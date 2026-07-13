#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


try:
    from .build_dataset import counter_dict, dumps_json, empty_summary, update_summary
    from .build_feature_dataset import FeatureShardWriter, process_row
    from .data import convert_flat_eval_row, normalize_fen_key_fast, terminal_fixture_rows
except ImportError:  # pragma: no cover - lets this script run directly
    from build_dataset import counter_dict, dumps_json, empty_summary, update_summary  # type: ignore
    from build_feature_dataset import FeatureShardWriter, process_row  # type: ignore
    from data import convert_flat_eval_row, normalize_fen_key_fast, terminal_fixture_rows  # type: ignore


COLUMNS = ("fen", "line", "depth", "knodes", "cp", "mate")


def stable_bucket(text: str, modulus: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: clean_scalar(row.get(key)) for key in COLUMNS}


def row_score(row: dict[str, Any]) -> tuple[int, int, int]:
    try:
        depth = int(row.get("depth", 0) or 0)
    except (TypeError, ValueError):
        depth = 0
    try:
        knodes = int(row.get("knodes", 0) or 0)
    except (TypeError, ValueError):
        knodes = 0
    has_line = 1 if str(row.get("line", "") or "").strip() else 0
    return depth, knodes, has_line


def is_better_candidate(new_row: dict[str, Any], old_row: dict[str, Any]) -> bool:
    return row_score(new_row) > row_score(old_row)


def expand_input(value: str) -> list[str]:
    path = Path(value)
    if value.startswith("hf://"):
        return [value]
    if path.is_dir():
        return [str(item) for item in sorted(path.rglob("*.parquet"))]
    if any(ch in value for ch in "*?[]"):
        return sorted(glob.glob(value))
    return [value]


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyarrow is required for parquet cleaning. Install training requirements.") from exc

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    columns = [column for column in COLUMNS if column in available]
    missing = set(COLUMNS) - set(columns)
    if "fen" in missing or "depth" in missing:
        raise SystemExit(f"{path} is missing required parquet columns: {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        table = batch.to_pydict()
        rows = len(next(iter(table.values()))) if table else 0
        for index in range(rows):
            yield clean_row({column: table[column][index] for column in columns})


def parse_hf_source(source: str) -> tuple[str, str | None, str]:
    raw = source.removeprefix("hf://").strip("/")
    parts = raw.split("/")
    if len(parts) < 2:
        raise ValueError("HF source must look like hf://namespace/repo[/config[/split]]")
    dataset = "/".join(parts[:2])
    config = parts[2] if len(parts) >= 3 and parts[2] else None
    split = parts[3] if len(parts) >= 4 and parts[3] else "train"
    return dataset, config, split


def iter_hf_rows(source: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("datasets is required for hf:// streaming. Prefer downloading parquet shards locally.") from exc

    dataset, config, split = parse_hf_source(source)
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config is not None:
        kwargs["name"] = config
    stream = load_dataset(dataset, **kwargs)
    for row in stream:
        yield clean_row(row)


def iter_input_rows(inputs: Iterable[str], batch_size: int) -> Iterator[dict[str, Any]]:
    for raw in inputs:
        for item in expand_input(raw):
            if item.startswith("hf://"):
                yield from iter_hf_rows(item)
            else:
                yield from iter_parquet_rows(Path(item), batch_size=batch_size)


def write_jsonl_row(fp: Any, row: dict[str, Any]) -> None:
    fp.write(dumps_json(row) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean denormalized Lichess/HF Stockfish eval parquet into v2 rows or HalfKP shards."
    )
    parser.add_argument("--input", action="append", required=True,
                        help="Local parquet file/dir/glob, or hf://namespace/repo[/config[/split]]. Repeatable.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Optional cleaned v2 JSONL output.")
    parser.add_argument("--output-features-dir", type=Path, default=None,
                        help="Optional direct HalfKP feature shard output directory.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--min-depth", type=int, default=18)
    parser.add_argument("--min-knodes", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=0,
                        help="Stop after this many unique sampled FEN candidates; 0 scans all inputs.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum non-fixture rows to write; 0 writes all candidates.")
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1, help="Only consider every Nth input row.")
    parser.add_argument("--hash-sample-mod", type=int, default=0,
                        help="Deterministically keep FENs whose hash modulo this value is below --hash-sample-keep.")
    parser.add_argument("--hash-sample-keep", type=int, default=1)
    parser.add_argument("--max-per-bucket", type=int, default=0)
    parser.add_argument("--terminal-fixtures", action="store_true")
    parser.add_argument("--assume-unique", action="store_true",
                        help="Stream rows directly without FEN dedupe/best-row selection. Use for already deduped datasets.")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast approximate labeling/bucketing; recommended for large cleaning jobs.")
    parser.add_argument("--uncompressed-features", action="store_true",
                        help="Write faster, larger feature shards by skipping npz compression.")
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--parquet-batch-size", type=int, default=65_536)
    parser.add_argument("--log-every", type=int, default=1_000_000)
    return parser


def write_converted_row(converted: dict[str, Any],
                        json_fp: Any,
                        writer: FeatureShardWriter | None,
                        process_args: argparse.Namespace,
                        seen_fens: set[str],
                        bucket_seen: Counter[str],
                        accepted_summary: dict[str, Counter[str]],
                        written_summary: dict[str, Counter[str]],
                        output_stats: Counter[str]) -> bool:
    if writer is not None:
        if not process_row(
            converted,
            writer,
            process_args,
            seen_fens,
            bucket_seen,
            accepted_summary,
            written_summary,
            output_stats,
        ):
            return False
        if json_fp is not None:
            write_jsonl_row(json_fp, converted)
        return True

    bucket = str(converted["bucket"])
    if process_args.max_per_bucket > 0 and bucket_seen[bucket] >= process_args.max_per_bucket:
        output_stats["bucket_cap_skipped"] += 1
        return False
    if process_args.limit > 0 and output_stats["written_rows"] >= process_args.limit:
        output_stats["limit_skipped"] += 1
        return False
    if json_fp is not None:
        write_jsonl_row(json_fp, converted)
    output_stats["accepted"] += 1
    output_stats["written_rows"] += 1
    bucket_seen[bucket] += 1
    update_summary(accepted_summary, converted)
    update_summary(written_summary, converted)
    return True


def append_terminal_fixtures(args: argparse.Namespace,
                            json_fp: Any,
                            writer: FeatureShardWriter | None,
                            process_args: argparse.Namespace,
                            seen_fens: set[str],
                            bucket_seen: Counter[str],
                            accepted_summary: dict[str, Counter[str]],
                            written_summary: dict[str, Counter[str]],
                            output_stats: Counter[str]) -> None:
    if not args.terminal_fixtures:
        return
    for fixture in terminal_fixture_rows():
        if writer is not None:
            process_row(
                fixture,
                writer,
                process_args,
                seen_fens,
                bucket_seen,
                accepted_summary,
                written_summary,
                output_stats,
                enforce_bucket_cap=False,
                enforce_limit=False,
            )
            if json_fp is not None:
                write_jsonl_row(json_fp, fixture)
        else:
            if json_fp is not None:
                write_jsonl_row(json_fp, fixture)
            output_stats["accepted"] += 1
            output_stats["written_rows"] += 1
            update_summary(accepted_summary, fixture)
            update_summary(written_summary, fixture)
        output_stats["terminal_fixtures"] += 1


def run_assume_unique(args: argparse.Namespace) -> dict[str, Any]:
    json_fp = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        json_fp = args.output_jsonl.open("w", encoding="utf-8")

    writer: FeatureShardWriter | None = None
    if args.output_features_dir is not None:
        writer = FeatureShardWriter(
            args.output_features_dir,
            rows_per_shard=args.rows_per_shard,
            compressed=not args.uncompressed_features,
        )

    process_args = argparse.Namespace(
        no_dedupe=True,
        max_per_bucket=args.max_per_bucket,
        limit=args.limit,
    )
    stats: Counter[str] = Counter()
    output_stats: Counter[str] = Counter()
    bucket_seen: Counter[str] = Counter()
    accepted_summary = empty_summary()
    written_summary = empty_summary()
    dummy_seen: set[str] = set()

    try:
        for row in iter_input_rows(args.input, batch_size=args.parquet_batch_size):
            stats["input_rows"] += 1
            if args.max_input_rows > 0 and stats["input_rows"] > args.max_input_rows:
                stats["stopped_max_input_rows"] = 1
                break
            if args.sample_every > 1 and ((stats["input_rows"] - 1) % args.sample_every) != 0:
                stats["stride_skipped_rows"] += 1
                continue

            fen = str(row.get("fen", "") or "").strip()
            if not fen:
                stats["missing_fen"] += 1
                continue
            fen_key = normalize_fen_key_fast(fen)
            if args.hash_sample_mod > 0 and stable_bucket(fen_key, args.hash_sample_mod) >= args.hash_sample_keep:
                stats["hash_skipped_rows"] += 1
                continue
            row["fen"] = fen_key

            converted = convert_flat_eval_row(
                row,
                min_depth=args.min_depth,
                min_knodes=args.min_knodes,
                score_perspective="white",
                source="lichess-hf-unique",
                fast=args.fast,
            )
            if converted is None:
                stats["conversion_rejected"] += 1
                continue
            stats["unique_candidates"] += 1
            if args.candidate_limit > 0 and stats["unique_candidates"] > args.candidate_limit:
                stats["stopped_candidate_limit"] = 1
                break
            write_converted_row(
                converted,
                json_fp,
                writer,
                process_args,
                dummy_seen,
                bucket_seen,
                accepted_summary,
                written_summary,
                output_stats,
            )
            if args.log_every > 0 and stats["input_rows"] % args.log_every == 0:
                print(
                    f"[clean-unique] input={stats['input_rows']} candidates={stats['unique_candidates']} "
                    f"written={output_stats['written_rows']}",
                    file=sys.stderr,
                    flush=True,
                )
            if args.limit > 0 and output_stats["written_rows"] >= args.limit:
                output_stats["stopped_written_limit"] = 1
                break

        append_terminal_fixtures(
            args,
            json_fp,
            writer,
            process_args,
            dummy_seen,
            bucket_seen,
            accepted_summary,
            written_summary,
            output_stats,
        )
    finally:
        if json_fp is not None:
            json_fp.close()

    manifest: dict[str, Any] | None = None
    if writer is not None:
        writer.flush()
        manifest = writer.manifest(",".join(args.input))
        manifest_path = args.output_features_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2, sort_keys=True)

    return {
        "input": args.input,
        "output_jsonl": str(args.output_jsonl) if args.output_jsonl is not None else None,
        "output_features_dir": str(args.output_features_dir) if args.output_features_dir is not None else None,
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "candidate_limit": args.candidate_limit,
        "limit": args.limit,
        "max_input_rows": args.max_input_rows,
        "sample_every": args.sample_every,
        "hash_sample_mod": args.hash_sample_mod,
        "hash_sample_keep": args.hash_sample_keep,
        "max_per_bucket": args.max_per_bucket,
        "terminal_fixtures": bool(args.terminal_fixtures),
        "assume_unique": True,
        "fast": bool(args.fast),
        "compressed_features": not args.uncompressed_features,
        "input_stats": counter_dict(stats),
        "output_stats": counter_dict(output_stats),
        "candidate_rows": int(stats["unique_candidates"]),
        "feature_manifest": manifest,
        "accepted_distribution": {name: counter_dict(counter) for name, counter in sorted(accepted_summary.items())},
        "written_distribution": {name: counter_dict(counter) for name, counter in sorted(written_summary.items())},
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.output_jsonl is None and args.output_features_dir is None:
        raise SystemExit("provide --output-jsonl and/or --output-features-dir")
    if args.hash_sample_mod < 0 or args.hash_sample_keep < 0:
        raise SystemExit("hash sample values must be non-negative")
    if args.hash_sample_mod > 0 and args.hash_sample_keep > args.hash_sample_mod:
        raise SystemExit("--hash-sample-keep cannot exceed --hash-sample-mod")

    if args.assume_unique:
        summary_payload = run_assume_unique(args)
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            with args.summary.open("w", encoding="utf-8") as fp:
                json.dump(summary_payload, fp, indent=2, sort_keys=True)
        print(json.dumps(summary_payload, indent=2, sort_keys=True), flush=True)
        return 0

    candidates: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()

    for row in iter_input_rows(args.input, batch_size=args.parquet_batch_size):
        stats["input_rows"] += 1
        if args.max_input_rows > 0 and stats["input_rows"] > args.max_input_rows:
            stats["stopped_max_input_rows"] = 1
            break
        if args.sample_every > 1 and ((stats["input_rows"] - 1) % args.sample_every) != 0:
            stats["stride_skipped_rows"] += 1
            continue

        fen = str(row.get("fen", "") or "").strip()
        if not fen:
            stats["missing_fen"] += 1
            continue
        fen_key = normalize_fen_key_fast(fen)
        if args.hash_sample_mod > 0 and stable_bucket(fen_key, args.hash_sample_mod) >= args.hash_sample_keep:
            stats["hash_skipped_rows"] += 1
            continue

        try:
            depth = int(row.get("depth", 0) or 0)
            knodes = int(row.get("knodes", 0) or 0)
        except (TypeError, ValueError):
            stats["bad_depth_or_knodes"] += 1
            continue
        if depth < args.min_depth or knodes < args.min_knodes:
            stats["rejected_depth_or_knodes"] += 1
            continue
        if row.get("cp") is None and row.get("mate") is None:
            stats["missing_score"] += 1
            continue

        row["fen"] = fen_key
        old = candidates.get(fen_key)
        if old is None:
            candidates[fen_key] = row
            stats["unique_candidates"] += 1
        elif is_better_candidate(row, old):
            candidates[fen_key] = row
            stats["candidate_replaced"] += 1
        else:
            stats["duplicate_candidate_skipped"] += 1

        if args.log_every > 0 and stats["input_rows"] % args.log_every == 0:
            print(
                f"[clean] input={stats['input_rows']} unique={len(candidates)} "
                f"dupes={stats['duplicate_candidate_skipped']} replaced={stats['candidate_replaced']}",
                file=sys.stderr,
                flush=True,
            )
        if args.candidate_limit > 0 and len(candidates) >= args.candidate_limit:
            stats["stopped_candidate_limit"] = 1
            break

    json_fp = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        json_fp = args.output_jsonl.open("w", encoding="utf-8")

    writer: FeatureShardWriter | None = None
    if args.output_features_dir is not None:
        writer = FeatureShardWriter(
            args.output_features_dir,
            rows_per_shard=args.rows_per_shard,
            compressed=not args.uncompressed_features,
        )

    process_args = argparse.Namespace(
        no_dedupe=True,
        max_per_bucket=args.max_per_bucket,
        limit=args.limit,
    )
    dummy_seen: set[str] = set()
    bucket_seen: Counter[str] = Counter()
    accepted_summary = empty_summary()
    written_summary = empty_summary()
    output_stats: Counter[str] = Counter()

    try:
        for flat_row in candidates.values():
            converted = convert_flat_eval_row(
                flat_row,
                min_depth=args.min_depth,
                min_knodes=args.min_knodes,
                score_perspective="white",
                source="lichess-hf-cleaned",
                fast=args.fast,
            )
            if converted is None:
                output_stats["conversion_rejected"] += 1
                continue
            if args.max_per_bucket > 0:
                bucket = str(converted["bucket"])
                if bucket_seen[bucket] >= args.max_per_bucket:
                    output_stats["bucket_cap_skipped"] += 1
                    continue
            if args.limit > 0 and output_stats["written_rows"] >= args.limit:
                output_stats["limit_skipped"] += 1
                break

            if not write_converted_row(
                converted,
                json_fp,
                writer,
                process_args,
                dummy_seen,
                bucket_seen,
                accepted_summary,
                written_summary,
                output_stats,
            ):
                continue

        append_terminal_fixtures(
            args,
            json_fp,
            writer,
            process_args,
            dummy_seen,
            bucket_seen,
            accepted_summary,
            written_summary,
            output_stats,
        )
    finally:
        if json_fp is not None:
            json_fp.close()

    manifest: dict[str, Any] | None = None
    if writer is not None:
        writer.flush()
        manifest = writer.manifest(",".join(args.input))
        manifest_path = args.output_features_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2, sort_keys=True)

    summary_payload = {
        "input": args.input,
        "output_jsonl": str(args.output_jsonl) if args.output_jsonl is not None else None,
        "output_features_dir": str(args.output_features_dir) if args.output_features_dir is not None else None,
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "candidate_limit": args.candidate_limit,
        "limit": args.limit,
        "max_input_rows": args.max_input_rows,
        "sample_every": args.sample_every,
        "hash_sample_mod": args.hash_sample_mod,
        "hash_sample_keep": args.hash_sample_keep,
        "max_per_bucket": args.max_per_bucket,
        "terminal_fixtures": bool(args.terminal_fixtures),
        "assume_unique": False,
        "fast": bool(args.fast),
        "compressed_features": not args.uncompressed_features,
        "input_stats": counter_dict(stats),
        "output_stats": counter_dict(output_stats),
        "candidate_rows": len(candidates),
        "feature_manifest": manifest,
        "accepted_distribution": {name: counter_dict(counter) for name, counter in sorted(accepted_summary.items())},
        "written_distribution": {name: counter_dict(counter) for name, counter in sorted(written_summary.items())},
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2, sort_keys=True)

    print(json.dumps(summary_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
