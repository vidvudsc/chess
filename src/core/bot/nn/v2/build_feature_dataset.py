#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


NN_ROOT = Path(__file__).resolve().parents[1]
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from features import DUMMY_FEATURE_INDEX, encode_fen  # noqa: E402

try:
    from .build_dataset import (
        convert_source_item,
        counter_dict,
        empty_summary,
        iter_source_items,
        update_summary,
    )
    from .data import terminal_fixture_rows
    from .precompute_features import (
        MAX_HALFKP_FEATURES,
        pad_features,
        sample_weight,
        save_shard,
        split_bucket_for_fen,
    )
except ImportError:  # pragma: no cover - lets this script run directly
    from build_dataset import (  # type: ignore
        convert_source_item,
        counter_dict,
        empty_summary,
        iter_source_items,
        update_summary,
    )
    from data import terminal_fixture_rows  # type: ignore
    from precompute_features import (  # type: ignore
        MAX_HALFKP_FEATURES,
        pad_features,
        sample_weight,
        save_shard,
        split_bucket_for_fen,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build NN v2 HalfKP feature shards directly from Lichess eval JSONL."
    )
    parser.add_argument("--input", required=True, help="Input .jsonl, .jsonl.zst, URL, hf:// source, or '-'.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for .npz shards and manifest.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional build summary JSON path.")
    parser.add_argument("--min-depth", type=int, default=18, help="Minimum Stockfish depth.")
    parser.add_argument("--min-knodes", type=int, default=0, help="Minimum Stockfish knodes when present.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after writing this many feature rows; 0 keeps all.")
    parser.add_argument("--max-per-bucket", type=int, default=0, help="First-N cap per side/phase/eval/tactical bucket.")
    parser.add_argument("--fast", action="store_true",
                        help="Use faster approximate conversion before feature encoding.")
    parser.add_argument("--max-input-rows", type=int, default=0, help="Stop after reading this many input rows.")
    parser.add_argument("--skip-input-rows", type=int, default=0, help="Skip this many input rows before converting.")
    parser.add_argument("--sample-every", type=int, default=1, help="Only try every Nth row after skipped rows.")
    parser.add_argument("--hf-page-size", type=int, default=100, help="HF Dataset Viewer page size, max 100.")
    parser.add_argument("--stop-after-accepted", type=int, default=0, help="Stop after this many accepted eval rows.")
    parser.add_argument("--terminal-fixtures", action="store_true",
                        help="Append canonical checkmate/stalemate/insufficient-material feature rows.")
    parser.add_argument("--no-dedupe", action="store_true", help="Allow duplicate normalized FEN keys.")
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=100_000, help="Progress interval in input rows.")
    return parser


class FeatureShardWriter:
    def __init__(self, output_dir: Path, rows_per_shard: int, compressed: bool = True) -> None:
        self.output_dir = output_dir
        self.rows_per_shard = rows_per_shard
        self.compressed = compressed
        self.shards: list[dict[str, Any]] = []
        self.rows = 0
        self.skipped = 0
        self.white: list[list[int]] = []
        self.white_len: list[int] = []
        self.black: list[list[int]] = []
        self.black_len: list[int] = []
        self.stm_white: list[int] = []
        self.target: list[float] = []
        self.weight: list[float] = []
        self.split_bucket: list[int] = []

    def add(self, row: dict[str, Any]) -> bool:
        try:
            white_half, black_half, side_white = encode_fen(str(row["fen"]))
        except (KeyError, ValueError):
            self.skipped += 1
            return False

        white_features, white_count = pad_features(white_half or [DUMMY_FEATURE_INDEX])
        black_features, black_count = pad_features(black_half or [DUMMY_FEATURE_INDEX])
        self.white.append(white_features)
        self.white_len.append(white_count)
        self.black.append(black_features)
        self.black_len.append(black_count)
        self.stm_white.append(1 if side_white else 0)
        self.target.append(float(row["target_stm"]))
        self.weight.append(sample_weight(row))
        self.split_bucket.append(split_bucket_for_fen(str(row["fen"])))
        self.rows += 1

        if len(self.target) >= self.rows_per_shard:
            self.flush()
        return True

    def flush(self) -> None:
        if not self.target:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shards.append(save_shard(
            self.output_dir,
            len(self.shards),
            self.white,
            self.white_len,
            self.black,
            self.black_len,
            self.stm_white,
            self.target,
            self.weight,
            self.split_bucket,
            compressed=self.compressed,
        ))
        self.white = []
        self.white_len = []
        self.black = []
        self.black_len = []
        self.stm_white = []
        self.target = []
        self.weight = []
        self.split_bucket = []

    def manifest(self, input_name: str) -> dict[str, Any]:
        return {
            "input": input_name,
            "rows": self.rows,
            "skipped": self.skipped,
            "max_halfkp_features": MAX_HALFKP_FEATURES,
            "dummy_feature_index": DUMMY_FEATURE_INDEX,
            "compressed": self.compressed,
            "shards": self.shards,
        }


def should_skip_input(stats: Counter[str], args: argparse.Namespace) -> bool:
    if args.max_input_rows > 0 and stats["input_rows"] > args.max_input_rows:
        stats["stopped_max_input_rows"] = 1
        return True
    if stats["input_rows"] <= args.skip_input_rows:
        stats["skipped_input_rows"] += 1
        return True
    if args.sample_every > 1 and ((stats["input_rows"] - args.skip_input_rows - 1) % args.sample_every) != 0:
        stats["stride_skipped_rows"] += 1
        return True
    return False


def process_row(row: dict[str, Any],
                writer: FeatureShardWriter,
                args: argparse.Namespace,
                seen_fens: set[str],
                bucket_seen: Counter[str],
                accepted_summary: dict[str, Counter[str]],
                written_summary: dict[str, Counter[str]],
                stats: Counter[str],
                enforce_bucket_cap: bool = True,
                enforce_limit: bool = True) -> bool:
    fen_key = str(row["fen"])
    if not args.no_dedupe:
        if fen_key in seen_fens:
            stats["duplicate_fen"] += 1
            return False
        seen_fens.add(fen_key)

    stats["accepted"] += 1
    update_summary(accepted_summary, row)
    bucket = str(row["bucket"])
    bucket_seen[bucket] += 1
    if enforce_bucket_cap and args.max_per_bucket > 0 and bucket_seen[bucket] > args.max_per_bucket:
        stats["bucket_cap_skipped"] += 1
        return False
    if enforce_limit and args.limit > 0 and stats["written_rows"] >= args.limit:
        stats["limit_skipped"] += 1
        return False
    if not writer.add(row):
        stats["feature_skipped"] += 1
        return False

    stats["written_rows"] += 1
    update_summary(written_summary, row)
    return True


def main() -> int:
    args = build_parser().parse_args()
    args.stream_output = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    writer = FeatureShardWriter(args.output_dir, rows_per_shard=args.rows_per_shard)
    seen_fens: set[str] = set()
    bucket_seen: Counter[str] = Counter()
    accepted_summary = empty_summary()
    written_summary = empty_summary()
    stats: Counter[str] = Counter()

    for source_item in iter_source_items(args):
        stats["input_rows"] += 1
        if should_skip_input(stats, args):
            if stats.get("stopped_max_input_rows"):
                break
            continue
        if source_item.row.get("__bad_json__"):
            stats["bad_json"] += 1
            continue

        item = convert_source_item(
            source_item,
            min_depth=args.min_depth,
            min_knodes=args.min_knodes,
            score_perspective="white",
            fast=args.fast,
        )
        if item is None:
            stats["rejected"] += 1
            continue

        process_row(
            item,
            writer,
            args,
            seen_fens,
            bucket_seen,
            accepted_summary,
            written_summary,
            stats,
        )

        if args.log_every > 0 and stats["input_rows"] % args.log_every == 0:
            print(
                f"[features] input={stats['input_rows']} accepted={stats['accepted']} "
                f"written={stats['written_rows']} feature_skipped={stats['feature_skipped']} "
                f"rejected={stats['rejected']}",
                file=sys.stderr,
                flush=True,
            )
        if args.stop_after_accepted > 0 and stats["accepted"] >= args.stop_after_accepted:
            stats["stopped_after_accepted"] = 1
            break
        if args.limit > 0 and stats["written_rows"] >= args.limit:
            stats["stopped_after_written_limit"] = 1
            break

    if args.terminal_fixtures:
        for item in terminal_fixture_rows():
            process_row(
                item,
                writer,
                args,
                seen_fens,
                bucket_seen,
                accepted_summary,
                written_summary,
                stats,
                enforce_bucket_cap=False,
                enforce_limit=False,
            )

    writer.flush()
    manifest = writer.manifest(str(args.input))
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)

    summary_payload = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "manifest": str(manifest_path),
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "limit": args.limit,
        "max_per_bucket": args.max_per_bucket,
        "fast": bool(args.fast),
        "max_input_rows": args.max_input_rows,
        "skip_input_rows": args.skip_input_rows,
        "sample_every": args.sample_every,
        "stop_after_accepted": args.stop_after_accepted,
        "terminal_fixtures": bool(args.terminal_fixtures),
        "dedupe": not args.no_dedupe,
        "rows_per_shard": args.rows_per_shard,
        "stats": counter_dict(stats),
        "written_rows": int(stats["written_rows"]),
        "feature_rows": writer.rows,
        "feature_skipped": writer.skipped,
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
