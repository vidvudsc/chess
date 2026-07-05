#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import urlopen

try:
    from .data import convert_flat_eval_row, convert_lichess_eval_row, terminal_fixture_rows
except ImportError:  # pragma: no cover - lets this script run directly
    from data import convert_flat_eval_row, convert_lichess_eval_row, terminal_fixture_rows


HF_DATASETS_API = "https://datasets-server.huggingface.co"


@dataclass(frozen=True)
class SourceItem:
    row: dict[str, Any]
    flat: bool
    source_name: str


@dataclass(frozen=True)
class HfDatasetSpec:
    dataset: str
    config: str
    split: str


class StreamHandle:
    def __init__(self, text: TextIO, procs: list[subprocess.Popen[bytes]] | None = None) -> None:
        self.text = text
        self.procs = procs or []

    def close(self) -> None:
        if self.text is not sys.stdin:
            self.text.close()
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def check(self, allow_early_stop: bool = False) -> None:
        for proc in self.procs:
            code = proc.poll()
            if allow_early_stop and code in (23, 70, -signal.SIGTERM, -signal.SIGPIPE):
                continue
            if code not in (None, 0, -signal.SIGTERM, -signal.SIGPIPE):
                raise SystemExit(f"stream process exited with status {code}: {' '.join(proc.args)}")


class TextIOWrapperForBytes:
    def __init__(self, raw: IO[bytes]) -> None:
        self.raw = raw

    def __iter__(self) -> Iterator[str]:
        for line in self.raw:
            yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.raw.close()


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def is_hf_source(value: str) -> bool:
    return value.startswith("hf://")


def parse_hf_source(value: str) -> HfDatasetSpec:
    raw = value.removeprefix("hf://").strip("/")
    parts = raw.split("/")
    if len(parts) < 2:
        raise ValueError("HF source must look like hf://namespace/repo[/config[/split]]")
    dataset = "/".join(parts[:2])
    config = parts[2] if len(parts) >= 3 and parts[2] else "default"
    split = parts[3] if len(parts) >= 4 and parts[3] else "train"
    return HfDatasetSpec(dataset=dataset, config=config, split=split)


def hf_rows_url(spec: HfDatasetSpec, offset: int, length: int) -> str:
    return (
        f"{HF_DATASETS_API}/rows?"
        f"dataset={quote(spec.dataset, safe='')}"
        f"&config={quote(spec.config, safe='')}"
        f"&split={quote(spec.split, safe='')}"
        f"&offset={offset}&length={length}"
    )


def fetch_json(url: str, retries: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(min(2 ** attempt, 10))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to fetch {url}")


def iter_hf_source(source: str, page_size: int) -> Iterator[SourceItem]:
    spec = parse_hf_source(source)
    offset = 0
    length = max(1, min(page_size, 100))
    while True:
        payload = fetch_json(hf_rows_url(spec, offset=offset, length=length))
        rows = payload.get("rows", [])
        if not isinstance(rows, list) or not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            if isinstance(row, dict):
                yield SourceItem(row=row, flat=True, source_name=f"hf:{spec.dataset}")
        offset += len(rows)
        total = payload.get("num_rows_total")
        if isinstance(total, int) and offset >= total:
            break


def open_text_stream(source: str) -> StreamHandle:
    if source == "-":
        return StreamHandle(sys.stdin)
    if is_url(source):
        curl = subprocess.Popen(
            ["curl", "-L", "--fail", "--silent", "--show-error", source],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if curl.stdout is None:
            raise RuntimeError("curl did not provide stdout")
        if source.endswith(".zst"):
            zstd = subprocess.Popen(["zstdcat"], stdin=curl.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            curl.stdout.close()
            if zstd.stdout is None:
                raise RuntimeError("zstdcat did not provide stdout")
            return StreamHandle(TextIOWrapperForBytes(zstd.stdout), [curl, zstd])
        return StreamHandle(TextIOWrapperForBytes(curl.stdout), [curl])

    path = Path(source)
    if path.suffix == ".zst":
        proc = subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stdout is None:
            raise RuntimeError("zstdcat did not provide stdout")
        return StreamHandle(TextIOWrapperForBytes(proc.stdout), [proc])
    return StreamHandle(path.open("r", encoding="utf-8"))


def reservoir_add(items: list[dict[str, Any]],
                  item: dict[str, Any],
                  seen: int,
                  limit: int,
                  rng: random.Random) -> None:
    if limit <= 0:
        items.append(item)
        return
    if len(items) < limit:
        items.append(item)
        return
    slot = rng.randrange(seen)
    if slot < limit:
        items[slot] = item


def update_summary(summary: dict[str, Counter[str]], item: dict[str, Any]) -> None:
    bucket = str(item.get("bucket", "unknown"))
    parts = bucket.split("/")
    summary["bucket"][bucket] += 1
    if len(parts) == 4:
        summary["side_to_move"][parts[0]] += 1
        summary["phase"][parts[1]] += 1
        summary["eval_band"][parts[2]] += 1
        summary["tactical"][parts[3]] += 1
    summary["terminal"][str(item.get("terminal", "none"))] += 1
    summary["mate"]["mate" if item.get("is_mate") else "non_mate"] += 1


def empty_summary() -> dict[str, Counter[str]]:
    return {
        "bucket": Counter(),
        "side_to_move": Counter(),
        "phase": Counter(),
        "eval_band": Counter(),
        "tactical": Counter(),
        "terminal": Counter(),
        "mate": Counter(),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary = empty_summary()
    for item in rows:
        update_summary(summary, item)
    return {name: counter_dict(counter) for name, counter in sorted(summary.items())}


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def write_jsonl_row(fp: TextIO, row: dict[str, Any]) -> None:
    fp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a clean NNUE v2 value dataset from Lichess eval JSONL.")
    parser.add_argument("--input", required=True, help="Input .jsonl, .jsonl.zst, URL, or '-' for stdin.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--min-depth", type=int, default=18, help="Minimum Stockfish depth.")
    parser.add_argument("--min-knodes", type=int, default=0, help="Minimum Stockfish knodes when present.")
    parser.add_argument("--limit", type=int, default=0, help="Global reservoir sample size; 0 keeps all accepted rows.")
    parser.add_argument("--max-per-bucket", type=int, default=0, help="Reservoir cap per side/phase/eval/tactical bucket.")
    parser.add_argument("--stream-output", action="store_true",
                        help="Write rows as they are accepted. Scales to large datasets; uses first-N per bucket, not reservoir sampling.")
    parser.add_argument("--max-input-rows", type=int, default=0, help="Stop after reading this many input rows.")
    parser.add_argument("--skip-input-rows", type=int, default=0, help="Skip this many input rows before converting.")
    parser.add_argument("--sample-every", type=int, default=1, help="Only try every Nth row after skipped rows.")
    parser.add_argument("--hf-page-size", type=int, default=100, help="HF Dataset Viewer page size, max 100.")
    parser.add_argument("--stop-after-accepted", type=int, default=0,
                        help="Stop streaming after this many accepted rows; use this for live URL sampling.")
    parser.add_argument("--terminal-fixtures", action="store_true",
                        help="Append canonical checkmate/stalemate/insufficient-material rows for both sides.")
    parser.add_argument("--no-dedupe", action="store_true", help="Allow duplicate normalized FEN keys.")
    parser.add_argument("--seed", type=int, default=20260704, help="Sampling seed.")
    parser.add_argument("--log-every", type=int, default=100000, help="Progress interval in input rows.")
    return parser


def iter_source_items(args: argparse.Namespace) -> Iterator[SourceItem]:
    if is_hf_source(args.input):
        yield from iter_hf_source(args.input, page_size=args.hf_page_size)
        return

    handle = open_text_stream(args.input)
    try:
        for raw in handle.text:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                yield SourceItem(row={"__bad_json__": True}, flat=False, source_name=args.input)
                continue
            yield SourceItem(row=row, flat=False, source_name=args.input)
    finally:
        handle.close()
        handle.check(allow_early_stop=args.stop_after_accepted > 0 or args.max_input_rows > 0)


def convert_source_item(item: SourceItem,
                        min_depth: int,
                        min_knodes: int,
                        score_perspective: str) -> dict[str, Any] | None:
    if item.row.get("__bad_json__"):
        return None
    if item.flat:
        return convert_flat_eval_row(
            item.row,
            min_depth=min_depth,
            min_knodes=min_knodes,
            score_perspective=score_perspective,
            source=item.source_name,
        )
    return convert_lichess_eval_row(
        item.row,
        min_depth=min_depth,
        min_knodes=min_knodes,
        score_perspective=score_perspective,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.stream_output and args.output == Path("-"):
        raise SystemExit("--stream-output requires a filesystem output path")
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    bucket_rows: dict[str, list[dict[str, Any]]] = {}
    bucket_seen: Counter[str] = Counter()
    seen_fens: set[str] = set()
    summary = empty_summary()
    written_summary = empty_summary()
    stats = Counter()

    output_fp: TextIO | None = None
    if args.stream_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_fp = args.output.open("w", encoding="utf-8")

    try:
        for source_item in iter_source_items(args):
            stats["input_rows"] += 1
            if args.max_input_rows > 0 and stats["input_rows"] > args.max_input_rows:
                stats["stopped_max_input_rows"] = 1
                break
            if stats["input_rows"] <= args.skip_input_rows:
                stats["skipped_input_rows"] += 1
                continue
            if args.sample_every > 1 and ((stats["input_rows"] - args.skip_input_rows - 1) % args.sample_every) != 0:
                stats["stride_skipped_rows"] += 1
                continue
            if source_item.row.get("__bad_json__"):
                stats["bad_json"] += 1
                continue

            item = convert_source_item(
                source_item,
                min_depth=args.min_depth,
                min_knodes=args.min_knodes,
                score_perspective="white",
            )
            if item is None:
                stats["rejected"] += 1
                continue

            fen_key = str(item["fen"])
            if not args.no_dedupe:
                if fen_key in seen_fens:
                    stats["duplicate_fen"] += 1
                    continue
                seen_fens.add(fen_key)

            stats["accepted"] += 1
            update_summary(summary, item)
            bucket = str(item["bucket"])

            if args.stream_output:
                bucket_seen[bucket] += 1
                if args.max_per_bucket > 0 and bucket_seen[bucket] > args.max_per_bucket:
                    stats["bucket_cap_skipped"] += 1
                    continue
                if args.limit > 0 and stats["written_rows"] >= args.limit:
                    stats["limit_skipped"] += 1
                    continue
                assert output_fp is not None
                write_jsonl_row(output_fp, item)
                stats["written_rows"] += 1
                update_summary(written_summary, item)
            elif args.max_per_bucket > 0:
                bucket_seen[bucket] += 1
                reservoir = bucket_rows.setdefault(bucket, [])
                reservoir_add(reservoir, item, bucket_seen[bucket], args.max_per_bucket, rng)
            else:
                reservoir_add(rows, item, stats["accepted"], args.limit, rng)

            if args.log_every > 0 and stats["input_rows"] % args.log_every == 0:
                print(
                    f"[data] input={stats['input_rows']} accepted={stats['accepted']} "
                    f"written={stats['written_rows']} dupes={stats['duplicate_fen']} "
                    f"rejected={stats['rejected']}",
                    file=sys.stderr,
                    flush=True,
                )
            if args.stop_after_accepted > 0 and stats["accepted"] >= args.stop_after_accepted:
                stats["stopped_after_accepted"] = 1
                break
            if args.stream_output and args.limit > 0 and stats["written_rows"] >= args.limit:
                stats["stopped_after_written_limit"] = 1
                break
    finally:
        if output_fp is not None:
            output_fp.close()

    if not args.stream_output and args.max_per_bucket > 0:
        rows = []
        for bucket in sorted(bucket_rows):
            rows.extend(bucket_rows[bucket])
        if args.limit > 0 and len(rows) > args.limit:
            sampled: list[dict[str, Any]] = []
            for idx, item in enumerate(rows, start=1):
                reservoir_add(sampled, item, idx, args.limit, rng)
            rows = sampled

    if args.terminal_fixtures:
        if args.stream_output:
            existing_fens = set(seen_fens) if not args.no_dedupe else set()
            with args.output.open("a", encoding="utf-8") as fp:
                for item in terminal_fixture_rows():
                    if not args.no_dedupe and str(item["fen"]) in existing_fens:
                        continue
                    write_jsonl_row(fp, item)
                    existing_fens.add(str(item["fen"]))
                    stats["terminal_fixtures"] += 1
                    stats["written_rows"] += 1
                    update_summary(written_summary, item)
        else:
            existing_fens = {str(row["fen"]) for row in rows}
            for item in terminal_fixture_rows():
                if not args.no_dedupe and str(item["fen"]) in existing_fens:
                    continue
                rows.append(item)
                existing_fens.add(str(item["fen"]))
                stats["terminal_fixtures"] += 1

    if not args.stream_output:
        write_jsonl(args.output, rows)
        stats["written_rows"] = len(rows)
        written_distribution = summarize_rows(rows)
    else:
        written_distribution = {name: counter_dict(counter) for name, counter in sorted(written_summary.items())}

    summary_payload = {
        "input": str(args.input),
        "output": str(args.output),
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "limit": args.limit,
        "max_per_bucket": args.max_per_bucket,
        "stream_output": bool(args.stream_output),
        "max_input_rows": args.max_input_rows,
        "skip_input_rows": args.skip_input_rows,
        "sample_every": args.sample_every,
        "stop_after_accepted": args.stop_after_accepted,
        "terminal_fixtures": bool(args.terminal_fixtures),
        "dedupe": not args.no_dedupe,
        "stats": counter_dict(stats),
        "written_rows": int(stats["written_rows"]),
        "accepted_distribution": {name: counter_dict(counter) for name, counter in sorted(summary.items())},
        "written_distribution": written_distribution,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2, sort_keys=True)

    print(json.dumps(summary_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
