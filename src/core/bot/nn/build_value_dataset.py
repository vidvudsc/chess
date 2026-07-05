#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import IO, Iterator, TextIO


def open_text_stream(path: Path) -> tuple[TextIO, subprocess.Popen[bytes] | None]:
    if str(path) == "-":
        return sys.stdin, None
    if path.suffix == ".zst":
        proc = subprocess.Popen(
            ["zstdcat", str(path)],
            stdout=subprocess.PIPE,
            stderr=None,
        )
        if proc.stdout is None:
            raise RuntimeError("zstdcat did not provide stdout")
        return TextIOWrapperForBytes(proc.stdout), proc
    return path.open("r", encoding="utf-8"), None


class TextIOWrapperForBytes:
    def __init__(self, raw: IO[bytes]) -> None:
        self.raw = raw

    def __iter__(self) -> Iterator[str]:
        for line in self.raw:
            yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.raw.close()


def best_eval(row: dict, min_depth: int) -> dict | None:
    best: dict | None = None
    best_depth = -1
    for item in row.get("evals", []):
        try:
            depth = int(item.get("depth", 0))
        except (TypeError, ValueError):
            continue
        if depth < min_depth or depth < best_depth:
            continue
        pvs = item.get("pvs", [])
        if not pvs:
            continue
        pv = pvs[0]
        if "cp" not in pv and "mate" not in pv:
            continue
        best = {
            "depth": depth,
            "pv": pv,
        }
        best_depth = depth
    return best


def evaluation_text(pv: dict) -> str:
    if "mate" in pv:
        mate = int(pv["mate"])
        return f"M{mate}" if mate >= 0 else f"-M{abs(mate)}"
    cp = int(pv["cp"])
    return f"{cp / 100.0:.2f}"


def convert_row(row: dict, min_depth: int) -> dict | None:
    fen = str(row.get("fen", "")).strip()
    if not fen:
        return None
    selected = best_eval(row, min_depth)
    if selected is None:
        return None
    pv = selected["pv"]
    line = str(pv.get("line", "")).strip()
    return {
        "fen": fen,
        "depth": int(selected["depth"]),
        "evaluation": evaluation_text(pv),
        "best_move": line.split()[0] if line else "",
        "best_line": line,
        "source": "lichess-evals",
    }


def maybe_keep(index: int, limit: int, rng: random.Random, reservoir: list[dict], item: dict) -> None:
    if limit <= 0:
        reservoir.append(item)
        return
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    slot = rng.randrange(index + 1)
    if slot < limit:
        reservoir[slot] = item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Lichess Stockfish eval JSONL into local value-training JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .jsonl, .jsonl.zst, or '-' for stdin.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--min-depth", type=int, default=18, help="Minimum Stockfish eval depth to keep.")
    parser.add_argument("--limit", type=int, default=0, help="Reservoir sample this many rows, 0 keeps all.")
    parser.add_argument("--seed", type=int, default=20260704, help="Reservoir sampling seed.")
    parser.add_argument("--log-every", type=int, default=100000, help="Progress interval in input rows.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    kept: list[dict] = []
    seen = 0
    converted = 0

    stream, proc = open_text_stream(args.input)
    try:
        for raw in stream:
            seen += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            item = convert_row(row, args.min_depth)
            if item is None:
                continue
            maybe_keep(converted, args.limit, rng, kept, item)
            converted += 1
            if args.log_every > 0 and seen % args.log_every == 0:
                print(f"[data] seen={seen} converted={converted} kept={len(kept)}", file=sys.stderr, flush=True)
    finally:
        if stream is not sys.stdin:
            stream.close()
        if proc is not None:
            proc.wait()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        for item in kept:
            fp.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(f"[done] seen={seen} converted={converted} wrote={len(kept)} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
