#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training_data import iter_training_samples, summarize_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect streamed NN training data without rewriting the source JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Path to stockfish SAN JSONL")
    parser.add_argument("--min-depth", type=int, default=15, help="Minimum depth to keep")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all", help="Hash-based split selector")
    parser.add_argument("--exclude-mates", action="store_true", help="Drop mate-labelled rows")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of example rows to print")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to scan after filtering")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    depth_counts = Counter()
    samples = []
    kept = []

    for sample in iter_training_samples(
        args.input,
        min_depth=args.min_depth,
        include_mates=not args.exclude_mates,
        split=args.split,
    ):
        kept.append(sample)
        depth_counts[sample.depth] += 1
        if len(samples) < args.sample_count:
            samples.append(sample)
        if args.limit > 0 and len(kept) >= args.limit:
            break

    summary = summarize_samples(kept)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("top_depths:")
    for depth, count in depth_counts.most_common(10):
        print(f"  depth {depth}: {count}")
    print("samples:")
    for sample in samples:
        print(json.dumps({
            "fen": sample.fen,
            "stm_white": sample.stm_white,
            "depth": sample.depth,
            "evaluation_raw": sample.evaluation_raw,
            "is_mate": sample.is_mate,
            "value_target": sample.value_target,
            "cp_target": sample.cp_target,
            "best_move_san": sample.best_move_san,
            "weight": sample.weight,
        }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
