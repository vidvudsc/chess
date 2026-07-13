#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


NN_ROOT = Path(__file__).resolve().parents[1]
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from features import DUMMY_FEATURE_INDEX, encode_fen  # noqa: E402


MAX_HALFKP_FEATURES = 30


def sample_weight(row: dict[str, Any]) -> float:
    try:
        depth = int(row.get("depth", 18))
    except (TypeError, ValueError):
        depth = 18
    weight = math.sqrt(max(1, depth) / 18.0)
    weight = max(0.70, min(weight, 2.50))
    if bool(row.get("is_mate", False)):
        weight *= 0.75
    terminal = str(row.get("terminal", "none"))
    if terminal != "none":
        weight *= 0.75
    return weight


def pad_features(values: list[int]) -> tuple[list[int], int]:
    clipped = list(values[:MAX_HALFKP_FEATURES])
    length = len(clipped)
    if length < MAX_HALFKP_FEATURES:
        clipped.extend([DUMMY_FEATURE_INDEX] * (MAX_HALFKP_FEATURES - length))
    return clipped, length


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute NN v2 HalfKP features into compact NumPy shards.")
    parser.add_argument("--input", type=Path, required=True, help="v2 JSONL dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for .npz shards and manifest.")
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--uncompressed", action="store_true",
                        help="Write faster, larger .npz shards with np.savez instead of np.savez_compressed.")
    return parser


def save_shard(output_dir: Path,
               shard_index: int,
               white: list[list[int]],
               white_len: list[int],
               black: list[list[int]],
               black_len: list[int],
               stm_white: list[int],
               target: list[float],
               weight: list[float],
               split_bucket: list[int] | None = None,
               compressed: bool = True) -> dict[str, Any]:
    name = f"features_{shard_index:05d}.npz"
    path = output_dir / name
    savez = np.savez_compressed if compressed else np.savez
    arrays = dict(
        white=np.asarray(white, dtype=np.uint32),
        white_len=np.asarray(white_len, dtype=np.uint8),
        black=np.asarray(black, dtype=np.uint32),
        black_len=np.asarray(black_len, dtype=np.uint8),
        stm_white=np.asarray(stm_white, dtype=np.bool_),
        target=np.asarray(target, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
    )
    if split_bucket is not None:
        arrays["split_bucket"] = np.asarray(split_bucket, dtype=np.uint16)
    savez(path, **arrays)
    return {"file": name, "rows": len(target), "compressed": compressed}


def split_bucket_for_fen(fen: str, modulus: int = 1000) -> int:
    digest = hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    white: list[list[int]] = []
    white_len: list[int] = []
    black: list[list[int]] = []
    black_len: list[int] = []
    stm_white: list[int] = []
    target: list[float] = []
    weight: list[float] = []
    split_bucket: list[int] = []
    shards: list[dict[str, Any]] = []
    rows = 0
    skipped = 0

    def flush() -> None:
        nonlocal white, white_len, black, black_len, stm_white, target, weight, split_bucket
        if not target:
            return
        shards.append(save_shard(
            args.output_dir,
            len(shards),
            white,
            white_len,
            black,
            black_len,
            stm_white,
            target,
            weight,
            split_bucket,
            compressed=not args.uncompressed,
        ))
        white = []
        white_len = []
        black = []
        black_len = []
        stm_white = []
        target = []
        weight = []
        split_bucket = []

    with args.input.open("r", encoding="utf-8") as fp:
        for raw in fp:
            if args.limit > 0 and rows >= args.limit:
                break
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if "target_stm" not in row or "fen" not in row:
                skipped += 1
                continue
            try:
                white_half, black_half, side_white = encode_fen(str(row["fen"]))
            except ValueError:
                skipped += 1
                continue
            white_features, white_count = pad_features(white_half or [DUMMY_FEATURE_INDEX])
            black_features, black_count = pad_features(black_half or [DUMMY_FEATURE_INDEX])
            white.append(white_features)
            white_len.append(white_count)
            black.append(black_features)
            black_len.append(black_count)
            stm_white.append(1 if side_white else 0)
            target.append(float(row["target_stm"]))
            weight.append(sample_weight(row))
            split_bucket.append(split_bucket_for_fen(str(row["fen"])))
            rows += 1
            if len(target) >= args.rows_per_shard:
                flush()
            if args.log_every > 0 and rows % args.log_every == 0:
                print(f"[precompute] rows={rows} skipped={skipped} shards={len(shards)}", flush=True)

    flush()
    manifest = {
        "input": str(args.input),
        "rows": rows,
        "skipped": skipped,
        "max_halfkp_features": MAX_HALFKP_FEATURES,
        "dummy_feature_index": DUMMY_FEATURE_INDEX,
        "compressed": not args.uncompressed,
        "shards": shards,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
