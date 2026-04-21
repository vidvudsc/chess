from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


DEFAULT_CP_CLIP = 1200
DEFAULT_CP_SCALE = 600.0
DEFAULT_VAL_BUCKETS = 10
DEFAULT_SPLIT_MODULUS = 1000


@dataclass(frozen=True)
class TrainingSample:
    fen: str
    stm_white: bool
    depth: int
    evaluation_raw: str
    is_mate: bool
    value_target: float
    cp_target: int
    best_move_san: str
    best_line_san: str
    weight: float


def fen_bucket(fen: str, modulus: int = DEFAULT_SPLIT_MODULUS) -> int:
    digest = hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % modulus


def split_matches(fen: str,
                  split: Optional[str],
                  val_buckets: int = DEFAULT_VAL_BUCKETS,
                  modulus: int = DEFAULT_SPLIT_MODULUS) -> bool:
    if split is None or split == "all":
        return True
    bucket = fen_bucket(fen, modulus)
    in_val = bucket < val_buckets
    if split == "val":
        return in_val
    if split == "train":
        return not in_val
    raise ValueError(f"unsupported split: {split}")


def parse_eval(raw: str,
               cp_clip: int = DEFAULT_CP_CLIP,
               cp_scale: float = DEFAULT_CP_SCALE) -> tuple[bool, float, int]:
    text = raw.strip()
    if text.startswith("-M") or text.startswith("M"):
        sign = 1
        if text.startswith("-M") or text.startswith("M-"):
            sign = -1
        return True, float(sign), sign * cp_clip

    pawns = float(text)
    cp = int(round(pawns * 100.0))
    cp = max(-cp_clip, min(cp_clip, cp))
    value_target = math.tanh(cp / cp_scale)
    return False, value_target, cp


def depth_weight(depth: int, is_mate: bool) -> float:
    weight = math.sqrt(max(depth, 1) / 15.0)
    weight = max(0.75, min(weight, 2.50))
    if is_mate:
        weight *= 0.75
    return weight


def iter_jsonl_rows(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_training_samples(path: Path,
                          min_depth: int = 15,
                          include_mates: bool = True,
                          split: Optional[str] = None,
                          val_buckets: int = DEFAULT_VAL_BUCKETS,
                          modulus: int = DEFAULT_SPLIT_MODULUS,
                          cp_clip: int = DEFAULT_CP_CLIP,
                          cp_scale: float = DEFAULT_CP_SCALE) -> Iterator[TrainingSample]:
    for row in iter_jsonl_rows(path):
        fen = str(row["fen"])
        fen_parts = fen.split()
        if len(fen_parts) < 2:
            continue
        stm_white = fen_parts[1] == "w"
        depth = int(row["depth"])
        if depth < min_depth:
            continue
        if not split_matches(fen, split, val_buckets=val_buckets, modulus=modulus):
            continue

        evaluation_raw = str(row["evaluation"])
        is_mate, value_target, cp_target = parse_eval(
            evaluation_raw,
            cp_clip=cp_clip,
            cp_scale=cp_scale,
        )
        if is_mate and not include_mates:
            continue
        if not stm_white:
            value_target = -value_target
            cp_target = -cp_target

        yield TrainingSample(
            fen=fen,
            stm_white=stm_white,
            depth=depth,
            evaluation_raw=evaluation_raw,
            is_mate=is_mate,
            value_target=value_target,
            cp_target=cp_target,
            best_move_san=str(row["best_move"]),
            best_line_san=str(row["best_line"]),
            weight=depth_weight(depth, is_mate),
        )


def count_training_samples(path: Path,
                           min_depth: int = 15,
                           include_mates: bool = True,
                           split: Optional[str] = None,
                           val_buckets: int = DEFAULT_VAL_BUCKETS,
                           modulus: int = DEFAULT_SPLIT_MODULUS) -> int:
    rows = 0
    for row in iter_jsonl_rows(path):
        fen = str(row["fen"])
        fen_parts = fen.split()
        if len(fen_parts) < 2:
            continue
        depth = int(row["depth"])
        if depth < min_depth:
            continue
        if not split_matches(fen, split, val_buckets=val_buckets, modulus=modulus):
            continue
        evaluation_raw = str(row["evaluation"])
        is_mate = evaluation_raw.startswith("-M") or evaluation_raw.startswith("M")
        if is_mate and not include_mates:
            continue
        rows += 1
    return rows


def summarize_samples(samples: Iterable[TrainingSample]) -> Dict[str, object]:
    rows = 0
    mates = 0
    min_depth: Optional[int] = None
    max_depth: Optional[int] = None
    sum_depth = 0
    sum_weight = 0.0

    for sample in samples:
        rows += 1
        if sample.is_mate:
            mates += 1
        sum_depth += sample.depth
        sum_weight += sample.weight
        if min_depth is None or sample.depth < min_depth:
            min_depth = sample.depth
        if max_depth is None or sample.depth > max_depth:
            max_depth = sample.depth

    if rows == 0:
        return {
            "rows": 0,
            "mates": 0,
            "mean_depth": 0.0,
            "mean_weight": 0.0,
            "min_depth": None,
            "max_depth": None,
        }

    return {
        "rows": rows,
        "mates": mates,
        "mate_fraction": mates / rows,
        "mean_depth": sum_depth / rows,
        "mean_weight": sum_weight / rows,
        "min_depth": min_depth,
        "max_depth": max_depth,
    }
