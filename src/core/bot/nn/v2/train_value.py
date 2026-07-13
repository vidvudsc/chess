#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


NN_ROOT = Path(__file__).resolve().parents[1]
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from export_inference import export_checkpoint  # noqa: E402
from features import (  # noqa: E402
    DUMMY_FEATURE_INDEX,
    HALFKA_DUMMY_FEATURE_INDEX,
    MIRRORED_DUMMY_FEATURE_INDEX,
    MIRRORED_HALFKA_DUMMY_FEATURE_INDEX,
    encode_fen,
    encode_fen_halfka,
)
from model import build_value_model  # noqa: E402

try:
    from .data import convert_flat_eval_row, terminal_fixture_rows  # type: ignore
except ImportError:  # pragma: no cover - lets this script run directly
    from data import convert_flat_eval_row, terminal_fixture_rows  # type: ignore


DEFAULT_OUTPUT_DIR = NN_ROOT / "model" / "v2"
DEFAULT_EXPORT_PATH = NN_ROOT / "model" / "nn_eval.bin"
PARQUET_COLUMNS = ("fen", "line", "depth", "knodes", "cp", "score", "mate")


@dataclass
class Batch:
    white_indices: torch.Tensor
    white_offsets: torch.Tensor
    black_indices: torch.Tensor
    black_offsets: torch.Tensor
    stm_white: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor


class TrainingInterrupted(Exception):
    def __init__(self, metrics: dict[str, float]) -> None:
        super().__init__("training interrupted")
        self.metrics = metrics


def stable_bucket(text: str, modulus: int = 1000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def split_matches(fen: str, split: str, val_buckets: int, modulus: int = 1000) -> bool:
    bucket = stable_bucket(fen, modulus=modulus)
    in_val = bucket < val_buckets
    if split == "train":
        return not in_val
    if split == "val":
        return in_val
    if split == "all":
        return True
    raise ValueError(f"unsupported split: {split}")


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


def expand_parquet_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        raw = Path(value)
        if raw.is_dir():
            paths.extend(sorted(raw.rglob("*.parquet")))
        elif any(ch in value for ch in "*?[]"):
            paths.extend(Path(item) for item in sorted(glob.glob(value, recursive=True)))
        else:
            paths.append(raw)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def iter_parquet_eval_rows(paths: list[Path],
                           batch_size: int,
                           worker_id: int = 0,
                           num_workers: int = 1,
                           shuffle: bool = False,
                           seed: int = 0) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyarrow is required for parquet training. Install training requirements.") from exc

    row_groups: list[tuple[Path, int]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            row_groups.append((path, row_group))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(row_groups)

    for unit_index, (path, row_group) in enumerate(row_groups):
        if unit_index % num_workers != worker_id:
            continue
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        columns = [column for column in PARQUET_COLUMNS if column in available]
        missing = set(("fen", "depth")) - set(columns)
        if missing:
            raise SystemExit(f"{path} is missing required parquet columns: {sorted(missing)}")
        batch_rng = random.Random(seed ^ (unit_index * 0x9E3779B1) ^ worker_id)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, row_groups=[row_group]):
            table = batch.to_pydict()
            rows = len(next(iter(table.values()))) if table else 0
            indices = list(range(rows))
            if shuffle:
                batch_rng.shuffle(indices)
            for index in indices:
                yield {column: clean_scalar(table[column][index]) for column in columns}


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
    try:
        row_weight = float(row.get("weight", 1.0))
    except (TypeError, ValueError):
        row_weight = 1.0
    weight *= max(0.05, min(row_weight, 32.0))
    return weight


def iter_rows(path: Path,
              split: str,
              val_buckets: int,
              sample_limit: int,
              worker_id: int = 0,
              num_workers: int = 1) -> Iterator[dict[str, Any]]:
    yielded = 0
    accepted = 0
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            if sample_limit > 0 and yielded >= sample_limit:
                break
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            fen = str(row.get("fen", ""))
            if not fen or not split_matches(fen, split, val_buckets):
                continue
            if "target_stm" not in row:
                continue
            assigned = accepted % num_workers == worker_id
            accepted += 1
            if not assigned:
                continue
            yielded += 1
            yield row


class V2ValueDataset(IterableDataset):
    def __init__(self,
                 input_path: Path,
                 split: str,
                 val_buckets: int,
                 sample_limit: int,
                 encoder: Callable[[str], tuple[list[int], list[int], bool]] = encode_fen) -> None:
        super().__init__()
        self.input_path = input_path
        self.split = split
        self.val_buckets = val_buckets
        self.sample_limit = sample_limit
        self.encoder = encoder

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        sample_limit = self.sample_limit
        if sample_limit > 0 and num_workers > 1:
            sample_limit = math.ceil(sample_limit / num_workers)
        for row in iter_rows(
            self.input_path,
            split=self.split,
            val_buckets=self.val_buckets,
            sample_limit=sample_limit,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            white_half, black_half, stm_white = self.encoder(str(row["fen"]))
            yield {
                "white_half": white_half or [DUMMY_FEATURE_INDEX],
                "black_half": black_half or [DUMMY_FEATURE_INDEX],
                "stm_white": stm_white,
                "target": float(row["target_stm"]),
                "weight": sample_weight(row),
            }


class PrecomputedFeatureDataset(IterableDataset):
    def __init__(self,
                 feature_dir: Path,
                 split: str,
                 val_buckets: int,
                 sample_limit: int,
                 shuffle: bool = False,
                 seed: int = 0) -> None:
        super().__init__()
        self.feature_dir = feature_dir
        self.split = split
        self.val_buckets = val_buckets
        self.sample_limit = sample_limit
        self.shuffle = shuffle
        self.seed = seed
        manifest_path = feature_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as fp:
            self.manifest = json.load(fp)
        self.shards = [feature_dir / item["file"] for item in self.manifest.get("shards", [])]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        sample_limit = self.sample_limit
        if sample_limit > 0 and num_workers > 1:
            sample_limit = math.ceil(sample_limit / num_workers)
        yielded = 0
        index = 0
        shards = list(self.shards)
        rng = np.random.default_rng(self.seed)
        if self.shuffle:
            rng.shuffle(shards)
        for shard in shards:
            with np.load(shard) as data:
                white = data["white"]
                white_len = data["white_len"]
                black = data["black"]
                black_len = data["black_len"]
                stm_white = data["stm_white"]
                target = data["target"]
                weight = data["weight"]
                split_bucket = data["split_bucket"] if "split_bucket" in data else None
                row_indices = np.arange(target.shape[0])
                if self.shuffle:
                    rng.shuffle(row_indices)
                for row_idx in row_indices:
                    if sample_limit > 0 and yielded >= sample_limit:
                        return
                    bucket = int(split_bucket[row_idx]) if split_bucket is not None else index % 1000
                    assigned = index % num_workers == worker_id
                    index += 1
                    if not assigned:
                        continue
                    in_val = bucket < self.val_buckets
                    if self.split == "train" and in_val:
                        continue
                    if self.split == "val" and not in_val:
                        continue
                    yielded += 1
                    yield {
                        "white_half": white[row_idx, :int(white_len[row_idx])].tolist(),
                        "black_half": black[row_idx, :int(black_len[row_idx])].tolist(),
                        "stm_white": bool(stm_white[row_idx]),
                        "target": float(target[row_idx]),
                        "weight": float(weight[row_idx]),
                    }


class ParquetValueDataset(IterableDataset):
    def __init__(self,
                 inputs: list[str],
                 split: str,
                 val_buckets: int,
                 sample_limit: int,
                 min_depth: int,
                 min_knodes: int,
                 cp_scale: float,
                 batch_size: int,
                 fast: bool,
                 terminal_fixtures: bool,
                 shuffle: bool,
                 seed: int,
                 encoder: Callable[[str], tuple[list[int], list[int], bool]] = encode_fen) -> None:
        super().__init__()
        self.paths = expand_parquet_inputs(inputs)
        if not self.paths:
            raise ValueError(f"no parquet files matched: {inputs}")
        self.split = split
        self.val_buckets = val_buckets
        self.sample_limit = sample_limit
        self.min_depth = min_depth
        self.min_knodes = min_knodes
        self.cp_scale = cp_scale
        self.batch_size = batch_size
        self.fast = fast
        self.terminal_fixtures = terminal_fixtures
        self.shuffle = shuffle
        self.seed = seed
        self.encoder = encoder

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        sample_limit = self.sample_limit
        if sample_limit > 0 and num_workers > 1:
            sample_limit = math.ceil(sample_limit / num_workers)

        yielded = 0
        for row in iter_parquet_eval_rows(
            self.paths,
            batch_size=self.batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
            shuffle=self.shuffle,
            seed=self.seed,
        ):
            if sample_limit > 0 and yielded >= sample_limit:
                return
            converted = convert_flat_eval_row(
                row,
                min_depth=self.min_depth,
                min_knodes=self.min_knodes,
                score_perspective="white",
                source="lichess-hf-parquet",
                fast=self.fast,
                cp_scale=self.cp_scale,
            )
            if converted is None:
                continue
            fen = str(converted["fen"])
            if not split_matches(fen, self.split, self.val_buckets):
                continue
            try:
                white_half, black_half, stm_white = self.encoder(fen)
            except ValueError:
                continue
            yielded += 1
            yield {
                "white_half": white_half or [DUMMY_FEATURE_INDEX],
                "black_half": black_half or [DUMMY_FEATURE_INDEX],
                "stm_white": stm_white,
                "target": float(converted["target_stm"]),
                "weight": sample_weight(converted),
            }
        if self.terminal_fixtures and self.split in {"train", "all"} and worker_id == 0:
            for converted in terminal_fixture_rows():
                fen = str(converted["fen"])
                try:
                    white_half, black_half, stm_white = self.encoder(fen)
                except ValueError:
                    continue
                yield {
                    "white_half": white_half or [DUMMY_FEATURE_INDEX],
                    "black_half": black_half or [DUMMY_FEATURE_INDEX],
                    "stm_white": stm_white,
                    "target": float(converted["target_stm"]),
                    "weight": sample_weight(converted),
                }


class RepeatedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, repeat: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.repeat = max(1, repeat)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for _ in range(self.repeat):
            yield from self.dataset


class ChainedIterableDataset(IterableDataset):
    def __init__(self, datasets: list[IterableDataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for dataset in self.datasets:
            yield from dataset


def collate(samples: list[dict[str, Any]]) -> Batch:
    white_indices: list[int] = []
    white_offsets: list[int] = []
    black_indices: list[int] = []
    black_offsets: list[int] = []
    stm_white: list[bool] = []
    target: list[float] = []
    weight: list[float] = []
    for sample in samples:
        white_offsets.append(len(white_indices))
        black_offsets.append(len(black_indices))
        white_indices.extend(list(sample["white_half"]))
        black_indices.extend(list(sample["black_half"]))
        stm_white.append(bool(sample["stm_white"]))
        target.append(float(sample["target"]))
        weight.append(float(sample["weight"]))
    return Batch(
        white_indices=torch.tensor(white_indices, dtype=torch.long),
        white_offsets=torch.tensor(white_offsets, dtype=torch.long),
        black_indices=torch.tensor(black_indices, dtype=torch.long),
        black_offsets=torch.tensor(black_offsets, dtype=torch.long),
        stm_white=torch.tensor(stm_white, dtype=torch.bool),
        target=torch.tensor(target, dtype=torch.float32),
        weight=torch.tensor(weight, dtype=torch.float32),
    )


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        white_indices=batch.white_indices.to(device),
        white_offsets=batch.white_offsets.to(device),
        black_indices=batch.black_indices.to(device),
        black_offsets=batch.black_offsets.to(device),
        stm_white=batch.stm_white.to(device),
        target=batch.target.to(device),
        weight=batch.weight.to(device),
    )


def mirror_halfkp_indices(indices: torch.Tensor,
                          planes: int = 10,
                          source_dummy: int = DUMMY_FEATURE_INDEX,
                          target_dummy: int = MIRRORED_DUMMY_FEATURE_INDEX) -> torch.Tensor:
    stride = planes * 64
    king = torch.div(indices, stride, rounding_mode="floor")
    remainder = indices.remainder(stride)
    plane = torch.div(remainder, 64, rounding_mode="floor")
    piece_square = remainder.remainder(64)
    mirror = king.remainder(8) < 4
    king = torch.where(mirror, torch.bitwise_xor(king, 7), king)
    piece_square = torch.where(mirror, torch.bitwise_xor(piece_square, 7), piece_square)
    king_bucket = torch.div(king, 8, rounding_mode="floor") * 4 + king.remainder(8) - 4
    mapped = (king_bucket * planes + plane) * 64 + piece_square
    return torch.where(
        indices == source_dummy,
        torch.full_like(mapped, target_dummy),
        mapped,
    )


def mirror_batch_features(batch: Batch, halfka: bool = False) -> Batch:
    kwargs = {
        "planes": 11,
        "source_dummy": HALFKA_DUMMY_FEATURE_INDEX,
        "target_dummy": MIRRORED_HALFKA_DUMMY_FEATURE_INDEX,
    } if halfka else {}
    return Batch(
        white_indices=mirror_halfkp_indices(batch.white_indices, **kwargs),
        white_offsets=batch.white_offsets,
        black_indices=mirror_halfkp_indices(batch.black_indices, **kwargs),
        black_offsets=batch.black_offsets,
        stm_white=batch.stm_white,
        target=batch.target,
        weight=batch.weight,
    )


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             mirrored: bool = False,
             halfka: bool = False) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    batches = 0
    rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if mirrored:
            batch = mirror_batch_features(batch, halfka=halfka)
        pred = model(
            batch.white_indices,
            batch.white_offsets,
            batch.black_indices,
            batch.black_offsets,
            batch.stm_white,
        )
        loss = weighted_loss(pred, batch.target, batch.weight)
        mae = (torch.abs(pred - batch.target) * batch.weight).sum() / batch.weight.sum().clamp_min(1e-8)
        total_loss += float(loss.item())
        total_mae += float(mae.item())
        batches += 1
        rows += int(batch.target.numel())
    if batches == 0:
        return {"loss": 0.0, "mae": 0.0, "batches": 0.0, "rows": 0.0}
    return {
        "loss": total_loss / batches,
        "mae": total_mae / batches,
        "batches": float(batches),
        "rows": float(rows),
    }


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                log_every: int,
                mirrored: bool = False,
                halfka: bool = False,
                save_callback: Callable[[int, dict[str, float], str], None] | None = None) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    batches = 0
    rows = 0
    started = time.time()

    def metrics() -> dict[str, float]:
        if batches == 0:
            return {"loss": 0.0, "mae": 0.0, "batches": 0.0, "rows": 0.0, "elapsed_s": time.time() - started}
        return {
            "loss": total_loss / batches,
            "mae": total_mae / batches,
            "batches": float(batches),
            "rows": float(rows),
            "elapsed_s": time.time() - started,
        }

    try:
        for idx, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            if mirrored:
                batch = mirror_batch_features(batch, halfka=halfka)
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                batch.white_indices,
                batch.white_offsets,
                batch.black_indices,
                batch.black_offsets,
                batch.stm_white,
            )
            loss = weighted_loss(pred, batch.target, batch.weight)
            mae = (torch.abs(pred - batch.target) * batch.weight).sum() / batch.weight.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_mae += float(mae.item())
            batches += 1
            rows += int(batch.target.numel())
            current = metrics()
            if log_every > 0 and idx % log_every == 0:
                print(
                    f"[train] batch={idx} rows={rows} "
                    f"loss={current['loss']:.5f} mae={current['mae']:.5f}",
                    flush=True,
                )
            if save_callback is not None:
                save_callback(idx, current, "periodic")
    except KeyboardInterrupt as exc:
        current = metrics()
        if save_callback is not None:
            save_callback(batches, current, "interrupt")
        raise TrainingInterrupted(current) from exc
    if batches == 0:
        return {"loss": 0.0, "mae": 0.0, "batches": 0.0, "rows": 0.0, "elapsed_s": 0.0}
    return metrics()


def save_checkpoint(output_dir: Path,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    train_metrics: dict[str, float],
                    val_metrics: dict[str, float],
                    args: argparse.Namespace,
                    name: str | None = None,
                    update_best: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    path = output_dir / (name if name is not None else f"checkpoint_epoch_{epoch:02d}.pt")
    torch.save(payload, path)
    if update_best:
        torch.save(payload, output_dir / "best.pt")
    return path


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the NNUE v2 value model from v2 dataset rows.")
    parser.add_argument("--input", type=Path, default=None, help="v2 JSONL dataset.")
    parser.add_argument("--input-parquet", action="append", default=[],
                        help="Local parquet file/dir/glob with fen/depth/cp/mate fields. Repeatable.")
    parser.add_argument("--input-features", type=Path, default=None,
                        help="Precomputed feature directory from precompute_features.py.")
    parser.add_argument("--input-repeat", type=int, default=1,
                        help="When mixing --input JSONL with parquet, repeat JSONL train rows this many times.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-path", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional compatible checkpoint used to initialize the model.")
    parser.add_argument("--halfka-new-features-only", action="store_true",
                        help="Freeze the warm-started network and train only the new HalfKAv2 king rows.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--arch",
                        choices=[
                            "v2",
                            "linear-head",
                            "linear-head-crelu",
                            "linear-head-screlu",
                            "linear-head-screlu-hm",
                            "linear-head-screlu-hm-buckets",
                            "linear-head-screlu-hm-buckets-psqt",
                            "linear-head-screlu-halfka-hm-buckets-psqt",
                            "bottleneck-head",
                            "bottleneck-head-crelu",
                            "bottleneck-head-screlu",
                        ],
                        default="v2",
                        help="Model architecture. *-screlu uses squared clipped ReLU and fixed integer activation scales.")
    parser.add_argument("--feature-dim", type=int, default=160)
    parser.add_argument("--bottleneck-dim", type=int, default=64,
                        help="Bottleneck width for --arch bottleneck-head.")
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-depth", type=int, default=18)
    parser.add_argument("--min-knodes", type=int, default=0)
    parser.add_argument("--cp-scale", type=float, default=600.0,
                        help="Centipawn scale used for tanh/WDL-ish targets and exported cp mapping.")
    parser.add_argument("--train-samples", type=int, default=200000)
    parser.add_argument("--val-samples", type=int, default=50000)
    parser.add_argument("--val-buckets", type=int, default=10)
    parser.add_argument("--allow-train-val-fallback", action="store_true",
                        help="Use train/all rows for validation if the hash val split is empty.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--parquet-batch-size", type=int, default=262144)
    parser.add_argument("--slow-parquet-labels", action="store_true",
                        help="Use python-chess terminal/tactical labeling for parquet rows; much slower.")
    parser.add_argument("--shuffle-parquet", action="store_true",
                        help="Shuffle parquet row-group order and rows within each read batch.")
    parser.add_argument("--terminal-fixtures", action="store_true",
                        help="Append canonical terminal rows when training directly from parquet.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every-batches", type=int, default=0,
                        help="Save a partial checkpoint/export every N train batches. 0 disables periodic saves.")
    parser.add_argument("--no-export", action="store_true")
    return parser


def make_loader(dataset: IterableDataset,
                batch_size: int,
                args: argparse.Namespace,
                device: torch.device) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def main() -> int:
    args = build_parser().parse_args()
    if args.input is None and args.input_features is None and not args.input_parquet:
        raise SystemExit("provide --input, --input-features, or --input-parquet")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[init] input={args.input}", flush=True)
    if args.input_parquet:
        print(f"[init] input_parquet={args.input_parquet}", flush=True)
    if args.input_features is not None:
        print(f"[init] input_features={args.input_features}", flush=True)
    print(f"[init] output_dir={args.output_dir}", flush=True)
    print(f"[init] device={device}", flush=True)
    halfka = args.arch == "linear-head-screlu-halfka-hm-buckets-psqt"
    encoder = encode_fen_halfka if halfka else encode_fen

    def make_dataset(split: str, sample_limit: int) -> IterableDataset:
        if args.input_features is not None:
            if halfka:
                raise ValueError("HalfKAv2 training requires FEN-backed JSONL or parquet input")
            return PrecomputedFeatureDataset(
                args.input_features,
                split=split,
                val_buckets=args.val_buckets,
                sample_limit=sample_limit,
                shuffle=bool(args.shuffle_parquet and split == "train"),
                seed=args.seed,
            )
        if args.input_parquet:
            parquet_dataset = ParquetValueDataset(
                args.input_parquet,
                split=split,
                val_buckets=args.val_buckets,
                sample_limit=sample_limit,
                min_depth=args.min_depth,
                min_knodes=args.min_knodes,
                cp_scale=args.cp_scale,
                batch_size=args.parquet_batch_size,
                fast=not args.slow_parquet_labels,
                terminal_fixtures=args.terminal_fixtures,
                shuffle=args.shuffle_parquet,
                seed=args.seed,
                encoder=encoder,
            )
            if args.input is None or split == "val":
                return parquet_dataset
            json_dataset: IterableDataset = V2ValueDataset(
                args.input,
                split="all" if split == "train" else split,
                val_buckets=args.val_buckets,
                sample_limit=0,
                encoder=encoder,
            )
            if split == "train":
                json_dataset = RepeatedIterableDataset(json_dataset, args.input_repeat)
            return ChainedIterableDataset([parquet_dataset, json_dataset])
        if args.input is None:
            raise ValueError("--input is required when not using --input-features or --input-parquet")
        return V2ValueDataset(
            args.input,
            split=split,
            val_buckets=args.val_buckets,
            sample_limit=sample_limit,
            encoder=encoder,
        )

    train_loader = make_loader(
        make_dataset("train", args.train_samples),
        args.batch_size,
        args,
        device,
    )

    def make_val_loader(split: str) -> DataLoader:
        return make_loader(
            make_dataset(split, args.val_samples),
            args.batch_size,
            args,
            device,
        )

    val_loader = make_val_loader("val")
    model = build_value_model(
        args.arch,
        accumulator_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = payload["model_state"]
        if halfka and state["accumulator.weight"].shape[0] == MIRRORED_DUMMY_FEATURE_INDEX + 1:
            target = model.state_dict()
            for name, value in state.items():
                if name not in {"accumulator.weight", "psqt.weight"}:
                    target[name] = value
            for name in ("accumulator.weight", "psqt.weight"):
                source = state[name]
                destination = target[name]
                for king_bucket in range(32):
                    for plane in range(10):
                        old_start = (king_bucket * 10 + plane) * 64
                        new_start = (king_bucket * 11 + plane) * 64
                        destination[new_start:new_start + 64].copy_(source[old_start:old_start + 64])
                    king_start = (king_bucket * 11 + 10) * 64
                    destination[king_start:king_start + 64].zero_()
                target[name] = destination
            model.load_state_dict(target)
            print("[init] expanded HalfKP weights into king-aware HalfKAv2", flush=True)
        else:
            model.load_state_dict(state)
        print(f"[init] loaded checkpoint={args.init_checkpoint}", flush=True)
    if args.halfka_new_features_only:
        if not halfka or args.init_checkpoint is None:
            raise ValueError("--halfka-new-features-only requires HalfKAv2 and --init-checkpoint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name in ("accumulator", "psqt"):
            weight = getattr(model, name).weight
            weight.requires_grad_(True)
            mask = torch.zeros_like(weight)
            for king_bucket in range(32):
                king_start = (king_bucket * 11 + 10) * 64
                mask[king_start:king_start + 64].fill_(1.0)
            weight.register_hook(lambda grad, row_mask=mask: grad * row_mask)
        print("[init] training only zero-initialized HalfKAv2 king rows", flush=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=0.0 if args.halfka_new_features_only else args.weight_decay,
    )
    mirrored = args.arch in {
        "linear-head-screlu-hm",
        "linear-head-screlu-hm-buckets",
        "linear-head-screlu-hm-buckets-psqt",
        "linear-head-screlu-halfka-hm-buckets-psqt",
    }

    best_loss: float | None = None
    best_path: Path | None = None
    history: list[dict[str, Any]] = []

    def export_partial(checkpoint_path: Path, epoch: int, batch_idx: int, reason: str) -> None:
        if args.no_export:
            return
        suffix = f"epoch_{epoch:02d}_{reason}"
        if batch_idx > 0:
            suffix = f"epoch_{epoch:02d}_batch_{batch_idx:06d}_{reason}"
        export_path = args.output_dir / f"nn_eval_{suffix}.bin"
        export_checkpoint(checkpoint_path, export_path)
        print(f"[export:{reason}] {export_path}", flush=True)

    for epoch in range(1, args.epochs + 1):
        def save_during_epoch(batch_idx: int, train_metrics: dict[str, float], reason: str) -> None:
            should_save = reason == "interrupt"
            if reason == "periodic" and args.save_every_batches > 0 and batch_idx > 0:
                should_save = (batch_idx % args.save_every_batches) == 0
            if not should_save:
                return
            safe_reason = "interrupt" if reason == "interrupt" else "partial"
            name = f"checkpoint_epoch_{epoch:02d}_{safe_reason}.pt"
            if batch_idx > 0:
                name = f"checkpoint_epoch_{epoch:02d}_batch_{batch_idx:06d}_{safe_reason}.pt"
            val_metrics = {"loss": 0.0, "mae": 0.0, "batches": 0.0, "rows": 0.0}
            checkpoint_path = save_checkpoint(
                args.output_dir,
                model,
                optimizer,
                epoch,
                train_metrics,
                val_metrics,
                args,
                name=name,
                update_best=False,
            )
            print(
                f"[checkpoint:{safe_reason}] {checkpoint_path} "
                f"rows={int(train_metrics.get('rows', 0.0))} loss={train_metrics.get('loss', 0.0):.5f} "
                f"mae={train_metrics.get('mae', 0.0):.5f}",
                flush=True,
            )
            export_partial(checkpoint_path, epoch, batch_idx, safe_reason)

        print(f"[epoch {epoch}] train", flush=True)
        try:
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.log_every,
                mirrored=mirrored,
                halfka=halfka,
                save_callback=save_during_epoch,
            )
        except TrainingInterrupted as exc:
            train_metrics = exc.metrics
            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": {"loss": 0.0, "mae": 0.0, "batches": 0.0, "rows": 0.0},
                "interrupted": True,
            })
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "history.json").open("w", encoding="utf-8") as fp:
                json.dump({"history": history, "best_loss": best_loss, "interrupted": True}, fp, indent=2, sort_keys=True)
            print(
                f"[interrupted] rows={int(train_metrics.get('rows', 0.0))} "
                f"loss={train_metrics.get('loss', 0.0):.5f} mae={train_metrics.get('mae', 0.0):.5f}",
                flush=True,
            )
            return 130
        print(f"[epoch {epoch}] val", flush=True)
        val_metrics = evaluate(model, val_loader, device, mirrored=mirrored, halfka=halfka)
        if val_metrics["rows"] == 0.0 and args.allow_train_val_fallback:
            print("[epoch {epoch}] val split empty; falling back to all rows for smoke validation".format(epoch=epoch), flush=True)
            val_loader = make_val_loader("all")
            val_metrics = evaluate(model, val_loader, device, mirrored=mirrored, halfka=halfka)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.5f} "
            f"train_mae={train_metrics['mae']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_mae={val_metrics['mae']:.5f}",
            flush=True,
        )
        if best_loss is None or val_metrics["loss"] <= best_loss:
            best_loss = val_metrics["loss"]
            best_path = save_checkpoint(args.output_dir, model, optimizer, epoch, train_metrics, val_metrics, args)
            if not args.no_export:
                export_checkpoint(best_path, args.export_path)
                print(f"[export] {args.export_path}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump({"history": history, "best_loss": best_loss}, fp, indent=2, sort_keys=True)
    print(f"[done] best={best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
