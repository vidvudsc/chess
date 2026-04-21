#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from features import DUMMY_FEATURE_INDEX, encode_fen
from model import HalfKpValueNet
from training_data import TrainingSample, count_training_samples, iter_training_samples
from export_inference import export_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "src" / "core" / "bot" / "nn" / "model"
DEFAULT_INPUT = REPO_ROOT / "data" / "labels" / "stockfish_san_10m.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR
DEFAULT_EXPORT_PATH = DEFAULT_MODEL_DIR / "nn_eval.bin"


@dataclass
class BatchTensors:
    white_indices: torch.Tensor
    white_offsets: torch.Tensor
    black_indices: torch.Tensor
    black_offsets: torch.Tensor
    stm_white: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor
    is_mate: torch.Tensor


def log(msg: str) -> None:
    print(msg, flush=True)


def jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def buffered_shuffle(items: Iterable[TrainingSample],
                     buffer_size: int,
                     seed: int) -> Iterator[TrainingSample]:
    rng = random.Random(seed)
    buf: List[TrainingSample] = []
    for item in items:
        if buffer_size <= 1:
            yield item
            continue
        if len(buf) < buffer_size:
            buf.append(item)
            continue
        idx = rng.randrange(buffer_size + 1)
        if idx < buffer_size:
            yield buf[idx]
            buf[idx] = item
        else:
            yield item
    rng.shuffle(buf)
    yield from buf


class StreamingHalfKpDataset(IterableDataset):
    def __init__(self,
                 input_path: Path,
                 split: str,
                 min_depth: int,
                 include_mates: bool,
                 shuffle_buffer: int,
                 sample_limit: int,
                 seed: int) -> None:
        super().__init__()
        self.input_path = input_path
        self.split = split
        self.min_depth = min_depth
        self.include_mates = include_mates
        self.shuffle_buffer = shuffle_buffer
        self.sample_limit = sample_limit
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, object]]:
        stream: Iterable[TrainingSample] = iter_training_samples(
            self.input_path,
            min_depth=self.min_depth,
            include_mates=self.include_mates,
            split=self.split,
        )
        if self.split == "train" and self.shuffle_buffer > 1:
            stream = buffered_shuffle(stream, self.shuffle_buffer, self.seed)
        if self.sample_limit > 0:
            stream = itertools.islice(stream, self.sample_limit)

        for sample in stream:
            white_half, black_half, stm_white = encode_fen(sample.fen)
            yield {
                "white_half": white_half,
                "black_half": black_half,
                "stm_white": stm_white,
                "target": sample.value_target,
                "weight": sample.weight,
                "is_mate": sample.is_mate,
            }


def collate_batch(samples: List[Dict[str, object]]) -> BatchTensors:
    white_indices: List[int] = []
    white_offsets: List[int] = []
    black_indices: List[int] = []
    black_offsets: List[int] = []
    stm_white: List[bool] = []
    target: List[float] = []
    weight: List[float] = []
    is_mate: List[bool] = []

    for sample in samples:
        white_offsets.append(len(white_indices))
        black_offsets.append(len(black_indices))
        white_half = list(sample["white_half"])
        black_half = list(sample["black_half"])
        if not white_half:
            white_half = [DUMMY_FEATURE_INDEX]
        if not black_half:
            black_half = [DUMMY_FEATURE_INDEX]
        white_indices.extend(white_half)
        black_indices.extend(black_half)
        stm_white.append(bool(sample["stm_white"]))
        target.append(float(sample["target"]))
        weight.append(float(sample["weight"]))
        is_mate.append(bool(sample["is_mate"]))

    return BatchTensors(
        white_indices=torch.tensor(white_indices, dtype=torch.long),
        white_offsets=torch.tensor(white_offsets, dtype=torch.long),
        black_indices=torch.tensor(black_indices, dtype=torch.long),
        black_offsets=torch.tensor(black_offsets, dtype=torch.long),
        stm_white=torch.tensor(stm_white, dtype=torch.bool),
        target=torch.tensor(target, dtype=torch.float32),
        weight=torch.tensor(weight, dtype=torch.float32),
        is_mate=torch.tensor(is_mate, dtype=torch.bool),
    )


def move_batch_to_device(batch: BatchTensors, device: torch.device) -> BatchTensors:
    return BatchTensors(
        white_indices=batch.white_indices.to(device),
        white_offsets=batch.white_offsets.to(device),
        black_indices=batch.black_indices.to(device),
        black_offsets=batch.black_offsets.to(device),
        stm_white=batch.stm_white.to(device),
        target=batch.target.to(device),
        weight=batch.weight.to(device),
        is_mate=batch.is_mate.to(device),
    )


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def estimated_batches(sample_limit: int, batch_size: int) -> int | None:
    if sample_limit <= 0:
        return None
    return max(1, math.ceil(sample_limit / batch_size))


def build_scheduler(args: argparse.Namespace,
                    optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
    schedule = args.lr_schedule
    if schedule == "none":
        return None
    if schedule == "cosine":
        eta_min = args.lr * args.min_lr_ratio
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=eta_min,
        )
    raise ValueError(f"unsupported lr schedule: {schedule}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def count_batches_from_stream(path: Path,
                              split: str,
                              min_depth: int,
                              include_mates: bool,
                              batch_size: int) -> int:
    rows = count_training_samples(
        path,
        min_depth=min_depth,
        include_mates=include_mates,
        split=split,
    )
    return max(1, math.ceil(rows / batch_size)) if rows > 0 else 0


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             progress_desc: str,
             progress_total: int | None,
             enable_tqdm: bool) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_weight = 0.0
    batches = 0

    iterator = loader
    progress = None
    if enable_tqdm and tqdm is not None:
        progress = tqdm(loader, desc=progress_desc, total=progress_total, leave=False)
        iterator = progress

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
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
        total_abs += float(mae.item())
        total_weight += float(batch.weight.sum().item())
        batches += 1
        if progress is not None:
            progress.set_postfix(loss=f"{total_loss / batches:.4f}", mae=f"{total_abs / batches:.4f}")

    if batches == 0:
        return {"loss": 0.0, "mae": 0.0, "batches": 0}

    return {
        "loss": total_loss / batches,
        "mae": total_abs / batches,
        "batches": float(batches),
        "weight_seen": total_weight,
    }


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    log_every: int,
                    progress_desc: str,
                    progress_total: int | None,
                    enable_tqdm: bool) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    batches = 0
    started = time.time()

    iterator = loader
    progress = None
    if enable_tqdm and tqdm is not None:
        progress = tqdm(loader, desc=progress_desc, total=progress_total, leave=False)
        iterator = progress

    for batch_idx, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device)
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
        if progress is not None:
            progress.set_postfix(loss=f"{total_loss / batches:.4f}", mae=f"{total_mae / batches:.4f}")

        if log_every > 0 and batch_idx % log_every == 0:
            if progress is None:
                elapsed = time.time() - started
                log(
                    f"[train] batch={batch_idx} "
                    f"loss={total_loss / batches:.4f} "
                    f"mae={total_mae / batches:.4f} "
                    f"elapsed={elapsed:.1f}s"
                )

    return {
        "loss": total_loss / max(batches, 1),
        "mae": total_mae / max(batches, 1),
        "batches": float(batches),
        "elapsed_s": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first value-only NNUE-style model from streamed JSONL data")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input stockfish SAN JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints and metadata")
    parser.add_argument("--min-depth", type=int, default=15, help="Minimum depth to keep")
    parser.add_argument("--exclude-mates", action="store_true", help="Drop mate rows")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine",
                        help="Learning-rate schedule across epochs")
    parser.add_argument("--min-lr-ratio", type=float, default=0.10,
                        help="Final LR ratio for cosine decay, as a fraction of the base LR")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--feature-dim", type=int, default=160, help="Accumulator dimension")
    parser.add_argument("--hidden-dim", type=int, default=40, help="Final hidden dimension")
    parser.add_argument("--train-samples-per-epoch", type=int, default=200000, help="Filtered train samples per epoch, 0 means full stream")
    parser.add_argument("--val-samples", type=int, default=50000, help="Filtered validation samples, 0 means full stream")
    parser.add_argument("--shuffle-buffer", type=int, default=20000, help="Buffered shuffle size for the training stream")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=100, help="Train progress interval in batches")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--export-best-path", type=Path, default=DEFAULT_EXPORT_PATH,
                        help="Path to write the current best inference model binary")
    return parser


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        requested = torch.device(raw)
        if requested.type == "mps":
            log("[init] MPS requested, but EmbeddingBag is not supported there in this pipeline; falling back to CPU")
            return torch.device("cpu")
        return requested
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(output_dir: Path,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                    epoch: int,
                    train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float],
                    args: argparse.Namespace,
                    best: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": jsonable_args(args),
    }
    epoch_path = output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
    torch.save(payload, epoch_path)
    if best:
        torch.save(payload, output_dir / "best.pt")


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[init] input={args.input}")
    log(f"[init] output_dir={output_dir}")
    log(f"[init] device={device}")
    log(f"[init] min_depth={args.min_depth} exclude_mates={args.exclude_mates}")

    train_dataset = StreamingHalfKpDataset(
        input_path=args.input,
        split="train",
        min_depth=args.min_depth,
        include_mates=not args.exclude_mates,
        shuffle_buffer=args.shuffle_buffer,
        sample_limit=args.train_samples_per_epoch,
        seed=args.seed,
    )
    val_dataset = StreamingHalfKpDataset(
        input_path=args.input,
        split="val",
        min_depth=args.min_depth,
        include_mates=not args.exclude_mates,
        shuffle_buffer=0,
        sample_limit=args.val_samples,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        num_workers=0,
    )

    enable_tqdm = not args.no_tqdm
    train_total_batches = estimated_batches(args.train_samples_per_epoch, args.batch_size)
    val_total_batches = estimated_batches(args.val_samples, args.batch_size)
    include_mates = not args.exclude_mates
    if enable_tqdm and train_total_batches is None:
        log("[init] counting filtered train rows for tqdm total...")
        train_total_batches = count_batches_from_stream(
            args.input,
            split="train",
            min_depth=args.min_depth,
            include_mates=include_mates,
            batch_size=args.batch_size,
        )
        log(f"[init] train_batches={train_total_batches}")
    if enable_tqdm and val_total_batches is None:
        log("[init] counting filtered val rows for tqdm total...")
        val_total_batches = count_batches_from_stream(
            args.input,
            split="val",
            min_depth=args.min_depth,
            include_mates=include_mates,
            batch_size=args.batch_size,
        )
        log(f"[init] val_batches={val_total_batches}")

    model = HalfKpValueNet(
        accumulator_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer)

    history = []
    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        log(f"[epoch {epoch}] train lr={current_lr(optimizer):.6g}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.log_every,
            progress_desc=f"train {epoch}/{args.epochs}",
            progress_total=train_total_batches,
            enable_tqdm=enable_tqdm,
        )
        log(f"[epoch {epoch}] val")
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            progress_desc=f"val {epoch}/{args.epochs}",
            progress_total=val_total_batches,
            enable_tqdm=enable_tqdm,
        )
        best = best_val_loss is None or val_metrics["loss"] < best_val_loss
        if best:
            best_val_loss = val_metrics["loss"]

        save_checkpoint(output_dir, model, optimizer, scheduler, epoch, train_metrics, val_metrics, args, best=best)
        if best and args.export_best_path:
            export_checkpoint(output_dir / "best.pt", args.export_best_path)
        if scheduler is not None:
            scheduler.step()
        history.append({
            "epoch": epoch,
            "lr": current_lr(optimizer),
            "train": train_metrics,
            "val": val_metrics,
        })
        log(
            f"[epoch {epoch}] "
            f"lr={current_lr(optimizer):.6g} "
            f"train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.4f}"
        )

    metadata = {
        "args": jsonable_args(args),
        "device": str(device),
        "history": history,
    }
    with (output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    log(f"[done] wrote checkpoints to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
