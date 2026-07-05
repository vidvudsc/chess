#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset


NN_ROOT = Path(__file__).resolve().parents[1]
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from export_inference import export_checkpoint  # noqa: E402
from features import DUMMY_FEATURE_INDEX, encode_fen  # noqa: E402
from model import HalfKpValueNet  # noqa: E402


DEFAULT_OUTPUT_DIR = NN_ROOT / "model" / "v2"
DEFAULT_EXPORT_PATH = NN_ROOT / "model" / "nn_eval.bin"


@dataclass
class Batch:
    white_indices: torch.Tensor
    white_offsets: torch.Tensor
    black_indices: torch.Tensor
    black_offsets: torch.Tensor
    stm_white: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor


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


def iter_rows(path: Path,
              split: str,
              val_buckets: int,
              sample_limit: int) -> Iterator[dict[str, Any]]:
    yielded = 0
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
            yielded += 1
            yield row


class V2ValueDataset(IterableDataset):
    def __init__(self,
                 input_path: Path,
                 split: str,
                 val_buckets: int,
                 sample_limit: int) -> None:
        super().__init__()
        self.input_path = input_path
        self.split = split
        self.val_buckets = val_buckets
        self.sample_limit = sample_limit

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in iter_rows(
            self.input_path,
            split=self.split,
            val_buckets=self.val_buckets,
            sample_limit=self.sample_limit,
        ):
            white_half, black_half, stm_white = encode_fen(str(row["fen"]))
            yield {
                "white_half": white_half or [DUMMY_FEATURE_INDEX],
                "black_half": black_half or [DUMMY_FEATURE_INDEX],
                "stm_white": stm_white,
                "target": float(row["target_stm"]),
                "weight": sample_weight(row),
            }


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


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    batches = 0
    rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
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
                log_every: int) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    batches = 0
    rows = 0
    started = time.time()
    for idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
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
        if log_every > 0 and idx % log_every == 0:
            print(
                f"[train] batch={idx} rows={rows} "
                f"loss={total_loss / batches:.5f} mae={total_mae / batches:.5f}",
                flush=True,
            )
    if batches == 0:
        return {"loss": 0.0, "mae": 0.0, "batches": 0.0, "rows": 0.0, "elapsed_s": 0.0}
    return {
        "loss": total_loss / batches,
        "mae": total_mae / batches,
        "batches": float(batches),
        "rows": float(rows),
        "elapsed_s": time.time() - started,
    }


def save_checkpoint(output_dir: Path,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    train_metrics: dict[str, float],
                    val_metrics: dict[str, float],
                    args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    path = output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
    torch.save(payload, path)
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
    parser.add_argument("--input", type=Path, required=True, help="v2 JSONL dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--export-path", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-samples", type=int, default=200000)
    parser.add_argument("--val-samples", type=int, default=50000)
    parser.add_argument("--val-buckets", type=int, default=10)
    parser.add_argument("--allow-train-val-fallback", action="store_true",
                        help="Use train/all rows for validation if the hash val split is empty.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-export", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[init] input={args.input}", flush=True)
    print(f"[init] output_dir={args.output_dir}", flush=True)
    print(f"[init] device={device}", flush=True)

    train_loader = DataLoader(
        V2ValueDataset(args.input, split="train", val_buckets=args.val_buckets, sample_limit=args.train_samples),
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=0,
    )
    def make_val_loader(split: str) -> DataLoader:
        return DataLoader(
            V2ValueDataset(args.input, split=split, val_buckets=args.val_buckets, sample_limit=args.val_samples),
            batch_size=args.batch_size,
            collate_fn=collate,
            num_workers=0,
        )

    val_loader = make_val_loader("val")
    model = HalfKpValueNet(accumulator_dim=args.feature_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss: float | None = None
    best_path: Path | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch}] train", flush=True)
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.log_every)
        print(f"[epoch {epoch}] val", flush=True)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["rows"] == 0.0 and args.allow_train_val_fallback:
            print("[epoch {epoch}] val split empty; falling back to all rows for smoke validation".format(epoch=epoch), flush=True)
            val_loader = make_val_loader("all")
            val_metrics = evaluate(model, val_loader, device)
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
