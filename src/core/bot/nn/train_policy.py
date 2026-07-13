#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

NN_ROOT = Path(__file__).resolve().parent
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from policy_model import ConvPolicyValueNet, MOVE_INDEX_COUNT  # noqa: E402


DEFAULT_OUTPUT_DIR = NN_ROOT / "model" / "policy"


class NpzPolicyDataset(Dataset):
    def __init__(self, path: Path, sample_limit: int = 0, eval_scale: float = 600.0) -> None:
        payload = np.load(path)
        states = payload["states"]
        plans = payload["plans"]
        evals = payload["evals"]
        total = int(states.shape[0])
        if sample_limit > 0:
            total = min(total, sample_limit)
        self.states = states[:total].astype(np.float32, copy=False)
        raw_moves = plans[:total, 0, 0] if plans.ndim == 3 else plans[:total, 0]
        self.moves = raw_moves.astype(np.int64, copy=False)
        raw_evals = evals[:total].astype(np.float32, copy=False)
        if raw_evals.size > 0 and float(np.nanmax(np.abs(raw_evals))) <= 1.25:
            self.evals = np.clip(raw_evals, -1.0, 1.0).astype(np.float32, copy=False)
        else:
            self.evals = np.tanh(raw_evals / max(float(eval_scale), 1.0)).astype(np.float32, copy=False)
        if self.states.ndim != 4 or self.states.shape[1:] != (15, 8, 8):
            raise ValueError(f"expected states shape (N,15,8,8), got {self.states.shape}")
        if np.any(self.moves < 0) or np.any(self.moves >= MOVE_INDEX_COUNT):
            raise ValueError("policy move index out of 0..4095 range")

    def __len__(self) -> int:
        return int(self.moves.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.states[index]),
            torch.tensor(int(self.moves[index]), dtype=torch.long),
            torch.tensor(float(self.evals[index]), dtype=torch.float32),
        )


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model: ConvPolicyValueNet,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer | None,
              device: torch.device,
              value_weight: float,
              log_every: int = 100) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_correct = 0
    total_top3 = 0
    total_top5 = 0
    total_rows = 0
    start = time.time()

    for batch_index, (states, moves, values) in enumerate(loader, start=1):
        states = states.to(device, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits, pred_values = model(states)
            policy_loss = F.cross_entropy(logits, moves)
            value_loss = F.mse_loss(pred_values, values)
            loss = policy_loss + value_weight * value_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        rows = int(states.shape[0])
        total_rows += rows
        total_loss += float(loss.item()) * rows
        total_policy += float(policy_loss.item()) * rows
        total_value += float(value_loss.item()) * rows
        total_correct += int((logits.argmax(dim=1) == moves).sum().item())
        max_k = min(5, logits.shape[1])
        topk = logits.topk(k=max_k, dim=1).indices
        total_top3 += int((topk[:, :min(3, max_k)] == moves.unsqueeze(1)).any(dim=1).sum().item())
        total_top5 += int((topk == moves.unsqueeze(1)).any(dim=1).sum().item())
        if log_every > 0 and batch_index % log_every == 0:
            print(
                f"[{'train' if training else 'val'}] batch={batch_index} rows={total_rows} "
                f"loss={total_loss / total_rows:.4f} policy={total_policy / total_rows:.4f} "
                f"value={total_value / total_rows:.4f} top1={total_correct / total_rows:.4f} "
                f"top5={total_top5 / total_rows:.4f}",
                flush=True,
            )

    rows_f = float(max(total_rows, 1))
    return {
        "rows": float(total_rows),
        "loss": total_loss / rows_f,
        "policy_loss": total_policy / rows_f,
        "value_loss": total_value / rows_f,
        "top1": total_correct / rows_f,
        "top3": total_top3 / rows_f,
        "top5": total_top5 / rows_f,
        "elapsed_s": time.time() - start,
    }


def save_checkpoint(output_dir: Path,
                    model: ConvPolicyValueNet,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    metrics: dict[str, Any],
                    args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    torch.save(payload, path)
    torch.save(payload, output_dir / "best.pt")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a chess policy/value network from Chess-Alpha-style NPZ tensors.")
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--value-hidden", type=int, default=128)
    parser.add_argument("--value-weight", type=float, default=0.10)
    parser.add_argument("--eval-scale", type=float, default=600.0,
                        help="Centipawn scale used for tanh value targets.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[init] input_npz={args.input_npz}", flush=True)
    print(f"[init] output_dir={args.output_dir}", flush=True)
    print(f"[init] device={device}", flush=True)

    dataset = NpzPolicyDataset(args.input_npz, sample_limit=args.sample_limit, eval_scale=args.eval_scale)
    val_count = int(math.ceil(len(dataset) * args.val_fraction))
    val_count = min(max(val_count, 1), max(len(dataset) - 1, 1))
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=generator)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    model = ConvPolicyValueNet(
        channels=args.channels,
        blocks=args.blocks,
        value_hidden=args.value_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    best_path: Path | None = None
    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch}] train", flush=True)
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.value_weight, args.log_every)
        print(f"[epoch {epoch}] val", flush=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, args.value_weight, args.log_every)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"[epoch {epoch}] train_top1={train_metrics['top1']:.4f} val_top1={val_metrics['top1']:.4f} "
            f"val_top3={val_metrics['top3']:.4f} val_top5={val_metrics['top5']:.4f} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}",
            flush=True,
        )
        if val_metrics["top1"] >= best_top1:
            best_top1 = val_metrics["top1"]
            best_path = save_checkpoint(args.output_dir, model, optimizer, epoch, row, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump({"history": history, "best_top1": best_top1, "best": str(best_path)}, fp, indent=2)
        fp.write("\n")
    print(f"[done] best={best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
