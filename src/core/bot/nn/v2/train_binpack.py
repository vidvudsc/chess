#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


NN_ROOT = Path(__file__).resolve().parents[1]
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from export_inference import export_checkpoint  # noqa: E402
from features import (  # noqa: E402
    DUMMY_FEATURE_INDEX,
    MIRRORED_DUMMY_FEATURE_INDEX,
)
from model import build_value_model  # noqa: E402


def import_binpack_loader(nnue_pytorch_dir: Path):
    root = nnue_pytorch_dir.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    previous = Path.cwd()
    os.chdir(root)
    try:
        import data_loader  # type: ignore
    finally:
        os.chdir(previous)
    return data_loader


def mirror_halfkp_indices(indices: torch.Tensor) -> torch.Tensor:
    valid = indices >= 0
    safe = indices.clamp_min(0).long()
    king = torch.div(safe, 640, rounding_mode="floor")
    remainder = safe.remainder(640)
    plane = torch.div(remainder, 64, rounding_mode="floor")
    piece_square = remainder.remainder(64)
    mirror = king.remainder(8) < 4
    king = torch.where(mirror, torch.bitwise_xor(king, 7), king)
    piece_square = torch.where(mirror, torch.bitwise_xor(piece_square, 7), piece_square)
    king_bucket = torch.div(king, 8, rounding_mode="floor") * 4 + king.remainder(8) - 4
    mapped = (king_bucket * 10 + plane) * 64 + piece_square
    return torch.where(valid, mapped, torch.full_like(mapped, MIRRORED_DUMMY_FEATURE_INDEX))


def model_batch(raw: tuple[torch.Tensor, ...], device: torch.device, mirrored: bool = False):
    us, _them, white, black, outcome, score, _piece_count = raw
    batch_size, width = white.shape
    if mirrored:
        white = mirror_halfkp_indices(white.to(device)).reshape(-1)
        black = mirror_halfkp_indices(black.to(device)).reshape(-1)
    else:
        white = white.masked_fill(white < 0, DUMMY_FEATURE_INDEX).reshape(-1).long().to(device)
        black = black.masked_fill(black < 0, DUMMY_FEATURE_INDEX).reshape(-1).long().to(device)
    offsets = torch.arange(0, batch_size * width, width, dtype=torch.long, device=device)
    return (
        white,
        offsets,
        black,
        offsets,
        us.reshape(-1).bool().to(device),
        outcome.reshape(-1).to(device),
        score.reshape(-1).to(device),
    )


def wdl_eval_loss(pred: torch.Tensor,
                  score: torch.Tensor,
                  outcome: torch.Tensor,
                  cp_scale: float,
                  score_lambda: float,
                  power: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Stockfish binpack scores use 208 units per pawn; convert to centipawns.
    score_cp = score * (100.0 / 208.0)
    eval_probability = 0.5 * (1.0 + torch.tanh(score_cp / cp_scale))
    target_probability = score_lambda * eval_probability + (1.0 - score_lambda) * outcome
    pred_probability = 0.5 * (1.0 + pred)
    error = torch.abs(pred_probability - target_probability)
    return torch.mean(error.pow(power)), target_probability


def save(output_dir: Path,
         model: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         args: argparse.Namespace,
         batch_index: int,
         metrics: dict[str, float],
         name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_metrics": metrics,
        "val_metrics": {},
        "batch_index": batch_index,
        "args": {
            k: str(v) if isinstance(v, Path)
            else [str(item) for item in v] if isinstance(v, list)
            else v
            for k, v in vars(args).items()
        },
    }
    checkpoint = output_dir / f"{name}.pt"
    torch.save(payload, checkpoint)
    export_checkpoint(checkpoint, output_dir / f"{name}.bin")
    return checkpoint


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the engine NNUE directly from Stockfish binpack data.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--extra-input", type=Path, action="append", default=[],
                   help="Additional binpacks mixed by the native sparse loader.")
    p.add_argument("--nnue-pytorch-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Optional model checkpoint used to initialize a compatible architecture.")
    p.add_argument("--arch", default="linear-head-screlu")
    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--cp-scale", type=float, default=600.0)
    p.add_argument("--score-lambda", type=float, default=0.70)
    p.add_argument("--loss-power", type=float, default=2.5)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--batches", type=int, default=6000)
    p.add_argument("--loader-workers", type=int, default=1)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lr-end", type=float, default=None,
                   help="Final cosine-decayed learning rate. Defaults to --lr (constant schedule).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--unfiltered", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_loader = import_binpack_loader(args.nnue_pytorch_dir)
    config = data_loader.DataloaderSkipConfig(
        filtered=not args.unfiltered,
        wld_filtered=not args.unfiltered,
        soft_early_fen_skipping=20 if not args.unfiltered else -1,
    )
    provider = data_loader.SparseBatchProvider(
        "LegacyHalfKP",
        [str(path.resolve()) for path in [args.input, *args.extra_input]],
        args.batch_size,
        cyclic=False,
        num_workers=args.loader_workers,
        config=config,
        use_pinned_memory=device.type == "cuda",
        device="cpu",
    )
    model = build_value_model(
        args.arch,
        accumulator_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        print(f"[init] loaded checkpoint={args.init_checkpoint}", flush=True)
    mirrored = args.arch in {
        "linear-head-screlu-hm",
        "linear-head-screlu-hm-buckets",
        "linear-head-screlu-hm-buckets-psqt",
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_end = args.lr if args.lr_end is None else args.lr_end
    started = time.time()
    loss_sum = 0.0
    mae_sum = 0.0
    rows = 0

    for batch_index in range(1, args.batches + 1):
        progress = (batch_index - 1) / max(args.batches - 1, 1)
        learning_rate = lr_end + 0.5 * (args.lr - lr_end) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        try:
            raw = next(provider)
        except StopIteration:
            print(f"[data] exhausted after {batch_index - 1} batches", flush=True)
            break
        white, white_offsets, black, black_offsets, stm, outcome, score = model_batch(raw, device, mirrored)
        optimizer.zero_grad(set_to_none=True)
        pred = model(white, white_offsets, black, black_offsets, stm)
        loss, target_probability = wdl_eval_loss(
            pred, score, outcome, args.cp_scale, args.score_lambda, args.loss_power
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mae = torch.mean(torch.abs(0.5 * (1.0 + pred) - target_probability))
        loss_sum += float(loss.item())
        mae_sum += float(mae.item())
        rows += int(pred.numel())
        metrics = {
            "loss": loss_sum / batch_index,
            "probability_mae": mae_sum / batch_index,
            "rows": float(rows),
            "elapsed_s": time.time() - started,
            "learning_rate": learning_rate,
        }
        if args.log_every and batch_index % args.log_every == 0:
            rate = rows / max(metrics["elapsed_s"], 1e-9)
            print(
                f"[train] batch={batch_index}/{args.batches} rows={rows} "
                f"loss={metrics['loss']:.6f} pmae={metrics['probability_mae']:.5f} "
                f"positions_s={rate:.0f}",
                flush=True,
            )
        if args.save_every and batch_index % args.save_every == 0:
            save(args.output_dir, model, optimizer, args, batch_index, metrics, f"batch_{batch_index:06d}")

    completed_batches = rows // args.batch_size
    final_metrics = {
        "loss": loss_sum / max(completed_batches, 1),
        "probability_mae": mae_sum / max(completed_batches, 1),
        "rows": float(rows),
        "elapsed_s": time.time() - started,
    }
    checkpoint = save(args.output_dir, model, optimizer, args, completed_batches, final_metrics, "best")
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    print(f"[done] checkpoint={checkpoint} rows={rows}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
