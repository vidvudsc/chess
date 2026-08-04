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
    MIRRORED_HALFKA_DIM,
    MIRRORED_HALFKA_THREATS_DUMMY_FEATURE_INDEX,
)
from model import build_value_model  # noqa: E402
from v2.train_value import initialize_from_checkpoint  # noqa: E402


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


def remap_native_halfka_threats(indices: torch.Tensor) -> torch.Tensor:
    """Compress nnue-pytorch's 12 HalfKA planes into the engine's 11 planes."""
    native_halfka_dim = 32 * 12 * 64
    valid = indices >= 0
    halfka = valid & (indices < native_halfka_dim)
    safe = indices.clamp_min(0).long()
    # nnue-pytorch numbers mirrored king buckets from h8 back toward e1,
    # whereas the engine numbers them from e1 toward h8. Squares are already
    # oriented in the sparse indices, but the bucket order must be reversed.
    bucket = 31 - torch.div(safe, 12 * 64, rounding_mode="floor")
    within = safe.remainder(12 * 64)
    plane12 = torch.div(within, 64, rounding_mode="floor")
    square = within.remainder(64)
    plane11 = torch.where(
        plane12 >= 10,
        torch.full_like(plane12, 10),
        torch.div(plane12, 2, rounding_mode="floor") + plane12.remainder(2) * 5,
    )
    halfka_index = (bucket * 11 + plane11) * 64 + square
    threat_index = MIRRORED_HALFKA_DIM + safe - native_halfka_dim
    mapped = torch.where(halfka, halfka_index, threat_index)
    return torch.where(
        valid, mapped, torch.full_like(mapped, MIRRORED_HALFKA_THREATS_DUMMY_FEATURE_INDEX)
    )


def native_model_batch(raw: tuple[torch.Tensor, ...], device: torch.device, feature_set: str):
    us, _them, white, black, outcome, score, _piece_count = raw
    if feature_set != "HalfKAv2_hm+Full_Threats":
        raise ValueError(f"unsupported native feature mapping: {feature_set}")
    white = remap_native_halfka_threats(white)
    black = remap_native_halfka_threats(black)
    batch_size, width = white.shape
    offsets = torch.arange(0, batch_size * width, width, dtype=torch.long, device=device)
    return (
        white.reshape(-1).long().to(device),
        offsets,
        black.reshape(-1).long().to(device),
        offsets,
        us.reshape(-1).bool().to(device),
        outcome.reshape(-1).to(device),
        score.reshape(-1).to(device),
    )


def configure_new_channel_training(model: torch.nn.Module,
                                   source_state: dict[str, torch.Tensor]) -> int:
    """Freeze a widened model except for newly added transformer channels."""
    source_accumulator = source_state.get("accumulator.weight")
    if source_accumulator is None or not hasattr(model, "accumulator"):
        raise ValueError("source and target must have an accumulator")
    accumulator = model.accumulator.weight
    source_width = int(source_accumulator.shape[1])
    target_width = int(accumulator.shape[1])
    if source_width <= 0 or source_width >= target_width:
        raise ValueError(
            f"target accumulator width {target_width} must exceed source width {source_width}"
        )
    if not hasattr(model, "accumulator_factor") or not hasattr(model, "fc1_weight"):
        raise ValueError("new-channel training requires a factorized bucketed linear head")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    accumulator.requires_grad_(True)
    accumulator_mask = torch.zeros_like(accumulator)
    accumulator_mask[:, source_width:].fill_(1.0)
    accumulator.register_hook(lambda grad, row_mask=accumulator_mask: grad * row_mask)

    factor = model.accumulator_factor.weight
    if factor.shape[1] != target_width:
        raise ValueError("accumulator factor width does not match accumulator width")
    factor.requires_grad_(True)
    factor_mask = torch.zeros_like(factor)
    factor_mask[:, source_width:].fill_(1.0)
    factor.register_hook(lambda grad, row_mask=factor_mask: grad * row_mask)

    fc1_weight = model.fc1_weight
    if fc1_weight.ndim != 3 or fc1_weight.shape[2] != target_width * 2:
        raise ValueError("fc1 input width does not match both accumulator perspectives")
    fc1_weight.requires_grad_(True)
    fc1_mask = torch.zeros_like(fc1_weight)
    fc1_mask[:, :, source_width:target_width].fill_(1.0)
    fc1_mask[:, :, target_width + source_width:target_width * 2].fill_(1.0)
    fc1_weight.register_hook(lambda grad, channel_mask=fc1_mask: grad * channel_mask)
    return source_width


def configure_new_hidden_training(model: torch.nn.Module,
                                  source_state: dict[str, torch.Tensor]) -> int:
    """Freeze a warm start except for newly added linear-head hidden units."""
    source_fc1 = source_state.get("fc1_weight")
    source_out = source_state.get("out_weight")
    if source_fc1 is None or source_out is None:
        raise ValueError("source must have a bucketed linear head")
    if not hasattr(model, "fc1_weight") or not hasattr(model, "out_weight"):
        raise ValueError("target must have a bucketed linear head")

    fc1_weight = model.fc1_weight
    out_weight = model.out_weight
    if (source_fc1.ndim != 3 or fc1_weight.ndim != 3 or
            source_fc1.shape[0] != fc1_weight.shape[0] or
            source_fc1.shape[2] != fc1_weight.shape[2]):
        raise ValueError("new-hidden training requires identical buckets and head inputs")
    source_hidden = int(source_fc1.shape[1])
    target_hidden = int(fc1_weight.shape[1])
    if source_hidden <= 0 or source_hidden >= target_hidden:
        raise ValueError(
            f"target hidden width {target_hidden} must exceed source width {source_hidden}"
        )
    if (source_out.ndim != 2 or out_weight.ndim != 2 or
            source_out.shape[0] != out_weight.shape[0] or
            int(source_out.shape[1]) != source_hidden or
            int(out_weight.shape[1]) != target_hidden):
        raise ValueError("output head does not match the widened hidden layer")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    fc1_weight.requires_grad_(True)
    fc1_mask = torch.zeros_like(fc1_weight)
    fc1_mask[:, source_hidden:].fill_(1.0)
    fc1_weight.register_hook(lambda grad, hidden_mask=fc1_mask: grad * hidden_mask)

    fc1_bias = model.fc1_bias
    fc1_bias.requires_grad_(True)
    bias_mask = torch.zeros_like(fc1_bias)
    bias_mask[:, source_hidden:].fill_(1.0)
    fc1_bias.register_hook(lambda grad, hidden_mask=bias_mask: grad * hidden_mask)

    out_weight.requires_grad_(True)
    out_mask = torch.zeros_like(out_weight)
    out_mask[:, source_hidden:].fill_(1.0)
    out_weight.register_hook(lambda grad, hidden_mask=out_mask: grad * hidden_mask)
    return source_hidden


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
         loader_batch_index: int,
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
        "loader_batch_index": loader_batch_index,
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
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Directory of raw .binpack shards; combined with --input/--extra-input.")
    p.add_argument("--extra-input", type=Path, action="append", default=[],
                   help="Additional binpacks mixed by the native sparse loader.")
    p.add_argument("--nnue-pytorch-dir", type=Path, required=True)
    p.add_argument("--feature-set", default="LegacyHalfKP",
                   help="Native loader feature order. V13 requires HalfKAv2_hm+Full_Threats.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Optional model checkpoint used to initialize a compatible architecture.")
    p.add_argument("--resume-checkpoint", type=Path, default=None,
                   help="Resume model, optimizer, metrics, and global batch index from a saved checkpoint.")
    p.add_argument("--resume-fresh-loader", action="store_true",
                   help="Resume optimizer/schedule state but start the supplied dataset at its beginning.")
    p.add_argument("--resume-loader-batch", type=int, default=None,
                   help="Override native-loader batches to fast-forward when resuming.")
    p.add_argument("--new-features-only", action="store_true",
                   help="Freeze the warm-started model and train only new HalfKA king/threat rows.")
    p.add_argument("--new-channels-only", action="store_true",
                   help="For a widened warm start, train only added accumulator channels and head inputs.")
    p.add_argument("--new-hidden-only", action="store_true",
                   help="For a wider linear head, train only newly added hidden units and outputs.")
    p.add_argument(
        "--freeze-accumulator",
        action="store_true",
        help=(
            "Freeze the warm-started feature transformer while fine-tuning "
            "the bucketed value head and PSQT."
        ),
    )
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
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise SystemExit("--init-checkpoint and --resume-checkpoint are mutually exclusive")
    staged_modes = sum((
        bool(args.new_features_only),
        bool(args.new_channels_only),
        bool(args.new_hidden_only),
        bool(args.freeze_accumulator),
    ))
    if staged_modes > 1:
        raise SystemExit(
            "--new-features-only, --new-channels-only, --new-hidden-only, "
            "and --freeze-accumulator are mutually exclusive"
        )
    if staged_modes and args.init_checkpoint is None:
        raise SystemExit("staged training modes require --init-checkpoint")
    if args.resume_fresh_loader and args.resume_checkpoint is None:
        raise SystemExit("--resume-fresh-loader requires --resume-checkpoint")
    if args.resume_loader_batch is not None and args.resume_checkpoint is None:
        raise SystemExit("--resume-loader-batch requires --resume-checkpoint")
    if args.resume_loader_batch is not None and args.resume_loader_batch < 0:
        raise SystemExit("--resume-loader-batch must be non-negative")
    if args.resume_fresh_loader and args.resume_loader_batch is not None:
        raise SystemExit("--resume-fresh-loader and --resume-loader-batch are mutually exclusive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_loader = import_binpack_loader(args.nnue_pytorch_dir)
    config = data_loader.DataloaderSkipConfig(
        filtered=not args.unfiltered,
        wld_filtered=not args.unfiltered,
        soft_early_fen_skipping=20 if not args.unfiltered else -1,
    )
    input_paths = ([args.input] if args.input is not None else []) + list(args.extra_input)
    if args.input_dir is not None:
        input_paths.extend(sorted(args.input_dir.glob("*.binpack")))
    input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    if not input_paths:
        raise SystemExit("provide --input, --extra-input, or --input-dir with .binpack files")
    print(f"[data] binpacks={len(input_paths)}", flush=True)
    provider = data_loader.SparseBatchProvider(
        args.feature_set,
        [str(path) for path in input_paths],
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
    resume_payload = None
    start_batch = 0
    loader_batch_index = 0
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model_state"], strict=True)
        start_batch = int(resume_payload.get("batch_index", 0))
        loader_batch_index = int(resume_payload.get("loader_batch_index", start_batch))
        if args.resume_fresh_loader:
            loader_batch_index = 0
        elif args.resume_loader_batch is not None:
            loader_batch_index = args.resume_loader_batch
        if start_batch < 0 or start_batch >= args.batches:
            raise SystemExit(
                f"resume batch {start_batch} must be between 0 and --batches {args.batches - 1}"
            )
        print(
            f"[resume] checkpoint={args.resume_checkpoint} batch={start_batch}/{args.batches} "
            f"loader_batch={loader_batch_index}",
            flush=True,
        )
    init_payload = None
    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        copied = initialize_from_checkpoint(
            model, init_payload["model_state"], halfka="halfka" in args.arch
        )
        print(f"[init] checkpoint={args.init_checkpoint} copied={','.join(copied)}", flush=True)
    if args.new_features_only:
        if args.feature_set != "HalfKAv2_hm+Full_Threats" or args.init_checkpoint is None:
            raise SystemExit("--new-features-only requires a warm-started HalfKAv2_hm+Full_Threats model")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        weight = model.accumulator.weight
        weight.requires_grad_(True)
        mask = torch.zeros_like(weight)
        for king_bucket in range(32):
            king_start = (king_bucket * 11 + 10) * 64
            mask[king_start:king_start + 64].fill_(1.0)
        mask[MIRRORED_HALFKA_DIM:MIRRORED_HALFKA_THREATS_DUMMY_FEATURE_INDEX].fill_(1.0)
        weight.register_hook(lambda grad, row_mask=mask: grad * row_mask)
        print("[init] training only new HalfKAv2 king and Full Threats rows", flush=True)
    if args.new_channels_only:
        if args.feature_set != "HalfKAv2_hm+Full_Threats" or init_payload is None:
            raise SystemExit("--new-channels-only requires a warm-started Full Threats model")
        try:
            source_width = configure_new_channel_training(model, init_payload["model_state"])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"[init] training only widened transformer channels "
            f"{source_width}:{model.accumulator.weight.shape[1]}",
            flush=True,
        )
    if args.new_hidden_only:
        if args.feature_set != "HalfKAv2_hm+Full_Threats" or init_payload is None:
            raise SystemExit("--new-hidden-only requires a warm-started Full Threats model")
        try:
            source_hidden = configure_new_hidden_training(model, init_payload["model_state"])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"[init] training only widened head units "
            f"{source_hidden}:{model.fc1_weight.shape[1]}",
            flush=True,
        )
    if args.freeze_accumulator:
        if init_payload is None:
            raise SystemExit("--freeze-accumulator requires --init-checkpoint")
        for parameter in model.accumulator.parameters():
            parameter.requires_grad_(False)
        if hasattr(model, "accumulator_factor"):
            for parameter in model.accumulator_factor.parameters():
                parameter.requires_grad_(False)
        print("[init] feature transformer frozen; training value head and PSQT", flush=True)
    mirrored = args.arch in {
        "linear-head-screlu-hm",
        "linear-head-screlu-hm-buckets",
        "linear-head-screlu-hm-buckets-psqt",
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=0.0 if (args.new_features_only or args.new_channels_only or
                             args.new_hidden_only or args.freeze_accumulator)
        else args.weight_decay,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        resumed_weight_decay = (
            0.0 if (args.new_features_only or args.new_channels_only or
                    args.new_hidden_only or args.freeze_accumulator)
            else args.weight_decay
        )
        for group in optimizer.param_groups:
            group["weight_decay"] = resumed_weight_decay
    lr_end = args.lr if args.lr_end is None else args.lr_end
    previous_metrics = resume_payload.get("train_metrics", {}) if resume_payload is not None else {}
    previous_elapsed = float(previous_metrics.get("elapsed_s", 0.0))
    started = time.time() - previous_elapsed
    loss_sum = float(previous_metrics.get("loss", 0.0)) * start_batch
    mae_sum = float(previous_metrics.get("probability_mae", 0.0)) * start_batch
    rows = int(previous_metrics.get("rows", start_batch * args.batch_size))

    if loader_batch_index:
        skipped_started = time.time()
        for skipped_batch in range(1, loader_batch_index + 1):
            try:
                next(provider)
            except StopIteration:
                raise SystemExit(
                    "data exhausted while fast-forwarding at batch "
                    f"{skipped_batch - 1}/{loader_batch_index}"
                )
            if args.log_every and skipped_batch % max(args.log_every * 10, 1000) == 0:
                elapsed = time.time() - skipped_started
                print(
                    f"[resume] loader_fast_forward={skipped_batch}/{loader_batch_index} "
                    f"batches_s={skipped_batch / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )
        print(
            f"[resume] loader_fast_forward={loader_batch_index} "
            f"elapsed_s={time.time() - skipped_started:.1f}",
            flush=True,
        )
    elif start_batch:
        print("[resume] starting supplied dataset from its beginning", flush=True)

    completed_batches = start_batch
    for batch_index in range(start_batch + 1, args.batches + 1):
        progress = (batch_index - 1) / max(args.batches - 1, 1)
        learning_rate = lr_end + 0.5 * (args.lr - lr_end) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        try:
            raw = next(provider)
        except StopIteration:
            print(f"[data] exhausted after {batch_index - 1} batches", flush=True)
            break
        loader_batch_index += 1
        if args.feature_set == "LegacyHalfKP":
            white, white_offsets, black, black_offsets, stm, outcome, score = model_batch(
                raw, device, mirrored
            )
        else:
            white, white_offsets, black, black_offsets, stm, outcome, score = native_model_batch(
                raw, device, args.feature_set
            )
        optimizer.zero_grad(set_to_none=True)
        pred = model(white, white_offsets, black, black_offsets, stm)
        loss, target_probability = wdl_eval_loss(
            pred, score, outcome, args.cp_scale, args.score_lambda, args.loss_power
        )
        loss.backward()
        optimizer.step()
        completed_batches = batch_index
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
            save(
                args.output_dir,
                model,
                optimizer,
                args,
                batch_index,
                loader_batch_index,
                metrics,
                f"batch_{batch_index:06d}",
            )

    final_metrics = {
        "loss": loss_sum / max(completed_batches, 1),
        "probability_mae": mae_sum / max(completed_batches, 1),
        "rows": float(rows),
        "elapsed_s": time.time() - started,
    }
    checkpoint = save(
        args.output_dir,
        model,
        optimizer,
        args,
        completed_batches,
        loader_batch_index,
        final_metrics,
        "best",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    print(f"[done] checkpoint={checkpoint} rows={rows}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
