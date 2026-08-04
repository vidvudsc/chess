#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NN_ROOT = ROOT / "src" / "core" / "bot" / "nn"
sys.path.insert(0, str(NN_ROOT))

from export_inference import export_checkpoint  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpolate two compatible NN checkpoints and export the result."
    )
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint to receive the update vector. With this set, "
            "output = target + alpha * (candidate - base)."
        ),
    )
    p.add_argument("--alpha", type=float, required=True,
                   help="Interpolation or update-vector scale in [0, 1].")
    p.add_argument("--output-checkpoint", type=Path, required=True)
    p.add_argument("--output-bin", type=Path, default=None)
    return p


def main() -> int:
    args = parser().parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    target = (
        torch.load(args.target, map_location="cpu", weights_only=False)
        if args.target is not None
        else base
    )
    base_state = base["model_state"]
    candidate_state = candidate["model_state"]
    target_state = target["model_state"]
    if base_state.keys() != candidate_state.keys() or base_state.keys() != target_state.keys():
        raise SystemExit("checkpoint model-state keys differ")

    blended_state: dict[str, torch.Tensor] = {}
    for name, base_value in base_state.items():
        candidate_value = candidate_state[name]
        target_value = target_state[name]
        if (base_value.shape != candidate_value.shape or
                base_value.shape != target_value.shape or
                base_value.dtype != candidate_value.dtype or
                base_value.dtype != target_value.dtype):
            raise SystemExit(f"incompatible tensor: {name}")
        if torch.is_floating_point(base_value):
            if args.target is None:
                blended = torch.lerp(
                    base_value.float(), candidate_value.float(), args.alpha
                )
            else:
                blended = (
                    target_value.float() +
                    args.alpha * (candidate_value.float() - base_value.float())
                )
            blended_state[name] = blended.to(base_value.dtype)
        else:
            if (not torch.equal(base_value, candidate_value) or
                    not torch.equal(base_value, target_value)):
                raise SystemExit(f"non-floating tensor differs: {name}")
            blended_state[name] = base_value.clone()

    payload = {
        "epoch": max(
            int(base.get("epoch", 0)),
            int(candidate.get("epoch", 0)),
            int(target.get("epoch", 0)),
        ),
        "model_state": blended_state,
        "train_metrics": {},
        "val_metrics": {},
        "batch_index": 0,
        "loader_batch_index": 0,
        "args": target.get("args", candidate.get("args", base.get("args", {}))),
        "interpolation": {
            "base": str(args.base),
            "candidate": str(args.candidate),
            "target": str(args.target) if args.target is not None else None,
            "alpha": args.alpha,
        },
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_checkpoint)

    output_bin = args.output_bin
    if output_bin is None:
        output_bin = args.output_checkpoint.with_suffix(".bin")
    export_checkpoint(args.output_checkpoint, output_bin)
    print(
        f"[done] alpha={args.alpha:.6f} checkpoint={args.output_checkpoint} "
        f"bin={output_bin}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
