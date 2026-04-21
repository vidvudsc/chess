#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from features import DUMMY_FEATURE_INDEX, HALFKP_DIM
from model import HalfKpValueNet
from training_data import DEFAULT_CP_SCALE

MAGIC = b"CHNNUE1\0"
VERSION = 2


def quantize_tensor(values: torch.Tensor, qmax: int, dtype: np.dtype) -> tuple[np.ndarray, float]:
    arr = values.detach().cpu().contiguous().float().numpy()
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    scale = max_abs / float(qmax) if max_abs > 1e-12 else 1.0
    q = np.clip(np.round(arr / scale), -qmax, qmax).astype(dtype, copy=False)
    return q, float(scale)


def load_model_from_checkpoint(checkpoint_path: Path) -> tuple[HalfKpValueNet, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    args = payload.get("args", {})
    feature_dim = int(args.get("feature_dim", 128))
    hidden_dim = int(args.get("hidden_dim", 32))
    model = HalfKpValueNet(accumulator_dim=feature_dim, hidden_dim=hidden_dim)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, args


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    model, _args = load_model_from_checkpoint(checkpoint_path)
    feature_dim = int(model.accumulator.embedding_dim)
    hidden_dim = int(model.fc2.out_features)

    acc_w = model.accumulator.weight.detach().cpu().contiguous().float()
    fc1_w = model.fc1.weight.detach().cpu().contiguous().float()
    fc1_b = model.fc1.bias.detach().cpu().contiguous().float()
    fc2_w = model.fc2.weight.detach().cpu().contiguous().float()
    fc2_b = model.fc2.bias.detach().cpu().contiguous().float()
    out_w = model.out.weight.detach().cpu().contiguous().float().view(-1)
    out_b = model.out.bias.detach().cpu().contiguous().float().view(-1)

    acc_w_q, acc_scale = quantize_tensor(acc_w, 32767, np.int16)
    fc1_w_q, fc1_scale = quantize_tensor(fc1_w, 127, np.int8)
    fc2_w_q, fc2_scale = quantize_tensor(fc2_w, 127, np.int8)
    out_w_q, out_scale = quantize_tensor(out_w, 127, np.int8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack(
            "<IIIII f f f f f",
            VERSION,
            HALFKP_DIM,
            feature_dim,
            hidden_dim,
            DUMMY_FEATURE_INDEX,
            float(DEFAULT_CP_SCALE),
            acc_scale,
            fc1_scale,
            fc2_scale,
            out_scale,
        ))
        fp.write(acc_w_q.tobytes(order="C"))
        fp.write(fc1_w_q.tobytes(order="C"))
        fp.write(fc1_b.numpy().tobytes(order="C"))
        fp.write(fc2_w_q.tobytes(order="C"))
        fp.write(fc2_b.numpy().tobytes(order="C"))
        fp.write(out_w_q.tobytes(order="C"))
        fp.write(out_b.numpy().tobytes(order="C"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained NN checkpoint to the C inference binary format")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a training checkpoint .pt file")
    parser.add_argument("--output", type=Path, required=True, help="Output inference model path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_checkpoint(args.checkpoint, args.output)
    print(f"[done] exported inference model to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
