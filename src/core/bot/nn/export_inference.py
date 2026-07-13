#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from features import DUMMY_FEATURE_INDEX, HALFKP_DIM
from model import build_value_model
MAGIC = b"CHNNUE1\0"
VERSION_V2 = 2
VERSION_LINEAR_HEAD = 3
VERSION_BOTTLENECK_HEAD = 4
VERSION_LINEAR_HEAD_FIXED = 5
VERSION_BOTTLENECK_HEAD_FIXED = 6
VERSION_LINEAR_HEAD_SCRELU = 7
VERSION_BOTTLENECK_HEAD_SCRELU = 8
VERSION_LINEAR_HEAD_SCRELU_I16_ACC = 9
VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS = 10
VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT = 11
ACT_QMAX = 127
PSQT_ARCHES = {
    "linear-head-screlu-hm-buckets-psqt",
    "linear-head-screlu-halfka-hm-buckets-psqt",
}
BUCKET_ARCHES = {"linear-head-screlu-hm-buckets", *PSQT_ARCHES}


def quantize_tensor(values: torch.Tensor, qmax: int, dtype: np.dtype) -> tuple[np.ndarray, float]:
    arr = values.detach().cpu().contiguous().float().numpy()
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    scale = max_abs / float(qmax) if max_abs > 1e-12 else 1.0
    q = np.clip(np.round(arr / scale), -qmax, qmax).astype(dtype, copy=False)
    return q, float(scale)


def load_model_from_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    # Checkpoints are produced by our own trainer; args may hold Path objects,
    # which the torch>=2.6 weights-only unpickler rejects.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    arch = str(args.get("arch", "v2"))
    feature_dim = int(args.get("feature_dim", 128))
    hidden_dim = int(args.get("hidden_dim", 32))
    bottleneck_dim = int(args.get("bottleneck_dim", 64))
    model = build_value_model(
        arch,
        accumulator_dim=feature_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, args


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    model, args = load_model_from_checkpoint(checkpoint_path)
    arch = str(args.get("arch", "v2"))
    cp_scale = float(args.get("cp_scale", 600.0))
    feature_dim = int(model.accumulator.embedding_dim)
    if arch in BUCKET_ARCHES:
        hidden_dim = int(model.fc1_weight.shape[1])
    else:
        hidden_dim = int(
            model.fc1.out_features
            if arch in {"linear-head", "linear-head-crelu", "linear-head-screlu", "linear-head-screlu-hm"}
            else model.fc2.out_features
        )
    halfkp_dim = int(model.accumulator.num_embeddings - 1)
    dummy_feature_index = int(model.accumulator.padding_idx)
    bottleneck_dim = int(model.fc1.out_features) if arch in {
        "bottleneck-head",
        "bottleneck-head-crelu",
        "bottleneck-head-screlu",
    } else 0

    acc_w = model.accumulator.weight.detach().cpu().contiguous().float()
    psqt_w_q = None
    psqt_scale = 1.0
    if arch in BUCKET_ARCHES:
        num_buckets = int(model.fc1_weight.shape[0])
        fc1_w = model.fc1_weight.detach().cpu().contiguous().float()
        fc1_b = model.fc1_bias.detach().cpu().contiguous().float()
        out_w = model.out_weight.detach().cpu().contiguous().float().view(-1)
        out_b = model.out_bias.detach().cpu().contiguous().float().view(-1)
        if arch in PSQT_ARCHES:
            psqt_w = model.psqt.weight.detach().cpu().contiguous().float()
            psqt_w_q, psqt_scale = quantize_tensor(psqt_w, 32767, np.int16)
        else:
            psqt_w_q = None
            psqt_scale = 1.0
    else:
        num_buckets = 0
        fc1_w = model.fc1.weight.detach().cpu().contiguous().float()
        fc1_b = model.fc1.bias.detach().cpu().contiguous().float()
        out_w = model.out.weight.detach().cpu().contiguous().float().view(-1)
        out_b = model.out.bias.detach().cpu().contiguous().float().view(-1)

    # Packed SCReLU inference keeps the incrementally updated accumulator in
    # int16. Limiting each feature row to +/-255 keeps the worst legal chess
    # position comfortably inside that range while halving stack copy traffic.
    acc_qmax = 255 if arch in {"linear-head-screlu", "linear-head-screlu-hm", *BUCKET_ARCHES} else 32767
    acc_w_q, acc_scale = quantize_tensor(acc_w, acc_qmax, np.int16)
    fc1_w_q, fc1_scale = quantize_tensor(fc1_w, 127, np.int8)
    out_w_q, out_scale = quantize_tensor(out_w, 127, np.int8)
    if arch in {"v2", "bottleneck-head", "bottleneck-head-crelu", "bottleneck-head-screlu"}:
        fc2_w = model.fc2.weight.detach().cpu().contiguous().float()
        fc2_b = model.fc2.bias.detach().cpu().contiguous().float()
        fc2_w_q, fc2_scale = quantize_tensor(fc2_w, 127, np.int8)
        if arch == "bottleneck-head-screlu":
            version = VERSION_BOTTLENECK_HEAD_SCRELU
        elif arch == "bottleneck-head-crelu":
            version = VERSION_BOTTLENECK_HEAD_FIXED
        elif arch == "bottleneck-head":
            version = VERSION_BOTTLENECK_HEAD
        else:
            version = VERSION_V2
    elif arch == "linear-head":
        fc2_b = None
        fc2_w_q = None
        fc2_scale = 1.0
        version = VERSION_LINEAR_HEAD
    elif arch == "linear-head-crelu":
        fc2_b = None
        fc2_w_q = None
        fc2_scale = 1.0
        version = VERSION_LINEAR_HEAD_FIXED
    elif arch in {"linear-head-screlu", "linear-head-screlu-hm"}:
        fc2_b = None
        fc2_w_q = None
        fc2_scale = 1.0
        version = VERSION_LINEAR_HEAD_SCRELU_I16_ACC
    elif arch == "linear-head-screlu-hm-buckets":
        fc2_b = None
        fc2_w_q = None
        fc2_scale = 1.0
        version = VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS
    elif arch in PSQT_ARCHES:
        fc2_b = None
        fc2_w_q = None
        fc2_scale = 1.0
        version = VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT
    else:
        raise ValueError(f"unsupported export architecture: {arch}")
    fixed_activation = arch in {
        "linear-head-crelu",
        "bottleneck-head-crelu",
        "linear-head-screlu",
        "linear-head-screlu-hm",
        "linear-head-screlu-hm-buckets",
        "linear-head-screlu-hm-buckets-psqt",
        "linear-head-screlu-halfka-hm-buckets-psqt",
        "bottleneck-head-screlu",
    }
    act0_scale = 1.0 / float(ACT_QMAX)
    act1_scale = 1.0 / float(ACT_QMAX)
    act2_scale = 1.0 / float(ACT_QMAX)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack(
            "<IIIII f f f f f",
            version,
            halfkp_dim,
            feature_dim,
            hidden_dim,
            dummy_feature_index,
            cp_scale,
            acc_scale,
            fc1_scale,
            fc2_scale,
            out_scale,
        ))
        if fixed_activation:
            fp.write(struct.pack("<fff", act0_scale, act1_scale, act2_scale))
        if arch in BUCKET_ARCHES:
            fp.write(struct.pack("<I", num_buckets))
        if arch in PSQT_ARCHES:
            fp.write(struct.pack("<f", psqt_scale))
        if arch == "bottleneck-head":
            fp.write(struct.pack("<I", bottleneck_dim))
        if arch == "bottleneck-head-crelu":
            fp.write(struct.pack("<I", bottleneck_dim))
        if arch == "bottleneck-head-screlu":
            fp.write(struct.pack("<I", bottleneck_dim))
        fp.write(acc_w_q.tobytes(order="C"))
        fp.write(fc1_w_q.tobytes(order="C"))
        fp.write(fc1_b.numpy().tobytes(order="C"))
        if arch in {"v2", "bottleneck-head", "bottleneck-head-crelu", "bottleneck-head-screlu"}:
            fp.write(fc2_w_q.tobytes(order="C"))
            fp.write(fc2_b.numpy().tobytes(order="C"))
        fp.write(out_w_q.tobytes(order="C"))
        fp.write(out_b.numpy().tobytes(order="C"))
        if psqt_w_q is not None:
            fp.write(psqt_w_q.tobytes(order="C"))


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
