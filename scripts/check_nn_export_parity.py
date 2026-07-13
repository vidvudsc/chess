#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
NN_ROOT = ROOT / "src" / "core" / "bot" / "nn"
sys.path.insert(0, str(NN_ROOT))

from export_inference import export_checkpoint  # noqa: E402
from features import MIRRORED_HALFKA_DIM, MIRRORED_HALFKP_DIM, encode_fen, encode_fen_halfka  # noqa: E402
from model import build_value_model  # noqa: E402


DEFAULT_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 13",
    "r3r1k1/pp1n1ppp/2p2n2/3p4/3P1B2/2N2N2/PP3PPP/R3R1K1 b - - 0 14",
    "8/5pk1/6p1/8/3P4/5KP1/8/8 w - - 0 42",
    "8/1p3pk1/p5p1/3P4/8/1P3KP1/8/8 b - - 0 39",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
    "7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
]


def ensure_probe() -> Path:
    probe = ROOT / "bin" / "nn_eval_probe"
    if not probe.exists():
        subprocess.run(["make", "bin/nn_eval_probe"], cwd=ROOT, check=True)
    return probe


def load_checkpoint(checkpoint: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu")
    args = payload.get("args", {})
    model = build_value_model(
        str(args.get("arch", "v2")),
        accumulator_dim=int(args.get("feature_dim", 128)),
        hidden_dim=int(args.get("hidden_dim", 32)),
        bottleneck_dim=int(args.get("bottleneck_dim", 64)),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, args


def mirrored_index(index: int, planes: int = 10) -> int:
    king, remainder = divmod(index, planes * 64)
    plane, piece_square = divmod(remainder, 64)
    if king % 8 < 4:
        king ^= 7
        piece_square ^= 7
    king_bucket = (king // 8) * 4 + king % 8 - 4
    return (king_bucket * planes + plane) * 64 + piece_square


def tensorize_fen(fen: str,
                  mirrored: bool = False,
                  halfka: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    white, black, stm_white = encode_fen_halfka(fen) if halfka else encode_fen(fen)
    if mirrored:
        planes = 11 if halfka else 10
        dummy = MIRRORED_HALFKA_DIM if halfka else MIRRORED_HALFKP_DIM
        white = [mirrored_index(index, planes) for index in white] or [dummy]
        black = [mirrored_index(index, planes) for index in black] or [dummy]
    return (
        torch.tensor(white, dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor(black, dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([stm_white], dtype=torch.bool),
    )


def pytorch_cp(model: torch.nn.Module, fen: str, cp_scale: float) -> float:
    with torch.no_grad():
        halfka = model.accumulator.num_embeddings == MIRRORED_HALFKA_DIM + 1
        mirrored = halfka or model.accumulator.num_embeddings == MIRRORED_HALFKP_DIM + 1
        value = model(*tensorize_fen(fen, mirrored=mirrored, halfka=halfka)).item()
    return value * cp_scale


def c_cps(probe: Path, model_bin: Path, fens: list[str]) -> list[int]:
    proc = subprocess.run(
        [str(probe), str(model_bin), *fens],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows: list[int] = []
    for line in proc.stdout.splitlines():
        cp, _fen = line.split("\t", 1)
        rows.append(int(cp))
    if len(rows) != len(fens):
        raise RuntimeError(f"expected {len(fens)} probe rows, got {len(rows)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PyTorch NN checkpoint output with exported C inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-bin", type=Path, default=None,
                        help="Existing exported nn_eval.bin. If omitted, export to a temp file next to the checkpoint.")
    parser.add_argument("--fen", action="append", default=[], help="Extra FEN to check; repeatable.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-mean-abs-cp", type=float, default=20.0)
    parser.add_argument("--max-abs-cp", type=float, default=80.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    model_bin = args.model_bin.resolve() if args.model_bin else checkpoint.with_name("parity_nn_eval.bin")
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    if args.model_bin is None or not model_bin.exists():
        export_checkpoint(checkpoint, model_bin)

    model, meta = load_checkpoint(checkpoint)
    cp_scale = float(meta.get("cp_scale", 600.0))
    probe = ensure_probe()
    fens = DEFAULT_FENS + list(args.fen)
    py_rows = [pytorch_cp(model, fen, cp_scale) for fen in fens]
    c_rows = c_cps(probe, model_bin, fens)
    rows = [
        {
            "fen": fen,
            "pytorch_cp": py,
            "c_cp": c,
            "abs_diff_cp": abs(py - c),
        }
        for fen, py, c in zip(fens, py_rows, c_rows)
    ]
    diffs = [row["abs_diff_cp"] for row in rows]
    summary = {
        "checkpoint": str(checkpoint),
        "model_bin": str(model_bin),
        "positions": len(rows),
        "mean_abs_diff_cp": mean(diffs),
        "max_abs_diff_cp": max(diffs),
        "rows": rows,
    }

    print(json.dumps(summary, indent=2), flush=True)
    if args.report is not None:
        args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.report.resolve().open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
            fp.write("\n")

    if summary["mean_abs_diff_cp"] > args.max_mean_abs_cp or summary["max_abs_diff_cp"] > args.max_abs_cp:
        raise SystemExit(
            "export parity failed: "
            f"mean_abs={summary['mean_abs_diff_cp']:.2f}, max_abs={summary['max_abs_diff_cp']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
