"""Per-square saliency maps for the production NNUE net.

Two layers per position, both from the real trained model:
- contrib_cp: leave-one-out influence — white-POV eval change (cp) when the
  piece on that square is removed (kings excluded; removal would be illegal).
- energy: accumulator activation energy — L1 mass of the first-layer rows
  attributable to that square's features (piece-placement + threat rows,
  both perspectives), i.e. how loudly the square drives the accumulator.
"""
import json
import math
import sys
from pathlib import Path

import torch

import os

ROOT = Path(os.environ.get("CHESS_ROOT", "/Users/vidvudscalitis/Desktop/CODING/Chess"))
sys.path.insert(0, str(ROOT / "src" / "core" / "bot" / "nn"))

from export_inference import load_model_from_checkpoint  # noqa: E402
from features import (  # noqa: E402
    HALFKA_DIM,
    MIRRORED_HALFKA_DIM,
    encode_fen_halfka_threats,
)
from v2.train_value import mirror_halfkp_indices  # noqa: E402


def to_model_space(indices):
    """Map encode_fen_halfka_threats output into the mirrored model space."""
    t = torch.tensor(indices, dtype=torch.long)
    base = mirror_halfkp_indices(
        t, planes=11,
        source_dummy=-1_000_000, target_dummy=-1_000_000,
    )
    threat = MIRRORED_HALFKA_DIM + (t - HALFKA_DIM)
    return torch.where(t < HALFKA_DIM, base, threat)

SCRATCH = Path(__file__).parent
CHECKPOINT = Path(os.environ.get(
    "NN_CHECKPOINT", str(SCRATCH / "prod_checkpoint.pt")))
CP_SCALE = 600.0

POSITIONS = [
    ("Italian middlegame — pieces aimed at f7/kingside",
     "r1bq1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8"),
    ("Queenside space vs kingside attack",
     "r2q1rk1/pb2bppp/1pn1pn2/2p5/2P5/1PN1PN2/PB1P1PPP/R2Q1RK1 w - - 0 10"),
    ("Outside passed pawn endgame",
     "8/5pk1/6p1/1P6/8/6P1/5PK1/8 w - - 0 40"),
    ("King and pawn endgame — opposition",
     "8/8/4k3/8/8/4K3/4P3/8 w - - 0 50"),
]


def fen_board(fen):
    """Return {square_index: piece_char} from the FEN board field."""
    board = {}
    ranks = fen.split()[0].split("/")
    for r, row in enumerate(ranks):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                board[(7 - r) * 8 + f] = ch
                f += 1
    return board


def fen_without(fen, sq):
    """FEN with the piece on square index sq removed."""
    parts = fen.split()
    rows = parts[0].split("/")
    r = 7 - sq // 8
    f = sq % 8
    row = rows[r]
    expanded = []
    for ch in row:
        if ch.isdigit():
            expanded.extend(["1"] * int(ch))
        else:
            expanded.append(ch)
    expanded[f] = "1"
    out = []
    run = 0
    for ch in expanded:
        if ch == "1":
            run += 1
        else:
            if run:
                out.append(str(run))
                run = 0
            out.append(ch)
    if run:
        out.append(str(run))
    rows[r] = "".join(out)
    parts[0] = "/".join(rows)
    return " ".join(parts)


def evaluate(model, fen):
    raw_white, raw_black, stm_white = encode_fen_halfka_threats(fen)
    white = to_model_space(raw_white)
    black = to_model_space(raw_black)
    with torch.no_grad():
        out = model(
            white,
            torch.tensor([0], dtype=torch.long),
            black,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([stm_white], dtype=torch.bool),
        )
    value = float(out.item())
    value = max(min(value, 0.999999), -0.999999)
    cp_stm = CP_SCALE * math.atanh(value)
    cp_white = cp_stm if stm_white else -cp_stm
    return cp_white, set(white.tolist()), set(black.tolist())


def main():
    model, args = load_model_from_checkpoint(CHECKPOINT)
    print("arch:", args.get("arch"), flush=True)
    emb = model.accumulator.weight.detach()
    row_l1 = emb.abs().sum(dim=1)

    report = []
    for label, fen in POSITIONS:
        board = fen_board(fen)
        cp_full, wfeat_full, bfeat_full = evaluate(model, fen)
        squares = {}
        for sq, piece in board.items():
            entry = {"piece": piece, "contrib_cp": None, "energy": None}
            if piece.upper() != "K":
                reduced = fen_without(fen, sq)
                cp_r, wfeat_r, bfeat_r = evaluate(model, reduced)
                entry["contrib_cp"] = round(cp_full - cp_r, 1)
                lost = (wfeat_full - wfeat_r) | (bfeat_full - bfeat_r)
                if lost:
                    idx = torch.tensor(sorted(lost), dtype=torch.long)
                    entry["energy"] = round(float(row_l1[idx].sum().item()), 2)
            squares[sq] = entry
        report.append({
            "label": label,
            "fen": fen,
            "cp_white": round(cp_full, 1),
            "squares": squares,
        })
        print(f"{label}: eval {cp_full:+.0f}cp white-POV, "
              f"{len(board)} pieces mapped", flush=True)

    out_path = SCRATCH / "nn_saliency.json"
    out_path.write_text(json.dumps(report))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
