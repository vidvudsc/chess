#!/usr/bin/env python3
"""Emit the engine's CURRENT weights as a TUNED line for texel_tune.py.

Everything is parsed from hce_eval.c: hce_piece_value, the named scalar
constants (k_iso_mg .. k_pawn_threat_major_eg), the stage-B constants, and
the PST tables. texel_tune.py's exact-reconstruction gate verifies the vector
against a real tunedump, so any parse/ordering drift fails loudly.

Usage:
    texel_current_defaults.py [--eval-c src/core/engine/hce_eval.c] > cur.txt
    texel_tune.py --feats dump.txt --initial-tuned-file cur.txt ...
"""
import argparse
import re
import sys
from pathlib import Path

# Tuner scalar order after the 5 material values.
SCALAR_CONST_NAMES = [
    "k_iso_mg", "k_iso_eg", "k_dbl_mg", "k_dbl_eg",
    "k_mob_n_mg", "k_mob_n_eg", "k_mob_b_mg", "k_mob_b_eg",
    "k_mob_r_mg", "k_mob_r_eg", "k_mob_q_mg", "k_mob_q_eg",
    "k_rook_open_mg", "k_rook_open_eg", "k_rook_semi_mg", "k_rook_semi_eg",
    "k_passed_mg_scale", "k_passed_eg_scale",
    "k_king_mg_scale", "k_king_eg_scale",
    "k_hanging_mg_scale", "k_hanging_eg_scale",
    "k_queen_mg_scale", "k_queen_eg_scale",
    "k_pawn_push_mg", "k_pawn_push_eg",
    "k_pawn_threat_minor_mg", "k_pawn_threat_minor_eg",
    "k_pawn_threat_major_mg", "k_pawn_threat_major_eg",
]
STAGEB_CONST_NAMES = [
    "k_safe_check_n_mg", "k_safe_check_n_eg",
    "k_safe_check_b_mg", "k_safe_check_b_eg",
    "k_safe_check_r_mg", "k_safe_check_r_eg",
    "k_safe_check_q_mg", "k_safe_check_q_eg",
    "k_bishop_pair_mg", "k_bishop_pair_eg",
    "k_mob_safe_n_mg", "k_mob_safe_n_eg",
    "k_mob_safe_b_mg", "k_mob_safe_b_eg",
    "k_mob_safe_r_mg", "k_mob_safe_r_eg",
    "k_mob_safe_q_mg", "k_mob_safe_q_eg",
]
# Tuner PST order (piece index 0..5): K, Q, B, N, R, P.
PST_MG_NAMES = ["k_king_mid_pst", "k_queen_pst", "k_bishop_pst",
                "k_knight_pst", "k_rook_pst", "k_pawn_pst"]
PST_EG_NAMES = ["k_king_end_pst", "k_queen_pst_eg", "k_bishop_pst_eg",
                "k_knight_pst_eg", "k_rook_pst_eg", "k_pawn_pst_eg"]


def parse_const(src, name):
    m = re.search(rf"static const int {name} = (-?\d+);", src)
    if m is None:
        raise SystemExit(f"constant {name} not found")
    return int(m.group(1))


def parse_int_array(src, name, count):
    m = re.search(rf"{name}\[{count}\]\s*=\s*\{{(.*?)\}};", src, re.S)
    if m is None:
        raise SystemExit(f"array {name} not found")
    vals = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(vals) != count:
        raise SystemExit(f"array {name}: expected {count} values, got {len(vals)}")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-c", default=None)
    args = ap.parse_args()
    path = (Path(args.eval_c) if args.eval_c
            else Path(__file__).resolve().parents[1] / "src/core/engine/hce_eval.c")
    src = path.read_text(encoding="utf-8")

    m = re.search(r"hce_piece_value\[PIECE_TYPE_COUNT\]\s*=\s*\{(.*?)\};", src, re.S)
    if m is None:
        raise SystemExit("hce_piece_value not found")
    pv = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(pv) != 6:
        raise SystemExit(f"hce_piece_value: expected 6 values, got {len(pv)}")
    # Enum order K,Q,B,N,R,P -> tuner material order q,n,b,r,p.
    values = [pv[1], pv[3], pv[2], pv[4], pv[5]]

    values += [parse_const(src, name) for name in SCALAR_CONST_NAMES]
    values += [parse_const(src, name) for name in STAGEB_CONST_NAMES]
    pr_mg = parse_int_array(src, "k_passer_rank_mg", 6)
    pr_eg = parse_int_array(src, "k_passer_rank_eg", 6)
    for r in range(6):
        values += [pr_mg[r], pr_eg[r]]

    for name in PST_MG_NAMES:
        values += parse_int_array(src, name, 64)
    for name in PST_EG_NAMES:
        values += parse_int_array(src, name, 64)

    print("TUNED " + " ".join(str(v) for v in values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
