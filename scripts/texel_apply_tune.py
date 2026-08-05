#!/usr/bin/env python3
"""Apply a TUNED weight line from texel_tune.py to hce_eval.c.

Reads the machine-readable `TUNED ...` line (803 integers):
    35 scalars (established terms, residual scales, and pawn activity)
    384 mg PST values (K, Q, B, N, R, P each 64 squares)
    384 eg PST values

Then patches:
  - hce_piece_value[PIECE_TYPE_COUNT]
  - scalar constants in eval_side (isolated, doubled, mobility, rook files)
  - k_*_pst and k_*_pst_eg arrays
"""
import argparse
import re
import sys
from pathlib import Path

N_SCALAR = 65
N_PST = 6 * 64
PST_PIECES = 6


def parse_tuned_line(line):
    parts = line.strip().split()
    if len(parts) >= 1 and parts[0] == "TUNED":
        parts = parts[1:]
    vals = [int(x) for x in parts]
    expected = N_SCALAR + 2 * N_PST
    legacy_expected = 21 + 2 * N_PST
    prev_expected = 35 + 2 * N_PST
    if len(vals) == legacy_expected:
        extra_defaults = [
            100, 100, -100, -100, -100, -100, -100, -100,
            0, 0, 0, 0, 0, 0,
        ]
        vals = vals[:21] + extra_defaults + vals[21:]
    if len(vals) == prev_expected:
        vals = vals[:35] + [0] * 30 + vals[35:]
    if len(vals) != expected:
        raise SystemExit(f"expected {expected} tuned integers, got {len(vals)}")
    return vals


def fmt_array(arr, indent=4):
    lines = []
    for r in range(8):
        row = ", ".join(f"{int(arr[r * 8 + c]):4d}" for c in range(8))
        lines.append(" " * indent + row + ",")
    return "\n".join(lines)


def patch_eval_c(path, vals):
    text = path.read_text(encoding="utf-8")

    scalar = vals[:N_SCALAR]
    mg = vals[N_SCALAR:N_SCALAR + N_PST]
    eg = vals[N_SCALAR + N_PST:]

    mat_q, mat_n, mat_b, mat_r, mat_p = scalar[0:5]
    iso_mg, iso_eg, dbl_mg, dbl_eg = scalar[5:9]
    mob_n_mg, mob_n_eg, mob_b_mg, mob_b_eg = scalar[9:13]
    mob_r_mg, mob_r_eg, mob_q_mg, mob_q_eg = scalar[13:17]
    rook_open_mg, rook_open_eg, rook_semi_mg, rook_semi_eg = scalar[17:21]
    (passed_mg_scale, passed_eg_scale,
     king_mg_scale, king_eg_scale,
     hanging_mg_scale, hanging_eg_scale,
     queen_mg_scale, queen_eg_scale) = scalar[21:29]
    (pawn_push_mg, pawn_push_eg,
     pawn_threat_minor_mg, pawn_threat_minor_eg,
     pawn_threat_major_mg, pawn_threat_major_eg) = scalar[29:35]
    stageb = scalar[35:65]

    # Piece enum order in engine: KING=0, QUEEN=1, BISHOP=2, KNIGHT=3, ROOK=4, PAWN=5.
    # Scalar order from tuner: mat_q, mat_n, mat_b, mat_r, mat_p.
    piece_values = [0, mat_q, mat_b, mat_n, mat_r, mat_p]

    # Patch hce_piece_value array.
    def pv_repl(m):
        return "const int hce_piece_value[PIECE_TYPE_COUNT] = {\n" + "\n".join(
            f"    {v}," for v in piece_values
        ) + "\n};"

    text = re.sub(
        r"const int hce_piece_value\[PIECE_TYPE_COUNT\] = \{[^}]+\};",
        pv_repl,
        text,
        count=1,
    )

    # All per-term scalars are named constants now; patch them uniformly.
    const_values = {
        "k_iso_mg": iso_mg, "k_iso_eg": iso_eg,
        "k_dbl_mg": dbl_mg, "k_dbl_eg": dbl_eg,
        "k_mob_n_mg": mob_n_mg, "k_mob_n_eg": mob_n_eg,
        "k_mob_b_mg": mob_b_mg, "k_mob_b_eg": mob_b_eg,
        "k_mob_r_mg": mob_r_mg, "k_mob_r_eg": mob_r_eg,
        "k_mob_q_mg": mob_q_mg, "k_mob_q_eg": mob_q_eg,
        "k_rook_open_mg": rook_open_mg, "k_rook_open_eg": rook_open_eg,
        "k_rook_semi_mg": rook_semi_mg, "k_rook_semi_eg": rook_semi_eg,
    }
    stageb_names = [
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
    for name, value in zip(stageb_names, stageb[:18]):
        const_values[name] = value
    for name, value in const_values.items():
        text, replaced = re.subn(
            rf"static const int {name} = -?\d+;",
            f"static const int {name} = {value};",
            text,
            count=1,
        )
        if replaced != 1:
            raise SystemExit(f"failed to patch {name} in {path}")
    # Per-rank passer arrays: stage-B params 18..29 are (mg, eg) pairs.
    pr_mg = [stageb[18 + 2 * r] for r in range(6)]
    pr_eg = [stageb[19 + 2 * r] for r in range(6)]
    for name, arr in (("k_passer_rank_mg", pr_mg), ("k_passer_rank_eg", pr_eg)):
        body = ", ".join(str(v) for v in arr)
        text, replaced = re.subn(
            rf"static const int {name}\[6\] = \{{[^}}]*\}};",
            f"static const int {name}[6] = {{{body}}};",
            text,
            count=1,
        )
        if replaced != 1:
            raise SystemExit(f"failed to patch {name} in {path}")

    scale_values = {
        "k_passed_mg_scale": passed_mg_scale,
        "k_passed_eg_scale": passed_eg_scale,
        "k_king_mg_scale": king_mg_scale,
        "k_king_eg_scale": king_eg_scale,
        "k_hanging_mg_scale": hanging_mg_scale,
        "k_hanging_eg_scale": hanging_eg_scale,
        "k_queen_mg_scale": queen_mg_scale,
        "k_queen_eg_scale": queen_eg_scale,
    }
    for name, value in scale_values.items():
        text, replaced = re.subn(
            rf"static const int {name} = -?\d+;",
            f"static const int {name} = {value};",
            text,
            count=1,
        )
        if replaced != 1:
            raise SystemExit(f"failed to patch {name} in {path}")

    pawn_values = {
        "k_pawn_push_mg": pawn_push_mg,
        "k_pawn_push_eg": pawn_push_eg,
        "k_pawn_threat_minor_mg": pawn_threat_minor_mg,
        "k_pawn_threat_minor_eg": pawn_threat_minor_eg,
        "k_pawn_threat_major_mg": pawn_threat_major_mg,
        "k_pawn_threat_major_eg": pawn_threat_major_eg,
    }
    for name, value in pawn_values.items():
        text, replaced = re.subn(
            rf"static const int {name} = -?\d+;",
            f"static const int {name} = {value};",
            text,
            count=1,
        )
        if replaced != 1:
            raise SystemExit(f"failed to patch {name} in {path}")

    # Patch PST arrays.  Order in the tuned vector is K, Q, B, N, R, P.
    piece_table_names = {
        0: ("k_king_mid_pst", "k_king_end_pst"),
        1: ("k_queen_pst", "k_queen_pst_eg"),
        2: ("k_bishop_pst", "k_bishop_pst_eg"),
        3: ("k_knight_pst", "k_knight_pst_eg"),
        4: ("k_rook_pst", "k_rook_pst_eg"),
        5: ("k_pawn_pst", "k_pawn_pst_eg"),
    }

    for p in range(PST_PIECES):
        mg_name, eg_name = piece_table_names[p]
        mg_arr = mg[p * 64:(p + 1) * 64]
        eg_arr = eg[p * 64:(p + 1) * 64]

        def make_repl(arr):
            body = fmt_array(arr)
            return lambda m: f"static const int {m.group(1)}[64] = {{\n{body}\n}};"

        text = re.sub(
            rf"static const int ({re.escape(mg_name)})\[64\] = \{{[^}}]+\}};",
            make_repl(mg_arr),
            text,
            count=1,
        )
        text = re.sub(
            rf"static const int ({re.escape(eg_name)})\[64\] = \{{[^}}]+\}};",
            make_repl(eg_arr),
            text,
            count=1,
        )

    path.write_text(text, encoding="utf-8")
    print(f"patched {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-c", type=Path, default=Path("src/core/engine/hce_eval.c"))
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--tuned-line", help="The full TUNED line as a string.")
    group.add_argument("--tuned-file", type=Path, help="File containing the TUNED line.")
    args = ap.parse_args()

    if args.tuned_line:
        line = args.tuned_line
    else:
        lines = [l for l in args.tuned_file.read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("TUNED ")]
        if not lines:
            raise SystemExit(f"no TUNED line found in {args.tuned_file}")
        line = lines[-1]

    vals = parse_tuned_line(line)
    patch_eval_c(args.eval_c, vals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
