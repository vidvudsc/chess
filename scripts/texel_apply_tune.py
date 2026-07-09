#!/usr/bin/env python3
"""Apply a TUNED weight line from texel_tune.py to hce_eval.c.

Reads the machine-readable `TUNED ...` line (789 integers):
    21 scalars (mat_q, mat_n, mat_b, mat_r, mat_p, iso_mg, iso_eg, ...)
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

N_SCALAR = 21
N_PST = 6 * 64
PST_PIECES = 6


def parse_tuned_line(line):
    parts = line.strip().split()
    if len(parts) >= 1 and parts[0] == "TUNED":
        parts = parts[1:]
    vals = [int(x) for x in parts]
    expected = N_SCALAR + 2 * N_PST
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

    # Patch scalar constants.  Use regexes that match the surrounding code.
    text = re.sub(
        r"eval_term_add\(&terms\.pawn_structure, -?\d+, -?\d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->isolated",
        lambda m: f"eval_term_add(&terms.pawn_structure, {iso_mg}, {iso_eg});\n"
                  f"                    if (feat != NULL) {{\n"
                  f"                        feat->isolated",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.pawn_structure, -?\d+, -?\d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->doubled",
        lambda m: f"eval_term_add(&terms.pawn_structure, {dbl_mg}, {dbl_eg});\n"
                  f"                    if (feat != NULL) {{\n"
                  f"                        feat->doubled",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.mobility, knight_mob \* \d+, knight_mob \* \d+\);",
        f"eval_term_add(&terms.mobility, knight_mob * {mob_n_mg}, knight_mob * {mob_n_eg});",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.mobility, bishop_mob \* \d+, bishop_mob \* \d+\);",
        f"eval_term_add(&terms.mobility, bishop_mob * {mob_b_mg}, bishop_mob * {mob_b_eg});",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.mobility, rook_mob \* \d+, rook_mob \* \d+\);",
        f"eval_term_add(&terms.mobility, rook_mob * {mob_r_mg}, rook_mob * {mob_r_eg});",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.mobility, queen_mob \* \d+, queen_mob \* \d+\);",
        f"eval_term_add(&terms.mobility, queen_mob * {mob_q_mg}, queen_mob * {mob_q_eg});",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.rook_files, \d+, \d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->rook_open",
        lambda m: f"eval_term_add(&terms.rook_files, {rook_open_mg}, {rook_open_eg});\n"
                  f"                        if (feat != NULL) {{\n"
                  f"                            feat->rook_open",
        text,
        count=1,
    )
    text = re.sub(
        r"eval_term_add\(&terms\.rook_files, \d+, \d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->rook_semi",
        lambda m: f"eval_term_add(&terms.rook_files, {rook_semi_mg}, {rook_semi_eg});\n"
                  f"                        if (feat != NULL) {{\n"
                  f"                            feat->rook_semi",
        text,
        count=1,
    )

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
