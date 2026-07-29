#!/usr/bin/env python3
"""Apply a TUNED weight line from texel_tune.py to hce_eval.c.

Reads the machine-readable `TUNED ...` line (815 integers):
    47 scalars (established terms, pawn activity, and HCE-v2 structure)
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

N_CURRENT_SCALAR = 35
N_SCALAR = 47
N_PST = 6 * 64
PST_PIECES = 6


def parse_tuned_line(line):
    parts = line.strip().split()
    if len(parts) >= 1 and parts[0] == "TUNED":
        parts = parts[1:]
    vals = [int(x) for x in parts]
    expected = N_SCALAR + 2 * N_PST
    current_expected = N_CURRENT_SCALAR + 2 * N_PST
    legacy_expected = 21 + 2 * N_PST
    if len(vals) == legacy_expected:
        extra_defaults = [
            100, 100, -100, -100, -100, -100, -100, -100,
            0, 0, 0, 0, 0, 0,
        ]
        vals = vals[:21] + extra_defaults + [0] * 12 + vals[21:]
    elif len(vals) == current_expected:
        vals = vals[:N_CURRENT_SCALAR] + [0] * 12 + vals[N_CURRENT_SCALAR:]
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

    def replace_exact(pattern, replacement, label, *, flags=0):
        nonlocal text
        text, replaced = re.subn(
            pattern,
            replacement,
            text,
            count=1,
            flags=flags,
        )
        if replaced != 1:
            raise SystemExit(f"failed to patch {label} in {path}")

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
    (connected_pawn_mg, connected_pawn_eg,
     phalanx_pawn_mg, phalanx_pawn_eg,
     backward_pawn_mg, backward_pawn_eg,
     knight_outpost_mg, knight_outpost_eg,
     bishop_pair_mg, bishop_pair_eg,
     rook_behind_passer_mg, rook_behind_passer_eg) = scalar[35:47]

    # Piece enum order in engine: KING=0, QUEEN=1, BISHOP=2, KNIGHT=3, ROOK=4, PAWN=5.
    # Scalar order from tuner: mat_q, mat_n, mat_b, mat_r, mat_p.
    piece_values = [0, mat_q, mat_b, mat_n, mat_r, mat_p]

    # Patch hce_piece_value array.
    def pv_repl(m):
        return "const int hce_piece_value[PIECE_TYPE_COUNT] = {\n" + "\n".join(
            f"    {v}," for v in piece_values
        ) + "\n};"

    replace_exact(
        r"const int hce_piece_value\[PIECE_TYPE_COUNT\] = \{[^}]+\};",
        pv_repl,
        "hce_piece_value",
    )

    # Patch scalar constants.  Use regexes that match the surrounding code.
    replace_exact(
        r"eval_term_add\(&terms\.pawn_structure, -?\d+, -?\d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->isolated",
        lambda m: f"eval_term_add(&terms.pawn_structure, {iso_mg}, {iso_eg});\n"
                  f"                        if (feat != NULL) {{\n"
                  f"                            feat->isolated",
        "isolated pawn weights",
    )
    replace_exact(
        r"eval_term_add\(&terms\.pawn_structure, -?\d+, -?\d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->doubled",
        lambda m: f"eval_term_add(&terms.pawn_structure, {dbl_mg}, {dbl_eg});\n"
                  f"                        if (feat != NULL) {{\n"
                  f"                            feat->doubled",
        "doubled pawn weights",
    )
    mobility_replacement = (
        "eval_term_add(&terms.mobility,\n"
        f"                  knight_mob * {mob_n_mg} + "
        f"bishop_mob * {mob_b_mg} + rook_mob * {mob_r_mg} + "
        f"queen_mob * {mob_q_mg},\n"
        f"                  knight_mob * {mob_n_eg} + "
        f"bishop_mob * {mob_b_eg} + rook_mob * {mob_r_eg} + "
        f"queen_mob * {mob_q_eg});"
    )
    replace_exact(
        r"eval_term_add\(&terms\.mobility,\s*"
        r"knight_mob\s*\*\s*-?\d+.*?\);",
        mobility_replacement,
        "combined mobility weights",
        flags=re.DOTALL,
    )
    replace_exact(
        r"eval_term_add\(&terms\.rook_files, \d+, \d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->rook_open",
        lambda m: f"eval_term_add(&terms.rook_files, {rook_open_mg}, {rook_open_eg});\n"
                  f"                        if (feat != NULL) {{\n"
                  f"                            feat->rook_open",
        "open-file rook weights",
    )
    replace_exact(
        r"eval_term_add\(&terms\.rook_files, \d+, \d+\);\s*\n\s*if \(feat != NULL\) \{\s*\n\s*feat->rook_semi",
        lambda m: f"eval_term_add(&terms.rook_files, {rook_semi_mg}, {rook_semi_eg});\n"
                  f"                        if (feat != NULL) {{\n"
                  f"                            feat->rook_semi",
        "semi-open-file rook weights",
    )

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

    v2_values = {
        "k_connected_pawn_mg": connected_pawn_mg,
        "k_connected_pawn_eg": connected_pawn_eg,
        "k_phalanx_pawn_mg": phalanx_pawn_mg,
        "k_phalanx_pawn_eg": phalanx_pawn_eg,
        "k_backward_pawn_mg": backward_pawn_mg,
        "k_backward_pawn_eg": backward_pawn_eg,
        "k_knight_outpost_mg": knight_outpost_mg,
        "k_knight_outpost_eg": knight_outpost_eg,
        "k_bishop_pair_mg": bishop_pair_mg,
        "k_bishop_pair_eg": bishop_pair_eg,
        "k_rook_behind_passer_mg": rook_behind_passer_mg,
        "k_rook_behind_passer_eg": rook_behind_passer_eg,
    }
    for name, value in v2_values.items():
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

        replace_exact(
            rf"static const int ({re.escape(mg_name)})\[64\] = \{{[^}}]+\}};",
            make_repl(mg_arr),
            mg_name,
        )
        replace_exact(
            rf"static const int ({re.escape(eg_name)})\[64\] = \{{[^}}]+\}};",
            make_repl(eg_arr),
            eg_name,
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
