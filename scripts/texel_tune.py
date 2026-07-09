#!/usr/bin/env python3
"""Texel-tune the HCE eval weights from a feature dump.

Input: the file produced by `chess_uci`'s `tunedump` command, one line per
quiet position:

    <label> <phase> <eval_true> <white 399 feats> <black 399 feats>

where each 399-feature block is:
    mat_q mat_n mat_b mat_r mat_p isolated doubled
    mob_n mob_b mob_r mob_q rook_open rook_semi
    pst[K,Q,B,N,R,P][64] flattened
    residual_mg residual_eg

The engine's per-side eval is  total = (mg*phase + eg*(24-phase)) / 24  with an
integer truncation per side. Dropping that truncation makes white_total -
black_total a *linear* function of the tunable weights, so we fit them with
gradient descent through the Texel sigmoid against the game result.

Steps: (1) verify the exact integer reconstruction matches eval_true, (2) fit
the sigmoid scale K, (3) gradient-descent the weights on a train split while
watching a validation split, (4) print old vs new integer weights.
"""
import argparse
import sys

import numpy as np

# Number of scalar (material + positional) features and per-side layout.
N_SCALAR = 21
SIDE_OLD = 15
PST_PIECES = 6
PST_SQUARES = 64
N_PST = PST_PIECES * PST_SQUARES
N_PARAMS = N_SCALAR + 2 * N_PST  # 789

# Current engine PST tables (from src/core/engine/hce_eval.c).
K_PAWN_PST = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.float64)

K_KNIGHT_PST = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.float64)

K_BISHOP_PST = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.float64)

K_ROOK_PST = np.array([
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.float64)

K_QUEEN_PST = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
], dtype=np.float64)

K_KING_MID_PST = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
], dtype=np.float64)

K_KING_END_PST = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
], dtype=np.float64)

# Piece enum order: KING=0, QUEEN=1, BISHOP=2, KNIGHT=3, ROOK=4, PAWN=5.
_PST_TABLES_MG = [
    np.zeros(64, dtype=np.float64),   # KING (zero in the pre-king-PST baseline)
    K_QUEEN_PST,
    K_BISHOP_PST,
    K_KNIGHT_PST,
    K_ROOK_PST,
    K_PAWN_PST,
]
# The engine uses mg = table, eg = table/2 (trunc toward zero) for non-king pieces.
_PST_TABLES_EG = [
    np.sign(t) * (np.abs(t) // 2) for t in _PST_TABLES_MG
]
_PST_DEFAULT_MG = np.concatenate(_PST_TABLES_MG)
_PST_DEFAULT_EG = np.concatenate(_PST_TABLES_EG)

PARAM_NAMES = (
    ["mat_q", "mat_n", "mat_b", "mat_r", "mat_p",
     "iso_mg", "iso_eg", "dbl_mg", "dbl_eg",
     "mob_n_mg", "mob_n_eg", "mob_b_mg", "mob_b_eg",
     "mob_r_mg", "mob_r_eg", "mob_q_mg", "mob_q_eg",
     "rook_open_mg", "rook_open_eg", "rook_semi_mg", "rook_semi_eg"]
    + [f"pst{p}_{s}_mg" for p in range(PST_PIECES) for s in range(PST_SQUARES)]
    + [f"pst{p}_{s}_eg" for p in range(PST_PIECES) for s in range(PST_SQUARES)]
)

# Current engine scalar defaults (post 44dc6b5 texel tune).
_SCALAR_DEFAULTS = np.array([
    1235, 409, 466, 537, 100,   # material q n b r p (knight=409, bishop=466)
    -13, -16, -17, -15,         # isolated, doubled (mg, eg)
    8, 4, 9, 4, 9, 6, 9, 2,     # mobility n,b,r,q (mg, eg)
    19, 12, 11, 6,              # rook open, semi (mg, eg)
], dtype=np.float64)

DEFAULTS = np.concatenate([
    _SCALAR_DEFAULTS,
    _PST_DEFAULT_MG,
    _PST_DEFAULT_EG,
])

# Column indices within the first 15 old scalar features.
F_MATQ, F_MATN, F_MATB, F_MATR, F_MATP = 0, 1, 2, 3, 4
F_ISO, F_DBL = 5, 6
F_MN, F_MB, F_MR, F_MQ = 7, 8, 9, 10
F_ROPEN, F_RSEMI = 11, 12
F_RESMG, F_RESEG = 13, 14


def trunc_div24(a):
    """C integer division by 24 truncating toward zero (vectorized)."""
    q = np.abs(a) // 24
    return np.where(a < 0, -q, q).astype(np.int64)


def build(feats_path):
    raw = np.atleast_2d(np.loadtxt(feats_path))
    label = raw[:, 0].astype(np.float64)
    phase = raw[:, 1].astype(np.int64)
    eval_true = raw[:, 2].astype(np.int64)
    side_feats = SIDE_OLD + N_PST
    w = raw[:, 3:3 + side_feats]
    b = raw[:, 3 + side_feats:3 + 2 * side_feats]
    return label, phase, eval_true, w, b


def split_side(side):
    """Return (old_scalar, pst) from a side feature vector."""
    old = side[:, :SIDE_OLD].astype(np.int64)
    pst = side[:, SIDE_OLD:].astype(np.int64).reshape(-1, PST_PIECES, PST_SQUARES)
    return old, pst


def side_totals_int(side, phase, theta):
    """Exact integer per-side total using the current weights."""
    old, pst = split_side(side)
    scalar = theta[:N_SCALAR]
    wmg = theta[N_SCALAR:N_SCALAR + N_PST].reshape(PST_PIECES, PST_SQUARES)
    weg = theta[N_SCALAR + N_PST:].reshape(PST_PIECES, PST_SQUARES)

    mat = (old[:, F_MATQ] * scalar[0] + old[:, F_MATN] * scalar[1] +
           old[:, F_MATB] * scalar[2] + old[:, F_MATR] * scalar[3] +
           old[:, F_MATP] * scalar[4]).astype(np.int64)
    ps_mg = np.sum(pst * wmg, axis=(1, 2)).astype(np.int64)
    ps_eg = np.sum(pst * weg, axis=(1, 2)).astype(np.int64)

    mg = (mat + ps_mg +
          old[:, F_ISO] * scalar[5] + old[:, F_DBL] * scalar[7] +
          old[:, F_MN] * scalar[9] + old[:, F_MB] * scalar[11] +
          old[:, F_MR] * scalar[13] + old[:, F_MQ] * scalar[15] +
          old[:, F_ROPEN] * scalar[17] + old[:, F_RSEMI] * scalar[19] +
          old[:, F_RESMG]).astype(np.int64)
    eg = (mat + ps_eg +
          old[:, F_ISO] * scalar[6] + old[:, F_DBL] * scalar[8] +
          old[:, F_MN] * scalar[10] + old[:, F_MB] * scalar[12] +
          old[:, F_MR] * scalar[14] + old[:, F_MQ] * scalar[16] +
          old[:, F_ROPEN] * scalar[18] + old[:, F_RSEMI] * scalar[20] +
          old[:, F_RESEG]).astype(np.int64)
    return trunc_div24(mg * phase + eg * (24 - phase))


def design_matrix(phase, w, b):
    """X (N x N_PARAMS) and c (N,) so eval_white_float ~= X @ theta + c."""
    n = w.shape[0]
    ph = phase.astype(np.float64)
    mgw = ph / 24.0
    egw = (24.0 - ph) / 24.0
    w_old, w_pst = split_side(w)
    b_old, b_pst = split_side(b)
    d_old = (w_old - b_old).astype(np.float64)
    d_pst = (w_pst - b_pst).astype(np.float64).reshape(n, N_PST)

    X = np.zeros((n, N_PARAMS), dtype=np.float64)
    # Material: phase-independent.
    X[:, 0] = d_old[:, F_MATQ]
    X[:, 1] = d_old[:, F_MATN]
    X[:, 2] = d_old[:, F_MATB]
    X[:, 3] = d_old[:, F_MATR]
    X[:, 4] = d_old[:, F_MATP]
    # mg/eg scalar pairs.
    X[:, 5] = d_old[:, F_ISO] * mgw
    X[:, 6] = d_old[:, F_ISO] * egw
    X[:, 7] = d_old[:, F_DBL] * mgw
    X[:, 8] = d_old[:, F_DBL] * egw
    X[:, 9] = d_old[:, F_MN] * mgw
    X[:, 10] = d_old[:, F_MN] * egw
    X[:, 11] = d_old[:, F_MB] * mgw
    X[:, 12] = d_old[:, F_MB] * egw
    X[:, 13] = d_old[:, F_MR] * mgw
    X[:, 14] = d_old[:, F_MR] * egw
    X[:, 15] = d_old[:, F_MQ] * mgw
    X[:, 16] = d_old[:, F_MQ] * egw
    X[:, 17] = d_old[:, F_ROPEN] * mgw
    X[:, 18] = d_old[:, F_ROPEN] * egw
    X[:, 19] = d_old[:, F_RSEMI] * mgw
    X[:, 20] = d_old[:, F_RSEMI] * egw
    # PST mg/eg.
    X[:, N_SCALAR:N_SCALAR + N_PST] = d_pst * mgw[:, None]
    X[:, N_SCALAR + N_PST:] = d_pst * egw[:, None]

    c = d_old[:, F_RESMG] * mgw + d_old[:, F_RESEG] * egw
    return X, c


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def load_tuned_defaults(path):
    with open(path, "r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp if line.strip().startswith("TUNED ")]
    if not lines:
        raise SystemExit(f"no TUNED line found in {path}")
    values = np.array([int(value) for value in lines[-1].split()[1:]], dtype=np.float64)
    if len(values) != N_PARAMS:
        raise SystemExit(f"expected {N_PARAMS} values in {path}, got {len(values)}")
    return values


def loss_for(K, evals, y):
    return np.mean((y - sigmoid(K * evals)) ** 2)


def fit_K(evals, y):
    best_K, best_L = 0.004, 1e9
    for K in np.linspace(0.0005, 0.02, 200):
        L = loss_for(K, evals, y)
        if L < best_L:
            best_L, best_K = L, K
    return best_K, best_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=3.0,
                    help="L2 pull toward defaults (relative), tames overfit.")
    ap.add_argument("--anchor-pawn", action="store_true", default=True,
                    help="Freeze pawn=100 so the eval stays in centipawns.")
    ap.add_argument("--no-anchor-pawn", dest="anchor_pawn", action="store_false")
    ap.add_argument("--freeze-material", action="store_true",
                    help="Hold all 5 material values fixed; tune only the "
                         "positional terms + PST.")
    ap.add_argument("--out-c", help="Optional path to write tuned PST/material C snippet.")
    ap.add_argument("--initial-tuned-file",
                    help="Use the last TUNED line in this file as the exact current defaults.")
    args = ap.parse_args()

    defaults = (load_tuned_defaults(args.initial_tuned_file)
                if args.initial_tuned_file else DEFAULTS.copy())

    label, phase, eval_true, w, b = build(args.feats)
    n = len(label)
    print(f"positions: {n}", file=sys.stderr)

    # (1) Exact integer verification against the engine's own eval.
    wt = side_totals_int(w, phase, defaults)
    bt = side_totals_int(b, phase, defaults)
    eval_white = wt - bt
    mism = np.sum(np.abs(eval_true - 12) != np.abs(eval_white))
    print(f"verify: |eval_true-12| != |recon| on {mism}/{n} rows", file=sys.stderr)
    if mism > 0:
        bad = np.where(np.abs(eval_true - 12) != np.abs(eval_white))[0][:5]
        for i in bad:
            print(f"  row {i}: true={eval_true[i]} recon_white={eval_white[i]}",
                  file=sys.stderr)
        print("ABORT: feature reconstruction is not exact.", file=sys.stderr)
        return 1
    print("verify: OK (linear features reproduce engine eval exactly)",
          file=sys.stderr)

    # Linear design for float tuning.
    X, c = design_matrix(phase, w, b)
    y = label
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    nval = int(n * args.val_frac)
    val, tr = idx[:nval], idx[nval:]

    theta = defaults.copy()
    evals_tr = X[tr] @ theta + c[tr]
    K, L0 = fit_K(evals_tr, y[tr])
    print(f"fit K={K:.5f}  baseline train loss={L0:.6f}  "
          f"val loss={loss_for(K, X[val] @ theta + c[val], y[val]):.6f}",
          file=sys.stderr)

    # (3) Gradient descent (Adam) on weights; refit K periodically.
    # Relative L2 pull toward defaults keeps low-count terms sane.
    reg_scale = np.maximum(np.abs(defaults), 20.0)
    m = np.zeros(N_PARAMS)
    v = np.zeros(N_PARAMS)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xtr, ctr, ytr = X[tr], c[tr], y[tr]
    ntr = len(tr)
    best_theta = theta.copy()
    best_iteration = 0
    best_val = loss_for(K, X[val] @ theta + c[val], y[val])
    for it in range(1, args.iters + 1):
        evals = Xtr @ theta + ctr
        p = sigmoid(K * evals)
        g = (Xtr.T @ (2.0 * (p - ytr) * p * (1.0 - p) * K)) / ntr
        g += args.l2 * 1e-3 * (theta - defaults) / (reg_scale ** 2)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** it)
        vhat = v / (1 - b2 ** it)
        theta -= args.lr * mhat / (np.sqrt(vhat) + eps)
        if args.freeze_material:
            theta[0:5] = defaults[0:5]
        elif args.anchor_pawn:
            theta[4] = 100.0
        if it % 500 == 0:
            K, _ = fit_K(Xtr @ theta + ctr, ytr)
            Ltr = loss_for(K, Xtr @ theta + ctr, ytr)
            Lval = loss_for(K, X[val] @ theta + c[val], y[val])
            if Lval < best_val:
                best_val = Lval
                best_theta = theta.copy()
                best_iteration = it
            print(f"  it {it:5d}  K={K:.5f}  train={Ltr:.6f}  val={Lval:.6f}",
                  file=sys.stderr)

    theta = best_theta
    print(f"selected iteration {best_iteration} with val loss={best_val:.6f}",
          file=sys.stderr)
    rounded = np.round(theta).astype(np.int64)
    print("\n# tuned scalar weights (round to int):", file=sys.stderr)
    for name, d0, t in zip(PARAM_NAMES[:N_SCALAR], defaults[:N_SCALAR], rounded[:N_SCALAR]):
        print(f"  {name:14s} {int(round(d0)):5d} -> {int(t):5d}",
              file=sys.stderr)

    # Machine-readable line for porting.
    print("TUNED " + " ".join(str(int(t)) for t in rounded))

    if args.out_c:
        write_c_snippet(args.out_c, rounded)
    return 0


def write_c_snippet(path, rounded):
    """Write a C snippet with tuned hce_piece_value and PST tables."""
    mat = rounded[:5]
    scalar = rounded[:N_SCALAR]
    pst_mg = rounded[N_SCALAR:N_SCALAR + N_PST].reshape(PST_PIECES, PST_SQUARES)
    pst_eg = rounded[N_SCALAR + N_PST:].reshape(PST_PIECES, PST_SQUARES)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("// Tuned HCE tables (machine-generated).\n")
        fp.write("const int hce_piece_value[PIECE_TYPE_COUNT] = {\n")
        fp.write("    0,\n")
        for i, name in enumerate(["QUEEN", "BISHOP", "KNIGHT", "ROOK", "PAWN"]):
            fp.write(f"    {mat[i]:4d},  // {name}\n")
        fp.write("};\n\n")
        _write_table(fp, "k_king_mid_pst", pst_mg[0])
        _write_table(fp, "k_king_end_pst", pst_eg[0])
        _write_table(fp, "k_queen_pst", pst_mg[1])
        _write_table(fp, "k_bishop_pst", pst_mg[2])
        _write_table(fp, "k_knight_pst", pst_mg[3])
        _write_table(fp, "k_rook_pst", pst_mg[4])
        _write_table(fp, "k_pawn_pst", pst_mg[5])
        fp.write("\n// Suggested scalar weights if tuning them separately:\n")
        for name, val in zip(PARAM_NAMES[:N_SCALAR], scalar):
            fp.write(f"// {name} = {val}\n")


def _write_table(fp, name, arr):
    fp.write(f"static const int {name}[64] = {{\n")
    for r in range(8):
        row = ", ".join(f"{int(arr[r * 8 + c]):4d}" for c in range(8))
        fp.write("    " + row + ",\n")
    fp.write("};\n")


if __name__ == "__main__":
    raise SystemExit(main())
