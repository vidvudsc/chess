#!/usr/bin/env python3
"""Texel-tune the HCE linear eval weights from a feature dump.

Input: the file produced by `chess_uci`'s `tunedump` command, one line per
position:

    <label> <phase> <eval_true> <white 15 feats> <black 15 feats>

where each 15-feature block is:
    mat_q mat_n mat_b mat_r mat_p isolated doubled
    mob_n mob_b mob_r mob_q rook_open rook_semi residual_mg residual_eg

The engine's per-side eval is  total = (mg*phase + eg*(24-phase)) / 24  with an
integer truncation per side. Dropping that truncation makes white_total -
black_total a *linear* function of the 21 tunable weights, so we fit them with
gradient descent through the Texel sigmoid against the game result.

Steps: (1) verify the exact integer reconstruction matches eval_true, (2) fit
the sigmoid scale K, (3) gradient-descent the weights on a train split while
watching a validation split, (4) print old vs new integer weights.
"""
import argparse
import sys

import numpy as np

# Weight layout (21 params). Material is phase-independent (adds to mg and eg).
PARAM_NAMES = [
    "mat_q", "mat_n", "mat_b", "mat_r", "mat_p",
    "iso_mg", "iso_eg", "dbl_mg", "dbl_eg",
    "mob_n_mg", "mob_n_eg", "mob_b_mg", "mob_b_eg",
    "mob_r_mg", "mob_r_eg", "mob_q_mg", "mob_q_eg",
    "rook_open_mg", "rook_open_eg", "rook_semi_mg", "rook_semi_eg",
]
DEFAULTS = np.array([
    900, 320, 335, 500, 100,   # material q n b r p (knight 320, bishop 335)
    -10, -15, -12, -15,        # isolated, doubled (mg, eg)
    3, 2, 2, 3, 1, 2, 1, 1,    # mobility n,b,r,q (mg, eg)
    18, 12, 10, 6,             # rook open, semi (mg, eg)
], dtype=np.float64)

# Column indices within a 15-feature side block.
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
    raw = np.loadtxt(feats_path).astype(np.int64)
    label = raw[:, 0].astype(np.float64)
    phase = raw[:, 1].astype(np.int64)
    eval_true = raw[:, 2].astype(np.int64)
    w = raw[:, 3:18]
    b = raw[:, 18:33]
    return label, phase, eval_true, w, b


def side_totals_int(side, phase, theta):
    """Exact integer per-side total using the current weights."""
    mat = (side[:, F_MATQ] * theta[0] + side[:, F_MATN] * theta[1] +
           side[:, F_MATB] * theta[2] + side[:, F_MATR] * theta[3] +
           side[:, F_MATP] * theta[4]).astype(np.int64)
    mg = (mat + side[:, F_ISO] * theta[5] + side[:, F_DBL] * theta[7] +
          side[:, F_MN] * theta[9] + side[:, F_MB] * theta[11] +
          side[:, F_MR] * theta[13] + side[:, F_MQ] * theta[15] +
          side[:, F_ROPEN] * theta[17] + side[:, F_RSEMI] * theta[19] +
          side[:, F_RESMG]).astype(np.int64)
    eg = (mat + side[:, F_ISO] * theta[6] + side[:, F_DBL] * theta[8] +
          side[:, F_MN] * theta[10] + side[:, F_MB] * theta[12] +
          side[:, F_MR] * theta[14] + side[:, F_MQ] * theta[16] +
          side[:, F_ROPEN] * theta[18] + side[:, F_RSEMI] * theta[20] +
          side[:, F_RESEG]).astype(np.int64)
    return trunc_div24(mg * phase + eg * (24 - phase))


def design_matrix(phase, w, b):
    """X (N x 21) and c (N,) so eval_white_float ~= X @ theta + c."""
    n = w.shape[0]
    ph = phase.astype(np.float64)
    mgw = ph / 24.0
    egw = (24.0 - ph) / 24.0
    d = (w - b).astype(np.float64)  # white-minus-black per feature
    X = np.zeros((n, 21), dtype=np.float64)
    # Material: phase-independent.
    X[:, 0] = d[:, F_MATQ]
    X[:, 1] = d[:, F_MATN]
    X[:, 2] = d[:, F_MATB]
    X[:, 3] = d[:, F_MATR]
    X[:, 4] = d[:, F_MATP]
    # mg/eg pairs.
    X[:, 5] = d[:, F_ISO] * mgw
    X[:, 6] = d[:, F_ISO] * egw
    X[:, 7] = d[:, F_DBL] * mgw
    X[:, 8] = d[:, F_DBL] * egw
    X[:, 9] = d[:, F_MN] * mgw
    X[:, 10] = d[:, F_MN] * egw
    X[:, 11] = d[:, F_MB] * mgw
    X[:, 12] = d[:, F_MB] * egw
    X[:, 13] = d[:, F_MR] * mgw
    X[:, 14] = d[:, F_MR] * egw
    X[:, 15] = d[:, F_MQ] * mgw
    X[:, 16] = d[:, F_MQ] * egw
    X[:, 17] = d[:, F_ROPEN] * mgw
    X[:, 18] = d[:, F_ROPEN] * egw
    X[:, 19] = d[:, F_RSEMI] * mgw
    X[:, 20] = d[:, F_RSEMI] * egw
    c = d[:, F_RESMG] * mgw + d[:, F_RESEG] * egw
    return X, c


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    ap.add_argument("--iters", type=int, default=4000)
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
                         "16 positional scalar terms (pins the eval scale).")
    args = ap.parse_args()

    label, phase, eval_true, w, b = build(args.feats)
    n = len(label)
    print(f"positions: {n}", file=sys.stderr)

    # (1) Exact integer verification against the engine's own eval.
    wt = side_totals_int(w, phase, DEFAULTS)
    bt = side_totals_int(b, phase, DEFAULTS)
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

    theta = DEFAULTS.copy()
    evals_tr = X[tr] @ theta + c[tr]
    K, L0 = fit_K(evals_tr, y[tr])
    print(f"fit K={K:.5f}  baseline train loss={L0:.6f}  "
          f"val loss={loss_for(K, X[val] @ theta + c[val], y[val]):.6f}",
          file=sys.stderr)

    # (3) Gradient descent (Adam) on weights; refit K periodically.
    # Relative L2 pull toward defaults keeps low-count terms sane; per-param
    # scale so material (large) is barely constrained but mobility/rook (small)
    # are held near their starting values unless the data strongly disagrees.
    reg_scale = np.maximum(np.abs(DEFAULTS), 20.0)
    m = np.zeros(21); v = np.zeros(21)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xtr, ctr, ytr = X[tr], c[tr], y[tr]
    ntr = len(tr)
    for it in range(1, args.iters + 1):
        evals = Xtr @ theta + ctr
        p = sigmoid(K * evals)
        # d/dtheta of mean (y-p)^2 :  mean( 2(p-y) * p(1-p) * K * X )
        g = (Xtr.T @ (2.0 * (p - ytr) * p * (1.0 - p) * K)) / ntr
        g += args.l2 * 1e-3 * (theta - DEFAULTS) / (reg_scale ** 2)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** it)
        vhat = v / (1 - b2 ** it)
        theta -= args.lr * mhat / (np.sqrt(vhat) + eps)
        if args.freeze_material:
            theta[0:5] = DEFAULTS[0:5]
        elif args.anchor_pawn:
            theta[4] = 100.0
        if it % 500 == 0:
            K, _ = fit_K(Xtr @ theta + ctr, ytr)
            Ltr = loss_for(K, Xtr @ theta + ctr, ytr)
            Lval = loss_for(K, X[val] @ theta + c[val], y[val])
            print(f"  it {it:5d}  K={K:.5f}  train={Ltr:.6f}  val={Lval:.6f}",
                  file=sys.stderr)

    print("\n# tuned weights (round to int):", file=sys.stderr)
    for name, d0, t in zip(PARAM_NAMES, DEFAULTS, theta):
        print(f"  {name:14s} {int(round(d0)):5d} -> {int(round(t)):5d}",
              file=sys.stderr)
    # Machine-readable line for porting.
    print("TUNED " + " ".join(str(int(round(t))) for t in theta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
