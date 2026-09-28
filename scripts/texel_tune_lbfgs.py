#!/usr/bin/env python3
"""L-BFGS fit of the HCE linear eval model (texel_tune.py layout).

Same model, exact-reconstruction gate and TUNED output line as
texel_tune.py, but minimized with scipy L-BFGS-B, which converges in a few
hundred iterations and suits low-noise labels such as Stockfish win
probabilities (see docs/HCE_NEXT_20260928.md). The pawn value stays 100.

    texel_current_defaults.py > cur.txt
    texel_tune_lbfgs.py --feats dump.txt --initial-tuned-file cur.txt \
        --l2 0.005 > tuned.txt
    texel_apply_tune.py --tuned-file tuned.txt

--l2 is the same relative pull toward the current weights as texel_tune.py
(which defaults to 3.0 for noisy game-result labels). --scalars-only keeps
the piece-square tables fixed. Set VECLIB_MAXIMUM_THREADS=1 (macOS) or
OPENBLAS_NUM_THREADS=1 to keep numpy on one core.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texel_tune as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--initial-tuned-file", required=True)
    ap.add_argument("--l2", type=float, default=0.005)
    ap.add_argument("--scalars-only", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    defaults = T.load_tuned_defaults(args.initial_tuned_file)
    label, phase, eval_true, w, b = T.build(args.feats)
    wt = T.side_totals_int(w, phase, defaults)
    bt = T.side_totals_int(b, phase, defaults)
    mism = int(np.sum(np.abs(eval_true - 12) != np.abs(wt - bt)))
    print(f"positions: {len(label)}  verify mismatches: {mism}", file=sys.stderr)
    if mism:
        print("ABORT: feature reconstruction is not exact.", file=sys.stderr)
        return 1

    X, c = T.design_matrix(phase, w, b)
    n = len(label)
    idx = np.random.default_rng(args.seed).permutation(n)
    nval = int(n * args.val_frac)
    val, tr = idx[:nval], idx[nval:]
    y = label
    K, loss0 = T.fit_K(X[tr] @ defaults + c[tr], y[tr])
    print(f"K={K:.5f} train={loss0:.6f} "
          f"val={T.loss_for(K, X[val] @ defaults + c[val], y[val]):.6f}", file=sys.stderr)

    active = np.arange(T.N_SCALAR if args.scalars_only else T.N_PARAMS)
    active = active[active != 4]  # pawn anchored at 100
    reg_scale = np.maximum(np.abs(defaults), 20.0)[active]
    base = defaults[active]
    xa = X[tr][:, active]
    fixed = X[tr] @ defaults + c[tr] - xa @ base
    ytr = y[tr]
    ntr = len(tr)
    scale = 1e4  # keep the objective well away from L-BFGS tolerances

    def objective(theta):
        ev = fixed + xa @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(K * ev, -60.0, 60.0)))
        r = p - ytr
        reg = args.l2 * 1e-3 * (theta - base) / reg_scale ** 2
        loss = np.mean(r * r) + 0.5 * args.l2 * 1e-3 * np.sum(((theta - base) / reg_scale) ** 2)
        grad = xa.T @ (2.0 * r * p * (1.0 - p) * K) / ntr + reg
        return loss * scale, grad * scale

    res = minimize(objective, base.copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "maxcor": 30, "gtol": 1e-10, "ftol": 1e-14})
    theta = defaults.copy()
    theta[active] = res.x
    print(f"iterations={res.nit} train={T.loss_for(K, X[tr] @ theta + c[tr], y[tr]):.6f} "
          f"val={T.loss_for(K, X[val] @ theta + c[val], y[val]):.6f}", file=sys.stderr)
    for i in range(T.N_SCALAR):
        print(f"  {T.PARAM_NAMES[i]:24s} {defaults[i]:6.0f} -> {theta[i]:7.1f}", file=sys.stderr)
    print("TUNED " + " ".join(str(int(round(v))) for v in theta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
