#!/usr/bin/env python3
"""Ridge-distill existing linear HCE parameters from Stockfish scores."""

import argparse

import numpy as np
from sklearn.linear_model import Ridge

import texel_tune


def load_scores(path):
    scores = []
    with open(path, "r", encoding="utf-8") as source:
        header = next(source).rstrip().split("\t")
        columns = {name: index for index, name in enumerate(header)}
        for line in source:
            fields = line.rstrip().split("\t")
            scores.append(float(fields[columns["sf_cp_white"]]))
    return np.asarray(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feats", required=True)
    parser.add_argument("--initial-tuned-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-score", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Fit one ridge strength instead of scanning defaults.")
    args = parser.parse_args()

    sf = load_scores(args.labels)
    _, phase, eval_true, white, black = texel_tune.build(args.feats)
    raw_features = np.atleast_2d(np.loadtxt(args.feats))
    qsearch_delta = raw_features[:, -1]
    if len(sf) != len(phase):
        raise SystemExit(f"label/feature mismatch: {len(sf)} vs {len(phase)}")
    defaults = texel_tune.load_tuned_defaults(args.initial_tuned_file)
    wt = texel_tune.side_totals_int(white, phase, defaults)
    bt = texel_tune.side_totals_int(black, phase, defaults)
    if np.any(np.abs(eval_true - 12) != np.abs(wt - bt)):
        raise SystemExit("feature reconstruction failed")

    x, residual = texel_tune.design_matrix(phase, white, black)
    baseline = x @ defaults + residual
    keep = ((np.abs(sf) <= args.max_score) & (np.abs(baseline) <= 2500) &
            (np.abs(qsearch_delta) < 50))
    x = x[keep]
    residual = residual[keep]
    sf = sf[keep]
    baseline = baseline[keep]

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sf))
    split = int(0.8 * len(order))
    train, validation = order[:split], order[split:]

    slope, intercept = np.polyfit(baseline[train], sf[train], 1)
    teacher = (sf - intercept) / slope
    active = np.arange(5, texel_tune.N_PARAMS)
    parameter_scale = np.full(texel_tune.N_PARAMS, 10.0)
    parameter_scale[:5] = 100.0
    parameter_scale[5:texel_tune.N_SCALAR] = 20.0
    z = x[:, active] * parameter_scale[active]
    target = teacher - baseline

    base_rmse = np.sqrt(np.mean((teacher[validation] - baseline[validation]) ** 2))
    best = None
    alphas = ((args.alpha,) if args.alpha > 0 else
              (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0))
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr", tol=1e-6)
        model.fit(z[train], target[train])
        prediction = baseline + z @ model.coef_
        rmse = np.sqrt(np.mean((teacher[validation] - prediction[validation]) ** 2))
        print(f"alpha={alpha:7.0f} val_rmse={rmse:.2f} gain={base_rmse-rmse:+.2f}")
        if best is None or rmse < best[0]:
            best = (rmse, alpha, model.coef_.copy())

    theta = defaults.copy()
    theta[active] += best[2] * parameter_scale[active]
    theta[4] = 100.0
    rounded = np.rint(theta).astype(int)
    with open(args.out, "w", encoding="utf-8") as target_file:
        target_file.write("TUNED " + " ".join(str(value) for value in rounded) + "\n")
    print(f"positions={len(sf)} baseline_rmse={base_rmse:.2f} "
          f"selected_alpha={best[1]:.0f} selected_rmse={best[0]:.2f} "
          f"teacher_scale={slope:.4f} intercept={intercept:.2f}")
    for name, old, new in zip(texel_tune.PARAM_NAMES[:texel_tune.N_SCALAR],
                              defaults[:texel_tune.N_SCALAR],
                              rounded[:texel_tune.N_SCALAR]):
        print(f"{name:24s} {int(old):5d} -> {new:5d}")


if __name__ == "__main__":
    main()
