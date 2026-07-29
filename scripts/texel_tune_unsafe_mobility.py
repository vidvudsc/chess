#!/usr/bin/env python3
"""Tune enemy-pawn-controlled mobility corrections over a frozen HCE eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import texel_tune as base
    import texel_tune_mobility_shape as extended
except ModuleNotFoundError:
    from scripts import texel_tune as base
    from scripts import texel_tune_mobility_shape as extended


N_FEATURES = 4
N_PARAMS = 2 * N_FEATURES
EXTRA_START = (
    extended.N_SHAPE_FEATURES +
    extended.N_KING_PRESSURE_FEATURES +
    extended.N_POSITIONAL_SPACE_FEATURES
)
PARAMETER_NAMES = [
    "unsafe_mob_n_mg", "unsafe_mob_n_eg",
    "unsafe_mob_b_mg", "unsafe_mob_b_eg",
    "unsafe_mob_r_mg", "unsafe_mob_r_eg",
    "unsafe_mob_q_mg", "unsafe_mob_q_eg",
]


def load_weights(path: str | None) -> np.ndarray:
    if not path:
        return np.zeros(N_PARAMS, dtype=np.float64)
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("UNSAFE_MOBILITY ")
    ]
    if not lines:
        raise SystemExit(f"no UNSAFE_MOBILITY line found in {path}")
    values = np.array(
        [int(value) for value in lines[-1].split()[1:]],
        dtype=np.float64,
    )
    if len(values) != N_PARAMS:
        raise SystemExit(
            f"expected {N_PARAMS} unsafe-mobility values in {path}, "
            f"got {len(values)}"
        )
    return values


def load_features(path: str):
    (
        label,
        phase,
        eval_true,
        white,
        white_extra,
        black,
        black_extra,
    ) = extended.load_extended_features(path)
    required = EXTRA_START + N_FEATURES
    if white_extra.shape[1] < required:
        raise ValueError(
            "feature dump has no unsafe-mobility columns; regenerate it "
            "with the current tunedump"
        )
    return (
        label,
        phase,
        eval_true,
        white,
        white_extra[:, EXTRA_START:required],
        black,
        black_extra[:, EXTRA_START:required],
    )


def correction_mg_eg(counts: np.ndarray, weights: np.ndarray):
    rounded = np.round(weights).astype(np.int64)
    mg = np.zeros(len(counts), dtype=np.int64)
    eg = np.zeros(len(counts), dtype=np.int64)
    for index in range(N_FEATURES):
        feature = counts[:, index].astype(np.int64)
        mg += feature * rounded[2 * index]
        eg += feature * rounded[2 * index + 1]
    return mg, eg


def design_matrix(
    phase: np.ndarray,
    white_counts: np.ndarray,
    black_counts: np.ndarray,
) -> np.ndarray:
    delta = (white_counts - black_counts).astype(np.float32)
    mg_weight = phase.astype(np.float32) / 24.0
    eg_weight = (24.0 - phase.astype(np.float32)) / 24.0
    design = np.zeros((len(phase), N_PARAMS), dtype=np.float32)
    for index in range(N_FEATURES):
        design[:, 2 * index] = delta[:, index] * mg_weight
        design[:, 2 * index + 1] = delta[:, index] * eg_weight
    return design


def verify_reconstruction(
    white: np.ndarray,
    white_counts: np.ndarray,
    black: np.ndarray,
    black_counts: np.ndarray,
    phase: np.ndarray,
    eval_true: np.ndarray,
    base_weights: np.ndarray,
    correction_weights: np.ndarray,
) -> np.ndarray:
    white_mg, white_eg = base.side_mg_eg_int(white, base_weights)
    black_mg, black_eg = base.side_mg_eg_int(black, base_weights)
    white_corr_mg, white_corr_eg = correction_mg_eg(
        white_counts,
        correction_weights,
    )
    black_corr_mg, black_corr_eg = correction_mg_eg(
        black_counts,
        correction_weights,
    )
    white_total = base.trunc_div24(
        (white_mg + white_corr_mg) * phase +
        (white_eg + white_corr_eg) * (24 - phase)
    )
    black_total = base.trunc_div24(
        (black_mg + black_corr_mg) * phase +
        (black_eg + black_corr_eg) * (24 - phase)
    )
    reconstructed_white = white_total - black_total
    return np.flatnonzero(
        np.abs(reconstructed_white) != np.abs(eval_true - 12)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats", required=True)
    parser.add_argument("--groups")
    parser.add_argument("--initial-tuned-file", required=True)
    parser.add_argument("--initial-unsafe-file")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    base_weights = base.load_tuned_defaults(args.initial_tuned_file)
    correction_weights = load_weights(args.initial_unsafe_file)
    (
        label,
        phase,
        eval_true,
        white,
        white_counts,
        black,
        black_counts,
    ) = load_features(args.feats)
    print(f"positions: {len(label)}", file=sys.stderr)

    bad = verify_reconstruction(
        white,
        white_counts,
        black,
        black_counts,
        phase,
        eval_true,
        base_weights,
        correction_weights,
    )
    print(
        f"verify: eval_true != reconstruction on "
        f"{'at least ' if len(bad) else ''}{len(bad)}/{len(label)} rows",
        file=sys.stderr,
    )
    if len(bad):
        print(
            "ABORT: unsafe-mobility reconstruction is not exact.",
            file=sys.stderr,
        )
        return 1
    print(
        "verify: OK (unsafe-mobility features reconstruct exactly)",
        file=sys.stderr,
    )
    if args.verify_only:
        return 0

    white_mg, white_eg = base.side_mg_eg_int(white, base_weights)
    black_mg, black_eg = base.side_mg_eg_int(black, base_weights)
    mg_weight = phase.astype(np.float32) / 24.0
    eg_weight = (24.0 - phase.astype(np.float32)) / 24.0
    fixed = (
        (white_mg - black_mg).astype(np.float32) * mg_weight +
        (white_eg - black_eg).astype(np.float32) * eg_weight
    )
    design = design_matrix(phase, white_counts, black_counts)

    if args.groups:
        groups = base.load_groups(args.groups, len(label))
        train, validation = base.grouped_split(
            groups,
            args.val_frac,
            args.seed,
        )
        print(
            f"split: {len(np.unique(groups[train]))} train games, "
            f"{len(np.unique(groups[validation]))} validation games",
            file=sys.stderr,
        )
    else:
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(label))
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(label) * args.val_frac)))
        validation = indices[:validation_count]
        train = indices[validation_count:]
        print("warning: using a row-random validation split", file=sys.stderr)

    theta = correction_weights.copy()
    train_x = design[train]
    train_fixed = fixed[train]
    train_y = label[train]
    K, baseline_train_loss = base.fit_K(
        train_fixed + train_x @ theta,
        train_y,
    )
    best_theta = theta.copy()
    best_iteration = 0
    best_validation_loss = base.loss_for(
        K,
        fixed[validation] + design[validation] @ theta,
        label[validation],
    )
    print(
        f"fit K={K:.5f} baseline train loss={baseline_train_loss:.6f} "
        f"val loss={best_validation_loss:.6f}",
        file=sys.stderr,
    )

    first_moment = np.zeros(N_PARAMS, dtype=np.float64)
    second_moment = np.zeros(N_PARAMS, dtype=np.float64)
    rng = np.random.default_rng(args.seed ^ 0x53414645)
    for iteration in range(1, args.iters + 1):
        if 0 < args.batch_size < len(train):
            batch = rng.integers(0, len(train), size=args.batch_size)
            batch_x = train_x[batch]
            batch_fixed = train_fixed[batch]
            batch_y = train_y[batch]
        else:
            batch_x = train_x
            batch_fixed = train_fixed
            batch_y = train_y
        evaluation = batch_fixed + batch_x @ theta
        probability = base.sigmoid(K * evaluation)
        gradient = (
            batch_x.T @ (
                2.0 * (probability - batch_y) *
                probability * (1.0 - probability) * K
            )
        ) / len(batch_y)
        gradient += (
            args.l2 * 1e-3 * (theta - correction_weights) /
            (20.0 ** 2)
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
        corrected_first = first_moment / (1.0 - 0.9 ** iteration)
        corrected_second = second_moment / (1.0 - 0.999 ** iteration)
        theta -= (
            args.lr * corrected_first /
            (np.sqrt(corrected_second) + 1e-8)
        )

        if iteration % 500 == 0:
            K, _ = base.fit_K(
                train_fixed + train_x @ theta,
                train_y,
            )
            train_loss = base.loss_for(
                K,
                train_fixed + train_x @ theta,
                train_y,
            )
            validation_loss = base.loss_for(
                K,
                fixed[validation] + design[validation] @ theta,
                label[validation],
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_theta = theta.copy()
                best_iteration = iteration
            print(
                f"  it {iteration:5d} K={K:.5f} train={train_loss:.6f} "
                f"val={validation_loss:.6f}",
                file=sys.stderr,
            )

    rounded = np.round(best_theta).astype(np.int64)
    print(
        f"selected iteration {best_iteration} with "
        f"val loss={best_validation_loss:.6f}",
        file=sys.stderr,
    )
    for name, old, new in zip(
        PARAMETER_NAMES,
        correction_weights,
        rounded,
    ):
        print(
            f"  {name:24s} {int(round(old)):5d} -> {int(new):5d}",
            file=sys.stderr,
        )
    print("UNSAFE_MOBILITY " + " ".join(str(int(value)) for value in rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
