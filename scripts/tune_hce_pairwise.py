#!/usr/bin/env python3
"""Fit pure-HCE correction terms to Stockfish quiet-move preferences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SIDE_OLD = 37
PST_FEATURES = 6 * 64
BASE_SIDE = SIDE_OLD + PST_FEATURES
EXTENDED_FEATURES = 18

FEATURES: list[tuple[str, int]] = [
    ("pawn_push", 20),
    ("pawn_threat_minor", 21),
    ("pawn_threat_major", 22),
    ("connected_pawn", 23),
    ("phalanx_pawn", 24),
    ("backward_pawn", 25),
    ("knight_outpost", 26),
    ("bishop_pair", 27),
    ("rook_behind_passer", 28),
    ("minor_threat_pawn", 29),
    ("minor_threat_minor", 30),
    ("minor_threat_major", 31),
    ("rook_threat_minor", 32),
    ("safe_push_threat_minor", 33),
    ("safe_push_threat_major", 34),
    ("restricted_mob_n", BASE_SIDE + 0),
    ("restricted_mob_b", BASE_SIDE + 1),
    ("restricted_mob_r", BASE_SIDE + 2),
    ("restricted_mob_q", BASE_SIDE + 3),
    ("active_mob_n", BASE_SIDE + 4),
    ("active_mob_b", BASE_SIDE + 5),
    ("active_mob_r", BASE_SIDE + 6),
    ("active_mob_q", BASE_SIDE + 7),
    ("king_ring_coverage", BASE_SIDE + 8),
    ("king_ring_double", BASE_SIDE + 9),
    ("bad_bishop_pawns", BASE_SIDE + 10),
    ("safe_space", BASE_SIDE + 11),
    ("deep_space", BASE_SIDE + 12),
    ("advanced_center_pawns", BASE_SIDE + 13),
    ("unsafe_mob_n", BASE_SIDE + 14),
    ("unsafe_mob_b", BASE_SIDE + 15),
    ("unsafe_mob_r", BASE_SIDE + 16),
    ("unsafe_mob_q", BASE_SIDE + 17),
]

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "pawn_activity": (
        "pawn_push",
        "pawn_threat_minor",
        "pawn_threat_major",
    ),
    "structure": (
        "connected_pawn",
        "phalanx_pawn",
        "backward_pawn",
        "knight_outpost",
        "bishop_pair",
        "rook_behind_passer",
    ),
    "threats": (
        "minor_threat_pawn",
        "minor_threat_minor",
        "minor_threat_major",
        "rook_threat_minor",
        "safe_push_threat_minor",
        "safe_push_threat_major",
    ),
    "mobility_shape": (
        "restricted_mob_n",
        "restricted_mob_b",
        "restricted_mob_r",
        "restricted_mob_q",
        "active_mob_n",
        "active_mob_b",
        "active_mob_r",
        "active_mob_q",
    ),
    "king_pressure": (
        "king_ring_coverage",
        "king_ring_double",
    ),
    "space": (
        "bad_bishop_pawns",
        "safe_space",
        "deep_space",
        "advanced_center_pawns",
    ),
    "unsafe_mobility": (
        "unsafe_mob_n",
        "unsafe_mob_b",
        "unsafe_mob_r",
        "unsafe_mob_q",
    ),
}


def active_feature_columns(value: str) -> tuple[list[str], list[int]]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if not requested or requested == ["all"]:
        return list(FEATURE_FAMILIES), list(range(2 * len(FEATURES)))
    if "all" in requested:
        raise ValueError("'all' cannot be combined with named feature families")
    unknown = sorted(set(requested) - FEATURE_FAMILIES.keys())
    if unknown:
        raise ValueError(f"unknown feature families: {', '.join(unknown)}")

    feature_by_name = {name: index for index, (name, _) in enumerate(FEATURES)}
    columns: list[int] = []
    for family in requested:
        for feature in FEATURE_FAMILIES[family]:
            index = feature_by_name[feature]
            columns.extend((2 * index, 2 * index + 1))
    return requested, columns


def named_feature_columns(value: str) -> tuple[list[str], list[int]]:
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if not requested:
        raise ValueError("at least one feature name is required")
    feature_by_name = {name: index for index, (name, _) in enumerate(FEATURES)}
    unknown = sorted(set(requested) - feature_by_name.keys())
    if unknown:
        raise ValueError(f"unknown features: {', '.join(unknown)}")
    columns: list[int] = []
    for feature in requested:
        index = feature_by_name[feature]
        columns.extend((2 * index, 2 * index + 1))
    return requested, columns


def expand_theta(theta: np.ndarray, columns: list[int]) -> np.ndarray:
    expanded = np.zeros(2 * len(FEATURES), dtype=np.float64)
    expanded[columns] = theta
    return expanded


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def load_feature_design(path: Path, metadata: list[dict[str, str]]):
    raw = np.atleast_2d(np.loadtxt(path, dtype=np.float32))
    if len(raw) != len(metadata):
        raise ValueError(
            f"feature rows ({len(raw)}) do not match metadata ({len(metadata)})"
        )
    side_size = (raw.shape[1] - 4) // 2
    if raw.shape[1] != 4 + 2 * side_size:
        raise ValueError("expected tunedumpall output with a qsearch-delta column")
    if side_size < BASE_SIDE + EXTENDED_FEATURES:
        raise ValueError(
            f"feature side has {side_size} columns; expected at least "
            f"{BASE_SIDE + EXTENDED_FEATURES}"
        )

    phase = raw[:, 1]
    eval_stm = raw[:, 2].astype(np.float64)
    white = raw[:, 3:3 + side_size]
    black_start = 3 + side_size
    black = raw[:, black_start:black_start + side_size]
    qsearch_delta = raw[:, -1].astype(np.float64)
    mg = phase / 24.0
    eg = (24.0 - phase) / 24.0
    root_sign = np.array(
        [1.0 if row["root_side"] == "w" else -1.0 for row in metadata],
        dtype=np.float64,
    )

    design = np.zeros((len(raw), 2 * len(FEATURES)), dtype=np.float64)
    for feature_index, (_, column) in enumerate(FEATURES):
        delta = (white[:, column] - black[:, column]).astype(np.float64)
        design[:, 2 * feature_index] = root_sign * delta * mg
        design[:, 2 * feature_index + 1] = root_sign * delta * eg

    # Every feature row is the child after the root move, so its side to move
    # is the root player's opponent. The +12 tempo term cancels inside a pair.
    base_root_eval = -eval_stm
    return design, base_root_eval, qsearch_delta


def build_pairs(
    design: np.ndarray,
    base_eval: np.ndarray,
    metadata: list[dict[str, str]],
    qsearch_delta: np.ndarray | None = None,
    max_qsearch_delta: float = float("inf"),
):
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        groups.setdefault(row["group_id"], []).append(index)

    pair_design: list[np.ndarray] = []
    pair_base: list[float] = []
    pair_target: list[float] = []
    pair_group: list[str] = []
    for group_id, indices in groups.items():
        best = [index for index in indices if metadata[index]["is_best"] == "1"]
        if len(best) != 1:
            continue
        best_index = best[0]
        if (qsearch_delta is not None and
                abs(qsearch_delta[best_index]) > max_qsearch_delta):
            continue
        best_score = float(metadata[best_index]["teacher_score_cp"])
        for alt_index in indices:
            if alt_index == best_index:
                continue
            if (qsearch_delta is not None and
                    abs(qsearch_delta[alt_index]) > max_qsearch_delta):
                continue
            alt_score = float(metadata[alt_index]["teacher_score_cp"])
            gap = best_score - alt_score
            if gap <= 0:
                continue
            pair_design.append(design[best_index] - design[alt_index])
            pair_base.append(base_eval[best_index] - base_eval[alt_index])
            pair_target.append(gap)
            pair_group.append(group_id)
    if not pair_design:
        raise ValueError("no valid best-versus-alternative pairs")
    return (
        np.stack(pair_design),
        np.asarray(pair_base),
        np.asarray(pair_target),
        np.asarray(pair_group),
    )


def grouped_split(groups: np.ndarray, val_frac: float, seed: int):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    nval = min(len(unique) - 1, max(1, int(round(len(unique) * val_frac))))
    is_val = np.isin(groups, unique[:nval])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def fit_huber_ridge(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    huber_delta: float,
    iterations: int = 8,
) -> np.ndarray:
    scale = np.sqrt(np.mean(design * design, axis=0))
    scale[scale < 1e-6] = 1.0
    x = design / scale
    theta = np.zeros(x.shape[1], dtype=np.float64)
    identity = np.eye(x.shape[1], dtype=np.float64)
    for _ in range(iterations):
        residual = target - x @ theta
        weights = np.minimum(1.0, huber_delta / np.maximum(np.abs(residual), 1e-9))
        root_weights = np.sqrt(weights)
        weighted_x = x * root_weights[:, None]
        weighted_y = target * root_weights
        lhs = weighted_x.T @ weighted_x + ridge * len(x) * identity
        rhs = weighted_x.T @ weighted_y
        theta = np.linalg.solve(lhs, rhs)
    return theta / scale


def metrics(
    theta: np.ndarray,
    design: np.ndarray,
    base: np.ndarray,
    teacher_gap: np.ndarray,
) -> dict[str, float]:
    predicted = base + design @ theta
    return {
        "accuracy": float(np.mean(predicted > 0.0)),
        "mae_cp": float(np.mean(np.abs(predicted - teacher_gap))),
        "margin_cp": float(np.mean(predicted)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="31,32,33")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=100.0)
    parser.add_argument("--max-weight", type=float, default=64.0)
    parser.add_argument(
        "--only-families",
        default="all",
        help=(
            "Comma-separated feature families to fit. Choices: "
            + ", ".join(FEATURE_FAMILIES)
            + ". The default fits all families."
        ),
    )
    parser.add_argument(
        "--only-features",
        default="",
        help=(
            "Comma-separated feature names to fit. This overrides the default "
            "all-family selection and cannot be combined with an explicit "
            "--only-families value."
        ),
    )
    parser.add_argument(
        "--max-qsearch-delta",
        type=float,
        default=120.0,
        help="Drop children where tactics move qsearch too far from static eval.",
    )
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("at least one seed is required")
    try:
        if args.only_features:
            if args.only_families != "all":
                raise ValueError(
                    "--only-features cannot be combined with --only-families"
                )
            active_features, active_columns = named_feature_columns(
                args.only_features
            )
            active_families: list[str] = []
        else:
            active_families, active_columns = active_feature_columns(
                args.only_families
            )
            active_features = [
                FEATURES[column // 2][0]
                for column in active_columns[::2]
            ]
    except ValueError as error:
        parser.error(str(error))
    metadata = load_metadata(args.metadata)
    row_design, row_base, qsearch_delta = load_feature_design(
        args.features, metadata
    )
    design, base, teacher_gap, groups = build_pairs(
        row_design,
        row_base,
        metadata,
        qsearch_delta,
        args.max_qsearch_delta,
    )
    residual_target = teacher_gap - base
    fit_design = design[:, active_columns]
    ridges = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    folds: list[dict[str, object]] = []
    selected_ridges: list[float] = []
    fold_thetas: list[np.ndarray] = []
    for seed in seeds:
        train, val = grouped_split(groups, args.val_frac, seed)
        candidates: list[tuple[float, float, float, np.ndarray]] = []
        for ridge in ridges:
            theta = fit_huber_ridge(
                fit_design[train],
                residual_target[train],
                ridge,
                args.huber_delta,
            )
            theta = np.clip(theta, -args.max_weight, args.max_weight)
            result = metrics(
                theta,
                fit_design[val],
                base[val],
                teacher_gap[val],
            )
            candidates.append((
                -result["accuracy"],
                result["mae_cp"],
                ridge,
                theta,
            ))
        _, _, ridge, theta = min(candidates, key=lambda item: item[:3])
        selected_ridges.append(ridge)
        fold_thetas.append(theta)
        folds.append({
            "seed": seed,
            "ridge": ridge,
            "baseline": metrics(
                np.zeros(fit_design.shape[1]),
                fit_design[val],
                base[val],
                teacher_gap[val],
            ),
            "tuned": metrics(
                theta,
                fit_design[val],
                base[val],
                teacher_gap[val],
            ),
            "validation_pairs": int(len(val)),
        })

    final_ridge = float(np.median(selected_ridges))
    theta = expand_theta(
        np.median(np.stack(fold_thetas), axis=0),
        active_columns,
    )
    expanded_folds = [
        expand_theta(fold_theta, active_columns)
        for fold_theta in fold_thetas
    ]
    rounded = np.rint(theta).astype(int)
    weights = {
        name: {"mg": int(rounded[2 * index]), "eg": int(rounded[2 * index + 1])}
        for index, (name, _) in enumerate(FEATURES)
    }
    stability = {
        name: {
            "mg": [int(round(values[2 * index])) for values in expanded_folds],
            "eg": [
                int(round(values[2 * index + 1]))
                for values in expanded_folds
            ],
        }
        for index, (name, _) in enumerate(FEATURES)
    }
    report = {
        "rows": len(metadata),
        "groups": int(len(np.unique(groups))),
        "pairs": int(len(groups)),
        "selected_ridge": final_ridge,
        "final_method": "coordinate_median_of_grouped_fits",
        "active_families": active_families,
        "active_features": active_features,
        "folds": folds,
        "full_baseline": metrics(
            np.zeros(design.shape[1]), design, base, teacher_gap
        ),
        "full_tuned": metrics(rounded.astype(float), design, base, teacher_gap),
        "weights": weights,
        "fold_weights": stability,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.out)

    print(f"pairs: {report['pairs']} from {report['groups']} roots")
    for fold in folds:
        baseline_result = fold["baseline"]
        tuned_result = fold["tuned"]
        print(
            f"seed {fold['seed']} ridge={fold['ridge']}: "
            f"accuracy {baseline_result['accuracy']:.3f} -> "
            f"{tuned_result['accuracy']:.3f}, MAE "
            f"{baseline_result['mae_cp']:.1f} -> {tuned_result['mae_cp']:.1f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
