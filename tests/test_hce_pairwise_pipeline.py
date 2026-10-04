#!/usr/bin/env python3
"""Focused tests for the Stockfish-to-HCE pairwise tuning pipeline."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABEL_SCRIPT = ROOT / "scripts" / "stockfish_label_multipv.py"
BUILD_SCRIPT = ROOT / "scripts" / "build_hce_pairwise_dataset.py"
TUNE_SCRIPT = ROOT / "scripts" / "tune_hce_pairwise.py"
APPLY_SCRIPT = ROOT / "scripts" / "apply_hce_pairwise.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_position_loading_deduplicates_and_skips_bad_fens() -> None:
    label = load_module(LABEL_SCRIPT, "hce_pairwise_label_test")
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    other = "8/8/8/8/8/8/4K3/7k w - - 0 1"
    with tempfile.TemporaryDirectory() as tmp:
        positions = Path(tmp) / "positions.txt"
        positions.write_text(
            f"{start};1.0\n{start};0.0\nnot-a-fen;0.5\n{other};0.5\n",
            encoding="utf-8",
        )
        loaded = label.load_positions(positions, count=10, seed=3)
    assert sorted(loaded) == sorted([start, other])


def test_pair_builder_keeps_aligned_quiet_children() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    labels = {
        "fen": start,
        "depth": 12,
        "moves": [
            {"uci": "e2e4", "score_cp": 42},
            {"uci": "d2d4", "score_cp": 22},
            {"uci": "g1f3", "score_cp": 12},
            {"uci": "a2a3", "score_cp": -400},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        labels_path = tmp_path / "labels.jsonl"
        positions_path = tmp_path / "children.txt"
        metadata_path = tmp_path / "children.tsv"
        labels_path.write_text(json.dumps(labels) + "\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(BUILD_SCRIPT),
                "--labels",
                str(labels_path),
                "--positions-out",
                str(positions_path),
                "--metadata-out",
                str(metadata_path),
                "--alternatives",
                "4",
                "--min-gap-cp",
                "15",
                "--max-gap-cp",
                "300",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        children = positions_path.read_text(encoding="utf-8").splitlines()
        with metadata_path.open("r", encoding="utf-8", newline="") as source:
            metadata = list(csv.DictReader(source, delimiter="\t"))

    assert len(children) == len(metadata) == 3
    assert [row["move"] for row in metadata] == ["e2e4", "d2d4", "g1f3"]
    assert [row["is_best"] for row in metadata] == ["1", "0", "0"]
    assert [row["teacher_gap_cp"] for row in metadata] == ["0", "20", "30"]
    assert len({row["group_id"] for row in metadata}) == 1


def test_feature_design_uses_root_perspective_and_child_eval_sign() -> None:
    tune = load_module(TUNE_SCRIPT, "hce_pairwise_design_test")
    side_size = tune.BASE_SIDE + tune.EXTENDED_FEATURES
    raw = np.zeros((2, 4 + 2 * side_size), dtype=np.float32)
    raw[:, 0] = 0.5
    raw[0, 1] = 24
    raw[1, 1] = 0
    raw[0, 2] = -50
    raw[1, 2] = 30
    feature_column = tune.FEATURES[0][1]
    raw[0, 3 + feature_column] = 2
    black_start = 3 + side_size
    raw[1, 3 + feature_column] = 1
    raw[1, black_start + feature_column] = 4
    metadata = [{"root_side": "w"}, {"root_side": "b"}]
    with tempfile.TemporaryDirectory() as tmp:
        features = Path(tmp) / "features.txt"
        np.savetxt(features, raw, fmt="%.1f")
        design, base, qsearch_delta = tune.load_feature_design(features, metadata)

    assert design[0, 0] == 2
    assert design[0, 1] == 0
    assert design[1, 0] == 0
    assert design[1, 1] == 3
    assert base.tolist() == [50, -30]
    assert qsearch_delta.tolist() == [0, 0]


def test_pair_construction_and_grouped_split_do_not_leak_roots() -> None:
    tune = load_module(TUNE_SCRIPT, "hce_pairwise_group_test")
    design = np.arange(24, dtype=np.float64).reshape(4, 6)
    base = np.array([30.0, 10.0, -5.0, -20.0])
    metadata = [
        {"group_id": "a", "is_best": "1", "teacher_score_cp": "50"},
        {"group_id": "a", "is_best": "0", "teacher_score_cp": "20"},
        {"group_id": "b", "is_best": "1", "teacher_score_cp": "15"},
        {"group_id": "b", "is_best": "0", "teacher_score_cp": "0"},
    ]
    pair_design, pair_base, target, groups = tune.build_pairs(
        design, base, metadata
    )
    assert np.array_equal(pair_design[0], design[0] - design[1])
    assert pair_base.tolist() == [20.0, 15.0]
    assert target.tolist() == [30.0, 15.0]
    train, val = tune.grouped_split(groups, val_frac=0.5, seed=9)
    assert set(groups[train]).isdisjoint(set(groups[val]))

    qsearch_delta = np.array([0.0, 200.0, 0.0, 0.0])
    filtered = tune.build_pairs(
        design,
        base,
        metadata,
        qsearch_delta=qsearch_delta,
        max_qsearch_delta=120.0,
    )
    assert filtered[2].tolist() == [15.0]


def test_huber_ridge_improves_a_simple_pairwise_fit() -> None:
    tune = load_module(TUNE_SCRIPT, "hce_pairwise_fit_test")
    design = np.ones((40, 1), dtype=np.float64)
    base = np.zeros(40, dtype=np.float64)
    teacher = np.full(40, 25.0, dtype=np.float64)
    theta = tune.fit_huber_ridge(
        design,
        teacher - base,
        ridge=0.001,
        huber_delta=100.0,
    )
    before = tune.metrics(np.zeros(1), design, base, teacher)
    after = tune.metrics(theta, design, base, teacher)
    assert after["accuracy"] > before["accuracy"]
    assert after["mae_cp"] < 1.0


def test_family_selection_keeps_inactive_features_zero() -> None:
    tune = load_module(TUNE_SCRIPT, "hce_pairwise_family_test")
    families, columns = tune.active_feature_columns(
        "mobility_shape,unsafe_mobility"
    )
    assert families == ["mobility_shape", "unsafe_mobility"]
    assert len(columns) == 24
    assert len(columns) == len(set(columns))

    expanded = tune.expand_theta(
        np.arange(1, len(columns) + 1, dtype=np.float64),
        columns,
    )
    feature_index = {
        name: index for index, (name, _) in enumerate(tune.FEATURES)
    }
    pawn_push = feature_index["pawn_push"]
    restricted_knight = feature_index["restricted_mob_n"]
    unsafe_queen = feature_index["unsafe_mob_q"]
    assert expanded[2 * pawn_push:2 * pawn_push + 2].tolist() == [0.0, 0.0]
    assert expanded[2 * restricted_knight] != 0.0
    assert expanded[2 * unsafe_queen + 1] != 0.0

    try:
        tune.active_feature_columns("mobility_shape,unknown")
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown feature family was accepted")

    features, feature_columns = tune.named_feature_columns(
        "pawn_push,pawn_threat_minor"
    )
    assert features == ["pawn_push", "pawn_threat_minor"]
    assert feature_columns == [0, 1, 2, 3]


def test_apply_updates_every_reported_constant() -> None:
    apply = load_module(APPLY_SCRIPT, "hce_pairwise_apply_test")
    report = {
        "weights": {
            "pawn_push": {"mg": 7, "eg": -2},
            "unsafe_mob_n": {"mg": -4, "eg": -1},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        eval_copy = tmp_path / "hce_eval.c"
        report_path = tmp_path / "report.json"
        eval_copy.write_text(
            (ROOT / "src" / "core" / "engine" / "hce_eval.c").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        report_path.write_text(json.dumps(report), encoding="utf-8")
        apply.patch_eval(eval_copy, report_path)
        patched = eval_copy.read_text(encoding="utf-8")
    assert "static const int k_pawn_push_mg = 7;" in patched
    assert "static const int k_pawn_push_eg = -2;" in patched
    assert "static const int k_unsafe_mob_n_mg = -4;" in patched
    assert "static const int k_unsafe_mob_n_eg = -1;" in patched


if __name__ == "__main__":
    test_position_loading_deduplicates_and_skips_bad_fens()
    test_pair_builder_keeps_aligned_quiet_children()
    test_feature_design_uses_root_perspective_and_child_eval_sign()
    test_pair_construction_and_grouped_split_do_not_leak_roots()
    test_huber_ridge_improves_a_simple_pairwise_fit()
    test_family_selection_keeps_inactive_features_zero()
    test_apply_updates_every_reported_constant()
    print("test_hce_pairwise_pipeline: OK")
