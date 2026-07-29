#!/usr/bin/env python3
"""Regression tests for game-grouped HCE Texel data and validation splits."""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "texel_build_dataset.py"
TUNE_SCRIPT = ROOT / "scripts" / "texel_tune.py"
APPLY_SCRIPT = ROOT / "scripts" / "texel_apply_tune.py"
MOBILITY_TUNE_SCRIPT = ROOT / "scripts" / "texel_tune_mobility_shape.py"
MOBILITY_APPLY_SCRIPT = ROOT / "scripts" / "texel_apply_mobility_shape.py"
KING_TUNE_SCRIPT = ROOT / "scripts" / "texel_tune_king_pressure.py"
KING_APPLY_SCRIPT = ROOT / "scripts" / "texel_apply_king_pressure.py"
KING_PIPELINE_SCRIPT = (
    ROOT / "scripts" / "run_hce_king_pressure_pipeline.py"
)
SPACE_TUNE_SCRIPT = ROOT / "scripts" / "texel_tune_positional_space.py"
SPACE_APPLY_SCRIPT = ROOT / "scripts" / "texel_apply_positional_space.py"
SPACE_PIPELINE_SCRIPT = (
    ROOT / "scripts" / "run_hce_positional_space_pipeline.py"
)
PAWN_ACTIVITY_PIPELINE_SCRIPT = (
    ROOT / "scripts" / "run_hce_pawn_activity_pipeline.py"
)
UNSAFE_MOBILITY_TUNE_SCRIPT = (
    ROOT / "scripts" / "texel_tune_unsafe_mobility.py"
)
UNSAFE_MOBILITY_APPLY_SCRIPT = (
    ROOT / "scripts" / "texel_apply_unsafe_mobility.py"
)
UNSAFE_MOBILITY_PIPELINE_SCRIPT = (
    ROOT / "scripts" / "run_hce_unsafe_mobility_pipeline.py"
)
THREAT_RECOVERY_SCRIPT = (
    ROOT / "scripts" / "run_hce_threat_recovery_sweep.py"
)
REMOTE_PIPELINE_SCRIPT = ROOT / "scripts" / "run_hce_v2_remote_pipeline.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    added_paths = [str(ROOT), str(ROOT / "scripts")]
    sys.path[:0] = added_paths
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[:len(added_paths)]
    return module


def test_grouped_split_has_no_game_leakage() -> None:
    tune = load_module(TUNE_SCRIPT, "texel_tune_test")
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    train, val = tune.grouped_split(groups, val_frac=0.25, seed=7)
    assert len(train) + len(val) == len(groups)
    assert set(groups[train]).isdisjoint(set(groups[val]))
    assert len(set(groups[val])) == 1


def test_dataset_group_sidecar_stays_aligned() -> None:
    pgn = """[Event "g1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "g2"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 0-1
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pgn_path = tmp_path / "games.pgn"
        data_path = tmp_path / "positions.txt"
        groups_path = tmp_path / "groups.txt"
        pgn_path.write_text(pgn, encoding="utf-8")
        subprocess.run(
            [
                sys.executable, str(BUILD_SCRIPT),
                "--pgn", str(pgn_path),
                "--out", str(data_path),
                "--groups-out", str(groups_path),
                "--group-prefix", "fixture",
                "--skip-opening", "2",
                "--skip-tail", "1",
                "--per-game", "3",
            ],
            check=True,
        )
        positions = data_path.read_text(encoding="utf-8").splitlines()
        groups = groups_path.read_text(encoding="utf-8").splitlines()
        assert len(positions) == len(groups) == 6
        assert groups[:3] == ["fixture-1"] * 3
        assert groups[3:] == ["fixture-2"] * 3


def test_apply_tune_accepts_previous_vector_shapes() -> None:
    apply_tune = load_module(APPLY_SCRIPT, "texel_apply_tune_test")
    for scalar_count in (21, 35, 47, 59):
        values = [0] * (scalar_count + 2 * 6 * 64)
        parsed = apply_tune.parse_tuned_line(
            "TUNED " + " ".join(str(value) for value in values))
        assert len(parsed) == 59 + 2 * 6 * 64
        if scalar_count < 47:
            assert parsed[35:47] == [0] * 12
        if scalar_count < 59:
            assert parsed[47:59] == [0] * 12


def test_apply_tune_updates_combined_mobility_expression() -> None:
    apply_tune = load_module(APPLY_SCRIPT, "texel_apply_tune_mobility_test")
    values = [0] * (59 + 2 * 6 * 64)
    values[9:17] = [5, 3, 10, -1, 8, 7, 5, 2]
    with tempfile.TemporaryDirectory() as tmp:
        eval_copy = Path(tmp) / "hce_eval.c"
        eval_copy.write_text(
            (ROOT / "src" / "core" / "engine" / "hce_eval.c").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        apply_tune.patch_eval_c(eval_copy, values)
        patched = eval_copy.read_text(encoding="utf-8")
        assert (
            "knight_mob * 5 + bishop_mob * 10 + "
            "rook_mob * 8 + queen_mob * 5"
        ) in patched
        assert (
            "knight_mob * 3 + bishop_mob * -1 + "
            "rook_mob * 7 + queen_mob * 2"
        ) in patched


def test_hce_v2_feature_detectors() -> None:
    tune = load_module(TUNE_SCRIPT, "texel_tune_feature_test")
    fixtures = [
        # A supported pawn chain plus adjacent pawns on the fourth rank.
        "7k/8/8/8/3PPP2/2P5/8/7K w - - 0 1",
        # White d3 is backward: c4 is ahead and black e5 controls d4.
        "7k/8/8/4p3/2P5/3P4/8/7K w - - 0 1",
        # Pawn-supported d5 knight and two bishops.
        "7k/8/8/3N4/2P5/8/BB6/7K w - - 0 1",
        # Rook on d2 is behind a clear passed pawn on d5.
        "7k/8/8/3P4/8/8/3R4/7K w - - 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        features_path = tmp_path / "features.txt"
        positions_path.write_text(
            "".join(f"{fen};0.5\n" for fen in fixtures),
            encoding="utf-8",
        )
        subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=(
                f"tunedump {positions_path} {features_path}\n"
                "quit\n"
            ),
            text=True,
            check=True,
            capture_output=True,
        )
        _, _, _, white, _ = tune.build(features_path)
        assert len(white) == len(fixtures)
        old, _ = tune.split_side(white)

        assert old[0, tune.F_CONNECTED] >= 1
        assert old[0, tune.F_PHALANX] >= 2
        assert old[1, tune.F_BACKWARD] == 1
        assert old[2, tune.F_KNIGHT_OUTPOST] == 1
        assert old[2, tune.F_BISHOP_PAIR] == 1
        assert old[3, tune.F_ROOK_BEHIND_PASSER] == 1


def test_hce_v3_threat_feature_detectors() -> None:
    tune = load_module(TUNE_SCRIPT, "texel_tune_threat_feature_test")
    fixtures = [
        # Knight d4 attacks an undefended pawn on f5.
        "7k/8/8/5p2/3N4/8/8/7K w - - 0 1",
        # Knight d4 attacks an undefended bishop on f5 and rook on b5.
        "7k/8/8/1r3b2/3N4/8/8/7K w - - 0 1",
        # Rook a1 attacks an undefended bishop on a5.
        "7k/8/8/b7/8/8/8/R6K w - - 0 1",
        # A safe d4-d5 push would attack the knight c6 and rook e6.
        "7k/8/2n1r3/8/3P4/8/8/7K w - - 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        features_path = tmp_path / "features.txt"
        positions_path.write_text(
            "".join(f"{fen};0.5\n" for fen in fixtures),
            encoding="utf-8",
        )
        subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=(
                f"tunedumpall {positions_path} {features_path}\n"
                "quit\n"
            ),
            text=True,
            check=True,
            capture_output=True,
        )
        raw = np.atleast_2d(np.loadtxt(features_path, dtype=np.float32))
        side_features = tune.SIDE_OLD + tune.N_PST
        white = raw[:, 3:3 + side_features]
        old, _ = tune.split_side(white)

        assert old[0, tune.F_MINOR_THREAT_PAWN] == 1
        assert old[1, tune.F_MINOR_THREAT_MINOR] == 1
        assert old[1, tune.F_MINOR_THREAT_MAJOR] == 1
        assert old[2, tune.F_ROOK_THREAT_MINOR] == 1
        assert old[3, tune.F_SAFE_PUSH_THREAT_MINOR] == 1
        assert old[3, tune.F_SAFE_PUSH_THREAT_MAJOR] == 1


def test_hce_mobility_shape_feature_detectors() -> None:
    tune = load_module(
        MOBILITY_TUNE_SCRIPT,
        "texel_tune_mobility_shape_feature_test",
    )
    fixtures = [
        "k7/8/8/8/8/8/8/N6K w - - 0 1",
        "k7/8/8/8/8/8/1P1P4/2B4K w - - 0 1",
        "k7/8/8/8/8/8/P7/RN5K w - - 0 1",
        "k7/8/8/8/8/8/PP6/QN5K w - - 0 1",
        "k7/8/8/8/3N4/8/8/7K w - - 0 1",
        "k7/8/8/8/3B4/8/8/7K w - - 0 1",
        "k7/8/8/8/3R4/8/8/7K w - - 0 1",
        "k7/8/8/8/3Q4/8/8/7K w - - 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        features_path = tmp_path / "features.txt"
        positions_path.write_text(
            "".join(f"{fen};0.5\n" for fen in fixtures),
            encoding="utf-8",
        )
        subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=(
                f"tunedumpall {positions_path} {features_path}\n"
                "quit\n"
            ),
            text=True,
            check=True,
            capture_output=True,
        )
        _, _, _, _, white_shape, _, _ = tune.load_features(features_path)

        for index in range(4):
            assert white_shape[index, index] == 1
        for index in range(4):
            assert white_shape[index + 4, index + 4] == 1


def test_apply_mobility_shape_updates_all_constants() -> None:
    apply_shape = load_module(
        MOBILITY_APPLY_SCRIPT,
        "texel_apply_mobility_shape_test",
    )
    values = list(range(-8, 8))
    with tempfile.TemporaryDirectory() as tmp:
        eval_copy = Path(tmp) / "hce_eval.c"
        eval_copy.write_text(
            (ROOT / "src" / "core" / "engine" / "hce_eval.c").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        apply_shape.patch_eval_c(eval_copy, values)
        patched = eval_copy.read_text(encoding="utf-8")
        for name, value in zip(apply_shape.SHAPE_NAMES, values):
            assert f"static const int k_{name} = {value};" in patched


def test_hce_king_pressure_feature_detectors() -> None:
    tune = load_module(
        KING_TUNE_SCRIPT,
        "texel_tune_king_pressure_feature_test",
    )
    fen = "k6r/8/2b5/8/8/8/8/7K w - - 0 1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        features_path = tmp_path / "features.txt"
        positions_path.write_text(f"{fen};0.5\n", encoding="utf-8")
        subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=f"tunedumpall {positions_path} {features_path}\nquit\n",
            text=True,
            check=True,
            capture_output=True,
        )
        _, _, _, _, white_counts, _, _ = tune.load_features(features_path)
        assert white_counts[0, 0] == 3
        assert white_counts[0, 1] == 1


def test_apply_king_pressure_updates_all_constants() -> None:
    apply_pressure = load_module(
        KING_APPLY_SCRIPT,
        "texel_apply_king_pressure_test",
    )
    values = [-8, -3, -5, -2]
    with tempfile.TemporaryDirectory() as tmp:
        eval_copy = Path(tmp) / "hce_eval.c"
        eval_copy.write_text(
            (ROOT / "src" / "core" / "engine" / "hce_eval.c").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        apply_pressure.patch_eval_c(eval_copy, values)
        patched = eval_copy.read_text(encoding="utf-8")
        for name, value in zip(apply_pressure.PARAMETER_NAMES, values):
            assert f"static const int k_{name} = {value};" in patched


def test_king_pressure_pipeline_uses_coordinate_median() -> None:
    pipeline = load_module(
        KING_PIPELINE_SCRIPT,
        "hce_king_pressure_pipeline_test",
    )
    vectors = [
        [-8, -3, -4, -2],
        [-5, -7, -6, -1],
        [-7, -4, -3, -5],
    ]
    assert pipeline.coordinate_median(vectors) == [-7, -4, -4, -2]


def test_hce_positional_space_feature_detectors() -> None:
    tune = load_module(
        SPACE_TUNE_SCRIPT,
        "texel_tune_positional_space_feature_test",
    )
    fen = "k7/8/8/8/3P4/8/1P1P4/2B4K w - - 0 1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        features_path = tmp_path / "features.txt"
        positions_path.write_text(f"{fen};0.5\n", encoding="utf-8")
        subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=f"tunedumpall {positions_path} {features_path}\nquit\n",
            text=True,
            check=True,
            capture_output=True,
        )
        _, _, _, _, white_counts, _, _ = tune.load_features(features_path)
        assert white_counts[0].tolist() == [3, 4, 2, 1]


def test_apply_positional_space_updates_all_constants() -> None:
    apply_space = load_module(
        SPACE_APPLY_SCRIPT,
        "texel_apply_positional_space_test",
    )
    values = [-4, -2, 12, 4, -4, -1, 22, 8]
    with tempfile.TemporaryDirectory() as tmp:
        eval_copy = Path(tmp) / "hce_eval.c"
        eval_copy.write_text(
            (ROOT / "src" / "core" / "engine" / "hce_eval.c").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        apply_space.patch_eval_c(eval_copy, values)
        patched = eval_copy.read_text(encoding="utf-8")
        for name, value in zip(apply_space.PARAMETER_NAMES, values):
            assert f"static const int k_{name} = {value};" in patched


def test_positional_space_pipeline_uses_coordinate_median() -> None:
    pipeline = load_module(
        SPACE_PIPELINE_SCRIPT,
        "run_hce_positional_space_pipeline_test",
    )
    vectors = [
        [-5, -2, 16, -1, -6, 0, 18, 8],
        [-4, -1, 18, 1, -3, 2, 20, 11],
        [-3, 0, 17, 0, -4, 1, 19, 10],
    ]
    assert pipeline.coordinate_median(vectors) == [
        -4, -1, 17, 0, -4, 1, 19, 10,
    ]


def test_hce_unsafe_mobility_feature_detectors() -> None:
    tune = load_module(
        UNSAFE_MOBILITY_TUNE_SCRIPT,
        "texel_tune_unsafe_mobility_feature_test",
    )
    fixtures = [
        # Knight d4 can move to b5/f5, both controlled by black pawns.
        "7k/8/2p1p3/8/3N4/8/8/7K w - - 0 1",
        # Bishop d4 can move to c5/e5, both controlled by black pawns.
        "7k/8/1p3p2/8/3B4/8/8/7K w - - 0 1",
        # Rook d4 can move to d5, controlled by both black pawns.
        "7k/8/2p1p3/8/3R4/8/8/7K w - - 0 1",
        # Queen d4 can move to c5/e5, both controlled by black pawns.
        "7k/8/1p3p2/8/3Q4/8/8/7K w - - 0 1",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        features_path = tmp_path / "features.txt"
        positions_path.write_text(
            "".join(f"{fen};0.5\n" for fen in fixtures),
            encoding="utf-8",
        )
        subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=f"tunedumpall {positions_path} {features_path}\nquit\n",
            text=True,
            check=True,
            capture_output=True,
        )
        _, _, _, _, white_counts, _, _ = tune.load_features(features_path)
        assert white_counts.tolist() == [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 2],
        ]


def test_apply_unsafe_mobility_updates_all_constants() -> None:
    apply_unsafe = load_module(
        UNSAFE_MOBILITY_APPLY_SCRIPT,
        "texel_apply_unsafe_mobility_test",
    )
    values = [-5, -2, -7, -3, -4, -1, -3, 0]
    with tempfile.TemporaryDirectory() as tmp:
        eval_copy = Path(tmp) / "hce_eval.c"
        eval_copy.write_text(
            (ROOT / "src" / "core" / "engine" / "hce_eval.c").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        apply_unsafe.patch_eval_c(eval_copy, values)
        patched = eval_copy.read_text(encoding="utf-8")
        for name, value in zip(apply_unsafe.PARAMETER_NAMES, values):
            assert f"static const int k_{name} = {value};" in patched


def test_unsafe_mobility_pipeline_uses_coordinate_median() -> None:
    pipeline = load_module(
        UNSAFE_MOBILITY_PIPELINE_SCRIPT,
        "run_hce_unsafe_mobility_pipeline_test",
    )
    vectors = [
        [-7, -2, -8, -4, -5, -2, -4, 0],
        [-5, -3, -6, -2, -4, -1, -3, -1],
        [-6, -1, -7, -3, -6, 0, -2, 1],
    ]
    assert pipeline.coordinate_median(vectors) == [
        -6, -2, -7, -3, -5, -1, -3, 0,
    ]


def test_pawn_activity_pipeline_uses_coordinate_median() -> None:
    pipeline = load_module(
        PAWN_ACTIVITY_PIPELINE_SCRIPT,
        "run_hce_pawn_activity_pipeline_test",
    )
    baseline = list(range(pipeline.N_PARAMS))
    low = baseline.copy()
    high = baseline.copy()
    for index in range(pipeline.PAWN_START, pipeline.PAWN_START + 6):
        low[index] -= 3
        high[index] += 5
    assert pipeline.coordinate_median([high, baseline, low]) == baseline


def test_threat_recovery_scales_only_safe_push_weights() -> None:
    recovery = load_module(
        THREAT_RECOVERY_SCRIPT,
        "run_hce_threat_recovery_sweep_test",
    )
    source = list(range(recovery.N_TUNED))
    source[recovery.SAFE_PUSH_START:recovery.SAFE_PUSH_END] = [9, 4, 4, 2]

    direct = recovery.variant_vector(source, "zero_safe_push")
    half = recovery.variant_vector(source, "half_safe_push")

    assert direct[recovery.SAFE_PUSH_START:recovery.SAFE_PUSH_END] == [0, 0, 0, 0]
    assert half[recovery.SAFE_PUSH_START:recovery.SAFE_PUSH_END] == [4, 2, 2, 1]
    assert direct[:recovery.SAFE_PUSH_START] == source[:recovery.SAFE_PUSH_START]
    assert half[recovery.SAFE_PUSH_END:] == source[recovery.SAFE_PUSH_END:]


def test_tunedump_rejects_misaligned_groups() -> None:
    fen = "7k/8/8/8/8/8/8/7K w - - 0 1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        positions_path = tmp_path / "positions.txt"
        groups_in_path = tmp_path / "groups-in.txt"
        features_path = tmp_path / "features.txt"
        groups_out_path = tmp_path / "groups-out.txt"
        positions_path.write_text(
            f"{fen};0.5\n{fen};0.5\n",
            encoding="utf-8",
        )
        groups_in_path.write_text("game-1\n", encoding="utf-8")
        completed = subprocess.run(
            [str(ROOT / "bin" / "chess_uci")],
            input=(
                f"tunedump {positions_path} {features_path} "
                f"{groups_in_path} {groups_out_path}\n"
                "quit\n"
            ),
            text=True,
            check=True,
            capture_output=True,
        )
        assert "group sidecar ended early" in completed.stdout
        assert not features_path.exists()
        assert not groups_out_path.exists()


def test_remote_pipeline_uses_coordinate_median() -> None:
    pipeline = load_module(REMOTE_PIPELINE_SCRIPT, "hce_v2_remote_pipeline_test")
    first = [0] * pipeline.N_TUNED
    second = [0] * pipeline.N_TUNED
    third = [0] * pipeline.N_TUNED
    first[pipeline.V2_START] = 19
    second[pipeline.V2_START] = 11
    third[pipeline.V2_START] = 13
    median = pipeline.median_vector([first, second, third])
    assert median[pipeline.V2_START] == 13


def test_established_retune_guards_bishop_endgame_mobility() -> None:
    pipeline = load_module(
        ROOT / "scripts" / "run_hce_established_retune.py",
        "hce_established_retune_guard_test",
    )
    baseline = [0] * pipeline.N_TUNED
    tuned = [0] * pipeline.N_TUNED
    baseline[pipeline.BISHOP_EG_MOBILITY_INDEX] = 3
    tuned[pipeline.BISHOP_EG_MOBILITY_INDEX] = -1

    guarded, adjustments = pipeline.apply_behavioral_guards(tuned, baseline)

    assert guarded[pipeline.BISHOP_EG_MOBILITY_INDEX] == 3
    assert tuned[pipeline.BISHOP_EG_MOBILITY_INDEX] == -1
    assert adjustments == [{
        "parameter": "mob_b_eg",
        "raw": -1,
        "guarded": 3,
    }]


def test_established_retune_expands_previous_baseline_vector() -> None:
    pipeline = load_module(
        ROOT / "scripts" / "run_hce_established_retune.py",
        "hce_established_retune_baseline_test",
    )
    previous_length = pipeline.N_TUNED - pipeline.N_V2_SCALAR
    previous = list(range(previous_length))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.log"
        path.write_text(
            "TUNED " + " ".join(str(value) for value in previous) + "\n",
            encoding="utf-8",
        )
        expanded = pipeline.parse_baseline_vector(path)

    assert len(expanded) == pipeline.N_TUNED
    assert expanded[:pipeline.N_CURRENT_SCALAR] == \
        previous[:pipeline.N_CURRENT_SCALAR]
    assert expanded[
        pipeline.N_CURRENT_SCALAR:
        pipeline.N_CURRENT_SCALAR + pipeline.N_V2_SCALAR
    ] == [0] * pipeline.N_V2_SCALAR
    assert expanded[pipeline.N_CURRENT_SCALAR + pipeline.N_V2_SCALAR:] == \
        previous[pipeline.N_CURRENT_SCALAR:]


def test_remote_pipeline_summarizes_paired_match() -> None:
    pipeline = load_module(REMOTE_PIPELINE_SCRIPT, "hce_v2_match_summary_test")
    report = {
        "head_to_head": [{
            "candidate": "cand",
            "games": 120,
            "points": 64.0,
            "elo_diff": 23.2,
            "elo_ci_low": -20.0,
            "elo_ci_high": 66.0,
            "probability_better": 0.85,
            "paired_positions": 60,
            "paired": {
                "elo_diff": 29.0,
                "elo_ci_low": -12.0,
                "elo_ci_high": 70.0,
                "probability_better": 0.91,
            },
        }],
        "standings": [
            {"name": "cand", "engine_failures": 0},
            {"name": "base", "engine_failures": 0},
        ],
    }
    summary = pipeline.summarize_match(report, expected_games=120)
    assert summary["elo_diff"] == 23.2
    assert summary["paired_probability_better"] == 0.91
    assert summary["engine_failures"] == 0


if __name__ == "__main__":
    test_grouped_split_has_no_game_leakage()
    test_dataset_group_sidecar_stays_aligned()
    test_apply_tune_accepts_previous_vector_shapes()
    test_apply_tune_updates_combined_mobility_expression()
    test_hce_v2_feature_detectors()
    test_hce_v3_threat_feature_detectors()
    test_hce_mobility_shape_feature_detectors()
    test_apply_mobility_shape_updates_all_constants()
    test_hce_king_pressure_feature_detectors()
    test_apply_king_pressure_updates_all_constants()
    test_king_pressure_pipeline_uses_coordinate_median()
    test_hce_positional_space_feature_detectors()
    test_apply_positional_space_updates_all_constants()
    test_positional_space_pipeline_uses_coordinate_median()
    test_pawn_activity_pipeline_uses_coordinate_median()
    test_hce_unsafe_mobility_feature_detectors()
    test_apply_unsafe_mobility_updates_all_constants()
    test_unsafe_mobility_pipeline_uses_coordinate_median()
    test_threat_recovery_scales_only_safe_push_weights()
    test_tunedump_rejects_misaligned_groups()
    test_remote_pipeline_uses_coordinate_median()
    test_established_retune_guards_bishop_endgame_mobility()
    test_established_retune_expands_previous_baseline_vector()
    test_remote_pipeline_summarizes_paired_match()
    print("test_texel_pipeline: OK")
