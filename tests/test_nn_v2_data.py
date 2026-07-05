from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bot.nn.v2.data import (  # noqa: E402
    CP_CLIP,
    best_eval_item,
    convert_flat_eval_row,
    buckets_for_position,
    convert_lichess_eval_row,
    label_from_eval,
    normalize_fen_key,
)


def assert_close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"expected {expected}, got {actual}")


def test_white_perspective_eval_flips_to_side_to_move() -> None:
    white = label_from_eval("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "1.50", perspective="white")
    black = label_from_eval("4k3/8/8/8/8/8/8/4KQ2 b - - 0 1", "1.50", perspective="white")

    if white.cp_stm != 150:
        raise AssertionError(f"expected white stm cp 150, got {white.cp_stm}")
    if black.cp_stm != -150:
        raise AssertionError(f"expected black stm cp -150, got {black.cp_stm}")
    if not white.value_stm > 0.0 or not black.value_stm < 0.0:
        raise AssertionError("side-to-move values should have opposite signs")


def test_terminal_labels_override_eval() -> None:
    mate = label_from_eval("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "0.00", perspective="white")
    stalemate = label_from_eval("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "9.00", perspective="white")
    bare_kings = label_from_eval("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "9.00", perspective="white")

    if mate.terminal != "checkmate" or mate.cp_stm != -CP_CLIP or not mate.is_mate:
        raise AssertionError(f"bad mate label: {mate}")
    if stalemate.terminal != "stalemate" or stalemate.cp_stm != 0:
        raise AssertionError(f"bad stalemate label: {stalemate}")
    if bare_kings.terminal != "insufficient_material" or bare_kings.cp_stm != 0:
        raise AssertionError(f"bad insufficient-material label: {bare_kings}")


def test_best_eval_prefers_depth_then_knodes() -> None:
    row = [
        {"depth": 18, "knodes": 500, "pvs": [{"cp": 10, "line": "e2e4"}]},
        {"depth": 20, "knodes": 10, "pvs": [{"cp": 20, "line": "d2d4"}]},
        {"depth": 20, "knodes": 30, "pvs": [{"cp": 30, "line": "g1f3"}]},
    ]

    picked = best_eval_item(row, min_depth=18, min_knodes=0)
    if picked is None or picked["pv"]["line"] != "g1f3":
        raise AssertionError(f"unexpected picked eval: {picked}")
    if best_eval_item(row, min_depth=21, min_knodes=0) is not None:
        raise AssertionError("depth filter should reject all rows")
    picked_with_knodes = best_eval_item(row, min_depth=18, min_knodes=100)
    if picked_with_knodes is None or picked_with_knodes["pv"]["line"] != "e2e4":
        raise AssertionError(f"unexpected knodes-filtered eval: {picked_with_knodes}")


def test_convert_lichess_eval_row_adds_targets_and_buckets() -> None:
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "evals": [
            {"depth": 17, "knodes": 9999, "pvs": [{"cp": 99, "line": "e2e4"}]},
            {"depth": 22, "knodes": 100, "pvs": [{"cp": 34, "line": "e2e4 e7e5"}]},
        ],
    }

    out = convert_lichess_eval_row(row, min_depth=18)
    if out is None:
        raise AssertionError("expected converted row")
    if out["fen"] != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -":
        raise AssertionError(f"unexpected normalized fen: {out['fen']}")
    if out["depth"] != 22 or out["best_move"] != "e2e4":
        raise AssertionError(f"unexpected converted metadata: {out}")
    if out["cp_stm"] != 34 or out["bucket"] != "white/opening/equal/quiet":
        raise AssertionError(f"unexpected target/bucket: {out}")


def test_convert_flat_eval_row_supports_hf_schema() -> None:
    row = {
        "fen": "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
        "line": "f7g7 e6e2 h8d8",
        "depth": 46,
        "knodes": 4189972,
        "cp": 69,
        "mate": None,
    }

    out = convert_flat_eval_row(row, min_depth=18, source="hf:Lichess/chess-position-evaluations")
    if out is None:
        raise AssertionError("expected converted flat row")
    if out["depth"] != 46 or out["knodes"] != 4189972:
        raise AssertionError(f"unexpected flat metadata: {out}")
    if out["best_move"] != "f7g7":
        raise AssertionError(f"expected best move from line, got {out}")
    if out["cp_stm"] != -69:
        raise AssertionError(f"expected black side-to-move flip, got {out}")
    if out["source"] != "hf:Lichess/chess-position-evaluations":
        raise AssertionError(f"unexpected source: {out}")


def test_bucket_detects_black_side_and_capture() -> None:
    fen = "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"
    label = label_from_eval(fen, "-9.00", perspective="white")
    buckets = buckets_for_position(fen, label, "e1e2")

    if buckets.side_to_move != "white":
        raise AssertionError(f"unexpected side bucket: {buckets}")
    if buckets.eval_band != "losing":
        raise AssertionError(f"unexpected eval bucket: {buckets}")
    if buckets.tactical != "capture":
        raise AssertionError(f"unexpected tactical bucket: {buckets}")


def test_normalize_fen_drops_halfmove_and_fullmove() -> None:
    a = normalize_fen_key("4k3/8/8/8/8/8/8/4K3 w - - 12 55")
    b = normalize_fen_key("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    if a != b:
        raise AssertionError(f"expected same normalized key, got {a} vs {b}")


def main() -> None:
    test_white_perspective_eval_flips_to_side_to_move()
    test_terminal_labels_override_eval()
    test_best_eval_prefers_depth_then_knodes()
    test_convert_lichess_eval_row_adds_targets_and_buckets()
    test_convert_flat_eval_row_supports_hf_schema()
    test_bucket_detects_black_side_and_capture()
    test_normalize_fen_drops_halfmove_and_fullmove()
    assert_close(label_from_eval("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "0.00").value_stm, 0.0)
    print("test_nn_v2_data: OK")


if __name__ == "__main__":
    main()
