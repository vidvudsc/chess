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
    decompress_dummy_chess_fen,
    fast_phase_from_fen,
    label_from_eval,
    material_sanity_ok,
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


def test_fast_convert_preserves_label_fields() -> None:
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 7 9",
        "evals": [
            {"depth": 22, "knodes": 100, "pvs": [{"cp": 34, "line": "e7e5 g1f3"}]},
        ],
    }

    exact = convert_lichess_eval_row(row, min_depth=18, fast=False)
    fast = convert_lichess_eval_row(row, min_depth=18, fast=True)
    if exact is None or fast is None:
        raise AssertionError("expected both conversions")
    for key in ["fen", "depth", "knodes", "evaluation", "target_stm", "cp_stm", "is_mate", "terminal", "best_move"]:
        if exact[key] != fast[key]:
            raise AssertionError(f"fast conversion changed {key}: exact={exact} fast={fast}")
    if fast["bucket"].split("/")[:3] != exact["bucket"].split("/")[:3]:
        raise AssertionError(f"fast conversion changed non-tactical bucket fields: exact={exact} fast={fast}")


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


def test_convert_flat_eval_row_supports_dummy_chess_binary_fen() -> None:
    raw = [
        1, 138, 192, 118, 193, 139, 155, 188, 11, 187, 192, 188, 27, 172,
        85, 197, 92, 85, 193, 92, 21, 192, 83, 193, 85, 44, 17, 3, 66,
        129, 129, 255, 0, 1, 0,
    ]
    fen = decompress_dummy_chess_fen(raw)
    if fen != "rn1qk2r/pbpp1ppp/1p2pn2/4P3/3P4/2P2P2/P1PB2PP/R2QKBNR b KQkq - 0 1":
        raise AssertionError(f"unexpected decompressed dummy_chess FEN: {fen}")

    out = convert_flat_eval_row(
        {"fen": raw, "score": 120, "depth": 22, "knodes": 6125},
        source="hf:theoden8/nnue-chess-dataset",
        fast=True,
    )
    if out is None:
        raise AssertionError("expected converted dummy_chess row")
    if out["fen"] != "rn1qk2r/pbpp1ppp/1p2pn2/4P3/3P4/2P2P2/P1PB2PP/R2QKBNR b KQkq -":
        raise AssertionError(f"unexpected normalized dummy_chess FEN: {out}")
    if out["cp_stm"] != -120:
        raise AssertionError(f"score should be white-perspective and flip for black STM: {out}")
    if out["source"] != "hf:theoden8/nnue-chess-dataset":
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


def test_fast_phase_handles_partial_hf_fens() -> None:
    start_like = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    middlegame = "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - -"
    endgame = "8/5pk1/6p1/8/8/5PP1/6KP/8 w - -"

    if fast_phase_from_fen(start_like) != "opening":
        raise AssertionError("full-material partial FEN should classify as opening")
    if fast_phase_from_fen(middlegame) != "middlegame":
        raise AssertionError("reduced-material partial FEN should classify as middlegame")
    if fast_phase_from_fen(endgame) != "endgame":
        raise AssertionError("queenless low-material partial FEN should classify as endgame")


def test_material_sanity_rejects_impossible_promotions() -> None:
    normal = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    promoted = "qnbqkbnq/8/8/8/8/8/8/QNBQKBNQ w - -"
    impossible = "1B1B1B1B/1kB1B1B1/1B1B1B1B/P1B1B1B1/PB1B1B1B/P1B1BKB1/PB1B1B1B/B1B1B1B1 w - -"
    if not material_sanity_ok(normal):
        raise AssertionError("normal start position should pass material sanity")
    if not material_sanity_ok(promoted):
        raise AssertionError("possible promoted material should pass material sanity")
    if material_sanity_ok(impossible):
        raise AssertionError("impossible bishop wall should fail material sanity")

    row = {
        "fen": impossible,
        "depth": 55,
        "knodes": 1000,
        "cp": 6979,
        "mate": None,
    }
    if convert_flat_eval_row(row, min_depth=18) is not None:
        raise AssertionError("flat conversion should reject impossible material")


def main() -> None:
    test_white_perspective_eval_flips_to_side_to_move()
    test_terminal_labels_override_eval()
    test_best_eval_prefers_depth_then_knodes()
    test_convert_lichess_eval_row_adds_targets_and_buckets()
    test_fast_convert_preserves_label_fields()
    test_convert_flat_eval_row_supports_hf_schema()
    test_convert_flat_eval_row_supports_dummy_chess_binary_fen()
    test_bucket_detects_black_side_and_capture()
    test_normalize_fen_drops_halfmove_and_fullmove()
    test_fast_phase_handles_partial_hf_fens()
    test_material_sanity_rejects_impossible_promotions()
    assert_close(label_from_eval("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "0.00").value_stm, 0.0)
    print("test_nn_v2_data: OK")


if __name__ == "__main__":
    main()
