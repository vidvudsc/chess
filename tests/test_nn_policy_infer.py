from __future__ import annotations

import sys
from pathlib import Path

import chess


ROOT = Path(__file__).resolve().parents[1]
NN_ROOT = ROOT / "src" / "core" / "bot" / "nn"
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from policy_infer import encode_board, move_to_index  # noqa: E402


def test_encode_board_startpos_planes() -> None:
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (15, 8, 8)
    assert planes[:12].sum() == 32.0
    assert planes[12].sum() == 64.0
    assert planes[13].sum() == 0.0
    assert planes[14].sum() == 0.0


def test_encode_board_last_move_planes() -> None:
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    planes = encode_board(board)
    assert planes[12].sum() == 0.0
    assert planes[13, chess.E2 // 8, chess.E2 % 8] == 1.0
    assert planes[14, chess.E4 // 8, chess.E4 % 8] == 1.0


def test_move_to_index_uses_from_square_times_64_plus_to_square() -> None:
    move = chess.Move.from_uci("e2e4")
    assert move_to_index(move) == chess.E2 * 64 + chess.E4
