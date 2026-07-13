#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import chess
except ImportError as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit("python-chess is required for policy inference") from exc

NN_ROOT = Path(__file__).resolve().parent
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from policy_model import ConvPolicyValueNet, MOVE_INDEX_COUNT  # noqa: E402


PIECE_TO_PLANE = {
    chess.Piece(chess.PAWN, chess.WHITE): 0,
    chess.Piece(chess.KNIGHT, chess.WHITE): 1,
    chess.Piece(chess.BISHOP, chess.WHITE): 2,
    chess.Piece(chess.ROOK, chess.WHITE): 3,
    chess.Piece(chess.QUEEN, chess.WHITE): 4,
    chess.Piece(chess.KING, chess.WHITE): 5,
    chess.Piece(chess.PAWN, chess.BLACK): 6,
    chess.Piece(chess.KNIGHT, chess.BLACK): 7,
    chess.Piece(chess.BISHOP, chess.BLACK): 8,
    chess.Piece(chess.ROOK, chess.BLACK): 9,
    chess.Piece(chess.QUEEN, chess.BLACK): 10,
    chess.Piece(chess.KING, chess.BLACK): 11,
}


def square_to_row_col(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def move_to_index(move: chess.Move) -> int:
    return int(move.from_square) * 64 + int(move.to_square)


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((15, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[piece]
        row, col = square_to_row_col(square)
        planes[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    if board.move_stack:
        last = board.peek()
        from_row, from_col = square_to_row_col(last.from_square)
        to_row, to_col = square_to_row_col(last.to_square)
        planes[13, from_row, from_col] = 1.0
        planes[14, to_row, to_col] = 1.0
    return planes


def load_model(checkpoint_path: Path, device: torch.device) -> ConvPolicyValueNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    model = ConvPolicyValueNet(
        channels=int(args.get("channels", 64)),
        blocks=int(args.get("blocks", 3)),
        value_hidden=int(args.get("value_hidden", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def rank_legal_moves(model: ConvPolicyValueNet,
                     board: chess.Board,
                     device: torch.device,
                     limit: int = 10) -> tuple[list[tuple[chess.Move, float]], float]:
    features = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(features)
        probs = torch.softmax(logits[0], dim=0).detach().cpu()
        value_f = float(value.item())

    ranked: list[tuple[chess.Move, float]] = []
    for move in board.legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < MOVE_INDEX_COUNT:
            ranked.append((move, float(probs[idx].item())))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit], value_f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank legal chess moves with a trained policy/value checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--moves", nargs="*", default=[],
                        help="Optional UCI move history to apply after the FEN.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    board = chess.Board(args.fen)
    for raw_move in args.moves:
        board.push(chess.Move.from_uci(raw_move))
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    ranked, value = rank_legal_moves(model, board, device, limit=args.top)
    print(f"value={value:.4f}")
    for move, prob in ranked:
        print(f"{move.uci()} {prob:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
