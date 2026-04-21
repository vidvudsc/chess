from __future__ import annotations

from typing import List, Tuple

import chess


PIECE_PLANES = 10
SQUARES = 64
HALFKP_DIM = SQUARES * PIECE_PLANES * SQUARES
DUMMY_FEATURE_INDEX = HALFKP_DIM


def ensure_full_fen(fen: str) -> str:
    if fen.count(" ") == 3:
        return f"{fen} 0 1"
    return fen


def board_from_fen(fen: str) -> chess.Board:
    return chess.Board(ensure_full_fen(fen))


def orient_square(square: chess.Square, perspective: chess.Color) -> chess.Square:
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def relative_piece_plane(piece: chess.Piece, perspective: chess.Color) -> int:
    own = piece.color == perspective
    base = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }[piece.piece_type]
    return base if own else base + 5


def halfkp_index(king_square: chess.Square,
                 piece_plane: int,
                 piece_square: chess.Square,
                 perspective: chess.Color) -> int:
    oriented_king = orient_square(king_square, perspective)
    oriented_piece = orient_square(piece_square, perspective)
    return (oriented_king * PIECE_PLANES + piece_plane) * SQUARES + oriented_piece


def encode_halfkp(board: chess.Board, perspective: chess.Color) -> List[int]:
    king_square = board.king(perspective)
    if king_square is None:
        raise ValueError("board is missing a king")

    features: List[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        piece_plane = relative_piece_plane(piece, perspective)
        features.append(halfkp_index(king_square, piece_plane, square, perspective))
    if not features:
        features.append(DUMMY_FEATURE_INDEX)
    return features


def encode_fen(fen: str) -> Tuple[List[int], List[int], bool]:
    board = board_from_fen(fen)
    white_half = encode_halfkp(board, chess.WHITE)
    black_half = encode_halfkp(board, chess.BLACK)
    stm_white = board.turn == chess.WHITE
    return white_half, black_half, stm_white
