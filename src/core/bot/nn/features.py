from __future__ import annotations

from typing import List, Tuple

import chess


PIECE_PLANES = 10
SQUARES = 64
HALFKP_DIM = SQUARES * PIECE_PLANES * SQUARES
DUMMY_FEATURE_INDEX = HALFKP_DIM
MIRRORED_HALFKP_DIM = 32 * PIECE_PLANES * SQUARES
MIRRORED_DUMMY_FEATURE_INDEX = MIRRORED_HALFKP_DIM
HALFKA_PLANES = 11
HALFKA_DIM = SQUARES * HALFKA_PLANES * SQUARES
HALFKA_DUMMY_FEATURE_INDEX = HALFKA_DIM
MIRRORED_HALFKA_DIM = 32 * HALFKA_PLANES * SQUARES
MIRRORED_HALFKA_DUMMY_FEATURE_INDEX = MIRRORED_HALFKA_DIM
_PIECE_BASE = {
    "p": 0,
    "n": 1,
    "b": 2,
    "r": 3,
    "q": 4,
}


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


def _mirror_square(square: int) -> int:
    return square ^ 56


def _orient_square_fast(square: int, perspective_white: bool) -> int:
    return square if perspective_white else _mirror_square(square)


def _halfkp_index_fast(king_square: int,
                       piece_plane: int,
                       piece_square: int,
                       perspective_white: bool) -> int:
    oriented_king = _orient_square_fast(king_square, perspective_white)
    oriented_piece = _orient_square_fast(piece_square, perspective_white)
    return (oriented_king * PIECE_PLANES + piece_plane) * SQUARES + oriented_piece


def _parse_fen_pieces(fen: str) -> tuple[list[tuple[int, str]], int, int, bool]:
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError("FEN is missing active color")

    placement = parts[0]
    stm_white = parts[1] == "w"
    pieces: list[tuple[int, str]] = []
    white_king = -1
    black_king = -1
    rank = 7
    file = 0

    for char in placement:
        if char == "/":
            if file != 8:
                raise ValueError("invalid FEN rank width")
            rank -= 1
            file = 0
            continue
        if char.isdigit():
            file += int(char)
            if file > 8:
                raise ValueError("invalid FEN rank width")
            continue
        lower = char.lower()
        if lower not in "kpnbrq" or file >= 8 or rank < 0:
            raise ValueError("invalid FEN piece placement")
        square = rank * 8 + file
        if lower == "k":
            if char.isupper():
                white_king = square
            else:
                black_king = square
        else:
            pieces.append((square, char))
        file += 1

    if rank != 0 or file != 8:
        raise ValueError("invalid FEN board")
    if white_king < 0 or black_king < 0:
        raise ValueError("board is missing a king")
    return pieces, white_king, black_king, stm_white


def _encode_halfkp_fast(pieces: list[tuple[int, str]],
                        king_square: int,
                        perspective_white: bool) -> List[int]:
    features: List[int] = []
    for square, symbol in pieces:
        piece_white = symbol.isupper()
        base = _PIECE_BASE[symbol.lower()]
        piece_plane = base if piece_white == perspective_white else base + 5
        features.append(_halfkp_index_fast(king_square, piece_plane, square, perspective_white))
    if not features:
        features.append(DUMMY_FEATURE_INDEX)
    return features


def encode_fen_slow(fen: str) -> Tuple[List[int], List[int], bool]:
    board = board_from_fen(fen)
    white_half = encode_halfkp(board, chess.WHITE)
    black_half = encode_halfkp(board, chess.BLACK)
    stm_white = board.turn == chess.WHITE
    return white_half, black_half, stm_white


def encode_fen(fen: str) -> Tuple[List[int], List[int], bool]:
    pieces, white_king, black_king, stm_white = _parse_fen_pieces(fen)
    white_half = _encode_halfkp_fast(pieces, white_king, True)
    black_half = _encode_halfkp_fast(pieces, black_king, False)
    return white_half, black_half, stm_white


def _halfka_index_fast(king_square: int,
                       piece_plane: int,
                       piece_square: int,
                       perspective_white: bool) -> int:
    oriented_king = _orient_square_fast(king_square, perspective_white)
    oriented_piece = _orient_square_fast(piece_square, perspective_white)
    return (oriented_king * HALFKA_PLANES + piece_plane) * SQUARES + oriented_piece


def _encode_halfka_fast(pieces: list[tuple[int, str]],
                        own_king: int,
                        enemy_king: int,
                        perspective_white: bool) -> List[int]:
    features: List[int] = []
    for square, symbol in pieces:
        piece_white = symbol.isupper()
        base = _PIECE_BASE[symbol.lower()]
        plane = base if piece_white == perspective_white else base + 5
        features.append(_halfka_index_fast(own_king, plane, square, perspective_white))
    # HalfKAv2 compresses both kings into one plane. Their squares distinguish
    # the always-present own king from the enemy king without another plane.
    features.append(_halfka_index_fast(own_king, 10, own_king, perspective_white))
    features.append(_halfka_index_fast(own_king, 10, enemy_king, perspective_white))
    return features


def encode_fen_halfka(fen: str) -> Tuple[List[int], List[int], bool]:
    pieces, white_king, black_king, stm_white = _parse_fen_pieces(fen)
    white_half = _encode_halfka_fast(pieces, white_king, black_king, True)
    black_half = _encode_halfka_fast(pieces, black_king, white_king, False)
    return white_half, black_half, stm_white
