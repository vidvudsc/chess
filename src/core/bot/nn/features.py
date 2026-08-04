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
FULL_THREATS_DIM = 59808
HALFKA_THREATS_DIM = HALFKA_DIM + FULL_THREATS_DIM
HALFKA_THREATS_DUMMY_FEATURE_INDEX = HALFKA_THREATS_DIM
MIRRORED_HALFKA_THREATS_DIM = MIRRORED_HALFKA_DIM + FULL_THREATS_DIM
MIRRORED_HALFKA_THREATS_DUMMY_FEATURE_INDEX = MIRRORED_HALFKA_THREATS_DIM
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


# Stockfish Full_Threats v2 compression. Piece types use the conventional
# pawn, knight, bishop, rook, queen, king order and colors are relative to the
# accumulator perspective (us first, them second).
_THREAT_TARGET_MAP = (
    (-1, 0, -1, 1, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, -1, -1),
    (0, 1, 2, 3, 4, -1),
    (-1, -1, -1, -1, -1, -1),
)
_THREAT_VALID_TARGETS = (4, 10, 8, 8, 10, 0) * 2
_THREAT_TYPE_BY_SYMBOL = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def _threat_pseudo_targets(piece_type: int, square: int, white: bool) -> list[int]:
    rank, file = divmod(square, 8)
    targets: list[int] = []
    if piece_type == 0:
        if rank < 1 or rank > 6:
            return targets
        dr = 1 if white else -1
        for df in (-1, 1):
            nr, nf = rank + dr, file + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                targets.append(nr * 8 + nf)
    elif piece_type == 1:
        for dr, df in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                       (1, 2), (1, -2), (-1, 2), (-1, -2)):
            nr, nf = rank + dr, file + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                targets.append(nr * 8 + nf)
    else:
        directions: tuple[tuple[int, int], ...]
        if piece_type == 2:
            directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        elif piece_type == 3:
            directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        elif piece_type == 4:
            directions = ((1, 1), (1, -1), (-1, 1), (-1, -1),
                          (1, 0), (-1, 0), (0, 1), (0, -1))
        else:
            directions = ((1, 1), (1, -1), (-1, 1), (-1, -1),
                          (1, 0), (-1, 0), (0, 1), (0, -1))
        distance = 1 if piece_type == 5 else 7
        for dr, df in directions:
            for step in range(1, distance + 1):
                nr, nf = rank + dr * step, file + df * step
                if not (0 <= nr < 8 and 0 <= nf < 8):
                    break
                targets.append(nr * 8 + nf)
    return sorted(targets)


def _build_threat_tables() -> tuple[list[list[int]], list[int], list[int], list[int]]:
    rank_by_piece_square: list[list[int]] = [[0] * 4096 for _ in range(12)]
    from_offsets: list[int] = [0] * (12 * 64)
    geometry_sizes: list[int] = [0] * 12
    cumulative_offsets: list[int] = [0] * 12
    cumulative = 0
    for piece in range(12):
        piece_type = piece % 6
        white = piece < 6
        piece_total = 0
        for source in range(64):
            from_offsets[piece * 64 + source] = piece_total
            targets = _threat_pseudo_targets(piece_type, source, white)
            for rank, target in enumerate(targets):
                rank_by_piece_square[piece][source * 64 + target] = rank
            piece_total += len(targets)
        geometry_sizes[piece] = piece_total
        cumulative_offsets[piece] = cumulative
        cumulative += _THREAT_VALID_TARGETS[piece] * piece_total
    if cumulative != FULL_THREATS_DIM:
        raise AssertionError(f"Full_Threats table has {cumulative} rows, expected {FULL_THREATS_DIM}")
    return rank_by_piece_square, from_offsets, cumulative_offsets, geometry_sizes


(_THREAT_RANK, _THREAT_FROM_OFFSET,
 _THREAT_PIECE_OFFSET, _THREAT_GEOMETRY_SIZE) = _build_threat_tables()


def _full_threat_index(perspective_white: bool,
                       king_square: int,
                       attacker_symbol: str,
                       source: int,
                       target: int,
                       attacked_symbol: str) -> int | None:
    attacker_type = _THREAT_TYPE_BY_SYMBOL[attacker_symbol.lower()]
    attacked_type = _THREAT_TYPE_BY_SYMBOL[attacked_symbol.lower()]
    target_map = _THREAT_TARGET_MAP[attacker_type][attacked_type]
    if target_map < 0:
        return None
    orientation = (0 if (king_square & 7) < 4 else 7) ^ (0 if perspective_white else 56)
    source_oriented = source ^ orientation
    target_oriented = target ^ orientation
    perspective_color = attacker_symbol.isupper() == perspective_white
    attacked_color = attacked_symbol.isupper() == perspective_white
    attacker = attacker_type + (0 if perspective_color else 6)
    enemy = perspective_color != attacked_color
    if (source_oriented < target_oriented and attacker_type == attacked_type
            and (enemy or attacker_type != 0)):
        return None
    color_slot = 0 if attacked_color else 1
    geometry = _THREAT_GEOMETRY_SIZE[attacker]
    index = (
        _THREAT_PIECE_OFFSET[attacker]
        + (color_slot * (_THREAT_VALID_TARGETS[attacker] // 2) + target_map) * geometry
        + _THREAT_FROM_OFFSET[attacker * 64 + source_oriented]
        + _THREAT_RANK[attacker][source_oriented * 64 + target_oriented]
    )
    return index if 0 <= index < FULL_THREATS_DIM else None


def _encode_full_threats(pieces: list[tuple[int, str]],
                         white_king: int,
                         black_king: int,
                         perspective_white: bool) -> list[int]:
    board = {square: symbol for square, symbol in pieces}
    board[white_king] = "K"
    board[black_king] = "k"
    occupied = set(board)
    king_square = white_king if perspective_white else black_king
    active: list[int] = []
    for source, attacker in board.items():
        attacker_type = _THREAT_TYPE_BY_SYMBOL[attacker.lower()]
        if attacker_type == 5:
            continue
        pseudo = _threat_pseudo_targets(attacker_type, source, attacker.isupper())
        for target in pseudo:
            if attacker_type in (2, 3, 4):
                sr, sf = divmod(source, 8)
                tr, tf = divmod(target, 8)
                dr = 0 if tr == sr else (1 if tr > sr else -1)
                df = 0 if tf == sf else (1 if tf > sf else -1)
                square = source + dr * 8 + df
                blocked = False
                while square != target:
                    if square in occupied:
                        blocked = True
                        break
                    square += dr * 8 + df
                if blocked:
                    continue
            attacked = board.get(target)
            if attacked is None:
                continue
            index = _full_threat_index(
                perspective_white, king_square, attacker, source, target, attacked
            )
            if index is not None:
                active.append(index)
    return sorted(active)


def encode_fen_halfka_threats(fen: str) -> Tuple[List[int], List[int], bool]:
    pieces, white_king, black_king, stm_white = _parse_fen_pieces(fen)
    white = _encode_halfka_fast(pieces, white_king, black_king, True)
    black = _encode_halfka_fast(pieces, black_king, white_king, False)
    white.extend(HALFKA_DIM + index for index in _encode_full_threats(
        pieces, white_king, black_king, True
    ))
    black.extend(HALFKA_DIM + index for index in _encode_full_threats(
        pieces, white_king, black_king, False
    ))
    return white, black, stm_white
