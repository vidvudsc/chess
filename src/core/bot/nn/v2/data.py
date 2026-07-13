from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import chess


CP_CLIP = 1200
CP_SCALE = 600.0
ScorePerspective = Literal["white", "stm"]
TerminalKind = Literal["none", "checkmate", "stalemate", "insufficient_material", "seventyfive_moves"]
_DUMMY_CHESS_PIECES = "KQRBNPkqrbnp"
_DUMMY_CHESS_EMPTY = 0xC
_DUMMY_CHESS_NOPOS = 0xFF


@dataclass(frozen=True)
class ValueLabel:
    value_stm: float
    cp_stm: int
    is_mate: bool
    terminal: TerminalKind


@dataclass(frozen=True)
class PositionBuckets:
    side_to_move: Literal["white", "black"]
    phase: Literal["opening", "middlegame", "endgame"]
    eval_band: Literal["losing", "worse", "equal", "better", "winning"]
    tactical: Literal["quiet", "capture", "check", "promotion", "mate"]

    def key(self) -> str:
        return f"{self.side_to_move}/{self.phase}/{self.eval_band}/{self.tactical}"


def ensure_full_fen(fen: str) -> str:
    parts = fen.strip().split()
    if len(parts) == 4:
        return f"{fen.strip()} 0 1"
    return fen.strip()


def board_from_fen(fen: str) -> chess.Board:
    return chess.Board(ensure_full_fen(fen))


def decompress_dummy_chess_fen(data: bytes | bytearray | memoryview | list[int]) -> str:
    raw = bytes(data)
    if len(raw) < 8:
        raise ValueError("compressed dummy_chess FEN is too short")

    flags = raw[0]
    is_black = bool(flags & 1)
    is_chess960 = bool(flags & 2)
    is_crazyhouse = bool(flags & 4)
    meta_size = 6 + (13 if is_crazyhouse else 0)
    if len(raw) < 1 + meta_size:
        raise ValueError("compressed dummy_chess FEN is missing metadata")
    board_end = len(raw) - meta_size

    nibs: list[int] = []
    for byte in raw[1:board_end]:
        nibs.append((byte >> 4) & 0xF)
        nibs.append(byte & 0xF)

    board: list[str] = []
    i = 0
    while i < len(nibs) and len(board) < 64:
        nib = nibs[i]
        if nib == _DUMMY_CHESS_EMPTY and i + 1 < len(nibs):
            board.extend(" " for _ in range(nibs[i + 1] + 1))
            i += 2
        elif nib < len(_DUMMY_CHESS_PIECES):
            board.append(_DUMMY_CHESS_PIECES[nib])
            i += 1
        elif nib == 0xF:
            break
        else:
            i += 1
    if len(board) != 64:
        raise ValueError(f"compressed dummy_chess FEN decoded {len(board)} squares")

    ranks: list[str] = []
    for rank_start in range(0, 64, 8):
        empty = 0
        out = []
        for piece in board[rank_start:rank_start + 8]:
            if piece == " ":
                empty += 1
            else:
                if empty:
                    out.append(str(empty))
                    empty = 0
                out.append(piece)
        if empty:
            out.append(str(empty))
        ranks.append("".join(out))

    m = board_end
    white_castle_mask = raw[m]
    black_castle_mask = raw[m + 1]
    ep_square = raw[m + 2]
    halfmove = raw[m + 3]
    fullmove = raw[m + 4] | (raw[m + 5] << 8)

    castling = ""
    for mask, base in ((white_castle_mask, ord("A")), (black_castle_mask, ord("a"))):
        for file_index in range(7, -1, -1):
            if mask & (1 << file_index):
                ch = chr(base + file_index)
                if not is_chess960:
                    ch = {"H": "K", "A": "Q", "h": "k", "a": "q"}.get(ch, ch)
                castling += ch
    if not castling:
        castling = "-"

    if ep_square == _DUMMY_CHESS_NOPOS:
        ep = "-"
    else:
        ep = f"{chr(ord('a') + (ep_square % 8))}{ep_square // 8 + 1}"

    if fullmove <= 0:
        fullmove = 1
    return f"{'/'.join(ranks)} {'b' if is_black else 'w'} {castling} {ep} {halfmove} {fullmove}"


def coerce_fen(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bytes, bytearray, memoryview, list)):
        return decompress_dummy_chess_fen(value)
    return str(value).strip()


def normalize_fen_key(fen: str) -> str:
    board = board_from_fen(fen)
    parts = board.fen(en_passant="fen").split()
    return " ".join(parts[:4])


def normalize_fen_key_fast(fen: str) -> str:
    parts = fen.strip().split()
    if len(parts) < 4:
        return normalize_fen_key(fen)
    return " ".join(parts[:4])


def material_sanity_ok(fen: str) -> bool:
    parts = fen.strip().split()
    if not parts:
        return False
    counts: dict[str, int] = {}
    for char in parts[0]:
        if char.isalpha():
            counts[char] = counts.get(char, 0) + 1

    white = sum(counts.get(piece, 0) for piece in "PNBRQK")
    black = sum(counts.get(piece, 0) for piece in "pnbrqk")
    total = white + black
    pawns = counts.get("P", 0) + counts.get("p", 0)
    kings = counts.get("K", 0) + counts.get("k", 0)
    queens = counts.get("Q", 0) + counts.get("q", 0)
    rooks = counts.get("R", 0) + counts.get("r", 0)
    bishops = counts.get("B", 0) + counts.get("b", 0)
    knights = counts.get("N", 0) + counts.get("n", 0)

    if kings != 2 or total > 32 or pawns > 16:
        return False
    promoted_like = (
        max(0, queens - 2) +
        max(0, rooks - 4) +
        max(0, bishops - 4) +
        max(0, knights - 4)
    )
    missing_pawns = max(0, 16 - pawns)
    if promoted_like > missing_pawns:
        return False
    return queens <= 6 and rooks <= 8 and bishops <= 10 and knights <= 10


def side_to_move_from_fen(fen: str) -> Literal["white", "black"]:
    parts = fen.strip().split()
    return "black" if len(parts) > 1 and parts[1] == "b" else "white"


def fullmove_from_fen(fen: str) -> int:
    parts = fen.strip().split()
    if len(parts) >= 6:
        try:
            return int(parts[5])
        except ValueError:
            return 1
    return 1


def fast_phase_from_fen(fen: str) -> Literal["opening", "middlegame", "endgame"]:
    parts = fen.strip().split()
    has_fullmove = len(parts) >= 6
    if has_fullmove and fullmove_from_fen(fen) <= 10:
        return "opening"
    placement = parts[0] if parts else ""
    queens = placement.count("q") + placement.count("Q")
    rooks = placement.count("r") + placement.count("R")
    minors = placement.count("b") + placement.count("B") + placement.count("n") + placement.count("N")
    pawns = placement.count("p") + placement.count("P")
    if queens == 0 and rooks <= 2 and minors <= 4:
        return "endgame"
    ranks = placement.split("/")
    home_back_rank_pieces = 0
    if len(ranks) == 8:
        home_back_rank_pieces = sum(1 for char in ranks[0] if char in "rnbqk") + \
            sum(1 for char in ranks[7] if char in "RNBQK")
    if not has_fullmove and queens >= 2 and rooks >= 4 and minors >= 7 and pawns >= 14 and home_back_rank_pieces >= 12:
        return "opening"
    return "middlegame"


def value_from_cp(cp: int, cp_clip: int = CP_CLIP, cp_scale: float = CP_SCALE) -> float:
    clipped = max(-cp_clip, min(cp_clip, int(cp)))
    return math.tanh(clipped / cp_scale)


def parse_eval_to_cp(raw: str, cp_clip: int = CP_CLIP) -> tuple[bool, int]:
    text = raw.strip()
    if not text:
        raise ValueError("empty eval")
    if text.startswith("-M") or text.startswith("M-"):
        return True, -cp_clip
    if text.startswith("M"):
        return True, cp_clip
    pawns = float(text)
    cp = int(round(pawns * 100.0))
    return False, max(-cp_clip, min(cp_clip, cp))


def terminal_label(board: chess.Board, cp_clip: int = CP_CLIP) -> ValueLabel | None:
    if board.is_checkmate():
        return ValueLabel(value_stm=-1.0, cp_stm=-cp_clip, is_mate=True, terminal="checkmate")
    if board.is_stalemate():
        return ValueLabel(value_stm=0.0, cp_stm=0, is_mate=False, terminal="stalemate")
    if board.is_insufficient_material():
        return ValueLabel(value_stm=0.0, cp_stm=0, is_mate=False, terminal="insufficient_material")
    if board.is_seventyfive_moves():
        return ValueLabel(value_stm=0.0, cp_stm=0, is_mate=False, terminal="seventyfive_moves")
    return None


def label_from_eval(fen: str,
                    evaluation: str,
                    perspective: ScorePerspective = "white",
                    cp_clip: int = CP_CLIP,
                    cp_scale: float = CP_SCALE) -> ValueLabel:
    board = board_from_fen(fen)
    terminal = terminal_label(board, cp_clip=cp_clip)
    if terminal is not None:
        return terminal

    is_mate, cp = parse_eval_to_cp(evaluation, cp_clip=cp_clip)
    if perspective == "white":
        cp_stm = cp if board.turn == chess.WHITE else -cp
    elif perspective == "stm":
        cp_stm = cp
    else:
        raise ValueError(f"unsupported score perspective: {perspective}")
    return ValueLabel(
        value_stm=value_from_cp(cp_stm, cp_clip=cp_clip, cp_scale=cp_scale),
        cp_stm=cp_stm,
        is_mate=is_mate,
        terminal="none",
    )


def material_phase(board: chess.Board) -> Literal["opening", "middlegame", "endgame"]:
    if board.fullmove_number <= 10:
        return "opening"
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
    minors = (
        len(board.pieces(chess.BISHOP, chess.WHITE)) +
        len(board.pieces(chess.BISHOP, chess.BLACK)) +
        len(board.pieces(chess.KNIGHT, chess.WHITE)) +
        len(board.pieces(chess.KNIGHT, chess.BLACK))
    )
    if queens == 0 and rooks <= 2 and minors <= 4:
        return "endgame"
    return "middlegame"


def eval_band(cp_stm: int) -> Literal["losing", "worse", "equal", "better", "winning"]:
    if cp_stm <= -500:
        return "losing"
    if cp_stm <= -120:
        return "worse"
    if cp_stm < 120:
        return "equal"
    if cp_stm < 500:
        return "better"
    return "winning"


def move_tactical_bucket(board: chess.Board, uci: str, is_mate: bool) -> Literal["quiet", "capture", "check", "promotion", "mate"]:
    if is_mate:
        return "mate"
    if not uci:
        return "quiet"
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return "quiet"
    if move not in board.legal_moves:
        return "quiet"
    if move.promotion is not None:
        return "promotion"
    if board.is_capture(move):
        return "capture"
    after = board.copy(stack=False)
    after.push(move)
    if after.is_check():
        return "check"
    return "quiet"


def buckets_for_position(fen: str, label: ValueLabel, best_move_uci: str = "") -> PositionBuckets:
    board = board_from_fen(fen)
    return PositionBuckets(
        side_to_move="white" if board.turn == chess.WHITE else "black",
        phase=material_phase(board),
        eval_band=eval_band(label.cp_stm),
        tactical=move_tactical_bucket(board, best_move_uci, label.is_mate),
    )


def fast_tactical_bucket(best_move_uci: str, is_mate: bool) -> Literal["quiet", "capture", "check", "promotion", "mate"]:
    if is_mate:
        return "mate"
    if len(best_move_uci) >= 5:
        return "promotion"
    return "quiet"


def fast_buckets_for_position(fen: str, label: ValueLabel, best_move_uci: str = "") -> PositionBuckets:
    return PositionBuckets(
        side_to_move=side_to_move_from_fen(fen),
        phase=fast_phase_from_fen(fen),
        eval_band=eval_band(label.cp_stm),
        tactical=fast_tactical_bucket(best_move_uci, label.is_mate),
    )


def label_from_eval_fast(fen: str,
                         evaluation: str,
                         perspective: ScorePerspective = "white",
                         cp_clip: int = CP_CLIP,
                         cp_scale: float = CP_SCALE) -> ValueLabel:
    is_mate, cp = parse_eval_to_cp(evaluation, cp_clip=cp_clip)
    if perspective == "white":
        cp_stm = cp if side_to_move_from_fen(fen) == "white" else -cp
    elif perspective == "stm":
        cp_stm = cp
    else:
        raise ValueError(f"unsupported score perspective: {perspective}")
    return ValueLabel(
        value_stm=value_from_cp(cp_stm, cp_clip=cp_clip, cp_scale=cp_scale),
        cp_stm=cp_stm,
        is_mate=is_mate,
        terminal="none",
    )


def best_eval_item(evals: Iterable[dict[str, Any]],
                   min_depth: int,
                   min_knodes: int = 0) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_key = (-1, -1)
    for item in evals:
        try:
            depth = int(item.get("depth", 0))
            knodes = int(item.get("knodes", 0))
        except (TypeError, ValueError):
            continue
        if depth < min_depth or knodes < min_knodes:
            continue
        pvs = item.get("pvs", [])
        if not isinstance(pvs, list) or not pvs:
            continue
        pv = pvs[0]
        if not isinstance(pv, dict) or ("cp" not in pv and "mate" not in pv):
            continue
        key = (depth, knodes)
        if key >= best_key:
            best = {"depth": depth, "knodes": knodes, "pv": pv}
            best_key = key
    return best


def eval_text_from_pv(pv: dict[str, Any]) -> str:
    if "mate" in pv:
        mate = int(pv["mate"])
        return f"M{mate}" if mate >= 0 else f"-M{abs(mate)}"
    cp = int(pv["cp"])
    return f"{cp / 100.0:.2f}"


def eval_text_from_flat_fields(row: dict[str, Any]) -> str | None:
    mate = row.get("mate")
    if mate is not None:
        mate_int = int(mate)
        return f"M{mate_int}" if mate_int >= 0 else f"-M{abs(mate_int)}"

    if str(row.get("eval_type", "")).lower() == "mate":
        value = int(row["eval_value"])
        return f"M{value}" if value >= 0 else f"-M{abs(value)}"

    cp = row.get("cp")
    if cp is None:
        cp = row.get("score")
    if cp is None and str(row.get("eval_type", "")).lower() == "cp":
        cp = row.get("eval_value")
    if cp is None:
        return None
    return f"{int(cp) / 100.0:.2f}"


def convert_lichess_eval_row(row: dict[str, Any],
                             min_depth: int = 18,
                             min_knodes: int = 0,
                             score_perspective: ScorePerspective = "white",
                             fast: bool = False,
                             cp_scale: float = CP_SCALE) -> dict[str, Any] | None:
    try:
        fen = coerce_fen(row.get("fen", ""))
    except (TypeError, ValueError):
        return None
    if not fen:
        return None
    if not material_sanity_ok(fen):
        return None
    selected = best_eval_item(row.get("evals", []), min_depth=min_depth, min_knodes=min_knodes)
    if selected is None:
        return None
    pv = selected["pv"]
    evaluation = eval_text_from_pv(pv)
    line = str(pv.get("line", "")).strip()
    best_move = line.split()[0] if line else ""
    if fast:
        label = label_from_eval_fast(fen, evaluation, perspective=score_perspective, cp_scale=cp_scale)
        buckets = fast_buckets_for_position(fen, label, best_move)
        normalized_fen = normalize_fen_key_fast(fen)
    else:
        label = label_from_eval(
            fen,
            evaluation,
            perspective=score_perspective,
            cp_scale=cp_scale,
        )
        buckets = buckets_for_position(fen, label, best_move)
        normalized_fen = normalize_fen_key(fen)
    return {
        "fen": normalized_fen,
        "depth": selected["depth"],
        "knodes": selected["knodes"],
        "evaluation": evaluation,
        "score_perspective": score_perspective,
        "target_stm": label.value_stm,
        "cp_stm": label.cp_stm,
        "is_mate": label.is_mate,
        "terminal": label.terminal,
        "best_move": best_move,
        "best_line": line,
        "bucket": buckets.key(),
        "source": "lichess-evals",
    }


def convert_flat_eval_row(row: dict[str, Any],
                          min_depth: int = 18,
                          min_knodes: int = 0,
                          score_perspective: ScorePerspective = "white",
                          source: str = "flat-evals",
                          fast: bool = False,
                          cp_scale: float = CP_SCALE) -> dict[str, Any] | None:
    try:
        fen = coerce_fen(row.get("fen", ""))
    except (TypeError, ValueError):
        return None
    if not fen:
        return None
    if not material_sanity_ok(fen):
        return None
    try:
        depth = int(row.get("depth", 0))
        knodes = int(row.get("knodes", 0) or 0)
    except (TypeError, ValueError):
        return None
    if depth < min_depth or knodes < min_knodes:
        return None

    evaluation = eval_text_from_flat_fields(row)
    if evaluation is None:
        return None
    line = str(row.get("line", "") or row.get("best_line", "")).strip()
    best_move = str(row.get("best_move", "") or row.get("move", "")).strip()
    if not best_move and line:
        best_move = line.split()[0]
    if best_move and not line:
        line = best_move

    if fast:
        label = label_from_eval_fast(fen, evaluation, perspective=score_perspective, cp_scale=cp_scale)
        buckets = fast_buckets_for_position(fen, label, best_move)
        normalized_fen = normalize_fen_key_fast(fen)
    else:
        label = label_from_eval(
            fen,
            evaluation,
            perspective=score_perspective,
            cp_scale=cp_scale,
        )
        buckets = buckets_for_position(fen, label, best_move)
        normalized_fen = normalize_fen_key(fen)
    return {
        "fen": normalized_fen,
        "depth": depth,
        "knodes": knodes,
        "evaluation": evaluation,
        "score_perspective": score_perspective,
        "target_stm": label.value_stm,
        "cp_stm": label.cp_stm,
        "is_mate": label.is_mate,
        "terminal": label.terminal,
        "best_move": best_move,
        "best_line": line,
        "bucket": buckets.key(),
        "source": source,
    }


def terminal_fixture_rows() -> list[dict[str, Any]]:
    fens = [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/1k6/1q6/K7 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/1k6/2q5/K7 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    ]
    rows: list[dict[str, Any]] = []
    for fen in fens:
        label = label_from_eval(fen, "0.00", perspective="white")
        buckets = buckets_for_position(fen, label, "")
        rows.append({
            "fen": normalize_fen_key(fen),
            "depth": 99,
            "knodes": 0,
            "evaluation": "0.00",
            "score_perspective": "white",
            "target_stm": label.value_stm,
            "cp_stm": label.cp_stm,
            "is_mate": label.is_mate,
            "terminal": label.terminal,
            "best_move": "",
            "best_line": "",
            "bucket": buckets.key(),
            "source": "terminal-fixture",
        })
    return rows
