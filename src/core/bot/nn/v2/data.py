from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import chess


CP_CLIP = 1200
CP_SCALE = 600.0
ScorePerspective = Literal["white", "stm"]
TerminalKind = Literal["none", "checkmate", "stalemate", "insufficient_material", "seventyfive_moves"]


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


def normalize_fen_key(fen: str) -> str:
    board = board_from_fen(fen)
    parts = board.fen(en_passant="fen").split()
    return " ".join(parts[:4])


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
    if cp is None and str(row.get("eval_type", "")).lower() == "cp":
        cp = row.get("eval_value")
    if cp is None:
        return None
    return f"{int(cp) / 100.0:.2f}"


def convert_lichess_eval_row(row: dict[str, Any],
                             min_depth: int = 18,
                             min_knodes: int = 0,
                             score_perspective: ScorePerspective = "white") -> dict[str, Any] | None:
    fen = str(row.get("fen", "")).strip()
    if not fen:
        return None
    selected = best_eval_item(row.get("evals", []), min_depth=min_depth, min_knodes=min_knodes)
    if selected is None:
        return None
    pv = selected["pv"]
    evaluation = eval_text_from_pv(pv)
    line = str(pv.get("line", "")).strip()
    best_move = line.split()[0] if line else ""
    label = label_from_eval(
        fen,
        evaluation,
        perspective=score_perspective,
    )
    buckets = buckets_for_position(fen, label, best_move)
    return {
        "fen": normalize_fen_key(fen),
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
                          source: str = "flat-evals") -> dict[str, Any] | None:
    fen = str(row.get("fen", "")).strip()
    if not fen:
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

    label = label_from_eval(
        fen,
        evaluation,
        perspective=score_perspective,
    )
    buckets = buckets_for_position(fen, label, best_move)
    return {
        "fen": normalize_fen_key(fen),
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
