#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine


ROOT = Path(__file__).resolve().parents[1]
NN_ROOT = ROOT / "src" / "core" / "bot" / "nn"
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from v2.data import buckets_for_position, label_from_eval, normalize_fen_key  # noqa: E402


DEFAULT_REPORT_GLOB = "current/nnue_search_tuning/*.json"
DEFAULT_OUT_DIR = ROOT / "current" / "nnue_blunders"


@dataclass
class CandidateEval:
    move: str
    child_fen: str
    cp_white: int
    cp_stm_after: int
    target_stm_after: float
    rank: int
    is_played: bool
    is_best: bool
    pv: list[str]


@dataclass
class BlunderRow:
    report: str
    game_index: int
    ply: int
    nn_color: str
    start_fen: str
    fen: str
    played_move: str
    sf_best_move: str
    loss_cp: int
    best_cp_white: int
    played_cp_white: int
    side_to_move: str
    result: str
    termination: str
    candidates: list[CandidateEval]


def cp_from_score(score: chess.engine.PovScore, board: chess.Board, mate_score: int) -> int:
    pov = score.pov(chess.WHITE)
    cp = pov.score(mate_score=mate_score)
    if cp is None:
        return 0
    return max(-mate_score, min(mate_score, int(cp)))


def eval_child(engine: chess.engine.SimpleEngine,
               board: chess.Board,
               move: chess.Move,
               depth: int,
               mate_score: int) -> tuple[int, list[str]]:
    child = board.copy(stack=False)
    child.push(move)
    info = engine.analyse(child, chess.engine.Limit(depth=depth))
    score = info.get("score")
    pv = info.get("pv", [])
    cp_white = cp_from_score(score, child, mate_score) if isinstance(score, chess.engine.PovScore) else 0
    return cp_white, [m.uci() for m in pv]


def analyse_position(engine: chess.engine.SimpleEngine,
                     board: chess.Board,
                     played: chess.Move,
                     depth: int,
                     multipv: int,
                     mate_score: int,
                     cp_scale: float) -> tuple[int, int, str, list[CandidateEval]]:
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    if isinstance(infos, dict):
        infos = [infos]

    candidate_moves: list[chess.Move] = []
    pvs_by_move: dict[str, list[str]] = {}
    cp_by_move: dict[str, int] = {}
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        uci = move.uci()
        if move not in candidate_moves:
            candidate_moves.append(move)
        pvs_by_move[uci] = [m.uci() for m in pv]
        score = info.get("score")
        if isinstance(score, chess.engine.PovScore):
            cp_by_move[uci] = cp_from_score(score, board, mate_score)

    if played not in candidate_moves:
        candidate_moves.append(played)

    rows: list[CandidateEval] = []
    for rank, move in enumerate(candidate_moves, start=1):
        child = board.copy(stack=False)
        child.push(move)
        uci = move.uci()
        if uci in cp_by_move:
            parent_cp_white = cp_by_move[uci]
            # Convert parent score after choosing move into child-position
            # white score. This is equivalent to analysing the child, but
            # keeps the PV top moves cheap.
            cp_white = parent_cp_white
            pv = pvs_by_move.get(uci, [])
        else:
            cp_white, pv = eval_child(engine, board, move, depth=depth, mate_score=mate_score)
        eval_text = f"{cp_white / 100.0:.2f}"
        label = label_from_eval(child.fen(), eval_text, perspective="white", cp_scale=cp_scale)
        rows.append(
            CandidateEval(
                move=uci,
                child_fen=normalize_fen_key(child.fen()),
                cp_white=cp_white,
                cp_stm_after=label.cp_stm,
                target_stm_after=label.value_stm,
                rank=rank,
                is_played=(move == played),
                is_best=False,
                pv=pv,
            )
        )

    if not rows:
        return 0, 0, "", []

    # For the side to move at the parent, larger is better for White and
    # smaller is better for Black.
    reverse = board.turn == chess.WHITE
    rows.sort(key=lambda item: item.cp_white, reverse=reverse)
    for rank, row in enumerate(rows, start=1):
        row.rank = rank
        row.is_best = rank == 1
    best = rows[0]
    played_row = next(row for row in rows if row.is_played)
    if board.turn == chess.WHITE:
        loss = best.cp_white - played_row.cp_white
    else:
        loss = played_row.cp_white - best.cp_white
    return int(loss), best.cp_white, best.move, rows


def iter_report_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        raw = Path(pattern)
        if raw.is_file():
            paths.append(raw)
        else:
            paths.extend(sorted(ROOT.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def nn_to_move(game: dict[str, Any], board: chess.Board) -> bool:
    return (board.turn == chess.WHITE and game.get("white") == "nn") or (
        board.turn == chess.BLACK and game.get("black") == "nn"
    )


def mine_report(engine: chess.engine.SimpleEngine,
                report_path: Path,
                depth: int,
                multipv: int,
                threshold_cp: int,
                max_positions: int,
                mate_score: int,
                cp_scale: float) -> list[BlunderRow]:
    with report_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    rows: list[BlunderRow] = []
    for game in payload.get("games", []):
        if game.get("winner") == "draw":
            continue
        if game.get("white") != "nn" and game.get("black") != "nn":
            continue
        nn_color = "white" if game.get("white") == "nn" else "black"
        board = chess.Board(str(game["start_fen"]))
        for ply, move_uci in enumerate(game.get("moves_uci", []), start=1):
            try:
                move = chess.Move.from_uci(str(move_uci))
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            if nn_to_move(game, board):
                loss, best_cp_white, best_move, candidates = analyse_position(
                    engine,
                    board,
                    move,
                    depth=depth,
                    multipv=multipv,
                    mate_score=mate_score,
                    cp_scale=cp_scale,
                )
                if loss >= threshold_cp:
                    played = next(item for item in candidates if item.is_played)
                    rows.append(
                        BlunderRow(
                            report=str(report_path),
                            game_index=int(game.get("index", 0)),
                            ply=ply,
                            nn_color=nn_color,
                            start_fen=str(game["start_fen"]),
                            fen=normalize_fen_key(board.fen()),
                            played_move=move.uci(),
                            sf_best_move=best_move,
                            loss_cp=loss,
                            best_cp_white=best_cp_white,
                            played_cp_white=played.cp_white,
                            side_to_move="white" if board.turn == chess.WHITE else "black",
                            result=str(game.get("result", "")),
                            termination=str(game.get("termination", "")),
                            candidates=candidates,
                        )
                    )
                    if max_positions > 0 and len(rows) >= max_positions:
                        return rows
            board.push(move)
    return rows


def write_training_rows(blunders: list[BlunderRow], output: Path, source: str) -> int:
    count = 0
    seen: set[tuple[str, str]] = set()
    with output.open("w", encoding="utf-8") as fp:
        for blunder in blunders:
            for candidate in blunder.candidates:
                key = (candidate.child_fen, candidate.move)
                if key in seen:
                    continue
                seen.add(key)
                eval_text = f"{candidate.cp_white / 100.0:.2f}"
                label = label_from_eval(candidate.child_fen, eval_text, perspective="white")
                buckets = buckets_for_position(candidate.child_fen, label, "")
                row = {
                    "fen": candidate.child_fen,
                    "depth": 0,
                    "knodes": 0,
                    "evaluation": eval_text,
                    "score_perspective": "white",
                    "target_stm": label.value_stm,
                    "cp_stm": label.cp_stm,
                    "is_mate": label.is_mate,
                    "terminal": label.terminal,
                    "best_move": "",
                    "best_line": " ".join(candidate.pv),
                    "bucket": buckets.key(),
                    "source": source,
                    "move_from_parent": candidate.move,
                    "parent_fen": blunder.fen,
                    "parent_played_move": blunder.played_move,
                    "parent_sf_best_move": blunder.sf_best_move,
                    "parent_loss_cp": blunder.loss_cp,
                    "candidate_rank": candidate.rank,
                    "candidate_is_played": candidate.is_played,
                    "candidate_is_best": candidate.is_best,
                    "weight": min(12.0, 1.0 + blunder.loss_cp / 120.0),
                }
                fp.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine NNUE-vs-classic games for Stockfish-labeled NN blunders.")
    parser.add_argument("--report", action="append", default=[],
                        help=f"Report path or glob relative to repo. Default: {DEFAULT_REPORT_GLOB}")
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--multipv", type=int, default=6)
    parser.add_argument("--threshold-cp", type=int, default=180)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--mate-score", type=int, default=1200)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    patterns = args.report or [DEFAULT_REPORT_GLOB]
    reports = iter_report_paths(patterns)
    if not reports:
        raise SystemExit(f"no reports matched: {patterns}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_blunders: list[BlunderRow] = []
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:
        for report in reports:
            print(f"[mine] {report}", flush=True)
            all_blunders.extend(
                mine_report(
                    engine=engine,
                    report_path=report,
                    depth=args.depth,
                    multipv=args.multipv,
                    threshold_cp=args.threshold_cp,
                    max_positions=max(0, args.max_positions - len(all_blunders)) if args.max_positions > 0 else 0,
                    mate_score=args.mate_score,
                    cp_scale=args.cp_scale,
                )
            )
            if args.max_positions > 0 and len(all_blunders) >= args.max_positions:
                break

    all_blunders.sort(key=lambda item: item.loss_cp, reverse=True)
    blunder_path = args.out_dir / "blunders.jsonl"
    with blunder_path.open("w", encoding="utf-8") as fp:
        for item in all_blunders:
            payload = asdict(item)
            fp.write(json.dumps(payload, sort_keys=True) + "\n")

    training_path = args.out_dir / "critical_child_positions.jsonl"
    training_rows = write_training_rows(all_blunders, training_path, source="nnue-blunder-child")
    summary = {
        "generated_at_unix": time.time(),
        "elapsed_s": time.time() - started,
        "reports": [str(path) for path in reports],
        "stockfish": args.stockfish,
        "depth": args.depth,
        "multipv": args.multipv,
        "threshold_cp": args.threshold_cp,
        "blunders": len(all_blunders),
        "training_rows": training_rows,
        "blunders_path": str(blunder_path),
        "training_path": str(training_path),
        "top": [asdict(item) for item in all_blunders[:20]],
    }
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    print(f"[done] blunders={len(all_blunders)} training_rows={training_rows}", flush=True)
    print(f"[done] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
