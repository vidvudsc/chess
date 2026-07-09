#!/usr/bin/env python3
"""Generate self-play games with the HCE engine for Texel tuning.

Plays ENGINE vs ENGINE from a list of FENs, both colors, fixed movetime.
Writes a single PGN with standard headers so scripts/texel_build_dataset.py
(and similar tools) can consume it.

Usage:
    texel_selfplay.py --engine src/core/engine/chess_uci \
        --positions-file data/positions/lichess_equal_positions.fen \
        --out-pgn selfplay.pgn --think-ms 120 --concurrency 6
"""
import argparse
import datetime
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import chess
import chess.engine
import chess.pgn


def load_fens(path: Path) -> List[str]:
    fens: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.count(" ") == 3:
                line = f"{line} 0 1"
            fens.append(line)
    if not fens:
        raise RuntimeError(f"no positions loaded from {path}")
    return fens


def play_game(engine_path: str, think_ms: int, max_plies: int, start_fen: str,
              white_name: str, black_name: str) -> chess.pgn.Game:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        if "MoveTime" in engine.options:
            engine.configure({"MoveTime": think_ms})
        if "BookFile" in engine.options:
            engine.configure({"BookFile": ""})

        board = chess.Board(start_fen)
        limit = chess.engine.Limit(time=max(0.001, think_ms / 1000.0))
        game = chess.pgn.Game()
        game.headers["Event"] = "HCE texel selfplay"
        game.headers["Site"] = "?"
        game.headers["Date"] = datetime.date.today().isoformat().replace("-", ".")
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = start_fen

        node = game
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None or len(board.move_stack) >= max_plies:
                break
            result = engine.play(board, limit)
            if result.move is None or result.move not in board.legal_moves:
                # Forfeit the side that failed to move.
                break
            board.push(result.move)
            node = node.add_variation(result.move)

        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            game.headers["Result"] = outcome.result()
            game.headers["Termination"] = str(outcome.termination)
        else:
            game.headers["Result"] = "1/2-1/2"
            game.headers["Termination"] = "max_plies"
        return game
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def worker_init(engine_path: str, think_ms: int):
    t = threading.current_thread()
    t._hce_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    if "MoveTime" in t._hce_engine.options:
        t._hce_engine.configure({"MoveTime": think_ms})
    if "BookFile" in t._hce_engine.options:
        t._hce_engine.configure({"BookFile": ""})


def toggle_stm(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 2:
        parts[1] = "b" if parts[1] == "w" else "w"
    return " ".join(parts)


def worker_play(args) -> chess.pgn.Game:
    idx, total, engine_path, think_ms, max_plies, start_fen, white_name, black_name = args
    t = threading.current_thread()
    engine = getattr(t, "_hce_engine", None)
    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        if "MoveTime" in engine.options:
            engine.configure({"MoveTime": think_ms})
        if "BookFile" in engine.options:
            engine.configure({"BookFile": ""})
        t._hce_engine = engine

    board = chess.Board(start_fen)
    limit = chess.engine.Limit(time=max(0.001, think_ms / 1000.0))
    game = chess.pgn.Game()
    game.headers["Event"] = "HCE texel selfplay"
    game.headers["Site"] = "?"
    game.headers["Date"] = datetime.date.today().isoformat().replace("-", ".")
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["SetUp"] = "1"
    game.headers["FEN"] = start_fen

    node = game
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None or len(board.move_stack) >= max_plies:
            break
        try:
            result = engine.play(board, limit)
        except Exception as exc:
            print(f"[game {idx}/{total}] engine error: {exc}", file=sys.stderr)
            break
        if result.move is None or result.move not in board.legal_moves:
            break
        board.push(result.move)
        node = node.add_variation(result.move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        game.headers["Result"] = outcome.result()
        game.headers["Termination"] = str(outcome.termination)
    else:
        game.headers["Result"] = "1/2-1/2"
        game.headers["Termination"] = "max_plies"
    print(f"[game {idx}/{total}] {white_name} vs {black_name} "
          f"result={game.headers['Result']} plies={len(board.move_stack)} "
          f"term={game.headers['Termination']}", file=sys.stderr)
    return game


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to the UCI engine binary.")
    ap.add_argument("--positions-file", required=True)
    ap.add_argument("--out-pgn", required=True)
    ap.add_argument("--think-ms", type=int, default=120)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-games", type=int, default=0,
                    help="If >0, cap total games (rounds down to pairs).")
    args = ap.parse_args()

    engine_path = Path(args.engine).expanduser().resolve()
    if not engine_path.exists():
        raise SystemExit(f"engine not found: {engine_path}")

    fens = load_fens(Path(args.positions_file).expanduser())
    tasks = []
    total_pairs = len(fens)
    if args.max_games > 0:
        total_pairs = min(total_pairs, args.max_games // 2)
    for i, fen in enumerate(fens[:total_pairs]):
        tasks.append((2 * i + 1, total_pairs * 2, str(engine_path), args.think_ms,
                      args.max_plies, fen, "HCE", "HCE"))
        tasks.append((2 * i + 2, total_pairs * 2, str(engine_path), args.think_ms,
                      args.max_plies, toggle_stm(fen), "HCE", "HCE"))

    out_path = Path(args.out_pgn).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"playing {len(tasks)} games from {total_pairs} positions "
          f"(think {args.think_ms}ms, max-plies {args.max_plies})", file=sys.stderr)
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.concurrency,
                               initializer=worker_init,
                               initargs=(str(engine_path), args.think_ms)) as pool:
        # initializer is optional; worker_play reopens engine if thread-local missing.
        for game in pool.map(worker_play, tasks):
            fout.write(str(game))
            fout.write("\n\n")
            fout.flush()

    elapsed = time.perf_counter() - started
    print(f"wrote {len(tasks)} games to {out_path} in {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
