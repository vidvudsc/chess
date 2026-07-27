#!/usr/bin/env python3
"""Generate self-play games with the HCE engine for Texel tuning.

Plays ENGINE vs ENGINE from a list of FENs at fixed movetime.
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

_worker_engines = []
_worker_engines_lock = threading.Lock()


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


def configure_engine(engine, think_ms: int) -> None:
    options = {}
    if "MoveTime" in engine.options:
        options["MoveTime"] = think_ms
    if "BookFile" in engine.options:
        options["BookFile"] = ""
    if "Backend" in engine.options:
        options["Backend"] = "classic"
    if "Threads" in engine.options:
        options["Threads"] = 1
    if options:
        engine.configure(options)


def register_worker_engine(engine) -> None:
    with _worker_engines_lock:
        _worker_engines.append(engine)


def close_worker_engines() -> None:
    with _worker_engines_lock:
        engines = list(_worker_engines)
        _worker_engines.clear()
    for engine in engines:
        try:
            engine.quit()
        except Exception:
            try:
                engine.close()
            except Exception:
                pass


def worker_init(engine_path: str, think_ms: int):
    t = threading.current_thread()
    t._hce_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    configure_engine(t._hce_engine, think_ms)
    register_worker_engine(t._hce_engine)


def mirror_colors(fen: str) -> str:
    return chess.Board(fen).mirror().fen()


def worker_play(args) -> chess.pgn.Game:
    idx, total, engine_path, think_ms, max_plies, start_fen, white_name, black_name = args
    t = threading.current_thread()
    engine = getattr(t, "_hce_engine", None)
    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        configure_engine(engine, think_ms)
        t._hce_engine = engine
        register_worker_engine(engine)

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
    engine_failed = False
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None or len(board.move_stack) >= max_plies:
            break
        try:
            result = engine.play(board, limit)
        except Exception as exc:
            print(f"[game {idx}/{total}] engine error: {exc}", file=sys.stderr)
            engine_failed = True
            break
        if result.move is None or result.move not in board.legal_moves:
            print(f"[game {idx}/{total}] engine returned no legal move",
                  file=sys.stderr)
            engine_failed = True
            break
        board.push(result.move)
        node = node.add_variation(result.move)

    outcome = board.outcome(claim_draw=True)
    if engine_failed:
        game.headers["Result"] = "*"
        game.headers["Termination"] = "engine_error"
    elif outcome is not None:
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
                    help="If >0, cap the number of generated games.")
    ap.add_argument("--paired-mirror", action="store_true",
                    help="Also play a correctly color-mirrored copy of each FEN.")
    args = ap.parse_args()

    engine_path = Path(args.engine).expanduser().resolve()
    if not engine_path.exists():
        raise SystemExit(f"engine not found: {engine_path}")

    fens = load_fens(Path(args.positions_file).expanduser())
    tasks = []
    positions_count = len(fens)
    if args.max_games > 0:
        games_per_position = 2 if args.paired_mirror else 1
        positions_count = min(
            positions_count,
            (args.max_games + games_per_position - 1) // games_per_position)
    selected = fens[:positions_count]
    start_fens = []
    for fen in selected:
        start_fens.append(fen)
        if args.paired_mirror:
            start_fens.append(mirror_colors(fen))
    if args.max_games > 0:
        start_fens = start_fens[:args.max_games]
    total_games = len(start_fens)
    for i, fen in enumerate(start_fens, start=1):
        tasks.append((i, total_games, str(engine_path), args.think_ms,
                      args.max_plies, fen, "HCE", "HCE"))

    out_path = Path(args.out_pgn).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"playing {len(tasks)} games from {positions_count} positions "
          f"(think {args.think_ms}ms, max-plies {args.max_plies})", file=sys.stderr)
    started = time.perf_counter()
    try:
        with out_path.open("w", encoding="utf-8") as fout, \
                ThreadPoolExecutor(max_workers=args.concurrency,
                                   initializer=worker_init,
                                   initargs=(str(engine_path), args.think_ms)) as pool:
            for game in pool.map(worker_play, tasks):
                fout.write(str(game))
                fout.write("\n\n")
                fout.flush()
    finally:
        close_worker_engines()

    elapsed = time.perf_counter() - started
    print(f"wrote {len(tasks)} games to {out_path} in {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
