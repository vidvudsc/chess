#!/usr/bin/env python3
"""Generate diverse, roughly balanced opening FENs for self-play data runs.

Random-walks a small number of plies from the start position, then keeps the
position only if the engine's quick eval is within a window (so self-play games
start playable, not pre-decided). Dedupes by FEN. Output feeds
texel_selfplay.py, which plays each FEN twice with colors swapped.
"""
import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine


def gen_batch(args_tuple):
    engine_path, count, plies_min, plies_max, window_cp, movetime_ms, seed = args_tuple
    rng = random.Random(seed)
    out = []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        attempts = 0
        while len(out) < count and attempts < count * 12:
            attempts += 1
            board = chess.Board()
            plies = rng.randint(plies_min, plies_max)
            ok = True
            for _ in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    ok = False
                    break
                board.push(rng.choice(moves))
            if not ok or board.is_game_over():
                continue
            info = engine.analyse(board, chess.engine.Limit(time=movetime_ms / 1000.0))
            score = info.get("score")
            if score is None:
                continue
            cp = score.pov(chess.WHITE).score(mate_score=30000)
            if cp is None or abs(cp) > window_cp:
                continue
            out.append(board.fen())
    finally:
        engine.quit()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--count", type=int, default=15000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plies-min", type=int, default=6)
    ap.add_argument("--plies-max", type=int, default=10)
    ap.add_argument("--window-cp", type=int, default=200)
    ap.add_argument("--movetime-ms", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_worker = args.count // args.concurrency + 1
    tasks = [
        (args.engine, per_worker, args.plies_min, args.plies_max,
         args.window_cp, args.movetime_ms, args.seed * 1000 + i)
        for i in range(args.concurrency)
    ]
    seen = set()
    with ProcessPoolExecutor(max_workers=args.concurrency) as pool, \
            open(args.out, "w", encoding="utf-8") as fout:
        for batch in pool.map(gen_batch, tasks):
            for fen in batch:
                key = " ".join(fen.split()[:4])
                if key in seen:
                    continue
                seen.add(key)
                fout.write(fen + "\n")
    print(f"wrote {len(seen)} unique openings to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
