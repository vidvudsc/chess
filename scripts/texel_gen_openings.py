#!/usr/bin/env python3
"""Generate diverse, roughly balanced opening FENs for self-play data runs.

Follows the HCE's preferred opening moves with occasional random deviations,
then keeps only positions whose quick eval remains inside a configurable
window. This produces much more natural variety than choosing every opening
move uniformly at random.
"""
import argparse
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine


def gen_batch(args_tuple):
    (engine_path, count, plies_min, plies_max, window_cp, movetime_ms,
     engine_move_ms, random_move_rate, seed) = args_tuple
    rng = random.Random(seed)
    out = []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        attempts = 0
        while len(out) < count and attempts < count * 30:
            attempts += 1
            board = chess.Board()
            plies = rng.randint(plies_min, plies_max)
            ok = True
            for _ in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    ok = False
                    break
                result = engine.play(
                    board,
                    chess.engine.Limit(time=max(0.001, engine_move_ms / 1000.0)))
                best_move = result.move
                if best_move is None or best_move not in board.legal_moves:
                    ok = False
                    break
                if rng.random() < random_move_rate and len(moves) > 1:
                    alternatives = [move for move in moves if move != best_move]
                    board.push(rng.choice(alternatives))
                else:
                    board.push(best_move)
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
    print(f"worker seed {seed}: accepted {len(out)}/{count} openings",
          file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--count", type=int, default=15000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Keep existing unique FENs in --out and top up to --count.")
    ap.add_argument("--plies-min", type=int, default=6)
    ap.add_argument("--plies-max", type=int, default=10)
    ap.add_argument("--window-cp", type=int, default=200)
    ap.add_argument("--movetime-ms", type=int, default=30)
    ap.add_argument("--engine-move-ms", type=int, default=5,
                    help="HCE time per guided opening move.")
    ap.add_argument("--random-move-rate", type=float, default=0.22,
                    help="Chance of a legal deviation from HCE's preferred move.")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.count <= 0:
        ap.error("--count must be positive")
    if args.concurrency <= 0:
        ap.error("--concurrency must be positive")
    if not 0.0 <= args.random_move_rate <= 1.0:
        ap.error("--random-move-rate must be between 0 and 1")
    if args.plies_min < 0 or args.plies_max < args.plies_min:
        ap.error("opening ply range is invalid")

    out_path = Path(args.out)
    seen = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as existing:
            for raw in existing:
                fen = raw.strip()
                if fen:
                    seen.add(" ".join(fen.split()[:4]))
        print(f"resuming with {len(seen)} existing unique openings",
              file=sys.stderr)
    output_mode = "a" if args.resume else "w"
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
            out_path.open(output_mode, encoding="utf-8") as fout:
        for round_index in range(8):
            remaining = args.count - len(seen)
            if remaining <= 0:
                break
            # Workers cannot share dedupe sets, so oversample each round and
            # refill any cross-worker collisions with fresh deterministic seeds.
            per_worker = math.ceil(remaining * 1.10 / args.concurrency) + 1
            tasks = [
                (args.engine, per_worker, args.plies_min, args.plies_max,
                 args.window_cp, args.movetime_ms, args.engine_move_ms,
                 args.random_move_rate,
                 args.seed * 1000 + round_index * args.concurrency + i)
                for i in range(args.concurrency)
            ]
            before = len(seen)
            for batch in pool.map(gen_batch, tasks):
                if len(seen) >= args.count:
                    break
                for fen in batch:
                    key = " ".join(fen.split()[:4])
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(fen + "\n")
                    if len(seen) >= args.count:
                        break
            print(f"round {round_index + 1}: {len(seen)}/{args.count} unique",
                  file=sys.stderr)
            if len(seen) == before:
                break
    print(f"wrote {len(seen)} unique openings to {args.out}", file=sys.stderr)
    if len(seen) < args.count:
        print(f"failed to reach requested count {args.count}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
