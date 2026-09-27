#!/usr/bin/env python3
"""Build a balanced endgame start-position suite from real games.

One position per game: ply >= 50, reduced material (phase <= 10 on the
Q4/R2/B1/N1 scale), not in check, previous move not a capture, more than
six men, and a Stockfish score within +-MAX_CP at a fixed depth.
"""
import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import chess.pgn

PHASE = {chess.QUEEN: 4, chess.ROOK: 2, chess.BISHOP: 1, chess.KNIGHT: 1}


def phase(board):
    return sum(len(board.pieces(p, c)) * w for p, w in PHASE.items() for c in chess.COLORS)


def candidates(pgn_path, rng):
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while (game := chess.pgn.read_game(f)) is not None:
            board = game.board()
            pool = []
            for ply, move in enumerate(game.mainline_moves(), start=1):
                capture = board.is_capture(move)
                board.push(move)
                if (ply >= 50 and not capture and not board.is_check()
                        and phase(board) <= 10 and len(board.piece_map()) > 6
                        and not board.is_game_over()):
                    pool.append(board.fen())
            if pool:
                yield rng.choice(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max-cp", type=int, default=120)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fens = []
    for path in args.pgn:
        fens.extend(candidates(path, rng))
    fens = sorted(set(fens))
    rng.shuffle(fens)

    def score(fen):
        with chess.engine.SimpleEngine.popen_uci(args.stockfish) as eng:
            eng.configure({"Threads": 1, "Hash": 32})
            info = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=args.depth))
            return fen, info["score"].white().score(mate_score=100000)

    kept = []
    batch = 200
    with ThreadPoolExecutor(args.workers) as pool:
        for i in range(0, len(fens), batch):
            for fen, cp in pool.map(score, fens[i:i + batch]):
                if cp is not None and abs(cp) <= args.max_cp:
                    kept.append(fen)
            print(f"scored {min(i + batch, len(fens))}/{len(fens)} kept {len(kept)}", flush=True)
            if len(kept) >= args.limit:
                break
    kept = kept[:args.limit]
    with open(args.out, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"wrote {len(kept)} positions to {args.out}")


if __name__ == "__main__":
    main()
