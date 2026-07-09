#!/usr/bin/env python3
"""Build a Texel-tuning dataset from a PGN of games.

Emits one line per sampled position: "<FEN>;<result>" where result is the game
outcome from White's perspective (1.0 win, 0.5 draw, 0.0 loss). Positions are
filtered to be "quiet enough" for static-eval tuning: side to move not in
check, skip the opening and the last few plies, and sample a handful of
positions per game so a single long game does not dominate.

Usage:
    texel_build_dataset.py --pgn games.pgn --out data.txt [--skip-opening 10]
                           [--skip-tail 6] [--per-game 10] [--drop-forfeit]
"""
import argparse
import sys

import chess
import chess.pgn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-opening", type=int, default=10,
                    help="Skip this many opening plies (book/theory noise).")
    ap.add_argument("--skip-tail", type=int, default=6,
                    help="Skip this many final plies (result already decided).")
    ap.add_argument("--per-game", type=int, default=12,
                    help="Max sampled positions per game (evenly spaced).")
    ap.add_argument("--drop-forfeit", action="store_true",
                    help="Drop games decided on time (noisy labels).")
    args = ap.parse_args()

    result_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    games = 0
    kept = 0
    positions = 0
    with open(args.pgn, encoding="utf-8", errors="replace") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        while True:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            games += 1
            res = game.headers.get("Result", "*")
            if res not in result_map:
                continue
            if args.drop_forfeit and game.headers.get("Termination") == "Time forfeit":
                continue
            label = result_map[res]

            board = game.board()
            moves = list(game.mainline_moves())
            n = len(moves)
            # Candidate ply indices (position AFTER playing moves[0..i-1]).
            lo = args.skip_opening
            hi = n - args.skip_tail
            if hi <= lo:
                continue

            # Collect quiet candidate FENs across the game, then evenly sample.
            candidates = []
            for i, mv in enumerate(moves):
                board.push(mv)
                ply = i + 1
                if ply < lo or ply > hi:
                    continue
                if board.is_check():
                    continue
                candidates.append(board.fen())
            if not candidates:
                continue
            kept += 1
            if len(candidates) <= args.per_game:
                chosen = candidates
            else:
                step = len(candidates) / args.per_game
                chosen = [candidates[int(k * step)] for k in range(args.per_game)]
            for fen in chosen:
                fout.write(f"{fen};{label}\n")
                positions += 1

    print(f"games read       : {games}", file=sys.stderr)
    print(f"games kept       : {kept}", file=sys.stderr)
    print(f"positions written: {positions}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
