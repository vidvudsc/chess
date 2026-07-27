#!/usr/bin/env python3
"""Compare two UCI binaries at fixed depth on the same FENs."""

import argparse
from pathlib import Path

import chess
import chess.engine


def score_key(info, board):
    score = info["score"].pov(board.turn)
    return (score.mate(), score.score())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-a", required=True)
    parser.add_argument("--engine-b", required=True)
    parser.add_argument("--positions-file", required=True)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    fens = []
    with Path(args.positions_file).open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.count(" ") == 3:
                line += " 0 1"
            fens.append(line)
            if len(fens) >= args.count:
                break

    mismatches = 0
    with chess.engine.SimpleEngine.popen_uci(args.engine_a) as engine_a, \
            chess.engine.SimpleEngine.popen_uci(args.engine_b) as engine_b:
        for index, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            limit = chess.engine.Limit(depth=args.depth)
            info_a = engine_a.analyse(board, limit)
            info_b = engine_b.analyse(board, limit)
            move_a = info_a["pv"][0] if info_a.get("pv") else None
            move_b = info_b["pv"][0] if info_b.get("pv") else None
            same = (move_a == move_b and
                    score_key(info_a, board) == score_key(info_b, board) and
                    info_a.get("nodes") == info_b.get("nodes"))
            if not same:
                mismatches += 1
                print(
                    f"{index}: A move={move_a} score={score_key(info_a, board)} "
                    f"nodes={info_a.get('nodes')} | B move={move_b} "
                    f"score={score_key(info_b, board)} nodes={info_b.get('nodes')}")

    print(f"fixed-depth comparison: {len(fens) - mismatches}/{len(fens)} identical")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
