#!/usr/bin/env python3
"""Label sampled FEN positions with a Stockfish centipawn score."""

import argparse
import concurrent.futures
import random
from pathlib import Path

import chess
import chess.engine


def load_positions(path, count, seed):
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            fen = line.split(";", 1)[0].strip()
            if not fen or fen in seen:
                continue
            seen.add(fen)
            rows.append(fen)
    random.Random(seed).shuffle(rows)
    return rows[:count]


def label_chunk(engine_path, depth, hash_mb, fens):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    labelled = []
    try:
        for fen in fens:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].pov(chess.WHITE).score(mate_score=30000)
            pv = info.get("pv", [])
            best_move = pv[0].uci() if pv else ""
            labelled.append((fen, score, info.get("depth", depth), best_move))
    finally:
        engine.quit()
    return labelled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    fens = load_positions(args.positions, args.count, args.seed)
    chunks = [fens[i::args.workers] for i in range(args.workers)]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(label_chunk, args.engine, args.depth, args.hash_mb, chunk)
            for chunk in chunks if chunk
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    order = {fen: index for index, fen in enumerate(fens)}
    results.sort(key=lambda row: order[row[0]])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        target.write("fen\tsf_cp_white\tdepth\tbest_move\n")
        for fen, score, depth, best_move in results:
            target.write(f"{fen}\t{score}\t{depth}\t{best_move}\n")
    print(f"labelled {len(results)} positions at depth {args.depth}: {output}")


if __name__ == "__main__":
    main()
