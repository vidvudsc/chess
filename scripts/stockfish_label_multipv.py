#!/usr/bin/env python3
"""Label positions with several Stockfish root moves and their scores."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path

import chess
import chess.engine


def load_positions(path: Path, count: int, seed: int) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for raw in source:
            fen = raw.split(";", 1)[0].strip()
            if not fen or fen in seen:
                continue
            try:
                chess.Board(fen)
            except ValueError:
                continue
            seen.add(fen)
            rows.append(fen)
    random.Random(seed).shuffle(rows)
    return rows[:count]


def label_chunk(
    engine_path: Path,
    depth: int,
    multipv: int,
    hash_mb: int,
    fens: list[str],
) -> list[dict[str, object]]:
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    engine.configure({"Threads": 1, "Hash": hash_mb})
    labelled: list[dict[str, object]] = []
    try:
        for fen in fens:
            board = chess.Board(fen)
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=min(multipv, board.legal_moves.count()),
            )
            if isinstance(infos, dict):
                infos = [infos]
            moves: list[dict[str, int | str]] = []
            for info in infos:
                pv = info.get("pv", [])
                if not pv:
                    continue
                score = info["score"].pov(board.turn).score(mate_score=30000)
                if score is None:
                    continue
                moves.append({
                    "uci": pv[0].uci(),
                    "score_cp": int(score),
                })
            if len(moves) >= 2:
                moves.sort(key=lambda row: int(row["score_cp"]), reverse=True)
                labelled.append({
                    "fen": fen,
                    "depth": int(infos[0].get("depth", depth)),
                    "moves": moves,
                })
    finally:
        engine.quit()
    return labelled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--multipv", type=int, default=6)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    if args.count <= 0 or args.depth <= 0 or args.multipv < 2:
        parser.error("count/depth must be positive and multipv must be at least 2")
    if args.workers <= 0 or args.hash_mb <= 0:
        parser.error("workers and hash-mb must be positive")
    if not args.engine.exists():
        parser.error(f"engine not found: {args.engine}")

    fens = load_positions(args.positions, args.count, args.seed)
    chunks = [fens[index::args.workers] for index in range(args.workers)]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                label_chunk,
                args.engine,
                args.depth,
                args.multipv,
                args.hash_mb,
                chunk,
            )
            for chunk in chunks
            if chunk
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())

    order = {fen: index for index, fen in enumerate(fens)}
    rows.sort(key=lambda row: order[str(row["fen"])])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
    temp.replace(args.out)
    print(
        f"labelled {len(rows)}/{len(fens)} positions at depth {args.depth} "
        f"with MultiPV {args.multipv}: {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
