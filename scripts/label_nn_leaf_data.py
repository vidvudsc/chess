#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import chess
import chess.engine


ROOT = Path(__file__).resolve().parents[1]
CP_CLIP = 1200
CP_SCALE = 600.0


def cp_to_target(cp: int) -> float:
    clipped = max(-CP_CLIP, min(CP_CLIP, int(cp)))
    return math.tanh(clipped / CP_SCALE)


def score_to_cp_and_mate(score: chess.engine.PovScore, board: chess.Board) -> tuple[int, int | None]:
    pov = score.pov(board.turn)
    mate = pov.mate()
    if mate is not None:
        cp = CP_CLIP if mate > 0 else -CP_CLIP
        return cp, int(mate)
    cp = pov.score()
    if cp is None:
        return 0, None
    return max(-CP_CLIP, min(CP_CLIP, int(cp))), None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label collected NN leaf FENs with Stockfish targets.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stockfish", type=Path, default=None)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--movetime-ms", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    in_path = args.input.resolve()
    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")
    out_path = args.output.resolve() if args.output is not None else in_path.with_suffix(".labeled.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stockfish = args.stockfish
    if stockfish is None:
        found = shutil.which("stockfish")
        if found is None:
            raise SystemExit("stockfish not found; pass --stockfish")
        stockfish = Path(found)

    limit = chess.engine.Limit(
        depth=max(1, args.depth) if args.movetime_ms <= 0 else None,
        time=max(1, args.movetime_ms) / 1000.0 if args.movetime_ms > 0 else None,
    )

    total = 0
    labeled = 0
    skipped = 0
    with chess.engine.SimpleEngine.popen_uci(str(stockfish)) as engine:
        try:
            engine.configure({"Threads": max(1, args.threads), "Hash": max(1, args.hash_mb)})
        except chess.engine.EngineError:
            pass
        with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
            for raw in src:
                if args.limit > 0 and labeled >= args.limit:
                    break
                total += 1
                try:
                    row = json.loads(raw)
                    fen = str(row["fen"])
                    board = chess.Board(fen)
                except Exception:
                    skipped += 1
                    continue

                try:
                    info = engine.analyse(board, limit)
                except Exception:
                    skipped += 1
                    continue
                cp, mate = score_to_cp_and_mate(info["score"], board)
                pv = info.get("pv") or []
                best_move = pv[0].uci() if pv else None
                out = {
                    "fen": fen,
                    "cp_stm": cp,
                    "target_stm": cp_to_target(cp),
                    "is_mate": mate is not None,
                    "mate": mate,
                    "sf_depth": int(info.get("depth") or args.depth),
                    "sf_seldepth": int(info.get("seldepth") or 0),
                    "best_move": best_move,
                    "leaf_phase": row.get("phase"),
                    "leaf_ply": row.get("ply"),
                    "leaf_depth": row.get("depth"),
                    "leaf_nodes": row.get("nodes"),
                    "leaf_nn_score_cp": row.get("score_cp"),
                    "score_perspective": "stm",
                }
                dst.write(json.dumps(out, separators=(",", ":"), sort_keys=True))
                dst.write("\n")
                labeled += 1
                if labeled % 1000 == 0:
                    print(f"[label] {labeled}", flush=True)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "stockfish": str(stockfish),
        "depth": args.depth,
        "movetime_ms": args.movetime_ms,
        "total_read": total,
        "labeled": labeled,
        "skipped": skipped,
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
