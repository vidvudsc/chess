#!/usr/bin/env python3
"""Build child-position pairs from Stockfish MultiPV labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import chess


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and move.promotion is None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--positions-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--alternatives", type=int, default=4)
    parser.add_argument("--min-gap-cp", type=int, default=15)
    parser.add_argument("--max-gap-cp", type=int, default=300)
    parser.add_argument("--allow-tactical", action="store_true")
    args = parser.parse_args()

    if args.alternatives <= 0:
        parser.error("alternatives must be positive")
    if args.min_gap_cp < 0 or args.max_gap_cp < args.min_gap_cp:
        parser.error("invalid score-gap range")

    args.positions_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    positions_temp = args.positions_out.with_suffix(args.positions_out.suffix + ".tmp")
    metadata_temp = args.metadata_out.with_suffix(args.metadata_out.suffix + ".tmp")

    roots_read = 0
    roots_kept = 0
    child_rows = 0
    with args.labels.open("r", encoding="utf-8") as source, \
            positions_temp.open("w", encoding="utf-8") as positions, \
            metadata_temp.open("w", encoding="utf-8", newline="") as metadata:
        writer = csv.writer(metadata, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "group_id",
            "root_side",
            "move",
            "teacher_score_cp",
            "teacher_gap_cp",
            "is_best",
        ])
        for raw in source:
            if not raw.strip():
                continue
            roots_read += 1
            row = json.loads(raw)
            try:
                board = chess.Board(str(row["fen"]))
            except (KeyError, ValueError):
                continue
            labelled = row.get("moves", [])
            if not isinstance(labelled, list) or len(labelled) < 2:
                continue

            legal: list[tuple[chess.Move, int]] = []
            for move_row in labelled:
                try:
                    move = chess.Move.from_uci(str(move_row["uci"]))
                    score = int(move_row["score_cp"])
                except (KeyError, TypeError, ValueError):
                    continue
                if move not in board.legal_moves:
                    continue
                if not args.allow_tactical and not is_quiet(board, move):
                    continue
                legal.append((move, score))
            if len(legal) < 2:
                continue
            legal.sort(key=lambda item: item[1], reverse=True)
            best_move, best_score = legal[0]
            alternatives = [
                (move, score)
                for move, score in legal[1:]
                if args.min_gap_cp <= best_score - score <= args.max_gap_cp
            ][:args.alternatives]
            if not alternatives:
                continue

            group_id = f"pair-{roots_kept:07d}"
            root_side = "w" if board.turn == chess.WHITE else "b"
            selected = [(best_move, best_score, 1)] + [
                (move, score, 0) for move, score in alternatives
            ]
            for move, score, is_best in selected:
                child = board.copy(stack=False)
                child.push(move)
                positions.write(f"{child.fen()};0.5\n")
                writer.writerow([
                    group_id,
                    root_side,
                    move.uci(),
                    score,
                    best_score - score,
                    is_best,
                ])
                child_rows += 1
            roots_kept += 1

    positions_temp.replace(args.positions_out)
    metadata_temp.replace(args.metadata_out)
    print(f"roots read: {roots_read}")
    print(f"roots kept: {roots_kept}")
    print(f"child rows: {child_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
