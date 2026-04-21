#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import shutil
from dataclasses import dataclass

import chess
import chess.engine


@dataclass
class Config:
    openings: pathlib.Path
    out: pathlib.Path
    count: int
    seed: int
    stockfish_path: str
    multipv: int
    pool_window_cp: int
    final_window_cp: int
    choose_margin_cp: int
    extra_plies_min: int
    extra_plies_max: int
    attempts_per_opening: int
    candidate_movetime_ms: int
    verify_movetime_ms: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build equal AI test-lab positions from curated opening seeds")
    parser.add_argument("--openings", default="data/openings/opening_games_100.txt")
    parser.add_argument("--out", default="data/positions/lichess_equal_positions.fen")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--pool-window-cp", type=int, default=90)
    parser.add_argument("--final-window-cp", type=int, default=45)
    parser.add_argument("--choose-margin-cp", type=int, default=28)
    parser.add_argument("--extra-plies-min", type=int, default=4)
    parser.add_argument("--extra-plies-max", type=int, default=14)
    parser.add_argument("--attempts-per-opening", type=int, default=40)
    parser.add_argument("--candidate-movetime-ms", type=int, default=8)
    parser.add_argument("--verify-movetime-ms", type=int, default=18)
    parser.add_argument("--stockfish", default="")
    args = parser.parse_args()

    sf = args.stockfish or os.environ.get("STOCKFISH_BIN", "") or (shutil.which("stockfish") or "")
    if not sf:
        raise SystemExit("Stockfish binary not found. Use --stockfish or STOCKFISH_BIN.")

    return Config(
        openings=pathlib.Path(args.openings),
        out=pathlib.Path(args.out),
        count=max(1, args.count),
        seed=args.seed,
        stockfish_path=sf,
        multipv=max(1, args.multipv),
        pool_window_cp=max(1, args.pool_window_cp),
        final_window_cp=max(1, args.final_window_cp),
        choose_margin_cp=max(1, args.choose_margin_cp),
        extra_plies_min=max(1, args.extra_plies_min),
        extra_plies_max=max(args.extra_plies_min, args.extra_plies_max),
        attempts_per_opening=max(1, args.attempts_per_opening),
        candidate_movetime_ms=max(1, args.candidate_movetime_ms),
        verify_movetime_ms=max(1, args.verify_movetime_ms),
    )


def parse_opening_lines(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    openings: list[tuple[str, list[str]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, seq = line.partition("|")
        moves = [tok for tok in seq.split() if tok]
        if not name or not moves:
            continue
        openings.append((name.strip(), moves))
    if not openings:
        raise SystemExit(f"No openings found in {path}")
    return openings


def build_board_from_uci(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


def normalized_position_token(board: chess.Board) -> str:
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    turn = "w" if board.turn == chess.WHITE else "b"
    return f"{board.board_fen()} {turn} {board.castling_xfen()} {ep}"


def canonical_position_key(board: chess.Board) -> str:
    base = normalized_position_token(board)
    mirrored = normalized_position_token(board.mirror())
    return base if base <= mirrored else mirrored


def analyse_white_cp(engine: chess.engine.SimpleEngine, board: chess.Board, movetime_ms: int) -> int | None:
    info = engine.analyse(board, chess.engine.Limit(time=max(0.001, movetime_ms / 1000.0)))
    score = info.get("score")
    if score is None:
        return None
    cp = score.white().score(mate_score=20000)
    if cp is None:
        return None
    return int(cp)


def root_move_choices(engine: chess.engine.SimpleEngine,
                      board: chess.Board,
                      movetime_ms: int,
                      multipv: int,
                      margin_cp: int) -> list[chess.Move]:
    info = engine.analyse(
        board,
        chess.engine.Limit(time=max(0.001, movetime_ms / 1000.0)),
        multipv=multipv,
    )
    rows = info if isinstance(info, list) else [info]
    scored: list[tuple[chess.Move, int]] = []
    for row in rows:
        pv = row.get("pv")
        score = row.get("score")
        if not pv or score is None:
            continue
        cp = score.pov(board.turn).score(mate_score=20000)
        if cp is None:
            continue
        scored.append((pv[0], int(cp)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[1], reverse=True)
    best = scored[0][1]
    choices = [move for move, cp in scored if cp >= best - margin_cp]
    return choices if choices else [scored[0][0]]


def select_equal_subset(candidates: list[tuple[str, int]], count: int, final_window_cp: int) -> list[tuple[str, int]]:
    filtered = [item for item in candidates if abs(item[1]) <= final_window_cp]
    zeros = [item for item in filtered if item[1] == 0]
    negatives = sorted((item for item in filtered if item[1] < 0), key=lambda item: (abs(item[1]), item[1], item[0]))
    positives = sorted((item for item in filtered if item[1] > 0), key=lambda item: (abs(item[1]), item[1], item[0]))

    target_each = min(len(negatives), len(positives), count // 2)
    selected = negatives[:target_each] + positives[:target_each]
    remaining = count - len(selected)
    if remaining > 0:
        leftovers = zeros + negatives[target_each:] + positives[target_each:]
        leftovers.sort(key=lambda item: (abs(item[1]), item[1], item[0]))
        selected.extend(leftovers[:remaining])
    return selected[:count]


def main() -> int:
    cfg = parse_args()
    openings = parse_opening_lines(cfg.openings)
    rng = random.Random(cfg.seed)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)

    target_pool = max(cfg.count * 2, cfg.count + 800)
    pool: dict[str, tuple[str, int]] = {}

    with chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path) as engine:
        try:
            engine.configure({"Threads": 1, "Hash": 64})
        except Exception:
            pass

        attempts = 0
        for round_idx in range(cfg.attempts_per_opening):
            order = openings[:]
            rng.shuffle(order)
            for _name, seed_moves in order:
                if len(pool) >= target_pool:
                    break
                board = build_board_from_uci(seed_moves)
                best_candidate: tuple[str, int] | None = None
                extra = rng.randint(cfg.extra_plies_min, cfg.extra_plies_max)

                for ply in range(extra):
                    if board.is_game_over():
                        break
                    choices = root_move_choices(
                        engine,
                        board,
                        cfg.candidate_movetime_ms,
                        cfg.multipv,
                        cfg.choose_margin_cp,
                    )
                    if not choices:
                        break
                    move = rng.choice(choices)
                    board.push(move)

                    if board.is_game_over() or board.is_check():
                        continue
                    if ply + 1 < cfg.extra_plies_min:
                        continue

                    cp = analyse_white_cp(engine, board, cfg.candidate_movetime_ms)
                    if cp is None or abs(cp) > cfg.pool_window_cp:
                        continue

                    key = canonical_position_key(board)
                    if key in pool:
                        continue
                    fen = board.fen(en_passant="fen")
                    if best_candidate is None or abs(cp) < abs(best_candidate[1]):
                        best_candidate = (fen, cp)

                attempts += 1
                if best_candidate is None:
                    continue
                key = canonical_position_key(chess.Board(best_candidate[0]))
                pool[key] = best_candidate
            if len(pool) >= target_pool:
                break

        verified: list[tuple[str, int]] = []
        for fen, _cp in pool.values():
            board = chess.Board(fen)
            cp = analyse_white_cp(engine, board, cfg.verify_movetime_ms)
            if cp is None:
                continue
            verified.append((fen, cp))

    selected = select_equal_subset(verified, cfg.count, cfg.final_window_cp)
    if len(selected) < cfg.count:
        raise SystemExit(
            f"Only found {len(selected)} clean positions within +/-{cfg.final_window_cp} cp. "
            f"Try increasing attempts or widening the final window."
        )

    rng.shuffle(selected)
    rows = [fen for fen, _cp in selected]
    cfg.out.write_text(
        "\n".join([
            "# AI test-lab positions rebuilt from curated opening seeds and Stockfish filtering",
            f"# source={cfg.openings}",
            f"# count={cfg.count} seed={cfg.seed} pool_window_cp={cfg.pool_window_cp} final_window_cp={cfg.final_window_cp}",
        ] + rows) + "\n",
        encoding="utf-8",
    )

    mean_cp = sum(cp for _fen, cp in selected) / len(selected)
    mean_abs_cp = sum(abs(cp) for _fen, cp in selected) / len(selected)
    print(f"selected={len(selected)} pool={len(verified)} attempts={attempts}")
    print(f"mean_cp={mean_cp:.2f}")
    print(f"mean_abs_cp={mean_abs_cp:.2f}")
    print(f"white_better={sum(1 for _fen, cp in selected if cp > 0)}")
    print(f"black_better={sum(1 for _fen, cp in selected if cp < 0)}")
    print(f"exact_zero={sum(1 for _fen, cp in selected if cp == 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
