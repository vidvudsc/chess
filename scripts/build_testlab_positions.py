#!/usr/bin/env python3
"""Build balanced AI Test Lab FEN positions from local PGN/.pgn.zst dumps.

Goal:
- produce positions from stronger games ("top" Elo filter)
- keep them separate from training by starting from a later game index
- store directly into data/positions/lichess_equal_positions.fen for the UI test lab
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, TextIO

import chess
import chess.engine
import chess.pgn

try:
    import zstandard as zstd
except Exception:
    zstd = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


@dataclass
class Config:
    pgn: pathlib.Path
    out: pathlib.Path
    count: int
    start_game_index: int
    max_games_scan: int
    min_avg_elo: int
    min_each_elo: int
    min_ply: int
    max_ply: int
    sample_every: int
    eval_window_cp: int
    material_window_cp: int
    movetime_ms: int
    stockfish_path: str
    quiet: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build AI test-lab equal positions from local PGN")
    p.add_argument("--pgn", required=True, help="Input .pgn or .pgn.zst")
    p.add_argument("--out", default="data/positions/lichess_equal_positions.fen", help="Output .fen path")
    p.add_argument("--count", type=int, default=500, help="How many positions to save")
    p.add_argument("--start-game-index", type=int, default=120000,
                   help="1-based game index to start from (use > training slice)")
    p.add_argument("--max-games-scan", type=int, default=200000,
                   help="Max games to scan after start index")
    p.add_argument("--min-avg-elo", type=int, default=2400, help="Minimum average Elo")
    p.add_argument("--min-each-elo", type=int, default=2200, help="Minimum Elo for both players")
    p.add_argument("--min-ply", type=int, default=12, help="First ply to sample in each game")
    p.add_argument("--max-ply", type=int, default=60, help="Last ply to sample in each game")
    p.add_argument("--sample-every", type=int, default=4, help="Sample every N plies in range")
    p.add_argument("--eval-window-cp", type=int, default=55, help="Keep positions with |eval_cp| <= N")
    p.add_argument("--material-window-cp", type=int, default=220, help="Quick material balance pre-filter")
    p.add_argument("--movetime-ms", type=int, default=20, help="Stockfish movetime per sampled position")
    p.add_argument("--stockfish", default="", help="Stockfish binary path (or STOCKFISH_BIN env)")
    p.add_argument("--quiet", action="store_true", help="Reduce output")
    a = p.parse_args()

    sf = a.stockfish or os.environ.get("STOCKFISH_BIN", "") or (shutil.which("stockfish") or "")
    if not sf:
        raise SystemExit("Stockfish binary not found. Use --stockfish or STOCKFISH_BIN.")

    return Config(
        pgn=pathlib.Path(a.pgn),
        out=pathlib.Path(a.out),
        count=max(1, a.count),
        start_game_index=max(1, a.start_game_index),
        max_games_scan=max(1, a.max_games_scan),
        min_avg_elo=max(0, a.min_avg_elo),
        min_each_elo=max(0, a.min_each_elo),
        min_ply=max(1, a.min_ply),
        max_ply=max(2, a.max_ply),
        sample_every=max(1, a.sample_every),
        eval_window_cp=max(1, a.eval_window_cp),
        material_window_cp=max(0, a.material_window_cp),
        movetime_ms=max(5, a.movetime_ms),
        stockfish_path=sf,
        quiet=a.quiet,
    )


@contextmanager
def open_text_auto(path: pathlib.Path) -> Iterator[TextIO]:
    suffixes = [s.lower() for s in path.suffixes]
    is_zst = suffixes[-1:] == [".zst"]

    if not is_zst:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            yield f
        return

    raw_f = path.open("rb")
    try:
        if zstd is not None:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(raw_f)
            text_f = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            try:
                yield text_f
            finally:
                text_f.close()
        else:
            zstd_bin = shutil.which("zstd")
            if not zstd_bin:
                raise RuntimeError(
                    "Input is .zst but no zstandard module and no `zstd` CLI found. "
                    "Install with: python3 -m pip install zstandard"
                )
            proc = subprocess.Popen([zstd_bin, "-dc", str(path)], stdout=subprocess.PIPE)
            if proc.stdout is None:
                raise RuntimeError("failed to open zstd pipe")
            text_f = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
            try:
                yield text_f
            finally:
                text_f.close()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"zstd failed with exit code {rc}")
    finally:
        raw_f.close()


def parse_elo(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        v = int(value)
    except Exception:
        return None
    if v <= 0:
        return None
    return v


def material_balance_cp(board: chess.Board) -> int:
    w = 0
    b = 0
    for piece, cp in PIECE_CP.items():
        w += cp * len(board.pieces(piece, chess.WHITE))
        b += cp * len(board.pieces(piece, chess.BLACK))
    return w - b


def canonical_position_key(board: chess.Board) -> int:
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    turn = "w" if board.turn == chess.WHITE else "b"
    base = f"{board.board_fen()} {turn} {board.castling_xfen()} {ep}"
    digest = hashlib.blake2b(base.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def position_is_candidate(board: chess.Board, cfg: Config, ply: int) -> bool:
    if ply < cfg.min_ply or ply > cfg.max_ply:
        return False
    if ((ply - cfg.min_ply) % cfg.sample_every) != 0:
        return False
    if board.is_game_over():
        return False
    if board.is_check():
        return False
    if abs(material_balance_cp(board)) > cfg.material_window_cp:
        return False
    return True


def eval_white_cp(engine: chess.engine.SimpleEngine, board: chess.Board, movetime_ms: int) -> Optional[int]:
    try:
        info = engine.analyse(board, chess.engine.Limit(time=max(0.001, movetime_ms / 1000.0)))
    except Exception:
        return None
    score = info.get("score")
    if score is None:
        return None
    cp = score.white().score(mate_score=20000)
    if cp is None:
        return None
    return int(cp)


def main() -> int:
    cfg = parse_args()
    if not cfg.pgn.exists():
        raise SystemExit(f"PGN not found: {cfg.pgn}")

    cfg.out.parent.mkdir(parents=True, exist_ok=True)

    engine = chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path)
    try:
        try:
            engine.configure({"Threads": 1, "Hash": 64})
        except Exception:
            pass

        selected: list[str] = []
        seen: set[int] = set()

        scanned_total = 0
        scanned_after_start = 0
        top_games_seen = 0
        eval_calls = 0

        progress = None
        if not cfg.quiet and tqdm is not None:
            progress = tqdm(total=cfg.count, desc="Collecting equal test positions", unit="pos", dynamic_ncols=True)

        with open_text_auto(cfg.pgn) as f:
            while len(selected) < cfg.count:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                scanned_total += 1
                if scanned_total < cfg.start_game_index:
                    continue

                scanned_after_start += 1
                if scanned_after_start > cfg.max_games_scan:
                    break

                w_elo = parse_elo(game.headers.get("WhiteElo", ""))
                b_elo = parse_elo(game.headers.get("BlackElo", ""))
                if w_elo is None or b_elo is None:
                    continue
                if w_elo < cfg.min_each_elo or b_elo < cfg.min_each_elo:
                    continue
                if (w_elo + b_elo) // 2 < cfg.min_avg_elo:
                    continue
                top_games_seen += 1

                board = game.board()
                ply = 0
                accepted_from_game = False

                for mv in game.mainline_moves():
                    board.push(mv)
                    ply += 1
                    if not position_is_candidate(board, cfg, ply):
                        continue

                    key = canonical_position_key(board)
                    if key in seen:
                        continue

                    cp = eval_white_cp(engine, board, cfg.movetime_ms)
                    eval_calls += 1
                    if cp is None:
                        continue
                    if abs(cp) > cfg.eval_window_cp:
                        continue

                    fen = board.fen()
                    selected.append(fen)
                    seen.add(key)
                    accepted_from_game = True
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix({
                            "games": scanned_after_start,
                            "top": top_games_seen,
                            "evals": eval_calls,
                        })
                    if len(selected) >= cfg.count:
                        break
                    # Keep one position per game for diversity.
                    break

                if not accepted_from_game:
                    continue

        if progress is not None:
            progress.close()

        if not selected:
            raise SystemExit(
                "No positions found. Try lowering --min-avg-elo or increasing --eval-window-cp / --max-games-scan."
            )

        with cfg.out.open("w", encoding="utf-8") as out_f:
            out_f.write(
                "# AI test-lab positions from local PGN (high-Elo + Stockfish-equal)\n"
                f"# source={cfg.pgn}\n"
                f"# start_game_index={cfg.start_game_index} max_games_scan={cfg.max_games_scan} "
                f"min_avg_elo={cfg.min_avg_elo} min_each_elo={cfg.min_each_elo} "
                f"eval_window_cp={cfg.eval_window_cp} movetime_ms={cfg.movetime_ms}\n"
            )
            for fen in selected:
                out_f.write(fen + "\n")

        if not cfg.quiet:
            print(
                f"Wrote {len(selected)} positions to {cfg.out} "
                f"(games_scanned={scanned_after_start}, top_games={top_games_seen}, eval_calls={eval_calls})",
                flush=True,
            )

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
