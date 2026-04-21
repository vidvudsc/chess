#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.engine


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENGINE_BIN = REPO_ROOT / "bin" / "chess_uci"
DEFAULT_POSITIONS = REPO_ROOT / "data" / "positions" / "lichess_equal_positions.fen"


@dataclass
class CompetitorSpec:
    name: str
    source_kind: str
    source_value: str
    uci_options: Dict[str, str]


@dataclass
class CompetitorBuild:
    name: str
    source_kind: str
    source_value: str
    engine_path: str
    temp_dir: Optional[str]
    uci_options: Dict[str, str]


@dataclass
class GameRecord:
    index: int
    round_name: str
    white: str
    black: str
    start_fen: str
    result: str
    winner: str
    plies: int
    termination: str
    moves_uci: List[str]
    elapsed_ms: Dict[str, int]


@dataclass
class ScheduledGame:
    round_name: str
    white: str
    black: str
    start_fen: str


def log(message: str) -> None:
    print(message, flush=True)


def parse_named_value(text: str, flag: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"{flag} requires NAME=VALUE")
    name, value = text.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{flag} requires NAME=VALUE")
    return name, value


def parse_named_uci_option(text: str) -> tuple[str, str, str]:
    if ":" not in text or "=" not in text:
        raise argparse.ArgumentTypeError("--uci-option requires NAME:OPTION=VALUE")
    competitor_name, rest = text.split(":", 1)
    option_name, value = rest.split("=", 1)
    competitor_name = competitor_name.strip()
    option_name = option_name.strip()
    value = value.strip()
    if not competitor_name or not option_name:
        raise argparse.ArgumentTypeError("--uci-option requires NAME:OPTION=VALUE")
    return competitor_name, option_name, value


def load_fens(path: Path) -> List[str]:
    fens: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.count(" ") == 3:
                line = f"{line} 0 1"
            fens.append(line)
    if not fens:
        raise RuntimeError(f"no positions loaded from {path}")
    return fens


def run_checked(cmd: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def build_worktree_engine() -> Path:
    run_checked(["make", "-j4", str(DEFAULT_ENGINE_BIN)])
    engine_path = DEFAULT_ENGINE_BIN
    if not engine_path.exists():
        raise RuntimeError(f"worktree engine missing after build: {engine_path}")
    return engine_path


def build_git_ref_engine(name: str, git_ref: str) -> CompetitorBuild:
    temp_root = Path(tempfile.mkdtemp(prefix=f"hce_test_lab_{name}_"))
    archive_cmd = [
        "sh",
        "-c",
        f"git archive --format=tar {git_ref} src/core/engine | tar -xf - -C {shlex_quote(str(temp_root))}",
    ]
    run_checked(archive_cmd, cwd=REPO_ROOT)
    engine_dir = temp_root / "src" / "core" / "engine"
    if not engine_dir.exists():
        raise RuntimeError(f"failed to extract engine subtree for {git_ref}")
    run_checked(["make", "-C", str(engine_dir), "-j4"])
    engine_path = engine_dir / "chess_uci"
    if not engine_path.exists():
        raise RuntimeError(f"git ref engine missing after build: {engine_path}")
    return CompetitorBuild(
        name=name,
        source_kind="git_ref",
        source_value=git_ref,
        engine_path=str(engine_path),
        temp_dir=str(temp_root),
        uci_options={},
    )


def resolve_path_engine(name: str, raw_path: str) -> CompetitorBuild:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        candidate = path / "chess_uci"
    else:
        candidate = path
    if not candidate.exists():
        raise RuntimeError(f"engine path not found for {name}: {candidate}")
    return CompetitorBuild(
        name=name,
        source_kind="path",
        source_value=str(path),
        engine_path=str(candidate),
        temp_dir=None,
        uci_options={},
    )


def prepare_competitors(specs: List[CompetitorSpec]) -> List[CompetitorBuild]:
    builds: List[CompetitorBuild] = []
    worktree_built = False
    worktree_path: Optional[Path] = None
    for spec in specs:
        if spec.source_kind == "worktree":
            if not worktree_built:
                worktree_path = build_worktree_engine()
                worktree_built = True
            assert worktree_path is not None
            builds.append(
                CompetitorBuild(
                    name=spec.name,
                    source_kind="worktree",
                    source_value="worktree",
                    engine_path=str(worktree_path),
                    temp_dir=None,
                    uci_options=dict(spec.uci_options),
                )
            )
        elif spec.source_kind == "git_ref":
            build = build_git_ref_engine(spec.name, spec.source_value)
            build.uci_options = dict(spec.uci_options)
            builds.append(build)
        elif spec.source_kind == "path":
            build = resolve_path_engine(spec.name, spec.source_value)
            build.uci_options = dict(spec.uci_options)
            builds.append(build)
        else:
            raise RuntimeError(f"unknown source kind: {spec.source_kind}")
    return builds


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def configure_engine(engine: chess.engine.SimpleEngine,
                     competitor: CompetitorBuild,
                     move_time_ms: int,
                     max_depth: int,
                     book_file: str) -> None:
    options: Dict[str, object] = {}
    user_options: Dict[str, object] = {}
    option_lookup = {name.lower(): name for name in engine.options}

    for raw_name, raw_value in competitor.uci_options.items():
        resolved_name = option_lookup.get(raw_name.lower(), raw_name)
        option_def = engine.options.get(resolved_name)
        if option_def is None:
            raise RuntimeError(f"{competitor.name}: engine option not found: {raw_name}")
        try:
            user_options[resolved_name] = option_def.parse(raw_value)
        except Exception as exc:
            raise RuntimeError(f"{competitor.name}: failed to parse {raw_name}={raw_value!r}: {exc}") from exc

    if "NNModel" in user_options:
        options["NNModel"] = user_options.pop("NNModel")
    if "Backend" in user_options:
        options["Backend"] = user_options.pop("Backend")
    for name, value in user_options.items():
        options[name] = value
    if "MoveTime" in engine.options:
        options["MoveTime"] = max(1, move_time_ms)
    if max_depth > 0 and "MaxDepth" in engine.options:
        options["MaxDepth"] = max_depth
    if book_file and "BookFile" in engine.options:
        options["BookFile"] = book_file
    if options:
        engine.configure(options)


def open_configured_engine(competitor: CompetitorBuild,
                           think_ms: int,
                           max_depth: int,
                           book_file: str) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(competitor.engine_path)
    configure_engine(engine, competitor, think_ms, max_depth, book_file)
    return engine


def score_to_elo(score_fraction: float) -> Optional[float]:
    if score_fraction <= 0.0 or score_fraction >= 1.0:
        return None
    return -400.0 * math.log10((1.0 / score_fraction) - 1.0)


def score_to_elo_clamped(score_fraction: float) -> float:
    score_fraction = min(max(score_fraction, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10((1.0 / score_fraction) - 1.0)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def score_for_player(record: GameRecord, player_name: str) -> float:
    if record.result == "1-0":
        return 1.0 if record.white == player_name else 0.0
    if record.result == "0-1":
        return 1.0 if record.black == player_name else 0.0
    return 0.5


def summarize_vs_baseline(records: List[GameRecord], baseline_name: str, candidate_name: str) -> Dict[str, object]:
    scores = [
        score_for_player(record, candidate_name)
        for record in records
        if {record.white, record.black} == {baseline_name, candidate_name}
    ]
    n = len(scores)
    if n == 0:
        return {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "games": 0,
            "score_fraction": 0.0,
            "points": 0.0,
            "elo_diff": None,
            "elo_ci_low": None,
            "elo_ci_high": None,
            "probability_better": None,
            "z_score": None,
            "p_value_one_sided": None,
        }

    mean = sum(scores) / n
    elo_diff = score_to_elo(mean)

    sample_var = 0.0
    se = None
    z_score = None
    probability_better = None
    p_value_one_sided = None
    elo_ci_low = None
    elo_ci_high = None
    if n >= 2:
        sample_var = sum((score - mean) ** 2 for score in scores) / (n - 1)
        se = math.sqrt(sample_var / n)
    if se is not None and se > 0.0:
        z_score = (mean - 0.5) / se
        probability_better = normal_cdf(z_score)
        p_value_one_sided = 1.0 - probability_better
        low = min(max(mean - 1.96 * se, 1e-6), 1.0 - 1e-6)
        high = min(max(mean + 1.96 * se, 1e-6), 1.0 - 1e-6)
        elo_ci_low = score_to_elo_clamped(low)
        elo_ci_high = score_to_elo_clamped(high)
    elif n > 0:
        if mean > 0.5:
            probability_better = 1.0
            p_value_one_sided = 0.0
        elif mean < 0.5:
            probability_better = 0.0
            p_value_one_sided = 1.0
        else:
            probability_better = 0.5
            p_value_one_sided = 0.5

    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "games": n,
        "points": sum(scores),
        "score_fraction": mean,
        "elo_diff": elo_diff,
        "elo_ci_low": elo_ci_low,
        "elo_ci_high": elo_ci_high,
        "probability_better": probability_better,
        "z_score": z_score,
        "p_value_one_sided": p_value_one_sided,
    }


def choose_termination(board: chess.Board, max_plies: int) -> tuple[str, str, str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = outcome.result()
        if result == "1-0":
            winner = "white"
        elif result == "0-1":
            winner = "black"
        else:
            winner = "draw"
        return result, winner, str(outcome.termination)
    if len(board.move_stack) >= max_plies:
        return "1/2-1/2", "draw", "max_plies"
    return "*", "ongoing", "ongoing"


def play_one_game(game_index: int,
                  round_name: str,
                  white_name: str,
                  black_name: str,
                  white_engine: chess.engine.SimpleEngine,
                  black_engine: chess.engine.SimpleEngine,
                  start_fen: str,
                  think_ms: int,
                  max_depth: int,
                  max_plies: int) -> GameRecord:
    board = chess.Board(start_fen)
    move_limit = chess.engine.Limit(time=max(0.001, think_ms / 1000.0))
    if max_depth > 0:
        move_limit = chess.engine.Limit(time=max(0.001, think_ms / 1000.0), depth=max_depth)

    moves_uci: List[str] = []
    elapsed_ms = {"white": 0, "black": 0}

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None or len(board.move_stack) >= max_plies:
            break

        actor_name = white_name if board.turn == chess.WHITE else black_name
        actor_engine = white_engine if board.turn == chess.WHITE else black_engine
        actor_key = "white" if board.turn == chess.WHITE else "black"
        started = time.perf_counter()
        try:
            result = actor_engine.play(board, move_limit)
        except chess.engine.EngineTerminatedError as exc:
            elapsed_ms[actor_key] += int((time.perf_counter() - started) * 1000.0)
            winner = "black" if board.turn == chess.WHITE else "white"
            result_text = "0-1" if winner == "black" else "1-0"
            return GameRecord(
                index=game_index,
                round_name=round_name,
                white=white_name,
                black=black_name,
                start_fen=start_fen,
                result=result_text,
                winner=winner,
                plies=len(board.move_stack),
                termination=f"engine_terminated_by_{actor_name}: {type(exc).__name__}: {exc}",
                moves_uci=moves_uci,
                elapsed_ms=elapsed_ms,
            )
        except chess.engine.EngineError as exc:
            elapsed_ms[actor_key] += int((time.perf_counter() - started) * 1000.0)
            winner = "black" if board.turn == chess.WHITE else "white"
            result_text = "0-1" if winner == "black" else "1-0"
            return GameRecord(
                index=game_index,
                round_name=round_name,
                white=white_name,
                black=black_name,
                start_fen=start_fen,
                result=result_text,
                winner=winner,
                plies=len(board.move_stack),
                termination=f"engine_error_by_{actor_name}: {type(exc).__name__}: {exc}",
                moves_uci=moves_uci,
                elapsed_ms=elapsed_ms,
            )
        elapsed_ms[actor_key] += int((time.perf_counter() - started) * 1000.0)

        if result.move is None or result.move not in board.legal_moves:
            winner = "black" if board.turn == chess.WHITE else "white"
            result_text = "0-1" if winner == "black" else "1-0"
            return GameRecord(
                index=game_index,
                round_name=round_name,
                white=white_name,
                black=black_name,
                start_fen=start_fen,
                result=result_text,
                winner=winner,
                plies=len(board.move_stack),
                termination=f"illegal_or_missing_move_by_{actor_name}",
                moves_uci=moves_uci,
                elapsed_ms=elapsed_ms,
            )

        board.push(result.move)
        moves_uci.append(result.move.uci())

    result_text, winner, termination = choose_termination(board, max_plies)
    return GameRecord(
        index=game_index,
        round_name=round_name,
        white=white_name,
        black=black_name,
        start_fen=start_fen,
        result=result_text,
        winner=winner,
        plies=len(board.move_stack),
        termination=termination,
        moves_uci=moves_uci,
        elapsed_ms=elapsed_ms,
    )


def build_schedule(competitors: List[CompetitorBuild],
                   positions: List[str],
                   games_per_pair: int,
                   positions_count: int,
                   paired_colors: bool,
                   seed: int) -> List[ScheduledGame]:
    random_gen = random.Random(seed)
    shuffled_positions = positions[:]
    random_gen.shuffle(shuffled_positions)
    if not shuffled_positions:
        raise RuntimeError("no positions available")

    schedule: List[ScheduledGame] = []
    position_index = 0

    def next_position() -> str:
        nonlocal position_index, shuffled_positions
        if position_index >= len(shuffled_positions):
            shuffled_positions = positions[:]
            random_gen.shuffle(shuffled_positions)
            position_index = 0
        fen = shuffled_positions[position_index]
        position_index += 1
        return fen

    for i in range(len(competitors)):
        for j in range(i + 1, len(competitors)):
            a = competitors[i]
            b = competitors[j]
            round_name = f"{a.name} vs {b.name}"

            if positions_count > 0:
                for _ in range(positions_count):
                    fen = next_position()
                    schedule.append(
                        ScheduledGame(
                            round_name=round_name,
                            white=a.name,
                            black=b.name,
                            start_fen=fen,
                        )
                    )
                    if paired_colors:
                        schedule.append(
                            ScheduledGame(
                                round_name=round_name,
                                white=b.name,
                                black=a.name,
                                start_fen=fen,
                            )
                        )
            else:
                for game_no in range(games_per_pair):
                    if game_no % 2 == 0:
                        white = a.name
                        black = b.name
                    else:
                        white = b.name
                        black = a.name
                    schedule.append(
                        ScheduledGame(
                            round_name=round_name,
                            white=white,
                            black=black,
                            start_fen=next_position(),
                        )
                    )

    return schedule


def run_lab(competitors: List[CompetitorBuild],
            positions: List[str],
            games_per_pair: int,
            positions_count: int,
            paired_colors: bool,
            think_ms: int,
            max_depth: int,
            max_plies: int,
            book_file: str,
            seed: int,
            baseline_name: Optional[str],
            restart_each_game: bool) -> Dict[str, object]:
    engines: Dict[str, chess.engine.SimpleEngine] = {}
    records: List[GameRecord] = []
    standings: Dict[str, Dict[str, float]] = {}
    competitor_by_name = {competitor.name: competitor for competitor in competitors}
    schedule = build_schedule(
        competitors=competitors,
        positions=positions,
        games_per_pair=games_per_pair,
        positions_count=positions_count,
        paired_colors=paired_colors,
        seed=seed,
    )

    try:
        for competitor in competitors:
            standings[competitor.name] = {
                "games": 0.0,
                "points": 0.0,
                "wins": 0.0,
                "draws": 0.0,
                "losses": 0.0,
                "engine_failures": 0.0,
            }
        if not restart_each_game:
            for competitor in competitors:
                engine = open_configured_engine(competitor, think_ms, max_depth, book_file)
                engines[competitor.name] = engine

        game_index = 0
        current_round = None
        for scheduled in schedule:
            if scheduled.round_name != current_round:
                current_round = scheduled.round_name
                round_games = sum(1 for item in schedule if item.round_name == current_round)
                log(f"[pairing] {current_round} ({round_games} games)")

            game_index += 1
            if restart_each_game:
                white_engine = None
                black_engine = None
                try:
                    white_engine = open_configured_engine(competitor_by_name[scheduled.white], think_ms, max_depth, book_file)
                    black_engine = open_configured_engine(competitor_by_name[scheduled.black], think_ms, max_depth, book_file)
                    record = play_one_game(
                        game_index=game_index,
                        round_name=scheduled.round_name,
                        white_name=scheduled.white,
                        black_name=scheduled.black,
                        white_engine=white_engine,
                        black_engine=black_engine,
                        start_fen=scheduled.start_fen,
                        think_ms=think_ms,
                        max_depth=max_depth,
                        max_plies=max_plies,
                    )
                finally:
                    for engine in (white_engine, black_engine):
                        if engine is None:
                            continue
                        try:
                            engine.quit()
                        except Exception:
                            pass
            else:
                record = play_one_game(
                    game_index=game_index,
                    round_name=scheduled.round_name,
                    white_name=scheduled.white,
                    black_name=scheduled.black,
                    white_engine=engines[scheduled.white],
                    black_engine=engines[scheduled.black],
                    start_fen=scheduled.start_fen,
                    think_ms=think_ms,
                    max_depth=max_depth,
                    max_plies=max_plies,
                )
            records.append(record)

            white_row = standings[record.white]
            black_row = standings[record.black]
            white_row["games"] += 1.0
            black_row["games"] += 1.0
            if record.result == "1-0":
                white_row["points"] += 1.0
                white_row["wins"] += 1.0
                black_row["losses"] += 1.0
            elif record.result == "0-1":
                black_row["points"] += 1.0
                black_row["wins"] += 1.0
                white_row["losses"] += 1.0
            else:
                white_row["points"] += 0.5
                black_row["points"] += 0.5
                white_row["draws"] += 1.0
                black_row["draws"] += 1.0

            if record.termination.startswith("engine_error_by_") or record.termination.startswith("engine_terminated_by_") or record.termination.startswith("illegal_or_missing_move_by_"):
                if record.termination.endswith(record.white) or f"by_{record.white}" in record.termination:
                    white_row["engine_failures"] += 1.0
                elif record.termination.endswith(record.black) or f"by_{record.black}" in record.termination:
                    black_row["engine_failures"] += 1.0

            log(
                f"[game {record.index}] {record.white} vs {record.black} "
                f"result={record.result} plies={record.plies} term={record.termination}"
            )
    finally:
        for engine in engines.values():
            try:
                engine.quit()
            except Exception:
                pass

    summary_rows = []
    total_games = max(1, len(records))
    for competitor in competitors:
        row = standings[competitor.name]
        score_fraction = row["points"] / row["games"] if row["games"] > 0.0 else 0.0
        elo = score_to_elo(score_fraction)
        summary_rows.append(
            {
                "name": competitor.name,
                "games": int(row["games"]),
                "points": row["points"],
                "wins": int(row["wins"]),
                "draws": int(row["draws"]),
                "losses": int(row["losses"]),
                "engine_failures": int(row["engine_failures"]),
                "score_fraction": score_fraction,
                "elo_estimate": elo,
                "share_of_total_points": row["points"] / total_games,
            }
        )

    summary_rows.sort(key=lambda item: (-item["points"], -item["wins"], item["name"]))

    baseline_summary = None
    head_to_head = []
    if baseline_name is not None:
        baseline_summary = {
            "name": baseline_name,
            "elo_reference": 0.0,
        }
        for competitor in competitors:
            if competitor.name == baseline_name:
                continue
            head_to_head.append(summarize_vs_baseline(records, baseline_name, competitor.name))

    return {
        "generated_at_unix": int(time.time()),
        "think_ms": think_ms,
        "max_depth": max_depth,
        "max_plies": max_plies,
        "seed": seed,
        "games_per_pair": games_per_pair,
        "positions_count": positions_count,
        "paired_colors": paired_colors,
        "total_games": len(records),
        "positions_file": str(DEFAULT_POSITIONS),
        "competitors": [asdict(competitor) for competitor in competitors],
        "baseline": baseline_summary,
        "head_to_head": head_to_head,
        "standings": summary_rows,
        "games": [asdict(record) for record in records],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-to-head or round-robin lab for HCE engine versions.")
    parser.add_argument("--include-worktree", metavar="NAME", action="append", default=[],
                        help="Build and include the current worktree engine under NAME.")
    parser.add_argument("--git-ref", metavar="NAME=REF", action="append", default=[],
                        help="Build and include src/core/engine from git REF.")
    parser.add_argument("--engine", metavar="NAME=PATH", action="append", default=[],
                        help="Include a prebuilt engine binary or engine directory.")
    parser.add_argument("--uci-option", metavar="NAME:OPTION=VALUE", action="append", default=[],
                        help="Apply a competitor-specific UCI option, for example nn:Backend=nn or nn:NNModel=/abs/model.bin.")
    parser.add_argument("--games", type=int, default=6, help="Games per pairing.")
    parser.add_argument("--positions-count", type=int, default=0,
                        help="Use this many FEN positions per pairing. When set, the lab samples from the FEN file instead of using --games directly.")
    parser.add_argument("--paired-colors", dest="paired_colors", action=argparse.BooleanOptionalAction, default=True,
                        help="When using --positions-count, play each FEN twice with colors swapped.")
    parser.add_argument("--baseline", default="",
                        help="Competitor name to anchor at Elo 0 in head-to-head reporting. Defaults to the first competitor when there are exactly two.")
    parser.add_argument("--think-ms", type=int, default=120, help="Per-move think time for each engine.")
    parser.add_argument("--max-depth", type=int, default=0, help="Optional depth cap (0 = engine default).")
    parser.add_argument("--max-plies", type=int, default=160, help="Draw the game if this ply limit is reached.")
    parser.add_argument("--positions-file", default=str(DEFAULT_POSITIONS), help="FEN list to sample from.")
    parser.add_argument("--book-file", default="", help="Book file to pass to engines. Empty disables it.")
    parser.add_argument("--seed", type=int, default=20260305, help="PRNG seed for position order.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--restart-each-game", action="store_true",
                        help="Launch fresh engine processes for every game instead of reusing long-lived ones.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    specs: List[CompetitorSpec] = []
    uci_options_by_name: Dict[str, Dict[str, str]] = {}
    for item in args.uci_option:
        competitor_name, option_name, value = parse_named_uci_option(item)
        uci_options_by_name.setdefault(competitor_name, {})[option_name] = value
    for name in args.include_worktree:
        specs.append(CompetitorSpec(name=name, source_kind="worktree", source_value="worktree",
                                    uci_options=dict(uci_options_by_name.get(name, {}))))
    for item in args.git_ref:
        name, value = parse_named_value(item, "--git-ref")
        specs.append(CompetitorSpec(name=name, source_kind="git_ref", source_value=value,
                                    uci_options=dict(uci_options_by_name.get(name, {}))))
    for item in args.engine:
        name, value = parse_named_value(item, "--engine")
        specs.append(CompetitorSpec(name=name, source_kind="path", source_value=value,
                                    uci_options=dict(uci_options_by_name.get(name, {}))))

    if len(specs) == 0:
        specs = [
            CompetitorSpec(name="baseline", source_kind="git_ref", source_value="HEAD", uci_options={}),
            CompetitorSpec(name="worktree", source_kind="worktree", source_value="worktree", uci_options={}),
        ]
    elif len(specs) < 2:
        raise SystemExit("need at least two competitors (use --include-worktree, --git-ref, or --engine)")

    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise SystemExit("competitor names must be unique")
    unknown_option_names = sorted(set(uci_options_by_name) - set(names))
    if unknown_option_names:
        raise SystemExit(f"--uci-option references unknown competitors: {', '.join(unknown_option_names)}")

    baseline_name = args.baseline.strip() or (specs[0].name if len(specs) == 2 else "")
    if baseline_name:
        if baseline_name not in names:
            raise SystemExit(f"--baseline references unknown competitor: {baseline_name}")
    else:
        baseline_name = None

    positions_path = Path(args.positions_file).expanduser()
    competitors = prepare_competitors(specs)
    positions = load_fens(positions_path)
    positions_count = max(0, args.positions_count)
    if positions_count == 0 and len(competitors) == 2:
        positions_count = len(positions)
    report = run_lab(
        competitors=competitors,
        positions=positions,
        games_per_pair=max(1, args.games),
        positions_count=positions_count,
        paired_colors=bool(args.paired_colors),
        think_ms=max(1, args.think_ms),
        max_depth=max(0, args.max_depth),
        max_plies=max(40, args.max_plies),
        book_file=args.book_file,
        seed=args.seed,
        baseline_name=baseline_name,
        restart_each_game=bool(args.restart_each_game),
    )
    report["positions_file"] = str(positions_path)

    print("\nStandings:")
    for row in report["standings"]:
        elo_text = "n/a" if row["elo_estimate"] is None else f"{row['elo_estimate']:+.1f}"
        print(
            f"  {row['name']:<16} pts={row['points']:.1f}/{row['games']} "
            f"w={row['wins']} d={row['draws']} l={row['losses']} elo={elo_text}"
        )

    if report["head_to_head"]:
        print("\nHead-to-head vs baseline:")
        for row in report["head_to_head"]:
            elo_text = "n/a" if row["elo_diff"] is None else f"{row['elo_diff']:+.1f}"
            ci_text = "n/a"
            if row["elo_ci_low"] is not None and row["elo_ci_high"] is not None:
                ci_text = f"[{row['elo_ci_low']:+.1f}, {row['elo_ci_high']:+.1f}]"
            prob_text = "n/a"
            if row["probability_better"] is not None:
                prob_text = f"{row['probability_better'] * 100.0:.1f}%"
            print(
                f"  {row['candidate']:<16} vs {row['baseline']:<16} "
                f"score={row['points']:.1f}/{row['games']} "
                f"elo={elo_text} ci95={ci_text} P(better)={prob_text}"
            )

    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is None:
        out_dir = REPO_ROOT / "current"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hce_test_lab_{int(time.time())}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")
    print(f"\nReport: {out_path}")

    for competitor in competitors:
        if competitor.temp_dir:
            shutil.rmtree(competitor.temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
