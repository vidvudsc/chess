#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import chess


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.bot.run import BotRunner, LichessApi, default_opening_book_path, parse_speed_list  # noqa: E402


def assert_close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"expected {expected}, got {actual}")


def test_clock_pair_accepts_milliseconds_and_seconds() -> None:
    assert BotRunner._clock_pair_to_seconds({"initial": 60000, "increment": 0}) == (60.0, 0.0)
    assert BotRunner._clock_pair_to_seconds({"initial": 180000, "increment": 2000}) == (180.0, 2.0)
    assert BotRunner._clock_pair_to_seconds({"initial": 300, "increment": 3}) == (300.0, 3.0)


def test_game_state_preserves_true_zero_increment() -> None:
    wtime, btime, winc, binc = BotRunner._state_clock_seconds(
        {"wtime": 0, "btime": 60500, "winc": 0, "binc": 0},
        initial_s=60.0,
        increment_s=2.0,
    )

    assert_close(wtime, 0.0)
    assert_close(btime, 60.5)
    assert_close(winc, 0.0)
    assert_close(binc, 0.0)


def test_game_state_subsecond_clock_values_are_milliseconds() -> None:
    wtime, btime, winc, binc = BotRunner._state_clock_seconds(
        {"wtime": 500, "btime": 900, "winc": 500, "binc": 2000},
        initial_s=180.0,
        increment_s=2.0,
    )

    assert_close(wtime, 0.5)
    assert_close(btime, 0.9)
    assert_close(winc, 0.5)
    assert_close(binc, 2.0)


def test_game_state_uses_fallback_only_for_missing_values() -> None:
    wtime, btime, winc, binc = BotRunner._state_clock_seconds(
        {"btime": 301000, "binc": 3000},
        initial_s=300.0,
        increment_s=3.0,
    )

    assert_close(wtime, 300.0)
    assert_close(btime, 301.0)
    assert_close(winc, 3.0)
    assert_close(binc, 3.0)


def test_missing_env_opening_book_falls_back_to_packaged_book() -> None:
    old_value = os.environ.get("CHESS_OPENING_BOOK")
    os.environ["CHESS_OPENING_BOOK"] = "/definitely/missing/opening_book.txt"
    try:
        path = Path(default_opening_book_path(ROOT / "src" / "core" / "bot"))
    finally:
        if old_value is None:
            os.environ.pop("CHESS_OPENING_BOOK", None)
        else:
            os.environ["CHESS_OPENING_BOOK"] = old_value

    if not path.exists():
        raise AssertionError(f"expected fallback opening book to exist, got {path}")
    if path.name not in {"opening_book.txt", "opening_games_100.txt"}:
        raise AssertionError(f"unexpected fallback opening book: {path}")


def test_bullet_no_increment_budget_is_capped() -> None:
    runner = object.__new__(BotRunner)
    board = chess.Board()

    budget, remaining, increment = runner._compute_move_budget(
        board,
        {"wtime": 60000, "btime": 60000, "winc": 0, "binc": 0},
        chess.WHITE,
        initial_s=60.0,
        increment_s=0.0,
        base_default=0.35,
        active_bot_games=1,
    )

    assert_close(remaining, 60.0)
    assert_close(increment, 0.0)
    assert_close(budget, 0.35)


def test_blitz_increment_budget_respects_hard_cap() -> None:
    runner = object.__new__(BotRunner)
    board = chess.Board()

    budget, remaining, increment = runner._compute_move_budget(
        board,
        {"wtime": 180000, "btime": 180000, "winc": 2000, "binc": 2000},
        chess.WHITE,
        initial_s=180.0,
        increment_s=2.0,
        base_default=1.0,
        active_bot_games=1,
    )

    assert_close(remaining, 180.0)
    assert_close(increment, 2.0)
    assert_close(budget, 1.5)


def test_concurrency_scales_time_budget() -> None:
    runner = object.__new__(BotRunner)
    board = chess.Board()
    state = {"wtime": 180000, "btime": 180000, "winc": 2000, "binc": 2000}

    solo_budget, _, _ = runner._compute_move_budget(
        board,
        state,
        chess.WHITE,
        initial_s=180.0,
        increment_s=2.0,
        base_default=1.0,
        active_bot_games=1,
    )
    busy_budget, _, _ = runner._compute_move_budget(
        board,
        state,
        chess.WHITE,
        initial_s=180.0,
        increment_s=2.0,
        base_default=1.0,
        active_bot_games=3,
    )

    if not busy_budget < solo_budget:
        raise AssertionError(f"expected concurrency to lower budget, got solo={solo_budget} busy={busy_budget}")
    assert_close(busy_budget, 0.9)


def test_low_clock_panic_budget_stays_small() -> None:
    runner = object.__new__(BotRunner)
    board = chess.Board()

    budget, remaining, increment = runner._compute_move_budget(
        board,
        {"wtime": 800, "btime": 800, "winc": 500, "binc": 500},
        chess.WHITE,
        initial_s=180.0,
        increment_s=0.5,
        base_default=0.25,
        active_bot_games=1,
    )

    assert_close(remaining, 0.8)
    assert_close(increment, 0.5)
    if not 0.05 <= budget <= 0.225:
        raise AssertionError(f"panic budget outside expected range: {budget}")


def test_dead_game_stream_errors_do_not_retry_forever() -> None:
    assert LichessApi._should_stop_stream_retry("/api/bot/game/stream/abc123", 404)
    assert LichessApi._should_stop_stream_retry("/api/bot/game/stream/abc123", 410)
    assert not LichessApi._should_stop_stream_retry("/api/bot/game/stream/abc123", 500)
    assert not LichessApi._should_stop_stream_retry("/api/stream/event", 404)


def test_incoming_speed_filter_defaults_to_all_speeds() -> None:
    assert BotRunner._incoming_speed_allowed("bullet", [])
    assert BotRunner._incoming_speed_allowed("ultraBullet", [])


def test_incoming_speed_filter_limits_fast_challenges() -> None:
    speeds = parse_speed_list("blitz, rapid, blitz")

    assert speeds == ["blitz", "rapid"]
    assert BotRunner._incoming_speed_allowed("blitz", speeds)
    assert BotRunner._incoming_speed_allowed("rapid", speeds)
    assert not BotRunner._incoming_speed_allowed("bullet", speeds)
    assert not BotRunner._incoming_speed_allowed("ultraBullet", speeds)


def main() -> None:
    test_clock_pair_accepts_milliseconds_and_seconds()
    test_game_state_preserves_true_zero_increment()
    test_game_state_subsecond_clock_values_are_milliseconds()
    test_game_state_uses_fallback_only_for_missing_values()
    test_missing_env_opening_book_falls_back_to_packaged_book()
    test_bullet_no_increment_budget_is_capped()
    test_blitz_increment_budget_respects_hard_cap()
    test_concurrency_scales_time_budget()
    test_low_clock_panic_budget_stays_small()
    test_dead_game_stream_errors_do_not_retry_forever()
    test_incoming_speed_filter_defaults_to_all_speeds()
    test_incoming_speed_filter_limits_fast_challenges()
    print("test_bot_time_management: OK")


if __name__ == "__main__":
    main()
