"""Offline reproductions of finished games leaving engines/slots behind."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import chess
import requests

from src.core.bot.run import ActiveGame, BotRunner, LichessApi


def runner():
    bot = object.__new__(BotRunner)
    bot.lock = threading.Lock()
    bot.active_games = {}
    bot.finished_games = {}
    bot.pending_slots = {}
    bot.bot_cooldowns = {}
    bot.cfg = SimpleNamespace(max_games=3, max_bot_games=2, max_human_games=1)
    return bot


def test_finish_cancels_stream_closes_engine_and_releases_slot():
    bot = runner()
    game = ActiveGame(threading.Thread(), "bot", engine=Mock())
    bot.active_games["finished"] = game
    bot._mark_game_done("finished", "event")
    assert game.stop.is_set()
    game.engine.close.assert_called_once()
    assert not bot.active_games
    bot._mark_game_done("finished", "thread", expected=game)
    game.engine.close.assert_called_once()
    # A stale account snapshot/event cannot restart the completed game.
    bot._start_game_thread("finished")
    assert not bot.active_games


def test_old_worker_cannot_remove_replacement():
    bot = runner()
    old = ActiveGame(threading.Thread(), "bot")
    new = ActiveGame(threading.Thread(), "bot", engine=Mock())
    bot.active_games["same"] = new
    bot._mark_game_done("same", expected=old)
    assert bot.active_games["same"] is new
    new.engine.close.assert_not_called()


def test_reconcile_recovers_missing_finish_without_killing_new_games():
    bot = runner()
    stale = ActiveGame(threading.Thread(), "bot", engine=Mock(),
                       started_at=time.monotonic() - 120)
    fresh = ActiveGame(threading.Thread(), "human", engine=Mock())
    live = ActiveGame(threading.Thread(), "bot", engine=Mock())
    bot.active_games = {"stale": stale, "fresh": fresh, "live": live}
    bot.api = Mock()
    bot.api.get_json.return_value = {"nowPlaying": [{"gameId": "live"}]}
    bot._reconcile_games()
    assert set(bot.active_games) == {"fresh", "live"}
    stale.engine.close.assert_called_once()
    fresh.engine.close.assert_not_called()
    live.engine.close.assert_not_called()


def test_bad_snapshot_and_network_error_preserve_games():
    bot = runner()
    game = ActiveGame(threading.Thread(), "bot", engine=Mock(), started_at=0)
    bot.active_games["live"] = game
    bot.api = Mock()
    for payload in [{}, {"nowPlaying": None}, {"nowPlaying": [{}]}]:
        bot.api.get_json.return_value = payload
        try:
            bot._reconcile_games()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid snapshot accepted")
    bot.api.get_json.side_effect = requests.ConnectionError("offline")
    try:
        bot._reconcile_games()
    except requests.ConnectionError:
        pass
    assert bot.active_games["live"] is game
    game.engine.close.assert_not_called()


def test_game_start_during_snapshot_is_not_removed():
    bot = runner()
    game = ActiveGame(threading.Thread(), "bot", engine=Mock(), started_at=0)
    def fetch(path):
        bot.active_games["new"] = game
        return {"nowPlaying": []}
    bot.api = SimpleNamespace(get_json=fetch)
    bot._reconcile_games()
    assert bot.active_games["new"] is game


def test_reconcile_resumes_game_with_correct_human_slot(monkeypatch):
    bot = runner()
    bot.api = Mock()
    bot.api.get_json.return_value = {"nowPlaying": [
        {"gameId": "resumed", "opponent": {"username": "human"}, "speed": "blitz"}
    ]}
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    bot._reconcile_games()
    assert bot.active_games["resumed"].slot_kind == "human"
    assert bot.active_games["resumed"].target == "human"


def test_stream_stops_on_keepalive_and_closes_response():
    api = LichessApi("fake", "https://example.invalid")
    stop = threading.Event()
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    def lines(**kwargs):
        assert kwargs["chunk_size"] == 1
        yield '{"type":"gameState"}'
        stop.set()
        yield ""
    response.iter_lines.side_effect = lines
    api._sessions.session = Mock()
    api.session.get.return_value = response
    assert list(api.stream_events("/api/bot/game/stream/x", stop=stop)) == [{"type": "gameState"}]
    response.__exit__.assert_called_once()
    api.session.get.assert_called_once()


def test_cancelled_stream_never_connects():
    api = LichessApi("fake", "https://example.invalid")
    api._sessions.session = Mock()
    stop = threading.Event()
    stop.set()
    assert list(api.stream_events("/api/bot/game/stream/x", stop=stop)) == []
    api.session.get.assert_not_called()


def test_stream_eof_backs_off_instead_of_busy_looping():
    api = LichessApi("fake", "https://example.invalid")
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.return_value = True
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_lines.return_value = iter([])
    api._sessions.session = Mock()
    api.session.get.return_value = response
    assert list(api.stream_events("/api/bot/game/stream/x", stop=stop)) == []
    stop.wait.assert_called_once_with(2.0)
    api.session.get.assert_called_once()


def test_stream_retry_backoff_is_cancellable():
    api = LichessApi("fake", "https://example.invalid")
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.return_value = True
    api._sessions.session = Mock()
    api.session.get.side_effect = requests.ConnectionError("disconnected")
    assert list(api.stream_events("/api/bot/game/stream/x", stop=stop)) == []
    stop.wait.assert_called_once_with(2.0)


def test_worker_closes_real_engine_when_game_stream_fails(monkeypatch):
    from pathlib import Path
    import pytest
    from src.core.bot import run
    engine_path = Path(__file__).resolve().parents[1] / "bin/chess_uci"
    bot = runner()
    bot.cfg = SimpleNamespace(token="fake", base_url="https://example.invalid",
                              engine_path=str(engine_path), think_time=0.1,
                              max_games=3, max_bot_games=2, max_human_games=1)
    bot._configure_engine = lambda engine, game_id: 1
    game = ActiveGame(threading.Thread(), "bot")
    bot.active_games["x"] = game
    api = Mock()
    api.stream_events.side_effect = RuntimeError("broken stream")
    monkeypatch.setattr(run, "LichessApi", lambda *args: api)
    with pytest.raises(RuntimeError, match="broken stream"):
        bot._play_game("x")
    assert game.engine.returncode.result(timeout=2) is not None
    assert "x" not in bot.active_games
    api.session.close.assert_called_once()


def test_external_finish_interrupts_real_pondering_engine():
    from pathlib import Path
    bot = runner()
    path = Path(__file__).resolve().parents[1] / "bin/chess_uci"
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    try:
        engine.configure({"BookFile": ""})
        game = ActiveGame(threading.Thread(), "bot", engine=engine)
        bot.active_games["ponder"] = game
        engine.analysis(chess.Board(), chess.engine.Limit(time=60))
        bot._mark_game_done("ponder", "event")
        assert engine.returncode.result(timeout=2) is not None
        assert game.stop.is_set()
        assert not bot.active_games
    finally:
        engine.close()


def test_worker_forces_close_if_quit_fails(monkeypatch):
    from src.core.bot import run
    bot = runner()
    bot.cfg = SimpleNamespace(token="fake", base_url="https://example.invalid",
                              engine_path="fake", think_time=0.1,
                              max_games=3, max_bot_games=2, max_human_games=1)
    bot._configure_engine = lambda engine, game_id: 1
    game = ActiveGame(threading.Thread(), "bot")
    bot.active_games["x"] = game
    engine = Mock()
    engine.quit.side_effect = TimeoutError("hung quit")
    api = Mock()
    api.stream_events.return_value = iter([])
    monkeypatch.setattr(run, "LichessApi", lambda *args: api)
    monkeypatch.setattr(run.chess.engine.SimpleEngine, "popen_uci", lambda *args: engine)
    bot._play_game("x")
    engine.close.assert_called()
    assert not bot.active_games


def test_hce_long_control_spends_available_time_and_keeps_reserve():
    bot = runner()
    board = chess.Board("8/1pr2kp1/1R5p/4PP1P/8/4K3/8/8 b - - 0 43")
    budget, _, _ = bot._compute_hce_move_budget(
        board, {"btime": 1619820, "binc": 5000}, chess.BLACK, 1800, 5, 2.5, 2)
    assert 30 < budget <= 54
    for initial, increment in [(60, 0), (180, 2), (300, 3), (600, 0), (1800, 5)]:
        remaining = float(initial)
        for _ in range(150):
            budget, _, _ = bot._compute_hce_move_budget(
                board, {"btime": remaining * 1000, "binc": increment * 1000},
                chess.BLACK, initial, increment, 0.35, 1)
            remaining += increment - budget - 0.03
            if increment or initial >= 600:
                assert remaining > 0, (initial, increment, remaining)


def test_hce_panic_policy_matches_existing_policy():
    bot = runner()
    for clock in [0, 0.1, 0.8, 2, 5, 10]:
        args = (chess.Board(), {"wtime": clock * 1000, "winc": 500},
                chess.WHITE, 180, 0.5, 0.35, 2)
        assert bot._compute_hce_move_budget(*args) == bot._compute_move_budget(*args)


def test_hce_reserves_clock_with_network_overhead():
    bot = runner()
    board = chess.Board("8/1pr2kp1/1R5p/4PP1P/8/4K3/8/8 b - - 0 43")
    for initial, increment in [(180, 2), (300, 3), (600, 0), (600, 10), (1800, 5)]:
        remaining = float(initial)
        for _ in range(100):
            budget, _, _ = bot._compute_hce_move_budget(
                board, {"btime": remaining * 1000, "binc": increment * 1000},
                chess.BLACK, initial, increment, 0.35, 1)
            remaining += increment - budget - 0.25
        assert remaining > 2.0, (initial, increment, remaining)


def _stall_bot(monkeypatch, clock):
    bot = runner()
    bot.api = Mock()
    monkeypatch.setattr("src.core.bot.run.time.monotonic", lambda: clock[0])
    return bot


def test_unanswered_first_move_is_aborted(monkeypatch):
    clock = [1000.0]
    bot = _stall_bot(monkeypatch, clock)
    snap = [{"gameId": "stuck", "isMyTurn": False, "lastMove": "e2e4",
             "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"}]
    bot._release_stalled_games(snap)
    clock[0] += 120.0
    bot._release_stalled_games(snap)
    bot.api.post.assert_not_called()
    clock[0] += 61.0
    bot._release_stalled_games(snap)
    bot.api.post.assert_called_once_with("/api/bot/game/stuck/abort")


def test_stall_watch_ignores_our_turn_and_resets_on_moves(monkeypatch):
    clock = [0.0]
    bot = _stall_bot(monkeypatch, clock)
    mine = [{"gameId": "g", "isMyTurn": True, "lastMove": "e7e5", "fen": "x w - - 0 2"}]
    bot._release_stalled_games(mine)
    clock[0] += 5000.0
    bot._release_stalled_games(mine)
    bot.api.post.assert_not_called()
    theirs = [{"gameId": "g", "isMyTurn": False, "lastMove": "g1f3", "fen": "x b - - 0 2"}]
    bot._release_stalled_games(theirs)
    clock[0] += 1199.0
    bot._release_stalled_games(theirs)
    bot.api.post.assert_not_called()
    clock[0] += 2.0
    bot._release_stalled_games(theirs)
    bot.api.post.assert_called_once_with("/api/bot/game/g/claim-victory")
    bot._release_stalled_games([])
    assert bot.stall_watch == {}
