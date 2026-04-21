#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

import chess
import chess.engine
import requests

LOG_FILE_PATH: Optional[Path] = None
LOG_FILE_LOCK = threading.Lock()
DEFAULT_LOG_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_LOG_KEEP_FILES = 5
MOVE_POST_MAX_ATTEMPTS = 4


def rotate_log_file(path: Path, max_bytes: int, keep_files: int) -> None:
    if max_bytes <= 0 or keep_files <= 0 or not path.exists():
        return
    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    try:
        path.replace(archived)
    except OSError:
        return

    archived_logs = sorted(
        path.parent.glob(f"{path.stem}_*{path.suffix}"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for old_path in archived_logs[keep_files:]:
        try:
            old_path.unlink()
        except OSError:
            pass


def configure_log_file(path: str, max_bytes: int = DEFAULT_LOG_MAX_BYTES, keep_files: int = DEFAULT_LOG_KEEP_FILES) -> None:
    global LOG_FILE_PATH
    if not path:
        LOG_FILE_PATH = None
        return

    resolved = Path(path).expanduser()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        rotate_log_file(resolved, max_bytes, keep_files)
        with resolved.open("a", encoding="utf-8") as fp:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.write(f"{ts} [boot] ========================================\n")
            fp.write(f"{ts} [boot] bot process started\n")
    except OSError as exc:
        LOG_FILE_PATH = None
        print(f"[warn] failed to open log file {resolved}: {exc}", file=sys.stderr, flush=True)
        return

    LOG_FILE_PATH = resolved


def log_event(scope: str, message: str, game_id: str = "") -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    game_part = f"[game {game_id}] " if game_id else ""
    line = f"{ts} [{scope}] {game_part}{message}"
    print(line, flush=True)

    if LOG_FILE_PATH is None:
        return
    try:
        with LOG_FILE_LOCK:
            with LOG_FILE_PATH.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
    except OSError:
        pass


@dataclass
class BotConfig:
    token: str
    base_url: str
    engine_path: str
    backend: str
    nn_model_path: str
    think_time: float
    max_depth: int
    max_games: int
    max_bot_games: int
    max_human_games: int
    accept_bots: bool
    accept_humans: bool
    log_file: str
    log_max_bytes: int
    log_keep_files: int
    book_path: str
    seek_specs: List["SeekSpec"] = field(default_factory=list)
    pair_poll_seconds: float = 6.0
    min_challenge_gap_seconds: float = 15.0


@dataclass(frozen=True)
class SeekSpec:
    initial_minutes: int
    increment_seconds: int
    rated: bool
    variant: str = "standard"
    color: str = "random"

    @property
    def label(self) -> str:
        suffix = "rated" if self.rated else "casual"
        return f"{self.initial_minutes}+{self.increment_seconds} {suffix}"

    def to_form(self) -> dict:
        return {
            "time": str(self.initial_minutes),
            "increment": str(self.increment_seconds),
            "rated": "true" if self.rated else "false",
            "variant": self.variant,
            "color": self.color,
        }


@dataclass
class ActiveGame:
    thread: threading.Thread
    slot_kind: str
    target: str = ""


@dataclass
class PendingSlot:
    slot_kind: str
    target: str = ""
    outgoing: bool = False
    created_at: float = 0.0


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer in {name}: {raw}") from exc


def env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number in {name}: {raw}") from exc


def env_list(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_seek_spec(text: str, rated: bool, variant: str, color: str) -> SeekSpec:
    parts = text.strip().lower().replace(" ", "")
    if "+" not in parts:
        raise argparse.ArgumentTypeError(f"invalid seek spec '{text}', expected MINUTES+INCREMENT")
    initial_str, increment_str = parts.split("+", 1)
    try:
        initial = int(initial_str)
        increment = int(increment_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seek spec '{text}', expected MINUTES+INCREMENT") from exc
    if initial <= 0 or increment < 0:
        raise argparse.ArgumentTypeError(f"invalid seek spec '{text}', expected positive minutes and non-negative increment")
    return SeekSpec(
        initial_minutes=initial,
        increment_seconds=increment,
        rated=rated,
        variant=variant,
        color=color,
    )


def default_log_path(here: Path) -> Path:
    if here.parent.name == "releases":
        return here.parent.parent / "shared" / "logs" / "chessbot.log"
    try:
        repo_root = here.parents[2]
    except IndexError:
        return here / "current" / "logs" / "chessbot.log"
    return repo_root / "current" / "logs" / "chessbot.log"


def default_nn_model_path(here: Path) -> str:
    env_path = os.environ.get("CHESS_NN_MODEL", "").strip()
    if env_path:
        return env_path
    for candidate in (
        here / "nn" / "model" / "nn_eval.bin",
        here / "nn_eval.bin",
        here.parent.parent / "shared" / "nn_eval.bin",
        here.parents[2] / "src" / "core" / "bot" / "nn" / "model" / "nn_eval.bin",
        here.parents[2] / "current" / "nn_eval.bin",
        here.parents[2] / "shared" / "nn_eval.bin",
        here.parents[3] / "current" / "nn_eval.bin",
        here.parents[3] / "shared" / "nn_eval.bin",
    ):
        if candidate.exists():
            return str(candidate)
    return ""

def load_build_info(here: Path) -> Dict[str, str]:
    info_path = here / "BUILD_INFO.txt"
    if not info_path.exists():
        return {}
    info: Dict[str, str] = {}
    try:
        for raw_line in info_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    except OSError:
        return {}
    return info


class LichessApi:
    def __init__(self, token: str, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get_json(self, path: str) -> dict:
        resp = self.session.get(self._url(path), timeout=20)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, data: Optional[dict] = None) -> requests.Response:
        resp = self.session.post(self._url(path), data=data, timeout=20)
        resp.raise_for_status()
        return resp

    def get_ndjson(self, path: str, limit: int = 100) -> List[dict]:
        items: List[dict] = []
        with self.session.get(self._url(path), stream=True, timeout=20) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="ignore").strip()
                else:
                    line = str(raw).strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(items) >= limit:
                    break
        return items

    def stream_events(self, path: str, reconnect: bool = True) -> Generator[dict, None, None]:
        attempt = 0
        while True:
            try:
                if attempt == 0:
                    log_event("stream", f"open {path}")
                else:
                    log_event("stream", f"reopen {path} (attempt {attempt})")
                with self.session.get(self._url(path), stream=True, timeout=(15, None)) as resp:
                    resp.raise_for_status()
                    if attempt > 0:
                        log_event("stream", f"stream restored on {path} after {attempt} retries")
                    attempt = 0
                    for raw in resp.iter_lines(decode_unicode=True):
                        if raw is None:
                            continue
                        if isinstance(raw, bytes):
                            line = raw.decode("utf-8", errors="ignore").strip()
                        else:
                            line = str(raw).strip()
                        if not line:
                            continue

                        # Lichess streams are usually NDJSON; keep SSE "data:" support as fallback.
                        payload = line[5:].strip() if line.startswith("data:") else line
                        if not payload or payload == "null":
                            continue
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            continue
            except requests.RequestException as exc:
                attempt += 1
                backoff_s = min(60.0, 2.0 * (1.7 ** min(attempt - 1, 10)))
                status_code = None
                if hasattr(exc, "response") and exc.response is not None:
                    status_code = exc.response.status_code
                if status_code is not None and 500 <= status_code < 600:
                    backoff_s = max(backoff_s, 4.0)
                log_event(
                    "stream",
                    f"reconnect after error on {path}: {exc} (retry {attempt} in {backoff_s:.1f}s)",
                )
                time.sleep(backoff_s)
            if not reconnect:
                return

    def get_stream_snapshot(self, path: str, read_timeout_s: float = 4.0) -> Optional[dict]:
        with self.session.get(self._url(path), stream=True, timeout=(15, read_timeout_s)) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="ignore").strip()
                else:
                    line = str(raw).strip()
                if not line:
                    continue
                payload = line[5:].strip() if line.startswith("data:") else line
                if not payload or payload == "null":
                    continue
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    continue
        return None


class BotRunner:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.api = LichessApi(cfg.token, cfg.base_url)
        self.username = self.api.get_json("/api/account").get("username", "")
        if not self.username:
            raise RuntimeError("Could not read Lichess username from /api/account")
        self.username_lc = self.username.lower()
        self.active_games: Dict[str, ActiveGame] = {}
        self.pending_slots: Dict[str, PendingSlot] = {}
        self.bot_cooldowns: Dict[str, float] = {}
        self.bot_cursor = 0
        self.spec_cursor = 0
        self.pair_backoff_until = 0.0
        self.next_pair_attempt_at = 0.0
        self.pair_rate_limit_count = 0
        self.lock = threading.Lock()
        log_event("init", f"logged in as {self.username}")
        log_event(
            "init",
            (
                f"think_time={cfg.think_time:.2f}s "
                f"max_depth={cfg.max_depth if cfg.max_depth > 0 else 'engine-default'} "
                f"max_games={cfg.max_games} bot_slots={cfg.max_bot_games} human_slots={cfg.max_human_games} "
                f"accept_bots={cfg.accept_bots} accept_humans={cfg.accept_humans}"
            ),
        )
        log_event("init", f"engine_path={cfg.engine_path}")
        log_event("init", f"backend={cfg.backend}")
        if cfg.nn_model_path:
            log_event("init", f"nn_model={cfg.nn_model_path}")
        if cfg.book_path:
            log_event("init", f"opening_book={cfg.book_path}")
        if cfg.seek_specs:
            log_event(
                "init",
                (
                    f"bot_pairings={', '.join(spec.label for spec in cfg.seek_specs)} "
                    f"poll={cfg.pair_poll_seconds:.1f}s gap={cfg.min_challenge_gap_seconds:.1f}s"
                ),
            )

    def _prune_pending_slots_locked(self) -> None:
        now = time.time()
        stale_ids = [
            challenge_id
            for challenge_id, slot in self.pending_slots.items()
            if slot.outgoing and (now - slot.created_at) > 45.0
        ]
        for challenge_id in stale_ids:
            slot = self.pending_slots.pop(challenge_id, None)
            if slot is not None and slot.target:
                self.bot_cooldowns[slot.target.lower()] = now + 90.0

    def _counts_locked(self) -> tuple[int, int, int, int, int]:
        self._prune_pending_slots_locked()
        active_total = len(self.active_games)
        active_bot = sum(1 for game in self.active_games.values() if game.slot_kind == "bot")
        active_human = sum(1 for game in self.active_games.values() if game.slot_kind == "human")
        pending_bot = sum(1 for slot in self.pending_slots.values() if slot.slot_kind == "bot")
        pending_human = sum(1 for slot in self.pending_slots.values() if slot.slot_kind == "human")
        return active_total, active_bot, active_human, pending_bot, pending_human

    def _counts_text_locked(self) -> str:
        active_total, active_bot, active_human, pending_bot, pending_human = self._counts_locked()
        return (
            f"active={active_total}/{self.cfg.max_games} "
            f"bot={active_bot}+{pending_bot}/{self.cfg.max_bot_games} "
            f"human={active_human}+{pending_human}/{self.cfg.max_human_games}"
        )

    def _can_reserve_slot_locked(self, slot_kind: str) -> bool:
        active_total, active_bot, active_human, pending_bot, pending_human = self._counts_locked()
        reserved_total = active_total + pending_bot + pending_human
        reserved_bot = active_bot + pending_bot
        reserved_human = active_human + pending_human

        if reserved_total >= self.cfg.max_games:
            return False
        if slot_kind == "bot":
            return reserved_bot < self.cfg.max_bot_games
        if slot_kind == "human":
            return reserved_human < self.cfg.max_human_games
        return False

    def _choose_bot_spec(self) -> SeekSpec:
        spec = self.cfg.seek_specs[self.spec_cursor % len(self.cfg.seek_specs)]
        self.spec_cursor = (self.spec_cursor + 1) % max(1, len(self.cfg.seek_specs))
        return spec

    @staticmethod
    def _retry_after_seconds(exc: requests.RequestException, default_seconds: float) -> float:
        response = getattr(exc, "response", None)
        if response is None:
            return default_seconds
        value = response.headers.get("Retry-After")
        if value is None:
            return default_seconds
        try:
            seconds = float(value.strip())
        except (TypeError, ValueError):
            return default_seconds
        return max(1.0, seconds)

    def _maintain_bot_pairings(self) -> None:
        if not self.cfg.seek_specs or self.cfg.max_bot_games <= 0:
            return

        while True:
            try:
                self._issue_outgoing_bot_challenges()
            except Exception as exc:
                log_event("pair", f"auto-challenge loop error: {exc}")
            time.sleep(max(1.0, self.cfg.pair_poll_seconds))

    def _reserve_pending_slot_locked(self,
                                     challenge_id: str,
                                     slot_kind: str,
                                     target: str = "",
                                     outgoing: bool = False) -> None:
        self.pending_slots[challenge_id] = PendingSlot(
            slot_kind=slot_kind,
            target=target,
            outgoing=outgoing,
            created_at=time.time(),
        )

    def _reserve_challenge_slot(self,
                                challenge_id: str,
                                slot_kind: str,
                                target: str = "",
                                outgoing: bool = False) -> bool:
        with self.lock:
            if not self._can_reserve_slot_locked(slot_kind):
                return False
            self._reserve_pending_slot_locked(challenge_id, slot_kind, target=target, outgoing=outgoing)
            return True

    def _release_pending_slot(self, challenge_id: str, cooldown_seconds: Optional[float] = None) -> None:
        with self.lock:
            slot = self.pending_slots.pop(challenge_id, None)
            if slot is not None and slot.target:
                if cooldown_seconds is None:
                    cooldown_seconds = 120.0 if slot.outgoing else 30.0
                self.bot_cooldowns[slot.target.lower()] = time.time() + max(1.0, cooldown_seconds)

    def _fetch_online_bots(self, limit: int = 80) -> List[dict]:
        return self.api.get_ndjson(f"/api/bot/online?nb={max(1, limit)}", limit=limit)

    def _eligible_bot_username_locked(self, bots: List[dict]) -> Optional[str]:
        if not bots:
            return None

        active_targets = {game.target.lower() for game in self.active_games.values() if game.target}
        pending_targets = {slot.target.lower() for slot in self.pending_slots.values() if slot.target}
        now = time.time()

        for _ in range(len(bots)):
            idx = self.bot_cursor % len(bots)
            self.bot_cursor = (self.bot_cursor + 1) % max(1, len(bots))
            entry = bots[idx] or {}
            username = str(entry.get("username") or entry.get("id") or "").strip()
            if not username:
                continue
            username_lc = username.lower()
            if username_lc == self.username_lc:
                continue
            if username_lc in active_targets or username_lc in pending_targets:
                continue
            cooldown_until = self.bot_cooldowns.get(username_lc, 0.0)
            if cooldown_until > now:
                continue
            return username
        return None

    def _create_outgoing_bot_challenge(self, reserved_id: str, username: str, spec: SeekSpec) -> bool:
        form = {
            "rated": "true" if spec.rated else "false",
            "variant": spec.variant,
            "color": spec.color,
            "clock.limit": str(spec.initial_minutes * 60),
            "clock.increment": str(spec.increment_seconds),
        }
        try:
            resp = self.api.post(f"/api/challenge/{username}", data=form)
            payload = resp.json()
            challenge_id = str(payload.get("id", "")).strip()
            if not challenge_id:
                with self.lock:
                    self.pending_slots.pop(reserved_id, None)
                log_event("pair", f"challenge to {username} missing id ({spec.label})")
                return False
            with self.lock:
                slot = self.pending_slots.pop(reserved_id, None)
                if slot is None:
                    return False
                self.pending_slots[challenge_id] = PendingSlot(
                    slot_kind=slot.slot_kind,
                    target=username,
                    outgoing=True,
                    created_at=slot.created_at,
                )
                self.bot_cooldowns[username.lower()] = time.time() + 90.0
                self.pair_rate_limit_count = 0
                counts_text = self._counts_text_locked()
            log_event("pair", f"challenged {username} {spec.label} id={challenge_id} ({counts_text})")
            return True
        except requests.RequestException as exc:
            status_code = None
            if hasattr(exc, "response") and exc.response is not None:
                status_code = exc.response.status_code
            with self.lock:
                self.pending_slots.pop(reserved_id, None)
                self.bot_cooldowns[username.lower()] = time.time() + 120.0
                if status_code == 429:
                    self.pair_rate_limit_count += 1
                    backoff_s = max(
                        self._retry_after_seconds(exc, 90.0),
                        min(900.0, 90.0 * float(self.pair_rate_limit_count)),
                    )
                    next_allowed = time.time() + backoff_s
                    self.pair_backoff_until = max(self.pair_backoff_until, next_allowed)
                    self.next_pair_attempt_at = max(self.next_pair_attempt_at, next_allowed)
                counts_text = self._counts_text_locked()
            if status_code == 429:
                wait_s = max(0.0, self.pair_backoff_until - time.time())
                log_event(
                    "pair",
                    (
                        f"challenge rate-limited user={username} {spec.label}: "
                        f"backing off for {wait_s:.0f}s limit_count={self.pair_rate_limit_count} ({counts_text})"
                    ),
                )
                return False
            log_event("pair", f"challenge failed user={username} {spec.label}: {exc}")
            return False

    def _issue_outgoing_bot_challenges(self) -> None:
        with self.lock:
            now = time.time()
            if self.pair_backoff_until > now:
                return
            if self.next_pair_attempt_at > now:
                return
            if not self._can_reserve_slot_locked("bot"):
                return

        online_bots = self._fetch_online_bots(limit=100)
        if not online_bots:
            log_event("pair", "no online bots returned from /api/bot/online")
            return

        with self.lock:
            if not self._can_reserve_slot_locked("bot"):
                return
            username = self._eligible_bot_username_locked(online_bots)
            if username is None:
                return
            spec = self._choose_bot_spec()
            now = time.time()
            reserved_id = f"outgoing:{username.lower()}:{int(now * 1000)}"
            self._reserve_pending_slot_locked(reserved_id, "bot", target=username, outgoing=True)
            self.bot_cooldowns[username.lower()] = now + 30.0
            self.next_pair_attempt_at = now + max(1.0, self.cfg.min_challenge_gap_seconds)
        self._create_outgoing_bot_challenge(reserved_id, username, spec)

    def run(self) -> None:
        if self.cfg.seek_specs:
            threading.Thread(target=self._maintain_bot_pairings, daemon=True).start()
        log_event("main", "listening for Lichess events")
        for event in self.api.stream_events("/api/stream/event", reconnect=True):
            etype = event.get("type")
            if etype == "challenge":
                self._handle_challenge(event.get("challenge", {}))
            elif etype == "gameStart":
                game_id = event.get("game", {}).get("id")
                if game_id:
                    self._start_game_thread(game_id)
            elif etype == "gameFinish":
                game_id = event.get("game", {}).get("id")
                if game_id:
                    self._mark_game_done(game_id, source="event")
            elif etype in {"challengeCanceled", "challengeDeclined"}:
                challenge = event.get("challenge", {}) or event
                challenge_id = challenge.get("id")
                if challenge_id:
                    cooldown_s = 900.0 if etype == "challengeDeclined" else 300.0
                    self._release_pending_slot(challenge_id, cooldown_s)
                log_event("event", f"received {etype}{' id=' + challenge_id if challenge_id else ''}")
            elif etype is not None:
                log_event("event", f"received {etype}")

    def _handle_challenge(self, challenge: dict) -> None:
        cid = challenge.get("id")
        if not cid:
            return

        challenger = challenge.get("challenger", {}) or {}
        challenger_name = challenger.get("name", "?")
        challenger_id = str(challenger.get("id", "")).lower()
        challenger_title = challenger.get("title", "")
        challenger_rating = challenger.get("rating")
        direction = str(challenge.get("direction", "")).lower()
        variant = challenge.get("variant", {}).get("key", "")
        speed = challenge.get("speed", "")
        rated = bool(challenge.get("rated", False))
        tc = challenge.get("timeControl", {}) or {}
        if tc.get("type") == "clock":
            tc_desc = f"{tc.get('limit', '?')}+{tc.get('increment', '?')}"
        else:
            tc_desc = tc.get("type", "unknown")
        log_event(
            "challenge",
            (
                f"incoming id={cid} from={challenger_title + ' ' if challenger_title else ''}"
                f"{challenger_name} ({challenger_rating}) rated={rated} "
                f"variant={variant} speed={speed} tc={tc_desc}"
            ),
        )

        # Outgoing challenges we created ourselves can appear on the same event stream.
        # They should not be "accepted" by us again; we just wait for gameStart or decline/cancel.
        if direction == "out" or challenger_id == self.username_lc:
            log_event("challenge", f"tracked outgoing id={cid} direction={direction or 'out'}", "")
            return

        variant = challenge.get("variant", {}).get("key", "")
        if variant != "standard":
            self._decline(cid, "standard", f"variant={variant}")
            return

        speed = challenge.get("speed", "")
        if speed == "correspondence":
            self._decline(cid, "timeControl", "correspondence")
            return

        challenger_title = challenge.get("challenger", {}).get("title")
        is_bot = challenger_title == "BOT"
        if is_bot and not self.cfg.accept_bots:
            self._decline(cid, "noBot", "bot challenge")
            return
        if (not is_bot) and not self.cfg.accept_humans:
            self._decline(cid, "onlyBot", "human challenge")
            return

        slot_kind = "bot" if is_bot else "human"
        if not self._reserve_challenge_slot(cid, slot_kind, target=challenger_name, outgoing=False):
            self._decline(cid, "later", f"busy ({slot_kind} slot unavailable)")
            return

        try:
            self.api.post(f"/api/challenge/{cid}/accept")
            with self.lock:
                counts_text = self._counts_text_locked()
            log_event("challenge", f"accepted id={cid} slot={slot_kind} ({counts_text})")
        except requests.RequestException as exc:
            self._release_pending_slot(cid)
            log_event("challenge", f"accept failed id={cid}: {exc}")

    def _decline(self, challenge_id: str, reason: str, why: str) -> None:
        try:
            self.api.post(f"/api/challenge/{challenge_id}/decline", data={"reason": reason})
            log_event("challenge", f"declined id={challenge_id}: {why} (reason={reason})")
        except requests.RequestException as exc:
            log_event("challenge", f"decline failed id={challenge_id}: {exc}")

    def _start_game_thread(self, game_id: str) -> None:
        with self.lock:
            if game_id in self.active_games:
                return
            pending = self.pending_slots.pop(game_id, None)
            slot_kind = pending.slot_kind if pending is not None else "bot"
            target = pending.target if pending is not None else ""
            t = threading.Thread(target=self._play_game, args=(game_id,), daemon=True)
            self.active_games[game_id] = ActiveGame(thread=t, slot_kind=slot_kind, target=target)
            t.start()
            counts_text = self._counts_text_locked()
        log_event("game", f"started slot={slot_kind} ({counts_text})", game_id)

    def _mark_game_done(self, game_id: str, source: str = "thread") -> None:
        with self.lock:
            removed = self.active_games.pop(game_id, None)
            pending = self.pending_slots.pop(game_id, None)
            if removed is not None and removed.target:
                self.bot_cooldowns[removed.target.lower()] = time.time() + 300.0
            if pending is not None and pending.target:
                self.bot_cooldowns[pending.target.lower()] = time.time() + 300.0
            counts_text = self._counts_text_locked()
        if removed is not None:
            log_event("game", f"finished source={source} slot={removed.slot_kind} ({counts_text})", game_id)

    def _is_game_active(self, game_id: str) -> bool:
        with self.lock:
            return game_id in self.active_games

    def _build_board(self, initial_fen: str, moves: str) -> Optional[chess.Board]:
        try:
            board = chess.Board() if initial_fen == "startpos" else chess.Board(initial_fen)
            if moves.strip():
                for uci in moves.split():
                    board.push_uci(uci)
            return board
        except ValueError:
            return None

    @staticmethod
    def _clock_pair_to_seconds(clock: dict) -> tuple[float, float]:
        def _to_float(value: object) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return 0.0
            return v if v > 0.0 else 0.0

        raw_initial = _to_float(clock.get("initial", 0))
        raw_increment = _to_float(clock.get("increment", 0))

        # Lichess realtime clock is milliseconds (e.g. 180000+2000).
        # Keep compatibility with second-based values if ever provided.
        is_millis = raw_initial >= 1000.0 or raw_increment >= 120.0
        scale = 1000.0 if is_millis else 1.0
        return raw_initial / scale, raw_increment / scale

    def _compute_game_think_time(self, clock: dict) -> tuple[float, float, float]:
        base_default = max(0.01, self.cfg.think_time)
        if not clock:
            return base_default, 0.0, 0.0

        initial_s, increment_s = self._clock_pair_to_seconds(clock)

        # Policy requested:
        # - any increment time control: think time = increment / 2
        # - no increment and under 10 minutes: 0.35s
        # - no increment and 10 minutes or longer: 2.5s
        if increment_s > 0.0:
            return max(0.05, increment_s * 0.5), initial_s, increment_s
        if 0.0 < initial_s < 600.0:
            return 0.35, initial_s, increment_s
        if initial_s >= 600.0:
            return 2.5, initial_s, increment_s
        return base_default, initial_s, increment_s

    @staticmethod
    def _state_clock_seconds(state: dict, initial_s: float, increment_s: float) -> tuple[float, float, float, float]:
        def _to_seconds(value: object, fallback: float) -> float:
            try:
                raw = float(value)
            except (TypeError, ValueError):
                return fallback
            if raw <= 0.0:
                return fallback
            if raw >= 1000.0:
                return raw / 1000.0
            return raw

        wtime = _to_seconds(state.get("wtime"), initial_s)
        btime = _to_seconds(state.get("btime"), initial_s)
        winc = _to_seconds(state.get("winc"), increment_s)
        binc = _to_seconds(state.get("binc"), increment_s)
        return wtime, btime, winc, binc

    @staticmethod
    def _estimate_moves_to_go(board: chess.Board) -> int:
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        minors = (
            len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        )

        if queens == 0 and rooks <= 2 and minors <= 3:
            return 14
        if board.fullmove_number <= 10:
            return 28
        if board.fullmove_number <= 20:
            return 24
        if board.fullmove_number <= 35:
            return 20
        return 16

    @staticmethod
    def _concurrency_scale(active_bot_games: int) -> float:
        if active_bot_games >= 3:
            return 0.60
        if active_bot_games >= 2:
            return 0.75
        return 1.0

    @classmethod
    def _hard_cap_seconds(cls,
                          initial_s: float,
                          increment_s: float,
                          remaining_s: float,
                          active_bot_games: int) -> float:
        if increment_s > 0.0:
            if initial_s <= 180.0 and increment_s <= 2.0:
                hard_cap = 1.50
            elif initial_s <= 300.0 and increment_s <= 3.0:
                hard_cap = 1.85
            elif initial_s <= 600.0:
                hard_cap = 2.50
            else:
                hard_cap = min(5.00, 1.50 + increment_s * 0.50)
        else:
            if initial_s <= 60.0:
                hard_cap = 0.35
            elif initial_s <= 180.0:
                hard_cap = 0.60
            elif initial_s <= 300.0:
                hard_cap = 1.00
            elif initial_s <= 600.0:
                hard_cap = 2.00
            else:
                hard_cap = 3.00

        hard_cap *= cls._concurrency_scale(active_bot_games)

        if remaining_s <= 10.0:
            hard_cap = min(hard_cap, max(0.10, remaining_s * 0.10 + increment_s * 0.35))
        if remaining_s <= 3.0:
            hard_cap = min(hard_cap, max(0.05, remaining_s / 5.0 + increment_s * 0.25))
        return max(0.05, hard_cap)

    def _active_bot_games(self) -> int:
        with self.lock:
            return sum(1 for game in self.active_games.values() if game.slot_kind == "bot")

    def _compute_move_budget(self,
                             board: chess.Board,
                             state: dict,
                             my_color: bool,
                             initial_s: float,
                             increment_s: float,
                             base_default: float,
                             active_bot_games: int) -> tuple[float, float, float]:
        wtime, btime, winc, binc = self._state_clock_seconds(state, initial_s, increment_s)
        remaining = wtime if my_color == chess.WHITE else btime
        increment = winc if my_color == chess.WHITE else binc
        if remaining <= 0.0:
            return max(0.05, base_default), remaining, increment

        moves_to_go = self._estimate_moves_to_go(board)
        reserve = max(0.15, increment * 2.0)
        if increment <= 0.0:
            reserve = max(reserve, remaining * 0.12)
        else:
            reserve = max(reserve, min(remaining * 0.08, increment * 3.0))

        usable = max(0.05, remaining - reserve)
        horizon = max(24.0, float(moves_to_go) * 4.0)
        budget = usable / horizon + increment * 0.35
        budget *= self._concurrency_scale(active_bot_games)

        legal_count = board.legal_moves.count()
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        minors = (
            len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        )
        simplified_endgame = queens == 0 and rooks <= 2 and minors <= 3

        if board.is_check():
            budget *= 1.40
        if legal_count <= 4:
            budget *= 1.20
        if legal_count <= 2:
            budget *= 1.15
        if simplified_endgame:
            budget *= 1.35
        if legal_count == 1:
            budget *= 0.65

        floor = 0.05
        if increment > 0.0:
            floor = max(floor, min(0.25, increment * 0.35))
        cap = min(8.0, max(0.10, usable * 0.55))
        if increment <= 0.0:
            cap = min(cap, max(0.10, remaining * 0.22))
        else:
            cap = min(cap, max(0.15, remaining * 0.25 + increment * 2.0))
        cap *= self._concurrency_scale(active_bot_games)

        if remaining <= 3.0:
            cap = min(cap, max(0.05, remaining / 5.0 + increment * 0.40))
        if remaining <= 1.0:
            cap = min(cap, 0.10 + increment * 0.25)

        hard_cap = self._hard_cap_seconds(initial_s, increment_s, remaining, active_bot_games)
        cap = min(cap, hard_cap)

        if budget < floor:
            budget = floor
        if budget > cap:
            budget = cap
        if budget < 0.01:
            budget = 0.01
        return budget, remaining, increment

    def _configure_engine(self, engine: chess.engine.SimpleEngine) -> None:
        options: dict = {}
        if self.cfg.backend == "nn" and "NNModel" in engine.options and self.cfg.nn_model_path:
            options["NNModel"] = self.cfg.nn_model_path
        if "Backend" in engine.options:
            options["Backend"] = self.cfg.backend
        if "MoveTime" in engine.options:
            options["MoveTime"] = max(1, int(self.cfg.think_time * 1000.0))
        if self.cfg.max_depth > 0 and "MaxDepth" in engine.options:
            options["MaxDepth"] = self.cfg.max_depth
        if self.cfg.book_path and "BookFile" in engine.options:
            options["BookFile"] = self.cfg.book_path
        if options:
            engine.configure(options)
        log_event("engine", f"configured options: {options if options else '(none)'}")

    @staticmethod
    def _request_status_code(exc: requests.RequestException) -> Optional[int]:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        return response.status_code

    @classmethod
    def _is_retryable_move_post_error(cls, exc: requests.RequestException) -> bool:
        status_code = cls._request_status_code(exc)
        if status_code is None:
            return True
        return status_code == 429 or 500 <= status_code < 600

    def _probe_game_after_move_post_error(
        self,
        api: LichessApi,
        game_id: str,
        initial_fen: str,
        my_color: bool,
        expected_ply: int,
        attempted_uci: str,
    ) -> tuple[str, str]:
        try:
            snapshot = api.get_stream_snapshot(f"/api/bot/game/stream/{game_id}")
        except requests.RequestException as exc:
            log_event("game", f"snapshot after move error failed: {exc}", game_id)
            return "unknown", "started"

        if not snapshot:
            return "unknown", "started"

        if snapshot.get("type") == "gameFull":
            snapshot_initial_fen = snapshot.get("initialFen", initial_fen)
            state = snapshot.get("state", {}) or {}
        else:
            snapshot_initial_fen = initial_fen
            state = snapshot

        status = str(state.get("status", "started") or "started")
        moves = str(state.get("moves", "") or "")
        if status != "started":
            return "terminal", status

        board = self._build_board(snapshot_initial_fen, moves)
        if board is None:
            return "unknown", status
        if board.move_stack and board.move_stack[-1].uci() == attempted_uci:
            return "move-seen", status
        if len(board.move_stack) > expected_ply or board.turn != my_color:
            return "position-advanced", status
        return "same-position", status

    def _post_move_with_recovery(
        self,
        api: LichessApi,
        game_id: str,
        initial_fen: str,
        my_color: bool,
        expected_ply: int,
        uci: str,
    ) -> tuple[bool, str]:
        last_error: Optional[requests.RequestException] = None
        for attempt in range(1, MOVE_POST_MAX_ATTEMPTS + 1):
            if not self._is_game_active(game_id):
                return False, "finished"
            try:
                api.post(f"/api/bot/game/{game_id}/move/{uci}")
                return True, "started"
            except requests.RequestException as exc:
                last_error = exc
                snapshot_result, snapshot_status = self._probe_game_after_move_post_error(
                    api,
                    game_id,
                    initial_fen,
                    my_color,
                    expected_ply,
                    uci,
                )
                if snapshot_result in {"move-seen", "position-advanced"}:
                    log_event(
                        "game",
                        f"move post recovered ({uci}) after error: {exc} [{snapshot_result}]",
                        game_id,
                    )
                    return True, snapshot_status
                if snapshot_result == "terminal":
                    log_event(
                        "game",
                        f"move post skipped ({uci}) after terminal status={snapshot_status}: {exc}",
                        game_id,
                    )
                    return False, snapshot_status
                if not self._is_retryable_move_post_error(exc) or attempt >= MOVE_POST_MAX_ATTEMPTS:
                    break
                backoff_s = min(2.0, 0.35 * attempt)
                log_event(
                    "game",
                    f"move post retry {attempt}/{MOVE_POST_MAX_ATTEMPTS - 1} ({uci}) after error: {exc}",
                    game_id,
                )
                time.sleep(backoff_s)

        if last_error is not None:
            log_event("game", f"move post failed ({uci}): {last_error}", game_id)
        return False, "started"

    @staticmethod
    def _recent_moves_text(board: chess.Board, count: int = 8) -> str:
        if not board.move_stack:
            return "-"
        return " ".join(move.uci() for move in board.move_stack[-count:])

    @staticmethod
    def _repetition_summary(board: chess.Board) -> str:
        return " ".join([
            f"turn={'w' if board.turn == chess.WHITE else 'b'}",
            f"halfmove={board.halfmove_clock}",
            f"fullmove={board.fullmove_number}",
            f"check={int(board.is_check())}",
            f"rep2={int(board.is_repetition(2))}",
            f"rep3={int(board.is_repetition(3))}",
            f"claim={int(board.can_claim_draw())}",
            f"fivefold={int(board.is_fivefold_repetition())}",
        ])

    def _should_log_repetition_focus(self, board: chess.Board) -> bool:
        return (
            board.halfmove_clock >= 80 or
            board.is_repetition(2) or
            board.is_repetition(3) or
            board.can_claim_draw() or
            board.is_fivefold_repetition()
        )

    @staticmethod
    def _score_fields(info: dict, my_color: bool) -> tuple[Optional[int], Optional[int]]:
        score = info.get("score")
        if score is None:
            return None, None
        try:
            pov_score = score.pov(my_color)
        except Exception:
            return None, None
        return pov_score.score(mate_score=32000), pov_score.mate()

    @staticmethod
    def _format_move_sequence(moves: object, limit: int = 6) -> str:
        try:
            move_list = list(moves)
        except TypeError:
            return str(moves)

        tokens: List[str] = []
        for move in move_list[:limit]:
            try:
                tokens.append(move.uci())
            except Exception:
                tokens.append(str(move))
        return " ".join(tokens) if tokens else "-"

    def _format_search_extra(self, key: str, value: object) -> str:
        if key in {"pv", "refutation"}:
            return self._format_move_sequence(value)
        if key == "currmove":
            try:
                return value.uci()
            except Exception:
                return str(value)
        if key == "currline":
            if isinstance(value, dict):
                parts: List[str] = []
                for cpu_id in sorted(value):
                    parts.append(f"{cpu_id}:{self._format_move_sequence(value[cpu_id], limit=4)}")
                return ",".join(parts) if parts else "-"
            return str(value)
        return str(value)

    def _format_search_info(self, info: dict, my_color: bool) -> str:
        bits: List[str] = []
        handled_keys = {
            "depth",
            "seldepth",
            "score",
            "nodes",
            "nps",
            "time",
            "pv",
            "multipv",
            "currmove",
            "currmovenumber",
            "hashfull",
            "tbhits",
            "cpuload",
            "refutation",
            "currline",
            "string",
        }
        depth = info.get("depth")
        seldepth = info.get("seldepth")
        nodes = info.get("nodes")
        nps = info.get("nps")
        time_s = info.get("time")
        score_cp, score_mate = self._score_fields(info, my_color)
        pv = info.get("pv")
        multipv = info.get("multipv")
        currmove = info.get("currmove")
        currmovenumber = info.get("currmovenumber")
        hashfull = info.get("hashfull")
        tbhits = info.get("tbhits")
        cpuload = info.get("cpuload")
        refutation = info.get("refutation")
        currline = info.get("currline")
        string = info.get("string")

        if depth is not None:
            bits.append(f"d={depth}")
        if seldepth is not None:
            bits.append(f"sd={seldepth}")
        if multipv is not None:
            bits.append(f"mpv={multipv}")
        if score_mate is not None:
            bits.append(f"mate={score_mate}")
        elif score_cp is not None:
            bits.append(f"cp={score_cp}")
        if nodes is not None:
            bits.append(f"nodes={nodes}")
        if nps is not None:
            bits.append(f"nps={nps}")
        if hashfull is not None:
            bits.append(f"hash={hashfull}")
        if tbhits is not None:
            bits.append(f"tbhits={tbhits}")
        if cpuload is not None:
            bits.append(f"cpu={cpuload}")
        if time_s is not None:
            try:
                bits.append(f"time={float(time_s):.3f}s")
            except (TypeError, ValueError):
                bits.append(f"time={time_s}")
        if currmove is not None:
            bits.append(f"curr={self._format_search_extra('currmove', currmove)}")
        if currmovenumber is not None:
            bits.append(f"currno={currmovenumber}")
        if pv:
            try:
                pv_moves = [move.uci() for move in pv[:6]]
                if pv_moves:
                    bits.append(f"root={pv_moves[0]}")
                    bits.append(f"pv={' '.join(pv_moves)}")
            except Exception:
                pass
        if refutation:
            bits.append(f"ref={self._format_search_extra('refutation', refutation)}")
        if currline:
            bits.append(f"line={self._format_search_extra('currline', currline)}")
        if string:
            bits.append(f"str={string}")

        for key in sorted(info):
            if key in handled_keys:
                continue
            bits.append(f"{key}={self._format_search_extra(key, info[key])}")
        return " ".join(bits)

    def _analyse_move(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        limit: chess.engine.Limit,
        my_color: bool,
        game_id: str,
        ply: int,
    ) -> tuple[Optional[chess.Move], dict]:
        final_info: dict = {}
        packet_index = 0

        with engine.analysis(
            board,
            limit,
            info=chess.engine.INFO_ALL,
        ) as analysis:
            while True:
                info = analysis.next()
                if info is None:
                    break
                packet_index += 1
                final_info = dict(analysis.info)
                formatted = self._format_search_info(info, my_color)
                if formatted:
                    log_event("search", f"ply={ply} pkt={packet_index} {formatted}", game_id)
                else:
                    log_event("search", f"ply={ply} pkt={packet_index} (empty)", game_id)
            best = analysis.wait()
            final_info = dict(analysis.info)

        if best.move is not None and "pv" not in final_info:
            final_info["pv"] = [best.move]
        return best.move, final_info

    def _play_game(self, game_id: str) -> None:
        api = LichessApi(self.cfg.token, self.cfg.base_url)
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.cfg.engine_path)
        except Exception as exc:
            log_event("engine", f"failed to start: {exc}", game_id)
            self._mark_game_done(game_id, source="engine-start-failed")
            return

        try:
            self._configure_engine(engine)
        except Exception as exc:
            log_event("engine", f"option setup failed: {exc}", game_id)

        initial_fen = "startpos"
        my_color: Optional[bool] = None
        last_played_ply = -1
        game_status = "started"
        game_think_time = self.cfg.think_time
        initial_s = 0.0
        increment_s = 0.0

        try:
            for event in api.stream_events(f"/api/bot/game/stream/{game_id}", reconnect=True):
                etype = event.get("type")
                if etype == "gameFull":
                    initial_fen = event.get("initialFen", "startpos")
                    white = event.get("white", {}) or {}
                    black = event.get("black", {}) or {}
                    white_id = white.get("id", "").lower()
                    black_id = black.get("id", "").lower()
                    if white_id == self.username_lc:
                        my_color = chess.WHITE
                    elif black_id == self.username_lc:
                        my_color = chess.BLACK
                    state = event.get("state", {}) or {}
                    speed = event.get("speed", "?")
                    rated = bool(event.get("rated", False))
                    perf = (event.get("perf", {}) or {}).get("name", "?")
                    clock = event.get("clock", {}) or {}
                    clock_desc = (
                        f"{clock.get('initial', '?')}+{clock.get('increment', '?')}"
                        if clock
                        else "?"
                    )
                    game_think_time, initial_s, increment_s = self._compute_game_think_time(clock)
                    my_color_name = "white" if my_color == chess.WHITE else ("black" if my_color == chess.BLACK else "?")
                    log_event(
                        "game",
                        (
                            f"meta {white.get('name', '?')}({white.get('rating', '?')}) vs "
                            f"{black.get('name', '?')}({black.get('rating', '?')}) "
                            f"rated={rated} speed={speed} perf={perf} clock={clock_desc} "
                            f"my_color={my_color_name}"
                        ),
                        game_id,
                    )
                    log_event(
                        "time",
                        (
                            f"policy initial={initial_s:.2f}s increment={increment_s:.2f}s "
                            f"think={game_think_time:.2f}s"
                        ),
                        game_id,
                    )
                    if initial_fen != "startpos":
                        log_event("game", f"initial FEN={initial_fen}", game_id)
                elif etype == "gameState":
                    state = event
                else:
                    continue

                if my_color is None:
                    continue

                status = state.get("status", "started")
                game_status = status
                if status != "started":
                    winner = state.get("winner", "")
                    wdraw = state.get("wdraw", False)
                    bdraw = state.get("bdraw", False)
                    log_event(
                        "game",
                        f"terminal status={status} winner={winner or '-'} wdraw={wdraw} bdraw={bdraw}",
                        game_id,
                    )
                    final_moves = state.get("moves", "")
                    final_board = self._build_board(initial_fen, final_moves)
                    if final_board is not None:
                        log_event(
                            "diag",
                            (
                                f"terminal {self._repetition_summary(final_board)} "
                                f"recent={self._recent_moves_text(final_board, 12)} "
                                f"fen={final_board.fen()}"
                            ),
                            game_id,
                        )
                    break

                moves = state.get("moves", "")
                board = self._build_board(initial_fen, moves)
                if board is None:
                    log_event("game", "invalid move list in stream", game_id)
                    break

                ply = len(board.move_stack)
                if board.turn != my_color or ply == last_played_ply:
                    continue

                active_bot_games = self._active_bot_games()
                move_budget_s, remaining_s, inc_s = self._compute_move_budget(
                    board,
                    state,
                    my_color,
                    initial_s,
                    increment_s,
                    game_think_time,
                    active_bot_games,
                )
                limit = chess.engine.Limit(time=move_budget_s)
                if self.cfg.max_depth > 0:
                    limit = chess.engine.Limit(time=move_budget_s, depth=self.cfg.max_depth)

                current_rep_summary = self._repetition_summary(board)
                if self._should_log_repetition_focus(board):
                    log_event(
                        "diag",
                        (
                            f"pre-move ply={ply} {current_rep_summary} "
                            f"recent={self._recent_moves_text(board, 12)} "
                            f"fen={board.fen()}"
                        ),
                        game_id,
                    )

                try:
                    result_move, result_info = self._analyse_move(
                        engine,
                        board,
                        limit,
                        my_color,
                        game_id,
                        ply,
                    )
                except Exception as exc:
                    log_event("engine", f"analyse failed: {exc}", game_id)
                    break

                if result_move is None:
                    log_event("engine", "returned no move", game_id)
                    break

                uci = result_move.uci()
                depth = result_info.get("depth")
                nodes = result_info.get("nodes")
                nps = result_info.get("nps")
                seldepth = result_info.get("seldepth")
                time_ms = result_info.get("time")
                pv = result_info.get("pv")
                score_cp, score_mate = self._score_fields(result_info, my_color)
                next_board = board.copy(stack=True)
                next_board.push(result_move)
                next_rep_summary = self._repetition_summary(next_board)
                post_ok, post_status = self._post_move_with_recovery(
                    api,
                    game_id,
                    initial_fen,
                    my_color,
                    ply,
                    uci,
                )
                if post_status != "started":
                    game_status = post_status
                if not post_ok:
                    if post_status == "finished":
                        log_event("game", "stop play after external finish", game_id)
                    break

                last_played_ply = ply
                info_bits = [
                    f"ply={ply}",
                    f"move={uci}",
                    f"budget={move_budget_s:.2f}s",
                    f"clock={remaining_s:.2f}s+{inc_s:.2f}s",
                    f"pre[{current_rep_summary}]",
                    f"post[{next_rep_summary}]",
                ]
                if depth is not None:
                    info_bits.append(f"d={depth}")
                if seldepth is not None:
                    info_bits.append(f"sd={seldepth}")
                if score_mate is not None:
                    info_bits.append(f"mate={score_mate}")
                elif score_cp is not None:
                    info_bits.append(f"cp={score_cp}")
                if nodes is not None:
                    info_bits.append(f"nodes={nodes}")
                if nps is not None:
                    info_bits.append(f"nps={nps}")
                if time_ms is not None:
                    try:
                        info_bits.append(f"time={float(time_ms):.3f}s")
                    except (TypeError, ValueError):
                        info_bits.append(f"time={time_ms}")
                if pv:
                    try:
                        pv_text = " ".join(move.uci() for move in pv[:4])
                        if pv_text:
                            info_bits.append(f"pv={pv_text}")
                    except Exception:
                        pass
                log_event("move", " ".join(info_bits), game_id)
                if self._should_log_repetition_focus(board) or self._should_log_repetition_focus(next_board):
                    log_event(
                        "diag",
                        (
                            f"post-move move={uci} recent={self._recent_moves_text(next_board, 12)} "
                            f"fen={next_board.fen()}"
                        ),
                        game_id,
                    )
        finally:
            try:
                engine.quit()
            except Exception:
                pass
            log_event("game", f"thread exit status={game_status}", game_id)
            self._mark_game_done(game_id, source="thread")


def parse_args() -> BotConfig:
    here = Path(__file__).resolve().parent
    default_engine_path = here / "engine" / "chess_uci"
    if not default_engine_path.exists():
        for candidate in (
            here.parents[2] / "bin" / "chess_uci",
            here.parents[3] / "bin" / "chess_uci",
        ):
            if candidate.exists():
                default_engine_path = candidate
                break
    default_log_file = os.environ.get("LICHESS_BOT_LOG_FILE", str(default_log_path(here)))
    default_backend = os.environ.get("LICHESS_BOT_BACKEND", os.environ.get("CHESS_ENGINE_BACKEND", "classic")).strip().lower() or "classic"
    default_nn_model = default_nn_model_path(here)
    default_book_path = os.environ.get("CHESS_OPENING_BOOK", "")
    if not default_book_path:
        for candidate in (
            here / "opening_book.txt",
            here / "opening_games_100.txt",
            here.parents[2] / "data" / "openings" / "opening_book.txt",
            here.parents[2] / "data" / "openings" / "opening_games_100.txt",
            here.parents[3] / "data" / "openings" / "opening_book.txt",
            here.parents[3] / "data" / "openings" / "opening_games_100.txt",
        ):
            if candidate.exists():
                default_book_path = str(candidate)
                break
    max_games_default = env_int("LICHESS_BOT_MAX_GAMES")
    max_bot_games_default = env_int("LICHESS_BOT_MAX_BOT_GAMES")
    max_human_games_default = env_int("LICHESS_BOT_MAX_HUMAN_GAMES")
    seek_default = env_list("LICHESS_BOT_AUTO_CHALLENGES") or env_list("LICHESS_BOT_SEEKS")
    seek_rated_default = env_bool("LICHESS_BOT_AUTO_CHALLENGE_RATED", env_bool("LICHESS_BOT_SEEK_RATED", False))
    accept_bots_default = env_bool("LICHESS_BOT_ACCEPT_BOTS", False)
    accept_humans_default = env_bool("LICHESS_BOT_ACCEPT_HUMANS", False)
    seek_variant_default = os.environ.get("LICHESS_BOT_AUTO_CHALLENGE_VARIANT",
                                          os.environ.get("LICHESS_BOT_SEEK_VARIANT", "standard"))
    seek_color_default = os.environ.get("LICHESS_BOT_AUTO_CHALLENGE_COLOR",
                                        os.environ.get("LICHESS_BOT_SEEK_COLOR", "random"))
    pair_poll_seconds_default = env_float("LICHESS_BOT_PAIR_POLL_SECONDS")
    min_challenge_gap_default = env_float("LICHESS_BOT_MIN_CHALLENGE_GAP_SECONDS")
    log_max_mb_default = env_float("LICHESS_BOT_LOG_MAX_MB")
    log_keep_files_default = env_int("LICHESS_BOT_LOG_KEEP_FILES")

    parser = argparse.ArgumentParser(description="Run Chess engine as a Lichess bot client.")
    parser.add_argument("--token", default=os.environ.get("LICHESS_BOT_TOKEN", ""), help="Lichess OAuth token (bot:play)")
    parser.add_argument("--base-url", default="https://lichess.org", help="Lichess API base URL")
    parser.add_argument("--engine-path", default=str(default_engine_path), help="Path to UCI engine binary")
    parser.add_argument("--backend", choices=["classic", "nn"], default=default_backend, help="Engine backend")
    parser.add_argument("--nn-model", default=default_nn_model, help="Path to NN inference model binary")
    parser.add_argument("--think-time", type=float, default=0.35, help="Think time per move in seconds")
    parser.add_argument("--max-depth", type=int, default=0, help="Optional depth cap (0 = engine default)")
    parser.add_argument("--max-games", type=int, default=max_games_default, help="Total concurrent game cap")
    parser.add_argument("--max-bot-games", type=int, default=max_bot_games_default, help="Concurrent slots reserved for bot auto-challenges/games")
    parser.add_argument("--max-human-games", type=int, default=max_human_games_default, help="Concurrent slots reserved for direct human challenges")
    parser.add_argument("--accept-bots", action="store_true", default=accept_bots_default, help="Accept bot challenges")
    parser.add_argument("--accept-humans", action="store_true", default=accept_humans_default, help="Accept human challenges")
    parser.add_argument("--bot-match", "--seek", action="append", dest="seek", default=seek_default,
                        help="Outgoing bot challenge spec, e.g. 1+0 or 3+2. Repeatable.")
    parser.add_argument("--bot-match-rated", "--seek-rated", action="store_true", dest="seek_rated",
                        default=seek_rated_default, help="Send rated outgoing bot challenges instead of casual")
    parser.add_argument("--bot-match-variant", "--seek-variant", choices=["standard"], dest="seek_variant",
                        default=seek_variant_default, help="Variant for outgoing bot challenges")
    parser.add_argument("--bot-match-color", "--seek-color", choices=["random", "white", "black"], dest="seek_color",
                        default=seek_color_default, help="Color preference for outgoing bot challenges")
    parser.add_argument("--pair-poll-seconds", type=float, default=pair_poll_seconds_default,
                        help="Seconds between outgoing bot pairing loop polls")
    parser.add_argument("--min-challenge-gap-seconds", type=float, default=min_challenge_gap_default,
                        help="Minimum spacing between outgoing challenge attempts")
    parser.add_argument("--log-file", default=default_log_file, help="Log file path")
    parser.add_argument("--log-max-mb", type=float, default=log_max_mb_default,
                        help="Rotate log after this many MiB (0 disables rotation)")
    parser.add_argument("--log-keep-files", type=int, default=log_keep_files_default,
                        help="How many rotated log files to keep")
    parser.add_argument("--book-path", default=default_book_path, help="Optional opening book file path")

    args = parser.parse_args()
    if not args.token:
        parser.error("missing --token (or LICHESS_BOT_TOKEN env var)")
    if not Path(args.engine_path).exists():
        parser.error(f"engine binary not found: {args.engine_path}")
    if args.backend == "nn" and not args.nn_model:
        parser.error("NN backend requires --nn-model (or CHESS_NN_MODEL / src/core/bot/nn/model/nn_eval.bin)")
    if args.nn_model and not Path(args.nn_model).exists():
        parser.error(f"nn model not found: {args.nn_model}")
    seek_specs = [parse_seek_spec(text, args.seek_rated, args.seek_variant, args.seek_color) for text in args.seek]

    default_total_games = 1 if args.max_games is None else args.max_games
    default_bot_slots = default_total_games if (args.accept_bots or seek_specs) else 0
    default_human_slots = default_total_games if args.accept_humans else 0
    max_bot_games = default_bot_slots if args.max_bot_games is None else args.max_bot_games
    max_human_games = default_human_slots if args.max_human_games is None else args.max_human_games

    if args.max_games is None and (args.max_bot_games is not None or args.max_human_games is not None):
        total_games = max(1, max_bot_games + max_human_games)
    else:
        total_games = default_total_games

    if total_games <= 0:
        parser.error("--max-games must be positive")
    if max_bot_games < 0 or max_human_games < 0:
        parser.error("--max-bot-games and --max-human-games must be non-negative")
    if max_bot_games > total_games or max_human_games > total_games:
        parser.error("per-type game caps cannot exceed --max-games")
    if seek_specs and max_bot_games <= 0:
        parser.error("outgoing bot challenges require at least one bot slot")

    pair_poll_seconds = 6.0 if args.pair_poll_seconds is None else max(1.0, args.pair_poll_seconds)
    min_challenge_gap_seconds = (
        15.0 if args.min_challenge_gap_seconds is None else max(1.0, args.min_challenge_gap_seconds)
    )
    log_max_mb = DEFAULT_LOG_MAX_BYTES / (1024.0 * 1024.0) if args.log_max_mb is None else max(0.0, args.log_max_mb)
    log_keep_files = DEFAULT_LOG_KEEP_FILES if args.log_keep_files is None else max(0, args.log_keep_files)

    return BotConfig(
        token=args.token,
        base_url=args.base_url,
        engine_path=args.engine_path,
        backend=args.backend,
        nn_model_path=args.nn_model,
        think_time=max(0.01, args.think_time),
        max_depth=max(0, args.max_depth),
        max_games=max(1, total_games),
        max_bot_games=max(0, max_bot_games),
        max_human_games=max(0, max_human_games),
        accept_bots=args.accept_bots,
        accept_humans=args.accept_humans,
        log_file=args.log_file,
        log_max_bytes=int(log_max_mb * 1024.0 * 1024.0),
        log_keep_files=log_keep_files,
        book_path=args.book_path,
        seek_specs=seek_specs,
        pair_poll_seconds=pair_poll_seconds,
        min_challenge_gap_seconds=min_challenge_gap_seconds,
    )


def main() -> int:
    cfg = parse_args()
    configure_log_file(cfg.log_file, cfg.log_max_bytes, cfg.log_keep_files)
    if LOG_FILE_PATH is not None:
        log_event("boot", f"logging to {LOG_FILE_PATH}")
    build_info = load_build_info(Path(__file__).resolve().parent)
    if build_info:
        parts = []
        for key in ("release", "built_at", "git_commit", "git_branch", "git_dirty"):
            value = build_info.get(key)
            if value:
                parts.append(f"{key}={value}")
        if parts:
            log_event("boot", "build " + " ".join(parts))
    runner = BotRunner(cfg)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
