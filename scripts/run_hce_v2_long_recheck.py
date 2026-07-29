#!/usr/bin/env python3
"""Run a fresh long confirmation of the saved six-feature HCE-v2 tune."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from run_hce_v2_remote_pipeline import executable_name, summarize_match
except ModuleNotFoundError:
    from scripts.run_hce_v2_remote_pipeline import (
        executable_name,
        summarize_match,
    )


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "current" / "hce_v2"
RUN_DIR = LAB / "v2_long_recheck"
BASELINE = LAB / executable_name("baseline_854193a")
CANDIDATE = LAB / executable_name("candidate_v2")
PRIOR_REPORT = LAB / "v2_gate_240g.json"
REPORT = RUN_DIR / "v2_recheck_480g.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: list[str], name: str) -> None:
    stdout_path = RUN_DIR / f"{name}.stdout.log"
    stderr_path = RUN_DIR / f"{name}.stderr.log"
    log("run: " + subprocess.list2cmdline([str(arg) for arg in args]))
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [str(arg) for arg in args],
            cwd=ROOT,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit {completed.returncode}; "
            f"see {stdout_path.name} and {stderr_path.name}"
        )


def probability(result: dict) -> float:
    paired = result.get("paired_probability_better")
    if paired is not None:
        return float(paired)
    raw = result.get("probability_better")
    return float(raw) if raw is not None else 0.0


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    status_path = RUN_DIR / "status.json"
    status: dict = {
        "status": "running",
        "started_at_unix": int(time.time()),
        "baseline": str(BASELINE),
        "candidate": str(CANDIDATE),
        "prior_report": str(PRIOR_REPORT),
        "promotion": "none",
    }
    write_json(status_path, status)

    try:
        for path in (BASELINE, CANDIDATE, PRIOR_REPORT):
            if not path.exists():
                raise RuntimeError(f"missing required artifact: {path}")

        run(
            [
                sys.executable,
                "scripts/run_hce_position_suite.py",
                "--engine",
                str(CANDIDATE),
                "--backend",
                "classic",
                "--out",
                str(RUN_DIR / "hce_suite.json"),
                "--fail-fast",
            ],
            "hce_suite",
        )
        status["hce_suite"] = "passed"
        write_json(status_path, status)

        run(
            [
                sys.executable,
                "src/core/bot/test_lab.py",
                "--engine",
                f"cand={CANDIDATE}",
                "--engine",
                f"base={BASELINE}",
                "--baseline",
                "base",
                "--positions-count",
                "240",
                "--paired-colors",
                "--think-ms",
                "120",
                "--max-plies",
                "200",
                "--positions-file",
                "data/positions/lichess_equal_positions.fen",
                "--seed",
                "20261201",
                "--concurrency",
                "5",
                "--out",
                str(REPORT),
            ],
            "v2_recheck_480g",
        )
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        result = summarize_match(report, expected_games=480)
        status["fresh_480"] = result

        prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
        status["prior_240"] = summarize_match(prior, expected_games=240)

        confirmed = (
            result["engine_failures"] == 0
            and result["elo_diff"] is not None
            and result["elo_diff"] > 0
            and probability(result) >= 0.90
        )
        status["status"] = "confirmed_candidate" if confirmed else "not_confirmed"
        status["promotion"] = "blocked_pending_review" if confirmed else "none"
        status["finished_at_unix"] = int(time.time())
        write_json(status_path, status)
        log(
            f"fresh 480: {result['elo_diff']:+.1f} Elo, "
            f"P={probability(result):.1%}; status={status['status']}"
        )
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        status["finished_at_unix"] = int(time.time())
        write_json(status_path, status)
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
