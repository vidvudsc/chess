#!/usr/bin/env python3
"""Screen a four-entry clustered transposition table for pure HCE."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
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
OUT = LAB / "tt_cluster_sweep"
BASELINE = LAB / executable_name("baseline_854193a")
CANDIDATE = OUT / executable_name("tt_cluster_candidate")
VARIANTS: OrderedDict[str, dict[str, str]] = OrderedDict(
    [
        ("control", {}),
        ("cluster4", {"HceTtClusterMode": "1"}),
    ]
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: list[str], name: str) -> None:
    stdout_path = OUT / f"{name}.stdout.log"
    stderr_path = OUT / f"{name}.stderr.log"
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


def option_args(options: dict[str, str]) -> list[str]:
    args: list[str] = []
    for option, value in options.items():
        args.extend(["--uci-option", f"cand:{option}={value}"])
    return args


def probability(result: dict) -> float:
    paired = result.get("paired_probability_better")
    if paired is not None:
        return float(paired)
    raw = result.get("probability_better")
    return float(raw) if raw is not None else 0.0


def build_and_test() -> None:
    run(["make", "-B", "bin/chess_uci", "-j5"], "build_candidate")
    shutil.copy2(
        ROOT / "bin" / executable_name("chess_uci"),
        CANDIDATE,
    )
    run(
        [
            sys.executable,
            "scripts/compare_uci_fixed_depth.py",
            "--engine-a",
            str(BASELINE),
            "--engine-b",
            str(CANDIDATE),
            "--positions-file",
            "data/positions/lichess_equal_positions.fen",
            "--depth",
            "9",
            "--count",
            "20",
        ],
        "fixed_depth_identity",
    )
    run(["make", "test"], "make_test")
    for name, options in VARIANTS.items():
        args = [
            sys.executable,
            "scripts/run_hce_position_suite.py",
            "--engine",
            str(CANDIDATE),
            "--backend",
            "classic",
            "--out",
            str(OUT / f"hce_suite_{name}.json"),
            "--fail-fast",
        ]
        for option, value in options.items():
            args.extend(["--uci-option", f"{option}={value}"])
        run(args, f"hce_suite_{name}")


def run_match(
    name: str,
    options: dict[str, str],
    *,
    games: int,
    positions: int,
    seed: int,
    stage: str,
) -> dict:
    report_path = OUT / f"{stage}_{name}_{games}g.json"
    args = [
        sys.executable,
        "src/core/bot/test_lab.py",
        "--engine",
        f"cand={CANDIDATE}",
        "--engine",
        f"base={BASELINE}",
        "--baseline",
        "base",
        "--positions-count",
        str(positions),
        "--paired-colors",
        "--think-ms",
        "120",
        "--max-plies",
        "200",
        "--positions-file",
        "data/positions/lichess_equal_positions.fen",
        "--seed",
        str(seed),
        "--concurrency",
        "5",
        "--out",
        str(report_path),
    ]
    args.extend(option_args(options))
    run(args, f"{stage}_{name}_{games}g")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return summarize_match(report, expected_games=games)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    status: dict = {
        "status": "running",
        "started_at_unix": int(time.time()),
        "baseline": str(BASELINE),
        "candidate": str(CANDIDATE),
        "variants": VARIANTS,
        "screens_60": {},
        "promotion": "none",
    }
    write_json(status_path, status)
    try:
        if not BASELINE.exists():
            raise RuntimeError(f"missing frozen baseline: {BASELINE}")
        build_and_test()
        status["build_and_tests"] = "passed"
        write_json(status_path, status)

        for name, options in VARIANTS.items():
            result = run_match(
                name,
                options,
                games=60,
                positions=30,
                seed=20261421,
                stage="screen",
            )
            status["screens_60"][name] = result
            write_json(status_path, status)
            log(
                f"screen {name}: {result['elo_diff']:+.1f} Elo, "
                f"P={probability(result):.1%}"
            )

        screen = status["screens_60"]["cluster4"]
        if (
            screen["engine_failures"] != 0
            or screen["elo_diff"] is None
            or screen["elo_diff"] <= -50.0
            or probability(screen) < 0.15
        ):
            status["status"] = "rejected_at_60"
            status["finished_at_unix"] = int(time.time())
            write_json(status_path, status)
            return 0

        gate = run_match(
            "cluster4",
            VARIANTS["cluster4"],
            games=120,
            positions=60,
            seed=20261422,
            stage="gate",
        )
        status["gate_120"] = gate
        write_json(status_path, status)
        if (
            gate["engine_failures"] != 0
            or gate["elo_diff"] is None
            or gate["elo_diff"] <= 0.0
            or probability(gate) < 0.75
        ):
            status["status"] = "not_confirmed_at_120"
            status["finished_at_unix"] = int(time.time())
            write_json(status_path, status)
            return 0

        confirmation = run_match(
            "cluster4",
            VARIANTS["cluster4"],
            games=240,
            positions=120,
            seed=20261423,
            stage="confirm",
        )
        status["confirmation_240"] = confirmation
        confirmed = (
            confirmation["engine_failures"] == 0
            and confirmation["elo_diff"] is not None
            and confirmation["elo_diff"] > 0.0
            and probability(confirmation) >= 0.90
        )
        status["status"] = (
            "confirmed_positive_not_promoted"
            if confirmed
            else "not_confirmed"
        )
        status["finished_at_unix"] = int(time.time())
        write_json(status_path, status)
        return 0
    except BaseException as exc:
        status["status"] = "failed"
        status["finished_at_unix"] = int(time.time())
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(status_path, status)
        log(status["error"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
