#!/usr/bin/env python3
"""Screen pure-HCE search selectivity settings against a frozen binary.

The same engine executable is used for candidate and baseline. Candidate-only
UCI options isolate each search setting without changing or rebuilding source.
Short screens share one opening seed; larger gates use fresh seeds.
"""

from __future__ import annotations

import json
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
SWEEP = LAB / "search_knob_sweep"
BASELINE = LAB / executable_name("baseline_854193a")

VARIANTS: OrderedDict[str, dict[str, str]] = OrderedDict(
    [
        ("control", {}),
        ("rfp70", {"HceRfpMargin": "70"}),
        ("rfp110", {"HceRfpMargin": "110"}),
        ("rfp140", {"HceRfpMargin": "140"}),
        ("null_base1", {"HceNullBase": "1"}),
        ("null_div5", {"HceNullDepthDivisor": "5"}),
        ("null_div6", {"HceNullDepthDivisor": "6"}),
        (
            "conservative_combo",
            {
                "HceRfpMargin": "110",
                "HceNullBase": "1",
                "HceNullDepthDivisor": "5",
            },
        ),
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
    stdout_path = SWEEP / f"{name}.stdout.log"
    stderr_path = SWEEP / f"{name}.stderr.log"
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


def option_args(candidate_name: str, options: dict[str, str]) -> list[str]:
    args: list[str] = []
    for option, value in options.items():
        args.extend(["--uci-option", f"{candidate_name}:{option}={value}"])
    return args


def run_match(
    name: str,
    options: dict[str, str],
    *,
    games: int,
    positions: int,
    seed: int,
) -> dict:
    report_path = SWEEP / f"{name}.json"
    args = [
        sys.executable,
        "src/core/bot/test_lab.py",
        "--engine",
        f"cand={BASELINE}",
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
    args.extend(option_args("cand", options))
    run(args, name)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return summarize_match(report, expected_games=games)


def run_hce_suite(name: str, options: dict[str, str]) -> None:
    report_path = SWEEP / f"hce_suite_{name}.json"
    args = [
        sys.executable,
        "scripts/run_hce_position_suite.py",
        "--engine",
        str(BASELINE),
        "--backend",
        "classic",
        "--out",
        str(report_path),
        "--fail-fast",
    ]
    for option, value in options.items():
        args.extend(["--uci-option", f"{option}={value}"])
    run(args, f"hce_suite_{name}")


def probability(result: dict) -> float:
    paired = result.get("paired_probability_better")
    if paired is not None:
        return float(paired)
    raw = result.get("probability_better")
    return float(raw) if raw is not None else 0.0


def main() -> int:
    SWEEP.mkdir(parents=True, exist_ok=True)
    status_path = SWEEP / "status.json"
    status: dict = {
        "status": "running",
        "started_at_unix": int(time.time()),
        "baseline": str(BASELINE),
        "variants": VARIANTS,
        "screens": {},
        "gates_120": {},
        "promotion": "none",
    }
    write_json(status_path, status)

    try:
        if not BASELINE.exists():
            raise RuntimeError(f"missing frozen baseline: {BASELINE}")

        run(["make", "test"], "make_test")

        for name, options in VARIANTS.items():
            result = run_match(
                f"screen_{name}_60g",
                options,
                games=60,
                positions=30,
                seed=20260821,
            )
            status["screens"][name] = result
            write_json(status_path, status)
            log(
                f"screen {name}: {result['elo_diff']:+.1f} Elo, "
                f"P={probability(result):.1%}"
            )

        ranked = sorted(
            (
                (name, result)
                for name, result in status["screens"].items()
                if name != "control"
                and result["engine_failures"] == 0
                and result["elo_diff"] is not None
                and result["elo_diff"] > 0
                and probability(result) >= 0.70
            ),
            key=lambda item: (
                probability(item[1]),
                item[1]["elo_diff"],
            ),
            reverse=True,
        )[:2]
        status["shortlist"] = [name for name, _ in ranked]
        write_json(status_path, status)

        for index, (name, _) in enumerate(ranked):
            options = VARIANTS[name]
            run_hce_suite(name, options)
            result = run_match(
                f"gate_{name}_120g",
                options,
                games=120,
                positions=60,
                seed=20260830 + index,
            )
            status["gates_120"][name] = result
            write_json(status_path, status)

        finalists = sorted(
            (
                (name, result)
                for name, result in status["gates_120"].items()
                if result["engine_failures"] == 0
                and result["elo_diff"] is not None
                and result["elo_diff"] > 0
                and probability(result) >= 0.75
            ),
            key=lambda item: (
                probability(item[1]),
                item[1]["elo_diff"],
            ),
            reverse=True,
        )
        if not finalists:
            status["status"] = "no_confirmable_candidate"
            status["finished_at_unix"] = int(time.time())
            write_json(status_path, status)
            return 0

        finalist = finalists[0][0]
        result_240 = run_match(
            f"confirm_{finalist}_240g",
            VARIANTS[finalist],
            games=240,
            positions=120,
            seed=20260910,
        )
        status["confirmation_240"] = {
            "candidate": finalist,
            **result_240,
        }
        confirmed = (
            result_240["engine_failures"] == 0
            and result_240["elo_diff"] is not None
            and result_240["elo_diff"] > 0
            and probability(result_240) >= 0.90
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
        raise


if __name__ == "__main__":
    raise SystemExit(main())
