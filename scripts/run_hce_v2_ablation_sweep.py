#!/usr/bin/env python3
"""Build and screen focused ablations of the saved HCE-v2 Texel tune.

This controller is intended for the isolated Windows ML-PC checkout. It never
deploys or replaces the frozen baseline. Short screens use a shared seed so the
variants see the same positions; longer gates use fresh seeds.
"""

from __future__ import annotations

import json
import os
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
SWEEP = LAB / "ablation_sweep"
EVAL_SOURCE = ROOT / "src" / "core" / "engine" / "hce_eval.c"
V2_START = 35
V2_COUNT = 12

VARIANTS: OrderedDict[str, tuple[int, ...]] = OrderedDict(
    [
        ("zero_control", ()),
        ("connected_only", (0, 1)),
        ("connected_phalanx", (0, 1, 2, 3)),
        ("pawn_trio", (0, 1, 2, 3, 4, 5)),
        ("non_pawn", (6, 7, 8, 9, 10, 11)),
        ("combined_no_phalanx", (0, 1, 4, 5, 6, 7, 8, 9, 10, 11)),
        ("combined_no_backward", (0, 1, 2, 3, 6, 7, 8, 9, 10, 11)),
        ("combined_all", tuple(range(V2_COUNT))),
    ]
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    args: list[str],
    name: str,
    *,
    input_text: str | None = None,
) -> None:
    stdout_path = SWEEP / f"{name}.stdout.log"
    stderr_path = SWEEP / f"{name}.stderr.log"
    log("run: " + subprocess.list2cmdline([str(arg) for arg in args]))
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [str(arg) for arg in args],
            cwd=ROOT,
            input=input_text,
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


def read_tuned_vector(path: Path) -> list[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("TUNED "):
            values = [int(value) for value in line.split()[1:]]
            if len(values) != 815:
                raise RuntimeError(
                    f"{path} contains {len(values)} tuned values, expected 815"
                )
            return values
    raise RuntimeError(f"{path} contains no TUNED line")


def write_variant(
    name: str,
    median: list[int],
    enabled_indices: tuple[int, ...],
) -> Path:
    values = median.copy()
    values[V2_START:V2_START + V2_COUNT] = [0] * V2_COUNT
    for index in enabled_indices:
        values[V2_START + index] = median[V2_START + index]
    path = SWEEP / f"{name}.tuned.txt"
    path.write_text(
        "TUNED " + " ".join(str(value) for value in values) + "\n",
        encoding="utf-8",
    )
    return path


def build_candidates(median: list[int]) -> dict[str, dict[str, str]]:
    backup = SWEEP / "hce_eval.zero_weights.c"
    shutil.copy2(EVAL_SOURCE, backup)
    built: dict[str, dict[str, str]] = {}
    verify_positions = LAB / "candidate_verify_positions.txt"
    if not verify_positions.exists():
        raise RuntimeError(f"missing verification sample: {verify_positions}")

    try:
        for name, enabled in VARIANTS.items():
            tuned_path = write_variant(name, median, enabled)
            run(
                [
                    sys.executable,
                    "scripts/texel_apply_tune.py",
                    "--eval-c",
                    str(EVAL_SOURCE),
                    "--tuned-file",
                    str(tuned_path),
                ],
                f"apply_{name}",
            )
            run(["make", "bin/chess_uci", "-j6"], f"build_{name}")
            candidate = SWEEP / executable_name(name)
            shutil.copy2(
                ROOT / "bin" / executable_name("chess_uci"),
                candidate,
            )

            features = SWEEP / f"{name}.verify.features.txt"
            command = f"tunedump {verify_positions} {features}\nquit\n"
            run(
                [str(candidate)],
                f"verify_dump_{name}",
                input_text=command,
            )
            run(
                [
                    sys.executable,
                    "scripts/texel_tune.py",
                    "--feats",
                    str(features),
                    "--initial-tuned-file",
                    str(tuned_path),
                    "--only-v2-features",
                    "--iters",
                    "0",
                ],
                f"verify_exact_{name}",
            )
            built[name] = {
                "binary": str(candidate),
                "tuned_file": str(tuned_path),
            }
    finally:
        shutil.copyfile(backup, EVAL_SOURCE)
        os.utime(EVAL_SOURCE, None)
        run(["make", "-B", "bin/chess_uci", "-j6"], "restore_zero_engine")
    return built


def run_match(
    name: str,
    candidate: Path,
    *,
    games: int,
    positions: int,
    seed: int,
) -> dict:
    report_path = SWEEP / f"{name}.json"
    run(
        [
            sys.executable,
            "src/core/bot/test_lab.py",
            "--engine",
            f"cand={candidate}",
            "--engine",
            f"base={LAB / executable_name('baseline_854193a')}",
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
        ],
        name,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return summarize_match(report, expected_games=games)


def test_shortlisted(name: str, tuned_file: Path) -> None:
    backup = SWEEP / "hce_eval.pretest.zero_weights.c"
    shutil.copy2(EVAL_SOURCE, backup)
    try:
        run(
            [
                sys.executable,
                "scripts/texel_apply_tune.py",
                "--eval-c",
                str(EVAL_SOURCE),
                "--tuned-file",
                str(tuned_file),
            ],
            f"apply_test_{name}",
        )
        run(["make", "-B", "bin/chess_uci", "-j6"], f"rebuild_test_{name}")
        run(["make", "test"], f"make_test_{name}")
        run(["make", "hce_suite"], f"hce_suite_{name}")
    finally:
        shutil.copyfile(backup, EVAL_SOURCE)
        os.utime(EVAL_SOURCE, None)
        run(["make", "-B", "bin/chess_uci", "-j6"], "restore_after_tests")


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
        "baseline": str(LAB / executable_name("baseline_854193a")),
        "screens": {},
        "gates_120": {},
    }
    write_json(status_path, status)

    try:
        median = read_tuned_vector(LAB / "v2_median_tuned.txt")
        status["candidates"] = build_candidates(median)
        write_json(status_path, status)

        for index, name in enumerate(VARIANTS):
            result = run_match(
                f"screen_{name}_60g",
                Path(status["candidates"][name]["binary"]),
                games=60,
                positions=30,
                seed=20260801,
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
                if result["engine_failures"] == 0
                and result["elo_diff"] is not None
                and result["elo_diff"] > 0
                and probability(result) >= 0.70
                and name != "zero_control"
            ),
            key=lambda item: (
                probability(item[1]),
                item[1]["elo_diff"],
            ),
            reverse=True,
        )[:2]

        for index, (name, _) in enumerate(ranked):
            tuned_file = Path(status["candidates"][name]["tuned_file"])
            test_shortlisted(name, tuned_file)
            result = run_match(
                f"gate_{name}_120g",
                Path(status["candidates"][name]["binary"]),
                games=120,
                positions=60,
                seed=20260810 + index,
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
            status["promotion"] = "none"
            write_json(status_path, status)
            return 0

        finalist = finalists[0][0]
        result_240 = run_match(
            f"confirm_{finalist}_240g",
            Path(status["candidates"][finalist]["binary"]),
            games=240,
            positions=120,
            seed=20260820,
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
        status["promotion"] = "none"
        write_json(status_path, status)
        return 0
    except BaseException as exc:
        status["status"] = "failed"
        status["finished_at_unix"] = int(time.time())
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["promotion"] = "none"
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
