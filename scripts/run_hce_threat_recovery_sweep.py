#!/usr/bin/env python3
"""Recover conservative candidates from the over-aggressive threat tune."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

from run_hce_v2_remote_pipeline import (
    LAB,
    ROOT,
    copy_first_lines,
    executable_name,
    summarize_match,
    write_json_atomic,
)


OUT = LAB / "threat_recovery"
SOURCE_TUNE = LAB / "threat_v3" / "median_tuned.txt"
BASELINE = LAB / executable_name("baseline_854193a")
N_TUNED = 827
SAFE_PUSH_START = 55
SAFE_PUSH_END = 59
VARIANTS = OrderedDict(
    [
        ("direct_only", "zero_safe_push"),
        ("half_safe_push", "half_safe_push"),
    ]
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(
    args: list[str],
    name: str,
    *,
    input_text: str | None = None,
) -> None:
    stdout_path = OUT / f"{name}.stdout.log"
    stderr_path = OUT / f"{name}.stderr.log"
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


def parse_tuned(path: Path) -> list[int]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("TUNED ")
    ]
    if not lines:
        raise RuntimeError(f"no TUNED line in {path}")
    values = [int(value) for value in lines[-1].split()[1:]]
    if len(values) != N_TUNED:
        raise RuntimeError(
            f"{path} has {len(values)} weights, expected {N_TUNED}"
        )
    return values


def variant_vector(source: list[int], mode: str) -> list[int]:
    values = source.copy()
    if mode == "zero_safe_push":
        values[SAFE_PUSH_START:SAFE_PUSH_END] = [0, 0, 0, 0]
    elif mode == "half_safe_push":
        values[SAFE_PUSH_START:SAFE_PUSH_END] = [
            int(value / 2)
            for value in values[SAFE_PUSH_START:SAFE_PUSH_END]
        ]
    else:
        raise RuntimeError(f"unknown threat recovery mode: {mode}")
    return values


def probability(result: dict) -> float:
    paired = result.get("paired_probability_better")
    if paired is not None:
        return float(paired)
    raw = result.get("probability_better")
    return float(raw) if raw is not None else 0.0


def build_candidates(status: dict) -> None:
    eval_source = ROOT / "src" / "core" / "engine" / "hce_eval.c"
    backup = OUT / "hce_eval.zero.c"
    shutil.copy2(eval_source, backup)
    source_vector = parse_tuned(SOURCE_TUNE)
    verify_positions = OUT / "verify_positions.txt"
    copy_first_lines(
        LAB / "combined_positions.txt",
        verify_positions,
        limit=5000,
    )

    try:
        for name, mode in VARIANTS.items():
            vector = variant_vector(source_vector, mode)
            tune_path = OUT / f"{name}_tuned.txt"
            tune_path.write_text(
                "TUNED " + " ".join(str(value) for value in vector) + "\n",
                encoding="utf-8",
            )
            shutil.copyfile(backup, eval_source)
            os.utime(eval_source, None)
            run(
                [
                    sys.executable,
                    "scripts/texel_apply_tune.py",
                    "--eval-c",
                    str(eval_source),
                    "--tuned-file",
                    str(tune_path),
                ],
                f"apply_{name}",
            )
            run(
                ["make", "-B", "bin/chess_uci", "-j5"],
                f"build_{name}",
            )
            candidate = OUT / executable_name(name)
            shutil.copy2(
                ROOT / "bin" / executable_name("chess_uci"),
                candidate,
            )
            features = OUT / f"{name}_verify_features.txt"
            run(
                [str(candidate)],
                f"tunedump_{name}",
                input_text=f"tunedump {verify_positions} {features}\nquit\n",
            )
            run(
                [
                    sys.executable,
                    "scripts/texel_tune.py",
                    "--feats",
                    str(features),
                    "--initial-tuned-file",
                    str(tune_path),
                    "--only-v3-features",
                    "--iters",
                    "0",
                ],
                f"exact_verify_{name}",
            )
            run(["make", "test"], f"make_test_{name}")
            run(
                [
                    sys.executable,
                    "scripts/run_hce_position_suite.py",
                    "--engine",
                    str(candidate),
                    "--backend",
                    "classic",
                    "--out",
                    str(OUT / f"hce_suite_{name}.json"),
                    "--fail-fast",
                ],
                f"hce_suite_{name}",
            )
            status["candidate_tests"][name] = {
                "exact_reconstruction": True,
                "make_test": "passed",
                "hce_suite": "passed",
                "safe_push_weights": vector[
                    SAFE_PUSH_START:SAFE_PUSH_END
                ],
            }
            write_json_atomic(OUT / "status.json", status)
    finally:
        shutil.copyfile(backup, eval_source)
        os.utime(eval_source, None)
        run(
            ["make", "-B", "bin/chess_uci", "-j5"],
            "restore_default",
        )


def run_match(
    candidate_name: str,
    *,
    games: int,
    positions: int,
    seed: int,
    stage: str,
) -> dict:
    report_path = OUT / f"{stage}_{candidate_name}_{games}g.json"
    run(
        [
            sys.executable,
            "src/core/bot/test_lab.py",
            "--engine",
            f"cand={OUT / executable_name(candidate_name)}",
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
        ],
        f"{stage}_{candidate_name}_{games}g",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return summarize_match(report, expected_games=games)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    status: dict = {
        "status": "running",
        "started_at_unix": int(time.time()),
        "source_tune": str(SOURCE_TUNE),
        "baseline": str(BASELINE),
        "candidate_tests": {},
        "screens_60": {},
        "gates_120": {},
        "promotion": "none",
    }
    write_json_atomic(status_path, status)
    try:
        if not SOURCE_TUNE.exists():
            raise RuntimeError(f"missing source tune: {SOURCE_TUNE}")
        if not BASELINE.exists():
            raise RuntimeError(f"missing frozen baseline: {BASELINE}")
        build_candidates(status)

        for name in VARIANTS:
            result = run_match(
                name,
                games=60,
                positions=30,
                seed=20261331,
                stage="screen",
            )
            status["screens_60"][name] = result
            write_json_atomic(status_path, status)
            log(
                f"screen {name}: {result['elo_diff']:+.1f} Elo, "
                f"P={probability(result):.1%}"
            )

        survivors = [
            name
            for name, result in status["screens_60"].items()
            if result["engine_failures"] == 0
            and result["elo_diff"] is not None
            and result["elo_diff"] > -50.0
            and probability(result) >= 0.15
        ]
        for index, name in enumerate(survivors):
            result = run_match(
                name,
                games=120,
                positions=60,
                seed=20261341 + index,
                stage="gate",
            )
            status["gates_120"][name] = result
            write_json_atomic(status_path, status)

        finalists = sorted(
            (
                (name, result)
                for name, result in status["gates_120"].items()
                if result["engine_failures"] == 0
                and result["elo_diff"] is not None
                and result["elo_diff"] > 0.0
                and probability(result) >= 0.75
            ),
            key=lambda item: (probability(item[1]), item[1]["elo_diff"]),
            reverse=True,
        )
        if not finalists:
            status["status"] = "no_confirmable_candidate"
            status["finished_at_unix"] = int(time.time())
            write_json_atomic(status_path, status)
            return 0

        finalist = finalists[0][0]
        confirmation = run_match(
            finalist,
            games=240,
            positions=120,
            seed=20261351,
            stage="confirm",
        )
        status["confirmation_240"] = {
            "candidate": finalist,
            **confirmation,
        }
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
        write_json_atomic(status_path, status)
        return 0
    except BaseException as exc:
        status["status"] = "failed"
        status["finished_at_unix"] = int(time.time())
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(status_path, status)
        log(status["error"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
