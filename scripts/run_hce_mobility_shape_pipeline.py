#!/usr/bin/env python3
"""Tune and Elo-test nonlinear pure-HCE mobility corrections."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_hce_v2_remote_pipeline import (
    LAB,
    ROOT,
    copy_first_lines,
    executable_name,
    summarize_match,
    write_json_atomic,
)


OUT = LAB / "mobility_shape"
N_SHAPE = 16
SHAPE_NAMES = [
    "restricted_mob_n_mg", "restricted_mob_n_eg",
    "restricted_mob_b_mg", "restricted_mob_b_eg",
    "restricted_mob_r_mg", "restricted_mob_r_eg",
    "restricted_mob_q_mg", "restricted_mob_q_eg",
    "active_mob_n_mg", "active_mob_n_eg",
    "active_mob_b_mg", "active_mob_b_eg",
    "active_mob_r_mg", "active_mob_r_eg",
    "active_mob_q_mg", "active_mob_q_eg",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class MobilityShapePipeline:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.status_path = OUT / "status.json"
        self.summary = {
            "status": "running",
            "started_at_unix": int(time.time()),
            "feature_family": "nonlinear mobility shape",
            "stages": {},
            "promotion": "none",
        }
        write_json_atomic(self.status_path, self.summary)

    def mark(self, stage: str, **details) -> None:
        self.summary["stages"][stage] = {
            "completed_at_unix": int(time.time()),
            **details,
        }
        write_json_atomic(self.status_path, self.summary)
        log(f"completed stage: {stage}")

    def run(
        self,
        args: list[str],
        name: str,
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

    def dump_features(self) -> None:
        positions = LAB / "combined_positions.txt"
        groups_raw = LAB / "combined_groups_raw.txt"
        if not positions.exists() or not groups_raw.exists():
            raise RuntimeError("the grouped HCE-v2 corpus is missing")
        self.run(["make", "-B", "bin/chess_uci", "-j5"], "build_feature_engine")
        feature_engine = OUT / executable_name("feature_engine")
        shutil.copy2(ROOT / "bin" / executable_name("chess_uci"), feature_engine)
        features = OUT / "features.txt"
        groups = OUT / "groups.txt"
        self.run(
            [str(feature_engine)],
            "quiet_tunedump",
            input_text=(
                f"tunedump {positions} {features} {groups_raw} {groups}\n"
                "quit\n"
            ),
        )
        feature_rows = count_nonempty_lines(features)
        group_rows = count_nonempty_lines(groups)
        if feature_rows != group_rows or feature_rows < 50000:
            raise RuntimeError(
                f"invalid quiet feature/group rows: {feature_rows}/{group_rows}"
            )
        self.run(
            [
                sys.executable,
                "scripts/texel_tune_mobility_shape.py",
                "--feats", str(features),
                "--initial-tuned-file", str(LAB / "baseline_tune.log"),
                "--verify-only",
            ],
            "zero_weight_exact_verify",
        )
        self.mark("features", positions=feature_rows, exact_reconstruction=True)

    def tune(self) -> None:
        vectors: list[list[int]] = []
        for seed in (41, 42, 43):
            name = f"tune_seed{seed}"
            self.run(
                [
                    sys.executable,
                    "scripts/texel_tune_mobility_shape.py",
                    "--feats", str(OUT / "features.txt"),
                    "--groups", str(OUT / "groups.txt"),
                    "--initial-tuned-file", str(LAB / "baseline_tune.log"),
                    "--iters", "5000",
                    "--l2", "0.5",
                    "--seed", str(seed),
                ],
                name,
            )
            vectors.append(parse_shape_vector(OUT / f"{name}.stdout.log"))
        median = [
            sorted(vector[index] for vector in vectors)[1]
            for index in range(N_SHAPE)
        ]
        shape_path = OUT / "median_shape.txt"
        shape_path.write_text(
            "MOBILITY_SHAPE " +
            " ".join(str(value) for value in median) +
            "\n",
            encoding="utf-8",
        )
        stability = {
            name: [vector[index] for vector in vectors]
            for index, name in enumerate(SHAPE_NAMES)
        }
        self.mark(
            "tuning",
            median_weights=dict(zip(SHAPE_NAMES, median)),
            seed_weights=stability,
        )

    def build_and_test_candidate(self) -> None:
        eval_source = ROOT / "src" / "core" / "engine" / "hce_eval.c"
        backup = OUT / "hce_eval.zero.c"
        shutil.copy2(eval_source, backup)
        try:
            self.run(
                [
                    sys.executable,
                    "scripts/texel_apply_mobility_shape.py",
                    "--eval-c", str(eval_source),
                    "--shape-file", str(OUT / "median_shape.txt"),
                ],
                "apply_median",
            )
            self.run(["make", "-B", "bin/chess_uci", "-j5"], "build_candidate")
            candidate = OUT / executable_name("candidate")
            shutil.copy2(ROOT / "bin" / executable_name("chess_uci"), candidate)
            self.verify_candidate(candidate)
            self.run(["make", "test"], "make_test")
            self.run(
                [
                    sys.executable,
                    "scripts/run_hce_position_suite.py",
                    "--engine", str(candidate),
                    "--backend", "classic",
                    "--out", str(OUT / "hce_suite.json"),
                    "--fail-fast",
                ],
                "hce_suite",
            )
        finally:
            shutil.copyfile(backup, eval_source)
            os.utime(eval_source, None)
            self.run(["make", "-B", "bin/chess_uci", "-j5"], "restore_default")
        self.mark(
            "candidate_tests",
            exact_reconstruction=True,
            make_test="passed",
            hce_suite="passed",
        )

    def verify_candidate(self, candidate: Path) -> None:
        positions = OUT / "verify_positions.txt"
        copy_first_lines(LAB / "combined_positions.txt", positions, limit=5000)
        features = OUT / "verify_features.txt"
        self.run(
            [str(candidate)],
            "candidate_tunedump",
            input_text=f"tunedump {positions} {features}\nquit\n",
        )
        self.run(
            [
                sys.executable,
                "scripts/texel_tune_mobility_shape.py",
                "--feats", str(features),
                "--initial-tuned-file", str(LAB / "baseline_tune.log"),
                "--initial-shape-file", str(OUT / "median_shape.txt"),
                "--verify-only",
            ],
            "candidate_exact_verify",
        )

    def run_elo_gates(self) -> None:
        screen = self.run_match(60, 30, 20261321, "screen_60g")
        self.mark("elo_60", **screen)
        probability = paired_probability(screen)
        if (
            screen["engine_failures"] != 0
            or screen["elo_diff"] is None
            or screen["elo_diff"] <= -50.0
            or (probability is not None and probability < 0.15)
        ):
            self.finish("rejected_at_60")
            return

        gate = self.run_match(120, 60, 20261322, "gate_120g")
        self.mark("elo_120", **gate)
        if (
            gate["engine_failures"] != 0
            or gate["elo_diff"] is None
            or gate["elo_diff"] <= -50.0
        ):
            self.finish("rejected_at_120")
            return

        confirmation = self.run_match(240, 120, 20261323, "confirm_240g")
        self.mark("elo_240", **confirmation)
        probability = paired_probability(confirmation)
        confirmed = (
            confirmation["engine_failures"] == 0
            and confirmation["elo_diff"] is not None
            and confirmation["elo_diff"] > 0.0
            and probability is not None
            and probability >= 0.90
        )
        self.finish(
            "confirmed_positive_not_promoted" if confirmed else "not_confirmed"
        )

    def run_match(
        self,
        games: int,
        positions: int,
        seed: int,
        name: str,
    ) -> dict:
        report_path = OUT / f"{name}.json"
        self.run(
            [
                sys.executable,
                "src/core/bot/test_lab.py",
                "--engine", f"cand={OUT / executable_name('candidate')}",
                "--engine", f"base={LAB / executable_name('baseline_854193a')}",
                "--baseline", "base",
                "--positions-count", str(positions),
                "--paired-colors",
                "--think-ms", "120",
                "--max-plies", "200",
                "--positions-file", "data/positions/lichess_equal_positions.fen",
                "--seed", str(seed),
                "--concurrency", "5",
                "--out", str(report_path),
            ],
            name,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return summarize_match(report, expected_games=games)

    def finish(self, status: str) -> None:
        self.summary["status"] = status
        self.summary["finished_at_unix"] = int(time.time())
        self.summary["promotion"] = "none"
        write_json_atomic(self.status_path, self.summary)
        log(f"pipeline finished: {status}")

    def fail(self, exc: BaseException) -> None:
        self.summary["status"] = "failed"
        self.summary["finished_at_unix"] = int(time.time())
        self.summary["error"] = f"{type(exc).__name__}: {exc}"
        self.summary["promotion"] = "none"
        write_json_atomic(self.status_path, self.summary)
        log(self.summary["error"])


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(1 for line in stream if line.strip())


def parse_shape_vector(path: Path) -> list[int]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("MOBILITY_SHAPE ")
    ]
    if not lines:
        raise ValueError(f"no MOBILITY_SHAPE line in {path}")
    vector = [int(value) for value in lines[-1].split()[1:]]
    if len(vector) != N_SHAPE:
        raise ValueError(
            f"{path} has {len(vector)} values, expected {N_SHAPE}"
        )
    return vector


def paired_probability(result: dict) -> float | None:
    paired = result.get("paired_probability_better")
    return paired if paired is not None else result.get("probability_better")


def main() -> int:
    pipeline = MobilityShapePipeline()
    try:
        pipeline.dump_features()
        pipeline.tune()
        pipeline.build_and_test_candidate()
        pipeline.run_elo_gates()
    except BaseException as exc:
        pipeline.fail(exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
