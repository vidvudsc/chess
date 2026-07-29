#!/usr/bin/env python3
"""Retune the established HCE scalar/PST evaluation on the large corpus.

This unattended ML-PC controller waits for the preceding search sweep, fits
three game-grouped mini-batch runs, takes their coordinate-wise median, verifies
the exact engine reconstruction, runs all tests, and applies paired Elo gates.
It never replaces the frozen baseline or deploys a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
SWEEP = LAB / "established_retune"
BASELINE = LAB / executable_name("baseline_854193a")
CANDIDATE = SWEEP / executable_name("established_retune_candidate")
EVAL_SOURCE = ROOT / "src" / "core" / "engine" / "hce_eval.c"
N_TUNED = 815
N_CURRENT_SCALAR = 21
N_V2_SCALAR = 26
BISHOP_EG_MOBILITY_INDEX = 12


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: list[str], name: str, *, input_text: str | None = None) -> None:
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


def parse_tuned_vector(path: Path) -> list[int]:
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
            f"{path} has {len(values)} tuned values, expected {N_TUNED}"
        )
    return values


def parse_baseline_vector(path: Path) -> list[int]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("TUNED ")
    ]
    if not lines:
        raise RuntimeError(f"no TUNED line in {path}")
    values = [int(value) for value in lines[-1].split()[1:]]
    previous_length = N_TUNED - N_V2_SCALAR
    if len(values) == previous_length:
        values = (
            values[:N_CURRENT_SCALAR]
            + [0] * N_V2_SCALAR
            + values[N_CURRENT_SCALAR:]
        )
    if len(values) != N_TUNED:
        raise RuntimeError(
            f"{path} has {len(values)} tuned values, expected "
            f"{previous_length} or {N_TUNED}"
        )
    return values


def median_vector(vectors: list[list[int]]) -> list[int]:
    if len(vectors) != 3:
        raise RuntimeError("median tune requires exactly three vectors")
    return [
        sorted(vector[index] for vector in vectors)[1]
        for index in range(N_TUNED)
    ]


def apply_behavioral_guards(
    tuned: list[int],
    baseline: list[int],
) -> tuple[list[int], list[dict[str, int | str]]]:
    if len(tuned) != N_TUNED or len(baseline) != N_TUNED:
        raise RuntimeError("behavioral guards require complete tuned vectors")

    guarded = tuned.copy()
    adjustments: list[dict[str, int | str]] = []

    # Bishop mobility must remain a reward in endings. Letting this coordinate
    # cross below the proven baseline rewards a bishop for being boxed in by
    # its own pawns and breaks the engine's bad-bishop evaluation invariant.
    floor = baseline[BISHOP_EG_MOBILITY_INDEX]
    if guarded[BISHOP_EG_MOBILITY_INDEX] < floor:
        adjustments.append({
            "parameter": "mob_b_eg",
            "raw": guarded[BISHOP_EG_MOBILITY_INDEX],
            "guarded": floor,
        })
        guarded[BISHOP_EG_MOBILITY_INDEX] = floor

    return guarded, adjustments


def copy_first_lines(source: Path, destination: Path, limit: int) -> None:
    written = 0
    with source.open("r", encoding="utf-8", errors="replace") as src, \
            destination.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line.rstrip("\r\n") + "\n")
            written += 1
            if written >= limit:
                break
    if written == 0:
        raise RuntimeError(f"no rows copied from {source}")


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as source:
        return sum(1 for line in source if line.strip())


def probability(result: dict) -> float:
    paired = result.get("paired_probability_better")
    if paired is not None:
        return float(paired)
    raw = result.get("probability_better")
    return float(raw) if raw is not None else 0.0


class Retune:
    def __init__(self, *, resume_from_median: bool = False) -> None:
        SWEEP.mkdir(parents=True, exist_ok=True)
        self.status_path = SWEEP / "status.json"
        self.candidate_options: dict[str, str] = {}
        self.resume_from_median = resume_from_median
        self.status: dict = {
            "status": "waiting",
            "started_at_unix": int(time.time()),
            "baseline": str(BASELINE),
            "candidate": str(CANDIDATE),
            "promotion": "none",
            "tunes": {},
        }
        write_json(self.status_path, self.status)

    def save(self) -> None:
        write_json(self.status_path, self.status)

    def wait_for_search_sweep(self) -> None:
        prior_path = LAB / "check_extension_sweep" / "status.json"
        deadline = time.monotonic() + 6 * 3600
        while True:
            if prior_path.exists():
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                prior_status = prior.get("status")
                if prior_status not in {"running", "waiting"}:
                    self.status["prior_search_status"] = prior_status
                    if prior_status == "confirmed_candidate":
                        candidate_name = prior["confirmation_240"]["candidate"]
                        self.candidate_options = dict(
                            prior["variants"][candidate_name]
                        )
                        self.status["stacked_search_candidate"] = candidate_name
                        self.status["candidate_options"] = self.candidate_options
                    self.status["status"] = "running"
                    self.save()
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError("check-extension sweep did not finish in 6h")
            log("waiting for check-extension sweep")
            time.sleep(30)

    def prepare_feature_corpus(self) -> tuple[Path, Path, Path]:
        positions = LAB / "combined_positions.txt"
        raw_groups = LAB / "combined_groups_raw.txt"
        initial = LAB / "baseline_tune.log"
        for path in (positions, raw_groups, initial):
            if not path.exists():
                raise RuntimeError(f"missing tuning input: {path}")

        feature_engine = SWEEP / executable_name("established_feature_engine")
        features = SWEEP / "established_features.txt"
        groups = SWEEP / "established_groups.txt"

        run(["make", "-B", "bin/chess_uci", "-j5"], "build_feature_engine")
        shutil.copy2(
            ROOT / "bin" / executable_name("chess_uci"),
            feature_engine,
        )
        command = (
            f"tunedump {positions} {features} {raw_groups} {groups}\n"
            "quit\n"
        )
        run([str(feature_engine)], "dump_feature_corpus", input_text=command)

        feature_rows = count_nonempty_lines(features)
        group_rows = count_nonempty_lines(groups)
        if feature_rows != group_rows:
            raise RuntimeError(
                f"fresh feature/group mismatch: {feature_rows} vs {group_rows}"
            )
        if feature_rows < 50000:
            raise RuntimeError(
                f"fresh quiet filter produced only {feature_rows} positions"
            )

        run(
            [
                sys.executable,
                "scripts/texel_tune.py",
                "--feats",
                str(features),
                "--initial-tuned-file",
                str(initial),
                "--retune-established",
                "--iters",
                "0",
            ],
            "verify_feature_corpus",
        )
        self.status["feature_corpus"] = {
            "positions": feature_rows,
            "groups": group_rows,
            "exact_reconstruction": True,
        }
        self.save()
        return features, groups, initial

    def tune(self) -> Path:
        features, groups, initial = self.prepare_feature_corpus()
        vectors = []
        for seed in (31, 32, 33):
            name = f"seed{seed}"
            run(
                [
                    sys.executable,
                    "scripts/texel_tune.py",
                    "--feats",
                    str(features),
                    "--groups",
                    str(groups),
                    "--initial-tuned-file",
                    str(initial),
                    "--retune-established",
                    "--batch-size",
                    "8192",
                    "--iters",
                    "3000",
                    "--l2",
                    "3.0",
                    "--seed",
                    str(seed),
                ],
                f"tune_{name}",
            )
            vector = parse_tuned_vector(SWEEP / f"tune_{name}.stdout.log")
            vectors.append(vector)
            self.status["tunes"][name] = "completed"
            self.save()

        median = median_vector(vectors)
        tuned_path = SWEEP / "established_median_tuned.txt"
        tuned_path.write_text(
            "TUNED " + " ".join(str(value) for value in median) + "\n",
            encoding="utf-8",
        )
        self.status["median_tuned_file"] = str(tuned_path)
        self.save()
        return tuned_path

    def guard_candidate(self, tuned_path: Path) -> Path:
        baseline_path = LAB / "baseline_tune.log"
        baseline = parse_baseline_vector(baseline_path)
        tuned = parse_tuned_vector(tuned_path)
        guarded, adjustments = apply_behavioral_guards(tuned, baseline)
        guarded_path = SWEEP / "established_guarded_tuned.txt"
        guarded_path.write_text(
            "TUNED " + " ".join(str(value) for value in guarded) + "\n",
            encoding="utf-8",
        )
        self.status["guarded_tuned_file"] = str(guarded_path)
        self.status["behavioral_guard_adjustments"] = adjustments
        self.save()
        return guarded_path

    def build_and_test(self, tuned_path: Path) -> None:
        backup = SWEEP / "hce_eval.pre_retune.c"
        shutil.copy2(EVAL_SOURCE, backup)
        try:
            run(
                [
                    sys.executable,
                    "scripts/texel_apply_tune.py",
                    "--eval-c",
                    str(EVAL_SOURCE),
                    "--tuned-file",
                    str(tuned_path),
                ],
                "apply_tune",
            )
            run(["make", "-B", "bin/chess_uci", "-j5"], "build_candidate")
            shutil.copy2(
                ROOT / "bin" / executable_name("chess_uci"),
                CANDIDATE,
            )

            sample = SWEEP / "verify_positions.txt"
            verify_features = SWEEP / "verify_features.txt"
            copy_first_lines(LAB / "combined_positions.txt", sample, 5000)
            run(
                [str(CANDIDATE)],
                "verify_tunedump",
                input_text=f"tunedump {sample} {verify_features}\nquit\n",
            )
            run(
                [
                    sys.executable,
                    "scripts/texel_tune.py",
                    "--feats",
                    str(verify_features),
                    "--initial-tuned-file",
                    str(tuned_path),
                    "--retune-established",
                    "--iters",
                    "0",
                ],
                "verify_exact",
            )
            run(["make", "test"], "make_test")
            suite_args = [
                sys.executable,
                "scripts/run_hce_position_suite.py",
                "--engine",
                str(CANDIDATE),
                "--backend",
                "classic",
                "--out",
                str(SWEEP / "hce_suite.json"),
                "--fail-fast",
            ]
            for option, value in self.candidate_options.items():
                suite_args.extend(["--uci-option", f"{option}={value}"])
            run(suite_args, "hce_suite")
        finally:
            shutil.copyfile(backup, EVAL_SOURCE)
            os.utime(EVAL_SOURCE, None)
            run(["make", "-B", "bin/chess_uci", "-j5"], "restore_engine")

        self.status["candidate_tests"] = {
            "exact_reconstruction": True,
            "make_test": "passed",
            "hce_suite": "passed",
        }
        self.save()

    def run_match(
        self,
        name: str,
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
        for option, value in self.candidate_options.items():
            args.extend(["--uci-option", f"cand:{option}={value}"])
        run(args, name)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return summarize_match(report, expected_games=games)

    def gate(self) -> None:
        screen = self.run_match(
            "screen_60g",
            games=60,
            positions=30,
            seed=20261021,
        )
        self.status["screen_60"] = screen
        self.save()
        if (
            screen["engine_failures"] != 0
            or screen["elo_diff"] is None
            or screen["elo_diff"] <= 0
            or probability(screen) < 0.70
        ):
            self.status["status"] = "rejected_at_60"
            return

        gate = self.run_match(
            "gate_120g",
            games=120,
            positions=60,
            seed=20261030,
        )
        self.status["gate_120"] = gate
        self.save()
        if (
            gate["engine_failures"] != 0
            or gate["elo_diff"] is None
            or gate["elo_diff"] <= 0
            or probability(gate) < 0.75
        ):
            self.status["status"] = "rejected_at_120"
            return

        confirmation = self.run_match(
            "confirm_240g",
            games=240,
            positions=120,
            seed=20261040,
        )
        self.status["confirmation_240"] = confirmation
        confirmed = (
            confirmation["engine_failures"] == 0
            and confirmation["elo_diff"] is not None
            and confirmation["elo_diff"] > 0
            and probability(confirmation) >= 0.90
        )
        self.status["status"] = (
            "confirmed_candidate" if confirmed else "rejected_at_240"
        )
        if confirmed:
            self.status["promotion"] = "blocked_pending_review"

    def execute(self) -> int:
        try:
            if not BASELINE.exists():
                raise RuntimeError(f"missing frozen baseline: {BASELINE}")
            self.wait_for_search_sweep()
            if self.resume_from_median:
                tuned_path = SWEEP / "established_median_tuned.txt"
                if not tuned_path.exists():
                    raise RuntimeError(
                        f"missing saved median tune: {tuned_path}"
                    )
                parse_tuned_vector(tuned_path)
                self.status["resumed_from_median"] = True
                self.status["median_tuned_file"] = str(tuned_path)
                self.save()
            else:
                tuned_path = self.tune()
            tuned_path = self.guard_candidate(tuned_path)
            self.build_and_test(tuned_path)
            self.gate()
            self.status["finished_at_unix"] = int(time.time())
            self.save()
            return 0
        except Exception as exc:
            self.status["status"] = "failed"
            self.status["error"] = str(exc)
            self.status["finished_at_unix"] = int(time.time())
            self.save()
            log(f"ERROR: {exc}")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from-median",
        action="store_true",
        help="Reuse the saved three-seed median and repeat candidate validation.",
    )
    args = parser.parse_args()
    return Retune(resume_from_median=args.resume_from_median).execute()


if __name__ == "__main__":
    raise SystemExit(main())
