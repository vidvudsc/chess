#!/usr/bin/env python3
"""Run the HCE-v2 data, tuning, verification, and Elo pipeline unattended.

This controller is intended for the isolated Windows ML-PC checkout. It waits
for the scheduled self-play job, builds a game-grouped dataset, tunes the 12
HCE-v2 weights across three held-out splits, tests a median candidate, and
runs paired 120/240-game gates. It never deploys or replaces the frozen
baseline.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "current" / "hce_v2"
V2_NAMES = [
    "connected_pawn_mg",
    "connected_pawn_eg",
    "phalanx_pawn_mg",
    "phalanx_pawn_eg",
    "backward_pawn_mg",
    "backward_pawn_eg",
    "knight_outpost_mg",
    "knight_outpost_eg",
    "bishop_pair_mg",
    "bishop_pair_eg",
    "rook_behind_passer_mg",
    "rook_behind_passer_eg",
]
N_TUNED = 815
V2_START = 35


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Pipeline:
    def __init__(self, timeout_hours: float) -> None:
        self.timeout_seconds = max(1.0, timeout_hours * 3600.0)
        self.status_path = LAB / "pipeline_status.json"
        self.failure_path = LAB / "pipeline_failed.txt"
        self.summary = {
            "status": "running",
            "started_at_unix": int(time.time()),
            "root": str(ROOT),
            "stages": {},
        }
        LAB.mkdir(parents=True, exist_ok=True)
        self.failure_path.unlink(missing_ok=True)
        write_json_atomic(self.status_path, self.summary)

    def mark(self, stage: str, **details) -> None:
        self.summary["stages"][stage] = {
            "completed_at_unix": int(time.time()),
            **details,
        }
        write_json_atomic(self.status_path, self.summary)
        log(f"completed stage: {stage}")

    def run(self, args: list[str], name: str, input_text: str | None = None) -> None:
        stdout_path = LAB / f"{name}.stdout.log"
        stderr_path = LAB / f"{name}.stderr.log"
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
                f"see {stdout_path.name} and {stderr_path.name}")

    def wait_for_selfplay(self) -> None:
        exit_path = LAB / "selfplay.exit"
        pgn_path = LAB / "selfplay_25000.pgn"
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if exit_path.exists():
                raw_exit = exit_path.read_text(
                    encoding="utf-8", errors="replace").strip()
                if raw_exit:
                    if raw_exit != "0":
                        raise RuntimeError(f"self-play exited with {raw_exit!r}")
                    break
                # cmd.exe opens the redirection target before executing `echo`.
                # If the watcher lands in that tiny window, the complete PGN is
                # a stronger success signal than the still-empty handoff file.
                if pgn_path.exists() and count_pgn_games(pgn_path) == 25000:
                    log("self-play exit file was empty, but the 25,000-game "
                        "PGN is complete")
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"self-play did not finish within {self.timeout_seconds / 3600:.1f}h")
            log("waiting for HCEV2Selfplay to write selfplay.exit")
            time.sleep(30)
        if not pgn_path.exists() or pgn_path.stat().st_size == 0:
            raise RuntimeError("self-play reported success but produced no PGN")
        games = count_pgn_games(pgn_path)
        if games != 25000:
            raise RuntimeError(f"expected 25000 self-play games, found {games}")
        self.mark("selfplay", games=games, bytes=pgn_path.stat().st_size)

    def build_grouped_corpus(self) -> None:
        python = sys.executable
        self.run(
            [
                python,
                "scripts/texel_build_dataset.py",
                "--pgn", str(LAB / "selfplay_25000.pgn"),
                "--out", str(LAB / "selfplay_positions.txt"),
                "--groups-out", str(LAB / "selfplay_groups.txt"),
                "--group-prefix", "selfplay",
                "--drop-max-plies",
                "--skip-opening", "10",
                "--skip-tail", "6",
                "--per-game", "12",
            ],
            "build_selfplay_dataset",
        )
        position_inputs = [
            LAB / "vidbot_positions.txt",
            LAB / "selfplay_positions.txt",
        ]
        group_inputs = [
            LAB / "vidbot_groups.txt",
            LAB / "selfplay_groups.txt",
        ]
        source_counts = []
        for positions, groups in zip(position_inputs, group_inputs):
            position_count = count_nonempty_lines(positions)
            group_count = count_nonempty_lines(groups)
            if position_count != group_count:
                raise RuntimeError(
                    f"{positions.name} has {position_count} rows but "
                    f"{groups.name} has {group_count}")
            source_counts.append(position_count)
        concatenate_files(position_inputs, LAB / "combined_positions.txt")
        concatenate_files(group_inputs, LAB / "combined_groups_raw.txt")
        combined = count_nonempty_lines(LAB / "combined_positions.txt")
        if combined != sum(source_counts):
            raise RuntimeError("combined corpus row count is inconsistent")
        self.mark(
            "grouped_corpus",
            vidbot_positions=source_counts[0],
            selfplay_positions=source_counts[1],
            combined_positions=combined,
        )

    def dump_quiet_features(self) -> None:
        self.run(["make", "bin/chess_uci", "-j6"], "build_feature_engine")
        feature_engine = LAB / executable_name("feature_engine")
        shutil.copy2(ROOT / "bin" / executable_name("chess_uci"), feature_engine)
        command = (
            f"tunedump {LAB / 'combined_positions.txt'} "
            f"{LAB / 'combined_features.txt'} "
            f"{LAB / 'combined_groups_raw.txt'} "
            f"{LAB / 'combined_groups.txt'}\n"
            "quit\n"
        )
        self.run([str(feature_engine)], "quiet_tunedump", input_text=command)
        feature_rows = count_nonempty_lines(LAB / "combined_features.txt")
        group_rows = count_nonempty_lines(LAB / "combined_groups.txt")
        if feature_rows != group_rows:
            raise RuntimeError(
                f"quiet feature/group mismatch: {feature_rows} vs {group_rows}")
        if feature_rows < 50000:
            raise RuntimeError(
                f"quiet filter produced only {feature_rows} positions")
        self.mark(
            "quiet_features",
            positions=feature_rows,
            groups=count_unique_lines(LAB / "combined_groups.txt"),
        )

    def tune_median_candidate(self) -> list[int]:
        initial = LAB / "baseline_tune.log"
        if not initial.exists():
            raise RuntimeError(f"missing exact initial weights: {initial}")
        vectors = []
        for seed in (11, 12, 13):
            name = f"v2_seed{seed}_tune"
            self.run(
                [
                    sys.executable,
                    "scripts/texel_tune.py",
                    "--feats", str(LAB / "combined_features.txt"),
                    "--groups", str(LAB / "combined_groups.txt"),
                    "--initial-tuned-file", str(initial),
                    "--only-v2-features",
                    "--iters", "5000",
                    "--l2", "0.3",
                    "--seed", str(seed),
                ],
                name,
            )
            tuned_stdout = LAB / f"{name}.stdout.log"
            vectors.append(parse_tuned_vector(tuned_stdout))
        median = median_vector(vectors)
        median_path = LAB / "v2_median_tuned.txt"
        median_path.write_text(
            "TUNED " + " ".join(str(value) for value in median) + "\n",
            encoding="utf-8",
        )
        weights = dict(zip(V2_NAMES, median[V2_START:V2_START + len(V2_NAMES)]))
        stability = {
            name: [vector[V2_START + index] for vector in vectors]
            for index, name in enumerate(V2_NAMES)
        }
        self.mark("tuning", median_weights=weights, seed_weights=stability)
        return median

    def build_and_test_candidate(self) -> None:
        eval_source = ROOT / "src" / "core" / "engine" / "hce_eval.c"
        source_backup = LAB / "hce_eval.zero_weights.c"
        shutil.copy2(eval_source, source_backup)
        try:
            self.run(
                [
                    sys.executable,
                    "scripts/texel_apply_tune.py",
                    "--eval-c", str(eval_source),
                    "--tuned-file", str(LAB / "v2_median_tuned.txt"),
                ],
                "apply_v2_median",
            )
            self.run(["make", "bin/chess_uci", "-j6"], "build_candidate")
            candidate = LAB / executable_name("candidate_v2")
            shutil.copy2(ROOT / "bin" / executable_name("chess_uci"), candidate)
            self.verify_candidate(candidate)
            self.run(["make", "test"], "candidate_make_test")
            self.run(["make", "hce_suite"], "candidate_hce_suite")
        finally:
            shutil.copy2(source_backup, eval_source)
        self.run(["make", "bin/chess_uci", "-j6"], "restore_zero_weight_build")
        self.mark(
            "candidate_tests",
            exact_reconstruction=True,
            make_test="passed",
            hce_suite="passed",
        )

    def verify_candidate(self, candidate: Path) -> None:
        sample_positions = LAB / "candidate_verify_positions.txt"
        copy_first_lines(
            LAB / "combined_positions.txt",
            sample_positions,
            limit=5000,
        )
        features = LAB / "candidate_verify_features.txt"
        command = f"tunedumpall {sample_positions} {features}\nquit\n"
        self.run([str(candidate)], "candidate_verify_tunedump", input_text=command)
        self.run(
            [
                sys.executable,
                "scripts/texel_tune.py",
                "--feats", str(features),
                "--initial-tuned-file", str(LAB / "v2_median_tuned.txt"),
                "--only-v2-features",
                "--iters", "0",
            ],
            "candidate_exact_verify",
        )

    def run_elo_gates(self) -> None:
        result_120 = self.run_match(
            games=120,
            positions=60,
            seed=20260730,
            name="v2_gate_120g",
        )
        self.mark("elo_120", **result_120)
        if result_120["engine_failures"] != 0:
            raise RuntimeError("120-game gate had engine failures")
        if result_120["elo_diff"] is None or result_120["elo_diff"] < -50.0:
            self.summary["status"] = "rejected_at_120"
            self.finish()
            return
        result_240 = self.run_match(
            games=240,
            positions=120,
            seed=20260731,
            name="v2_gate_240g",
        )
        self.mark("elo_240", **result_240)
        if result_240["engine_failures"] != 0:
            raise RuntimeError("240-game confirmation had engine failures")
        probability = result_240["paired_probability_better"]
        if probability is None:
            probability = result_240["probability_better"]
        confirmed = (
            result_240["elo_diff"] is not None
            and result_240["elo_diff"] > 0.0
            and probability is not None
            and probability >= 0.90
        )
        self.summary["status"] = (
            "confirmed_positive_not_promoted" if confirmed else "not_confirmed")
        self.finish()

    def run_match(self, games: int, positions: int, seed: int, name: str) -> dict:
        report_path = LAB / f"{name}.json"
        self.run(
            [
                sys.executable,
                "src/core/bot/test_lab.py",
                "--engine", f"cand={LAB / executable_name('candidate_v2')}",
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

    def finish(self) -> None:
        self.summary["finished_at_unix"] = int(time.time())
        self.summary["promotion"] = "none"
        write_json_atomic(self.status_path, self.summary)
        write_json_atomic(LAB / "pipeline_summary.json", self.summary)
        log(f"pipeline finished: {self.summary['status']}")

    def fail(self, exc: BaseException) -> None:
        self.summary["status"] = "failed"
        self.summary["finished_at_unix"] = int(time.time())
        self.summary["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(self.status_path, self.summary)
        self.failure_path.write_text(self.summary["error"] + "\n", encoding="utf-8")
        log(self.summary["error"])


def executable_name(stem: str) -> str:
    return stem + (".exe" if os.name == "nt" else "")


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(1 for line in stream if line.strip())


def count_unique_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return len({line.strip() for line in stream if line.strip()})


def count_pgn_games(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(1 for line in stream if line.startswith("[Event "))


def concatenate_files(inputs: Iterable[Path], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for path in inputs:
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    destination.write(line.rstrip("\r\n") + "\n")


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


def parse_tuned_vector(path: Path) -> list[int]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("TUNED ")
    ]
    if not lines:
        raise ValueError(f"no TUNED line in {path}")
    vector = [int(value) for value in lines[-1].split()[1:]]
    if len(vector) != N_TUNED:
        raise ValueError(f"{path} has {len(vector)} tuned values, expected {N_TUNED}")
    return vector


def median_vector(vectors: list[list[int]]) -> list[int]:
    if len(vectors) != 3:
        raise ValueError("median candidate requires exactly three tuning seeds")
    lengths = {len(vector) for vector in vectors}
    if lengths != {N_TUNED}:
        raise ValueError(f"inconsistent tuned vector lengths: {sorted(lengths)}")
    return [
        sorted(vector[index] for vector in vectors)[1]
        for index in range(N_TUNED)
    ]


def summarize_match(report: dict, expected_games: int) -> dict:
    head_to_head = report.get("head_to_head", [])
    if len(head_to_head) != 1:
        raise ValueError("match report must contain exactly one head-to-head result")
    row = head_to_head[0]
    if row.get("games") != expected_games:
        raise ValueError(
            f"match report has {row.get('games')} games, expected {expected_games}")
    candidate = row.get("candidate")
    standings = {
        item.get("name"): item for item in report.get("standings", [])
    }
    paired = row.get("paired", {})
    return {
        "games": row["games"],
        "points": row["points"],
        "elo_diff": row.get("elo_diff"),
        "elo_ci_low": row.get("elo_ci_low"),
        "elo_ci_high": row.get("elo_ci_high"),
        "probability_better": row.get("probability_better"),
        "paired_positions": row.get("paired_positions", 0),
        "paired_elo_diff": paired.get("elo_diff"),
        "paired_elo_ci_low": paired.get("elo_ci_low"),
        "paired_elo_ci_high": paired.get("elo_ci_high"),
        "paired_probability_better": paired.get("probability_better"),
        "engine_failures": standings.get(candidate, {}).get("engine_failures", -1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selfplay-timeout-hours",
        type=float,
        default=12.0,
        help="How long to wait for the independently scheduled self-play task.",
    )
    args = parser.parse_args()
    pipeline = Pipeline(args.selfplay_timeout_hours)
    try:
        pipeline.wait_for_selfplay()
        pipeline.build_grouped_corpus()
        pipeline.dump_quiet_features()
        pipeline.tune_median_candidate()
        pipeline.build_and_test_candidate()
        pipeline.run_elo_gates()
    except BaseException as exc:
        pipeline.fail(exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
