from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "core" / "bot" / "nn" / "v2" / "build_dataset.py"


def write_rows(path: Path) -> None:
    rows = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "evals": [{"depth": 22, "knodes": 120, "pvs": [{"cp": 34, "line": "e2e4 e7e5"}]}],
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 9",
            "evals": [{"depth": 23, "knodes": 999, "pvs": [{"cp": 50, "line": "d2d4 d7d5"}]}],
        },
        {
            "fen": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            "evals": [{"depth": 30, "knodes": 10, "pvs": [{"cp": 0, "line": ""}]}],
        },
        {
            "fen": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            "evals": [{"depth": 30, "knodes": 10, "pvs": [{"cp": 900, "line": ""}]}],
        },
        {
            "fen": "4k3/8/8/8/8/8/8/4KQ2 b - - 0 1",
            "evals": [{"depth": 10, "knodes": 10, "pvs": [{"cp": 150, "line": "e8e7"}]}],
        },
    ]
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_build_dataset_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "out.jsonl"
        summary_path = tmp_path / "summary.json"
        write_rows(input_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--summary",
                str(summary_path),
                "--min-depth",
                "18",
                "--max-per-bucket",
                "2",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stdout_summary = json.loads(result.stdout)

        if len(rows) != 3:
            raise AssertionError(f"expected 3 written rows, got {len(rows)}: {rows}")
        if summary["stats"]["duplicate_fen"] != 1:
            raise AssertionError(f"expected one duplicate, got {summary}")
        if summary["stats"]["rejected"] != 1:
            raise AssertionError(f"expected one rejected shallow row, got {summary}")
        if summary["written_distribution"]["terminal"].get("checkmate") != 1:
            raise AssertionError(f"expected checkmate terminal row, got {summary}")
        if summary["written_distribution"]["terminal"].get("stalemate") != 1:
            raise AssertionError(f"expected stalemate terminal row, got {summary}")
        if stdout_summary["written_rows"] != summary["written_rows"]:
            raise AssertionError("stdout and summary file disagree")


def test_build_dataset_stops_after_accepted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "out.jsonl"
        write_rows(input_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--min-depth",
                "18",
                "--stop-after-accepted",
                "2",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        summary = json.loads(result.stdout)
        rows = output_path.read_text(encoding="utf-8").splitlines()
        if summary["stats"].get("stopped_after_accepted") != 1:
            raise AssertionError(f"expected early stop marker, got {summary}")
        if summary["stats"]["accepted"] != 2 or len(rows) != 2:
            raise AssertionError(f"expected exactly 2 accepted rows, got {summary} rows={rows}")


def test_build_dataset_terminal_fixtures_and_stride() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "out.jsonl"
        write_rows(input_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--min-depth",
                "18",
                "--skip-input-rows",
                "1",
                "--sample-every",
                "2",
                "--terminal-fixtures",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        summary = json.loads(result.stdout)
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        terminals = {row["terminal"] for row in rows}
        if summary["stats"].get("skipped_input_rows") != 1:
            raise AssertionError(f"expected one skipped input row, got {summary}")
        if summary["stats"].get("terminal_fixtures", 0) < 5:
            raise AssertionError(f"expected most terminal fixtures to be added, got {summary}")
        if sum(summary["written_distribution"]["terminal"].values()) < 6:
            raise AssertionError(f"expected terminal fixture coverage in written rows, got {summary}")
        for expected in {"checkmate", "stalemate", "insufficient_material"}:
            if expected not in terminals:
                raise AssertionError(f"missing terminal fixture {expected}: {rows}")


def test_build_dataset_stream_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "stream.jsonl"
        summary_path = tmp_path / "summary.json"
        write_rows(input_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--summary",
                str(summary_path),
                "--min-depth",
                "18",
                "--stream-output",
                "--max-per-bucket",
                "1",
                "--no-dedupe",
                "--terminal-fixtures",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        summary = json.loads(result.stdout)
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        file_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary["stream_output"]:
            raise AssertionError(f"expected stream output summary, got {summary}")
        if summary["written_rows"] != len(rows):
            raise AssertionError(f"summary row count disagrees with output: {summary} rows={rows}")
        if summary["stats"].get("bucket_cap_skipped", 0) < 1:
            raise AssertionError(f"expected bucket cap skip, got {summary}")
        if summary["written_distribution"]["terminal"].get("checkmate", 0) < 1:
            raise AssertionError(f"expected checkmate written row, got {summary}")
        if file_summary["written_rows"] != summary["written_rows"]:
            raise AssertionError("summary file and stdout disagree")


def main() -> None:
    test_build_dataset_cli()
    test_build_dataset_stops_after_accepted()
    test_build_dataset_terminal_fixtures_and_stride()
    test_build_dataset_stream_output()
    print("test_nn_v2_build_dataset: OK")


if __name__ == "__main__":
    main()
