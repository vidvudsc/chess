from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "core" / "bot" / "nn" / "v2" / "clean_hf_evals.py"


def write_parquet(path: Path) -> None:
    rows = [
        {
            "fen": "4k3/8/8/8/8/8/8/4KQ2 w - -",
            "line": "f1f7 e8d8",
            "depth": 18,
            "knodes": 100,
            "cp": 450,
            "mate": None,
        },
        {
            "fen": "4k3/8/8/8/8/8/8/4KQ2 w - -",
            "line": "f1f8 e8d7",
            "depth": 24,
            "knodes": 50,
            "cp": 800,
            "mate": None,
        },
        {
            "fen": "4k3/8/8/8/8/8/8/4KQ2 w - -",
            "line": "f1a6 e8d7",
            "depth": 24,
            "knodes": 500,
            "cp": 900,
            "mate": None,
        },
        {
            "fen": "4k3/8/8/8/8/8/8/4KR2 b - -",
            "line": "e8d8 e1e8",
            "depth": 22,
            "knodes": 120,
            "cp": -650,
            "mate": None,
        },
        {
            "fen": "7k/6Q1/6K1/8/8/8/8/8 b - -",
            "line": "",
            "depth": 28,
            "knodes": 25,
            "cp": None,
            "mate": -1,
        },
        {
            "fen": "4k3/8/8/8/8/8/8/4KB2 w - -",
            "line": "b1c2",
            "depth": 8,
            "knodes": 10,
            "cp": 20,
            "mate": None,
        },
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_clean_hf_evals_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "evals.parquet"
        jsonl_path = tmp_path / "clean.jsonl"
        feature_dir = tmp_path / "features"
        summary_path = tmp_path / "summary.json"
        write_parquet(parquet_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(parquet_path),
                "--output-jsonl",
                str(jsonl_path),
                "--output-features-dir",
                str(feature_dir),
                "--summary",
                str(summary_path),
                "--min-depth",
                "18",
                "--limit",
                "3",
                "--fast",
                "--terminal-fixtures",
                "--rows-per-shard",
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
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
        summary_file = json.loads(summary_path.read_text(encoding="utf-8"))

        queen_rows = [row for row in rows if row["fen"].startswith("4k3/8/8/8/8/8/8/4KQ2")]
        if len(queen_rows) != 1:
            raise AssertionError(f"expected deduped queen row, got {rows}")
        if queen_rows[0]["depth"] != 24 or queen_rows[0]["knodes"] != 500:
            raise AssertionError(f"expected best depth/knodes row, got {queen_rows[0]}")
        if queen_rows[0]["best_move"] != "f1a6" or queen_rows[0]["cp_stm"] != 900:
            raise AssertionError(f"expected line/best move to survive cleaning, got {queen_rows[0]}")
        if summary["input_stats"]["rejected_depth_or_knodes"] != 1:
            raise AssertionError(f"expected shallow row rejection, got {summary}")
        if summary["input_stats"]["candidate_replaced"] != 2:
            raise AssertionError(f"expected two duplicate replacements, got {summary}")
        if summary["output_stats"]["terminal_fixtures"] < 5:
            raise AssertionError(f"expected terminal fixtures, got {summary}")
        if manifest["rows"] != summary["output_stats"]["written_rows"]:
            raise AssertionError(f"feature manifest row count mismatch: {manifest} {summary}")
        if summary_file["candidate_rows"] != summary["candidate_rows"]:
            raise AssertionError("summary file and stdout disagree")


def test_clean_hf_evals_supports_normalized_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "normalized.parquet"
        feature_dir = tmp_path / "features"
        rows = [
            {
                "fen": "4k3/8/8/8/8/8/8/4KQ2 w - -",
                "depth": 30,
                "cp": 700,
                "mate": None,
            },
            {
                "fen": "4k3/8/8/8/8/8/8/4KR2 b - -",
                "depth": 26,
                "cp": -500,
                "mate": None,
            },
            {
                "fen": "7k/6Q1/6K1/8/8/8/8/8 b - -",
                "depth": 40,
                "cp": None,
                "mate": -1,
            },
        ]
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(parquet_path),
                "--output-features-dir",
                str(feature_dir),
                "--min-depth",
                "18",
                "--fast",
                "--assume-unique",
                "--rows-per-shard",
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
        manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
        if not summary["assume_unique"]:
            raise AssertionError(f"expected streaming unique mode, got {summary}")
        if summary["candidate_rows"] != 3:
            raise AssertionError(f"expected normalized candidates, got {summary}")
        if summary["output_stats"]["written_rows"] != 3 or manifest["rows"] != 3:
            raise AssertionError(f"expected three feature rows, got manifest={manifest} summary={summary}")
        if summary["input_stats"].get("rejected_depth_or_knodes", 0) != 0:
            raise AssertionError(f"normalized rows should not require knodes by default: {summary}")


def main() -> None:
    test_clean_hf_evals_cli()
    test_clean_hf_evals_supports_normalized_schema()
    print("test_nn_v2_clean_hf_evals: OK")


if __name__ == "__main__":
    main()
