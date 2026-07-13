from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "core" / "bot" / "nn" / "v2" / "build_feature_dataset.py"


def write_rows(path: Path) -> None:
    rows = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "evals": [{"depth": 22, "knodes": 120, "pvs": [{"cp": 34, "line": "e2e4 e7e5"}]}],
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "evals": [{"depth": 24, "knodes": 180, "pvs": [{"cp": -20, "line": "e7e5 e2e4"}]}],
        },
        {
            "fen": "4k3/8/8/8/8/8/8/4KQ2 b - - 0 1",
            "evals": [{"depth": 10, "knodes": 10, "pvs": [{"cp": 150, "line": "e8e7"}]}],
        },
        {
            "fen": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            "evals": [{"depth": 30, "knodes": 10, "pvs": [{"cp": 900, "line": ""}]}],
        },
    ]
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_build_feature_dataset_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.jsonl"
        output_dir = tmp_path / "features"
        summary_path = tmp_path / "summary.json"
        write_rows(input_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--summary",
                str(summary_path),
                "--min-depth",
                "18",
                "--limit",
                "3",
                "--rows-per-shard",
                "2",
                "--terminal-fixtures",
                "--fast",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        summary = json.loads(result.stdout)
        file_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        shard_rows = sum(int(item["rows"]) for item in manifest["shards"])

        if summary["stats"]["rejected"] != 1:
            raise AssertionError(f"expected one shallow rejected row, got {summary}")
        if summary["written_rows"] <= 3:
            raise AssertionError(f"terminal fixtures should be appended past the main limit: {summary}")
        if manifest["rows"] != summary["written_rows"] or shard_rows != manifest["rows"]:
            raise AssertionError(f"manifest and summary disagree: manifest={manifest} summary={summary}")
        if summary["written_distribution"]["terminal"].get("checkmate", 0) < 1:
            raise AssertionError(f"expected checkmate fixture coverage, got {summary}")
        if file_summary["feature_rows"] != summary["feature_rows"]:
            raise AssertionError("summary file and stdout disagree")


def main() -> None:
    test_build_feature_dataset_cli()
    print("test_nn_v2_build_feature_dataset: OK")


if __name__ == "__main__":
    main()
