from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "core" / "bot" / "nn" / "v2" / "train_value.py"
UCI = ROOT / "bin" / "chess_uci"


POSITIONS = [
    ("4k3/8/8/8/8/8/8/4KQ2 w - -", 0.95, 900),
    ("4k3/8/8/8/8/8/8/4KQ2 b - -", -0.95, -900),
    ("4k3/8/8/8/8/8/8/4KR2 w - -", 0.75, 600),
    ("4k3/8/8/8/8/8/8/4KR2 b - -", -0.75, -600),
    ("4k3/8/8/8/8/8/8/4KN2 w - -", 0.35, 220),
    ("4k3/8/8/8/8/8/8/4KN2 b - -", -0.35, -220),
    ("4k3/8/8/8/8/8/8/4KB2 w - -", 0.30, 180),
    ("4k3/8/8/8/8/8/8/4KB2 b - -", -0.30, -180),
    ("4k3/8/8/8/8/8/4q3/4K3 w - -", -0.95, -900),
    ("4k3/8/8/8/8/8/4q3/4K3 b - -", 0.95, 900),
]


def write_dataset(path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for idx, (fen, target, cp) in enumerate(POSITIONS):
            row = {
                "fen": fen,
                "depth": 24 + idx % 4,
                "knodes": 1000 + idx,
                "evaluation": f"{cp / 100.0:.2f}",
                "score_perspective": "stm",
                "target_stm": target,
                "cp_stm": cp,
                "is_mate": False,
                "terminal": "none",
                "best_move": "",
                "best_line": "",
                "bucket": "white/opening/equal/quiet",
                "source": "smoke",
            }
            fp.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_train_export_and_uci_load() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v2.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval.bin"
        write_dataset(dataset)

        subprocess.run(
            [
                sys.executable,
                str(TRAIN),
                "--input",
                str(dataset),
                "--output-dir",
                str(out_dir),
                "--export-path",
                str(export),
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--feature-dim",
                "8",
                "--hidden-dim",
                "4",
                "--train-samples",
                "20",
                "--val-samples",
                "20",
                "--val-buckets",
                "1",
                "--lr",
                "0.01",
                "--log-every",
                "0",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        magic = export.read_bytes()[:8]
        if magic != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {magic!r}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout:
            raise AssertionError(f"UCI did not become ready:\n{uci.stdout}\n{uci.stderr}")
        if "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept NN backend:\n{uci.stdout}\n{uci.stderr}")


def main() -> None:
    test_train_export_and_uci_load()
    print("test_nn_v2_train_smoke: OK")


if __name__ == "__main__":
    main()
