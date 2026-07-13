from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "core" / "bot" / "nn" / "train_policy.py"


def write_tiny_policy_npz(path: Path) -> None:
    rng = np.random.default_rng(1234)
    rows = 64
    states = np.zeros((rows, 15, 8, 8), dtype=np.float32)
    plans = np.zeros((rows, 3, 1), dtype=np.int64)
    evals = np.zeros((rows,), dtype=np.float32)
    for idx in range(rows):
        from_sq = idx % 64
        to_sq = (idx * 7 + 11) % 64
        states[idx] = rng.normal(0.0, 0.1, size=(15, 8, 8)).astype(np.float32)
        states[idx, 0, from_sq // 8, from_sq % 8] = 1.0
        states[idx, 1, to_sq // 8, to_sq % 8] = 1.0
        plans[idx, :, 0] = [from_sq * 64 + to_sq, from_sq * 64 + ((to_sq + 1) % 64), from_sq * 64 + ((to_sq + 2) % 64)]
        evals[idx] = np.tanh((idx - rows / 2) / 32.0)
    np.savez_compressed(path, states=states, plans=plans, evals=evals)


def test_policy_train_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "policy_smoke.npz"
        out_dir = tmp_path / "policy_model"
        write_tiny_policy_npz(dataset)
        subprocess.run(
            [
                sys.executable,
                str(TRAIN),
                "--input-npz",
                str(dataset),
                "--output-dir",
                str(out_dir),
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--channels",
                "8",
                "--blocks",
                "1",
                "--value-hidden",
                "16",
                "--sample-limit",
                "64",
                "--val-fraction",
                "0.25",
                "--device",
                "cpu",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        history_path = out_dir / "history.json"
        checkpoint_path = out_dir / "best.pt"
        if not history_path.exists() or not checkpoint_path.exists():
            raise AssertionError("policy training did not write expected artifacts")
        with history_path.open("r", encoding="utf-8") as fp:
            history = json.load(fp)
        if history["history"][0]["train"]["rows"] != 48.0:
            raise AssertionError(f"unexpected train rows: {history}")


def main() -> None:
    test_policy_train_smoke()
    print("test_nn_policy_train_smoke: OK")


if __name__ == "__main__":
    main()
