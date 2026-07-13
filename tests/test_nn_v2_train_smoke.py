from __future__ import annotations

import json
import importlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "core" / "bot" / "nn" / "v2" / "train_value.py"
PRECOMPUTE = ROOT / "src" / "core" / "bot" / "nn" / "v2" / "precompute_features.py"
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
                "--terminal-fixtures",
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


def test_train_linear_head_export_and_search() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v3.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval_v3.bin"
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
                "--arch",
                "linear-head",
                "--feature-dim",
                "16",
                "--hidden-dim",
                "6",
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
                "--terminal-fixtures",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        raw = export.read_bytes()
        if raw[:8] != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {raw[:8]!r}")
        version = struct.unpack_from("<I", raw, 8)[0]
        if version != 3:
            raise AssertionError(f"linear-head export should use version 3, got {version}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "position startpos\n"
                "go movetime 20\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout or "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept linear-head NN backend:\n{uci.stdout}\n{uci.stderr}")
        if "bestmove " not in uci.stdout:
            raise AssertionError(f"UCI did not search with linear-head NN:\n{uci.stdout}\n{uci.stderr}")


def test_train_linear_head_crelu_export_and_search() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v5.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval_v5.bin"
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
                "--arch",
                "linear-head-crelu",
                "--feature-dim",
                "16",
                "--hidden-dim",
                "6",
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
                "--terminal-fixtures",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        raw = export.read_bytes()
        if raw[:8] != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {raw[:8]!r}")
        version = struct.unpack_from("<I", raw, 8)[0]
        if version != 5:
            raise AssertionError(f"linear-head-crelu export should use version 5, got {version}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "position startpos\n"
                "go movetime 20\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout or "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept clipped linear NN backend:\n{uci.stdout}\n{uci.stderr}")
        if "bestmove " not in uci.stdout:
            raise AssertionError(f"UCI did not search with clipped linear NN:\n{uci.stdout}\n{uci.stderr}")


def test_train_bottleneck_head_export_and_search() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v4.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval_v4.bin"
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
                "1",
                "--batch-size",
                "4",
                "--arch",
                "bottleneck-head",
                "--feature-dim",
                "16",
                "--bottleneck-dim",
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
                "--terminal-fixtures",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        raw = export.read_bytes()
        if raw[:8] != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {raw[:8]!r}")
        version = struct.unpack_from("<I", raw, 8)[0]
        if version != 4:
            raise AssertionError(f"bottleneck export should use version 4, got {version}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "position startpos\n"
                "go movetime 20\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout or "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept bottleneck NN backend:\n{uci.stdout}\n{uci.stderr}")
        if "bestmove " not in uci.stdout:
            raise AssertionError(f"UCI did not search with bottleneck NN:\n{uci.stdout}\n{uci.stderr}")


def test_train_bottleneck_head_crelu_export_and_search() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v6.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval_v6.bin"
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
                "1",
                "--batch-size",
                "4",
                "--arch",
                "bottleneck-head-crelu",
                "--feature-dim",
                "16",
                "--bottleneck-dim",
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
                "--terminal-fixtures",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        raw = export.read_bytes()
        if raw[:8] != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {raw[:8]!r}")
        version = struct.unpack_from("<I", raw, 8)[0]
        if version != 6:
            raise AssertionError(f"bottleneck-head-crelu export should use version 6, got {version}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "position startpos\n"
                "go movetime 20\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout or "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept clipped bottleneck NN backend:\n{uci.stdout}\n{uci.stderr}")
        if "bestmove " not in uci.stdout:
            raise AssertionError(f"UCI did not search with clipped bottleneck NN:\n{uci.stdout}\n{uci.stderr}")


def test_train_linear_head_screlu_export_and_search() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v7.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval_v7.bin"
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
                "1",
                "--batch-size",
                "4",
                "--arch",
                "linear-head-screlu",
                "--feature-dim",
                "16",
                "--hidden-dim",
                "6",
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
                "--terminal-fixtures",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        raw = export.read_bytes()
        if raw[:8] != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {raw[:8]!r}")
        version = struct.unpack_from("<I", raw, 8)[0]
        if version != 9:
            raise AssertionError(f"linear-head-screlu export should use version 7, got {version}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "position startpos\n"
                "go movetime 20\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout or "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept SCReLU linear NN backend:\n{uci.stdout}\n{uci.stderr}")
        if "bestmove " not in uci.stdout:
            raise AssertionError(f"UCI did not search with SCReLU linear NN:\n{uci.stdout}\n{uci.stderr}")


def test_train_bottleneck_head_screlu_export_and_search() -> None:
    if not UCI.exists():
        raise AssertionError(f"missing UCI binary: {UCI}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v8.jsonl"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval_v8.bin"
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
                "1",
                "--batch-size",
                "4",
                "--arch",
                "bottleneck-head-screlu",
                "--feature-dim",
                "16",
                "--bottleneck-dim",
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
                "--terminal-fixtures",
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        raw = export.read_bytes()
        if raw[:8] != b"CHNNUE1\0":
            raise AssertionError(f"unexpected export magic: {raw[:8]!r}")
        version = struct.unpack_from("<I", raw, 8)[0]
        if version != 8:
            raise AssertionError(f"bottleneck-head-screlu export should use version 8, got {version}")

        uci = subprocess.run(
            [
                str(UCI),
            ],
            input=(
                "uci\n"
                f"setoption name NNModel value {export}\n"
                "setoption name Backend value nn\n"
                "isready\n"
                "position startpos\n"
                "go movetime 20\n"
                "quit\n"
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        if "readyok" not in uci.stdout or "backend set to nn" not in uci.stdout:
            raise AssertionError(f"UCI did not accept SCReLU bottleneck NN backend:\n{uci.stdout}\n{uci.stderr}")
        if "bestmove " not in uci.stdout:
            raise AssertionError(f"UCI did not search with SCReLU bottleneck NN:\n{uci.stdout}\n{uci.stderr}")


def test_precompute_train_export() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "tiny_v2.jsonl"
        feature_dir = tmp_path / "features"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval.bin"
        write_dataset(dataset)

        subprocess.run(
            [
                sys.executable,
                str(PRECOMPUTE),
                "--input",
                str(dataset),
                "--output-dir",
                str(feature_dir),
                "--rows-per-shard",
                "4",
                "--log-every",
                "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["rows"] != len(POSITIONS):
            raise AssertionError(f"bad feature manifest: {manifest}")

        subprocess.run(
            [
                sys.executable,
                str(TRAIN),
                "--input",
                str(dataset),
                "--input-features",
                str(feature_dir),
                "--output-dir",
                str(out_dir),
                "--export-path",
                str(export),
                "--epochs",
                "1",
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

        if export.read_bytes()[:8] != b"CHNNUE1\0":
            raise AssertionError("precomputed feature training did not export CHNNUE1")


def test_train_direct_from_normalized_parquet() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "normalized.parquet"
        out_dir = tmp_path / "model"
        export = tmp_path / "nn_eval.bin"
        rows = []
        for idx, (fen, _target, cp) in enumerate(POSITIONS):
            rows.append({
                "fen": fen,
                "depth": 24 + idx % 4,
                "cp": cp,
                "mate": None,
            })
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)

        subprocess.run(
            [
                sys.executable,
                str(TRAIN),
                "--input-parquet",
                str(parquet_path),
                "--output-dir",
                str(out_dir),
                "--export-path",
                str(export),
                "--epochs",
                "1",
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

        if export.read_bytes()[:8] != b"CHNNUE1\0":
            raise AssertionError("direct parquet training did not export CHNNUE1")


def test_json_repeat_is_sharded_across_workers() -> None:
    sys.path.insert(0, str(TRAIN.parent))
    train_value = importlib.import_module("train_value")

    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = Path(tmp) / "tiny_v2.jsonl"
        write_dataset(dataset_path)

        dataset = train_value.RepeatedIterableDataset(
            train_value.V2ValueDataset(dataset_path, split="all", val_buckets=0, sample_limit=0),
            repeat=3,
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        rows = list(loader)

    if len(rows) != len(POSITIONS) * 3:
        raise AssertionError(f"JSONL repeat duplicated across workers: got {len(rows)} rows")


def main() -> None:
    test_train_export_and_uci_load()
    test_train_linear_head_export_and_search()
    test_train_linear_head_crelu_export_and_search()
    test_train_bottleneck_head_export_and_search()
    test_train_bottleneck_head_crelu_export_and_search()
    test_train_linear_head_screlu_export_and_search()
    test_train_bottleneck_head_screlu_export_and_search()
    test_precompute_train_export()
    test_train_direct_from_normalized_parquet()
    test_json_repeat_is_sharded_across_workers()
    print("test_nn_v2_train_smoke: OK")


if __name__ == "__main__":
    main()
