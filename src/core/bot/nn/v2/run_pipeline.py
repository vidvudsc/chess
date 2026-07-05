#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
V2_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def verify_uci(model_path: Path) -> None:
    uci = ROOT / "bin" / "chess_uci"
    if not uci.exists():
        run(["make", "bin/chess_uci"])
    proc = subprocess.run(
        [str(uci)],
        cwd=ROOT,
        text=True,
        input=(
            "uci\n"
            f"setoption name NNModel value {model_path}\n"
            "setoption name Backend value nn\n"
            "isready\n"
            "quit\n"
        ),
        capture_output=True,
        check=True,
    )
    if "readyok" not in proc.stdout or "backend set to nn" not in proc.stdout:
        raise SystemExit(f"UCI failed to load NN model:\n{proc.stdout}\n{proc.stderr}")
    print("[ok] UCI loaded exported NN model", flush=True)


def maybe_run_match(model_path: Path, games: int, think_ms: int) -> None:
    if games <= 0:
        return
    run(["make", "bin/chess_uci"])
    out = ROOT / "current" / f"nnv2_smoke_match_{int(time.time())}.json"
    run([
        sys.executable,
        "src/core/bot/test_lab.py",
        "--engine",
        "classic=bin/chess_uci",
        "--engine",
        "nn=bin/chess_uci",
        "--uci-option",
        "nn:Backend=nn",
        "--uci-option",
        f"nn:NNModel={model_path}",
        "--positions-count",
        str(games),
        "--think-ms",
        str(think_ms),
        "--baseline",
        "classic",
        "--out",
        str(out),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream data, train NNUE v2, export, and verify.")
    parser.add_argument("--input", default=DEFAULT_EVAL_URL, help="Lichess eval URL/file/stdin.")
    parser.add_argument("--work-dir", type=Path, default=Path("current/nnv2_pipeline"))
    parser.add_argument("--dataset-name", default="stream_sample")
    parser.add_argument("--accepted", type=int, default=5000, help="Accepted rows to stream before stopping.")
    parser.add_argument("--limit", type=int, default=5000, help="Written row cap.")
    parser.add_argument("--max-per-bucket", type=int, default=500, help="Per-bucket cap.")
    parser.add_argument("--min-depth", type=int, default=18)
    parser.add_argument("--min-knodes", type=int, default=0)
    parser.add_argument("--skip-input-rows", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--stream-output", action=argparse.BooleanOptionalAction, default=False,
                        help="Write dataset rows directly to disk; use for million-row builds.")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Disable FEN dedupe in the dataset builder. Useful for large raw Lichess streams.")
    parser.add_argument("--terminal-fixtures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--match-games", type=int, default=0)
    parser.add_argument("--match-think-ms", type=int, default=120)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    work_dir = (ROOT / args.work_dir).resolve() if not args.work_dir.is_absolute() else args.work_dir
    dataset = work_dir / f"{args.dataset_name}.jsonl"
    summary = work_dir / f"{args.dataset_name}_summary.json"
    model_dir = work_dir / "model"
    export_path = work_dir / "nn_eval.bin"
    work_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        sys.executable,
        str(V2_DIR / "build_dataset.py"),
        "--input",
        args.input,
        "--output",
        str(dataset),
        "--summary",
        str(summary),
        "--min-depth",
        str(args.min_depth),
        "--min-knodes",
        str(args.min_knodes),
        "--limit",
        str(args.limit),
        "--max-per-bucket",
        str(args.max_per_bucket),
        "--stop-after-accepted",
        str(args.accepted),
        "--skip-input-rows",
        str(args.skip_input_rows),
        "--sample-every",
        str(args.sample_every),
    ]
    if args.terminal_fixtures:
        build_cmd.append("--terminal-fixtures")
    if args.stream_output:
        build_cmd.append("--stream-output")
    if args.no_dedupe:
        build_cmd.append("--no-dedupe")
    run(build_cmd)

    with summary.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if int(payload.get("written_rows", 0)) <= 0:
        raise SystemExit(f"dataset has no rows: {summary}")

    run([
        sys.executable,
        str(V2_DIR / "train_value.py"),
        "--input",
        str(dataset),
        "--output-dir",
        str(model_dir),
        "--export-path",
        str(export_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--feature-dim",
        str(args.feature_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--train-samples",
        str(args.train_samples),
        "--val-samples",
        str(args.val_samples),
        "--device",
        args.device,
        "--log-every",
        "25",
        "--allow-train-val-fallback",
    ])
    verify_uci(export_path)
    maybe_run_match(export_path, args.match_games, args.match_think_ms)
    print(f"[done] dataset={dataset}", flush=True)
    print(f"[done] summary={summary}", flush=True)
    print(f"[done] model={export_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
