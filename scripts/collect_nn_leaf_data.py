#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "bin" / "chess_uci"
DEFAULT_MODEL = ROOT / "current" / "runpod_small_sweep" / "nnv2_normalized_64x16" / "nn_eval.bin"
DEFAULT_ROOTS = [
    ROOT / "data" / "positions" / "lichess_equal_positions.fen",
    ROOT / "data" / "positions" / "hce_position_suite.jsonl",
]


def iter_roots(path: Path) -> Iterable[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if path.suffix == ".jsonl":
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fen = str(payload.get("fen") or "").strip()
                if fen:
                    yield f"position fen {fen}"
                continue
            yield f"position fen {line}"


def dedupe_leaf_log(raw_path: Path, out_path: Path) -> tuple[int, int]:
    seen: set[str] = set()
    total = 0
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            total += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            fen = row.get("fen")
            if not isinstance(fen, str) or fen in seen:
                continue
            seen.add(fen)
            dst.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            dst.write("\n")
            kept += 1
    return total, kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect NN search leaf FENs from local UCI search.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "current" / "nn_leaf_data" / "leaves.jsonl")
    parser.add_argument("--roots", type=Path, action="append", default=None)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--think-ms", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--keep-raw", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not ENGINE.exists():
        subprocess.run(["make", "bin/chess_uci"], cwd=ROOT, check=True)
    model = args.model.resolve()
    if not model.exists():
        raise SystemExit(f"missing NN model: {model}")

    root_paths = args.roots if args.roots is not None else DEFAULT_ROOTS
    position_commands: list[str] = []
    for path in root_paths:
        for command in iter_roots(path.resolve()):
            position_commands.append(command)
            if len(position_commands) >= args.positions:
                break
        if len(position_commands) >= args.positions:
            break
    if not position_commands:
        raise SystemExit("no root positions found")

    output = args.output.resolve()
    raw_output = output.with_suffix(output.suffix + ".raw")
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    commands = [
        "uci",
        f"setoption name NNModel value {model}",
        "setoption name Backend value nn",
        f"setoption name NNLeafLogLimit value {max(0, args.limit)}",
        f"setoption name NNLeafLog value {raw_output}",
        "isready",
    ]
    for command in position_commands:
        commands.append(command)
        commands.append(f"go movetime {max(1, args.think_ms)} depth {max(1, args.max_depth)}")
    commands.append("quit")

    timeout = max(30.0, len(position_commands) * max(1, args.think_ms) / 1000.0 + 30.0)
    proc = subprocess.run(
        [str(ENGINE)],
        cwd=ROOT,
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    if "backend set to nn" not in proc.stdout:
        raise SystemExit(f"engine did not enter NN backend:\n{proc.stdout}")
    if not raw_output.exists():
        raise SystemExit(f"engine did not create leaf log: {raw_output}")

    raw_rows, unique_rows = dedupe_leaf_log(raw_output, output)
    if not args.keep_raw:
        raw_output.unlink(missing_ok=True)

    summary = {
        "model": str(model),
        "output": str(output),
        "roots": [str(path.resolve()) for path in root_paths],
        "positions": len(position_commands),
        "think_ms": args.think_ms,
        "max_depth": args.max_depth,
        "limit": args.limit,
        "raw_rows": raw_rows,
        "unique_rows": unique_rows,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
