#!/usr/bin/env python3
"""Benchmark NPS for classic and NN backends across a few positions and time controls."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean


DEFAULT_FENS = [
    "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 13",
    "r3r1k1/pp1n1ppp/2p2n2/3p4/3P1B2/2N2N2/PP3PPP/R3R1K1 b - - 0 14",
    "8/5pk1/6p1/8/3P4/5KP1/8/8 w - - 0 42",
    "8/1p3pk1/p5p1/3P4/8/1P3KP1/8/8 b - - 0 39",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
]

INFO_RE = re.compile(r"^info .*?depth (?P<depth>-?\d+).*?nodes (?P<nodes>\d+).*?time (?P<time>\d+).*?nps (?P<nps>\d+)")


def uci_session(engine: Path, commands: list[str], timeout: float = 60.0) -> str:
    proc = subprocess.run(
        [str(engine)],
        text=True,
        input="\n".join(commands + ["quit"]) + "\n",
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return proc.stdout


def probe_backend(
    engine: Path,
    backend: str,
    model_path: Path | None,
    think_ms: int,
    fens: list[str],
) -> dict[str, object]:
    commands = ["uci"]
    if model_path is not None:
        commands.append(f"setoption name NNModel value {model_path}")
    commands.append(f"setoption name Backend value {backend}")
    commands.append("isready")
    for fen in fens:
        commands.append(f"position fen {fen}")
        commands.append(f"go movetime {think_ms}")
    out = uci_session(engine, commands, timeout=max(60.0, len(fens) * think_ms / 1000.0 + 20.0))

    rows: list[dict[str, int | str]] = []
    for line in out.splitlines():
        m = INFO_RE.match(line)
        if not m:
            continue
        rows.append({
            "depth": int(m.group("depth")),
            "nodes": int(m.group("nodes")),
            "time_ms": int(m.group("time")),
            "nps": int(m.group("nps")),
        })

    if len(rows) < len(fens):
        raise SystemExit(f"expected {len(fens)} info rows for {backend}, got {len(rows)}:\n{out}")
    rows = rows[-len(fens):]
    return {
        "backend": backend,
        "think_ms": think_ms,
        "positions": len(fens),
        "avg_depth": mean(row["depth"] for row in rows),
        "avg_nodes": mean(row["nodes"] for row in rows),
        "avg_nps": mean(row["nps"] for row in rows),
        "min_nps": min(row["nps"] for row in rows),
        "max_nps": max(row["nps"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NPS for classic and NN backends.")
    parser.add_argument("--engine", type=Path, default="bin/chess_uci", help="Path to chess_uci binary.")
    parser.add_argument("--nn-model", type=Path, default=None, help="Path to NN model (defaults to engine auto).")
    parser.add_argument("--think-ms", type=int, action="append", default=None, help="Think times to test.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--machine", default="local", help="Machine label for reporting.")
    args = parser.parse_args()

    engine = Path(args.engine).expanduser().resolve()
    if not engine.exists():
        raise SystemExit(f"engine not found: {engine}")

    think_times = args.think_ms or [120, 500, 1000]
    model_path = Path(args.nn_model).expanduser().resolve() if args.nn_model else None

    results: list[dict[str, object]] = []
    for think_ms in think_times:
        print(f"\n[think {think_ms}ms]")
        for backend, mp in [("classic", None), ("nn", model_path)]:
            try:
                probe = probe_backend(engine, backend, mp, think_ms, DEFAULT_FENS)
                results.append(probe)
                print(
                    f"  {backend:7s} depth={probe['avg_depth']:5.1f} "
                    f"nodes={probe['avg_nodes']:8.0f} nps={probe['avg_nps']:8.0f} "
                    f"[{probe['min_nps']}, {probe['max_nps']}]"
                )
            except Exception as exc:
                print(f"  {backend:7s} FAILED: {exc}")
                results.append({
                    "backend": backend,
                    "think_ms": think_ms,
                    "error": str(exc),
                })

    report = {
        "machine": args.machine,
        "engine": str(engine),
        "nn_model": str(model_path) if model_path else None,
        "timestamp": time.time(),
        "results": results,
    }

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        print(f"\n[wrote] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
