#!/usr/bin/env python3
"""Screen eval switches: ALL vs base, then ALL-minus-X vs ALL, one binary."""
import argparse
import subprocess
import sys
from pathlib import Path

SWITCHES = ["HcePawnPstFix", "HceKingPst", "HceKsFade", "HcePasser", "HceScale"]


def opts(name, on):
    out = []
    for sw in SWITCHES:
        out += ["--uci-option", f"{name}:{sw}={1 if sw in on else 0}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--positions-file", required=True)
    ap.add_argument("--positions-count", type=int, default=100)
    ap.add_argument("--clock", default="10+0.1")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="", help="comma list of run names")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [("all_vs_base", set(SWITCHES), set())]
    for sw in SWITCHES:
        runs.append((f"all_minus_{sw}_vs_all", set(SWITCHES) - {sw}, set(SWITCHES)))
    only = set(filter(None, args.only.split(",")))
    lab = Path(__file__).resolve().parent.parent / "src/core/bot/test_lab.py"
    for i, (name, cand, base) in enumerate(runs):
        if only and name not in only:
            continue
        report = out / f"{name}.json"
        if report.exists():
            continue
        cmd = [sys.executable, str(lab), "--engine", f"base={args.engine}", "--engine", f"cand={args.engine}",
               *opts("base", base), *opts("cand", cand), "--baseline", "base",
               "--positions-count", str(args.positions_count), "--clock", args.clock,
               "--positions-file", args.positions_file, "--concurrency", str(args.concurrency),
               "--seed", str(args.seed + i), "--out", str(report)]
        with open(out / f"{name}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)


if __name__ == "__main__":
    main()
