#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "bin" / "chess_uci"
REPORT_DIR = ROOT / "current" / "nnue_eval_matches"

DEFAULT_PROBE_FENS = [
    "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "2r2rk1/1bqnbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 13",
    "r3r1k1/pp1n1ppp/2p2n2/3p4/3P1B2/2N2N2/PP3PPP/R3R1K1 b - - 0 14",
    "8/5pk1/6p1/8/3P4/5KP1/8/8 w - - 0 42",
    "8/1p3pk1/p5p1/3P4/8/1P3KP1/8/8 b - - 0 39",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
]

DEFAULT_LINE_MOVES = [
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5a4",
    "g8f6",
    "e1g1",
    "f8e7",
    "f1e1",
    "b7b5",
    "a4b3",
    "d7d6",
    "c2c3",
    "e8g8",
    "h2h3",
    "c6b8",
    "d2d4",
    "b8d7",
    "c1g5",
    "c8b7",
    "b1d2",
    "h7h6",
]

INFO_RE = re.compile(r"^info .*?depth (?P<depth>-?\d+).*?nodes (?P<nodes>\d+).*?time (?P<time>\d+).*?nps (?P<nps>\d+)")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[run] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def ensure_engine() -> None:
    if not ENGINE.exists():
        run(["make", "bin/chess_uci"])


def uci_session(commands: list[str], timeout: float = 30.0) -> str:
    proc = subprocess.run(
        [str(ENGINE)],
        cwd=ROOT,
        text=True,
        input="\n".join(commands + ["quit"]) + "\n",
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return proc.stdout


def smoke_model(model_path: Path) -> None:
    out = uci_session([
        "uci",
        f"setoption name NNModel value {model_path}",
        "setoption name Backend value nn",
        "isready",
    ])
    if "readyok" not in out or "backend set to nn" not in out:
        raise SystemExit(f"NN UCI smoke failed for {model_path}:\n{out}")
    print(f"[ok] UCI loaded {model_path}", flush=True)


def line_position_commands(moves: list[str]) -> list[str]:
    commands = ["position startpos"]
    for index in range(1, len(moves) + 1):
        commands.append("position startpos moves " + " ".join(moves[:index]))
    return commands


def probe_backend_positions(name: str,
                            backend: str,
                            model_path: Path | None,
                            think_ms: int,
                            position_commands: list[str],
                            suite: str) -> dict[str, object]:
    commands = ["uci"]
    if model_path is not None:
        commands.append(f"setoption name NNModel value {model_path}")
    commands.append(f"setoption name Backend value {backend}")
    commands.append("isready")
    for position_command in position_commands:
        commands.append(position_command)
        commands.append(f"go movetime {think_ms}")
    out = uci_session(commands, timeout=max(30.0, len(position_commands) * think_ms / 1000.0 + 20.0))

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
            "raw": line,
        })

    if len(rows) < len(position_commands):
        raise SystemExit(f"expected at least {len(position_commands)} info rows for {name}, got {len(rows)}:\n{out}")
    rows = rows[-len(position_commands):]
    return {
        "name": name,
        "backend": backend,
        "suite": suite,
        "think_ms": think_ms,
        "positions": len(position_commands),
        "avg_depth": mean(row["depth"] for row in rows),
        "avg_nodes": mean(row["nodes"] for row in rows),
        "avg_nps": mean(row["nps"] for row in rows),
        "rows": rows,
    }


def probe_backend(name: str, backend: str, model_path: Path | None, think_ms: int, fens: list[str]) -> dict[str, object]:
    return probe_backend_positions(
        name,
        backend,
        model_path,
        think_ms,
        [f"position fen {fen}" for fen in fens],
        "fen",
    )


def probe_backend_line(name: str, backend: str, model_path: Path | None, think_ms: int) -> dict[str, object]:
    return probe_backend_positions(
        name,
        backend,
        model_path,
        think_ms,
        line_position_commands(DEFAULT_LINE_MOVES),
        "line",
    )


def run_match(model_path: Path, label: str, positions: int, think_ms: int, concurrency: int) -> Path:
    out = REPORT_DIR / f"{label}_vs_classic_{positions}pos_{think_ms}ms.json"
    cmd = [
        sys.executable,
        "src/core/bot/test_lab.py",
        "--engine",
        "classic=bin/chess_uci",
        "--engine",
        "nn=bin/chess_uci",
        "--uci-option",
        f"nn:NNModel={model_path}",
        "--uci-option",
        "nn:Backend=nn",
        "--positions-count",
        str(positions),
        "--think-ms",
        str(think_ms),
        "--concurrency",
        str(concurrency),
        "--baseline",
        "classic",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def summarize_match(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    h2h = payload.get("head_to_head") or []
    standings = payload.get("standings") or []
    nn_row = next((row for row in standings if row.get("name") == "nn"), {})
    return {
        "path": str(path),
        "head_to_head": h2h[0] if h2h else {},
        "nn_standing": nn_row,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exported CHNNUE1 models against the classic backend.")
    parser.add_argument("model", type=Path, nargs="+", help="Path(s) to nn_eval.bin candidates.")
    parser.add_argument("--label", action="append", default=[], help="Optional label(s), one per model.")
    parser.add_argument("--probe-think-ms", type=int, action="append", default=None)
    parser.add_argument("--match-positions", type=int, default=16)
    parser.add_argument("--match-think-ms", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-match", action="store_true")
    parser.add_argument("--skip-line-probe", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.probe_think_ms is None:
        args.probe_think_ms = [120, 1000]
    ensure_engine()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    labels = args.label or []
    if labels and len(labels) != len(args.model):
        raise SystemExit("--label count must match model count")

    report: dict[str, object] = {
        "generated_at_unix": time.time(),
        "models": [],
    }

    classic_fen_probes = [
        probe_backend("classic", "classic", None, think_ms, DEFAULT_PROBE_FENS)
        for think_ms in args.probe_think_ms
    ]
    report["classic_fen_probes"] = classic_fen_probes
    if not args.skip_line_probe:
        report["classic_line_probes"] = [
            probe_backend_line("classic", "classic", None, think_ms)
            for think_ms in args.probe_think_ms
        ]

    for idx, raw_model in enumerate(args.model):
        model_path = raw_model.resolve()
        if not model_path.exists():
            raise SystemExit(f"missing model: {model_path}")
        label = labels[idx] if labels else model_path.parent.name
        smoke_model(model_path)

        fen_probes = [
            probe_backend(label, "nn", model_path, think_ms, DEFAULT_PROBE_FENS)
            for think_ms in args.probe_think_ms
        ]
        line_probes = []
        if not args.skip_line_probe:
            line_probes = [
                probe_backend_line(label, "nn", model_path, think_ms)
                for think_ms in args.probe_think_ms
            ]
        matches = []
        if not args.skip_match:
            match_path = run_match(model_path, label, args.match_positions, args.match_think_ms, args.concurrency)
            matches.append(summarize_match(match_path))

        model_report = {
            "label": label,
            "model": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "fen_probes": fen_probes,
            "line_probes": line_probes,
            "matches": matches,
        }
        report["models"].append(model_report)

        for probe in fen_probes + line_probes:
            print(
                f"[probe:{probe['suite']}] {label} {probe['think_ms']}ms depth={probe['avg_depth']:.2f} "
                f"nps={probe['avg_nps']:.0f}",
                flush=True,
            )
        if matches:
            h2h = matches[-1]["head_to_head"]
            print(
                f"[match] {label} score={h2h.get('points')}/{h2h.get('games')} "
                f"elo={h2h.get('elo_diff')}",
                flush=True,
            )

    out = args.report
    if out is None:
        out = REPORT_DIR / f"nnue_candidate_eval_{int(time.time())}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
