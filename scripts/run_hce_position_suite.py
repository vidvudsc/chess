#!/usr/bin/env python3
"""Run a small HCE position-regression suite through the UCI engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT / "bin" / "chess_uci"
DEFAULT_SUITE = ROOT / "data" / "positions" / "hce_position_suite.jsonl"


def load_suite(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            case.setdefault("id", f"case_{line_no}")
            cases.append(case)
    if not cases:
        raise SystemExit(f"{path}: no suite cases found")
    return cases


def configure_engine(engine: chess.engine.SimpleEngine, args: argparse.Namespace) -> None:
    options: dict[str, object] = {}
    if "Backend" in engine.options:
        options["Backend"] = args.backend
    if "MoveTime" in engine.options:
        options["MoveTime"] = args.default_think_ms
    if "MaxDepth" in engine.options:
        options["MaxDepth"] = args.default_max_depth
    if args.nn_model and "NNModel" in engine.options:
        options["NNModel"] = str(args.nn_model)
    if options:
        engine.configure(options)


def legal_uci_set(board: chess.Board) -> set[str]:
    return {move.uci() for move in board.legal_moves}


def run_case(engine: chess.engine.SimpleEngine, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    board = chess.Board(case["fen"])
    think_ms = int(case.get("think_ms", args.default_think_ms))
    max_depth = int(case.get("max_depth", args.default_max_depth))
    limit = chess.engine.Limit(time=think_ms / 1000.0, depth=max_depth)

    started = time.perf_counter()
    result = engine.play(board, limit)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    move_uci = result.move.uci() if result.move is not None else "0000"

    legal = legal_uci_set(board)
    passed = result.move is not None and move_uci in legal
    reason = "ok" if passed else "engine returned no legal move"

    kind = case.get("kind", "avoid_move")
    if passed and kind == "avoid_move":
        avoid = set(case.get("avoid", []))
        if move_uci in avoid:
            passed = False
            reason = f"played avoided move {move_uci}"
    elif passed and kind == "prefer_move":
        prefer = set(case.get("prefer", []))
        if prefer and move_uci not in prefer:
            passed = False
            reason = f"expected one of {sorted(prefer)}, got {move_uci}"
    elif passed and kind == "allow_move":
        allow = set(case.get("allow", []))
        if allow and move_uci not in allow:
            passed = False
            reason = f"expected allowed move from {sorted(allow)}, got {move_uci}"
    elif kind not in {"avoid_move", "prefer_move", "allow_move"}:
        passed = False
        reason = f"unknown case kind {kind!r}"

    return {
        "id": case["id"],
        "kind": kind,
        "fen": case["fen"],
        "move": move_uci,
        "pass": passed,
        "reason": reason,
        "think_ms": think_ms,
        "max_depth": max_depth,
        "elapsed_ms": elapsed_ms,
        "tags": case.get("tags", []),
        "note": case.get("note", ""),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--backend", default="classic", choices=["classic", "nn", "experimental"])
    parser.add_argument("--nn-model", type=Path)
    parser.add_argument("--default-think-ms", type=int, default=120)
    parser.add_argument("--default-max-depth", type=int, default=10)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    engine_path = args.engine.resolve()
    suite_path = args.suite.resolve()
    if not engine_path.exists():
        raise SystemExit(f"engine not found: {engine_path}")
    cases = load_suite(suite_path)

    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "current" / f"hce_position_suite_{stamp}.json"
    if str(out_path) != "-":
        out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failed = 0
    started = time.perf_counter()
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        configure_engine(engine, args)
        for case in cases:
            row = run_case(engine, case, args)
            rows.append(row)
            mark = "PASS" if row["pass"] else "FAIL"
            if not row["pass"]:
                failed += 1
            print(f"{mark} {row['id']}: {row['move']} ({row['reason']})")
            if args.fail_fast and failed:
                break
    finally:
        engine.quit()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    report = {
        "engine": str(engine_path),
        "suite": str(suite_path),
        "backend": args.backend,
        "case_count": len(rows),
        "failed": failed,
        "elapsed_ms": elapsed_ms,
        "results": rows,
    }
    if str(out_path) == "-":
        print(json.dumps(report, indent=2))
    else:
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    print(f"hce_position_suite: {len(rows) - failed}/{len(rows)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
