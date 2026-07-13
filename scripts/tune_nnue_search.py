#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "bin" / "chess_uci"
DEFAULT_MODEL = ROOT / "current" / "runpod_linear256x32_wdl400_full_fast" / "best_smoke_nn_eval.bin"
DEFAULT_OUT_DIR = ROOT / "current" / "nnue_search_tuning"


def run(cmd: list[str]) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_engine() -> None:
    if not ENGINE.exists():
        run(["make", "bin/chess_uci"])


def safe_label(options: dict[str, int]) -> str:
    parts = []
    for key in sorted(options):
        short = key.removeprefix("NN")
        parts.append(f"{short}{options[key]}")
    return "_".join(parts).replace("-", "m")


def preset_configs(kind: str) -> list[dict[str, int]]:
    if kind == "scale":
        return [{"NNEvalScale": value} for value in (600, 750, 900, 1000, 1100, 1250, 1400)]
    if kind == "broad":
        configs: list[dict[str, int]] = []
        for scale in (750, 900, 1000, 1100, 1250):
            configs.append({"NNEvalScale": scale})
        for scale in (750, 900, 1000, 1100):
            for qdelta in (120, 180, 240):
                configs.append({"NNEvalScale": scale, "NNQDeltaMargin": qdelta})
        for scale in (750, 900, 1000, 1100):
            for margin in (80, 130, 180):
                configs.append({"NNEvalScale": scale, "NNStaticPruneMargin": margin})
        return configs
    if kind == "refine":
        return [
            {"NNEvalScale": 700, "NNQDeltaMargin": 120, "NNStaticPruneMargin": 80},
            {"NNEvalScale": 750, "NNQDeltaMargin": 120, "NNStaticPruneMargin": 80},
            {"NNEvalScale": 800, "NNQDeltaMargin": 120, "NNStaticPruneMargin": 80},
            {"NNEvalScale": 750, "NNQDeltaMargin": 180, "NNStaticPruneMargin": 80},
            {"NNEvalScale": 750, "NNQDeltaMargin": 120, "NNStaticPruneMargin": 130},
            {"NNEvalScale": 900, "NNQDeltaMargin": 120, "NNStaticPruneMargin": 80},
            {"NNEvalScale": 900, "NNQDeltaMargin": 180, "NNStaticPruneMargin": 130},
            {"NNEvalScale": 1000, "NNQDeltaMargin": 120, "NNStaticPruneMargin": 80},
            {"NNEvalScale": 1000, "NNQDeltaMargin": 180, "NNStaticPruneMargin": 130},
        ]
    raise ValueError(f"unknown preset: {kind}")


def load_summary(report_path: Path) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    h2h = payload.get("head_to_head") or []
    if not h2h:
        raise RuntimeError(f"missing head_to_head in {report_path}")
    row = h2h[0]
    return {
        "report": str(report_path),
        "games": row.get("games", 0),
        "points": row.get("points", 0.0),
        "score_fraction": row.get("score_fraction", 0.0),
        "elo_diff": row.get("elo_diff"),
        "elo_ci_low": row.get("elo_ci_low"),
        "elo_ci_high": row.get("elo_ci_high"),
        "probability_better": row.get("probability_better"),
    }


def run_candidate(model: Path,
                  out_dir: Path,
                  label: str,
                  options: dict[str, int],
                  positions: int,
                  think_ms: int,
                  concurrency: int,
                  seed: int,
                  book_file: str) -> dict[str, Any]:
    report_path = out_dir / f"{label}_{positions}pos_{think_ms}ms.json"
    cmd = [
        sys.executable,
        "src/core/bot/test_lab.py",
        "--engine",
        "classic=bin/chess_uci",
        "--engine",
        "nn=bin/chess_uci",
        "--uci-option",
        f"nn:NNModel={model}",
        "--uci-option",
        "nn:Backend=nn",
        "--positions-count",
        str(positions),
        "--think-ms",
        str(think_ms),
        "--concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--baseline",
        "classic",
        "--out",
        str(report_path),
    ]
    if book_file:
        cmd.extend(["--book-file", book_file])
    for name, value in sorted(options.items()):
        cmd.extend(["--uci-option", f"nn:{name}={value}"])
    run(cmd)
    summary = load_summary(report_path)
    summary["label"] = label
    summary["options"] = dict(options)
    print(
        f"[result] {label} score={summary['points']}/{summary['games']} "
        f"elo={summary['elo_diff']} options={options}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune NNUE UCI search options against classic HCE.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--preset", choices=["scale", "broad", "refine"], default="scale")
    parser.add_argument("--positions", type=int, default=64, help="FEN positions; paired colors doubles game count.")
    parser.add_argument("--think-ms", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--book-file", default="", help="Optional book path. Empty disables book.")
    parser.add_argument("--config", action="append", default=[],
                        help="Manual config like NNEvalScale=900,NNQDeltaMargin=180. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N configs.")
    return parser


def parse_config(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"bad --config item: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = int(value.strip())
    if not out:
        raise SystemExit("--config cannot be empty")
    return out


def main() -> int:
    args = build_parser().parse_args()
    ensure_engine()
    model = args.model.resolve()
    if not model.exists():
        raise SystemExit(f"missing model: {model}")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [parse_config(raw) for raw in args.config] if args.config else preset_configs(args.preset)
    seen: set[str] = set()
    unique_configs: list[dict[str, int]] = []
    for config in configs:
        key = json.dumps(config, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique_configs.append(config)
    if args.limit > 0:
        unique_configs = unique_configs[:args.limit]

    summaries = []
    started = time.time()
    for index, options in enumerate(unique_configs, start=1):
        label = f"{args.preset}_{index:02d}_{safe_label(options)}"
        summaries.append(
            run_candidate(
                model=model,
                out_dir=out_dir,
                label=label,
                options=options,
                positions=args.positions,
                think_ms=args.think_ms,
                concurrency=args.concurrency,
                seed=args.seed,
                book_file=args.book_file,
            )
        )
        ranked = sorted(summaries, key=lambda row: (row["score_fraction"], row["points"]), reverse=True)
        best = ranked[0]
        print(
            f"[leader] {best['label']} score={best['points']}/{best['games']} "
            f"elo={best['elo_diff']} options={best['options']}",
            flush=True,
        )

    ranked = sorted(summaries, key=lambda row: (row["score_fraction"], row["points"]), reverse=True)
    summary_path = out_dir / f"{args.preset}_{args.positions}pos_{args.think_ms}ms_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "generated_at_unix": time.time(),
                "elapsed_s": time.time() - started,
                "model": str(model),
                "preset": args.preset,
                "positions": args.positions,
                "think_ms": args.think_ms,
                "concurrency": args.concurrency,
                "seed": args.seed,
                "ranked": ranked,
            },
            fp,
            indent=2,
            sort_keys=True,
        )
    print(f"[done] {summary_path}", flush=True)
    if ranked:
        best = ranked[0]
        print(f"[best] {best['label']} {best['points']}/{best['games']} elo={best['elo_diff']} {best['options']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
