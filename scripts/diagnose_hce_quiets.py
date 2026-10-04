#!/usr/bin/env python3
"""Replay quiet-move errors with isolated search ablations; never edit live code.

This is a diagnostic on selected losing positions, not a playing-strength test.
Every HCE search starts in a fresh process because ucinewgame retains its TT.
Stockfish compares all candidate moves together at the same root and depth.
"""

import argparse
import hashlib
import json
import random
import shutil
import statistics
import subprocess
from pathlib import Path

import chess
import chess.engine


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one diagnostic patch target: {old!r}")
    return text.replace(old, new, 1)


def build_variants(out):
    binaries = {}
    original = ROOT / "src/core/engine"
    for name in ("baseline", "no_lmp", "no_lmr", "no_lmp_lmr"):
        source = out / name
        if source.exists():
            raise FileExistsError(f"Use a fresh output directory: {source}")
        shutil.copytree(original, source,
                        ignore=shutil.ignore_patterns("build", "chess_uci"))
        path = source / "hce_search.c"
        text = path.read_text()
        if name in ("no_lmp", "no_lmp_lmr"):
            text = replace_once(text, "            depth <= 8 &&",
                                "            false && /* diagnostic: disable LMP */")
        if name in ("no_lmr", "no_lmp_lmr"):
            text = replace_once(text, "                if (reduction < 0) {",
                                "                reduction = 0; /* diagnostic: disable LMR */\n"
                                "                if (reduction < 0) {")
        path.write_text(text)
        with (out / f"build_{name}.log").open("w") as log:
            subprocess.run(["make", "-C", str(source), "-j4"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[name] = source / "chess_uci"
        print(f"built {name}", flush=True)
    return binaries


def load_cases(path, maximum, seed):
    cases, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not 50 <= row["loss_cp"] <= 2000:
            continue
        if max(abs(row["best_cp"]), abs(row["played_cp"])) >= 10000:
            continue
        board = chess.Board(row["fen"])
        moves = [chess.Move.from_uci(row[key]) for key in ("played", "best")]
        if any(move not in board.legal_moves for move in moves):
            raise ValueError(f"Illegal recorded move: {row}")
        if moves[0] == moves[1] or any(board.is_capture(m) or m.promotion for m in moves):
            continue
        if row["fen"] in seen:
            continue
        seen.add(row["fen"])
        cases.append(row)
    random.Random(seed).shuffle(cases)
    return cases[:maximum] if maximum else cases


def search(binary, board, ms, empty_book):
    with chess.engine.SimpleEngine.popen_uci(str(binary), timeout=30) as engine:
        engine.configure({"Backend": "classic", "Threads": 1, "MaxDepth": 32,
                          "BookFile": str(empty_book)})
        result = engine.play(board, chess.engine.Limit(time=ms / 1000),
                             info=chess.engine.INFO_ALL)
        if result.move not in board.legal_moves:
            raise ValueError(f"Illegal result: {result.move}")
        if result.info.get("depth", 0) < 1:
            raise ValueError("Search returned no completed depth (possibly a book move)")
        score = result.info.get("score")
        return {"move": result.move.uci(), "depth": result.info.get("depth"),
                "nodes": result.info.get("nodes"), "time": result.info.get("time"),
                "score_cp": score.pov(board.turn).score(mate_score=30000) if score else None,
                "pv": [m.uci() for m in result.info.get("pv", [])]}


def judge(engine, board, moves, depth):
    # Fresh game plus explicit Clear Hash prevents prior cases affecting labels.
    engine.configure({"Clear Hash": None})
    discovery = engine.analyse(board, chess.engine.Limit(depth=depth), game=object())
    moves = set(moves) | {discovery["pv"][0].uci()}
    comparison = engine.analyse(
        board, chess.engine.Limit(depth=depth),
        root_moves=[chess.Move.from_uci(m) for m in sorted(moves)], multipv=len(moves))
    result = {}
    for info in comparison:
        score = info["score"].pov(board.turn)
        result[info["pv"][0].uci()] = {
            "cp": score.score(mate_score=30000), "mate": score.mate(),
            "depth": info["depth"], "pv": [m.uci() for m in info["pv"]]}
    if set(result) != moves:
        raise ValueError("Teacher did not score every candidate")
    return result


def summarize(rows, arms):
    # Keep mate horizons separate from centipawn averages.
    finite = [r for r in rows if all(s["mate"] is None for s in r["teacher"].values())]
    confirmed = [r for r in finite if r["played_regret_cp"] >= 50]
    competitive = [r for r in confirmed if abs(max(s["cp"] for s in r["teacher"].values())) <= 300]
    summaries = {}
    for label, subset in (("all_nonmate", finite), ("reconfirmed_errors", confirmed),
                          ("competitive_reconfirmed_errors", competitive)):
        summaries[label] = {"positions": len(subset), "arms": {}}
        for arm in arms:
            vals = [r["searches"][arm] for r in subset]
            summaries[label]["arms"][arm] = {
                "within_30cp": sum(v["regret_cp"] <= 30 for v in vals),
                "mean_regret_cp": statistics.mean(v["regret_cp"] for v in vals) if vals else None,
                "median_depth": statistics.median(v["depth"] for v in vals) if vals else None,
                "better_than_long_by_30cp": sum(
                    r["searches"]["baseline_long"]["regret_cp"] - r["searches"][arm]["regret_cp"] >= 30
                    for r in subset),
                "worse_than_long_by_30cp": sum(
                    r["searches"][arm]["regret_cp"] - r["searches"]["baseline_long"]["regret_cp"] >= 30
                    for r in subset),
            }
    summaries["mate_horizon_cases"] = len(rows) - len(finite)
    return summaries


def follow_up(out, teacher_path, count=8, time_ms=12000, depth=18):
    """Give persistent errors another 10x time and independently deepen labels.

    Take at most one case per game, ranked by baseline regret, with the best
    reference move within +/-300 cp. This deliberately selects difficult but
    competitive cases and must not be read as a random strength sample.
    """
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    persistent = [r for r in rows
                  if all(s["mate"] is None for s in r["teacher"].values())
                  and abs(max(s["cp"] for s in r["teacher"].values())) <= 300
                  and min(s["regret_cp"] for s in r["searches"].values()) > 30]
    persistent.sort(key=lambda r: r["searches"]["baseline_long"]["regret_cp"], reverse=True)
    selected, games = [], set()
    for row in persistent:
        game = row["case"]["game_id"]
        if game not in games:
            games.add(game)
            selected.append(row)
        if len(selected) >= count:
            break
    result_path = out / "followup.jsonl"
    with result_path.open("x") as target, chess.engine.SimpleEngine.popen_uci(
            str(teacher_path), timeout=120) as teacher:
        teacher.configure({"Threads": 1, "Hash": 64})
        for i, row in enumerate(selected):
            board = chess.Board(row["case"]["fen"])
            deep = search(out / "baseline/chess_uci", board, time_ms, out / "empty_book.txt")
            moves = list(row["teacher"]) + [deep["move"]]
            scores = judge(teacher, board, moves, depth)
            best = max(s["cp"] for s in scores.values())
            deep["regret_cp"] = best - scores[deep["move"]]["cp"]
            result = {"case": row["case"], "deep": deep, "teacher": scores,
                      "budget_ms": time_ms, "teacher_depth": depth,
                      "earlier_moves_rejudged": {
                          a: {"move": s["move"], "regret_cp": best - scores[s["move"]]["cp"]}
                          for a, s in row["searches"].items()}}
            target.write(json.dumps(result) + "\n")
            target.flush()
            print(f"deep follow-up {i + 1}/{len(selected)}: {deep['regret_cp']} cp", flush=True)
    return result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--short-ms", type=int, default=120)
    parser.add_argument("--long-ms", type=int, default=1200)
    parser.add_argument("--teacher", type=Path, default=Path("/opt/homebrew/bin/stockfish"))
    parser.add_argument("--teacher-depth", type=int, default=14)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = load_cases(args.input, args.max_positions, args.seed)
    if not cases:
        raise ValueError("No quiet errors found")
    binaries = build_variants(out)
    empty_book = out / "empty_book.txt"
    empty_book.write_text("")
    arms = {"baseline_short": ("baseline", args.short_ms),
            "baseline_long": ("baseline", args.long_ms),
            "no_lmp": ("no_lmp", args.long_ms),
            "no_lmr": ("no_lmr", args.long_ms),
            "no_lmp_lmr": ("no_lmp_lmr", args.long_ms)}
    manifest = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "input": str(args.input.resolve()), "input_sha256": digest(args.input),
                "positions": len(cases), "seed": args.seed, "arms": arms,
                "teacher_depth": args.teacher_depth, "teacher_sha256": digest(args.teacher),
                "binaries": {k: digest(v) for k, v in binaries.items()},
                "limitations": "Selected historical losses; FEN replay omits repetition history; single-thread cold TT; no Elo claim."}
    rows = []
    with chess.engine.SimpleEngine.popen_uci(str(args.teacher), timeout=60) as teacher:
        teacher.configure({"Threads": 1, "Hash": 64})
        manifest["teacher_id"] = teacher.id
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (out / "results.jsonl").open("w") as target:
            for i, case in enumerate(cases):
                board = chess.Board(case["fen"])
                searches = {arm: search(binaries[name], board, ms, empty_book)
                            for arm, (name, ms) in arms.items()}
                moves = [case["played"], case["best"]] + [s["move"] for s in searches.values()]
                scores = judge(teacher, board, moves, args.teacher_depth)
                best = max(s["cp"] for s in scores.values())
                for value in searches.values():
                    value["regret_cp"] = best - scores[value["move"]]["cp"]
                row = {"case": case, "searches": searches, "teacher": scores,
                       "played_regret_cp": best - scores[case["played"]]["cp"]}
                rows.append(row)
                target.write(json.dumps(row) + "\n")
                target.flush()
                (out / "summary.json").write_text(json.dumps(summarize(rows, arms), indent=2) + "\n")
                if (i + 1) % 10 == 0 or i + 1 == len(cases):
                    print(f"completed {i + 1}/{len(cases)}", flush=True)
    print(json.dumps(summarize(rows, arms), indent=2))


if __name__ == "__main__":
    main()
