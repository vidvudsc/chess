#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import torch

ROOT = Path(__file__).resolve().parents[1]
NN_ROOT = ROOT / "src" / "core" / "bot" / "nn"
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from policy_infer import load_model, rank_legal_moves  # noqa: E402
from policy_model import ConvPolicyValueNet  # noqa: E402


MATE_SCORE = 100000


@dataclass
class PositionResult:
    fen: str
    hce_move: str
    policy_hce_move: str
    policy_value_move: str
    policy_top5: list[str]
    hce_in_policy_top5: bool
    hce_in_policy_topk: bool
    policy_hce_matches_hce: bool
    policy_value_matches_hce: bool
    hce_score_cp: int | None
    policy_hce_score_cp: int | None
    policy_value_score_cp: int | None


def generate_random_positions(count: int, seed: int, min_plies: int, max_plies: int) -> list[chess.Board]:
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    attempts = 0
    while len(positions) < count and attempts < count * 20:
        attempts += 1
        board = chess.Board()
        plies = rng.randint(min_plies, max_plies)
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over(claim_draw=True):
            continue
        if board.legal_moves.count() < 2:
            continue
        positions.append(board)
    return positions


def hce_play(engine: chess.engine.SimpleEngine,
             board: chess.Board,
             time_s: float,
             depth: int,
             root_moves: list[chess.Move] | None = None) -> chess.Move:
    limit = chess.engine.Limit(time=max(0.001, time_s))
    if depth > 0:
        limit = chess.engine.Limit(time=max(0.001, time_s), depth=depth)
    result = engine.play(board, limit, root_moves=root_moves)
    if result.move is None:
        raise RuntimeError("HCE returned no move")
    return result.move


def hce_score_after_move(engine: chess.engine.SimpleEngine,
                         board: chess.Board,
                         move: chess.Move,
                         time_s: float,
                         depth: int) -> int | None:
    child = board.copy(stack=False)
    child.push(move)
    limit = chess.engine.Limit(time=max(0.001, time_s))
    if depth > 0:
        limit = chess.engine.Limit(time=max(0.001, time_s), depth=depth)
    try:
        info = engine.analyse(child, limit)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
        return None
    score = info.get("score")
    if score is None:
        return None
    # Child is opponent-to-move, so ask how good the child is for the root side.
    return score.pov(board.turn).score(mate_score=MATE_SCORE)


def policy_hce_root_move(engine: chess.engine.SimpleEngine,
                         board: chess.Board,
                         root_moves: list[chess.Move],
                         time_s: float,
                         depth: int) -> tuple[chess.Move, int | None]:
    best_move = root_moves[0]
    best_score: int | None = None
    for move in root_moves:
        score = hce_score_after_move(engine, board, move, time_s, depth)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_move = move
    return best_move, best_score


def policy_soft_hce_move(engine: chess.engine.SimpleEngine,
                         board: chess.Board,
                         ranked_moves: list[chess.Move],
                         bonus: int,
                         time_s: float,
                         depth: int) -> chess.Move:
    engine.configure({
        "PolicyRootHints": " ".join(move.uci() for move in ranked_moves),
        "PolicyRootBonus": max(0, int(bonus)),
    })
    return hce_play(engine, board, time_s, depth)


def terminal_value_stm(board: chess.Board, ply: int) -> int | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0
    return MATE_SCORE - ply if outcome.winner == board.turn else -MATE_SCORE + ply


def nn_value_cp_stm(model: ConvPolicyValueNet,
                    board: chess.Board,
                    device: torch.device,
                    cache: dict[str, float]) -> int:
    key = board.transposition_key() if hasattr(board, "transposition_key") else board.board_fen() + str(board.turn)
    key_s = str(key)
    if key_s not in cache:
        _, value = rank_legal_moves(model, board, device, limit=1)
        cache[key_s] = value
    # The Chess-Alpha value convention is not documented well enough to trust as
    # a calibrated cp eval. Keep this as a bounded ordering signal only.
    return int(cache[key_s] * 1000.0)


def policy_value_negamax(model: ConvPolicyValueNet,
                         board: chess.Board,
                         device: torch.device,
                         depth: int,
                         alpha: int,
                         beta: int,
                         breadth: int,
                         cache: dict[str, float],
                         ply: int = 0) -> tuple[int, chess.Move | None]:
    term = terminal_value_stm(board, ply)
    if term is not None:
        return term, None
    if depth <= 0:
        return nn_value_cp_stm(model, board, device, cache), None

    ranked, _ = rank_legal_moves(model, board, device, limit=max(1, breadth))
    moves = [move for move, _ in ranked]
    if not moves:
        return nn_value_cp_stm(model, board, device, cache), None

    best_score = -MATE_SCORE
    best_move: chess.Move | None = None
    for move in moves:
        board.push(move)
        score, _ = policy_value_negamax(model, board, device, depth - 1, -beta, -alpha, breadth, cache, ply + 1)
        score = -score
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_score, best_move


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    engine = chess.engine.SimpleEngine.popen_uci(str(args.engine))
    engine.configure({"Backend": "classic", "MoveTime": int(args.hce_time_ms), "MaxDepth": int(args.hce_depth)})
    positions = generate_random_positions(args.positions, args.seed, args.min_plies, args.max_plies)

    results: list[PositionResult] = []
    start = time.time()
    try:
        for idx, board in enumerate(positions, start=1):
            engine.configure({"PolicyRootHints": "", "PolicyRootBonus": 0})
            hce_move = hce_play(engine, board, args.hce_time_ms / 1000.0, args.hce_depth)
            ranked, _ = rank_legal_moves(model, board, device, limit=max(args.policy_topk, args.policy_value_breadth))
            policy_moves = [move for move, _ in ranked]
            root_moves = policy_moves[:args.policy_topk]
            if args.policy_hce_mode == "hard-root":
                policy_hce_move, policy_hce_score = policy_hce_root_move(
                    engine,
                    board,
                    root_moves,
                    args.policy_hce_child_time_ms / 1000.0,
                    args.policy_hce_child_depth,
                )
            else:
                policy_hce_move = policy_soft_hce_move(
                    engine,
                    board,
                    root_moves,
                    args.policy_root_bonus,
                    args.hce_time_ms / 1000.0,
                    args.hce_depth,
                )
                policy_hce_score = None
            pv_score, policy_value_move = policy_value_negamax(
                model,
                board.copy(stack=False),
                device,
                args.policy_value_depth,
                -MATE_SCORE,
                MATE_SCORE,
                args.policy_value_breadth,
                {},
            )
            if policy_value_move is None:
                policy_value_move = policy_moves[0]

            hce_score = hce_score_after_move(engine, board, hce_move, args.eval_time_ms / 1000.0, args.eval_depth)
            if policy_hce_score is None or args.rescore_policy_hce:
                policy_hce_score = hce_score_after_move(engine, board, policy_hce_move, args.eval_time_ms / 1000.0, args.eval_depth)
            policy_value_score = hce_score_after_move(engine, board, policy_value_move, args.eval_time_ms / 1000.0, args.eval_depth)
            result = PositionResult(
                fen=board.fen(),
                hce_move=hce_move.uci(),
                policy_hce_move=policy_hce_move.uci(),
                policy_value_move=policy_value_move.uci(),
                policy_top5=[move.uci() for move in policy_moves[:5]],
                hce_in_policy_top5=hce_move in policy_moves[:5],
                hce_in_policy_topk=hce_move in root_moves,
                policy_hce_matches_hce=policy_hce_move == hce_move,
                policy_value_matches_hce=policy_value_move == hce_move,
                hce_score_cp=hce_score,
                policy_hce_score_cp=policy_hce_score,
                policy_value_score_cp=policy_value_score,
            )
            results.append(result)
            if args.log_every > 0 and idx % args.log_every == 0:
                print(f"[eval] {idx}/{len(positions)}", flush=True)
    finally:
        engine.quit()

    def avg_loss(field: str) -> float | None:
        losses: list[int] = []
        for row in results:
            base = row.hce_score_cp
            other = getattr(row, field)
            if base is not None and other is not None:
                losses.append(base - other)
        if not losses:
            return None
        return sum(losses) / len(losses)

    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {
        "positions": len(results),
        "elapsed_s": time.time() - start,
        "hce_in_policy_top5": sum(r.hce_in_policy_top5 for r in results) / max(1, len(results)),
        "hce_in_policy_topk": sum(r.hce_in_policy_topk for r in results) / max(1, len(results)),
        "policy_hce_matches_hce": sum(r.policy_hce_matches_hce for r in results) / max(1, len(results)),
        "policy_value_matches_hce": sum(r.policy_value_matches_hce for r in results) / max(1, len(results)),
        "avg_policy_hce_loss_cp": avg_loss("policy_hce_score_cp"),
        "avg_policy_value_loss_cp": avg_loss("policy_value_score_cp"),
        "settings": settings,
        "examples": [r.__dict__ for r in results[: args.examples]],
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare policy-assisted search prototypes against current HCE.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "current" / "policy_runs" / "chess_alpha_full_cpu" / "best.pt")
    parser.add_argument("--engine", type=Path, default=ROOT / "bin" / "chess_uci")
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--min-plies", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=46)
    parser.add_argument("--hce-time-ms", type=int, default=50)
    parser.add_argument("--hce-depth", type=int, default=8)
    parser.add_argument("--policy-topk", type=int, default=5)
    parser.add_argument("--policy-hce-mode", choices=("soft", "hard-root"), default="soft")
    parser.add_argument("--policy-root-bonus", type=int, default=30000)
    parser.add_argument("--policy-hce-child-time-ms", type=int, default=15)
    parser.add_argument("--policy-hce-child-depth", type=int, default=6)
    parser.add_argument("--rescore-policy-hce", action="store_true")
    parser.add_argument("--policy-value-depth", type=int, default=2)
    parser.add_argument("--policy-value-breadth", type=int, default=5)
    parser.add_argument("--eval-time-ms", type=int, default=30)
    parser.add_argument("--eval-depth", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--json-out", type=Path, default=ROOT / "current" / "policy_runs" / "policy_search_eval.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = evaluate(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2, default=str))
    print(f"[done] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
