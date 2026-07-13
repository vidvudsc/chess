#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chess
import chess.engine
import torch

ROOT = Path(__file__).resolve().parents[1]
NN_ROOT = ROOT / "src" / "core" / "bot" / "nn"
if str(NN_ROOT) not in sys.path:
    sys.path.insert(0, str(NN_ROOT))

from policy_infer import load_model, rank_legal_moves  # noqa: E402


class PolicyHceUci:
    def __init__(self, engine_path: Path, checkpoint: Path, device: str = "cpu") -> None:
        self.engine_path = engine_path
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.model = load_model(checkpoint, self.device)
        self.engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        self.board = chess.Board()
        self.movetime_ms = int(os.environ.get("POLICY_HCE_MOVETIME_MS", "350"))
        self.max_depth = int(os.environ.get("POLICY_HCE_MAX_DEPTH", "16"))
        self.policy_topk = int(os.environ.get("POLICY_HCE_TOPK", "10"))
        self.policy_bonus = int(os.environ.get("POLICY_HCE_BONUS", "30000"))
        self.child_options: dict[str, object] = {}

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass

    def print_uci(self) -> None:
        print("id name Chess Policy-HCE Wrapper", flush=True)
        print("id author Codex + vidvudscalitis", flush=True)
        print("option name MoveTime type spin default 350 min 1 max 10000", flush=True)
        print("option name MaxDepth type spin default 16 min 1 max 32", flush=True)
        print("option name PolicyTopK type spin default 10 min 0 max 64", flush=True)
        print("option name PolicyRootBonus type spin default 30000 min 0 max 1000000", flush=True)
        print("option name Backend type combo default classic var classic var nn var experimental", flush=True)
        print("option name NNModel type string default auto", flush=True)
        print("option name BookFile type string default auto", flush=True)
        print("uciok", flush=True)

    def setoption(self, line: str) -> None:
        rest = line[len("setoption"):].strip()
        if not rest.startswith("name "):
            return
        rest = rest[5:]
        if " value " in rest:
            name, value = rest.split(" value ", 1)
            value = value.strip()
        else:
            name, value = rest, ""
        name = name.strip()
        low = name.lower()
        if low == "movetime":
            self.movetime_ms = max(1, int(value))
            return
        if low == "maxdepth":
            self.max_depth = max(1, int(value))
            return
        if low == "policytopk":
            self.policy_topk = max(0, min(64, int(value)))
            return
        if low == "policyrootbonus":
            self.policy_bonus = max(0, int(value))
            return
        child_name = {
            "backend": "Backend",
            "nnmodel": "NNModel",
            "bookfile": "BookFile",
            "openingbook": "BookFile",
        }.get(low)
        if child_name is not None:
            self.child_options[child_name] = value
            self.engine.configure({child_name: value})

    def position(self, line: str) -> None:
        tokens = line.split()
        if len(tokens) < 2:
            return
        idx = 1
        if tokens[idx] == "startpos":
            board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            fen_parts: list[str] = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            board = chess.Board(" ".join(fen_parts))
        else:
            return
        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            for raw in tokens[idx:]:
                board.push(chess.Move.from_uci(raw))
        self.board = board

    def parse_go(self, line: str) -> chess.engine.Limit:
        tokens = line.split()
        movetime_ms = self.movetime_ms
        depth = self.max_depth
        wtime = btime = -1
        winc = binc = 0
        idx = 1
        while idx < len(tokens):
            tok = tokens[idx]
            val = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if tok == "movetime":
                movetime_ms = max(1, int(val))
                idx += 2
            elif tok == "depth":
                depth = max(1, int(val))
                idx += 2
            elif tok == "wtime":
                wtime = int(val)
                idx += 2
            elif tok == "btime":
                btime = int(val)
                idx += 2
            elif tok == "winc":
                winc = int(val)
                idx += 2
            elif tok == "binc":
                binc = int(val)
                idx += 2
            elif tok == "infinite":
                movetime_ms = 2000
                idx += 1
            else:
                idx += 1
        if "movetime" not in tokens and (wtime > 0 or btime > 0):
            remaining = wtime if self.board.turn == chess.WHITE else btime
            inc = winc if self.board.turn == chess.WHITE else binc
            if remaining > 0:
                movetime_ms = max(20, min(5000, remaining // 25 + inc // 2))
        return chess.engine.Limit(time=max(0.001, movetime_ms / 1000.0), depth=depth)

    def go(self, line: str) -> None:
        limit = self.parse_go(line)
        hints = ""
        if self.policy_topk > 0 and self.policy_bonus > 0 and not self.board.is_game_over(claim_draw=True):
            ranked, _ = rank_legal_moves(self.model, self.board, self.device, limit=self.policy_topk)
            hints = " ".join(move.uci() for move, _ in ranked)
        self.engine.configure({
            "PolicyRootHints": hints,
            "PolicyRootBonus": self.policy_bonus if hints else 0,
        })
        try:
            result = self.engine.play(self.board, limit)
        except Exception as exc:
            print(f"info string policy-hce child error: {exc}", flush=True)
            print("bestmove 0000", flush=True)
            return
        if result.move is None:
            print("bestmove 0000", flush=True)
            return
        print(f"bestmove {result.move.uci()}", flush=True)

    def loop(self) -> None:
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                if line == "uci":
                    self.print_uci()
                elif line == "isready":
                    print("readyok", flush=True)
                elif line.startswith("setoption "):
                    self.setoption(line)
                elif line == "ucinewgame":
                    self.board = chess.Board()
                    try:
                        self.engine.ucinewgame()
                    except Exception:
                        pass
                elif line.startswith("position "):
                    self.position(line)
                elif line.startswith("go"):
                    self.go(line)
                elif line == "stop":
                    continue
                elif line == "quit":
                    return
        finally:
            self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="UCI wrapper that injects NN policy root hints into the HCE engine.")
    parser.add_argument("--engine", type=Path, default=ROOT / "bin" / "chess_uci")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "current" / "policy_runs" / "chess_alpha_64x3_mps_4ep" / "best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    PolicyHceUci(args.engine, args.checkpoint, args.device).loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
