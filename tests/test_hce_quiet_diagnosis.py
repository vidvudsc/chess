#!/usr/bin/env python3
"""Check diagnostic sample selection and handling of uncertain teacher labels."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_hce_quiets import judge, load_cases, summarize


class QuietDiagnosisTests(unittest.TestCase):
    def test_teacher_compares_every_move_at_the_same_root_and_depth(self):
        board = chess.Board()
        moves = [chess.Move.from_uci(m) for m in ("d2d4", "e2e4", "g1f3")]
        engine = Mock()
        engine.analyse.side_effect = [
            {"pv": [moves[2]]},
            [{"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(10), board.turn),
              "depth": 14} for move in moves],
        ]
        scores = judge(engine, board, ["e2e4", "d2d4"], 14)
        self.assertEqual(set(scores), {m.uci() for m in moves})
        for call in engine.analyse.call_args_list:
            self.assertIs(call.args[0], board)
            self.assertEqual(call.args[1].depth, 14)
        comparison = engine.analyse.call_args_list[1]
        self.assertEqual(set(comparison.kwargs["root_moves"]), set(moves))
        self.assertEqual(comparison.kwargs["multipv"], 3)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_filter_deduplicates_and_excludes_captures_promotions_and_mates(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        quiet = dict(fen=fen, played="e2e4", best="d2d4", loss_cp=80,
                     played_cp=0, best_cp=80)
        capture = dict(quiet, fen="4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
                       played="e4d5", best="e4e5")
        promotion = dict(quiet, fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                         played="a7a8q", best="e1d1")
        rows = [dict(quiet, loss_cp=20), dict(quiet, best_cp=30000),
                capture, promotion, dict(quiet, best="e2e4"), quiet, quiet]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(load_cases(path, 0, 1), [quiet])

    def test_summary_excludes_mate_horizons_and_rechecks_historical_errors(self):
        def row(played, base, variant, mate=None):
            return {"played_regret_cp": played,
                    "teacher": {"e2e4": {"mate": mate, "cp": 0}},
                    "searches": {
                        "baseline_long": {"regret_cp": base, "depth": 14},
                        "variant": {"regret_cp": variant, "depth": 12}}}
        rows = [row(100, 80, 20), row(20, 0, 40), row(30000, 30000, 0, 4)]
        result = summarize(rows, ["baseline_long", "variant"])
        self.assertEqual(result["mate_horizon_cases"], 1)
        self.assertEqual(result["all_nonmate"]["positions"], 2)
        subset = result["reconfirmed_errors"]
        self.assertEqual(subset["positions"], 1)
        variant = subset["arms"]["variant"]
        self.assertEqual(variant["mean_regret_cp"], 20)
        self.assertEqual(variant["within_30cp"], 1)
        self.assertEqual(variant["better_than_long_by_30cp"], 1)
        self.assertEqual(variant["worse_than_long_by_30cp"], 0)


if __name__ == "__main__":
    unittest.main()
