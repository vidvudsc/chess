from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core" / "bot" / "nn"))

from features import encode_fen, encode_fen_halfka, encode_fen_slow  # noqa: E402
from v2.train_binpack import mirror_halfkp_indices  # noqa: E402
from v2.train_value import mirror_halfkp_indices as mirror_training_indices  # noqa: E402


FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
    "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - -",
    "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -",
    "8/5pk1/6p1/8/8/5PP1/6KP/8 w - -",
    "7k/6Q1/6K1/8/8/8/8/8 b - -",
    "4k3/8/8/8/8/8/8/4K3 w - -",
]


def test_fast_halfkp_matches_python_chess_encoder() -> None:
    for fen in FENS:
        fast_white, fast_black, fast_stm = encode_fen(fen)
        slow_white, slow_black, slow_stm = encode_fen_slow(fen)
        if sorted(fast_white) != sorted(slow_white):
            raise AssertionError(f"white features differ for {fen}: {fast_white} vs {slow_white}")
        if sorted(fast_black) != sorted(slow_black):
            raise AssertionError(f"black features differ for {fen}: {fast_black} vs {slow_black}")
        if fast_stm != slow_stm:
            raise AssertionError(f"side to move differs for {fen}: {fast_stm} vs {slow_stm}")


def test_fast_halfkp_rejects_missing_king() -> None:
    try:
        encode_fen("8/8/8/8/8/8/8/4K3 w - -")
    except ValueError:
        return
    raise AssertionError("missing black king should be rejected")


def test_mirrored_halfkp_collapses_horizontal_symmetry() -> None:
    def legacy_index(king: int, plane: int, square: int) -> int:
        return (king * 10 + plane) * 64 + square

    left = legacy_index(26, 3, 10)
    right = legacy_index(26 ^ 7, 3, 10 ^ 7)
    mapped = mirror_halfkp_indices(torch.tensor([left, right, -1], dtype=torch.long))
    if mapped[0].item() != mapped[1].item():
        raise AssertionError(f"horizontal mirror did not collapse: {mapped.tolist()}")
    if not (0 <= mapped[0].item() < 20480):
        raise AssertionError(f"mirrored feature out of range: {mapped[0].item()}")
    if mapped[2].item() != 20480:
        raise AssertionError(f"mirrored dummy index mismatch: {mapped[2].item()}")


def test_halfka_adds_both_kings_and_mirrors_in_range() -> None:
    fen = "4k3/8/8/8/8/8/8/4K3 w - -"
    white, black, stm = encode_fen_halfka(fen)
    if not stm or len(white) != 2 or len(black) != 2:
        raise AssertionError(f"HalfKAv2 must encode both kings per perspective: {white}, {black}")
    mapped = mirror_training_indices(
        torch.tensor(white + black, dtype=torch.long),
        planes=11,
        source_dummy=45056,
        target_dummy=22528,
    )
    if not torch.all((mapped >= 0) & (mapped < 22528)):
        raise AssertionError(f"mirrored HalfKAv2 feature out of range: {mapped.tolist()}")


def main() -> None:
    test_fast_halfkp_matches_python_chess_encoder()
    test_fast_halfkp_rejects_missing_king()
    print("test_nn_features: OK")


if __name__ == "__main__":
    main()
