from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core" / "bot" / "nn"))

from features import MIRRORED_DUMMY_FEATURE_INDEX, encode_fen  # noqa: E402
from model import (  # noqa: E402
    NUM_MATERIAL_BUCKETS,
    build_value_model,
    material_buckets_from_features,
)


def _bucket_for_counts(non_king_counts: list[int]) -> torch.Tensor:
    dummy = MIRRORED_DUMMY_FEATURE_INDEX
    width = max(non_king_counts) + 1
    rows = []
    for count in non_king_counts:
        rows.extend([0] * count + [dummy] * (width - count))
    indices = torch.tensor(rows, dtype=torch.long)
    offsets = torch.arange(0, len(non_king_counts) * width, width, dtype=torch.long)
    return material_buckets_from_features(indices, offsets, dummy)


def test_material_bucket_boundaries() -> None:
    # piece_count = non-king features + 2 kings; bucket = (piece_count - 1) // 4
    buckets = _bucket_for_counts([0, 1, 2, 6, 10, 14, 18, 22, 26, 30])
    expected = torch.tensor([(c + 2 - 1) // 4 for c in [0, 1, 2, 6, 10, 14, 18, 22, 26, 30]])
    expected = expected.clamp(max=NUM_MATERIAL_BUCKETS - 1)
    assert torch.equal(buckets, expected)
    assert buckets.max().item() == NUM_MATERIAL_BUCKETS - 1
    assert buckets.min().item() == 0


def test_bucketed_model_uses_distinct_heads() -> None:
    torch.manual_seed(7)
    model = build_value_model(
        "linear-head-screlu-hm-buckets", accumulator_dim=32, hidden_dim=8
    )
    model.eval()

    # startpos (32 pieces) must land in the last bucket, K vs K in the first
    bucket_full = material_buckets_from_features(
        torch.zeros(30, dtype=torch.long), torch.tensor([0]), MIRRORED_DUMMY_FEATURE_INDEX
    )
    bucket_bare = material_buckets_from_features(
        torch.tensor([MIRRORED_DUMMY_FEATURE_INDEX]),
        torch.tensor([0]),
        MIRRORED_DUMMY_FEATURE_INDEX,
    )
    assert bucket_full.item() == NUM_MATERIAL_BUCKETS - 1
    assert bucket_bare.item() == 0

    # swapping one bucket's head weights must change only that bucket's output
    x_indices = torch.zeros(30, dtype=torch.long)
    x_offsets = torch.tensor([0], dtype=torch.long)
    stm_t = torch.tensor([True])
    with torch.no_grad():
        before = model(x_indices, x_offsets, x_indices, x_offsets, stm_t).item()
        model.out_bias[0] += 0.5  # bucket 0 head; startpos uses bucket 7
        unchanged = model(x_indices, x_offsets, x_indices, x_offsets, stm_t).item()
        model.out_bias[NUM_MATERIAL_BUCKETS - 1] += 0.5
        changed = model(x_indices, x_offsets, x_indices, x_offsets, stm_t).item()
    assert unchanged == before
    assert changed != before


def test_psqt_head_uses_front_perspective_and_material_bucket() -> None:
    model = build_value_model(
        "linear-head-screlu-hm-buckets-psqt", accumulator_dim=16, hidden_dim=8
    )
    model.eval()
    with torch.no_grad():
        model.accumulator.weight.zero_()
        model.fc1_weight.zero_()
        model.fc1_bias.zero_()
        model.out_weight.zero_()
        model.out_bias.zero_()
        model.psqt.weight.zero_()
        model.psqt.weight[0, NUM_MATERIAL_BUCKETS - 1] = 0.25
        model.psqt.weight[1, NUM_MATERIAL_BUCKETS - 1] = -0.5

    white = torch.zeros(30, dtype=torch.long)
    black = torch.ones(30, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    white_stm = model(white, offsets, black, offsets, torch.tensor([True])).item()
    black_stm = model(white, offsets, black, offsets, torch.tensor([False])).item()
    assert white_stm > 0.99
    assert black_stm < -0.99
