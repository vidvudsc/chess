#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NN_V2 = ROOT / "src" / "core" / "bot" / "nn" / "v2"
sys.path.insert(0, str(NN_V2))

from train_binpack import (  # noqa: E402
    configure_new_channel_training,
    configure_new_hidden_training,
    remap_native_halfka_threats,
)
from train_value import initialize_from_checkpoint  # noqa: E402
from model import build_value_model  # noqa: E402


def test_native_halfka_remap_and_halfkp_warm_start() -> None:
    native_halfka = 32 * 12 * 64
    combined_dummy = 32 * 11 * 64 + 59808
    raw = torch.tensor([
        0,
        64,
        10 * 64,
        11 * 64,
        12 * 64 + 7,
        native_halfka + 123,
        -1,
    ])
    expected = torch.tensor([
        31 * 11 * 64,
        (31 * 11 + 5) * 64,
        (31 * 11 + 10) * 64,
        (31 * 11 + 10) * 64,
        30 * 11 * 64 + 7,
        32 * 11 * 64 + 123,
        combined_dummy,
    ])
    actual = remap_native_halfka_threats(raw)
    if not torch.equal(actual, expected):
        raise AssertionError(f"native HalfKA remap mismatch: {actual.tolist()}")

    source = build_value_model("linear-head-screlu-hm-buckets", 8, 4)
    target = build_value_model(
        "linear-head-screlu-halfka-threats-hm-buckets-psqt", 8, 4
    )
    initialize_from_checkpoint(target, source.state_dict(), halfka=True)
    first_new_row = 32 * 11 * 64
    if torch.count_nonzero(target.accumulator.weight[first_new_row:]).item() != 0:
        raise AssertionError("warm start must zero every new threat and padding row")
    for king_bucket in range(32):
        start = (king_bucket * 11 + 10) * 64
        if torch.count_nonzero(target.accumulator.weight[start:start + 64]).item() != 0:
            raise AssertionError("warm start must zero new merged-king rows")


def test_wider_threat_model_warm_start_preserves_every_feature_row() -> None:
    torch.manual_seed(7)
    source = build_value_model(
        "linear-head-screlu-halfka-threats-hm-buckets-psqt", 8, 4
    )
    target = build_value_model(
        "linear-head-screlu-halfka-threats-hm-buckets-psqt", 12, 6
    )
    source_state = source.state_dict()
    copied = initialize_from_checkpoint(target, source_state, halfka=True)
    real_rows = source.accumulator.weight.shape[0] - 1
    if not torch.equal(
        target.accumulator.weight[:real_rows, :8],
        source.accumulator.weight[:real_rows],
    ):
        raise AssertionError("wider warm start dropped HalfKAv2 or Full Threats rows")
    if torch.count_nonzero(target.accumulator.weight[-1]).item() != 0:
        raise AssertionError("wider warm start must keep the padding row zero")
    if not any("HalfKAv2 overlap" in name for name in copied):
        raise AssertionError(f"missing threat-model warm-start marker: {copied}")
    white = torch.tensor([0, 511, 22528, 40000], dtype=torch.long)
    black = torch.tensor([63, 1024, 22529, 50000], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    stm_white = torch.tensor([True])
    with torch.no_grad():
        source_value = source(white, offsets, black, offsets, stm_white)
        target_value = target(white, offsets, black, offsets, stm_white)
    torch.testing.assert_close(source_value, target_value, rtol=0.0, atol=1e-7)

    source_width = configure_new_channel_training(target, source_state)
    target.zero_grad(set_to_none=True)
    target(white, offsets, black, offsets, stm_white).sum().backward()
    if torch.count_nonzero(target.accumulator.weight.grad[:, :source_width]).item() != 0:
        raise AssertionError("staged widening changed existing accumulator channels")
    front_old = target.fc1_weight.grad[:, :, :source_width]
    back_old = target.fc1_weight.grad[:, :, target.accumulator.embedding_dim:
                                      target.accumulator.embedding_dim + source_width]
    if torch.count_nonzero(front_old).item() != 0 or torch.count_nonzero(back_old).item() != 0:
        raise AssertionError("staged widening changed existing head inputs")
    front_new = target.fc1_weight.grad[:, :, source_width:target.accumulator.embedding_dim]
    back_new = target.fc1_weight.grad[:, :,
                                      target.accumulator.embedding_dim + source_width:]
    if torch.count_nonzero(front_new).item() == 0 and torch.count_nonzero(back_new).item() == 0:
        raise AssertionError("staged widening did not train any new head input")
    if target.out_weight.grad is not None:
        raise AssertionError("staged widening must freeze the proven output head")


def test_wider_hidden_head_is_exact_and_only_trains_new_units() -> None:
    torch.manual_seed(11)
    source = build_value_model(
        "linear-head-screlu-halfka-threats-hm-buckets-psqt", 8, 4
    )
    target = build_value_model(
        "linear-head-screlu-halfka-threats-hm-buckets-psqt", 8, 7
    )
    source_state = source.state_dict()
    initialize_from_checkpoint(target, source_state, halfka=True)

    white = torch.tensor([0, 511, 22528, 40000], dtype=torch.long)
    black = torch.tensor([63, 1024, 22529, 50000], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    stm_white = torch.tensor([True])
    with torch.no_grad():
        source_value = source(white, offsets, black, offsets, stm_white)
        target_value = target(white, offsets, black, offsets, stm_white)
    torch.testing.assert_close(source_value, target_value, rtol=0.0, atol=1e-7)
    if torch.count_nonzero(target.fc1_weight[:, 4:]).item() == 0:
        raise AssertionError("new hidden units lost their useful random initialization")
    if torch.count_nonzero(target.out_weight[:, 4:]).item() != 0:
        raise AssertionError("new output weights must start at zero for exact warm start")

    source_hidden = configure_new_hidden_training(target, source_state)
    target.zero_grad(set_to_none=True)
    target(white, offsets, black, offsets, stm_white).sum().backward()
    if source_hidden != 4:
        raise AssertionError(f"wrong source hidden width: {source_hidden}")
    if torch.count_nonzero(target.fc1_weight.grad[:, :source_hidden]).item() != 0:
        raise AssertionError("staged head widening changed proven hidden units")
    if torch.count_nonzero(target.out_weight.grad[:, :source_hidden]).item() != 0:
        raise AssertionError("staged head widening changed proven output weights")
    if torch.count_nonzero(target.out_weight.grad[:, source_hidden:]).item() == 0:
        raise AssertionError("new hidden units did not receive an output gradient")
    if target.accumulator.weight.grad is not None:
        raise AssertionError("staged head widening must freeze the feature transformer")


def main() -> int:
    test_native_halfka_remap_and_halfkp_warm_start()
    test_wider_threat_model_warm_start_preserves_every_feature_row()
    test_wider_hidden_head_is_exact_and_only_trains_new_units()
    print("test_nn_binpack_mapping: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
