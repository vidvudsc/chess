#!/usr/bin/env python3
"""Verify that pairwise weights reconstruct the candidate HCE exactly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import texel_tune
import tune_hce_pairwise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--initial-tuned-file", type=Path, required=True)
    args = parser.parse_args()

    raw = np.atleast_2d(np.loadtxt(args.features, dtype=np.float32))
    with args.metadata.open("r", encoding="utf-8", newline="") as source:
        metadata = list(csv.DictReader(source, delimiter="\t"))
    if len(raw) != len(metadata):
        raise ValueError("feature and metadata row counts differ")

    side_size = (raw.shape[1] - 4) // 2
    if raw.shape[1] != 4 + 2 * side_size:
        raise ValueError("expected tunedumpall rows with qsearch delta")
    base_side = tune_hce_pairwise.BASE_SIDE
    if side_size < base_side + tune_hce_pairwise.EXTENDED_FEATURES:
        raise ValueError("feature dump does not contain all pairwise features")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    weights = report["weights"]
    baseline_weights = texel_tune.load_tuned_defaults(args.initial_tuned_file)
    phase = raw[:, 1].astype(np.int64)
    eval_true = raw[:, 2].astype(np.int64)
    white = raw[:, 3:3 + side_size]
    black_start = 3 + side_size
    black = raw[:, black_start:black_start + side_size]

    totals: list[np.ndarray] = []
    for side in (white, black):
        base_mg, base_eg = texel_tune.side_mg_eg_int(
            side[:, :base_side], baseline_weights
        )
        correction_mg = np.zeros(len(side), dtype=np.int64)
        correction_eg = np.zeros(len(side), dtype=np.int64)
        for name, column in tune_hce_pairwise.FEATURES:
            count = side[:, column].astype(np.int64)
            correction_mg += count * int(weights[name]["mg"])
            correction_eg += count * int(weights[name]["eg"])
        totals.append(texel_tune.trunc_div24(
            (base_mg + correction_mg) * phase +
            (base_eg + correction_eg) * (24 - phase)
        ))

    white_eval = totals[0] - totals[1]
    child_sign = np.array(
        [-1 if row["root_side"] == "w" else 1 for row in metadata],
        dtype=np.int64,
    )
    reconstructed = white_eval * child_sign + 12
    bad = np.flatnonzero(reconstructed != eval_true)
    if len(bad):
        sample = ", ".join(
            f"row {index}: engine={eval_true[index]} recon={reconstructed[index]}"
            for index in bad[:5]
        )
        raise SystemExit(f"reconstruction failed on {len(bad)}/{len(raw)}: {sample}")
    print(f"reconstruction exact on {len(raw)}/{len(raw)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
