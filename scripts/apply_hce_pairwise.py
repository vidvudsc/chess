#!/usr/bin/env python3
"""Apply pairwise-tuned handcrafted weights to the pure HCE evaluator."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def patch_eval(eval_path: Path, report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights = report.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("report has no pairwise weights")

    source = eval_path.read_text(encoding="utf-8")
    for name, phases in weights.items():
        if not isinstance(phases, dict):
            raise ValueError(f"invalid phases for {name}")
        for phase in ("mg", "eg"):
            value = int(phases[phase])
            constant = f"k_{name}_{phase}"
            pattern = re.compile(
                rf"(static const int {re.escape(constant)} = )-?\d+(;)"
            )
            source, count = pattern.subn(rf"\g<1>{value}\g<2>", source)
            if count != 1:
                raise ValueError(
                    f"expected one definition for {constant}, found {count}"
                )

    eval_path.write_text(source, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eval-c", type=Path, required=True)
    args = parser.parse_args()
    patch_eval(args.eval_c, args.report)
    print(f"patched {args.eval_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
