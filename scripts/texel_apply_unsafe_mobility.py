#!/usr/bin/env python3
"""Apply an UNSAFE_MOBILITY line to the zero-default HCE corrections."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from texel_tune_unsafe_mobility import N_PARAMS, PARAMETER_NAMES


def parse_line(line: str) -> list[int]:
    parts = line.strip().split()
    if parts and parts[0] == "UNSAFE_MOBILITY":
        parts = parts[1:]
    values = [int(part) for part in parts]
    if len(values) != N_PARAMS:
        raise SystemExit(
            f"expected {N_PARAMS} unsafe-mobility integers, got {len(values)}"
        )
    return values


def patch_eval_c(path: Path, values: list[int]) -> None:
    text = path.read_text(encoding="utf-8")
    for parameter, value in zip(PARAMETER_NAMES, values):
        constant = f"k_{parameter}"
        text, replacements = re.subn(
            rf"static const int {re.escape(constant)} = -?\d+;",
            f"static const int {constant} = {value};",
            text,
            count=1,
        )
        if replacements != 1:
            raise SystemExit(f"failed to patch {constant} in {path}")
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-c",
        type=Path,
        default=Path("src/core/engine/hce_eval.c"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--unsafe-line")
    source.add_argument("--unsafe-file", type=Path)
    args = parser.parse_args()

    if args.unsafe_line:
        line = args.unsafe_line
    else:
        lines = [
            candidate
            for candidate in args.unsafe_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if candidate.strip().startswith("UNSAFE_MOBILITY ")
        ]
        if not lines:
            raise SystemExit(
                f"no UNSAFE_MOBILITY line found in {args.unsafe_file}"
            )
        line = lines[-1]
    patch_eval_c(args.eval_c, parse_line(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
