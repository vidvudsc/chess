#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".h", ".py", ".sh", ".md"}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    lines: int


def git_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return [
            p.relative_to(ROOT)
            for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ]
    return [Path(line) for line in out.splitlines() if line.strip()]


def source_files(paths: Iterable[Path]) -> list[SourceFile]:
    files: list[SourceFile] = []
    for rel in paths:
        if rel.suffix not in SOURCE_SUFFIXES:
            continue
        path = ROOT / rel
        try:
            lines = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            continue
        files.append(SourceFile(rel, lines))
    return sorted(files, key=lambda item: (item.path.parts, str(item.path)))


def area_for(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("src/core/engine/"):
        return "engine"
    if text.startswith("src/app/"):
        return "app"
    if text.startswith("src/core/bot/nn/"):
        return "nn-training"
    if text.startswith("src/core/bot/"):
        return "bot"
    if text.startswith("tests/"):
        return "tests"
    if text.startswith("scripts/"):
        return "scripts"
    if text.startswith("docs/"):
        return "docs"
    return "other"


def print_area_summary(files: list[SourceFile]) -> None:
    totals: dict[str, tuple[int, int]] = {}
    for item in files:
        count, lines = totals.get(area_for(item.path), (0, 0))
        totals[area_for(item.path)] = (count + 1, lines + item.lines)

    print("Area summary")
    for area, (count, lines) in sorted(totals.items()):
        print(f"  {area:12} {count:3d} files {lines:6d} lines")


def print_hotspots(files: list[SourceFile], warn_lines: int) -> None:
    print("\nLargest files")
    for item in sorted(files, key=lambda src: src.lines, reverse=True)[:12]:
        marker = "  WARN" if item.lines >= warn_lines else "      "
        print(f"{marker} {item.lines:5d} {item.path}")


def quoted_includes(path: Path) -> list[str]:
    includes: list[str] = []
    try:
        lines = (ROOT / path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return includes
    for raw in lines:
        line = raw.strip()
        if not line.startswith("#include \""):
            continue
        parts = line.split('"')
        if len(parts) >= 2:
            includes.append(parts[1])
    return includes


def boundary_violations(files: list[SourceFile]) -> list[str]:
    violations: list[str] = []
    for item in files:
        text = item.path.as_posix()
        includes = quoted_includes(item.path)

        if text.startswith("src/core/engine/"):
            for inc in includes:
                if inc.startswith("../app/") or inc in {
                    "ui.h",
                    "assets.h",
                    "game_log.h",
                    "ai_test_runner.h",
                    "chess_ai_worker.h",
                    "stockfish_eval_worker.h",
                }:
                    violations.append(f"{item.path}: engine includes app header {inc}")

        if text == "src/core/bot/run.py":
            try:
                body = (ROOT / item.path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                body = ""
            if "import torch" in body or "from torch" in body:
                violations.append(f"{item.path}: live bot runtime imports torch")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Report codebase architecture hotspots and boundary violations.")
    parser.add_argument("--warn-lines", type=int, default=1000, help="Line count that marks a file as a size hotspot.")
    args = parser.parse_args()

    files = source_files(git_files())
    print_area_summary(files)
    print_hotspots(files, warn_lines=args.warn_lines)

    violations = boundary_violations(files)
    if violations:
        print("\nBoundary violations")
        for violation in violations:
            print(f"  FAIL {violation}")
        return 1

    print("\nBoundary violations: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
