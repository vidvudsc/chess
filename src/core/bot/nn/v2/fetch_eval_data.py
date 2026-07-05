#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch the public Lichess Stockfish eval dump.")
    parser.add_argument("--url", default=LICHESS_EVAL_URL, help="Source URL.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/lichess_db_eval.jsonl.zst"))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    parser.add_argument("--check-tools", action="store_true", help="Only check required local tools.")
    return parser


def check_tools() -> None:
    missing = []
    if shutil.which("zstdcat") is None:
        missing.append("zstdcat")
    if missing:
        raise SystemExit(f"missing required tool(s): {', '.join(missing)}")
    print("[ok] required tools available", flush=True)


def download(url: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"[skip] {output} already exists; use --force to overwrite", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    print(f"[fetch] {url}", flush=True)
    print(f"[fetch] -> {output}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fp:
        shutil.copyfileobj(response, fp, length=1024 * 1024)
    tmp.replace(output)
    print(f"[done] wrote {output}", flush=True)


def smoke_zstd(path: Path) -> None:
    if not path.exists():
        return
    proc = subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    line = proc.stdout.readline()
    proc.terminate()
    _, stderr = proc.communicate(timeout=10)
    if not line:
        raise SystemExit(f"zstdcat smoke check failed for {path}: {stderr.decode(errors='replace')[:400]}")


def main() -> int:
    args = build_parser().parse_args()
    check_tools()
    if args.check_tools:
        return 0
    download(args.url, args.output, args.force)
    smoke_zstd(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
