#!/usr/bin/env python3
"""Stream a zstd file to disk without holding the archive in memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import zstandard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.source.open("rb") as source, args.destination.open("wb") as destination:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            while chunk := reader.read(max(1, args.chunk_mb) * 1024 * 1024):
                destination.write(chunk)
                written += len(chunk)
                if written % (1024 * 1024 * 1024) < len(chunk):
                    print(f"[decompress] {written / 1e9:.1f} GB", flush=True)
    print(f"[done] bytes={written} destination={args.destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
