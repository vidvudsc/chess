#!/usr/bin/env python3
"""Download a deterministic, evenly sampled set of raw Hugging Face binpacks."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--max-file-mb", type=float, default=200.0)
    args = parser.parse_args()

    api = f"https://huggingface.co/api/datasets/{args.repo}/tree/main?recursive=true&expand=false&limit=1000"
    with urllib.request.urlopen(api) as response:
        rows = json.load(response)
    maximum = int(args.max_file_mb * 1_000_000)
    files = sorted(
        (row for row in rows if row.get("type") == "file"
         and row.get("path", "").endswith(".binpack")
         and 0 < int(row.get("size", 0)) <= maximum),
        key=lambda row: row["path"],
    )
    if not files:
        raise SystemExit("no matching raw .binpack files found")
    count = min(max(args.count, 1), len(files))
    if count == 1:
        selected = [files[len(files) // 2]]
    else:
        selected = [files[round(i * (len(files) - 1) / (count - 1))] for i in range(count)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(selected, 1):
        path = row["path"]
        destination = args.output_dir / Path(path).name
        quoted = urllib.parse.quote(path, safe="/")
        url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{quoted}?download=true"
        print(f"[download {index}/{count}] {path} ({int(row['size']) / 1e6:.1f} MB)", flush=True)
        subprocess.run(
            ["curl.exe", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(destination), url],
            check=True,
        )
        if destination.stat().st_size != int(row["size"]):
            raise SystemExit(f"size mismatch for {destination}")
        manifest.append({"path": str(destination), "source": path, "size": int(row["size"])})

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] files={len(manifest)} bytes={sum(row['size'] for row in manifest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
