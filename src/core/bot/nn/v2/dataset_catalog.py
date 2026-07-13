#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    priority: int
    source: str
    hf_repo: str
    local_dir: str
    include: tuple[str, ...]
    format: str
    role: str
    notes: str


CATALOG: tuple[DatasetEntry, ...] = (
    DatasetEntry(
        key="stockfish-test80-2024",
        priority=1,
        source="linrock/test80-2024",
        hf_repo="linrock/test80-2024",
        local_dir="/workspace/data/nnue/binpack/test80-2024",
        include=("*.binpack", "**/*.binpack"),
        format="binpack",
        role="primary-value",
        notes="Best public Stockfish-style NNUE source found: Leela T80 data converted to NNUE binpacks.",
    ),
    DatasetEntry(
        key="stockfish-test80-2023",
        priority=2,
        source="linrock/test80-2023",
        hf_repo="linrock/test80-2023",
        local_dir="/workspace/data/nnue/binpack/test80-2023",
        include=("*.binpack", "**/*.binpack"),
        format="binpack",
        role="primary-value",
        notes="Strong older Stockfish/Leela binpacks; useful for mixing and regression against 2024-only data.",
    ),
    DatasetEntry(
        key="stockfish-bullet-training-data",
        priority=3,
        source="linrock/bullet-training-data",
        hf_repo="linrock/bullet-training-data",
        local_dir="/workspace/data/nnue/bullet/linrock-bullet",
        include=("*",),
        format="bullet/binpack",
        role="primary-value",
        notes="Bullet-oriented public training data; good if we move the trainer to Bullet or add a converter.",
    ),
    DatasetEntry(
        key="lichess-stockfish-normalized",
        priority=4,
        source="mateuszgrzyb/lichess-stockfish-normalized",
        hf_repo="mateuszgrzyb/lichess-stockfish-normalized",
        local_dir="/workspace/data/hf/lichess-normalized",
        include=("*.parquet", "**/*.parquet"),
        format="parquet",
        role="fallback-value",
        notes="316M deduplicated Stockfish eval rows. Easy for this repo, but needs quiet filtering and engine validation.",
    ),
    DatasetEntry(
        key="lichess-position-evaluations",
        priority=5,
        source="Lichess/chess-position-evaluations",
        hf_repo="Lichess/chess-position-evaluations",
        local_dir="/workspace/data/hf/lichess-position-evals",
        include=("default/train/*.parquet", "**/*.parquet"),
        format="parquet",
        role="rich-value-pv",
        notes="Richer Lichess eval source with PV/line context; heavier and needs cleaning/dedupe.",
    ),
    DatasetEntry(
        key="deepmind-chessbench",
        priority=6,
        source="google-deepmind/searchless_chess",
        hf_repo="",
        local_dir="/workspace/data/chessbench",
        include=(),
        format="custom",
        role="policy-search",
        notes="Use for policy or root move ordering, not the first NNUE value replacement.",
    ),
)


def hf_download_command(entry: DatasetEntry) -> str:
    if not entry.hf_repo:
        return f"# {entry.key}: no direct Hugging Face dataset repo; follow {entry.source}"
    parts = [
        "hf",
        "download",
        entry.hf_repo,
        "--repo-type",
        "dataset",
        "--local-dir",
        entry.local_dir,
        "--max-workers",
        "16",
    ]
    for pattern in entry.include:
        parts.extend(["--include", pattern])
    return " ".join(shlex.quote(part) for part in parts)


def emit_manifest(path: Path) -> None:
    payload = {
        "purpose": "NNUE dataset acquisition order for the Chess engine.",
        "best_first": [entry.key for entry in sorted(CATALOG, key=lambda item: item.priority)],
        "datasets": [asdict(entry) for entry in sorted(CATALOG, key=lambda item: item.priority)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def emit_download_script(path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Install first if needed: pip install -U huggingface_hub[hf_transfer]",
        "# Optional faster transfers: export HF_HUB_ENABLE_HF_TRANSFER=1",
        "",
    ]
    for entry in sorted(CATALOG, key=lambda item: item.priority):
        lines.append(f"# {entry.priority}. {entry.key}: {entry.notes}")
        lines.append(f"mkdir -p {shlex.quote(entry.local_dir)}")
        lines.append(hf_download_command(entry))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit the recommended NNUE dataset manifest and download commands.")
    parser.add_argument("--manifest", type=Path, default=Path("current/nnue_dataset_catalog.json"))
    parser.add_argument("--download-script", type=Path, default=Path("current/download_nnue_datasets.sh"))
    parser.add_argument("--print", action="store_true", help="Print the catalog and commands to stdout.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    emit_manifest(args.manifest)
    emit_download_script(args.download_script)
    if args.print:
        for entry in sorted(CATALOG, key=lambda item: item.priority):
            print(f"{entry.priority}. {entry.key} [{entry.format}] {entry.role}")
            print(f"   {entry.notes}")
            print(f"   {hf_download_command(entry)}")
    print(f"[done] wrote {args.manifest} and {args.download_script}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
