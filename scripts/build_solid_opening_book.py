#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REJECT_TERMS = {
    "Alekhine",
    "Benko",
    "Bird",
    "Budapest",
    "Dutch",
    "Evans",
    "Kings Gambit",
    "Modern Defense",
    "Nimzowitsch",
    "Scandinavian",
    "Wade",
}

WEIGHT_BY_TERM = (
    (("Ruy Lopez", "Italian", "Giuoco", "Four Knights", "Petrov", "Scotch", "Two Knights"), 9),
    (("Caro-Kann", "French", "Sicilian", "QGD", "Slav", "Queens Pawn", "London"), 7),
    (("Catalan", "English", "Reti", "Nimzo Indian", "Queens Indian", "Bogo Indian"), 5),
    (("Kings Indian Defense", "Grunfeld", "QGA"), 4),
    (("Benoni", "Pirc", "Philidor", "Vienna", "Kings Pawn"), 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a conservative weighted opening book from curated seed lines.")
    parser.add_argument("--input", default="data/openings/opening_games_100.txt")
    parser.add_argument("--output", default="data/openings/opening_book.txt")
    parser.add_argument("--include-risky", action="store_true", help="Keep low-weight offbeat/gambit lines instead of dropping them.")
    return parser.parse_args()


def line_weight(name: str, include_risky: bool) -> int:
    if not include_risky and any(term in name for term in DEFAULT_REJECT_TERMS):
        return 0
    for terms, weight in WEIGHT_BY_TERM:
        if any(term in name for term in terms):
            return weight
    return 2 if include_risky else 1


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    kept: list[tuple[str, str, int]] = []
    dropped = 0
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, sequence = line.split("|", 1)
        except ValueError:
            continue
        weight = line_weight(name.strip(), bool(args.include_risky))
        if weight <= 0:
            dropped += 1
            continue
        kept.append((name.strip(), sequence.strip(), weight))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        fp.write("# Conservative weighted opening book generated from opening_games_100.txt\n")
        fp.write("# Format: name|uci-sequence|weight\n")
        for name, sequence, weight in kept:
            fp.write(f"{name}|{sequence}|{weight}\n")

    print(f"wrote {output_path} kept={len(kept)} dropped={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
