# Opening Book Data

Engine runtime opening book file:

- `data/openings/opening_book.txt`
- `data/openings/opening_games_100.txt` (100 curated opening game seeds)

Format:

- `history|move|weight`
- `history` is space-separated UCI history from start position
- use `-` for root history (empty)

Example:

```text
-|e2e4|120
e2e4|e7e5|83
e2e4 e7e5|g1f3|71
```

Build from PGN/ECO:

```bash
Opening book generation scripts are no longer part of the repo.
The runtime book file is `data/openings/opening_book.txt`.
```

Build from the curated 100-game seed set:

```bash
If you want to reseed the book from custom games, generate a new `opening_book.txt`
outside the repo and drop it into `data/openings/opening_book.txt`.
```

Behavior note:

- If `opening_book.txt` is loaded, the engine uses that external book as the opening source.
- Built-in fallback lines are only used when no external book is loaded.
