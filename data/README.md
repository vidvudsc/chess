# Data Layout

This folder stores engine test data, opening-book data, and shared analysis inputs.

- `data/pgn/`
  - source PGN dumps (for labeling / position extraction)
- `data/labels/`
  - labeled CSV datasets and analysis outputs
- `data/positions/`
  - FEN position sets (AI test lab / benchmarks)
- `data/openings/`
  - runtime opening book files (`opening_book.txt`) and notes
- `data/cache/`
  - cached preprocessing outputs (`.npz`)
- `data/misc/`
  - imported or one-off datasets

Current files:
- `data/pgn/lichess_db_standard_rated_2016-02.pgn.zst`
- `data/labels/labels_2016_02_500k_60ms.csv`
- `data/positions/lichess_equal_positions.fen`
- `data/openings/README.md`
- `data/openings/opening_games_100.txt`
- `data/cache/halfkp_500k_s600.npz`

Temporary/active outputs may still be kept at `data/` root while a long labeling run is in progress (for safe resume), e.g. `data/labels_eval_quality.csv`.

Binary engine/runtime artifacts are stored outside `data/`.
