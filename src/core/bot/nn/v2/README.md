# NNUE v2 Lab

This folder is the clean experiment area for the next neural engine. Keep it
separate from the live bot runtime until a model beats `classic` in test-lab.

## Current Pieces

- `data.py`: pure data contracts and labeling helpers
- `build_dataset.py`: converts Lichess Stockfish eval JSONL into v2 rows
- `fetch_eval_data.py`: downloads the public Lichess eval dump
- `train_value.py`: trains and exports a value-only `CHNNUE1` model

The v2 row format stores both the raw label and the search-ready target:

- normalized FEN key
- depth and knodes
- side-to-move target in centipawns
- side-to-move target as bounded value
- mate/terminal flags
- side/phase/eval/tactical bucket

## Local Smoke Test

```bash
make test_nn_v2
```

This test covers:

- side-to-move target flipping
- mate, stalemate, and insufficient-material labels
- depth/knodes filtering
- FEN dedupe keys
- dataset builder CLI
- tiny CPU training
- `CHNNUE1` export
- C UCI model loading

## Data Source

Primary source:

- URL: `https://database.lichess.org/lichess_db_eval.jsonl.zst`
- License: Lichess database exports are CC0
- Format: one JSON position per line
- Useful fields: `fen`, `evals[].depth`, `evals[].knodes`, `evals[].pvs[0].cp`,
  `evals[].pvs[0].mate`, `evals[].pvs[0].line`
- Selection rule: use the highest-depth eval and first PV

Check local tools:

```bash
make nn_v2_check_tools
```

Fetch the eval dump:

```bash
make nn_v2_fetch_eval
```

Hugging Face sources worth using:

- `Lichess/chess-position-evaluations`
  - Best default HF source for value training.
  - Columns: `fen`, `line`, `depth`, `knodes`, `cp`, `mate`.
  - Huge denormalized parquet table; duplicate FENs are normal because multiple
    PV rows can exist for one position.
- `Lichess/fishnet-evals`
  - Simpler `fen`, `cp`, `mate`, `move` schema.
  - Good fallback, but it does not expose depth in the previewed schema.
- `mateuszgrzyb/lichess-stockfish-normalized`
  - Deduplicated and ML-ready with `fen`, `depth`, `cp`, `mate`.
  - Useful, but license is CC-BY-4.0 rather than CC0.
- `avewright/chess-positions-lichess-sf`
  - Has `fen`, `best_move`, `eval_type`, `eval_value`, WDL fields, phase,
    legal-move count, and depth.
  - Interesting later for WDL/policy work.
- `thomas-schweich/pawn-stockfish-100m`
  - Stockfish self-play with per-legal-move evaluations.
  - Useful for future policy distillation, but shards are very large.

Stream/paginate from Hugging Face without downloading the whole dataset:

```bash
make nn_v2_build_dataset ARGS="\
  --input hf://Lichess/chess-position-evaluations/default/train \
  --output current/nnv2_hf_sample.jsonl \
  --summary current/nnv2_hf_sample_summary.json \
  --min-depth 18 \
  --limit 5000 \
  --max-per-bucket 500 \
  --stop-after-accepted 5000 \
  --terminal-fixtures"
```

Run the whole HF smoke pipeline:

```bash
make nn_v2_pipeline ARGS="\
  --input hf://Lichess/chess-position-evaluations/default/train \
  --accepted 5000 \
  --limit 5000 \
  --max-per-bucket 500 \
  --epochs 1 \
  --device cpu"
```

## Build A Small Dataset

```bash
make nn_v2_build_dataset ARGS="\
  --input data/raw/lichess_db_eval.jsonl.zst \
  --output data/labels/nnv2_eval_2m.jsonl \
  --summary current/nnv2_eval_2m_summary.json \
  --min-depth 18 \
  --min-knodes 0 \
  --limit 2000000 \
  --max-per-bucket 100000"
```

Use `--max-per-bucket` to avoid a dataset that is accidentally all quiet equal
openings. For the first serious run, inspect the summary before training.

## Train Locally Or On RunPod

Tiny local:

```bash
make nn_v2_train ARGS="\
  --input data/labels/nnv2_eval_2m.jsonl \
  --epochs 1 \
  --batch-size 512 \
  --feature-dim 64 \
  --hidden-dim 16 \
  --train-samples 50000 \
  --val-samples 10000 \
  --device cpu"
```

First RunPod-sized run:

```bash
make nn_v2_train ARGS="\
  --input data/labels/nnv2_eval_2m.jsonl \
  --epochs 4 \
  --batch-size 1024 \
  --feature-dim 160 \
  --hidden-dim 40 \
  --train-samples 500000 \
  --val-samples 100000 \
  --device cuda"
```

## Serious 20M Run Shape

For serious training, do not use the Hugging Face Dataset Viewer API. It is
for previews and small samples and can rate-limit. Use the raw Lichess eval
dump instead.

Two good options:

1. Stream from the public `.zst` URL and stop after the selected sample is
   written.
2. On a RunPod with enough storage, download the `.zst` once and build multiple
   datasets from the local file.

Download once on RunPod:

```bash
mkdir -p /workspace/data/raw
curl -L --fail --retry 10 --retry-delay 10 \
  -o /workspace/data/raw/lichess_db_eval.jsonl.zst \
  https://database.lichess.org/lichess_db_eval.jsonl.zst
```

Build a broad 20M selected set without keeping rows in RAM:

```bash
make nn_v2_build_dataset ARGS="\
  --input /workspace/data/raw/lichess_db_eval.jsonl.zst \
  --output /workspace/data/labels/nnv2_eval_20m.jsonl \
  --summary /workspace/data/labels/nnv2_eval_20m_summary.json \
  --min-depth 18 \
  --limit 20000000 \
  --max-per-bucket 300000 \
  --sample-every 20 \
  --stream-output \
  --no-dedupe \
  --terminal-fixtures \
  --log-every 1000000"
```

If storage is tight, stream directly instead of downloading:

```bash
make nn_v2_build_dataset ARGS="\
  --input https://database.lichess.org/lichess_db_eval.jsonl.zst \
  --output /workspace/data/labels/nnv2_eval_20m.jsonl \
  --summary /workspace/data/labels/nnv2_eval_20m_summary.json \
  --min-depth 18 \
  --limit 20000000 \
  --max-per-bucket 300000 \
  --sample-every 20 \
  --stream-output \
  --no-dedupe \
  --terminal-fixtures \
  --log-every 1000000"
```

Train from the selected set:

```bash
make nn_v2_train ARGS="\
  --input /workspace/data/labels/nnv2_eval_20m.jsonl \
  --output-dir /workspace/current/nnv2_20m/model \
  --export-path /workspace/current/nnv2_20m/nn_eval.bin \
  --epochs 6 \
  --batch-size 4096 \
  --feature-dim 256 \
  --hidden-dim 64 \
  --train-samples 20000000 \
  --val-samples 500000 \
  --device cuda \
  --log-every 200"
```

Notes:

- `--stream-output` makes the builder scale to millions of rows by writing rows
  directly instead of keeping them in Python lists.
- `--no-dedupe` is recommended for the raw Lichess eval stream because it avoids
  holding a multi-million-FEN set in RAM. Do not use it for denormalized HF
  parquet/Viewer rows unless duplicates are desired.
- `--sample-every 20` scans broadly through the dump instead of taking a narrow
  early slice. For a different budget, use roughly
  `source_rows / desired_rows` as the stride.
- Inspect the summary before training. If it is still phase-skewed, increase
  scanning breadth or split the run into multiple different offsets/strides.

## First Acceptance Criteria

Before spending GPU money:

1. `make test`
2. `make nn_v2_check_tools`
3. Mate, stalemate, and insufficient-material rows are present.
4. No single bucket dominates the written rows.
5. A tiny model can overfit a tiny sample locally.
6. Exported `CHNNUE1` loads in `bin/chess_uci`.
