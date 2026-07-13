# NNUE v2 Lab

This folder is the clean experiment area for the next neural engine. Keep it
separate from the live bot runtime until a model beats `classic` in test-lab.

## Current Pieces

- `data.py`: pure data contracts and labeling helpers
- `build_dataset.py`: converts Lichess Stockfish eval JSONL into v2 rows
- `build_feature_dataset.py`: converts eval JSONL directly into HalfKP feature shards
- `clean_hf_evals.py`: cleans denormalized HF/Lichess parquet rows into v2 rows or HalfKP shards
- `precompute_features.py`: converts existing v2 row JSONL into HalfKP feature shards
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

## Best Public Dataset Order

Generate the local manifest and download script:

```bash
make nnue_dataset_catalog ARGS="\
  --manifest current/nnue_dataset_catalog.json \
  --download-script current/download_nnue_datasets.sh \
  --print"
```

The current best-first order is:

1. `linrock/test80-2024` Stockfish/Leela NNUE binpacks
2. `linrock/test80-2023` Stockfish/Leela NNUE binpacks
3. `linrock/bullet-training-data` if we move conversion/training to Bullet
4. `mateuszgrzyb/lichess-stockfish-normalized` as the fast local parquet fallback
5. `Lichess/chess-position-evaluations` when PV context matters
6. `google-deepmind/searchless_chess` for policy/search help, not first value NNUE

The binpack datasets are the strongest public trail for NNUE value training.
`train_binpack.py` now consumes them directly through Stockfish's official
`nnue-pytorch` loader. Apply `nnue_pytorch_legacy_halfkp.patch` to that loader
to expose the engine's exact 40,960-input HalfKP feature layout. The trainer
uses both the search score and game outcome instead of CP-only regression.

Example direct-binpack run:

```bash
git clone --depth 1 https://github.com/official-stockfish/nnue-pytorch.git /workspace/nnue-pytorch
git -C /workspace/nnue-pytorch apply \
  "$PWD/src/core/bot/nn/v2/nnue_pytorch_legacy_halfkp.patch"
cmake -S /workspace/nnue-pytorch/data_loader/cpp \
  -B /workspace/nnue-pytorch/build -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/nnue-pytorch/build --target training_data_loader -j

python src/core/bot/nn/v2/train_binpack.py \
  --input /workspace/data/train.binpack \
  --nnue-pytorch-dir /workspace/nnue-pytorch \
  --output-dir current/test80_halfkp \
  --arch linear-head-screlu \
  --feature-dim 64 \
  --hidden-dim 16 \
  --batch-size 8192 \
  --batches 2000 \
  --score-lambda 0.70
```

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
  - Best default HF source for cleaned value training with PV context.
  - Columns: `fen`, `line`, `depth`, `knodes`, `cp`, `mate`.
  - Huge denormalized parquet table; duplicate FENs are normal because multiple
    PV rows can exist for one position.
  - Current HF parquet conversion has 20 shards and is about 41 GB total.
- `Lichess/fishnet-evals`
  - Simpler `fen`, `cp`, `mate`, `move` schema.
  - Good fallback, but it does not expose depth in the previewed schema.
- `mateuszgrzyb/lichess-stockfish-normalized`
  - Deduplicated and ML-ready with `fen`, `depth`, `cp`, `mate`.
  - Best fast value-only baseline because it is about 6.6 GB for 316M unique
    positions.
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

Clean a tiny local or downloaded HF parquet sample:

```bash
make nn_v2_clean_hf_evals ARGS="\
  --input /workspace/data/hf/lichess-position-evals/default/train/0000.parquet \
  --output-jsonl /workspace/data/labels/nnv2_clean_smoke.jsonl \
  --output-features-dir /workspace/data/features/nnv2_clean_smoke \
  --summary /workspace/data/features/nnv2_clean_smoke_summary.json \
  --min-depth 20 \
  --min-knodes 100 \
  --candidate-limit 100000 \
  --limit 80000 \
  --fast \
  --terminal-fixtures \
  --rows-per-shard 50000"
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

## Fast Normalized Value Baseline

For the first serious NNUE value baseline, use the normalized dataset. We are
not training a policy head yet, so losing PV/best-move data is acceptable for
this run.

Download once on RunPod:

```bash
mkdir -p /workspace/data/hf/lichess-normalized
hf download mateuszgrzyb/lichess-stockfish-normalized \
  --repo-type dataset \
  --include "*.parquet" \
  --local-dir /workspace/data/hf/lichess-normalized \
  --max-workers 16
```

The selected serious NNUE lane is the fixed/clipped bottleneck model:

- architecture: `bottleneck-head-crelu`
- features: HalfKP
- accumulator: 256 per perspective
- head: 512 clipped accumulator inputs -> 32 bottleneck -> 32 hidden -> value
- export format: `CHNNUE1` version 6
- target scale: `cp_scale=400`

This is the current answer to the size/speed question: train wider than the old
64x16 model, but only in the clipped quantized format that the C engine can run
with integer/SIMD inference. Use `--train-samples 0` for one full pass over all
rows that pass the split/filter.

```bash
make nnue_train_clipped256 ARGS="\
  --input-parquet '/workspace/data/hf/lichess-normalized/**/*.parquet' \
  --output-dir /workspace/current/nnue_clipped256_full/model \
  --export-path /workspace/current/nnue_clipped256_full/nn_eval.bin \
  --epochs 1 \
  --batch-size 16384 \
  --min-depth 18 \
  --train-samples 0 \
  --val-samples 500000 \
  --val-buckets 10 \
  --terminal-fixtures \
  --num-workers 4 \
  --pin-memory \
  --parquet-batch-size 262144 \
  --device cuda \
  --log-every 500"
```

If we want repeated training passes over a fixed 50M/100M sample, then build
feature shards. Use uncompressed shards when wall-clock matters more than disk.

```bash
make nn_v2_clean_hf_evals ARGS="\
  --input '/workspace/data/hf/lichess-normalized/**/*.parquet' \
  --output-features-dir /workspace/data/features/nnv2_normalized_100m \
  --summary /workspace/data/features/nnv2_normalized_100m_summary.json \
  --min-depth 18 \
  --hash-sample-mod 3 \
  --hash-sample-keep 1 \
  --candidate-limit 120000000 \
  --limit 100000000 \
  --max-per-bucket 2500000 \
  --assume-unique \
  --fast \
  --terminal-fixtures \
  --uncompressed-features \
  --rows-per-shard 1000000 \
  --parquet-batch-size 262144 \
  --log-every 2000000"
```

Use `--hash-sample-mod 6 --hash-sample-keep 1` for a roughly 50M row rehearsal.
Use all rows only after the 50M/100M summaries and engine tests look sane.

## Rich PV Dataset Shape

For serious training, do not use the Hugging Face Dataset Viewer API. It is
for previews and small samples and can rate-limit. Use local parquet shards or
the raw Lichess eval dump instead.

Best richer option:

1. Download `Lichess/chess-position-evaluations` parquet shards locally.
2. Clean them with `nn_v2_clean_hf_evals`, choosing the best `(depth, knodes)`
   row per FEN and preserving the PV line as `best_line`/`best_move`.
3. Train from the resulting HalfKP feature shards.

Download the HF parquet conversion once on RunPod:

```bash
mkdir -p /workspace/data/hf/lichess-position-evals
hf download Lichess/chess-position-evaluations \
  --repo-type dataset \
  --revision refs/convert/parquet \
  --include "default/train/*.parquet" \
  --local-dir /workspace/data/hf/lichess-position-evals \
  --max-workers 16
```

Build a broad cleaned 20M feature set:

```bash
make nn_v2_clean_hf_evals ARGS="\
  --input '/workspace/data/hf/lichess-position-evals/default/train/*.parquet' \
  --output-features-dir /workspace/data/features/nnv2_hf_clean_20m \
  --summary /workspace/data/features/nnv2_hf_clean_20m_summary.json \
  --min-depth 20 \
  --min-knodes 100 \
  --hash-sample-mod 50 \
  --hash-sample-keep 1 \
  --candidate-limit 25000000 \
  --limit 20000000 \
  --max-per-bucket 500000 \
  --fast \
  --terminal-fixtures \
  --rows-per-shard 1000000 \
  --parquet-batch-size 131072 \
  --log-every 1000000"
```

The hash sample makes the 20M rows broad across the whole dataset instead of
taking the first rows from sorted parquet files. If the summary is too skewed,
change `--hash-sample-mod`, `--hash-sample-keep`, or the bucket cap before
training.

Train the first serious value model:

```bash
make nn_v2_train ARGS="\
  --input /workspace/data/features/nnv2_hf_clean_20m/manifest.json \
  --input-features /workspace/data/features/nnv2_hf_clean_20m \
  --output-dir /workspace/current/nnv2_hf_clean_20m/model \
  --export-path /workspace/current/nnv2_hf_clean_20m/nn_eval.bin \
  --epochs 6 \
  --batch-size 8192 \
  --feature-dim 384 \
  --hidden-dim 128 \
  --train-samples 20000000 \
  --val-samples 500000 \
  --device cuda \
  --log-every 200"
```

Raw Lichess fallback:

1. Stream from the public `.zst` URL and stop after the selected sample is
   converted.
2. On a RunPod with enough storage, download the `.zst` once and build multiple
   datasets from the local file.

Download once on RunPod:

```bash
mkdir -p /workspace/data/raw
curl -L --fail --retry 10 --retry-delay 10 \
  -o /workspace/data/raw/lichess_db_eval.jsonl.zst \
  https://database.lichess.org/lichess_db_eval.jsonl.zst
```

Fastest broad 20M feature build:

```bash
make nn_v2_build_feature_dataset ARGS="\
  --input /workspace/data/raw/lichess_db_eval.jsonl.zst \
  --output-dir /workspace/data/features/nnv2_eval_20m \
  --summary /workspace/data/features/nnv2_eval_20m_summary.json \
  --min-depth 18 \
  --limit 20000000 \
  --max-per-bucket 300000 \
  --sample-every 20 \
  --fast \
  --no-dedupe \
  --terminal-fixtures \
  --rows-per-shard 1000000 \
  --log-every 1000000"
```

Fallback two-stage build if you want inspectable row JSONL:

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
  --fast \
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
  --fast \
  --no-dedupe \
  --terminal-fixtures \
  --log-every 1000000"
```

Precompute feature shards from the selected set:

```bash
make nn_v2_precompute_features ARGS="\
  --input /workspace/data/labels/nnv2_eval_20m.jsonl \
  --output-dir /workspace/data/features/nnv2_eval_20m \
  --rows-per-shard 1000000 \
  --log-every 1000000"
```

Notes:

- `--stream-output` makes the builder scale to millions of rows by writing rows
  directly instead of keeping them in Python lists.
- `nn_v2_build_feature_dataset` skips the label JSONL intermediate and is the
  preferred RunPod path once the sampling knobs are chosen.
- `nn_v2_clean_hf_evals` is now the preferred path for
  `Lichess/chess-position-evaluations` because it keeps `line`, `knodes`, and
  best-row selection while producing direct feature shards.
- `--fast` avoids `python-chess` board construction for normal non-fixture
  rows. It keeps side-to-move labels, eval bands, mate scores, and phase
  estimates, but tactical buckets become approximate (`mate`, `promotion`, or
  `quiet`). This is the recommended path for 20M+ value datasets.
- `nn_v2_precompute_features` removes JSON parsing and FEN feature extraction
  from every training epoch. For serious GPU runs, train from
  `--input-features`; keep `--input` too so checkpoint metadata still points to
  the source labels.
- `nn_v2_train --input-parquet` is preferred for the first normalized one-pass
  run because it skips the feature-build stage entirely. Use dataloader workers
  to overlap parquet decode/FEN encoding with GPU training.
- `--no-dedupe` is recommended for the raw Lichess eval stream because it avoids
  holding a multi-million-FEN set in RAM. Do not use it for denormalized HF
  parquet/Viewer rows unless duplicates are desired.
- `--sample-every 20` scans broadly through the dump instead of taking a narrow
  early slice. For a different budget, use roughly
  `source_rows / desired_rows` as the stride.
- Inspect the summary before training. If it is still phase-skewed, increase
  scanning breadth or split the run into multiple different offsets/strides.

## Architecture And GPU Plan

Current exported inference supports these `CHNNUE1` value families:

- HalfKP sparse features
- white and black perspective accumulators
- side-to-move ordered accumulator pair
- linear or bottleneck output heads
- dynamic ReLU quantization for older versions
- fixed clipped ReLU quantization for versions 5 and 6
- packed `int16` incremental accumulators for linear-head SCReLU version 9
- one bounded value output

New `linear-head-screlu` exports use version 9. Feature-transformer weights are
quantized to +/-255 so accumulator sums remain safe in `int16`; the runtime
uses half-width search-stack copies, packed SIMD updates, and integer activation
conversion. Versions 1-8 remain loadable for existing model files.

`linear-head-screlu-hm` uses the same packed runtime with horizontally mirrored
king buckets. For each perspective the king is mirrored onto files e-h together
with every piece square, reducing the feature table from 40,960 to 20,480 rows.
The direct binpack trainer remaps LegacyHalfKP indices on the GPU. Serious runs
should use cosine decay, for example `--lr 8e-4 --lr-end 8e-5`; constant LR was
observed to improve training loss after the strongest playing checkpoint while
reducing match strength.

The old clipped baseline was:

- `bottleneck-head-crelu`
- `feature_dim=256`
- `bottleneck_dim=32`
- `hidden_dim=32`
- `cp_scale=400`

The next serious engine-loadable lane is now:

- `bottleneck-head-screlu`
- `feature_dim=384`
- `bottleneck_dim=48`
- `hidden_dim=32`
- `cp_scale=400`
- export format: `CHNNUE1` version 8

SCReLU is squared clipped ReLU: `clamp(x, 0, 1)^2`. It is still cheap in the C
integer evaluator, but gives the output head a stronger nonlinearity than plain
clipped ReLU. This is the best architecture currently supported by our runtime.
Do not jump to 512 until SCReLU384 is tested, because accumulator size increases
memory bandwidth and accumulator-update cost.

RunPod command for the first SCReLU384 parquet fallback run:

```bash
make nnue_train_screlu384 ARGS="\
  --input-parquet '/workspace/data/hf/lichess-normalized/**/*.parquet' \
  --output-dir /workspace/current/nnue_screlu384_full/model \
  --export-path /workspace/current/nnue_screlu384_full/nn_eval.bin \
  --epochs 1 \
  --batch-size 16384 \
  --min-depth 20 \
  --train-samples 0 \
  --val-samples 500000 \
  --val-buckets 10 \
  --terminal-fixtures \
  --shuffle-parquet \
  --num-workers 12 \
  --pin-memory \
  --parquet-batch-size 262144 \
  --save-every-batches 5000 \
  --device cuda \
  --log-every 500"
```

Cleaning is mostly CPU, RAM, and disk I/O. A bigger GPU does not make parquet
dedupe much faster. Training is GPU-bound, so use the best affordable CUDA card
after feature shards exist:

- RTX 4090: good baseline, cheap, 24GB VRAM
- RTX 5090: preferred if close in price, 32GB VRAM helps batch size
- RTX 6000 Ada / L40S: useful if 48GB VRAM is cheap enough
- H100/H200: only worth it if the hourly price is sane; our current network is
  small enough that CPU input throughput can become the bottleneck

Do not deploy an NN model because its validation loss is lower. Promotion needs
engine matches against `classic` at short and longer time controls.

After training, evaluate with:

```bash
python3 scripts/evaluate_nnue_candidate.py \
  /workspace/current/nnue_clipped256_full/nn_eval.bin \
  --label nnue_clipped256_full \
  --match-positions 64 \
  --match-think-ms 120 \
  --concurrency 8 \
  --report /workspace/current/nnue_clipped256_full/eval_report.json
```

Then repeat at `--match-think-ms 1000` if the 120 ms result is close or better.

## First Acceptance Criteria

Before spending GPU money:

1. `make test`
2. `make nn_v2_check_tools`
3. Mate, stalemate, and insufficient-material rows are present.
4. No single bucket dominates the written rows.
5. A tiny model can overfit a tiny sample locally.
6. Exported `CHNNUE1` loads in `bin/chess_uci`.
