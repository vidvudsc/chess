# NN Training Plan

This directory is for Python-side NN training and dataset streaming. The engine-side C inference should live under `src/core/engine` later. Do not duplicate the source JSONL unless streaming becomes a real bottleneck.

## Dataset policy

Use the original JSONL directly:

- source file: `data/labels/stockfish_san_10m.jsonl`
- default filter: `depth >= 15`
- do not materialize a second cleaned JSONL as a first step
- split train/validation by stable FEN hash, not by file copies

Why:

- `depth >= 15` keeps almost all rows while removing the weakest tail
- depth quality is noisy, so weighting is better than a brutal hard cutoff
- a second 1.5 GB JSONL is wasted space unless it buys major training speed

## Mate rows

Depth `99` rows in this dataset are effectively mate rows, not normal deep centipawn rows.

Practical handling:

- keep them
- encode them as bounded extreme targets for value
- lower their sample weight a bit so they do not dominate

Do not treat `depth == 99` as ordinary "best quality centipawn" data.

## First training target

Start with a value network only.

Reason:

- the existing engine already has alpha-beta search
- replacing the static evaluator is the smallest integration step
- policy training can come later from `best_move`

Recommended value target:

- normal evals: convert pawns to centipawns, clip, then squash to `[-1, 1]`
- mate evals: map directly to `-1` / `+1`
- flip the target for Black-to-move positions so the network learns side-to-move value, matching the side-to-move ordered accumulators

Recommended weighting:

- depth weight uses a smooth square-root curve from the streaming loader
- `depth 15` starts around normal weight
- deeper rows get progressively more influence
- mate rows are slightly downweighted so they do not dominate

## First model shape

Use a value-only HalfKP-style network:

- two HalfKP accumulators:
  - White perspective
  - Black perspective
- concatenate them in side-to-move order
- feed through a small MLP
- output a bounded scalar value in `[-1, 1]`

Current prototype:

- sparse HalfKP-style features from FEN
- `EmbeddingBag` accumulator
- `128` accumulator dim
- `32` hidden dim
- `tanh` output

## Suggested rollout

1. Stream the JSONL with filtering in code
2. Train a small value-only model
3. Evaluate offline against the tactical regression suite and test lab
4. Integrate C inference into `src/core/engine`
5. Only then consider policy or full move ordering assistance

## Current helpers

- `training_data.py`: streaming rows, split logic, target conversion, sample weights
- `dataset_inspect.py`: quick dataset inspection without rewriting the file

Example:

```bash
python3 src/core/bot/nn/dataset_inspect.py \
  --input data/labels/stockfish_san_10m.jsonl \
  --min-depth 15 \
  --split train \
  --limit 50000
```
