# NN v2 Plan

This folder is for the next neural engine path. The old local legacy model is
not part of this plan.

## Rules

- Do not commit generated model binaries.
- Do not load legacy `CHNN` files.
- Train from reproducible public or locally generated data.
- Export only the local `CHNNUE1` inference format.
- Keep live Umbrel on `classic` until the NN backend beats it in head-to-head
  tests at blitz-like time controls.

## First Target

Start with value NNUE plus the existing alpha-beta search.

This is the smallest credible path to beating the handcrafted evaluator:

- the engine already has search, TT, qsearch, and move ordering
- `nn_eval.c` already supports quantized `CHNNUE1` inference
- `hce_search.c` can call NN eval through the same search
- we can test NN-vs-classic before changing the live bot

Full end-to-end policy/value search can come later, but it should not be the
first milestone. It needs far more games, compute, and infrastructure before it
can beat a conventional alpha-beta engine in blitz.

## Data Sources

Primary bootstrap source:

- Lichess public Stockfish evaluations JSONL
- Convert it with `build_value_dataset.py`
- Train value from side-to-move centipawn/mate targets

Secondary sources after the bootstrap works:

- Lichess rated game PGNs filtered to strong players and fast time controls
- Self-play positions from the current engine
- Tactical regression positions from our own losses
- Fresh Stockfish labels generated locally for positions where our engine
  disagrees with the bootstrap net

## Bootstrap Pipeline

Download or stream the Lichess eval file, then convert it:

```bash
python3 src/core/bot/nn/build_value_dataset.py \
  --input data/raw/lichess_db_eval.jsonl.zst \
  --output data/labels/lichess_eval_value_sample.jsonl \
  --min-depth 18 \
  --limit 2000000
```

Train a first small net:

```bash
python3 src/core/bot/nn/train.py \
  --input data/labels/lichess_eval_value_sample.jsonl \
  --epochs 4 \
  --batch-size 1024 \
  --feature-dim 160 \
  --hidden-dim 40 \
  --train-samples-per-epoch 500000 \
  --val-samples 100000
```

Then compare with the current classic backend:

```bash
python3 src/core/bot/test_lab.py \
  --engine classic=bin/chess_uci \
  --engine nn=bin/chess_uci \
  --uci-option nn:Backend=nn \
  --uci-option nn:NNModel=src/core/bot/nn/model/nn_eval.bin \
  --positions-count 100 \
  --think-ms 120 \
  --baseline classic
```

## Promotion Bar

A candidate net is not interesting until it passes all of these:

- `make test`
- tactical regression suite no worse than classic
- at least `+20 Elo` vs classic over a quick smoke match
- then `+40 Elo` or better over a larger paired-position match
- no speed collapse in `make bench` or UCI node logs

## Longer-Term Direction

After a value net is genuinely stronger, add policy-like help in this order:

1. NN move ordering head for root and main search ordering.
2. Tactical value head trained on blunder/loss positions.
3. Search-distilled labels from Stockfish or our own deeper search.
4. Only then consider end-to-end MCTS/policy-value play.

The practical target is not to make a tiny AlphaZero clone. It is to build a
fast CPU net that makes our alpha-beta search less blind than HCE.
