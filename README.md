# Chess

A chess project in C with a desktop app, a UCI engine, and a Lichess bot that
runs on an Umbrel home server. The engine contains two complete, selectable
backends: the current hand-crafted engine and an NNUE engine with its own
trained search profile. The live bot plays as
[Vidbot on Lichess](https://lichess.org/@/vidbot).

![Main Menu](docs/images/menu.png)

## What's inside

| Part | Where | What it does |
|------|-------|--------------|
| Desktop app | `src/app` | Play locally, watch engine games, and inspect evaluations |
| Engine | `src/core/engine` | Rules, move generation, HCE and NNUE evaluation/search, and UCI |
| Lichess bot | `src/core/bot` | Plays rated games on Lichess and manages clocks, seeks, and concurrency |
| NN tooling | `src/core/bot/nn` | Builds datasets, trains/exports NNUE and policy models, and evaluates candidates |
| Test lab | `src/core/bot/test_lab.py` | Runs paired engine matches with Elo estimates and confidence intervals |
| Experiment log | `docs/HCE_EXPERIMENTS.md` | Records HCE experiments and the match evidence behind them |

## Engine architecture

Only one backend searches a position. Selecting NNUE does not run HCE beside
it or blend their scores.

| Backend | Evaluation | Search | Intended use |
|---------|------------|--------|--------------|
| `classic` | Tapered hand-crafted evaluation in `hce_eval.c` | `hce_search.c` | Default and strongest validated HCE |
| `nn` | Incremental HalfKA-style NNUE in `nn_eval.c` | `nn_search.c` | Quantized trained model with NN-specific pruning parameters |
| `experimental` | Experimental HCE entry point | HCE search | Controlled local experiments |

The backends share legal move generation, board state, hashing, the opening
book, and the UCI/bot interfaces. Their search state, transposition tables,
piece-value support, and tuning profiles are isolated so a change calibrated
for one backend cannot silently alter the other backend's tree.

The classic engine includes magic-bitboard slider attacks, iterative
deepening, aspiration windows, PVS, null-move pruning, log-based late-move
reductions, late-move pruning, quiescence with SEE/delta pruning, killer and
history ordering, a shared atomic TT for lazy SMP, and soft/hard time budgets.
Its evaluation covers tapered material and piece-square tables, pawn
structure, mobility, rook files, king safety, and tactical pressure. Material,
PSTs, and several scalar terms have been jointly Texel-tuned.

The NN backend uses an incrementally updated quantized network, a dedicated
single-thread search implementation, backend-specific pruning and aspiration
settings, optional root policy hints, and separate model/export parity tests.

## Selecting a backend

Classic HCE is the default. In UCI, load the NN model before selecting `nn`;
the engine rejects NN selection when no valid model is loaded.

```text
setoption name Backend value classic

setoption name NNModel value /absolute/path/to/nn_eval.bin
setoption name Backend value nn
```

The Lichess bot uses the same selector:

```bash
export LICHESS_BOT_BACKEND=classic

# Or use NNUE:
export CHESS_NN_MODEL=/absolute/path/to/nn_eval.bin
export LICHESS_BOT_BACKEND=nn
```

## Building and testing

```bash
make                 # build the desktop app
make run             # play locally
make uci             # build bin/chess_uci for a chess GUI or bot
make test            # C tests plus NN dataset, training, policy, and bot-time tests
make hce_suite       # curated HCE tactical/positional regression suite
make bench           # engine benchmark
```

NN training and data targets are available from the same Makefile, including
`nn_v2_pipeline`, `nn_v2_train`, `nnue_train_bottleneck512`,
`nnue_train_clipped256`, `nnue_train_screlu384`, and `nn_policy_train`.

## Match validation

Engine changes are tested from paired positions with colors reversed. Reports
include wins/draws/losses, score, Elo estimate, a 95% confidence interval, and
`P(better)`. A short match screens candidates; longer independent matches
decide whether a playing-strength change is kept.

```bash
python3 src/core/bot/test_lab.py \
  --git-ref cand=HEAD --git-ref base=HEAD~1 --baseline base \
  --positions-count 60 --paired-colors \
  --think-ms 120 --concurrency 6

# Time-management changes need real clocks rather than fixed movetime.
python3 src/core/bot/test_lab.py ... --clock 60+0.5
```

NN candidates additionally use `scripts/check_nn_export_parity.py` to compare
PyTorch and exported C inference before match testing. HCE tuning tools include
the feature dumper, self-play/data builders, and `scripts/texel_tune.py`.

## Lichess deployment

The bot wraps the UCI engine, manages challenges and outgoing seeks, allocates
time by speed category, and supports concurrent games. Umbrel releases are
versioned by Git commit and retain rollback directories on the server. Current
public games and ratings are available on the
[Vidbot profile](https://lichess.org/@/vidbot).

```bash
make umbrel_bundle
make deploy_umbrel ARGS="user@host /target/path"
```

The deployment helper also uploads `current/nn_eval.bin` when present, so the
same packaged engine can run either backend through configuration.

## Screenshots

| | |
|---|---|
| ![Settings](docs/images/settings.png) | ![AI](docs/images/ai.png) |
| *Settings* | *AI panel* |

![AI test lab](docs/images/ai-test-lab.png)
*In-app AI test lab*

## Repository layout

```text
src/app/                         desktop UI and app workers
src/core/engine/hce_eval.c       hand-crafted evaluation
src/core/engine/hce_search.c     classic HCE search
src/core/engine/nn_eval.c        incremental quantized NNUE inference
src/core/engine/nn_search.c      NN-specific search and tuning profile
src/core/bot/                    Lichess bot and match test lab
src/core/bot/nn/                 NN dataset, training, export, and policy tools
tests/                           rules, perft, search, NN, and regression tests
data/                            opening book and position suites
docs/                            architecture notes and experiment history
```
