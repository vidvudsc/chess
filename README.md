# Chess

A chess project in C: a desktop app, a UCI engine, and a Lichess bot that runs
24/7 on a home server — plus the test lab used to validate every engine change
with real matches.

![Main Menu](docs/images/menu.png)

## What's inside

| Part | Where | What it does |
|------|-------|--------------|
| Desktop app | `src/app` | Play locally, watch engine games, inspect evals |
| Engine | `src/core/engine` | Bitboard move generation, search, evaluation, UCI binary |
| Lichess bot | `src/core/bot` | Plays rated games on Lichess, deployed to an Umbrel server |
| Test lab | `src/core/bot/test_lab.py` | Engine-vs-engine matches with Elo estimates and confidence intervals |
| Experiments log | `docs/HCE_EXPERIMENTS.md` | Every engine change and the match evidence behind it |

## The engine

Hand-crafted evaluation (HCE) with a classical alpha-beta search:

- **Board**: bitboards with magic bitboard slider attacks
- **Search**: iterative deepening, aspiration windows, PVS, transposition table
  with aging, null-move pruning, late-move reductions, check/promotion
  extensions, killer moves + history heuristic
- **Quiescence**: tactical-only move generation, SEE pruning, delta pruning
- **Evaluation**: tapered material/PST, pawn structure, mobility, king safety,
  and tactical pressure terms — every term match-validated (see the log)
- **Time management**: soft/hard budget split with extension on falling score
- **Opening book**: curated solid lines, weighted random selection

An optional NN evaluation backend (NNUE-style, trained with the tooling in
`src/core/bot/nn`) can replace the HCE eval at runtime.

## How changes get validated

No engine change ships on vibes. The workflow:

1. One change = one commit, built into its own binary
2. `test_lab.py` plays paired-color matches from equal positions
   (concurrent games, fixed movetime or real clocks with flag falls)
3. The report gives score, Elo estimate, 95% CI, and P(better)
4. Keep, revert, or escalate to a longer match — and log the verdict

```bash
# example: candidate commit vs its parent, 120 games, 6 in parallel
python3 src/core/bot/test_lab.py \
  --git-ref cand=HEAD --git-ref base=HEAD~1 --baseline base \
  --positions-count 60 --think-ms 120 --concurrency 6

# time-management changes need real clocks instead of fixed movetime
python3 src/core/bot/test_lab.py ... --clock 60+0.5
```

## Quick start

```bash
make            # build the desktop app
make run        # play locally
make uci        # build the UCI engine (bin/chess_uci) for any chess GUI
make test       # full test suite: rules, perft, clock, search, tactics
make bench      # engine benchmark
```

## The Lichess bot

The bot wraps the UCI engine, manages clocks and concurrency, and streams
games from the Lichess API. It deploys to an Umbrel home server as a systemd
service:

```bash
make umbrel_bundle                                   # package (source; builds on the server)
make deploy_umbrel ARGS="user@host /target/path"     # ship a release + restart
```

Releases are versioned by git commit with automatic rollback directories kept
on the server.

## Screenshots

| | |
|---|---|
| ![Settings](docs/images/settings.png) | ![AI](docs/images/ai.png) |
| *Settings* | *AI panel* |

![AI test lab](docs/images/ai-test-lab.png)
*In-app AI test lab*

## Repo layout

```
src/app/            desktop UI and app workers
src/core/engine/    rules, movegen, search, eval, UCI entrypoint
src/core/bot/       Lichess bot, test lab, NN training pipeline
tests/              rules, perft, clock, search, and tactical regression tests
data/               opening book, position suites
docs/               specs, architecture notes, experiments log
```
