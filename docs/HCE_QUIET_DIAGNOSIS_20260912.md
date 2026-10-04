# Quiet-move diagnosis — 2026-09-12

The broad search ablations do not justify changing the production defaults.
Disabling late-move pruning essentially tied the normal search; disabling
late-move reductions increased average reference-score loss. Extra time helped
average loss, but did not consistently recover the reference engine's choices.
These are selected-position diagnostics, not Elo measurements.

## Reproducible setup

- Baseline: `c3514fcd1c5d7f15e762135d0d46855d457b34b3`, current HCE sources.
  The isolated baseline matched all 22 original C source/header files.
- Corpus: 137 distinct positions from 20 historical VidBot losses, archived in
  `data/positions/hce_quiet_diagnostic_cases.jsonl`. Both the historical played
  and reference moves are legal non-captures, excluding promotions; checks
  are allowed. Historical loss is 50–2,000 cp, excluding mate-like scores.
- Original source: `current/worktrees/hce_2500/current/vidbot_recent_300_blunders_d11_v2.jsonl`.
  SHA-256: `9c0db6f5f2f5544a7075ff8c38f739149bfafe8e18575d9b34e44f042fbccf6e`.
  The archived corpus reproduces the original selected order with seed 20260912.
- HCE: classic backend, one thread, empty opening book, maximum depth 32.
  Each search starts in a new process: this engine's `ucinewgame` does not
  clear the transposition table, so reusing processes would confound the arms.
- All ablations are patched copies under `current/`; production sources are
  not edited. No-LMP disables the late quiet-move skip. No-LMR forces the
  computed late-move reduction to zero. Other search mechanisms remain active.
- Reference: locally installed **Stockfish 18**, one thread, 64 MB hash,
  depth 14. First discover a reference move, then compare it, both historical
  moves, and all HCE choices together using root-restricted MultiPV at the
  same depth. The historical labels are not assumed to remain correct.
- Error below means the score gap between the best reference-scored candidate
  and the chosen move, from the root side's perspective. A different move can
  receive full credit. It is not a comparison of HCE's own centipawn scale.

## Complete replay results

Nine positions contained a mate score in the new reference comparison and
were excluded from centipawn aggregates. All arms below use the same remaining
128 positions. "Within 30 cp" counts near-best choices, not exact move matches.

| Search | Budget | Mean error | Within 30 cp | Median completed depth |
|---|---:|---:|---:|---:|
| Baseline | 120 ms | 66.88 cp | 51/128 | 11 |
| Baseline | 1,200 ms | 57.62 cp | 48/128 | 15 |
| Disable LMP | 1,200 ms | 57.39 cp | 52/128 | 14 |
| Disable LMR | 1,200 ms | 65.76 cp | 49/128 | 11 |
| Disable both | 1,200 ms | 58.45 cp | 49/128 | 10 |

Against the 1,200 ms baseline, no-LMP improves 20 positions by at least 30 cp
and worsens 17. No-LMR improves 19 and worsens 24. The near-zero average change
from no-LMP is not evidence of a general strength gain, and the additional
depth cost from removing reductions does not translate into better choices.

The original played move is still at least 50 cp worse in only 87 of the 128
non-mate positions. On these re-confirmed errors, mean losses are 82.03 cp
(short baseline), 69.91 (long baseline), 72.57 (no-LMP), 82.69 (no-LMR), and
73.21 (both). The other 41 historical error labels were not reproduced by the
current reference configuration; engine version and depth differ from the
archived analysis, so this is label sensitivity rather than proof that the
old analysis was incorrect.

To avoid already decisive positions dominating the averages, a secondary
subset retains re-confirmed errors whose best reference score is within
±300 cp. On those 60 positions the corresponding means are 80.83, 66.95,
65.95, 77.43, and 75.53 cp. This subset also gives no compelling broad ablation.

Forty of the 128 non-mate positions remain more than 30 cp from the reference
choice in every arm. That establishes a useful investigation set, but does
not by itself prove an evaluation bug: all arms retain other selective search
mechanisms and finite horizons.

## Deeper follow-up

From the persistent cases, select the largest baseline errors with a best
reference score within ±300 cp, taking at most one position per game. Eight
positions received a fresh baseline search with a 12-second budget. Stockfish
18 independently rediscovered a reference move at depth 18 and rescored all
earlier candidates plus the new choice together at that depth.

Both error columns below use this new depth-18 comparison. There were no mate
scores in these eight comparisons. The subset is deliberately difficult and
is not representative of ordinary play.

| Game / ply | 1.2-second choice / error | 12-second choice / error | Completed depth |
|---|---|---|---:|
| tQBNYoBL / 26 | Ne5 / 207 cp | Ne5 / 207 cp | 16 |
| KoPumsdd / 112 | Be6 / 147 cp | Rc8 / 190 cp | 22 |
| Di1HcVmz / 58 | Bd6 / 130 cp | c5 / 0 cp | 19 |
| Jc2JQzrC / 58 | Qb5 / 162 cp | Qb5 / 162 cp | 19 |
| uWqdZBfo / 20 | Qd6 / 102 cp | Qd6 / 102 cp | 19 |
| qSfqhc0U / 33 | Nd4 / 114 cp | b4 / 0 cp | 20 |
| NvLEX85s / 116 | Rd8 / 173 cp | Rd8 / 173 cp | 20 |
| ufDBIIg7 / 34 | b5 / 64 cp | b5 / 64 cp | 19 |

Two cases recover the best reference-scored choice; six remain 64–207 cp
behind it. The persistent six are a better starting point for examining
quiet-move evaluation and selective-search traces than another global pruning
change. These results cannot identify a specific faulty evaluation term or
prove that still more search would not help.

## Limits and next experiment

This corpus is selected from losses, contains correlated positions from the
same games, and uses FEN replay without the original repetition history,
warm transposition table, live clock, or multi-thread search. The time figures
are maximum budgets; the engine's normal iteration-start policy can stop
early. Median actual search times were 84, 840, 908, 925, and 1,054 ms in table
order. Reference searches at finite depth are fallible.

Keep the current pruning defaults. Before retuning evaluation, use persistent
competitive positions with deeper, stable reference rankings; a candidate
should address a demonstrated failure and then pass independent paired games.
No candidate was promoted, no match-based Elo gain is claimed, and no deployment
was performed for this diagnosis.

## Running it again

From the repository root, with `python-chess`, a C compiler, make, and Stockfish:

```bash
python3 scripts/diagnose_hce_quiets.py \
  --input data/positions/hce_quiet_diagnostic_cases.jsonl \
  --out current/hce_quiet_diagnosis_fresh
python3 tests/test_hce_quiet_diagnosis.py
```

Use a fresh output directory. `--max-positions` provides a deterministic small
sample for smoke checks. `--teacher` selects a Stockfish binary; the default
is `/opt/homebrew/bin/stockfish`. Keep the machine free of other heavy jobs.

After the main replay, reproduce the selective 12-second follow-up with:

```bash
python3 -u - <<'PY'
from pathlib import Path
from scripts.diagnose_hce_quiets import follow_up
follow_up(Path('current/hce_quiet_diagnosis_fresh').resolve(),
          Path('/opt/homebrew/bin/stockfish'))
PY
```

Artifacts for this run are in `current/hce_quiet_diagnosis_20260912/`:
`manifest.json` records source/input/binary provenance, `results.jsonl` retains
every search and reference comparison, and `summary.json` records all subsets.
`followup.jsonl` retains the eight deeper searches and their reference rankings.
The four source snapshots, binaries, and compiler logs are retained there.
Three focused tests cover corpus filtering/deduplication, same-root/same-depth
reference comparisons, and exclusion of mate horizons and unconfirmed labels.
