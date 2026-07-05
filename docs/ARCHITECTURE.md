# Architecture

This project has three product surfaces that share one chess core:

- Desktop app: local play, UI inspection, and interactive engine testing.
- UCI engine: the stable process boundary used by match tools and bots.
- Lichess bot: production wrapper around the UCI engine on Umbrel.

The repo should keep those surfaces separate. Engine-strength work should be easy
to test locally, package for Umbrel, and roll back without dragging UI, training,
or deployment details into the search/eval code.

## Current Map

| Area | Path | Owns | Must Not Own |
| --- | --- | --- | --- |
| Engine core | `src/core/engine` | board state, rules, hashing, search, eval, UCI | UI, HTTP, Lichess, Stockfish process control, Umbrel layout |
| Desktop app | `src/app` | raylib UI, local workers, Stockfish debug bar, app test lab | engine rules/search internals beyond public headers |
| Lichess bot | `src/core/bot` | Lichess API loop, game threading, clocks, deployment env defaults | C engine internals, NN training dependencies in runtime |
| NN training | `src/core/bot/nn` | dataset streaming, PyTorch training, model export | live bot runtime imports |
| Tests | `tests` | rules, engine regressions, bot time-management tests | production secrets, network-dependent assertions |
| Scripts | `scripts` | packaging, deployment, datasets, plots | engine behavior |

## Stable Boundaries

The engine API boundary is `chess_ai.h`, `chess_io.h`, and `chess_rules.h`.
Anything outside `src/core/engine` should prefer these headers over reaching into
private HCE/NN internals. Tactical regression tests may include private headers
only when the test is explicitly guarding a private search/eval behavior.

The production bot boundary is UCI. The Lichess bot should configure and launch
`chess_uci`; it should not embed C-engine assumptions that are not exposed as UCI
options.

The Umbrel boundary is a release directory:

```text
chessbot/
  current -> releases/<release_id>
  releases/<release_id>/
  shared/
```

Deployments should create a new release, build it, move `current`, and restart
once. Live secrets stay in `chessbot.env` and should never be printed.

## Rules For Engine-Strength Work

1. One strength idea per patch.
2. Keep the live backend on `classic` unless an alternate backend wins a match
   gauntlet.
3. Preserve the Idea003 baseline until a later idea beats it with enough games.
4. Every accepted idea needs:
   - unit/regression tests for known failure modes,
   - a match result against the baseline,
   - a short note in `changes.md`.
5. Rejected ideas should be reverted, but their evidence should stay in
   `changes.md` so we stop retrying the same false positives.

## Known Hotspots

These files are allowed to exist for now, but they are the first refactor targets:

- `src/core/bot/run.py`: Lichess API, pairing policy, game loop, time policy,
  logging, and CLI parsing are all in one file.
- `src/core/engine/hce_search.c`: search, TT, move ordering, null move, qsearch,
  opening book probing, and root logic are all in one translation unit.
- `src/core/engine/nn_eval.c`: quantized `CHNNUE1` NNUE-style inference and
  accumulator updates live in one file.
- `src/app/ui.c`: layout, drawing, input handling, modals, and engine/worker
  orchestration are all in one file.
- `src/app/ai_test_runner.c`: Stockfish process control and local match logic
  are mixed with worker lifecycle.

## Refactor Sequence

Refactors should reduce risk before chasing strength:

1. Add or update tests around the current behavior.
2. Extract one concept behind the existing public API.
3. Run `make test`.
4. Avoid changing search/eval constants inside architecture patches.
5. Deploy only after tests pass and the release diff is understood.

Good first extractions:

- Bot: split `run.py` into `config`, `lichess_api`, `pairing`, `time_policy`,
  and `game_loop`, while keeping `run.py` as the CLI entrypoint.
- Search: extract TT and move ordering helpers from `hce_search.c` without
  changing search decisions.
- NN: keep data conversion, training, export, and C inference as separate
  steps with no committed generated model binaries.
- UI: separate layout/drawing from input actions in `ui.c`.

## Architecture Audit

Run:

```bash
make arch
```

The audit reports size hotspots and fails on hard boundary violations. Large
files are warnings, not failures, because this repo already has known oversized
files that need staged extraction.
