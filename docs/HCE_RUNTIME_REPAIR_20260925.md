# HCE runtime repair — September 25, 2026

## Confirmed failure and changes

The live classic bot had ten lingering engine children, some weeks old. Lichess
reported zero ongoing games while recent service logs repeatedly retained one
active slot. The finish-event handler removed bookkeeping without cancelling
the game stream or closing its engine. A missed terminal stream message could
therefore retain an engine indefinitely; a missed account finish event could
also leave a slot occupied. Reconnecting a stream after clean EOF had no backoff.

The runtime now:

- Signals cancellation and closes the engine when a game finishes externally.
- Checks cancellation on unbuffered stream keepalives and during retry backoff.
- Backs off after clean EOF rather than immediately reopening in a loop.
- Closes the engine transport even if a graceful UCI quit fails.
- Reconciles with `/api/account/playing` every 30 seconds, recovering missed
  starts and finishes. New games get a 60-second grace period; malformed or
  failed snapshots never remove games. Old workers cannot remove replacements.
- Suppresses briefly stale starts for completed games and restores the proper
  bot/human slot when recovering a game.

The dedicated runtime deployment copies the existing live release, replaces
only `run.py`, and verifies preserved engine/model/book hashes. It waits for
two idle account snapshots before restarting and rolls back on startup failure.
The chess service gets `CPUQuota=400%`, `CPUWeight=50`, `Nice=5`, and cgroup-wide
cleanup with a 15-second stop timeout. This is an aggregate four-logical-CPU
ceiling, not CPU affinity or a promise of four dedicated physical cores.

## HCE clock allocation

The server's August 5 release has a newer time policy and evaluator than the
local checkout. The HCE-specific policy preserves its fast-control caps and
80% increment share. Above five-minute controls, the ceiling now follows the
remaining clock (4% plus 80% of increment, capped at 60 seconds before the
concurrency reduction), instead of a fixed 8–18-second ceiling. The existing
planning horizon and clock reserve still determine the actual budget; panic
handling is unchanged. A sole legal move gets at most 100 ms.

Examples with a full clock and one active game:

| Clock | Opening budget | Rook-ending budget |
|---|---:|---:|
| 1+0 | 0.60 s | 0.60 s |
| 3+2 | 4.50 s | 4.50 s |
| 5+3 | 5.75 s | 5.75 s |
| 10+0 | 10.95 s | 24.00 s |
| 10+10 | 19.51 s | 32.00 s |
| 30+5 | 39.52 s | 60.00 s |

NN evaluation, search, training, model files, and the local NN clock-policy
function are unchanged. Lifecycle cleanup is shared runtime infrastructure.

## Positional experiments: not promoted

Three depth-18 reference comparisons from today's games are archived in
`data/positions/hce_live_diagnostic_cases_20260925.jsonl`. They are diagnostics,
not assertions that a finite-depth reference move is infallible.

The problematic 43...Kg8 rook ending was replayed using the actual deployed
engine sources, compiled locally, with one thread and no book:

- Baseline, 10.8 seconds: Kg8, depth 21.
- Null moves disabled with at most two non-pawn/non-king pieces: Kf8, depth 16,
  still missing the reference Rc4. Rejected before a match.
- Double the endgame rook-mobility coefficient from 4 to 8: Kg8, depth 20.
  Rejected before a match.
- Baseline with a 54-second maximum: Kg8, depth 23, stopped after 35.58 seconds.

A same-root depth-20 reference comparison scored Rc4 at -60 cp, Kf8 at -244 cp,
and Kg8 at -301 cp (Black's perspective). The null-move candidate's different
move was therefore not merely an equally good alternative to the reference.

Extra time did not fix this position. No evaluator/search coefficient change
is included, and no general positional-strength or Elo gain is claimed.

## Validation and reproduction

`make test` passes. Focused tests cover stream cancellation/backoff, actual
engine-process cleanup, failed quit, stale-worker races, missed finish/start
events, human slot restoration, malformed snapshots, low-clock behavior,
and 150-move clock simulations. The architecture audit has no boundary
violations. Shell syntax validation covers the runtime-only deploy helper.

The twelve paired 60+0.5 clock games tied 6–6 (four wins, four draws, four losses
for the new policy), with no engine failures or time forfeits. Six starting
positions are too few to establish an Elo gain; this checks execution and fast
control compatibility while keeping the underlying engine identical.

All four paired 600+0 rook-ending games drew (threefold repetition, fifty-move
rule, and two insufficient-material finishes), also without engine failures
or time forfeits. These two starting positions validate clock execution, not
general playing strength. The final full `make test` run passed, including
16 focused lifecycle tests; all 28 recorded NN source/model hashes match.

Additional clock-match results and rollout evidence live under
`current/hce_repair_20260925/`. The clock matches are a smoke check rather than
a statistically powered strength result. Both players use the exact same
deployed engine sources; only their Python move-budget policy differs.

Deploy the runtime without uploading a model or replacing engine code:

```sh
bash scripts/deploy_hce_runtime.sh umbrel
```

The active release records its previous release and hashes in
`RUNTIME_REPAIR.json`. Earlier releases are retained for rollback. Do not use
the general engine deployment helper as a substitute: it builds the local
engine and may upload a local NN model.

## Live rollout

Deployed September 25 at 15:28 UTC after the active game finished:
`20260925_152443_hce_runtime`, based on `20260805_140033_b9f5c5a`.
The service started successfully with `backend=classic`. At the initial idle
check, all old engine children were gone, tasks fell to 3, and service memory
was about 30 MiB (previously about 560 MiB). The kernel reports `cpu.max` as
`400000 100000` and `cpu.weight` as `50`, confirming the four-logical-CPU
aggregate ceiling; systemd reports `Nice=5`.

This verifies restart cleanup and effective resource settings. The real-engine
offline tests verify finish cancellation; long-term leak-free live operation
has not yet been established. The original engine build identity remains in
startup logs; `RUNTIME_REPAIR.json` identifies the new wrapper release.
