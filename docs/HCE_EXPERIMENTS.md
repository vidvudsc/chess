# HCE Experiments

## 2026-07-06: Tactical Patch Split Ablation
Status: both terms kept.

Hypothesis: the combined tactical patches (hanging-piece penalty and queen-trap
penalty) previously tested as useful, but one subterm might be noisy or dead
weight. Test each subterm independently.

Variants:
- `hce-queen-trap-off` (`291edb5`): `queen_trap = 0` in `eval_side`.
- `hce-hanging-off` (`7f55cb8`): `hanging = 0` in `eval_side`.

Validation:
- Queen-trap-off 60g:
  `current/baselines/ablate_queen_trap_off_60g_20260706.json`
  scored 29.0/60 vs current (`-11.6` Elo), CI95 `[-81.7, +57.5]`,
  `P(better)=37.0%`. This was within the +/-15 Elo gray zone, so it was rerun.
- Queen-trap-off 120g:
  `current/baselines/ablate_queen_trap_off_120g_20260706.json`
  scored 54.0/120 vs current (`-34.9` Elo), CI95 `[-84.3, +13.2]`,
  `P(better)=7.8%`. Keep queen-trap penalty.
- Hanging-off 60g:
  `current/baselines/ablate_hanging_off_60g_20260706.json`
  scored 27.5/60 vs current (`-29.0` Elo), CI95 `[-96.7, +36.5]`,
  `P(better)=19.3%`. Keep hanging-piece penalty.

Conclusion: neither tactical subterm is safe to delete at current strength.
The combined earlier result was not hiding a clearly removable half; both terms
appear to contribute enough tactical guidance to keep despite their static-eval
ugliness.

## 2026-07-06: HCE Branch Baseline Anchor
Status: baseline only; no engine changes.

Pinned current best HCE branch commit:
`80b2fff0c4cab2582e6786f1e733449656fe05bf`.

Snapshot binary:
`current/engine_snapshots/hce_best_80b2fff_20260706_chess_uci`
(`sha256=92ebd8ca2f1314a419cffc3107332d2d9a073333281e6dad87c9867cf6b1fe9c`).

NPS reference:
- Command shape: UCI `go depth 9` on the first five
  `data/positions/lichess_equal_positions.fen` middlegame positions.
- Report: `current/baselines/hce_best_80b2fff_nps_depth9_20260706.json`
- NPS values: 1,436,042; 1,446,244; 1,378,394; 1,336,608; 1,334,169.
- Mean NPS: 1,386,291. Median NPS: 1,378,394.
- Rerun after suspected local CPU contention:
  `current/baselines/hce_best_80b2fff_nps_depth9_20260706_rerun2.json`
  mean 1,328,248 / median 1,313,954. Keep the higher first run as the better
  speed reference, and treat roughly 4-5% NPS swing as normal desktop noise.

60-game sanity baseline:
- Match: current `bin/chess_uci` vs the snapshot binary, 30 positions with
  paired colors, `120ms/move`, concurrency 6, no book.
- Report: `current/baselines/hce_best_80b2fff_self_snapshot_60g_20260706.json`
- Result: HCE 29.0/60 vs snapshot 31.0/60 (`-11.6` Elo),
  CI95 `[-77.5, +53.5]`, `P(better)=36.3%`.
- Rerun after suspected local CPU contention:
  `current/baselines/hce_best_80b2fff_self_snapshot_60g_20260706_rerun2.json`
  scored exactly HCE 30.0/60 vs snapshot 30.0/60 (`0.0` Elo),
  CI95 `[-65.5, +65.5]`, `P(better)=50.0%`.
- Interpretation: same-code timed-search noise floor, not a regression signal.
  Future 60-game HCE experiments should clear this noise band or be rerun at
  120+ games before keeping behavior changes.

## 2026-07-06: Quick Wins — 2-Fold Repetition Scoring
Status: kept and shipped.
- Removed dead search plumbing (verification_plies, pv_move); node-identical.
- Repetition rework: score the FIRST repetition as a draw inside the search
  and scan only same-side history entries (stride 2). Match vs parent commit:
  70.5/120, +61.4 Elo, CI95 [+20.8, +103.8], P=99.9%
  (`current/repetition_2fold_120g.json`). Biggest single validated gain so
  far; old 3-fold-in-search let the engine shuffle into repetition draws from
  winning positions (seen as heavy THREEFOLD draw rates in earlier matches).

## 2026-07-06: Ablation Matrix + Consolidation
Status: weak trims kept and shipped; king safety kept intact; D validated.
- Kept after removal clearly lost (60g each): tactical eval patches (-58.5),
  reverse futility (-52.5), time management D under clocks (-40.7, P=3.3%,
  first clock-mode validation via new test_lab --clock).
- King-safety split looked removable per-term (shield +5.8, file pressure
  +34.9, attack units +23.2/0.0 rerun) but combined removal LOST at 120g:
  -31.9 Elo, P=3.8% (`current/consolidate_king_safety_120g.json`).
  Overlapping terms cover for each other; king safety stays whole.
- Weak trims combined WON at 120g: removing recapture extension, root-order
  bonus, and positional extras (outposts, bishop quality, connected-passer)
  scored 66/120, +34.9 Elo, CI95 [+2.1, +68.2], P=98.2%
  (`current/consolidate_weak_trims_120g.json`). Shipped as clean deletions;
  cleanup binary verified node-identical to the tested variant.
- Also fixed live-bot bug found during ablation: UCI position replay stopped
  at internal insufficient-material adjudication (KN vs KN), desyncing the
  board and emitting illegal bestmoves.

## 2026-07-05: Search Speed Pack + A/C/D (B reverted)
Status: kept (B reverted).
- Speed (node-identical, verified vs baseline binary at fixed depth): magic
  bitboard sliders, in-place null move with incremental hash, single-pass eval
  attack unions, classic eval cache, lazy interior static eval.
  NPS 565K -> 1.22M middlegame, 2.5x on kiwipete.
- A qsearch tactical movegen: +17.4 Elo vs speed base, CI95 [-29.7, +65.1],
  P=76.6% (60g). Kept. `current/chain_A_vs_speedbase_60g.json`
- B fail-soft: -34.9 Elo vs A, CI95 [-81.1, +10.2], P=6.5% (60g). Reverted
  in dba10ed. `current/chain_B_vs_A_60g.json`
- C TT 1M entries + generation aging: +11.6 Elo vs B, CI95 [-24.5, +48.0],
  P=73.5% (60g). Kept. `current/chain_C_vs_B_60g.json`
- D soft/hard time split + fail-low extension: dormant under fixed movetime,
  still needs a clock-based (wtime/btime) match or live Lichess validation.
- Final stack vs pre-session baseline: 35.5/60, +64.4 Elo,
  CI95 [+19.4, +111.7], P=99.8%. `current/final_stack_vs_baseline_60g.json`
- Test lab gained --concurrency (parallel games, ~6x faster matches).

This is the lightweight ledger for HCE changes. Each search/eval idea should
name the hypothesis, the exact files touched, the validation commands, and the
fallback path.

## 2026-07-05: Solid Weighted Opening Book

Status: kept.

Hypothesis: blitz/bullet losses are amplified by the broad 100-line opening
seed file, which can push the shallow HCE into sharp or offbeat structures
before it has enough depth to defend accurately. A conservative weighted book
should keep the same engine in more stable early positions without changing
search or eval.

Touched files:

- `scripts/build_solid_opening_book.py`
- `data/openings/opening_book.txt`
- `data/openings/README.md`
- `data/positions/startpos.fen`
- `.gitignore`
- `docs/HCE_EXPERIMENTS.md`

Validation results:

- Generated `data/openings/opening_book.txt` from
  `data/openings/opening_games_100.txt`: kept 78 lines, dropped 22 risky lines.
- UCI smoke test confirmed book usage beyond the first ply with
  `position startpos moves ...`.
- 32-game start-position match, same engine, broad book vs solid book:
  `current/hce_testlab_solid_book_vs_broad_32g_20260705.json`
  scored solid book 18.0/32 against broad book (`+43.7` Elo estimate,
  CI95 `[-7.3, +96.5]`, `P(better)=95.3%`).

Fallback:

- Regenerate a broader book with `python3 scripts/build_solid_opening_book.py --include-risky`
  or restore package fallback to `data/openings/opening_games_100.txt`.

## 2026-07-04: Promotion-First Move Ordering

Status: rejected and reverted.

Hypothesis: in blitz/bullet pawn races, quiet promotions are too tactical to sit
behind every ordinary capture. The search already extends promotions and treats
them as qsearch tactical moves, so main-search ordering should spend shallow
node budget on promotions before routine captures.

Touched files:

- `src/core/engine/hce_search.c`
- `docs/HCE_DIAGRAM.md`
- `docs/HCE_EXPERIMENTS.md`

Validation plan:

- Snapshot baseline:
  `current/engine_snapshots/hce_idea003_baseline_20260704_promo_order/chess_uci`
- `make hce_suite`
- `make test`
- `make bench`
- Short isolated match against the saved baseline with `make hce_testlab`.

Validation results:

- `make hce_suite`: 6/6 passed.
- `make test`: passed.
- `make bench`: passed.
- 32-game isolated short-time match:
  `current/hce_testlab_promo_order_vs_baseline_32g_20260704.json`
  scored promotion-order 15.5/32 against baseline (`-10.9` Elo estimate,
  CI95 `[-59.6, +37.5]`, `P(better)=32.9%`).
- Reverted. The idea was plausible for pawn races, but the measured ordering
  perturbation was not positive.

Fallback:

- Done: restored the promotion move-score bonus from `1600000 + value * 16` to
  `700000 + value * 8`.

## 2026-07-04: SEE Capture Ordering

Status: superseded and reverted after dirty-tree audit.

Hypothesis: in blitz/bullet, the engine wastes too much node budget searching
obviously losing captures before useful quiet moves. Ordering main-search
captures by SEE should keep winning/equal captures near the front while pushing
bad captures behind killers/history moves.

Touched files:

- `src/core/engine/hce_search.c`
- `docs/HCE_DIAGRAM.md`
- `data/positions/hce_position_suite.jsonl`
- `scripts/run_hce_position_suite.py`
- `Makefile`

Validation:

- `make hce_suite`: 6/6 passed.
- `make test`: passed.
- `make bench`: passed.
- Dirty-worktree smoke match:
  `make hce_testlab ARGS="--positions-count 8 --think-ms 40 --max-depth 8 --max-plies 120 --out current/hce_testlab_dirty_vs_head_20260704.json"`
  scored worktree 6.5/16 against `HEAD` (`-65.9` Elo estimate,
  CI95 `[-170.2, +27.5]`).

Fallback:

- Revert the `move_score` capture-ordering hunk in `src/core/engine/hce_search.c`.
- Keep the suite runner and docs unless they are wrong; they are infrastructure,
  not part of the playing-strength hypothesis.

Notes:

- This is meant to be measured against short-time tactical regressions first,
  then against `make hce_testlab ARGS="..."` snapshots for Elo-ish match data.
- Local suite reports:
  `current/hce_position_suite_20260704_164901.json`,
  `current/hce_position_suite_20260704_165129.json`
- The smoke match is not an isolated measurement of this SEE change because the
  worktree already contains broad older search/eval/NN edits. Treat it as a
  warning that the current full dirty engine needs smaller isolated experiments.
- Dirty-tree audit found that `changes.md` already tested SEE capture ordering
  against the kept Idea003 baseline and rejected it. The code was realigned to
  MVV/LVA capture ordering.

## 2026-07-04: Guarded Forward Futility

Status: superseded and reverted after dirty-tree audit.

Hypothesis: the dirty tree removed late-quiet forward futility pruning, which
likely wastes blitz/bullet node budget after move four at shallow depths. A
guarded version should restore search efficiency while avoiding the old risk of
dropping checking quiets.

Touched files:

- `src/core/engine/hce_search.c`
- `docs/HCE_EXPERIMENTS.md`

Validation plan:

- Snapshot baseline:
  `current/engine_snapshots/hce_see_baseline_20260704_1652/chess_uci`
- `make hce_suite`
- `make test`
- `make bench`
- Match current worktree against the saved snapshot with `make hce_testlab`.

Validation results:

- `make hce_suite`: 6/6 passed.
- `make test`: passed.
- `make bench`: passed.
- 32-game isolated snapshot match:
  `current/hce_testlab_futility_vs_see_20260704.json`
  scored futility 17.0/32 against the SEE-only baseline (`+21.7` Elo estimate,
  CI95 `[-53.2, +98.8]`, `P(better)=71.6%`).
- 64-game isolated snapshot match:
  `current/hce_testlab_futility_vs_see_64g_20260704.json`
  scored futility 32.5/64 against the SEE-only baseline (`+5.4` Elo estimate,
  CI95 `[-43.9, +55.0]`, `P(better)=58.6%`).

Fallback:

- Remove the `futility_candidate` block in `negamax` and keep the prior
  SEE-only baseline snapshot as the comparison point.

Notes:

- The larger match mostly erased the early positive signal. Keep this only as a
  provisional speed/search-shape experiment unless later tests show a clearer
  time-control win.
- Dirty-tree audit found that the kept Idea003 baseline explicitly removed
  shallow quiet futility and beat the old release over 120 games. The guarded
  futility variant was reverted to preserve that baseline.

## 2026-07-04: Endgame King-Safety Fadeout

Status: rejected after neutral local match; code reverted.

Hypothesis: carrying one-quarter of the king-safety penalty into the endgame can
make the HCE too timid with king activity in queenless/simple positions. Keep
the middlegame king danger unchanged, but fade the king-safety term to zero in
pure endgame blending.

Touched files:

- `src/core/engine/hce_eval.c`
- `docs/HCE_EXPERIMENTS.md`

Validation plan:

- Snapshot baseline:
  `current/engine_snapshots/hce_guarded_futility_baseline_20260704_1708/chess_uci`
- `make hce_suite`
- `make test`
- `make bench`
- Match current worktree against the saved guarded-futility snapshot with
  `make hce_testlab`.

Fallback:

- Restore `eval_term_add(&terms.king_safety_penalty, -king_danger, -(king_danger / 4));`.

Validation results:

- `make hce_suite`: 6/6 passed.
- `make test`: passed.
- `make bench`: passed.
- 64-game isolated snapshot match:
  `current/hce_testlab_kingfade_vs_guarded_64g_20260704.json`
  scored king-fade 32.0/64 against the guarded-futility baseline (`0.0` Elo
  estimate, CI95 `[-40.3, +40.3]`, `P(better)=50.0%`).
- The one-line eval change was reverted because it added no measurable strength.
