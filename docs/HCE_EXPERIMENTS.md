# HCE Experiments

## 2026-07-09: Countermove Heuristic
Status: rejected; not merged.

Hypothesis: remember the quiet reply that caused a beta cutoff against the
opponent's previous move, then order that reply highly when the same previous
move appears again. This should improve quiet move ordering beyond killers and
from/to history.

Variant:
- `hce-countermove` (`141cf88`): adds
  `countermove[side][prev_from][prev_to]`, rewards matching quiet moves below
  killers but above ordinary history, and updates the table on quiet beta
  cutoffs.

Validation:
- Build: `make uci` passed.
- 60-game match vs current HCE:
  `current/baselines/countermove_60g_20260709.json`
  scored 28.5/60 (`-17.4` Elo), CI95 `[-75.6, +39.9]`,
  `P(better)=27.5%`.

Conclusion: reject this shape. A countermove/continuation-history family may
still be worthwhile, but this simple bonus is not strong enough and leans
negative.

## 2026-07-09: Search Tuning Knobs + First LMR/RFP/Null Sweep
Status: infrastructure kept; no search-default change shipped.

Hypothesis: the next HCE gains are more likely in search shape than eval
deletions. Expose the main pruning/reduction constants through UCI so variants
can be tested without editing C for each attempt.

Infrastructure:
- `HceRfpMargin`: reverse-futility margin per remaining depth.
- `HceNullBase`: null-move base reduction.
- `HceNullDepthDivisor`: null-move depth divisor.
- `HceLmrBase`: late-move reduction base.
- `HceLmrDepthBonusAt`: depth threshold for the extra LMR ply.
- `HceLmrMoveBonusAt`: searched-move threshold for the extra LMR ply.

Default validation:
- Defaults are behavior-preserving: fixed-depth comparison against pre-knob HCE
  on 10 FENs at depth 9 had identical bestmove, score, nodes, and PV.
- `make hce_suite`: 6/6 passed.

Quick 40-game screen:
- `lmr_conservative` (`HceLmrDepthBonusAt=8`,
  `HceLmrMoveBonusAt=8`):
  `current/baselines/search_lmr_conservative_40g_20260709.json`
  scored 20.5/40 (`+8.7` Elo), `P(better)=58.2%`.
- `lmr_aggressive` (`HceLmrDepthBonusAt=5`,
  `HceLmrMoveBonusAt=4`):
  `current/baselines/search_lmr_aggressive_40g_20260709.json`
  scored 23.0/40 (`+52.5` Elo), `P(better)=90.1%`.
- `rfp_conservative` (`HceRfpMargin=120`):
  `current/baselines/search_rfp_conservative_40g_20260709.json`
  scored 20.5/40 (`+8.7` Elo), `P(better)=59.5%`.
- `null_aggressive` (`HceNullBase=3`):
  `current/baselines/search_null_aggressive_40g_20260709.json`
  scored 19.0/40 (`-17.4` Elo), `P(better)=32.9%`.

Confirmations:
- `lmr_aggressive` 120g:
  `current/baselines/search_lmr_aggressive_120g_20260709.json`
  scored 55.0/120 (`-29.0` Elo), `P(better)=11.6%`. Reject.
- `lmr_conservative` 120g:
  `current/baselines/search_lmr_conservative_120g_20260709.json`
  scored 63.0/120 (`+17.4` Elo), `P(better)=76.6%`.
- `lmr_conservative` 240g:
  `current/baselines/search_lmr_conservative_240g_20260709.json`
  scored 120.0/240 (`0.0` Elo), `P(better)=50.0%`. Reject as a default change.

Conclusion: keep the tuning knobs, but do not change default LMR/null/RFP yet.
The fast screen found plausible candidates, but longer matches showed the LMR
signals were noise. Future search work should use the knobs for broader sweeps
and only promote variants that survive 240+ games.

## 2026-07-08: Passed Pawn Half-Scale
Status: rejected; current passed-pawn weights kept.

Hypothesis: passed-pawn bonuses are large and may over-push the engine into
pawn-race optimism. Try half-strength passed-pawn bonuses while keeping the
same detection, outside/support modifiers, and caps.

Variant:
- `hce-passed-pawns-half` (`a48ce5d`): divide `passer_mg` and `passer_eg` by
  2 immediately before adding `terms.passed_pawns`.

Validation:
- Build: `make uci` passed cleanly.
- 60-game match vs current HCE:
  `current/baselines/tune_passed_pawns_half_60g_20260708.json`
  scored 29.5/60 (`-5.8` Elo), CI95 `[-78.6, +66.5]`,
  `P(better)=43.7%`.

Conclusion: reject. This is neutral-to-slightly-negative and has no cleanup
benefit, so keep the current passed-pawn weights.

## 2026-07-08: Tactical Penalty Half-Scale
Status: rejected; full tactical penalties kept.

Hypothesis: hanging-piece and queen-trap penalties are useful, but their static
swings may be too large now that search is faster. Try half weight for both
terms while keeping detection logic unchanged.

Variant:
- `hce-tactical-half` (`091a75b`): divide `hanging` and `queen_trap` by 2
  before adding them to eval.

Validation:
- Build: `make uci` passed, with only the existing unused `is_light_square`
  warning.
- 60-game match vs current HCE:
  `current/baselines/tune_tactical_half_60g_20260708.json`
  scored 31.0/60 (`+11.6` Elo), CI95 `[-61.4, +85.6]`,
  `P(better)=62.3%`.
- 120-game match vs current HCE:
  `current/baselines/tune_tactical_half_120g_20260708.json`
  scored 56.5/120 (`-20.3` Elo), CI95 `[-68.3, +27.0]`,
  `P(better)=20.0%`.

Conclusion: reject. The 60-game positive was noise; full-strength tactical
penalties are still better at this speed.

## 2026-07-08: Pawn Structure Penalty Ablations
Status: rejected for removal; isolated/doubled penalties kept.

Hypothesis: fixed isolated and doubled pawn penalties may be stale hand tuning.
They can over-penalize dynamic positions, and the first grouped test looked
promising enough to split.

Variants:
- `hce-pawn-penalties-off` (`455ea23`): zero all `terms.pawn_structure`,
  disabling isolated and doubled pawn penalties together while keeping passed
  pawns.
- `hce-isolated-pawn-off` (`8ff7e45`): remove only the isolated pawn penalty.
- `hce-doubled-pawn-off` (`d79e94b`): remove only the doubled pawn penalty.

Validation:
- Grouped removal 60g:
  `current/baselines/ablate_pawn_penalties_off_60g_20260708.json`
  scored 35.0/60 (`+58.5` Elo), CI95 `[-3.0, +123.8]`,
  `P(better)=96.9%`.
- Isolated-only removal 60g:
  `current/baselines/ablate_isolated_pawn_off_60g_20260708.json`
  scored 29.5/60 (`-5.8` Elo), CI95 `[-74.6, +62.6]`,
  `P(better)=43.3%`.
- Doubled-only removal 60g:
  `current/baselines/ablate_doubled_pawn_off_60g_20260708.json`
  scored 26.5/60 (`-40.7` Elo), CI95 `[-108.9, +24.4]`,
  `P(better)=11.1%`.
- Grouped removal 120g:
  `current/baselines/ablate_pawn_penalties_off_120g_20260708.json`
  scored 62.5/120 (`+14.5` Elo), CI95 `[-32.9, +62.4]`,
  `P(better)=72.6%`.
- Grouped removal 240g:
  `current/baselines/ablate_pawn_penalties_off_240g_20260708.json`
  scored 120.5/240 (`+1.4` Elo), CI95 `[-31.4, +34.3]`,
  `P(better)=53.4%`.

Conclusion: keep the penalties. The exciting 60-game grouped result evaporated
with more games, and the doubled-pawn split was clearly negative. Treat this as
a good reminder that 60 games can find candidates, not final truth, when the
signal is noisy.

## 2026-07-08: Staged TT Move Search
Status: rejected; not merged.

Hypothesis: try the transposition-table move before generating/scoring the full
legal move list. If the TT move cuts off, the node avoids full move generation;
otherwise generate the normal list and skip the already-searched TT move.

Variant:
- `hce-staged-tt-move` (`c03329d`): staged TT move in `negamax`.

Validation:
- Build: `make uci` passed, with only the existing unused `is_light_square`
  warning.
- Fixed-depth check on 10 FENs at depth 9: not node/PV identical; all 10
  positions changed node counts/PVs and median NPS was essentially flat
  (baseline 1,874,158 vs variant 1,880,529).
- 60-game match vs current HCE:
  `current/baselines/staged_tt_move_60g_20260708.json`
  scored 26.0/60 (`-46.6` Elo), CI95 `[-124.6, +27.0]`,
  `P(better)=10.7%`.

Conclusion: reject. This did not deliver the intended speed benefit and changed
search behavior enough to lose games. Keep the current full-list scoring/order
until a cleaner staged generator can preserve search semantics more carefully.

## 2026-07-08: Rook File Bonus Ablation
Status: rejected for removal; rook-file bonuses kept.

Hypothesis: rook open/semi-open file bonuses might overlap with mobility and
overvalue file occupation without actual penetration. Disable only
`terms.rook_files`, keeping rook PST and mobility intact.

Variant:
- `hce-rook-files-off` (`b96703d`): set `terms.rook_files.mg/eg = 0` in
  `eval_side` before blending.

Validation:
- Build: `make uci` passed, with only the existing unused `is_light_square`
  warning.
- 60-game match vs current HCE:
  `current/baselines/ablate_rook_files_off_60g_20260708.json`
  scored 30.0/60 (`0.0` Elo), CI95 `[-63.4, +63.4]`,
  `P(better)=50.0%`. This was in the +/-15 Elo gray zone, so it was rerun.
- 120-game match vs current HCE:
  `current/baselines/ablate_rook_files_off_120g_20260708.json`
  scored 55.0/120 (`-29.0` Elo), CI95 `[-74.6, +15.6]`,
  `P(better)=10.1%`.

Conclusion: keep rook-file bonuses. The neutral 60-game result was noise; the
larger run says removal is likely harmful.

## 2026-07-08: Bishop Pair Bonus Ablation
Status: removed and merged (`5722a80`, cleaned in `751d59e`).

Hypothesis: the standalone bishop-pair bonus may be stale hand tuning. Bishop
activity is already partly represented by PST and mobility, so the extra
`+24/+28` pair bonus might overvalue bishops in positions where the search
knows better.

Variant:
- `hce-bishop-pair-off` (`5722a80`): zero `terms.bishop_pair` before blending.
- Cleanup removed the internal bishop-pair accumulator and bonus block while
  leaving the public breakdown field compatible as zero.

Validation:
- Build: `make uci` passed, with only the existing unused `is_light_square`
  warning.
- 60-game match vs current HCE:
  `current/baselines/ablate_bishop_pair_off_60g_20260708.json`
  scored 31.5/60 (`+17.4` Elo), CI95 `[-42.1, +78.0]`,
  `P(better)=71.7%`.
- Cleaned version: `make hce_suite` passed 6/6.

Conclusion: remove. The result is modest and noisy, but it passes the removal
rule: the engine without the feature scored better than current, and the term
was pure hand-tuned eval weight with no tactical safety role.

## 2026-07-08: Tempo Bonus Ablation
Status: rejected for removal; tempo kept.

Hypothesis: the fixed `+12 cp` side-to-move bonus might be arbitrary or noisy.
Remove it in both normal eval and breakdown eval, then test against current HCE.

Variant:
- `hce-tempo-off` (`43681d0`): delete the two tempo additions in
  `hce_eval_cp_stm` and `hce_eval_breakdown`.

Validation:
- Build: `make uci` passed, with only the existing unused `is_light_square`
  warning.
- 60-game match vs current HCE:
  `current/baselines/ablate_tempo_off_60g_20260708.json`
  scored 25.5/60 (`-52.5` Elo), CI95 `[-127.8, +18.1]`,
  `P(better)=7.3%`.

Conclusion: keep the tempo bonus. The term is small but materially useful in
timed engine play, likely because it discourages sterile waiting moves and
helps the search value initiative correctly.

## 2026-07-08: Mobility Eval Ablation
Status: rejected for removal; mobility kept.

Hypothesis: the mobility term may be dead weight now that the search is faster,
and it costs repeated knight/slider attack counts in every eval. Disable the
term first without optimizing away the loops, so the match isolates chess value
rather than speed.

Variant:
- `hce-mobility-off` (`70873d5`): set `terms.mobility.mg/eg = 0` in
  `eval_side` before blending.

Validation:
- Build: `make uci` passed, with only the existing unused `is_light_square`
  warning.
- 60-game match vs current HCE:
  `current/baselines/ablate_mobility_off_60g_20260708.json`
  scored 28.5/60 (`-17.4` Elo), CI95 `[-84.6, +48.5]`,
  `P(better)=30.2%`.

Conclusion: do not delete mobility as a group. The result is not catastrophic,
but it leans negative and does not pass the ablation rule for removal. If we
return to this area, split by piece type or retune weights instead of removing
all mobility at once.

## 2026-07-08: Root GameState Copy Cleanup
Status: kept and merged (`a4ee219`).

Hypothesis: root search and aspiration retries still copied the full
`GameState`, including the large undo/hash history arrays. The interior null
move copy had already been removed; doing root moves with make/undo should be a
small pure speed win without changing fixed-depth search.

Change:
- `search_root` now makes and undoes root moves in-place instead of copying
  `GameState child = *root` for each candidate.
- Aspiration retries call `search_root(&root, ...)` directly instead of copying
  `GameState iter = root`.
- Root TT stores use the saved pre-search root hash.

Validation:
- Fixed-depth identity vs
  `current/engine_snapshots/hce_best_80b2fff_20260706_chess_uci` on 10 FENs at
  depth 9: identical bestmove, score, nodes, and PV for every position.
- NPS depth-9 comparison, 3 passes over the first five
  `data/positions/lichess_equal_positions.fen`: baseline median 1,309,475 NPS,
  root-copy-cleanup median 1,396,773 NPS (`+6.7%`).
- `make hce_suite`: 6/6 passed.
- `make test`: failed on
  `Search draw detection should not flatten a mere twofold repetition to a draw`.
  The clean `hce` baseline fails the same stale test, so this is not a
  regression from the root-copy patch.
- 60-game sanity match vs current HCE:
  `current/baselines/rootcopy_cleanup_60g_20260708.json`
  scored 31.0/60 (`+11.6` Elo), CI95 `[-51.4, +75.4]`,
  `P(better)=64.2%`.

Conclusion: keep. This is node-identical at fixed depth, measurably faster, and
the timed-game sanity check is not negative. The stale repetition unit test
should be cleaned up separately from this speed patch.

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
