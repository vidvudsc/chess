# HCE Experiments

## 2026-07-12: Deeper-HCE-trained quiet move policy
Status: rejected; engine restored exactly.

Trained a compact policy-PST from 1,557 quiet best moves selected by depth-14
HCE searches over 2,000 diverse positions. Features were relative source/to
squares by piece, castling, and destination pawn attack/support. Held-out top-1
teacher accuracy reached 7.7% with mean teacher rank 10.04.

- Full policy weight expanded depth-11 nodes by about 15%; rejected before a
  match.
- Quarter weight was node-neutral and scored 32.0/60, `+23.2` Elo,
  P(better)=74.7% (`current/move_policy_quarter_60g.json`). Below gate.
- Applying the quarter policy only to moves with zero dynamic history reduced
  fixed-depth nodes by 6%, but scored exactly 30.0/60, `0.0` Elo,
  P(better)=50.0% (`current/move_policy_cold_60g.json`).

Conclusion: the static policy can make alpha-beta cheaper, but its teacher
ranking does not improve timed play. Removed all policy code and generated
weights; do not confuse node reduction with strength.

## 2026-07-12: Residual-term and pawn-activity Texel follow-up
Status: tuning infrastructure kept; candidate weights rejected.

Extended the exact Texel decomposition to cover the existing passed-pawn,
king-danger, hanging-piece, and queen-trap terms as percentage scales. Added
zero-default features for unobstructed pawn pushes and pawn attacks on minor or
major pieces. Refreshed 2,571 VidBot games into 30,053 positions and combined
them with 11,894 paired self-play positions; qsearch filtering retained 27,156
quiet positions. Exact reconstruction passed on both baseline and candidate
dumps.

Results against the current baseline at fixed 120ms, paired colors:

- Residual scales were stable across three splits but scored 30.5/60,
  `+5.8` Elo, P(better)=56.5% (`current/residual_tune_60g.json`). Rejected.
- Pawn activity converged to push `+5/+1`, minor threat `+1/0`, major threat
  `0/0`, but scored 31.5/60, `+17.4` Elo, P(better)=70.4%
  (`current/pawn_activity_60g.json`). Directional but below the promotion gate;
  rejected rather than selecting on noise.

All candidate weights were restored to their behavior-preserving defaults.
The expanded zero-weight infrastructure matched score, nodes, and PV on 36/36
fixed-depth searches, with no measurable speed change. HCE suite 6/6, tactical,
and bot-time tests passed.

## 2026-07-12: Unified clock-aware live move budgeting
Status: kept; requires live observation after deployment.

Replaced the overlapping base-time, cap, and concurrency reductions in the
Lichess bot with one policy: estimate moves remaining, preserve a clock
reserve, spend 65% of increment, adjust for checks/forced moves/endgames, then
apply one time-control cap and one low-clock safety cap. Concurrency now reduces
only the final ceiling (90% for two games, 80% for three) instead of shrinking
both the calculated budget and its cap.

Representative solo ceilings are now 0.5s at 1+0, 1.48s at 1+1, 3.5s at 3+2,
4.5s at 5+3, 6s at 5+5 or 10+0, 10s at 10+10, and 12s at 15+10. Simulated
10+0 retained 173s after 80 engine moves; simulated 10+10 retained its full
600s base clock after 90 moves. Panic caps still dominate below 10s/3s/1s.

Validation: expanded bot-time tests cover bullet, blitz, rapid, concurrency,
low-clock panic, and long-game clock depletion. Python compilation, HCE suite
6/6, tactical, clock, rules, perft, and NN data/training smoke tests passed.
Full `make test` retains only the known stale twofold-repetition assertion.

## 2026-07-12: Remove the live UCI depth-16 ceiling
Status: kept; live-bot configuration improvement.

The current Umbrel release log contained 146 games and 6,974 engine moves.
After excluding opening-book and forced-mate moves, normal searches averaged
depth 14.78 with a median of 15, and 47.7% stopped exactly at the UCI default
`MaxDepth=16`. The cap affected 33.5% of bullet, 51.2% of blitz, and 61.8% of
rapid moves. Capped searches consumed only 34.3% of their assigned budget on
average (31.3% in rapid), proving that depth rather than time stopped them.

Changed the UCI default `MaxDepth` from 16 to the engine-supported maximum of
32. Time management remains the effective limit for every move. A 120ms
middlegame smoke search still stopped normally at depth 9 after 73ms, while a
quiet 1.5-second endgame reached depth 19 in 964ms instead of stopping at 16.

Validation: UCI reports default 32; HCE position suite 6/6, tactical, clock,
rules, perft, bot-time, and NN data/training smoke tests passed. Full `make
test` retains only the known stale twofold-repetition assertion in `test_ai`.

## 2026-07-12: Selective-depth follow-up after evaluator speedup
Status: rejected; engine restored to `07c5bf3` behavior.

Three search-allocation candidates were tested against the evaluator-speedup
baseline at fixed 120ms with paired colors. All passed the 6/6 HCE position
suite before match play:

- Raising the iterative-deepening start cutoff from 55% to 65% used about 18%
  more nodes and gained a completed ply on 4/30 probe positions, but scored
  27.0/60: `-34.9` Elo, P(better)=14.4%
  (`current/time65_vs_base_60g.json`).
- A PV-only singular extension with deep exact/lower-bound TT evidence scored
  30.5/60: `+5.8` Elo, P(better)=56.9%
  (`current/singular_pv_vs_base_60g.json`). Neutral; rejected.
- A modestly more conservative logarithmic LMR divisor (`2.25 -> 2.50`)
  slightly raised average probe depth, but scored 27.0/60: `-34.9` Elo,
  P(better)=15.9% (`current/lmr250_vs_base_60g.json`).

Conclusion: completed depth alone is a poor promotion metric. The current time
cutoff and LMR curve allocate nodes better in games, while singular proof
searches cost about as much depth as their tactical focus returns. Future
search work should improve move ordering directly instead of adding depth or
changing broad reduction thresholds.

## 2026-07-12: Cached pawn terms and consolidated attack evaluation
Status: kept; behavior-preserving speed improvement.

Runtime sampling put `eval_side` at roughly one third of CPU time and
`compute_attack_unions` at another 12%. The evaluator now caches material,
MG/EG PST, isolated/doubled, and passed-pawn terms by the two pawn bitboards.
It also computes slider attacks once per piece and reuses them for attack
unions, mobility, and king-pressure units instead of generating the same
attacks up to three times.

Validation:
- Clean `make uci`; HCE position suite 6/6, rules, clock, perft, tactical,
  bot-time, NN dataset, and NN training-smoke tests passed.
- The full `make test` reached only the pre-existing stale twofold-repetition
  assertion in `test_ai`, unchanged from the clean baseline.
- Exact fixed-depth identity on 100 FENs: same score, node count, and PV.
- Fresh Texel dump verification: 0/599 reconstruction mismatches.
- Six alternating pawn-cache-only trials showed a median `+5.1%` NPS.
- Four alternating trials of the complete optimization showed a stable median
  `+11.4%` NPS (all trials `+11.2%` to `+12.4%`). At 500ms over 20 positions,
  average completed depth moved from 12.45 to 12.65 and four positions gained
  one full ply.

Conclusion: keep. The search tree and evaluation are identical; the change
buys more nodes and occasional extra completed depth for the same clock.

## 2026-07-11: 240-game attribution audit of the merged chain
Status: all merged stages CONFIRMED at 240g; nothing slipped through.

Motivation: several candidate ideas looked good at 60/120g and collapsed at
240g (selection bias on noisy gates), raising the question of whether the
already-merged chain suffered the same. Re-validated every seam at 240 games
(120 fresh positions, paired colors, fixed 120ms, seeds 20260716/17):

| Match | Result |
|---|---|
| current hce (58d89c0) vs session-start (efbdfd2) | +117.2, CI [+82.2, +154.6], P=100% (`current/cumulative_240g.json`) |
| LMP (e818841) vs session-start | +64.4, CI [+31.9, +98.1], P=100% (`current/ladder_lmp_vs_orig_240g.json`) |
| linear texel (44dc6b5) vs LMP | +51.0, CI [+18.3, +84.7], P=99.9% (`current/ladder_texel_vs_lmp_240g.json`) |
| joint PST (58d89c0) vs linear texel | +46.6, CI [+13.2, +80.9], P=99.7% (`current/ladder_pst_vs_texel_240g.json`) |
| Jul-6 session (efbdfd2) vs pre-session (fe88312) | +147.2, CI [+117.6, +179.0], P=100% (`current/ladder_jul6_vs_pre_240g.json`) |

Notes: every stage's CI floor is well above zero — all three merged stages are
individually real. Point estimates at 120g ran ~1.5-2x hot vs 240g (LMP +123.7
-> +64.4), confirming the inflation-of-gated-winners effect; stage sums
(+162) exceed the direct cumulative (+117) as expected for chained Elo. The
joint-PST block (1a9b951 + 42ec83c + b388411), previously undeployed, is
confirmed +46.6 at 240g and is clear to ship.

## 2026-07-09: Post-Texel search follow-up matrix
Status: all rejected or neutral; engine restored to the joint-Texel baseline.

After promoting the joint material/PST/scalar tune, six isolated search ideas
were tested at fixed 120ms with paired colors. Every candidate passed the 6/6
HCE position suite before match play:

- Pawn-keyed correction history applied only to shallow static-eval pruning:
  27.0/60, -34.9 Elo, P(better)=17.2%
  (`current/kimi_correction_60g.json`).
- Guarded non-PV ProbCut over SEE-safe captures/promotions: 28.0/60,
  -23.2 Elo, P=20.8% (`current/kimi_probcut_60g.json`).
- Conservative depth-1/2 razoring: 25.0/60, -58.5 Elo, P=3.6%
  (`current/kimi_razor_60g.json`).
- LMP exemption for quiets with strong learned history: 30.5/60, +5.8 Elo,
  P=57.3% (`current/kimi_history_lmp_60g.json`). Neutral; rejected.
- Removing the duplicate TT-generation increment in isolation: 25.5/60,
  -52.5 Elo, P=5.6% (`current/kimi_tt_generation_60g.json`). The existing
  two-step generation behavior remains.
- One-legal-evasion check extension: 24.0/60, -70.4 Elo, P=2.6%
  (`current/kimi_forced_evasion_60g.json`).

The correction, ProbCut, and razoring candidates all reduced representative
fixed-depth node counts, but that extra pruning lost playing strength. The
forced-evasion extension lost despite adding tactical depth. Conclusion: the
current search is tightly calibrated around LMP/check extensions; do not keep
adding generic pruning or extensions. Future substantial work should use a
larger, game-grouped evaluation dataset or a fundamentally different evaluator,
not more single-condition search guesses.

## 2026-07-09: Joint Texel tune of material, PSTs, and eval scalars
Status: committed on `hce-kimi` (`2859044`); not yet merged to `hce`.

Hypothesis: the first linear Texel tune held PSTs fixed, so material and mobility
weights inflated to absorb eval scale. Jointly tuning material, middlegame/endgame
PSTs, and eval-phase scalars should remove that artifact and add real strength.

Pipeline:
- `tunedump` extended to emit PST features per piece/square/color and to filter
  quiet positions via `hce_qsearch_eval_cp_stm` (`|static - qsearch| < 50`).
- `src/core/engine/hce_eval.c` gained separate endgame PST tables; the eval now
  blends `mg` and `eg` PST values with the existing phase interpolation.
- `scripts/texel_selfplay.py` generated ~1,000 120ms self-play games from 1,003
  equal positions with paired colors.
- Datasets: 2,424 `vidbot` lichess games (`current/vidbot_20260712.pgn`),
  27,771 positions; self-play 11,894 positions; combined 39,665 positions;
  25,663 quiet after filtering.
- `scripts/texel_tune.py` jointly tuned material + mg/eg PSTs + scalars with the
  Texel sigmoid (L2 toward defaults, pawn anchored at 100). Feature reconstruction
  verified exact (0 mismatches).
- `scripts/texel_apply_tune.py` writes the tuned `TUNED` line back into
  `hce_eval.c`.

Validation (fixed 120ms/move, lichess equal positions, vs prior HCE baseline
`03afd16`):
- 60g (seed 20260712): +46.6 Elo, CI95 [-11.6, +107.5], P(better)=93.7%
  (`current/pst_tune_60g.json`).
- 120g confirm (seed 20260714): +43.7 Elo, CI95 [-4.6, +93.7], P(better)=96.2%
  (`current/pst_tune_120g.json`).

Conclusion: keep and commit. The PST/material/scalar joint tune is a clear
confirmed gain and becomes the new HCE baseline on this branch.

## 2026-07-09: Texel-tuned linear eval weights
Status: MERGED into `hce`.

First systematic tune, after four hand-tuned single-term attempts went
neutral/negative. Pipeline (committed 13dd7d0): `tunedump` emits an exact linear
feature decomposition of the eval; `scripts/texel_build_dataset.py` turns 2,424
`vidbot` lichess games into 27,771 labelled quiet positions; `scripts/
texel_tune.py` fits the 21 linear weights (material + isolated/doubled +
mobility + rook-files) through the Texel sigmoid vs game result. Feature
reconstruction verified exact (0/27,771). Pawn anchored at 100; L2 toward
defaults.

Tuned: material q/n/b/r = 900/320/335/500 -> 1235/409/466/537 (pawn 100);
isolated -10/-15 -> -13/-16; doubled -12/-15 -> -17/-15; mobility per square
knight 3/2->8/4, bishop 2/3->9/4, rook 1/2->9/6, queen 1/1->9/2; rook_open
18/12->19/12, rook_semi 10/6->11/6.

Validation (120ms, vs LMP baseline `e818841`):
- 60g (seed 20260709): +82.6 Elo, P 98.9% (`current/texel_vs_lmp_60g.json`).
- 120g (seed 20260711, independent): +34.9 Elo, CI95 [-15.9, +87.1], P 91.1%
  (`current/texel_vs_lmp_120g.json`).
- Pooled 180g (independent seeds): 103.0/180 ~= +50 Elo, P ~= 97%.

Note: PST/king-safety/passers were held fixed (residual), so material/mobility
inflated to grab eval scale — a known partial-tuning artifact. It still wins;
the next step (joint material+PST texel) should remove the artifact and gain
more.

## 2026-07-09: Safe Mobility (exclude enemy-pawn squares)
Status: rejected; reverted.

Hypothesis: mobility counted squares controlled by enemy pawns as real
mobility, which is wrong. Masked knight/bishop/rook/queen mobility with
`~own_occ & ~enemy_pawn_attacks`.

Validation (120ms, vs LMP baseline `e818841`): 60g `-17.4` Elo, CI95
`[-78.0, +42.1]`, P(better) 28.3% (`current/mob_vs_lmp_60g.json`). Leans
negative; reverted.

Lesson: the fix is principled but the mobility weights were fit to the old
inflated counts, so masking without re-fitting the weights loses. Fourth
neutral-to-negative hand-tune since LMP -> the eval weights need *joint*
tuning (texel), not one-term-at-a-time edits.

## 2026-07-09: Quadratic King Safety
Status: rejected; reverted.

Hypothesis: the king-danger term is linear (`attack_units * 5`); a quadratic
ramp should better reflect that stacking attackers are worth more than their
sum. Replaced with `danger = min(units,30)^2 / 2` (x3/4 without an enemy queen),
tuned to match the old linear at ~10 units.

Validation (120ms, vs LMP baseline `e818841`): 60g `-52.5` Elo, CI95
`[-114.0, +5.9]`, P(better) 3.9% (`current/ksq_vs_lmp_60g.json`). Clearly worse
(9-33-18). Reverted, no confirm needed.

Lesson: king-safety calibration is very sensitive and a hand-picked quadratic
over-penalizes (engine turns timid). This term belongs in a systematic tuning
pass (texel), not a hand guess.

## 2026-07-09: 1-ply Continuation History
Status: rejected; parked on branch `hce-conthist` (`6387dfe`), not merged.

Hypothesis: the earlier flat-bonus countermove failed (`-17.4` pre-LMP) because
a single fixed bonus is too coarse. A graded continuation-history table, updated
with the same depth^2 bonus/malus as the butterfly history, should order quiet
replies better — and should compound with LMP, which prunes late quiets hard.

Change: `g_cont_hist[prev_piece][prev_to][cur_piece][cur_to]` (color-agnostic,
static, cleared each search), folded additively into quiet move ordering;
rewarded on quiet beta cutoff, penalized for the failed quiets at that node.

Validation (fixed 120ms/move, vs LMP baseline `e818841`):
- 60g: `+23.2` Elo, CI95 `[-45.5, +93.8]`, P 74.7% (`current/cm_vs_lmp_60g.json`).
- 120g: `+11.6` Elo, CI95 `[-33.3, +56.9]`, P 69.4% (`current/cm_vs_lmp_120g.json`).
- 240g, fresh seed: `+4.3` Elo, CI95 `[-29.2, +38.0]`, P 60.0%
  (`current/cm_vs_lmp_240g.json`).

Conclusion: the stronger form beats the flat bonus (sign flipped positive) but
the signal decays to zero with sample size (+23 -> +11.6 -> +4.3). Neutral at
this time control; not worth the 2.3MB table. Second confirmation (after LMR)
that squeezing the search further yields little here — the bottleneck is eval.

## 2026-07-09: Log-based LMR + Late Move Pruning
Status: MERGED into `hce`.

Hypothesis: the search under-prunes. The old LMR used an additive scheme
(base + depth-step + move-step) that topped out around 2-3 plies, and there was
no late-move pruning at all — every quiet move was searched at shallow depth.

Changes (`src/core/engine/hce_search.c`):
- Log-based LMR base table `r = 0.75 + ln(depth)*ln(move)/2.25`, precomputed
  once, replacing the additive base + step bonuses (history/recapture
  adjustments kept). Removed the now-dead `ctx_lmr_*` knob helpers.
- Late move pruning in `negamax`: skip quiet moves once
  `searched >= 3 + depth*depth` at `depth <= 8`, guarded by having a real
  `best_score` and never while in check or near mate.

Validation (fixed 120ms/move, lichess equal positions, seed 20260709):
- Build clean, no warnings. `make test` unchanged (pre-existing `test_ai`
  twofold-repetition failure present on clean `hce` too, unrelated).
- Depth sanity (middlegame FEN, 2s): baseline d13 -> LMR d14 -> LMR+LMP d16.
- LMR alone vs base: 120g `+2.9` Elo, CI95 `[-41.8, +47.7]`, P(better) 55.1%
  (`current/lmr_vs_hce_120g.json`) — neutral. LMR is harmless substrate.
- LMR+LMP vs base: 60g `+70.4` Elo, CI95 `[+1.2, +145.8]`, P 97.7%
  (`current/lmp_vs_hce_60g.json`).
- LMR+LMP vs base (confirm): 120g `+123.7` Elo, CI95 `[+74.6, +178.0]`,
  P(better) 100.0%, 59-43-18 (`current/lmp_vs_hce_120g.json`). Decisive.

Lesson: reshaping reductions (LMR) did nothing on its own here; the win came
entirely from *pruning* late quiets (LMP). Depth converts to strength once the
tree is actually narrowed.

## 2026-07-09: Capture History Ordering
Status: rejected; not merged.

Hypothesis: MVV/LVA capture ordering is static and cannot learn which captures
actually cause cutoffs in this search. Add a capture-history table keyed by
side, attacking piece, victim piece, and destination square; reward capture
beta cutoffs and penalize earlier failed captures.

Variant:
- `hce-capture-history` (`4ae5b1f`): adds capture-history scoring and update
  flow for main-search captures.

Validation:
- Build: `make uci` passed.
- `make hce_suite`: 6/6 passed.
- 60-game match vs current HCE:
  `current/baselines/capture_history_60g_20260709.json`
  scored 28.5/60 (`-17.4` Elo), CI95 `[-80.2, +44.3]`,
  `P(better)=29.0%`.

Conclusion: reject this shape. Static MVV/LVA plus SEE pruning is still better
for this engine than the simple capture-history bonus/malus tried here.

## 2026-07-09: Qsearch In-Check Eval Skip
Status: kept and merged (`305cfdc`).

Hypothesis: quiescence cannot legally stand pat while in check, but the code
still evaluated the position before generating evasions. Skip that unused eval
for a free speed win without changing the searched tree.

Change:
- Only compute `stand_pat` in `quiescence` when the side to move is not in
  check.

Validation:
- Build: `make uci` passed.
- Fixed-depth comparison on 10 FENs at depth 9: identical bestmove, score,
  nodes, and PV.
- Median NPS moved from 1,807,825 to 1,822,278 (`+0.8%`).
- `make hce_suite`: 6/6 passed.

Conclusion: keep. It is a tiny but node-identical speed cleanup, and there is
no chess-behavior downside.

## 2026-07-09: Piece-Square History
Status: rejected; not merged.

Hypothesis: current history only tracks `from -> to`, so it cannot distinguish
different pieces moving to the same square. Add a smaller
`piece_history[color][piece][to]` table alongside existing history and update it
with the same quiet beta-cutoff/malus flow.

Variant:
- `hce-piece-history` (`bdddb1c`): adds piece-to-destination history to quiet
  move ordering.

Validation:
- Build: `make uci` passed.
- 60-game match vs current HCE:
  `current/baselines/piece_history_60g_20260709.json`
  scored exactly 30.0/60 (`0.0` Elo), CI95 `[-63.4, +63.4]`,
  `P(better)=50.0%`.

Conclusion: reject as a default change. This simple additive history did not
show a signal; more advanced continuation history needs a better design than
just another quiet score term.

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
