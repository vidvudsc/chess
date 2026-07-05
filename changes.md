# Engine Autoresearch Log

Branch: `engine-strength/king-safety-quiet-checks`

Goal: improve the HCE search/eval toward much stronger Lichess bot play without using AI/NN backends. Keep only changes that survive tests and statistically useful matches. When a change creates massive blunders, inspect and understand before continuing.

## 2026-07-05 HCE Audit: Solid Opening Book Kept

- Full HCE pass reviewed search/eval/book surfaces for already-rejected ideas
  and concrete fast-control weaknesses.
- Did not reapply rejected search/eval ideas such as SEE capture ordering,
  alpha-raising quiet history, broader quiet checks, null-move loosening, or
  broad king-safety coefficient changes.
- Added reproducible solid-book generation:
  `scripts/build_solid_opening_book.py` and `make opening_book`.
- Generated and tracked `data/openings/opening_book.txt` from the existing
  100 curated opening seeds: kept 78 solid/stable lines and dropped 22 risky
  offbeat/gambit lines.
- UCI smoke test confirmed the book is used from `position startpos moves ...`,
  so it affects real bot/game flows beyond the first ply.
- Same-engine start-position match, broad seed book vs solid weighted book:
  `current/hce_testlab_solid_book_vs_broad_32g_20260705.json`.
  Solid book scored 18.0/32, estimated `+43.7 Elo`, CI95 `[-7.3, +96.5]`,
  `P(better)=95.3%`.
- Validation passed: `make opening_book`, `make hce_suite`, `make test`,
  `make bench`, `make arch`, `git diff --check`, Umbrel package build, and
  staged package HCE suite
  `current/hce_position_suite_staged_solid_book_20260705.json`.

## 2026-07-04 Dirty Tree Audit

- Reviewed the dirty worktree as an integrated project stack rather than
  treating it as unrelated noise.
- Confirmed that the active HCE baseline should remain Idea003:
  shallow quiet futility removed, MVV/LVA capture ordering preserved.
- Reverted the new SEE capture-ordering experiment because prior Idea012 and
  Idea031 evidence already rejected SEE-based main-search capture ordering.
- Reverted the guarded forward-futility experiment because prior Idea003
  evidence showed full removal of shallow quiet futility was the kept +41 Elo
  baseline over 120 games.
- Saved a fallback snapshot of the reverted variant at
  `current/engine_snapshots/hce_pre_realign_see_guarded_futility_20260704/chess_uci`.
- Updated the architecture audit to include untracked non-ignored files, so it
  reports the actual dirty tree during active development.
- Added bot time-budget tests for bullet, blitz increment, concurrency scaling,
  and low-clock panic behavior.
- Packaged and built the Umbrel staged release locally; staged binary passed
  the HCE position suite:
  `current/hce_position_suite_staged_20260704.json`.
- Checked Umbrel live logs and found the bot retrying a finished game stream
  with 404 every minute. Added a terminal-stream guard so `/api/bot/game/stream/*`
  stops retrying on 404/410 and the game thread can exit cleanly. Staged
  package after the fix passed:
  `current/hce_position_suite_staged_after_streamfix_20260704.json`.

## Protocol

- Run unit/regression tests before any gauntlet result is trusted.
- Prefer paired-color matches from equal FENs.
- Track W/L/D, score, Elo estimate/CI when the run completes, and whether the idea is kept.
- Stop early when a candidate produces obvious massive losses, then inspect the positions.

## Baseline Calibration

### HCE vs Stockfish Skill 0

- Candidate: current pre-autoresearch HCE
- Opponent: Stockfish 17.1, `Skill Level=0`
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: HCE 14W / 0L / 2D, 15.0/16
- Elo estimate from report: +470 vs opponent, CI approximately [+306, +2400]
- Report: `current/hce_vs_sf0_probe.json`
- Status: baseline evidence only

### HCE vs Stockfish UCI_Elo 1800

- Candidate: current pre-autoresearch HCE
- Opponent: Stockfish 17.1, `UCI_LimitStrength=true`, `UCI_Elo=1800`
- Settings: 20 games, paired positions, 120 ms/move, no book
- Result: HCE 9W / 8L / 3D, 10.5/20
- Elo estimate from report: +17 vs opponent, CI roughly [-133, +175]
- Report: `current/hce_vs_sf1800_probe.json`
- Status: baseline evidence only

### HCE vs Stockfish UCI_Elo 2000

- Candidate: current pre-autoresearch HCE
- Opponent: Stockfish 17.1, `UCI_LimitStrength=true`, `UCI_Elo=2000`
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: HCE 7W / 3L / 6D, 10.0/16
- Elo estimate from report: +89 vs opponent, CI roughly [-45, +257]
- Report: `current/hce_vs_sf2000_probe.json`
- Status: noisy baseline evidence only

## Idea 001: Broader King Safety + Quiet Checks

### Hypothesis

The engine overrates unsafe middlegames where its home-rank king is exposed and where quiet checking moves or mate threats matter. Add bounded quiet checks to quiescence and increase king-danger penalties for line batteries, low escape count, and central home-rank exposure.

### Code Areas

- `src/core/engine/hce_search.c`
- `src/core/engine/hce_eval.c`
- `tests/test_tactical_regressions.c`

### Targeted Evidence

- Added regression for FEN:
  `4rk1r/5p2/p1nq1p2/1p1p4/1bpP2b1/2N2N2/PPPQ1PPP/2KR1B1R w - - 0 17`
- Before: HCE often chose the calm `f1e2`, while Stockfish preferred defensive moves such as `d1e1`, `h2h4`, or `c1b1`.
- After first tuning: targeted regression passed; at 1.0s and 1.5s HCE stopped choosing `f1e2`.

### Unit Tests

- `make test`: PASS after implementation.

### Match Evidence

- Match: new worktree vs old local release binary
- Settings: intended 60 games, paired positions, 120 ms/move, no book
- Stopped early due to bad signal.
- Observed before stop: new 0W / 2L / 5D, 2.5/7
- Status: **not kept as-is**

### Reproduction

- Match: new worktree vs old local release binary
- Settings: 8 games, paired positions, 120 ms/move, no book
- Result: new 0W / 4L / 4D, 2.0/8
- Elo estimate from report: -191 vs old, CI roughly [-464, -45]
- Report: `current/new_vs_old_idea001_repro_8g.json`
- Status: **rejected**

### Massive Blunder / Loss Follow-Up

- New played the early king-walk `e1d2` from:
  `r1bqkb1r/1p3ppp/p1n2n2/2pp4/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 1 8`
- Stockfish and the old engine both preferred normal development; the new line was quickly punished.
- After reverting the broad king-safety tuning, the isolated quiet-check qsearch change still chose `e1d2`, so the quiet-check inclusion itself was unsafe.
- Final decision: reject/revert both the broad king-safety tuning and the blanket quiet-check qsearch idea. Future quiet-check search must be more selective, probably only for checking moves that are legal evasions from tactically stable roots or that pass SEE/king-safety gates.

## Idea 002: Larger Transposition Table

### Hypothesis

The engine searches hundreds of thousands to more than a million nodes per move at deployed time controls, but the TT is only `2^18` single-entry buckets. Increasing it to `2^20` should reduce replacement pressure and improve search stability without changing eval personality.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- `HCE_TT_BITS`: 18 -> 20

### Unit Tests

- `make test`: PASS after implementation.

### Match Evidence

- Match: new worktree vs old local release binary
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 6W / 6L / 48D, 30.0/60
- Elo estimate from report: +0 vs old, CI approximately [-40, +40]
- Report: `current/new_vs_old_idea002_tt20_60g.json`

### Status

- **Rejected / reverted**. The change was statistically flat in the first useful gauntlet and did not justify keeping a larger memory footprint.

### Massive Blunder / Loss Follow-Up

- Worst loss-audit swing came from game 39:
  `3r2k1/pp2qppp/2p2n2/2b5/2Q5/1PN1pP1B/PBP3PP/R6K w - - 4 22`
- Stockfish preferred moves such as `a1e1`, `c3e2`, and `h3f5`; both old and new chose `c3a4`, so this was not introduced by the TT-size change.
- Another earlier game-39 inaccuracy (`d1d3`) was also shared by old and new.
- Decision: reject for no measurable gain, but keep the game-39 positions as future tactical-regression candidates.

## Idea 003: Disable Shallow Quiet Futility Pruning

### Hypothesis

The search is missing sharp middlegame resources at 120 ms because low-depth quiet-move futility pruning drops defensive or attacking quiet moves after only a few ordered moves have been searched. The game-39 `c3a4` miss from Idea 002 is depth-sensitive and disappears around depth 8 or 200 ms, so removing this pruning may improve tactical reliability more than it costs in depth.

### Code Areas

- `src/core/engine/hce_search.c`
- `tests/test_tactical_regressions.c`

### Patch

- Removed the `depth <= 3`, `searched >= 4` quiet-move futility skip in `negamax`.
- Added a regression for:
  `3r2k1/pp2qppp/2p2n2/2b5/2Q5/1PN1pP1B/PBP3PP/R6K w - - 4 22`
  requiring the engine to avoid `c3a4` at 120 ms.

### Targeted Evidence

- Before: at 120 ms, both old and current HCE chose `c3a4`; Stockfish rated it roughly -650 cp worse than alternatives in the loss audit.
- After patch: HCE chooses `c3e2` at 40 ms, 120 ms, and fixed depths 2 through 8 on the target position.

### Unit Tests

- `./bin/test_tactical_regressions`: PASS after implementation.

### Bench

- `make bench`: PASS; no meaningful raw movegen regression observed.

### Match Evidence

- Smoke match: new worktree vs old local release binary
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 1W / 2L / 13D, 7.5/16
- Elo estimate from report: -22 vs old, CI approximately [-100, +54]
- Report: `current/new_vs_old_idea003_no_quiet_futility_smoke_16g.json`
- Loss audit: no new obvious early catastrophic blunder; largest early drops were from already-worse positions. Proceeding to the normal 60-game gauntlet because the targeted tactical fix is real and the smoke interval is wide.
- Main match: new worktree vs old local release binary
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 6W / 5L / 49D, 30.5/60
- Elo estimate from report: +6 vs old, CI approximately [-32, +44]
- Report: `current/new_vs_old_idea003_no_quiet_futility_60g.json`
- Loss audit: no clear new massive blunder pattern in the five losses; worst loss-conversion moves were mostly from already bad positions.
- Narrow variant tried: keep quiet futility at `depth <= 2` only. Rejected before match because the new tactical regression was not robust and could still choose `c3a4`.
- Confirmation match: new worktree vs old local release binary
- Settings: 120 games, paired positions, 120 ms/move, no book
- Result: new 23W / 9L / 88D, 67.0/120
- Elo estimate from report: +41 vs old, CI approximately [+9, +73]
- Probability better from report: 99.4%
- Report: `current/new_vs_old_idea003_no_quiet_futility_confirm_120g.json`
- Loss audit: no new single catastrophic pattern like Idea 001. The largest Stockfish deltas were mostly forced mate conversions or moves from already bad positions; some opening/middlegame losses remain useful future tactical-regression material.

### Status

- **Kept.** This becomes the current search baseline for the next idea.

## Elo Plot

- Script: `scripts/plot_engine_elo.py`
- Output: `current/engine_elo_by_idea.png`
- The plot shows each idea's Elo delta vs the old local release. Rejected/flat ideas are shown as background points with CI bars; the front line tracks the best observed Elo so far.

## Guardrail: Opening King-Walk Regression

- Added a permanent tactical regression for the `c8d7` king-walk failure found during Idea 008:
  `2kr1b1r/ppp1nppp/3qpnb1/3p2B1/6PN/2NP3P/PPP1PPB1/R2Q1RK1 b - - 6 11`
- This is a guardrail, not an Elo-bearing idea. It should make future search/eval experiments fail faster when they revive this blunder family.

## Idea 029: Retest Larger TT After Idea003

### Hypothesis

Idea 002 tested `HCE_TT_BITS=20` against the old local release and was flat. After Idea003 removed shallow quiet futility pruning, the search explores more quiet continuations and may put more pressure on the transposition table. Retest the larger TT against the kept Idea003 baseline.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- `HCE_TT_BITS`: `18 -> 20`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea029_tt20_retest_smoke_16g.json`
  - 16 games, 1W / 12D / 3L, 7.0/16.
  - Elo: -43.7 vs Idea003, CI95 [-135.0, +41.9], P(better)=15.9%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 8, 9, 11.
- Stockfish depth-11 sweep:
  - Game 8, `f6d5` from `r1q2rk1/pbp1p1bp/2nppnp1/6N1/3P4/1QPB4/1P3PPP/RNB2RK1 b - - 3 14`: about -224 cp.
  - Game 9, `g4g5` from `2kr3r/ppp2pp1/2n1pnp1/2qpb3/6P1/2NP3P/PPPBPPB1/R2QR1K1 w - - 4 14`: about -212 cp.
  - Game 11, `g1h1` from `r3r1k1/pp1n1ppp/2q5/1R3Q2/2p1p3/2P1P3/PP1P2PP/R1B3K1 w - - 6 24`: about -185 cp.
- Conclusion: after Idea003, larger TT still does not improve strength and appears to perturb tactical choices negatively.

### Status

- Rejected and reverted.

## Idea 030: Store Actual Fail-High Score In TT Lower Bounds

### Hypothesis

The search stores transposition-table lower bounds as the cutoff threshold `beta` at immediate fail-high exits. Storing the actual searched `score` is still a valid lower bound and could give later probes a tighter bound, improving move ordering or cutoffs without changing eval terms.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In both recursive and root beta cutoffs, changed `tt_store(..., beta, HCE_TT_LOWER, move)` to `tt_store(..., score, HCE_TT_LOWER, move)`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea030_tt_lower_score_smoke_16g.json`
  - 16 games, 1W / 11D / 4L, 6.5/16.
  - Elo: -65.9 vs Idea003, CI95 [-170.2, +27.5], P(better)=8.4%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1, 4, 9, 10.
- Stockfish depth-11 sweep found several meaningful early/midgame drops:
  - Game 9, `g2h3` from `2kr4/pppn1pp1/2n1p1p1/3p1qP1/8/2NPP2r/PPPB2B1/1R1QR2K w - - 0 21`: about -566 cp, but direct 120 ms repro showed Idea003 also chooses `g2h3`; this is a shared HCE weakness rather than candidate-specific.
  - Game 10, `c6e7` from `r2qk2r/ppp2ppp/2n1pnb1/3p4/1b4PN/3P3P/PPP1PPB1/RNBQ1RK1 b kq - 2 9`: about -250 cp. Direct 120 ms repro diverged: Idea003 reached depth 8 and chose `d8d6`, while this candidate reached depth 6 and chose `c6e7`.
  - Game 1, `d1b1` from `r3k2r/1p3ppp/p1n1bn2/q2p2B1/8/2P2N2/P1P1QPPP/3RKB1R w Kkq - 1 13`: about -228 cp; direct repro showed both engines choose it.
- Conclusion: storing stronger lower-bound scores perturbs the timed search negatively and does not produce a cleaner tactical profile.

### Status

- Rejected and reverted.

## Idea 031: Positive SEE Capture Ordering

### Hypothesis

Idea 012's SEE-aware capture ordering failed after demoting negative-SEE captures heavily. A safer variant may still help: leave all captures searchable in the current MVV/LVA order, but add only a modest bonus for captures that static exchange evaluation says are clearly profitable.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In `move_score`, for capture moves only, added `static_exchange_eval(s, m) * 8` when SEE is positive.
- No negative-SEE penalty and no pruning changes.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; no obvious global speed cliff, though timed tactical repros showed lower effective depth in some positions.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea031_see_positive_order_smoke_16g.json`
  - 16 games, 0W / 13D / 3L, 6.5/16.
  - Elo: -65.9 vs Idea003, CI95 [-141.0, +3.5], P(better)=3.1%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 9, 10, 11.
- Stockfish depth-11 sweep:
  - Game 10, `d8e8` from `3k2rr/pR3R1p/p3pp2/3p4/6Pn/4P2P/PP1N1P2/6K1 b - - 0 28`: mate collapse, about -99,406 cp. Direct 120 ms repro diverged: Idea003 chose `d8c8`, candidate chose `d8e8`.
  - Game 9, `c3c4` from `2k4r/pppn1pp1/2nqp1p1/1R1p2P1/7r/2PPB2P/P1P1PPB1/3Q1RK1 w - - 5 18`: about -407 cp. Direct repro diverged but both moves were unhappy.
  - Game 11, `d1e2` from `r4rk1/pp1n1ppp/2q5/4p3/2p1b1P1/N1P1R3/PP1P1P1P/R1BQ2K1 w - - 3 18`: about -306 cp.
- Conclusion: even a positive-only SEE ordering bonus perturbs timed search depth/order badly enough to lose tactical reliability.

### Status

- Rejected and reverted.

## Idea 032: Narrow HCE QSearch Delta Margin

### Hypothesis

The last two failed ideas suggest extra search cleverness can cost effective depth at 120 ms. Tightening the HCE-only qsearch delta margin from 120 cp to 100 cp may prune hopeless captures earlier, preserve depth, and reduce tactical volatility while keeping the NN backend unchanged.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In quiescence delta pruning, changed the HCE margin from `120` to `100`; kept the NN backend margin at `160`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea032_qdelta100_smoke_16g.json`
  - 16 games, 2W / 9D / 5L, 6.5/16.
  - Elo: -65.9 vs Idea003, CI95 [-194.8, +46.7], P(better)=12.6%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 7, 8, 10, 11, 13; candidate wins: games 6, 9.
- Stockfish depth-11 sweep:
  - Game 10, `b8c8` from `1k1q4/2r2pp1/4p1p1/1Q4P1/3P3r/2R4P/PP2PP2/4K3 b - - 1 42`: about -567 cp in-game, though direct 120 ms repro was unstable and both engines preferred `c7b7`.
  - Game 8, `f6d7` from `1r2q2k/p1p1p1bp/2npQnp1/8/8/2N5/1P3PPP/R1B1R1K1 b - - 0 23`: about -255 cp; direct repro diverged, with candidate choosing `b8b6`.
  - Game 7, `a5a7` from `5r1k/p1p1p2p/6p1/R7/1q1Ppr2/8/1P2QPPP/5R1K w - - 3 27`: about -232 cp.
- Conclusion: the narrower margin makes games sharper but not stronger; it introduces or fails to prevent enough tactical collapses to reject.

### Status

- Rejected and reverted.

## Idea 033: Mild Missing King-Shield Penalty

### Hypothesis

Idea 016 raised the missing king-shield pawn penalty from 10 cp to 28 cp. It fixed the audited `c1e3` queen-attack collapse but lost Elo badly over 60 games. A smaller coefficient may capture some broken-castled-shelter value without over-penalizing open kings.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- In `king_shield_penalty`, changed missing immediate shield pawn penalty from `10` to `14`.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.

### Targeted Diagnostic

- Direct 120 ms repro on the `c1e3` collapse FEN:
  `3qk2r/rpp2pp1/p2b2p1/3p4/N2n2P1/3P3P/PP2PP2/R1BQ1RK1 w k - 0 18`
  - Idea003 baseline: `c1e3`.
  - Candidate: `c1e3`.

### Status

- Rejected before match and reverted. Even the mild shield increase revives the `c8d7` guardrail and does not solve the intended tactical weakness.

## Idea 034: More Aggressive Reverse Futility Margin

### Hypothesis

Multiple failed ideas added work and lost effective depth. The remaining reverse-futility pruning may be conservative for the HCE backend. Lowering the HCE margin from `90 * depth` to `75 * depth` should prune more static fail-high nodes and possibly recover time without touching quiet-futility pruning.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In shallow reverse futility, changed the HCE margin from `90 * depth` to `75 * depth`; kept the NN backend margin at `110 * depth`.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.

### Bench

- `make bench`: PASS, but the failed guardrail rejects the idea before match play.

### Status

- Rejected before match and reverted. More aggressive static fail-high pruning revives the same `c8d7` king-walk family.

## Idea 035: Lower Previous Root-Order Bonus

### Hypothesis

The root previous-iteration ordering bonus is very large: `4,000,000 + previous_score`. This makes shallow previous root order dominate ordinary capture and promotion ordering in later iterations. Reducing the bonus may let obvious tactical root moves compete while still preserving some iterative-deepening stability.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In `previous_root_order_bonus`, changed the bonus from `4,000,000 + score` to `500,000 + score`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea035_root_bonus_low_smoke_16g.json`
  - 16 games, 1W / 10D / 5L, 6.0/16.
  - Elo: -88.7 vs Idea003, CI95 [-206.4, +11.4], P(better)=4.2%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1, 2, 4, 13, 14; candidate win: game 15.
- Stockfish depth-11 sweep:
  - Game 14, `e8e7` from `r1b1k2r/p1p2ppp/1qpb4/4p3/2P1N1n1/5NP1/PPQ1PP1P/R1B1K2R b KQkq - 4 12`: about -487 cp. Direct 120 ms repro diverged from the baseline: Idea003 chose `d6b4`, while the candidate preferred a king move family.
  - Game 4, `d5a2` from `5r1k/P1R4p/4p1p1/3qp3/3nN1Q1/3P2P1/P3PP1P/6K1 b - - 0 33`: about -510 cp, but direct repro showed both engines choose it.
  - Game 2, `e5c4` from `r5k1/5ppp/p2Nb3/2P1n3/8/P3Q1P1/2PK1P1P/q4B1R b - - 0 23`: about -275 cp.
- Conclusion: the large previous-root ordering bonus is apparently protective at this time control. Reducing it makes root behavior noisier and weaker.

### Status

- Rejected and reverted.

## Idea 036: Back-Rank King Flight Penalty

### Hypothesis

The `c1e3` audit suggests the engine undervalues creating a king flight square from a broken back-rank shelter. A targeted term may help without repeating the broad king-shield failures: penalize a back-rank king with an enemy queen and at least two missing immediate shield pawns when it has very few safe adjacent squares. A king move such as `Kg2` can escape the term.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Added a `back_rank_king_flight_penalty` term inside `king_safety_penalty`.
- Penalty only applied when:
  - king is still on its home rank,
  - enemy has a queen,
  - immediate shield penalty is at least 20 cp,
  - safe adjacent king squares are scarce.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Targeted Diagnostic

- Direct 120 ms repro on the `c1e3` collapse FEN:
  `3qk2r/rpp2pp1/p2b2p1/3p4/N2n2P1/3P3P/PP2PP2/R1BQ1RK1 w k - 0 18`
  - Idea003 baseline: `c1e3`.
  - Candidate: `c1e3`.
- The idea did not solve the intended diagnostic.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea036_backrank_flight_smoke_16g.json`
  - 16 games, 1W / 13D / 2L, 7.5/16.
  - Elo: -21.7 vs Idea003, CI95 [-99.7, +54.1], P(better)=28.6%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1 and 11; candidate win: game 6.
- Stockfish depth-11 sweep:
  - Game 11, `f1d1` from `2kr1b1r/pp3ppp/1n2p3/5b2/1Np5/2P5/PPNP1PPP/R1B2RK1 w - - 0 16`: about -366 cp. Direct 120 ms repro diverged: Idea003 chose `g1h1`, candidate chose `f1d1`.
  - Game 1, `f4g3` from `2r3k1/5ppp/p7/8/5B2/1K1b4/5qPP/7R w - - 0 35`: mate collapse from an already lost position; direct repro showed both engines choose it.
- Conclusion: the targeted back-rank term was not enough to fix the intended position and introduced at least one worse defensive choice.

### Status

- Rejected and reverted.

## Idea 037: Verify Root-Child Ply

### Hypothesis

The search has an unused `verification_plies` hook that disables reverse futility, null move, and LMR while `ply` is below the configured threshold. Enabling it only through the first reply ply may catch tactical replies to root moves without making the entire tree expensive.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Set `ctx.verification_plies = 2` at search start.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.

### Bench

- `make bench`: PASS, but the failed guardrail rejects the idea before match play.

### Status

- Rejected before match and reverted. Verifying the first reply ply revives the same `c8d7` king-walk family.

## Idea 028: Wider Classic Aspiration Window

### Hypothesis

At 120 ms, aspiration fail-low/high re-searches can waste the time budget when scores jump between depths. The classic backend currently uses a narrow `24 + depth * 6` cp window. Widen it to `40 + depth * 8` to reduce re-search churn while leaving eval, pruning, and move ordering otherwise unchanged.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Classic aspiration window: `24 + depth * 6 -> 40 + depth * 8`.
- NN backend window unchanged.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.
  - Wider aspiration changed search timing/control enough to revive the known pawn-race king-walk failure.

### Bench

- `make bench`: PASS, but irrelevant because the tactical gate failed.

### Match Evidence

- None. Rejected before match.

### Status

- Rejected and reverted.

## Idea 027: Interior Previous-PV Move Ordering

### Hypothesis

The search records `ctx->pv_move[ply]` when a move raises alpha, but only the root has explicit previous-iteration ordering through `previous_root_order_bonus`. At non-root plies, a previous PV move can be buried behind captures, killers, or history. Give the stored PV move a high non-TT ordering score so iterative deepening re-searches the previous principal variation first without changing eval or pruning.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In `move_score`, after the TT move check, return a high score for `ctx->pv_move[ply]`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS. Kiwipete fixed-depth timing was slower in this run, so watch match results carefully.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea027_pv_order_smoke_16g.json`
  - 16 games, 2W / 9D / 5L, 6.5/16.
  - Elo: -65.9 vs Idea003, CI95 [-194.8, +46.7], P(better)=12.6%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1, 3, 5, 13, 16.
- Stockfish depth-11 sweep showed the previous-PV ordering was too disruptive:
  - Game 16, `d8e7` from `r2q4/pp3p2/3p1R2/1PpP2Q1/3kr3/8/P1P3PP/1R4K1 b - - 0 31`: about -523 cp.
  - Game 5, `g2b7` from `r1b1q2k/pppp1ppp/8/5r2/1QP2P2/P2RP1P1/1P4BP/4K2R w K - 5 24`: about -487 cp.
  - Game 3, `g5h4` from `r3k2r/pppq2pp/2n1bp2/3np1B1/1b6/3P1NP1/PP1NPPBP/R2Q1RK1 w kq - 0 11`: about -217 cp.
- Conclusion: stale interior PV ordering changes the move tree too aggressively at 120 ms.

### Status

- Rejected and reverted.

## Idea 026: Stronger Hanging Piece Penalty

### Hypothesis

Loss audits repeatedly show queen/rook and loose-piece tactical collapses. The HCE has a hanging-piece term, but the base values are modest relative to material and positional bonuses. Increase the static penalty for attacked loose major/minor pieces to make the engine less willing to leave valuable pieces tactically exposed.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Hanging-piece base penalty:
  - Queen: `36 -> 48`
  - Rook: `18 -> 24`
  - Bishop/knight: `14 -> 18`

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea026_hanging_penalty_smoke_16g.json`
  - 16 games, 1W / 10D / 5L, 6.0/16.
  - Elo: -88.7 vs Idea003, CI95 [-206.4, +11.4], P(better)=4.2%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 3, 4, 7, 8, 16.
- Stockfish depth-11 sweep found major tactical damage:
  - Game 7, `g1h1` from `r4r1k/pbp1p1bp/4Q1p1/2q5/2B5/2P2P1n/1P1N2PP/R1B2RK1 w - - 7 23`: immediate mate collapse.
  - Game 16, `c8d7` from `r1b4k/ppq1pr1p/3p1np1/1BpPp3/4P3/2PQB2P/P1P2PP1/1R1R3K b - - 13 22`: about -398 cp, another king-walk style failure.
  - Game 3, `d1b3` from `r2qk2r/ppp3pp/4bp2/2bnp3/1n6/3P1NP1/PP1BPPBP/RN1QK2R w KQkq - 2 10`: about -295 cp.
- Conclusion: stronger static hanging penalties distort tactical choices instead of improving safety.

### Status

- Rejected and reverted.

## Idea 025: Lower Knight Outpost Bonus

### Hypothesis

The HCE gives protected, unchallengeable advanced knights a large `+42 mg / +22 eg` outpost bonus. Several failed experiments and audits show the engine can overvalue long-term compensation while missing immediate danger. Reduce this bonus moderately to make tactical safety and material/king state dominate a little more often.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Knight outpost bonus: `+42 mg / +22 eg -> +30 mg / +16 eg`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea025_outpost_low_smoke_16g.json`
  - 16 games, 2W / 12D / 2L, 8.0/16.
  - Elo: +0.0 vs Idea003, CI95 [-89.9, +89.9], P(better)=50.0%.
  - No 60-game run because the smoke was flat and the loss audit showed no safety improvement.

### Blunder Audit

- Candidate losses: games 1, 13.
- Stockfish depth-11 sweep:
  - Game 1, `f3d4` from `r3r1k1/1b3ppp/p7/8/1n1p4/3B1N2/qBP2PPP/2Q2K1R w - - 2 22`: about -247 cp.
  - Game 1, `d1b1` from `r3k2r/1p3ppp/p1n1bn2/q2p2B1/8/2P2N2/P1P1QPPP/3RKB1R w Kkq - 1 13`: about -228 cp.
  - Game 13, `g4g5` from `2r5/2p1n1pp/4kp2/2Rpp3/6P1/P4N2/KP3P1P/8 w - - 4 26`: about -119 cp.
- Conclusion: lowering outpost value was flat and did not produce a cleaner loss profile.

### Status

- Rejected and reverted.

## Idea 024: More Aggressive Classic Null-Move Reduction

### Hypothesis

Most wider-search ideas have failed by changing tactical preferences, but the engine may still be depth-limited at 120 ms. Increase the classic backend null-move reduction by one ply to trade some verification for speed. If the tactical guardrails pass, self-play will tell whether the extra depth is worth the selectivity risk.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Classic null-move reduction: `2 + depth / 4 -> 3 + depth / 4`.
- NN backend reduction unchanged.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea024_nullmove_r3_smoke_16g.json`
  - 16 games, 0W / 12D / 4L, 6.0/16.
  - Elo: -88.7 vs Idea003, CI95 [-176.8, -10.7], P(better)=1.3%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1, 4, 8, 12.
- Stockfish depth-11 sweep found selectivity damage, not merely late conversion:
  - Game 8, `d7d5` from `r4rk1/pbpqp1bp/5np1/1B2p3/8/2P5/1P2QPPP/RNB1R1K1 b - - 1 17`: about -428 cp.
  - Game 12, `d8b8` from `3rkb1r/p2qpppp/Bn6/8/2p5/N1P5/PP1PQPPP/2R2K1R b k - 2 16`: about -320 cp.
  - Game 12, `g4f3` from `r3kb1r/pp2pppp/1nnq4/8/2p1B1b1/N1P2N2/PP1P1PPP/R1BQK2R b KQkq - 3 10`: about -211 cp.
- Conclusion: the extra null-move reduction saves nodes but misses too much tactics at 120 ms.

### Status

- Rejected and reverted.

## Idea 023: Lower Bishop Pair Bonus

### Hypothesis

Raising bishop-pair value revived a king-walk/pawn-race blunder, and the earlier broad bishop/knight rebalance did not survive. Test the opposite stabilization: reduce the bishop-pair bonus slightly so the engine is less willing to lean on long-term bishop compensation in sharp or race-like positions.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Bishop pair bonus: `+24 mg / +28 eg -> +16 mg / +20 eg`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea023_bishop_pair_low_smoke_16g.json`
  - 16 games, 1W / 12D / 3L, 7.0/16.
  - Elo: -43.7 vs Idea003, CI95 [-135.0, +41.9], P(better)=15.9%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1, 3, 16.
- Stockfish depth-11 sweep found early tactical deterioration:
  - Game 3, `d1b3` from `r2qk2r/ppp3pp/4bp2/2bnp3/1n6/3P1NP1/PP1BPPBP/RN1QK2R w KQkq - 2 10`: about -297 cp.
  - Game 1, `d1b1` from `r3k2r/1p3ppp/p1n1bn2/q2p2B1/8/2P2N2/P1P1QPPP/3RKB1R w Kkq - 1 13`: about -228 cp.
  - Game 16, `e8f8` from `r3k2r/ppNnppbp/3p1np1/3P4/1qp1P3/3Q1N2/PPPB1PPP/R3K2R b KQkq - 1 11`: about -202 cp.
- Conclusion: lowering bishop pair does not stabilize; it weakens compensation enough to choose worse tactical continuations.

### Status

- Rejected and reverted.

## Idea 022: Stronger Bishop Pair Bonus

### Hypothesis

The broad bishop/knight material rebalance had a strong positive smoke but failed the 60-game gate. A narrower version may capture the useful part: reward the two-bishop long-term advantage without making every bishop trade and every knight comparison different. Increase only the bishop-pair term.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Bishop pair bonus: `+24 mg / +28 eg -> +36 mg / +42 eg`.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.
  - The stronger bishop pair bonus is enough to revive the known king-walk/pawn-race blunder.

### Bench

- `make bench`: PASS, but irrelevant because the tactical gate failed.

### Match Evidence

- None. Rejected before match.

### Status

- Rejected and reverted.

## Idea 021: Do Not LMR Extended Quiet Checks

### Hypothesis

`search_move_extension` gives checking moves one extra ply, but late quiet checking moves can still enter the LMR branch and lose that depth immediately. Since blanket quiet checks in qsearch were unsafe, keep the search tree conservative, but make the existing check extension meaningful by skipping LMR when the quiet move already earned an extension.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Added `extension == 0` to the quiet LMR condition, so quiet checks with a one-ply extension are searched unreduced.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.
  - Allowing quiet checking extensions to bypass LMR revived the known pawn-race king-walk failure.

### Bench

- `make bench`: PASS, but irrelevant because the tactical gate failed.

### Match Evidence

- None. Rejected before match.

### Status

- Rejected and reverted.

## Idea 020: Raise Tempo Bonus to 16 cp

### Hypothesis

Idea 019 showed `+8 cp` tempo is exactly flat over 60 games. Test the other direction: a stronger side-to-move bonus may improve initiative, qsearch stand-pat choices, and tactical conversion. This is a narrow coefficient tune and should be rejected quickly if it creates tactical overreach.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Tempo bonus: `+12 cp -> +16 cp`.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.
  - Increasing tempo to 16 cp re-enabled the known pawn-race king-walk failure.

### Bench

- `make bench`: PASS, but irrelevant because the tactical gate failed.

### Match Evidence

- None. Rejected before match.

### Status

- Rejected and reverted.

## Idea 019: Lower Tempo Bonus to 8 cp

### Hypothesis

The HCE eval adds a fixed 12 cp side-to-move tempo. That value feeds static eval, qsearch stand-pat, futility, and move choice. Several failed ideas showed the engine can over-trust initiative and enter tactical danger; a slightly smaller tempo bonus may make shallow evaluations less optimistic without broad king-safety distortion.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Tempo bonus: `+12 cp -> +8 cp`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea019_tempo8_smoke_16g.json`
  - 16 games, 2W / 13D / 1L, 8.5/16.
  - Elo: +21.7 vs Idea003, CI95 [-54.1, +99.7], P(better)=71.4%.
- Main run vs kept Idea003 baseline: `current/new_vs_idea003_idea019_tempo8_60g.json`
  - 60 games, 7W / 46D / 7L, 30.0/60.
  - Elo: +0.0 vs Idea003, CI95 [-43.0, +43.0], P(better)=50.0%.

### Blunder Audit

- Candidate loss: game 6.
- Stockfish depth-11 sweep did not show an instant opening catastrophe. The candidate gradually entered a worse position; notable drops:
  - `g7g6` from `r1b3k1/ppppqppp/2n4r/2b3N1/2P1BP2/P3P3/1PQ3PP/R1BR3K b - - 4 20`: about -226 cp.
  - `c5d6` from `r1bn2k1/ppppqp1p/6pr/2bR2N1/2P1BP2/P1Q1P3/1P4PP/R1B4K b - - 3 22`: about -224 cp.
  - `d8e7` from `r1bq2k1/pppp1ppp/2n1r3/2b5/2P5/P1QBPN2/1P3PPP/R1B2R1K b - - 7 15`: about -205 cp.
- No smoke-level massive blunder; proceed to full gate.
- Full-run losses: games 17, 32, 37, 39, 42, 53, 54.
- Largest full-run tactical drops included:
  - Game 37, `d1b3` from `r3k2r/pbqp2p1/1pn5/5p1p/6nB/P2P1N2/4BPPP/R2Q1RK1 w kq - 0 16`: about -480 cp.
  - Game 39, `e2d3` from `3rr1k1/pp3ppp/2pq1n2/8/4Pp2/1P1n1P2/PBPPQ1PP/1R1K2R1 w - - 0 20`: about -473 cp.
  - Game 54, `e6a2` from `r3kb1r/pp3ppp/1qn1b3/2p1p3/4Q3/2PP1NP1/P3PP1P/R1B1KB1R b KQkq - 8 12`: about -319 cp.
- The full result is exactly flat; no reason to keep this coefficient.

### Status

- Rejected and reverted.

## Idea 018: Tiny Repetition Contempt

### Hypothesis

The gauntlets are dominated by threefold draws. Some are legitimate equal positions, but others look like the engine accepting harmless loops instead of continuing to press. Score terminal repetition draws as a tiny positive value for the side to move in the terminal node. Through negamax, the side choosing the repetition sees it as slightly unattractive, while normal stalemate/50-move/insufficient draws remain neutral.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Added `HCE_REPETITION_CONTEMPT_CP = 8`.
- `score_terminal_stm` now returns `+8` for `GAME_RESULT_DRAW_REPETITION` only.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea018_repetition_contempt_smoke_16g.json`
  - 16 games, 0W / 13D / 3L, 6.5/16.
  - Elo: -65.9 vs Idea003, CI95 [-141.0, +3.5], P(better)=3.1%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 1, 5, 13.
- Stockfish depth-11 sweep shows contempt caused risk-taking without useful conversion:
  - Game 5, `d5f6` from `r1b1r1k1/pppp1ppp/2nq1n2/2bNp3/2P5/P2PP3/1P3PPP/R1BQKBNR w KQ - 7 11`: about -286 cp from a good position.
  - Game 5, `c4d5` from `r1b4k/pppp4/3q2p1/2bnr3/2P5/P3P3/1PQ2PPP/R1BR2K1 w - - 0 21`: about -284 cp.
  - Game 13, `g8e8` from `6R1/r7/2R5/4p2p/2P1P1bk/8/P2P3P/7K w - - 1 59`: about -982 cp.
- Conclusion: tiny contempt did not eliminate repetitions in the smoke and converted some safe draws into losses.

### Status

- Rejected and reverted.

## Idea 017: Order Safe King Escapes Earlier

### Hypothesis

The `c1e3` queen-attack collapse appears partly time/order related: Stockfish prefers `g1g2`, but the engine often spends the final iteration on non-king quiets. Instead of broadly changing king-safety eval, give safe non-castling king moves a move-ordering bonus only when the king is still near home, has at least two missing immediate shield pawns, and the enemy has queen/rook material. This should help time-limited searches see practical king escapes without encouraging random king walks.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In quiet move scoring, add a bounded ordering bonus for safe king escape moves in exposed near-home king positions.

### Unit Tests

- Diagnostic gate failed before formal unit tests:
  - At `3qk2r/rpp2pp1/p2b2p1/3p4/N2n2P1/3P3P/PP2PP2/R1BQ1RK1 w k - 0 18`, the engine still chose `c1e3` at 120 ms even with a very large safe-king-move ordering bonus.

### Bench

- Not run; rejected before bench.

### Match Evidence

- None. Rejected before match.

### Status

- Rejected and reverted.

## Idea 016: Stronger Missing King-Shield Pawn Penalty

### Hypothesis

The rook-on-7th audit exposed a real kept-engine weakness: from `3qk2r/rpp2pp1/p2b2p1/3p4/N2n2P1/3P3P/PP2PP2/R1BQ1RK1 w k - 0 18`, the engine plays `c1e3` into a queen attack, while Stockfish strongly prefers `g1g2` to create king flight. The current shield penalty charges only 10 cp per missing pawn directly in front of the king, which may be too soft for opened castled positions. Increase that local shelter penalty and gate it on the new tactical regression before any match.

### Code Areas

- `src/core/engine/hce_eval.c`
- `tests/test_tactical_regressions.c`

### Patch

- Increased missing immediate king-shield pawn penalty from 10 cp to 28 cp.
- Added a tactical regression for the audited queen attack collapse, requiring the engine to avoid `c1e3`.

### Unit Tests

- `make test`: PASS after implementation, including the new `c1e3` guardrail.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea016_king_shield28_smoke_16g.json`
  - 16 games, 0W / 16D / 0L, 8.0/16.
  - Elo: +0.0 vs Idea003, P(better)=50.0%.
  - No losses, so no blunder audit required.
- Main run vs kept Idea003 baseline: `current/new_vs_idea003_idea016_king_shield28_60g.json`
  - 60 games, 5W / 45D / 10L, 27.5/60.
  - Elo: -29.0 vs Idea003, CI95 [-73.7, +14.8], P(better)=9.7%.

### Blunder Audit

- Candidate losses: games 3, 6, 12, 14, 20, 24, 32, 37, 46, 47.
- Stockfish depth-11 sweep found several large tactical drops, suggesting the broader king-shield coefficient distorts too many positions:
  - Game 12, `b6c8` from `1k3b1r/pp2p1pp/1n2p3/1Qp1rqN1/6R1/5P2/PP1P1P1P/R1B2K2 b - - 1 23`: about -694 cp.
  - Game 12, `h8g8` from `1kn1Qb1r/pp2p1pp/4p3/2p1rqN1/6R1/5P2/PP1P1P1P/R1B2K2 b - - 3 24`: about -658 cp.
  - Game 6, `f6h6` from `1rb4k/pp1p1ppp/3p1r2/3PP3/q1PR4/P3Q3/1P2BPPP/2K4R b - - 0 26`: about -523 cp.
- The new `c1e3` guardrail was useful diagnostically, but the coefficient fix costs Elo and is reverted. A more targeted king-danger/queen-access fix may be worth trying later.

### Status

- Rejected and reverted.

## Idea 015: Rook on the Seventh Rank Bonus

### Hypothesis

The engine draws many rook and late-middlegame positions by repetition. A modest classic HCE bonus for a rook on the 7th rank, only when the enemy king is trapped near home or enemy pawns are on that rank, should improve conversion pressure without broadly overvaluing random rook lifts.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Added `rook_on_seventh_target`.
- Award +22 mg / +34 eg through the existing rook-file/activity term when a rook is on the 7th rank and has enemy king or pawn targets.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea015_rook7_smoke_16g.json`
  - 16 games, 1W / 14D / 1L, 8.0/16.
  - Elo: +0.0 vs Idea003, CI95 [-62.8, +62.8], P(better)=50.0%.
  - No 60-game run because the only candidate loss was tactically ugly and the score showed no upside.

### Blunder Audit

- Candidate loss: game 9.
- Stockfish depth-12 sweep:
  - `g2d5` from `3qk2r/rpp2pp1/p2bp1p1/3n4/N2n2P1/3P3P/PP2PPB1/R1BQ1RK1 w k - 0 17`: about -300 cp.
  - `c1e3` from `3qk2r/rpp2pp1/p2b2p1/3p4/N2n2P1/3P3P/PP2PP2/R1BQ1RK1 w k - 0 18`: mate collapse, about -99830 cp.
- The rook-on-7th bonus was not directly active in that loss, but the eval nudge perturbed the tree into a forced mate line without showing positive score evidence.

### Status

- Rejected and reverted.

## Idea 014: Reverse Futility Only Through Depth 2

### Hypothesis

Full removal of reverse futility failed, but it may have been too expensive and noisy. A narrower setting that keeps the static beta cutoff at depths 1-2 while disabling only the depth-3 cutoff could reduce the worst optimistic static-eval misses without blowing up the tree as much.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Changed reverse futility pruning from `depth <= 3` to `depth <= 2`.

### Unit Tests

- `make test`: FAIL.
  - Failed permanent regression: `Avoid losing king walk c8d7 in pawn-race defense`.
  - The engine revived the known king-walk failure from `2k5/1p6/6p1/2K1P3/8/6P1/8/8 b - - 0 1`.

### Bench

- `make bench`: PASS, but irrelevant because the tactical gate failed.

### Match Evidence

- None. Rejected before match.

### Status

- Rejected and reverted.

## Idea 013: Disable Shallow Reverse Futility Pruning

### Hypothesis

The only kept gain so far came from removing shallow quiet futility pruning. The sibling pruning path still returns `beta` at depth <=3 from static evaluation alone. Disabling this reverse futility shortcut may preserve tactical verification in positions where HCE static eval is optimistic, at the cost of extra nodes.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Removed the `depth <= 3` static-eval beta cutoff before null-move pruning in `negamax`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea013_no_reverse_futility_smoke_16g.json`
  - 16 games, 0W / 13D / 3L, 6.5/16.
  - Elo: -65.9 vs Idea003, CI95 [-141.0, +3.5], P(better)=3.1%.
  - Failed smoke; no 60-game run.

### Blunder Audit

- Candidate losses: games 7, 13, 16.
- Stockfish depth-11 sweep found real tactical damage:
  - Game 7, `c7f4` from `7k/2Q1prbp/p3Nn2/3p1R2/3P4/1qP5/1P4PP/2B4K w - - 1 33`: about -539 cp.
  - Game 16, `g8f8` from `2r1r1k1/pp2pp2/q2p1n2/2pP2Q1/4P3/8/PPP2PPP/1R1R2K1 b - - 0 20`: about -452 cp.
  - Game 13, `b2b4` from `r5k1/p4ppp/1bN1rn2/4N3/4p3/6P1/PP3PKP/3R3R w - - 7 23`: about -263 cp.
- Conclusion: reverse futility pruning looks ugly in theory, but removing it made the engine tactically worse in the smoke sample.

### Status

- Rejected and reverted.

## Idea 012: SEE-Aware Capture Ordering

### Hypothesis

Main search orders captures with MVV-LVA only, so even obviously losing captures are searched before nearly every quiet move. QSearch already uses SEE to prune bad captures. Reusing the lightweight SEE in main move ordering should push losing captures below good captures and tactical quiets, improving node allocation without changing evaluation.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In `move_score`, compute `static_exchange_eval` for captures.
- If SEE is negative, subtract `950000` from the capture ordering score; otherwise add the SEE value.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; small expected overhead from SEE during capture scoring, no raw movegen failure.

### Match Evidence

- Smoke vs kept Idea003 baseline: `current/new_vs_idea003_idea012_see_capture_order_smoke_16g.json`
  - 16 games, 0W / 16D / 0L, 8.0/16.
  - Elo: +0.0 vs Idea003, P(better)=50.0%.
  - No losses, so no blunder audit required.
- Main run vs kept Idea003 baseline: `current/new_vs_idea003_idea012_see_capture_order_60g.json`
  - 60 games, 6W / 45D / 9L, 28.5/60.
  - Elo: -17.4 vs Idea003, CI95 [-62.1, +26.8], P(better)=22.0%.

### Blunder Audit

- Candidate losses: games 2, 4, 19, 31, 32, 35, 38, 47, 50.
- Stockfish depth-11 sweep found no single repeated opening catastrophe. Largest meaningful midgame drops were:
  - Game 19, `d7c8` from `r1r4k/pp1B1pbp/1q1p2p1/2p1P3/5P2/2n2Q2/PP1B1n1P/R3R1K1 w - - 0 24`: about -424 cp.
  - Game 4, `d6e6` from `1k1r3r/ppp2ppp/2nq4/3b2Q1/4p3/2PP2P1/P3PPBP/RR2N1K1 b - - 1 16`: about -421 cp.
  - Game 50, `f7g6` from `3r3r/pp3kpp/3b1pN1/2pPq3/2P1n1P1/4B3/PP2Q1BP/R2K3R b - - 0 19`: about -325 cp.
- Conclusion: no useful gain and a worse loss profile; SEE capture ordering appears to spend ordering precision in the wrong places for this search.

### Status

- Rejected and reverted.

## Idea 004: Milder Null-Move Pruning

### Hypothesis

The kept Idea 003 improved strength by searching more quiet defensive resources. Null-move pruning may still be too aggressive in sharp non-check positions, especially with the HCE eval. Reducing the classic backend null-move reduction from `2 + depth / 4` to `1 + depth / 4` should preserve most pruning while catching more tactics and zugzwang-adjacent danger.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Null-move reduction: `(classic ? 2 : 1) + depth / 4` -> `1 + depth / 4`

### Unit Tests

- `make test`: FAIL. The tactical regression suite brought back the known
  `c8d7` pawn-race king-walk blunder in process-level testing.

### Match Evidence

- Not run. Rejected before match because a known massive-blunder regression failed.

### Status

- **Rejected / reverted.**

## Idea 006: Ignore Very Shallow TT Moves For Ordering

### Hypothesis

Some bad moves appear in game-context searches but not from a fresh FEN, suggesting a persistent TT move-ordering effect. The TT probe currently returns a stored move for ordering even when the stored entry is shallower than the requested search. Very shallow depth-0/1 moves can be stale or tactical-noisy. Only using TT moves for ordering once the stored entry has at least depth 2 may reduce process-state blunders without disabling useful TT cutoffs.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- `tt_probe` now only writes `move_out` for ordering if the matching entry has `entry->depth >= 2`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; no meaningful raw movegen regression observed.

### Match Evidence

- Smoke match: new worktree vs Idea 003 baseline
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 1W / 2L / 13D, 7.5/16
- Elo estimate from report: -22 vs Idea 003 baseline, CI approximately [-100, +54]
- Report: `current/new_vs_idea003_idea006_ttmove_depth2_smoke_16g.json`
- Loss audit: early bad moves such as `e6f5`, `c1g5`, `f3d2`, and `g2c6` were shared by baseline and candidate on direct repro, so no candidate-specific massive blunder was found.
- Main match: new worktree vs Idea 003 baseline
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 6W / 6L / 48D, 30.0/60
- Elo estimate from report: +0 vs Idea 003 baseline, CI approximately [-40, +40]
- Report: `current/new_vs_idea003_idea006_ttmove_depth2_60g.json`

### Status

- **Rejected / reverted.** Flat result; no measurable gain from filtering very shallow TT ordering moves.

## Idea 007: Delay LMR Until Depth 4

### Hypothesis

Idea 003 gained strength by avoiding shallow pruning of quiet moves. The remaining late-move reduction still reduces quiet moves starting at depth 3 after only two searched moves. Delaying LMR until depth 4 may catch more shallow tactical resources while keeping deeper move-count pruning intact.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Quiet late-move reduction starts at `depth >= 4` instead of `depth >= 3`.

### Unit Tests

- `make test`: FAIL. The tactical regression suite brought back the known
  `c8d7` pawn-race king-walk blunder.

### Match Evidence

- Not run. Rejected before match because a known massive-blunder regression failed.

### Status

- **Rejected / reverted.**

### Massive Blunder / Loss Follow-Up

- Failing regression:
  `2k5/5pB1/8/1P3P1P/3NK3/1P6/8/8 b - - 2 49`
- This is the same tripwire that rejected Idea 004. Reducing shallow selectivity in the wrong place can re-enable a king-walk failure in pawn-race defense.
- Decision: reject immediately; do not gauntlet.

## Idea 008: Early Queen Development Penalty

### Hypothesis

Several shared loss motifs involve early queen moves while multiple minor pieces remain undeveloped. Add a small middlegame-only penalty for a queen that has left its home square while at least two home minors are still undeveloped. This should discourage fragile queen sorties without changing normal middlegame queen activity.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Added `undeveloped_minor_count`.
- If phase is high (`>= 18`), queen is away from home, and at least two home minors are undeveloped, apply `-8 cp` per undeveloped minor in the middlegame piece-square term.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; no meaningful raw movegen regression observed.

### Match Evidence

- Smoke match: new worktree vs Idea 003 baseline
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 0W / 2L / 14D, 7.0/16
- Elo estimate from report: -44 vs Idea 003 baseline, CI approximately [-105, +15]
- Report: `current/new_vs_idea003_idea008_queen_dev_smoke_16g.json`
- Loss audit: game 10 included `c8d7` from
  `2kr1b1r/ppp1nppp/3qpnb1/3p2B1/6PN/2NP3P/PPP1PPB1/R2Q1RK1 b - - 6 11`,
  a king-walk family blunder. This was enough to reject despite the unit regression suite passing.

### Status

- **Rejected / reverted.**

### Massive Blunder / Loss Follow-Up

- The queen-development penalty did not fail the fixed unit tripwire, but self-play found a related king-walk failure. Penalizing early queen movement changed the candidate's defensive preferences in sharp openings in a harmful way.
- Decision: reject immediately after smoke; do not gauntlet.

## Idea 009: Advanced Pawn Push Extension

### Hypothesis

Pawn races and advanced passers are recurring sources of decisive wins/losses. Search already extends checks, promotions, and recaptures, but a pawn push to the 6th or 7th rank can be promotion-critical before it is technically a promotion. Add a narrow one-ply extension for pawn moves reaching advance rank 5 or higher.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- In `search_move_extension`, extend pawn moves that reach the 6th/7th rank for the mover.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; some expected search overhead, no raw movegen failure.

### Match Evidence

- Smoke match: new worktree vs Idea 003 baseline
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 1W / 1L / 14D, 8.0/16
- Elo estimate from report: +0 vs Idea 003 baseline, CI approximately [-63, +63]
- Report: `current/new_vs_idea003_idea009_pawn_extend_smoke_16g.json`
- Main match: new worktree vs Idea 003 baseline
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 3W / 8L / 49D, 27.5/60
- Elo estimate from report: -29 vs Idea 003 baseline, CI approximately [-67, +8]
- Probability better from report: 6.4%
- Report: `current/new_vs_idea003_idea009_pawn_extend_60g.json`

### Status

- **Rejected / reverted.** The advanced-pawn extension was harmful in the first full gauntlet.

## Idea 010: Do Not Penalize Alpha-Raising Quiet Moves

### Hypothesis

The history heuristic records quiet moves searched before a later beta cutoff as failed quiets. Currently a quiet move that improves alpha but does not immediately cut off can still be added to `failed_quiets` and later penalized. That can punish useful PV-like quiet moves. Only quiet moves whose score fails to beat the pre-move alpha should receive the failed-quiet malus.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- Save `alpha_before_move` before searching each move.
- Add a quiet move to `failed_quiets` only when `score <= alpha_before_move`.

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; no meaningful raw movegen regression observed.

### Match Evidence

- Smoke match: new worktree vs Idea 003 baseline
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 2W / 1L / 13D, 8.5/16
- Elo estimate from report: +22 vs Idea 003 baseline, CI approximately [-54, +100]
- Report: `current/new_vs_idea003_idea010_history_alpha_smoke_16g.json`
- Loss audit: the single loss did not show a candidate-specific instant disaster; the largest move drop came from an already-lost position.
- Main match: new worktree vs Idea 003 baseline
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 7W / 8L / 45D, 29.5/60
- Elo estimate from report: -6 vs Idea 003 baseline, CI approximately [-50, +39]
- Report: `current/new_vs_idea003_idea010_history_alpha_60g.json`

### Status

- **Rejected / reverted.** The positive smoke did not survive the full gauntlet.

## Idea 011: Bishop/Knight Value Rebalance

### Hypothesis

The current HCE material values prefer knights over bishops (`N=335`, `B=320`). With separate outpost bonuses and bishop-pair bonuses already present, this may overvalue knights and undervalue ordinary bishop activity in open positions. Try a conservative classical rebalance: `N=325`, `B=330`.

### Code Areas

- `src/core/engine/hce_eval.c`

### Patch

- Knight value: `335 -> 325`
- Bishop value: `320 -> 330`

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; no meaningful raw movegen regression observed.

### Match Evidence

- Smoke match: new worktree vs Idea 003 baseline
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 4W / 1L / 11D, 9.5/16
- Elo estimate from report: +66 vs Idea 003 baseline, CI approximately [-28, +170]
- Report: `current/new_vs_idea003_idea011_bishop_knight_smoke_16g.json`
- Loss audit: no giant early candidate-specific blunder; the largest drops came in a late rook/pawn endgame.
- Main match: new worktree vs Idea 003 baseline
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 6W / 8L / 46D, 29.0/60
- Elo estimate from report: -12 vs Idea 003 baseline, CI approximately [-55, +31]
- Report: `current/new_vs_idea003_idea011_bishop_knight_60g.json`

### Status

- **Rejected / reverted.** The strong smoke did not survive the full gauntlet.

## Idea 005: Wider QSearch Delta Margin

### Hypothesis

After removing shallow quiet futility, the remaining tactical blind spots may come from quiescence pruning borderline captures too narrowly. Increasing the delta margin makes qsearch search more plausible captures without reintroducing quiet checks, which previously caused massive blunders.

### Code Areas

- `src/core/engine/hce_search.c`

### Patch

- QSearch delta margin: HCE `120 -> 180`, NN `160 -> 200`

### Unit Tests

- `make test`: PASS after implementation.

### Bench

- `make bench`: PASS; no meaningful raw movegen regression observed.

### Match Evidence

- Smoke match: new worktree vs Idea 003 baseline
- Settings: 16 games, paired positions, 120 ms/move, no book
- Result: new 1W / 2L / 13D, 7.5/16
- Elo estimate from report: -22 vs Idea 003 baseline, CI approximately [-100, +54]
- Report: `current/new_vs_idea003_idea005_qdelta_smoke_16g.json`
- Loss audit: the suspicious game-9 middlegame moves were shared by the Idea 003 baseline and the Idea 005 candidate on direct repro, so no candidate-specific massive blunder was found.
- Main match: new worktree vs Idea 003 baseline
- Settings: 60 games, paired positions, 120 ms/move, no book
- Result: new 10W / 5L / 45D, 32.5/60
- Elo estimate from report: +29 vs Idea 003 baseline, CI approximately [-15, +74]
- Report: `current/new_vs_idea003_idea005_qdelta_60g.json`
- Loss audit: no new single catastrophic pattern. The biggest early swings checked against the Idea 003 baseline were mostly shared, including `g5h7`, `b4d3`, and `f6d5` motifs.
- Confirmation match: new worktree vs Idea 003 baseline
- Settings: 120 games, paired positions, 120 ms/move, no book
- Result: new 9W / 11L / 100D, 59.0/120
- Elo estimate from report: -6 vs Idea 003 baseline, CI approximately [-31, +20]
- Probability better from report: 32.8%
- Report: `current/new_vs_idea003_idea005_qdelta_confirm_120g.json`
- Decision: the 60-game positive was a false signal; the longer confirmation did not improve on Idea 003.

### Status

- **Rejected / reverted.**
