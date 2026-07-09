# HCE — Next Steps & Handoff

Snapshot after the 2026-07-09 session. Goal: push the HCE engine from ~2000
lichess toward 2500. This session banked ~+160 self-play Elo (see bottom).

## 1. Current state

- Working branch: `hce`. Work happens in the git worktree
  `current/worktrees/hce_2500` (keeps the `nn` main tree and the shared
  `bin/chess_uci` untouched).
- `hce` HEAD chain (newest first):
  - `44dc6b5` eval: apply texel-tuned linear weights (+~50 Elo)
  - `13dd7d0` texel: feature-dump + tuning pipeline
  - `e818841` search: log-based LMR + late move pruning (+123.7)
- So the engine today = **LMP + texel**. Both are committed but **not yet
  deployed** to the lichess bot unless you ran the deploy.
- Parked (not merged): `hce-conthist` (`6387dfe`) 1-ply continuation history,
  neutral.
- Every experiment (wins and rejects) is logged in `docs/HCE_EXPERIMENTS.md`.

## 2. Deploy the current engine (LMP + texel) to lichess

**Run from the worktree, NOT the main repo** — the packager rsyncs the engine
from its own working tree, and the main repo is on the `nn` branch (would ship
the wrong engine):

```bash
cd current/worktrees/hce_2500
./scripts/deploy_umbrel_hce.sh umbrel
```

After deploy, check it's live and watch a few games:
```bash
ssh umbrel 'systemctl --user status chessbot.service'   # or the unit name in use
ssh umbrel 'journalctl --user -u chessbot.service -f'    # live logs
```
Then watch https://lichess.org/@/vidbot . Live rating drift over ~50+ games is
the real-world check (self-play Elo overstates; expect roughly half to
two-thirds of it live).

## 3. Verify before building further (sanity gates)

From the worktree:
```bash
make -C src/core/engine -j4            # builds clean, no warnings
make test                              # NOTE: test_ai has a PRE-EXISTING
                                       # twofold-repetition failure on clean hce
                                       # (unrelated to this work). Others pass.
```
Quick strength sanity (depth/nodes on a middlegame FEN):
```bash
printf 'position fen r1bqkb1r/pp2pppp/2n2n2/2pp4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6\ngo movetime 2000\nquit\n' | ./src/core/engine/chess_uci
```

## 4. Validation protocol (do NOT skip)

One change = one commit = one match. Matches via `src/core/bot/test_lab.py`
with `--engine name=PATH` (prebuilt binaries — avoids the background-`make`
hang on this machine). 60 games = signal (±45 Elo), 120 games = decision.
Pre-build both engines in the FOREGROUND, then run the match in the background.

```bash
# baseline = current hce; build both to stable paths first
python3 src/core/bot/test_lab.py \
  --engine cand=/abs/path/cand_uci --engine base=/abs/path/base_uci --baseline base \
  --positions-count 30 --paired-colors --think-ms 120 --max-plies 160 \
  --positions-file data/positions/lichess_equal_positions.fen \
  --seed <SEED> --concurrency 6 --out current/<name>.json
```
Rules learned this session:
- A promising 60g result almost always shrinks at 120g. Confirm before merging.
- Pooling independent seeds is valid (60g+120g fresh seed = 180g).
- Commit clear wins; shelve anything whose 120g CI straddles zero.

## 5. The plan, in priority order

### 5a. Make texel better (highest value) — "joint PST tune"
Today only the 21 LINEAR scalar weights were tuned; the piece-square tables
(PST), king-safety, passers were held FIXED in the residual. That forced
material/mobility to inflate to grab eval scale (a partial-tuning artifact — it
still won, but leaves Elo on the table). Steps:

1. **Quiet-filter the dataset via qsearch.** Keep a position only if
   `|static_eval - qsearch_eval| < ~50cp` (no pending tactic). Add this filter
   to the `tunedump` path (engine already has qsearch). Removes label noise —
   pure quality gain on every position.
2. **Self-play for volume.** 2.4k vidbot games (~28k positions) is too thin for
   ~768 PST params. Self-play a few thousand LMP-vs-LMP games from the 1003
   opening FENs at fast movetime, dump positions + game result. Combine with
   the vidbot set (real-world diversity).
3. **Extend the feature dump to PST** (mg/eg per piece per square). Then tune
   material + PST + scalars JOINTLY (pawn anchored at 100). Verify exact
   reconstruction (0 mismatches) as before. This kills the scale artifact.
4. Port tuned tables back into `k_*_pst` / `hce_piece_value`, rebuild, and run
   the 60g -> 120g match gate vs current `hce` (`44dc6b5`).

### 5b. Fold the failed hand-guesses into the tune (do them RIGHT)
These lost as hand-guesses; let data set them:
- **Passed pawns**: replace the capped formula with a tunable per-rank bonus
  (mg/eg) and tune it.
- **King safety**: replace linear `attack_units*5` with a tunable attack table
  (value per unit-count bucket). This is the correct version of the -52 Elo
  quadratic guess.
- **Mobility**: it's inside the linear tune already; re-check once PST joins.

### 5c. Search ideas still on the table (secondary — search is near ceiling
### at 120ms, eval is the bottleneck)
- Singular extensions (complex, ~+20-40).
- SEE pruning of quiets at low depth; razoring.
- 2-way TT buckets; bigger TT for longer games.
- Eval speed (pawn hash, lazy eval) only helps if depth converts — lower
  priority now that LMP already converts depth.

## 6. Reproduce the texel pipeline from scratch

```bash
# 1. download vidbot games (public BOT account)
curl -s -H "Accept: application/x-chess-pgn" \
  "https://lichess.org/api/games/user/vidbot?max=3000&clocks=false&evals=false" \
  -o games.pgn
# 2. build labelled quiet positions
python3 scripts/texel_build_dataset.py --pgn games.pgn --out positions.txt --drop-forfeit
# 3. dump exact linear features (engine must be built)
printf 'tunedump positions.txt features.txt\nquit\n' | ./src/core/engine/chess_uci
# 4. tune (verifies exact reconstruction first; prints TUNED weight line)
python3 scripts/texel_tune.py --feats features.txt --freeze-material   # or default
```
Tuner flags: `--anchor-pawn` (default on), `--freeze-material`, `--l2 <k>`.

## 7. Gotchas / lessons (don't relearn these)

- **Piece enum order is `KING,QUEEN,BISHOP,KNIGHT,ROOK,PAWN`** ->
  knight=320, bishop=335 (bishop > knight). Do not swap when reading
  `hce_piece_value`. (This caused a 15cp reconstruction bug; the exact-verify
  step caught it — always keep that gate.)
- **`make` hangs when launched from a background task** on this machine — build
  in FOREGROUND, run only matches in the background.
- **Codex may run matches in parallel** — don't rebuild the shared
  `bin/chess_uci`; work in the worktree, pass prebuilt `--engine` paths.
- **Deploy from the worktree**, never the main repo (branch mismatch).
- **Self-play Elo overstates real gains** — confirm live on lichess.
- No AI attribution in commits/PRs (git identity only).

## 8. Session scoreboard (2026-07-09)

| Change | vs | Games | Elo | Result |
|---|---|---|---|---|
| LMR + LMP | original hce | 120 | +123.7 (P=100%) | MERGED e818841 |
| Texel weights | LMP | 180 pooled | ~+50 (P~97%) | MERGED 44dc6b5 |
| Continuation history | LMP | 240 | +4.3 | parked hce-conthist |
| Quadratic king safety | LMP | 60 | -52.5 | rejected |
| Safe mobility | LMP | 60 | -17.4 | rejected |
| Log-LMR alone | hce | 120 | +2.9 | kept as LMP substrate |

Net committed: **~+160 self-play Elo** over the session-start engine.
