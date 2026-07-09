# Kimi overnight log (branch `hce-kimi`)

Running progress log for the autonomous overnight session. Branched from `hce`
at `03afd16` (= LMP + texel, the confirmed engine). Append newest entries at
the top. One entry per experiment: what was tried, the match result, and
whether it was committed or reverted.

## Status summary (keep this block current)
- Branch base: `hce` @ 03afd16
- Confirmed wins committed this session: 1 — joint material/PST/scalar Texel tune (`2859044`)
- Datasets built:
  - `current/vidbot_20260712.pgn` (2,424 games)
  - `current/selfplay_120ms_20260712.pgn` (~1,000 games)
  - `current/combined_features.txt` (25,663 quiet positions)
- Pending / left for review: (none)
- Baseline binary updated: `/tmp/base_uci` rebuilt from `2859044`

## Entries

### 2026-07-09 (post-midnight) — Joint Texel tune of material, PSTs, and eval scalars
- Hypothesis: the previous linear Texel tune fixed material/mobility/structure but
  held PSTs fixed, causing partial-tuning inflation. Jointly tuning material,
  middlegame/endgame PSTs, and eval-phase scalars should remove the artifact and
  gain real strength.
- Change:
  - Extended `tunedump` with PST features and quiet filtering via
    `hce_qsearch_eval_cp_stm`.
  - Added separate `_eg` PST tables in `hce_eval.c` and wired them into eval.
  - Extended `scripts/texel_tune.py` to tune material + mg/eg PSTs + scalars
    jointly, with exact feature reconstruction verification.
  - Added `scripts/texel_selfplay.py` and `scripts/texel_apply_tune.py` for data
    generation and applying tuned weights.
  - Trained on 25,663 quiet positions from 2,424 vidbot games + ~1,000 120ms
    self-play games.
- Build: clean, `make bin/chess_uci` passed.
- Match:
  - 60g (seed 20260712): cand 38.0/60, +46.6 Elo, CI95 [-11.6, +107.5],
    P(better)=93.7% (`current/pst_tune_60g.json`).
  - 120g confirm (seed 20260714): cand 67.5/120, +43.7 Elo, CI95 [-4.6, +93.7],
    P(better)=96.2% (`current/pst_tune_120g.json`).
- Result: COMMIT `2859044` on `hce-kimi`; `/tmp/base_uci` promoted to the new
  tuned build.

<!-- template:
### <time> — <short title>
- Hypothesis:
- Change:
- Build: clean / errors
- Match: <N>g vs <base>, Elo <x>, CI <..>, P(better) <..>  -> COMMIT <sha> / REVERTED
-->
