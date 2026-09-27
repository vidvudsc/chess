# HCE endgame repair — September 27, 2026

## Why

Reviewing live VidBot losses (see `HCE_RUNTIME_REPAIR_20260925.md`) left the
43...Kg8 rook-ending loss unexplained: more time did not fix it. Reading the
live evaluation against Stockfish 18 found concrete defects rather than
tuning noise.

| Defect | Evidence |
|---|---|
| King piece-square tables were all zero | Joint Texel tune `1a9b951` (2026-07-09) wrote zeros because the king feature plane was dead in the dump; `42ec83c` fixed the dump but kept zeros, and later refits never relearned them. |
| Pawn and queen tables were upside down | Tables are written rank 8 first but indexed from a1 (knight/bishop/rook tables were flipped, these were not). A pawn on a2 scored +21, on a7 +2; 1.e4 cost 12 cp. |
| King danger active in queenless endings | In `8/1pr2kp1/1R5p/4PP1P/8/4K3/8/8 b` black's king cost 26 cp on f7 and 2 cp on g8; HCE played Kg8 (Stockfish -5.4) over Rc4 (-0.7) and scored it -0.05. |
| No endgame knowledge | KPK draw `8/5k2/8/8/8/8/4P3/4K3 w` scored +1.66; passed pawns were rank-only and capped; no king proximity, rule of the square, draw scaling, or tablebases. |
| Test rig could not see endgames | `test_lab.py` drew games at 160 plies; 16.8% of 34,626 archived test games ended that way. |
| Texel reconstruction not exact | Stage-B per-rank passer terms were scaled by the passed-pawn scale; 6.7% of dump rows missed by 1 cp. |

## Changes

Five eval switches (UCI, all default on; all off reproduces the old eval):
`HcePawnPstFix`, `HceKingPst`, `HceKsFade` (king danger shrinks with the
enemy's pieces unless a queen remains), `HcePasser` (king proximity to the stop
square, free path, support, rook behind, rule of the square), `HceScale`
(lone minors, KNN, opposite bishops, single-pawn and same-wing rook endings).

Syzygy probing through vendored Fathom (MIT): WDL in search after zeroing
moves, DTZ at the root. `SyzygyPath` and `Hash` UCI options. 3-4-5-man tables
live in `shared/syzygy` on Umbrel (290 files, sizes verified).

Test rig: `--max-plies` default 600, a 600-position balanced endgame suite
(`data/positions/endgame_balanced.fen`, one position per game from VidBot and
self-play games, Stockfish depth 12 within 120 cp), `scripts/eval_switch_screen.py`.

Bot: `LICHESS_BOT_SYZYGY_PATH`, `LICHESS_BOT_HASH_MB`; stalled games are
aborted (opponent never made a first move, 3 min) or claimed (20 min); a
`DRAIN` file stops new games before a release switch.

## Results (10+0.1, paired colors, Threads=1)

| Match | Suite | Games | Score | Elo |
|---|---|---:|---:|---:|
| All switches vs all off | endgame | 200 | 66.0% (+79 =106 -15) | +115 |
| All switches vs all off | openings | 200 | 65.2% (+110 =41 -49) | +109 |
| All minus PawnPstFix vs all | endgame | 200 | 49.2% | -5 |
| All minus PawnPstFix vs all | openings | 91 | 46.2% | -27 |
| All minus KingPst vs all | endgame | 200 | 45.5% | -31 |
| Texel refit vs untuned (both all on) | openings | 200 | 55.8% (+83 =57 -60) | +40 (paired P=98%) |
| Texel refit vs untuned (both all on) | endgame | 200 | 49.5% | -3.5 (within noise) |

The refit (62,733 quiet self-play positions dumped with the switches on,
exact reconstruction, l2=3) raised rooks 638->656 and the queen 1299->1320,
cut middlegame mobility, and lowered the passed-pawn eg scale 122->114 now
that the new passer terms carry part of that value.

Tablebases vs none (endgame suite, 50 games): 48%, within noise. Audit of
every game with python-chess's Syzygy reader: the tablebase side never made a
move that worsened its WDL; every loss was already lost on entering the
tables. 150/150 random 3-5-man root positions got a WDL-preserving move.

## Reproduce

```sh
make -C src/core/engine
python3 scripts/eval_switch_screen.py --engine src/core/engine/chess_uci \
  --positions-file data/positions/endgame_balanced.fen --out-dir current/screen_eg
bash scripts/deploy_hce_release.sh umbrel     # waits for idle, rolls back on failure
```
