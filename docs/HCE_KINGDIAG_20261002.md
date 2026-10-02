# HCE king-zone diagonal pressure — October 2, 2026

## Why

Two VidBot losses flagged by a friend (lichess `qLFEFG1M`, `SeQWOM3L`) were
evaluation errors, not search depth: the live HCE scored the position before
18.h4?? at +74 where Stockfish 18 gives -38, and kept playing h4 at 1 s.

## Diagnosis

44,571 positions sampled from 4,020 of VidBot's rated Lichess games
(2026-08-16 .. 10-02), labelled by Stockfish 18 at depth 10
(`sf_winprob_labels.py`), compared with the static eval (`tunedumpall`,
quiet rows only):

| King-zone pressure from enemy bishops/queens (against White - against Black) | HCE error, cp (+ = too good for White) |
|---|---:|
| <= -5 | -49 |
| -4..-3 | -12 |
| 0 | -1 |
| +3..+4 | +13 |
| >= +5 | +52 |

Locked centres were the other suspect (game 2, HCE +280 vs Stockfish +40..+140).
The data says the opposite: with locked pawns the eval *underrates* a mobility
edge. Game 2's excess came from mobility and piece-square terms (UCI `eval`).
A locked-pawn mobility term (`HceLockedMob`) fixed that bias but lowered no
validation loss and did not beat the king term alone in games; it stays off.

## Change

`HceKingDiag` (default 14): penalty W*u^2/8 (mg, a quarter in eg) where u counts
the king-zone squares (king, neighbours, the three squares two ranks ahead) hit
by enemy bishops and queens, plus one per bishop/queen x-raying the zone through
a defender pawn. It lives outside the Texel-tuned terms, so the refit keeps it
fixed. All weights were then refit with L-BFGS (l2 0.002) on 119k quiet
positions: the previous 108k plus 36.5k from VidBot's games (the 8k from the
most recent games held out). Exact reconstruction: 0 mismatches.

| Variant | val loss | holdout rms (cp) | vs live, 200 games 10+0.1 |
|---|---:|---:|---:|
| live (302e213) | — | 148 | — |
| A: refit, no new term | 0.02124 | 146 | -3.5 |
| B: refit + HceKingDiag 14 | 0.02106 | 145 | +12.2 |
| C: B + HceLockedMob 6 | 0.02107 | 145 | +13.9 |

B, 1,000 games (five seeds, all positive: +12.2, +10.4, +17.4, +15.6, +45.4):
529/1000 (+350 =358 -292), **+20.2 Elo [+2.9, +37.5]**, P(better) 98.9%, no
engine errors or flag falls. Played on Umbrel at nice 19, Syzygy 3-5 for both.

Also added: UCI `eval` prints the static eval split into its terms.
