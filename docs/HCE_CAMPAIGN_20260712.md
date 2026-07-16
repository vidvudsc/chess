# HCE +100 Elo campaign

Baseline: `925597a`, frozen as `current/campaign_baseline_925597a_uci`.
Only changes developed after that commit count toward the campaign target.

## Confirmed total

`0 Elo`. No candidate has passed long confirmation yet.

## Rejected candidates

- Stockfish residual space (`36/10`): 30.0/60, 0.0 Elo, P=50.0%.
- Seventh-rank rook bonus: 29.0/60, -11.6 Elo, P=36.3%.
- Nonlinear bishop/rook mobility: +52.5 Elo at 60g, then +5.8 Elo at
  120g, P=59.8%.
- Stockfish-trained compact quiet policy: 28.5/60, -17.4 Elo, P=27.5%.
- One ply of quiet checks in qsearch: 26.0/60, -46.6 Elo, P=7.0%.
- Safe mobility with three-split Texel refit: 32.0/60, +23.2 Elo,
  P=74.1%; below gate.
- Thread-local full eval caches: +34.9 Elo at 60g, then -37.8 Elo at
  120g, P=4.7%.
- Packed atomic shared eval cache: 28.5/60, -17.4 Elo, P=29.6%.
- Atomic packed TT: rejected before match after a timed tactical regression.
- Deterministic lazy-SMP helper diversification: 25.0/60, -58.5 Elo,
  P=1.8%.
- Select the strictly deepest lazy-SMP helper result: +46.6 Elo at 60g,
  then -8.7 Elo at 120g, P=35.1%.
- Four-ply Stockfish-expanded opening book: 27.5/60, -29.0 Elo, P=16.8%.
- Full 789-parameter Stockfish distillation with material frozen and strong
  ridge regularization: 31.5/60, +17.4 Elo, P=69.3%; below gate.
- Four-million-entry direct-mapped TT at Threads=4: +88.7 Elo at 60g with a
  fully positive interval, then 53.5/120, -37.8 Elo, P=6.6%; reverted.
- More aggressive LMP (`2 + depth^2`): rejected before match after the HCE
  suite reproduced the forbidden pawn-race king walk; reverted.
- Bounded gravity quiet history with stronger updates and matching LMR
  thresholds: 26.0/60, -46.6 Elo, P=7.0%; reverted.
- Strongly regularized Stockfish-distilled king PST: +52.5 Elo at the
  deterministic Threads=1 60g screen, then 58.5/120, -8.7 Elo, paired
  P=35.8%; reverted.
- Extend passed-pawn moves to the sixth or seventh rank: 32.0/60, +23.2 Elo,
  paired P=78.2%; below gate and reverted.
- Restrict that extension to seventh-rank passers: +34.9 Elo at 60g, then
  62.0/120, +11.6 Elo, paired P=70.6%; reverted.
- Use Threads=4 instead of Threads=3 for a solo game: 29.5/60, -5.8 Elo,
  paired P=42.8%; the deployed 3/2/1 allocation remains unchanged.

All playing-code candidates above were reverted. The repeated 60-to-120
collapses reinforce the requirement for 240-game confirmation before any Elo
enters the campaign total.

## Correctness and measurement infrastructure

- Auditing lazy SMP found data races in both the shared transposition table
  and the full-evaluation cache. The replacement TT uses an atomic key lock
  plus packed atomic payload; the evaluation cache stores its complete key tag
  and score in one atomic word.
- The race-free engine is exactly identical to frozen `925597a` at Threads=1
  on 20/20 depth-9 positions (move, score, nodes, and depth). At Threads=4 it
  scored 61.5/120, +8.7 Elo, P=65.4%, so it is correctness infrastructure and
  contributes `0 Elo` to the campaign total.
- `test_lab.py` now also calculates confidence from paired-position averages
  when colors are swapped, avoiding the assumption that both games from one
  opening are independent samples.
- The VidBot blunder analyzer now compares the best and played move from the
  same root at the same Stockfish depth. Previously it searched the played
  move's child at full depth, effectively granting an extra ply and producing
  false 30,000 cp horizon losses.

## Diagnostic evidence

- Stockfish 17.1 labeled 5,000 diverse positions at depth 14. HCE/Stockfish
  score correlation was 0.81 after excluding decisive outliers, but three
  residual-derived eval candidates failed match transfer.
- A depth-12 Stockfish policy improved held-out top-1 move prediction from the
  earlier self-taught policy's 7.7% to 12.1% and reduced fixed-depth nodes by
  6.5%, but lost games. Node reduction alone is not a promotion signal.
- Re-analyzing 20 recent VidBot losses with both candidate moves searched from
  the same root produced 176 meaningful non-mate errors between 50 and 2,000
  centipawns. Of those, 130 were a quiet move instead of a better quiet move,
  spread across the opening (44), middlegame (68), and endgame (64). This points
  to broad quiet positional selection rather than one tactical move class.
