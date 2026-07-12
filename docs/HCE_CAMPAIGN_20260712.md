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

All playing-code candidates above were reverted. The repeated 60-to-120
collapses reinforce the requirement for 240-game confirmation before any Elo
enters the campaign total.

## Diagnostic evidence

- Stockfish 17.1 labeled 5,000 diverse positions at depth 14. HCE/Stockfish
  score correlation was 0.81 after excluding decisive outliers, but three
  residual-derived eval candidates failed match transfer.
- A depth-12 Stockfish policy improved held-out top-1 move prediction from the
  earlier self-taught policy's 7.7% to 12.1% and reduced fixed-depth nodes by
  6.5%, but lost games. Node reduction alone is not a promotion signal.
- Forty recent VidBot losses produced 1,992 analyzed bot moves. Most large
  non-mate losses were quiet positional moves; a Four Knights `...Qe7` instead
  of castling repeated in multiple games because the book ended one ply too
  early. Broad automatic book extension lost, so future book edits must be
  targeted and curated.
