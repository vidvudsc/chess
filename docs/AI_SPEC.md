# Chess AI Specification

## Scope
This document describes the current built-in AI in this project (C + raylib chess app), what it optimizes for, and what telemetry/logging it exposes.

## Engine Interface
- Entry point: `chess_ai_pick_move(const GameState*, const AiSearchConfig*, AiSearchResult*)`
- Unified facade entry point: `chess_engine_query(const GameState*, const ChessEngineRequest*, ChessEngineResponse*)`
- Lightweight eval endpoint for UI bars: `chess_ai_eval_cp(const GameState*)`
- Config:
  - `think_time_ms`: soft search budget in milliseconds
  - `max_depth`: iterative deepening limit
- Result:
  - `best_move`
  - `score_cp` (from side-to-move perspective)
  - `depth_reached`
  - `nodes`
  - `elapsed_ms`

## Search Core
- Algorithm: iterative deepening negamax with alpha-beta pruning.
- Principal variation search style re-search on non-first moves.
- Aspiration windows around previous iteration score.
- Quiescence search for tactical stabilization (captures/promotions + in-check handling).
- Quiescence includes a shallow quiet-check extension to better catch forcing tactical lines.
- Null-move pruning (non-check, non-trivial material positions) for faster fail-high cutoffs.
- Reverse-futility pruning at shallow depths.
- Stronger late move reduction (LMR) profile with history/check-aware reduction adjustments.
- Late move futility pruning for low-promise quiet moves.
- Extensions/reductions:
  - check extension (+1 ply when in check)
  - light late move reduction on quieter late moves

## Move Ordering
- Transposition-table move first
- MVV-LVA style capture ordering
- Promotion bonuses
- Killer moves (per ply)
- History heuristic table
- History malus on quiet moves that fail to improve during cutoffs

## Transposition Table
- Fixed-size hash table (`2^20` entries)
- Key: incremental zobrist hash from `GameState`
- Stores: best move, bound type (exact/lower/upper), depth, age, score
- Mate-score normalization for ply-correct reuse

## Evaluation
- Tapered midgame/endgame blend with phase weighting
- Material values (MG/EG)
- Piece-square tables (all pieces)
- Bishop pair bonus
- Pawn structure terms:
  - doubled pawns penalty
  - isolated pawns penalty
  - passed pawn bonuses by advance rank
- Rook activity:
  - open / semi-open files
  - rook on 7th rank bonus
- King handling:
  - king shield in middlegame
  - king danger/pressure terms from enemy piece attacks around king zone
  - king activity in endgame
  - mop-up pressure in simplified winning endgames
- Mobility/pressure:
  - piece mobility terms for knights/bishops/rooks/queens
  - pressure bonuses for attacks into enemy king zone
  - center-control bonus via pawn attacks

## Opening Behavior
- Built-in small hardcoded opening book from common starting positions
- If book position matches, chooses among legal book moves with variation seed

## UI Evaluation Bar Behavior
- Eval bar is optional (`Settings -> Eval Bar`)
- Display value is white-perspective centipawn score
- UI bar now animates toward target instead of snapping
- Sampling cadence is throttled for smoothness and lower overhead
- Tactical override in `chess_ai_eval_cp`:
  - detects terminal states
  - detects immediate checkmate-in-one for side-to-move only and maps it near mate score
- Additional tactical probe in `chess_ai_eval_cp`:
  - runs a small depth-limited search + quiescence (with shallow quiet checks) under a short time budget
  - improves bar accuracy in tactical positions (e.g., hanging queen / forcing checks)

## Stockfish Debug Compare Bar
- Optional second eval bar in Settings: `Stockfish Debug Bar`
- Runs in a separate background worker (`stockfish_eval_worker`) so the UI thread stays responsive.
- Queries external Stockfish via UCI (`go movetime ...`) and maps reported scores to white perspective.
- Worker degrades gracefully on transient failures (cooldown + retry) and reuses very recent same-FEN results.
- Designed for comparison/debugging only; gameplay move selection still uses the built-in engine.

## Worker/Threading Model
- AI move search runs on background worker thread (`chess_ai_worker`)
- UI thread polls results and applies them only if position signature still matches
- Prevents stale AI responses from being applied after state changes
- Added dedicated AI test runner worker (`ai_test_runner`) for long benchmark sessions (AI vs Stockfish).
- Core AI entry points are serialized internally because the engine currently uses shared global TT/eval caches.

## Logging/Debug Data
Per game JSON log is stored under:
- `/tmp/chess_logs`

Each move entry includes:
- ply/side/uci/fen_after
- think time and clocks
- AI metadata:
  - `ai_move`
  - `ai_depth`
  - `ai_nodes`
  - `ai_score_cp`

UI can open this folder directly from Settings via `Open Logs Folder`.

## Performance Notes
- Added mutable legal-move generation API for hot paths:
  - `chess_generate_legal_moves_mut(GameState*, Move out[256])`
- AI search/eval now use mutable movegen to avoid repeated full-state copy overhead in deep search.
- Added trusted internal move API for search nodes:
  - `chess_make_move_trusted(GameState*, Move legal_move)`
  - avoids full legal re-validation + full result recomputation in deep search.

## Current Constraints
- Single-threaded search per process (no parallel PV split / SMP search)
- Shared in-process TT/eval caches are currently globally serialized
- No Syzygy tablebases
- No endgame tablebase integration
- Opening book is intentionally small and embedded

## Planned Upgrade Directions
- Dedicated analysis worker separate from move worker
- Richer opening book loading from external file
- Tablebase probes for <= 5/6-man endings
- Incremental eval caches and more aggressive selective search

## AI Test Lab (UI)
- Accessible from Settings: `AI Test Lab`.
- Runs batch matches of internal AI vs Stockfish from balanced non-start positions.
- Reports live:
  - game progress,
  - W/D/L score,
  - estimated Elo (ladder mode) or Elo delta vs fixed setup,
  - ETA for completion.
- Opponent modes:
  - `Fixed Opponent`: fixed Stockfish skill/time (reports Elo delta vs that setup),
  - `Elo Ladder Sweep`: rotates across multiple Stockfish UCI-Elo brackets and estimates absolute Elo from aggregate score.
- Position source options:
  - built-in balanced set bundled in code,
  - external `data/positions/lichess_equal_positions.fen` file.
- Position fetch helper:
  - `make fetch_positions` / `make build_testlab_positions`
  - script: `scripts/build_testlab_positions.py`
