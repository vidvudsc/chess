#ifndef HCE_INTERNAL_H
#define HCE_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "chess_ai.h"
#include "chess_io.h"
#include "chess_opening_book.h"

#define HCE_INF 32000
#define HCE_MATE 30000
#define HCE_MATE_THRESHOLD 29000
#define HCE_MAX_DEPTH 32
#define HCE_MAX_PLY 96

extern const int hce_piece_value[PIECE_TYPE_COUNT];
extern const int hce_phase_inc[PIECE_TYPE_COUNT];

void hce_init_tables(void);
uint64_t hce_knight_attacks(int sq);
uint64_t hce_king_attacks(int sq);
uint64_t hce_pawn_attacks(int side, int sq);
uint64_t hce_bishop_attacks(int sq, uint64_t occ);
uint64_t hce_rook_attacks(int sq, uint64_t occ);
uint64_t hce_attackers_to_square(const GameState *s, int sq, int side);

int hce_eval_cp_stm(const GameState *s);
bool hce_eval_breakdown(const GameState *s, ChessEvalBreakdown *out);
int engine_eval_cp_stm(const GameState *s);
int hce_score_search_draw_stm(const GameState *s);
bool hce_pick_opening_move(const GameState *s, Move *out_move);
bool hce_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out);
int hce_probe_deep_eval_cp_stm(const GameState *state);

#endif
