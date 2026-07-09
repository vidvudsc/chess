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
int hce_experimental_eval_cp_stm(const GameState *s);
bool hce_eval_breakdown(const GameState *s, ChessEvalBreakdown *out);

// Texel-tuning support: the linear decomposition of one side's eval into raw
// feature counts plus a residual (the non-tuned mg/eg sums). Reconstructing
// total_mg = sum(weight*count) + residual_mg (and likewise eg), then blending
// by phase once, reproduces eval_side exactly. See scripts/texel_tune.py.
typedef struct HceTuneFeatures {
    int mat[PIECE_TYPE_COUNT];  // piece counts (value adds to both mg and eg)
    int isolated;               // isolated pawns
    int doubled;                // pawns on a doubled file
    int mob_n, mob_b, mob_r, mob_q;  // raw mobility square counts by piece
    int rook_open, rook_semi;   // rooks on open / half-open files
    int pst[PIECE_TYPE_COUNT][64];  // piece-square counts (view side, mirrored for black)
    int residual_mg;            // summed mg of all non-tuned terms
    int residual_eg;            // summed eg of all non-tuned terms
} HceTuneFeatures;

// Fills white/black feature structs and returns eval_cp_stm (incl. tempo) so a
// caller can verify the linear reconstruction against the real eval.
int hce_eval_tune_features(const GameState *s,
                           HceTuneFeatures *white_out,
                           HceTuneFeatures *black_out,
                           int *phase_out);
int engine_eval_cp_stm(const GameState *s);
int hce_score_search_draw_stm(const GameState *s);
int hce_qsearch_eval_cp_stm(const GameState *root);
bool hce_pick_opening_move(const GameState *s, Move *out_move);
bool hce_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out);
int hce_probe_deep_eval_cp_stm(const GameState *state);

#endif
