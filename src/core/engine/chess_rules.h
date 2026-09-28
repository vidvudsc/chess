#ifndef CHESS_RULES_H
#define CHESS_RULES_H

#include "chess_state.h"

void chess_init(GameState *s, const MatchConfig *cfg);
int chess_generate_legal_moves(const GameState *s, Move out[CHESS_MAX_MOVES]);
int chess_generate_legal_moves_mut(GameState *s, Move out[CHESS_MAX_MOVES]);
// Legal captures and promotions only. Does not generate quiet evasions, so
// callers must use the full generator when the side to move is in check.
int chess_generate_tactical_moves_mut(GameState *s, Move out[CHESS_MAX_MOVES]);
bool chess_make_move(GameState *s, Move m);
bool chess_make_move_trusted(GameState *s, Move legal_move);
bool chess_undo_move(GameState *s);
bool chess_in_check(const GameState *s, int side);
GameResult chess_update_result(GameState *s);
uint64_t chess_perft(GameState *s, int depth);
bool chess_is_move_legal(const GameState *s, Move m);
bool chess_has_mating_material(const GameState *s, int side);
// Magic-bitboard slider attacks shared by move generation and evaluators.
uint64_t chess_rook_attacks(int sq, uint64_t occ);
uint64_t chess_bishop_attacks(int sq, uint64_t occ);
// Build the attack tables (idempotent). The *_fast lookups below skip the
// readiness check and must only run after this (or any move generation).
void chess_attack_tables_init(void);

typedef struct ChessMagicEntry {
    uint64_t mask;
    uint64_t magic;
    uint64_t *attacks;
    int shift;
} ChessMagicEntry;
extern ChessMagicEntry g_chess_rook_magic[64];
extern ChessMagicEntry g_chess_bishop_magic[64];

static inline uint64_t chess_rook_attacks_fast(int sq, uint64_t occ) {
    const ChessMagicEntry *e = &g_chess_rook_magic[sq];
    return e->attacks[((occ & e->mask) * e->magic) >> e->shift];
}

static inline uint64_t chess_bishop_attacks_fast(int sq, uint64_t occ) {
    const ChessMagicEntry *e = &g_chess_bishop_magic[sq];
    return e->attacks[((occ & e->mask) * e->magic) >> e->shift];
}
void chess_set_result(GameState *s, GameResult result);
void chess_tick_clock(GameState *s, int delta_ms);

#endif
