#ifndef CHESS_HASH_H
#define CHESS_HASH_H

#include "chess_state.h"

void chess_hash_init(void);
uint64_t chess_hash_full(const GameState *s);

// Key tables (filled by chess_hash_init); the accessors are inline because
// every make/undo in the search uses them.
extern uint64_t g_chess_hash_piece_keys[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT][64];
extern uint64_t g_chess_hash_side_key;
extern uint64_t g_chess_hash_castle_keys[16];
extern uint64_t g_chess_hash_ep_file_keys[8];

static inline uint64_t chess_hash_piece_key(int color, int piece, int sq) {
    return g_chess_hash_piece_keys[color][piece][sq];
}

static inline uint64_t chess_hash_side_key(void) {
    return g_chess_hash_side_key;
}

static inline uint64_t chess_hash_castle_key(uint8_t rights) {
    return g_chess_hash_castle_keys[rights & 0xF];
}

static inline uint64_t chess_hash_ep_key(int ep_square) {
    if (ep_square == CHESS_NO_SQUARE) {
        return 0;
    }
    return g_chess_hash_ep_file_keys[square_file(ep_square)];
}

#endif
