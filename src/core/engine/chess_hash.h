#ifndef CHESS_HASH_H
#define CHESS_HASH_H

#include "chess_state.h"

void chess_hash_init(void);
uint64_t chess_hash_full(const GameState *s);
uint64_t chess_hash_piece_key(int color, int piece, int sq);
uint64_t chess_hash_side_key(void);
uint64_t chess_hash_castle_key(uint8_t rights);
uint64_t chess_hash_ep_key(int ep_square);

#endif
