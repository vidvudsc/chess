#include "chess_hash.h"

#include <stdbool.h>

static uint64_t g_piece_keys[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT][64];
static uint64_t g_side_key;
static uint64_t g_castle_keys[16];
static uint64_t g_ep_file_keys[8];
static bool g_inited = false;

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void chess_hash_init(void) {
    if (g_inited) {
        return;
    }

    uint64_t seed = 0xD1CEB00B5EED1234ULL;
    for (int c = 0; c < PIECE_COLOR_COUNT; ++c) {
        for (int p = 0; p < PIECE_TYPE_COUNT; ++p) {
            for (int sq = 0; sq < 64; ++sq) {
                g_piece_keys[c][p][sq] = splitmix64_next(&seed);
            }
        }
    }

    g_side_key = splitmix64_next(&seed);
    for (int i = 0; i < 16; ++i) {
        g_castle_keys[i] = splitmix64_next(&seed);
    }
    for (int i = 0; i < 8; ++i) {
        g_ep_file_keys[i] = splitmix64_next(&seed);
    }

    g_inited = true;
}

uint64_t chess_hash_piece_key(int color, int piece, int sq) {
    return g_piece_keys[color][piece][sq];
}

uint64_t chess_hash_side_key(void) {
    return g_side_key;
}

uint64_t chess_hash_castle_key(uint8_t rights) {
    return g_castle_keys[rights & 0xF];
}

uint64_t chess_hash_ep_key(int ep_square) {
    if (ep_square == CHESS_NO_SQUARE) {
        return 0;
    }
    return g_ep_file_keys[square_file(ep_square)];
}

uint64_t chess_hash_full(const GameState *s) {
    uint64_t h = 0;
    for (int color = 0; color < PIECE_COLOR_COUNT; ++color) {
        for (int piece = 0; piece < PIECE_TYPE_COUNT; ++piece) {
            uint64_t bb = s->bb[color][piece];
            while (bb != 0) {
                int sq = chess_pop_lsb(&bb);
                h ^= g_piece_keys[color][piece][sq];
            }
        }
    }

    h ^= g_castle_keys[s->castling_rights & 0xF];
    h ^= chess_hash_ep_key(s->ep_square);
    if (s->side_to_move == PIECE_BLACK) {
        h ^= g_side_key;
    }

    return h;
}
