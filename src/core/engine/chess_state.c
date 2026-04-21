#include "chess_state.h"

#include <string.h>

void chess_state_clear(GameState *s) {
    memset(s, 0, sizeof(*s));
    s->ep_square = CHESS_NO_SQUARE;
    s->result = GAME_RESULT_ONGOING;
    s->config.clock_enabled = false;
    s->config.initial_ms = 5 * 60 * 1000;
    s->config.increment_ms = 0;
    s->config.white_kind = PLAYER_LOCAL_HUMAN;
    s->config.black_kind = PLAYER_LOCAL_HUMAN;
    s->fullmove_number = 1;
    for (int sq = 0; sq < CHESS_BOARD_SQUARES; ++sq) {
        s->sq_piece[sq] = PIECE_NONE;
        s->sq_color[sq] = -1;
    }
}

void chess_state_rebuild_occupancy(GameState *s) {
    s->occ[PIECE_WHITE] = 0;
    s->occ[PIECE_BLACK] = 0;
    for (int sq = 0; sq < CHESS_BOARD_SQUARES; ++sq) {
        s->sq_piece[sq] = PIECE_NONE;
        s->sq_color[sq] = -1;
    }

    for (int p = 0; p < PIECE_TYPE_COUNT; ++p) {
        s->occ[PIECE_WHITE] |= s->bb[PIECE_WHITE][p];
        s->occ[PIECE_BLACK] |= s->bb[PIECE_BLACK][p];

        uint64_t wb = s->bb[PIECE_WHITE][p];
        while (wb != 0) {
            int sq = chess_pop_lsb(&wb);
            s->sq_piece[sq] = (int8_t)p;
            s->sq_color[sq] = PIECE_WHITE;
        }

        uint64_t bb = s->bb[PIECE_BLACK][p];
        while (bb != 0) {
            int sq = chess_pop_lsb(&bb);
            s->sq_piece[sq] = (int8_t)p;
            s->sq_color[sq] = PIECE_BLACK;
        }
    }
    s->occ_all = s->occ[PIECE_WHITE] | s->occ[PIECE_BLACK];
}

int chess_piece_on_square(const GameState *s, int sq, int *out_color) {
    int piece = s->sq_piece[sq];
    int color = s->sq_color[sq];
    if (piece != PIECE_NONE && color >= PIECE_WHITE && color <= PIECE_BLACK) {
        if (out_color != NULL) {
            *out_color = color;
        }
        return piece;
    }
    if (out_color != NULL) {
        *out_color = -1;
    }
    return PIECE_NONE;
}

int chess_pop_lsb(uint64_t *bb) {
    if (*bb == 0) {
        return -1;
    }
#if defined(__GNUC__) || defined(__clang__)
    int sq = __builtin_ctzll(*bb);
#else
    int sq = 0;
    uint64_t temp = *bb;
    while ((temp & 1ULL) == 0) {
        temp >>= 1;
        ++sq;
    }
#endif
    *bb &= (*bb - 1);
    return sq;
}

int chess_count_bits(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#else
    int c = 0;
    while (bb != 0) {
        bb &= (bb - 1);
        ++c;
    }
    return c;
#endif
}

int chess_find_king_square(const GameState *s, int side) {
    uint64_t bb = s->bb[side][PIECE_KING];
    if (bb == 0) {
        return -1;
    }
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb);
#else
    int sq = 0;
    while ((bb & 1ULL) == 0) {
        bb >>= 1;
        ++sq;
    }
    return sq;
#endif
}

void chess_reset_clocks(GameState *s) {
    s->clock_ms[PIECE_WHITE] = s->config.initial_ms;
    s->clock_ms[PIECE_BLACK] = s->config.initial_ms;
}
