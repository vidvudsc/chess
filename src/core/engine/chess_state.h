#ifndef CHESS_STATE_H
#define CHESS_STATE_H

#include "chess_types.h"

typedef struct UndoRecord {
    Move move;
    int captured_piece;
    int captured_square;
    uint8_t castling_rights_prev;
    int ep_square_prev;
    int halfmove_prev;
    int fullmove_prev;
    uint64_t hash_prev;
    GameResult result_prev;
    bool extended_state_saved;
    int clock_prev[2];
    int irreversible_ply_prev;
    Move last_move_prev;
    bool had_last_move_prev;
    int hash_history_count_prev;
} UndoRecord;

typedef struct GameState {
    uint64_t bb[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT];
    uint64_t occ[PIECE_COLOR_COUNT];
    uint64_t occ_all;
    int8_t sq_piece[CHESS_BOARD_SQUARES];
    int8_t sq_color[CHESS_BOARD_SQUARES];

    int side_to_move;
    uint8_t castling_rights;
    int ep_square;
    int halfmove_clock;
    int fullmove_number;

    uint64_t zobrist_hash;
    GameResult result;

    MatchConfig config;
    int clock_ms[2];

    UndoRecord undo_stack[CHESS_MAX_GAME_PLY];
    int ply;

    uint64_t hash_history[CHESS_MAX_GAME_PLY + 1];
    int hash_history_count;
    int irreversible_ply;

    Move last_move;
    bool has_last_move;
} GameState;

enum {
    CASTLE_WHITE_KING = 1 << 0,
    CASTLE_WHITE_QUEEN = 1 << 1,
    CASTLE_BLACK_KING = 1 << 2,
    CASTLE_BLACK_QUEEN = 1 << 3,
};

void chess_state_clear(GameState *s);
void chess_state_rebuild_occupancy(GameState *s);
int chess_piece_on_square(const GameState *s, int sq, int *out_color);
int chess_pop_lsb(uint64_t *bb);
int chess_count_bits(uint64_t bb);
int chess_find_king_square(const GameState *s, int side);
void chess_reset_clocks(GameState *s);

#endif
