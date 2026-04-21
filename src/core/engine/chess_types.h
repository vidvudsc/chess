#ifndef CHESS_TYPES_H
#define CHESS_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    CHESS_BOARD_SQUARES = 64,
    CHESS_MAX_MOVES = 256,
    CHESS_MAX_GAME_PLY = 4096,
    CHESS_NO_SQUARE = -1,
    CHESS_PROMO_NONE = 7,
};

typedef enum PieceColor {
    PIECE_WHITE = 0,
    PIECE_BLACK = 1,
    PIECE_COLOR_COUNT = 2,
} PieceColor;

typedef enum PieceType {
    PIECE_KING = 0,
    PIECE_QUEEN = 1,
    PIECE_BISHOP = 2,
    PIECE_KNIGHT = 3,
    PIECE_ROOK = 4,
    PIECE_PAWN = 5,
    PIECE_TYPE_COUNT = 6,
    PIECE_NONE = 7,
} PieceType;

typedef enum PlayerKind {
    PLAYER_LOCAL_HUMAN = 0,
    PLAYER_BOT_STUB = 1,
    PLAYER_REMOTE_STUB = 2,
} PlayerKind;

typedef enum MoveFlag {
    MOVE_FLAG_NONE = 0,
    MOVE_FLAG_CAPTURE = 1 << 0,
    MOVE_FLAG_DOUBLE_PUSH = 1 << 1,
    MOVE_FLAG_EN_PASSANT = 1 << 2,
    MOVE_FLAG_CASTLE = 1 << 3,
    MOVE_FLAG_PROMOTION = 1 << 4,
    MOVE_FLAG_CHECK_HINT = 1 << 5,
} MoveFlag;

typedef enum GameResult {
    GAME_RESULT_ONGOING = 0,
    GAME_RESULT_WHITE_WIN = 1,
    GAME_RESULT_BLACK_WIN = 2,
    GAME_RESULT_DRAW_STALEMATE = 3,
    GAME_RESULT_DRAW_REPETITION = 4,
    GAME_RESULT_DRAW_50 = 5,
    GAME_RESULT_DRAW_75 = 6,
    GAME_RESULT_DRAW_INSUFFICIENT = 7,
    GAME_RESULT_DRAW_AGREED = 8,
    GAME_RESULT_WIN_TIMEOUT = 9,
    GAME_RESULT_WHITE_WIN_RESIGN = 10,
    GAME_RESULT_BLACK_WIN_RESIGN = 11,
} GameResult;

typedef struct MatchConfig {
    bool clock_enabled;
    int initial_ms;
    int increment_ms;
    PlayerKind white_kind;
    PlayerKind black_kind;
} MatchConfig;

typedef uint32_t Move;

#define MOVE_FROM_SHIFT 0u
#define MOVE_TO_SHIFT 6u
#define MOVE_PIECE_SHIFT 12u
#define MOVE_PROMO_SHIFT 15u
#define MOVE_FLAGS_SHIFT 18u

#define MOVE_FROM_MASK 0x3Fu
#define MOVE_TO_MASK 0x3Fu
#define MOVE_PIECE_MASK 0x7u
#define MOVE_PROMO_MASK 0x7u
#define MOVE_FLAGS_MASK 0xFFu

static inline Move move_pack(int from, int to, int piece, int promo, uint32_t flags) {
    return ((uint32_t)(from & MOVE_FROM_MASK) << MOVE_FROM_SHIFT) |
           ((uint32_t)(to & MOVE_TO_MASK) << MOVE_TO_SHIFT) |
           ((uint32_t)(piece & MOVE_PIECE_MASK) << MOVE_PIECE_SHIFT) |
           ((uint32_t)(promo & MOVE_PROMO_MASK) << MOVE_PROMO_SHIFT) |
           ((uint32_t)(flags & MOVE_FLAGS_MASK) << MOVE_FLAGS_SHIFT);
}

static inline int move_from(Move m) {
    return (int)((m >> MOVE_FROM_SHIFT) & MOVE_FROM_MASK);
}

static inline int move_to(Move m) {
    return (int)((m >> MOVE_TO_SHIFT) & MOVE_TO_MASK);
}

static inline int move_piece(Move m) {
    return (int)((m >> MOVE_PIECE_SHIFT) & MOVE_PIECE_MASK);
}

static inline int move_promo(Move m) {
    return (int)((m >> MOVE_PROMO_SHIFT) & MOVE_PROMO_MASK);
}

static inline uint32_t move_flags(Move m) {
    return (uint32_t)((m >> MOVE_FLAGS_SHIFT) & MOVE_FLAGS_MASK);
}

static inline bool move_has_flag(Move m, MoveFlag flag) {
    return (move_flags(m) & (uint32_t)flag) != 0;
}

static inline int square_file(int sq) {
    return sq & 7;
}

static inline int square_rank(int sq) {
    return sq >> 3;
}

static inline int make_square(int file, int rank) {
    return rank * 8 + file;
}

#endif
