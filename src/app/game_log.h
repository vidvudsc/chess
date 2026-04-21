#ifndef GAME_LOG_H
#define GAME_LOG_H

#include <stdbool.h>

#include "chess_io.h"

typedef struct GameLogMove {
    int ply;
    int side;
    int think_ms;
    bool ai_move;
    int ai_depth;
    uint64_t ai_nodes;
    int ai_score_cp;
    int clock_white_ms;
    int clock_black_ms;
    char uci[6];
    char fen_after[256];
} GameLogMove;

typedef struct GameLog {
    char path[512];
    char start_fen[256];
    GameResult result;
    int elapsed_white_ms;
    int elapsed_black_ms;
    int move_count;
    GameLogMove moves[CHESS_MAX_GAME_PLY];
    bool active;
} GameLog;

void game_log_begin(GameLog *log, const GameState *state);
void game_log_record_move(GameLog *log,
                          const GameState *state_after,
                          Move move,
                          int mover_side,
                          int think_ms,
                          bool ai_move,
                          int ai_depth,
                          uint64_t ai_nodes,
                          int ai_score_cp);
void game_log_truncate(GameLog *log, int move_count);
void game_log_set_elapsed(GameLog *log, int white_ms, int black_ms);
void game_log_set_result(GameLog *log, GameResult result);
bool game_log_flush(const GameLog *log);
const char *game_log_path(const GameLog *log);
const char *game_log_directory(void);

#endif
