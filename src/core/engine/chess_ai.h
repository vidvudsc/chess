#ifndef CHESS_AI_H
#define CHESS_AI_H

#include <stddef.h>
#include <stdint.h>

#include "chess_rules.h"

typedef void (*AiSearchInfoCallback)(int depth,
                                     int score_cp,
                                     Move best_move,
                                     uint64_t nodes,
                                     int elapsed_ms,
                                     void *user_data);

typedef enum ChessAiBackend {
    CHESS_AI_BACKEND_CLASSIC = 0,
    CHESS_AI_BACKEND_NN = 1,
} ChessAiBackend;

typedef struct AiSearchConfig {
    int think_time_ms;
    int max_depth;
    AiSearchInfoCallback info_callback;
    void *info_user_data;
} AiSearchConfig;

typedef struct AiSearchResult {
    Move best_move;
    bool found_move;
    bool used_opening_book;
    int score_cp;
    int depth_reached;
    uint64_t nodes;
    int elapsed_ms;
} AiSearchResult;

typedef struct ChessEvalSideBreakdown {
    int material;
    int piece_square;
    int pawn_structure;
    int passed_pawns;
    int outposts;
    int bishop_quality;
    int rook_files;
    int mobility;
    int bishop_pair;
    int king_safety_penalty;
    int hanging_penalty;
    int queen_trap_penalty;
    int total;
} ChessEvalSideBreakdown;

typedef struct ChessEvalBreakdown {
    int phase;
    ChessEvalSideBreakdown white;
    ChessEvalSideBreakdown black;
    int score_cp_white;
    int score_cp_stm;
} ChessEvalBreakdown;

typedef enum ChessEngineRequestKind {
    CHESS_ENGINE_REQUEST_BEST_MOVE = 0,
    CHESS_ENGINE_REQUEST_EVAL_FAST = 1,
    CHESS_ENGINE_REQUEST_EVAL_DEEP = 2,
} ChessEngineRequestKind;

typedef struct ChessEngineRequest {
    ChessEngineRequestKind kind;
    int think_time_ms;
    int max_depth;
} ChessEngineRequest;

typedef struct ChessEngineResponse {
    bool ok;
    Move best_move;
    bool found_move;
    bool used_opening_book;
    int score_cp_white;
    int score_cp_stm;
    int depth_reached;
    uint64_t nodes;
    int elapsed_ms;
} ChessEngineResponse;

int chess_ai_eval_cp(const GameState *state);
int chess_ai_eval_fast_cp(const GameState *state);
bool chess_ai_eval_breakdown(const GameState *state, ChessEvalBreakdown *out);
bool chess_ai_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out);
bool chess_engine_query(const GameState *state, const ChessEngineRequest *req, ChessEngineResponse *out);

bool chess_ai_opening_book_set_path(const char *path);
bool chess_ai_opening_book_is_loaded(void);
const char *chess_ai_opening_book_path(void);

bool chess_ai_set_backend(ChessAiBackend backend);
ChessAiBackend chess_ai_get_backend(void);
const char *chess_ai_backend_name(ChessAiBackend backend);
bool chess_ai_set_nn_model_path(const char *path);
bool chess_ai_nn_model_is_loaded(void);
const char *chess_ai_nn_model_path(void);

#endif
