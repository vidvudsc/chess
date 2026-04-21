#ifndef AI_TEST_RUNNER_H
#define AI_TEST_RUNNER_H

#include <stdbool.h>
#include <stdint.h>

#include "chess_ai.h"

enum {
    AI_TEST_MAX_BRACKETS = 8,
};

typedef enum AiTestMode {
    AI_TEST_MODE_VS_STOCKFISH = 0,
    AI_TEST_MODE_INTERNAL = 1,
} AiTestMode;

typedef struct AiTestConfig {
    AiTestMode mode;
    ChessAiBackend our_backend;
    ChessAiBackend opponent_backend;
    uint64_t random_seed;
    int total_games;
    int our_think_ms;
    int stockfish_think_ms;
    int stockfish_skill_level;
    bool stockfish_ladder_enabled;
    int max_plies_per_game;
    bool use_lichess_positions;
    char positions_path[512];
    char nn_model_path[1024];
} AiTestConfig;

typedef struct AiTestStatus {
    bool running;
    bool completed;
    bool canceled;
    bool failed;
    char message[160];

    AiTestMode mode;
    ChessAiBackend our_backend;
    ChessAiBackend opponent_backend;

    int total_games;
    int games_completed;
    int current_game;
    int current_ply;
    int current_our_side; // PIECE_WHITE/PIECE_BLACK for "our" side in this game
    int positions_loaded;
    uint64_t seed_used;

    int wins;
    int draws;
    int losses;

    int current_opponent_elo;
    int current_opponent_skill;

    int bracket_count;
    int bracket_elo[AI_TEST_MAX_BRACKETS];
    int bracket_games[AI_TEST_MAX_BRACKETS];
    int bracket_wins[AI_TEST_MAX_BRACKETS];
    int bracket_draws[AI_TEST_MAX_BRACKETS];
    int bracket_losses[AI_TEST_MAX_BRACKETS];

    double score_pct; // global score percentage
    int estimated_elo;
    bool estimated_elo_is_delta;

    int elapsed_ms;
    int eta_ms;

    // Internal-engine search metrics gathered during test games (our side / opponent side).
    // For Stockfish opponents these may remain zero on the opponent side.
    int our_search_moves;
    int opp_search_moves;
    int our_search_depth_sum;
    int opp_search_depth_sum;
    uint64_t our_search_nodes_sum;
    uint64_t opp_search_nodes_sum;
    int64_t our_search_time_ms_sum;
    int64_t opp_search_time_ms_sum;
} AiTestStatus;

typedef struct AiTestRunner {
    bool initialized;
    void *impl;
} AiTestRunner;

bool ai_test_runner_init(AiTestRunner *runner);
void ai_test_runner_shutdown(AiTestRunner *runner);
bool ai_test_runner_start(AiTestRunner *runner, const AiTestConfig *cfg);
void ai_test_runner_stop(AiTestRunner *runner);
bool ai_test_runner_poll(AiTestRunner *runner, AiTestStatus *out_status);
bool ai_test_runner_get_status(AiTestRunner *runner, AiTestStatus *out_status);
bool ai_test_runner_is_running(AiTestRunner *runner);

#endif
