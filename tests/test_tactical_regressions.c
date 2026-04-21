#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "chess_ai.h"
#include "chess_io.h"

typedef struct TacticalRegression {
    const char *name;
    const char *fen;
    int think_time_ms;
    int max_depth;
    const char *avoid_uci;
    bool require_searched_move;
} TacticalRegression;

static void must(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        assert(cond);
    }
}

static void run_regression(const TacticalRegression *tc, GameState *state, const MatchConfig *cfg) {
    char err[128] = {0};
    AiSearchConfig ai_cfg = {
        .think_time_ms = tc->think_time_ms,
        .max_depth = tc->max_depth,
    };
    AiSearchResult result;
    char best_uci[6] = {0};

    chess_init(state, cfg);
    must(chess_load_fen(state, tc->fen, err, sizeof(err)), tc->name);
    must(chess_ai_pick_move(state, &ai_cfg, &result), tc->name);
    must(result.found_move, tc->name);
    must(chess_is_move_legal(state, result.best_move), tc->name);
    must(chess_move_to_uci(result.best_move, best_uci), tc->name);
    must(strcmp(best_uci, tc->avoid_uci) != 0, tc->name);
    if (tc->require_searched_move) {
        must(result.depth_reached > 0 || result.nodes > 0, tc->name);
    }
}

int main(void) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };
    GameState state;

    static const TacticalRegression regressions[] = {
        {
            .name = "Avoid depth-0 fallback queen-hanging wait move",
            .fen = "8/kr2Q3/p7/8/4P3/PPp5/2K2PPP/3R4 w - - 0 36",
            .think_time_ms = 40,
            .max_depth = 8,
            .avoid_uci = "f2f3",
            .require_searched_move = true,
        },
        {
            .name = "Avoid queen-hanging b2b3 blunder",
            .fen = "4k3/pp4Rp/2p5/1q3Q2/8/4P3/1PPK1PPP/8 w - - 3 28",
            .think_time_ms = 80,
            .max_depth = 8,
            .avoid_uci = "b2b3",
            .require_searched_move = true,
        },
        {
            .name = "Avoid immediate stalemate blunder b4b3",
            .fen = "8/6k1/p7/5nK1/1p3P2/4n3/8/3q4 b - - 3 52",
            .think_time_ms = 80,
            .max_depth = 8,
            .avoid_uci = "b4b3",
            .require_searched_move = true,
        },
        {
            .name = "Avoid losing king walk c8d7 in pawn-race defense",
            .fen = "2k5/5pB1/8/1P3P1P/3NK3/1P6/8/8 b - - 2 49",
            .think_time_ms = 200,
            .max_depth = 10,
            .avoid_uci = "c8d7",
            .require_searched_move = true,
        },
    };

    for (size_t i = 0; i < sizeof(regressions) / sizeof(regressions[0]); ++i) {
        run_regression(&regressions[i], &state, &cfg);
    }

    printf("test_tactical_regressions: OK\n");
    return 0;
}
