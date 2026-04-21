#include <assert.h>
#include <stdio.h>

#include "chess_io.h"

static void must(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        assert(cond);
    }
}

int main(void) {
    GameState s;
    MatchConfig cfg = {
        .clock_enabled = true,
        .initial_ms = 1000,
        .increment_ms = 200,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    chess_init(&s, &cfg);

    must(s.clock_ms[PIECE_WHITE] == 1000, "White starts with configured time");
    must(s.clock_ms[PIECE_BLACK] == 1000, "Black starts with configured time");

    chess_tick_clock(&s, 500);
    must(s.clock_ms[PIECE_WHITE] == 500, "White clock should decrement");

    Move e2e4;
    must(chess_move_from_uci(&s, "e2e4", &e2e4), "Find e2e4");
    must(chess_make_move(&s, e2e4), "Play e2e4");
    must(s.clock_ms[PIECE_WHITE] == 700, "Increment should be applied to mover");

    chess_tick_clock(&s, 1200);
    must(s.clock_ms[PIECE_BLACK] == 0, "Black should flag");
    must(s.result == GAME_RESULT_WIN_TIMEOUT, "Timeout should end game");

    printf("test_clock: OK\n");
    return 0;
}
