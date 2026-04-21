#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

#include "chess_io.h"

typedef struct PerftCase {
    const char *name;
    const char *fen;
    int depth;
    uint64_t expected;
} PerftCase;

static void must(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        assert(cond);
    }
}

int main(void) {
    static const PerftCase cases[] = {
        {"Kiwipete d1", "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 1, 45ULL},
        {"Kiwipete d2", "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 2, 1947ULL},
        {"Kiwipete d3", "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 3, 85877ULL},

        {"Position3 d1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14ULL},
        {"Position3 d2", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191ULL},
        {"Position3 d3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812ULL},

        {"Position4 d1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6ULL},
        {"Position4 d2", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264ULL},
        {"Position4 d3", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467ULL},

        {"Position5 d1", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44ULL},
        {"Position5 d2", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486ULL},
        {"Position5 d3", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379ULL},

        {"Position6 d1", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10", 1, 44ULL},
        {"Position6 d2", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10", 2, 1987ULL},
        {"Position6 d3", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10", 3, 83034ULL},
    };

    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    GameState s;
    chess_init(&s, &cfg);

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const PerftCase *tc = &cases[i];
        char err[128] = {0};
        must(chess_load_fen(&s, tc->fen, err, sizeof(err)), tc->name);

        uint64_t nodes = chess_perft(&s, tc->depth);
        if (nodes != tc->expected) {
            fprintf(stderr,
                    "%s mismatch: got=%" PRIu64 " expected=%" PRIu64 "\n",
                    tc->name,
                    nodes,
                    tc->expected);
            return 1;
        }
    }

    printf("test_perft_suite: OK\n");
    return 0;
}
