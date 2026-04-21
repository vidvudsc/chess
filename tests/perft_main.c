#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#include "chess_io.h"

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1000000.0;
}

int main(void) {
    GameState s;
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    const uint64_t expected[6] = {0, 20ULL, 400ULL, 8902ULL, 197281ULL, 4865609ULL};

    chess_init(&s, &cfg);
    for (int d = 1; d <= 5; ++d) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t nodes = chess_perft(&s, d);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = elapsed_ms(t0, t1);
        double nps = ms > 0.0 ? ((double)nodes * 1000.0 / ms) : 0.0;

        printf("depth %d: nodes=%" PRIu64 " expected=%" PRIu64 " time=%.2fms nps=%.0f\n",
               d,
               nodes,
               expected[d],
               ms,
               nps);

        if (nodes != expected[d]) {
            fprintf(stderr, "Perft mismatch at depth %d\n", d);
            return 1;
        }
    }

    return 0;
}
