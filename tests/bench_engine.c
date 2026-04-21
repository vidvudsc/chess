#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#include "chess_io.h"

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1000000.0;
}

static void bench_perft(const char *name, const char *fen, int depth) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    GameState s;
    chess_init(&s, &cfg);
    if (fen != NULL) {
        char err[128] = {0};
        if (!chess_load_fen(&s, fen, err, sizeof(err))) {
            fprintf(stderr, "bench_perft load fen failed: %s\n", err);
            return;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t nodes = chess_perft(&s, depth);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = elapsed_ms(t0, t1);
    double nps = ms > 0.0 ? ((double)nodes * 1000.0 / ms) : 0.0;

    printf("%s: depth=%d nodes=%" PRIu64 " time=%.2fms nps=%.0f\n", name, depth, nodes, ms, nps);
}

static void bench_movegen_cycles(void) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    GameState s;
    chess_init(&s, &cfg);

    const int movegen_iters = 3000000;
    const int makeundo_iters = 1000000;
    Move moves[CHESS_MAX_MOVES];
    uint64_t count_sum = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < movegen_iters; ++i) {
        count_sum += (uint64_t)chess_generate_legal_moves(&s, moves);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double mg_ms = elapsed_ms(t0, t1);
    double mg_calls_s = mg_ms > 0.0 ? ((double)movegen_iters * 1000.0 / mg_ms) : 0.0;
    printf("movegen_only: iters=%d time=%.2fms calls/s=%.0f avg_legal=%.2f\n",
           movegen_iters,
           mg_ms,
           mg_calls_s,
           (double)count_sum / (double)movegen_iters);

    count_sum = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < makeundo_iters; ++i) {
        int n = chess_generate_legal_moves(&s, moves);
        if (n > 0) {
            Move m = moves[i % n];
            chess_make_move(&s, m);
            chess_undo_move(&s);
        }
        count_sum += (uint64_t)n;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double mu_ms = elapsed_ms(t0, t1);
    double mu_cycles_s = mu_ms > 0.0 ? ((double)makeundo_iters * 1000.0 / mu_ms) : 0.0;
    printf("make_undo_cycle: iters=%d time=%.2fms cycles/s=%.0f avg_legal=%.2f\n",
           makeundo_iters,
           mu_ms,
           mu_cycles_s,
           (double)count_sum / (double)makeundo_iters);
}

static void load_bench_position(GameState *s, const char *fen) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    chess_init(s, &cfg);
    if (fen == NULL) {
        return;
    }

    char err[128] = {0};
    if (!chess_load_fen(s, fen, err, sizeof(err))) {
        fprintf(stderr, "load_bench_position failed: %s\n", err);
    }
}

static void bench_movegen_case(const char *name, const char *fen, int iters) {
    GameState s;
    load_bench_position(&s, fen);

    Move moves[CHESS_MAX_MOVES];
    uint64_t count_sum = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        count_sum += (uint64_t)chess_generate_legal_moves(&s, moves);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double copy_ms = elapsed_ms(t0, t1);
    double copy_calls_s = copy_ms > 0.0 ? ((double)iters * 1000.0 / copy_ms) : 0.0;

    count_sum = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        count_sum += (uint64_t)chess_generate_legal_moves_mut(&s, moves);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double mut_ms = elapsed_ms(t0, t1);
    double mut_calls_s = mut_ms > 0.0 ? ((double)iters * 1000.0 / mut_ms) : 0.0;

    count_sum = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        int n = chess_generate_legal_moves_mut(&s, moves);
        if (n > 0) {
            Move m = moves[i % n];
            chess_make_move_trusted(&s, m);
            chess_undo_move(&s);
        }
        count_sum += (uint64_t)n;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cycle_ms = elapsed_ms(t0, t1);
    double cycle_calls_s = cycle_ms > 0.0 ? ((double)iters * 1000.0 / cycle_ms) : 0.0;

    printf("%s: iters=%d legal(copy)=%.0f/s legal(mut)=%.0f/s makeundo=%.0f/s avg_legal=%.2f\n",
           name,
           iters,
           copy_calls_s,
           mut_calls_s,
           cycle_calls_s,
           (double)count_sum / (double)iters);
}

static void bench_movegen_matrix(void) {
    static const struct {
        const char *name;
        const char *fen;
        int iters;
    } cases[] = {
        {"mg_start", NULL, 800000},
        {"mg_kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 500000},
        {"check_evasion", "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", 700000},
        {"endgame", "8/2r5/1k6/8/4PP2/PP5P/1K1R4/8 w - - 5 43", 900000},
        {"promotion_race", "8/P6k/8/8/8/8/6Kp/8 w - - 0 1", 900000},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        bench_movegen_case(cases[i].name, cases[i].fen, cases[i].iters);
    }
}

int main(void) {
    printf("engine bench start\n");
    bench_perft("startpos", NULL, 5);
    bench_perft("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 4);
    bench_movegen_cycles();
    bench_movegen_matrix();
    return 0;
}
