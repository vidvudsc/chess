#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chess_io.h"
#include "chess_state.h"
#include "nn_eval.h"

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s <nn_eval.bin> <fen> [<fen> ...]\n", argv0);
    fprintf(stderr, "       %s <nn_eval.bin> --bench <iterations> <fen>\n", argv0);
    fprintf(stderr, "       %s <nn_eval.bin> --sequence <fen> <uci> [<uci> ...]\n", argv0);
}

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    if (!nn_eval_load_model(argv[1])) {
        fprintf(stderr, "failed to load NN model: %s\n", argv[1]);
        return 1;
    }

    if (argc == 5 && strcmp(argv[2], "--bench") == 0) {
        char *end = NULL;
        long iterations = strtol(argv[3], &end, 10);
        if (end == argv[3] || *end != '\0' || iterations <= 0) {
            usage(argv[0]);
            return 2;
        }
        GameState state;
        NnAccumulatorFrame frame;
        char err[160];
        memset(&state, 0, sizeof(state));
        memset(&frame, 0, sizeof(frame));
        if (!chess_load_fen(&state, argv[4], err, sizeof(err)) ||
            !nn_eval_build_frame(&state, &frame)) {
            fprintf(stderr, "failed to initialize benchmark: %s\n", err);
            return 1;
        }
        volatile int checksum = 0;
        struct timespec start;
        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (long i = 0; i < iterations; ++i) {
            checksum += nn_eval_cp_stm_from_frame(&state, &frame);
        }
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double seconds = elapsed_seconds(start, end_time);
        printf("iterations=%ld seconds=%.6f evals_per_second=%.0f checksum=%d\n",
               iterations,
               seconds,
               (double)iterations / seconds,
               checksum);
        return 0;
    }

    if (argc >= 5 && strcmp(argv[2], "--sequence") == 0) {
        GameState state;
        NnAccumulatorFrame frame;
        char err[160];
        memset(&state, 0, sizeof(state));
        memset(&frame, 0, sizeof(frame));
        if (!chess_load_fen(&state, argv[3], err, sizeof(err)) ||
            !nn_eval_build_frame(&state, &frame)) {
            fprintf(stderr, "failed to initialize sequence: %s\n", err);
            return 1;
        }
        for (int i = 4; i < argc; ++i) {
            Move move;
            NnAccumulatorFrame next;
            memset(&next, 0, sizeof(next));
            if (!chess_move_from_uci(&state, argv[i], &move) || !chess_make_move(&state, move)) {
                fprintf(stderr, "invalid sequence move: %s\n", argv[i]);
                return 1;
            }
            const UndoRecord *undo = &state.undo_stack[state.ply - 1];
            if (!nn_eval_update_frame(&state, undo, &frame, &next)) {
                fprintf(stderr, "incremental update failed after: %s\n", argv[i]);
                return 1;
            }
            int incremental = nn_eval_cp_stm_from_frame(&state, &next);
            int rebuilt = nn_eval_cp_stm(&state);
            if (incremental != rebuilt) {
                fprintf(stderr, "incremental mismatch after %s: %d != %d\n",
                        argv[i], incremental, rebuilt);
                return 1;
            }
            frame = next;
            printf("%s\t%d\n", argv[i], incremental);
        }
        return 0;
    }

    for (int i = 2; i < argc; ++i) {
        GameState state;
        char err[160];
        memset(&state, 0, sizeof(state));
        if (!chess_load_fen(&state, argv[i], err, sizeof(err))) {
            fprintf(stderr, "invalid FEN: %s: %s\n", argv[i], err);
            return 1;
        }
        printf("%d\t%s\n", nn_eval_cp_stm(&state), argv[i]);
    }
    return 0;
}
