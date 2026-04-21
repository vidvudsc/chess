#ifndef STOCKFISH_EVAL_WORKER_H
#define STOCKFISH_EVAL_WORKER_H

#include <stdbool.h>
#include <stdint.h>

#include "chess_rules.h"

typedef struct StockfishEvalConfig {
    int think_time_ms;
} StockfishEvalConfig;

typedef struct StockfishEvalResult {
    bool has_score;
    bool is_mate;
    int score_cp;      // White-perspective centipawns
    int mate_in;       // White-perspective mate distance in plies (sign indicates side)
    int depth;
    int elapsed_ms;
    uint64_t token;
} StockfishEvalResult;

typedef struct StockfishEvalWorker {
    bool initialized;
    void *impl;
} StockfishEvalWorker;

bool stockfish_eval_worker_init(StockfishEvalWorker *worker);
void stockfish_eval_worker_shutdown(StockfishEvalWorker *worker);
bool stockfish_eval_worker_request(StockfishEvalWorker *worker,
                                   const GameState *state,
                                   const StockfishEvalConfig *cfg,
                                   uint64_t token);
bool stockfish_eval_worker_poll(StockfishEvalWorker *worker, StockfishEvalResult *out_result);
bool stockfish_eval_worker_is_busy(StockfishEvalWorker *worker);
bool stockfish_eval_worker_is_available(const StockfishEvalWorker *worker);

#endif
