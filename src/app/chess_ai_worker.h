#ifndef CHESS_AI_WORKER_H
#define CHESS_AI_WORKER_H

#include <stdbool.h>

#include "chess_ai.h"

typedef struct AiWorker {
    bool initialized;
    void *impl;
} AiWorker;

bool ai_worker_init(AiWorker *worker);
void ai_worker_shutdown(AiWorker *worker);
bool ai_worker_request(AiWorker *worker, const GameState *state, const AiSearchConfig *cfg);
bool ai_worker_poll(AiWorker *worker, AiSearchResult *out_result);
bool ai_worker_get_progress(AiWorker *worker, AiSearchResult *out_result);
bool ai_worker_is_busy(AiWorker *worker);

#endif
