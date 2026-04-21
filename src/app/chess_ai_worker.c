#include "chess_ai_worker.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct AiWorkerImpl {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    bool running;
    bool has_job;
    bool busy;
    bool result_ready;
    bool progress_ready;

    GameState job_state;
    AiSearchConfig job_cfg;
    AiSearchResult result;
    AiSearchResult progress;
} AiWorkerImpl;

static void ai_worker_info_callback(int depth,
                                    int score_cp,
                                    Move best_move,
                                    uint64_t nodes,
                                    int elapsed_ms,
                                    void *user_data) {
    AiWorkerImpl *impl = (AiWorkerImpl *)user_data;
    if (impl == NULL) {
        return;
    }

    pthread_mutex_lock(&impl->mutex);
    impl->progress.best_move = best_move;
    impl->progress.found_move = (best_move != 0);
    impl->progress.used_opening_book = false;
    impl->progress.score_cp = score_cp;
    impl->progress.depth_reached = depth;
    impl->progress.nodes = nodes;
    impl->progress.elapsed_ms = elapsed_ms;
    impl->progress_ready = true;
    pthread_mutex_unlock(&impl->mutex);
}

static void *ai_worker_thread_main(void *arg) {
    AiWorkerImpl *impl = (AiWorkerImpl *)arg;

    for (;;) {
        pthread_mutex_lock(&impl->mutex);
        while (impl->running && !impl->has_job) {
            pthread_cond_wait(&impl->cond, &impl->mutex);
        }

        if (!impl->running) {
            pthread_mutex_unlock(&impl->mutex);
            break;
        }

        GameState state = impl->job_state;
        AiSearchConfig cfg = impl->job_cfg;
        impl->has_job = false;
        impl->busy = true;
        impl->result_ready = false;
        impl->progress_ready = false;
        memset(&impl->progress, 0, sizeof(impl->progress));
        pthread_mutex_unlock(&impl->mutex);

        AiSearchResult result;
        memset(&result, 0, sizeof(result));
        (void)chess_ai_pick_move(&state, &cfg, &result);

        pthread_mutex_lock(&impl->mutex);
        impl->result = result;
        impl->busy = false;
        impl->result_ready = true;
        pthread_mutex_unlock(&impl->mutex);
    }

    return NULL;
}

bool ai_worker_init(AiWorker *worker) {
    if (worker == NULL) {
        return false;
    }

    memset(worker, 0, sizeof(*worker));

    AiWorkerImpl *impl = (AiWorkerImpl *)calloc(1, sizeof(*impl));
    if (impl == NULL) {
        return false;
    }

    if (pthread_mutex_init(&impl->mutex, NULL) != 0) {
        free(impl);
        return false;
    }
    if (pthread_cond_init(&impl->cond, NULL) != 0) {
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return false;
    }

    impl->running = true;

    pthread_attr_t attr;
    bool attr_ok = (pthread_attr_init(&attr) == 0);
    if (attr_ok) {
        size_t stack_size = 8u * 1024u * 1024u;
        (void)pthread_attr_setstacksize(&attr, stack_size);
    }

    int create_rc = pthread_create(&impl->thread, attr_ok ? &attr : NULL, ai_worker_thread_main, impl);
    if (attr_ok) {
        pthread_attr_destroy(&attr);
    }

    if (create_rc != 0) {
        pthread_cond_destroy(&impl->cond);
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return false;
    }

    worker->impl = impl;
    worker->initialized = true;
    return true;
}

void ai_worker_shutdown(AiWorker *worker) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return;
    }

    AiWorkerImpl *impl = (AiWorkerImpl *)worker->impl;

    pthread_mutex_lock(&impl->mutex);
    impl->running = false;
    impl->has_job = false;
    pthread_cond_signal(&impl->cond);
    pthread_mutex_unlock(&impl->mutex);

    pthread_join(impl->thread, NULL);
    pthread_cond_destroy(&impl->cond);
    pthread_mutex_destroy(&impl->mutex);

    free(impl);
    worker->impl = NULL;
    worker->initialized = false;
}

bool ai_worker_request(AiWorker *worker, const GameState *state, const AiSearchConfig *cfg) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL || state == NULL) {
        return false;
    }

    AiWorkerImpl *impl = (AiWorkerImpl *)worker->impl;

    pthread_mutex_lock(&impl->mutex);
    if (!impl->running || impl->busy || impl->has_job) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    impl->job_state = *state;
    if (cfg != NULL) {
        impl->job_cfg = *cfg;
    } else {
        impl->job_cfg.think_time_ms = 700;
        impl->job_cfg.max_depth = 8;
        impl->job_cfg.info_callback = NULL;
        impl->job_cfg.info_user_data = NULL;
    }
    impl->job_cfg.info_callback = ai_worker_info_callback;
    impl->job_cfg.info_user_data = impl;

    impl->result_ready = false;
    impl->progress_ready = false;
    memset(&impl->progress, 0, sizeof(impl->progress));
    impl->has_job = true;
    pthread_cond_signal(&impl->cond);
    pthread_mutex_unlock(&impl->mutex);

    return true;
}

bool ai_worker_poll(AiWorker *worker, AiSearchResult *out_result) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return false;
    }

    AiWorkerImpl *impl = (AiWorkerImpl *)worker->impl;

    pthread_mutex_lock(&impl->mutex);
    if (!impl->result_ready) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    if (out_result != NULL) {
        *out_result = impl->result;
    }
    impl->result_ready = false;
    pthread_mutex_unlock(&impl->mutex);
    return true;
}

bool ai_worker_get_progress(AiWorker *worker, AiSearchResult *out_result) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL || out_result == NULL) {
        return false;
    }

    AiWorkerImpl *impl = (AiWorkerImpl *)worker->impl;

    pthread_mutex_lock(&impl->mutex);
    if (!impl->progress_ready) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    *out_result = impl->progress;
    pthread_mutex_unlock(&impl->mutex);
    return true;
}

bool ai_worker_is_busy(AiWorker *worker) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return false;
    }

    AiWorkerImpl *impl = (AiWorkerImpl *)worker->impl;

    pthread_mutex_lock(&impl->mutex);
    bool busy = impl->busy || impl->has_job;
    pthread_mutex_unlock(&impl->mutex);
    return busy;
}
