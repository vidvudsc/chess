#include "stockfish_eval_worker.h"

#include <ctype.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "chess_io.h"

#define STOCKFISH_LINE_MAX 1024
#define STOCKFISH_MATE_CP 30000

typedef struct StockfishEvalWorkerImpl {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    bool running;
    bool has_job;
    bool busy;
    bool result_ready;
    bool available;
    int consecutive_failures;
    int64_t retry_after_ms;

    char binary_path[512];
    char job_fen[256];
    int job_side_to_move;
    int job_think_time_ms;
    uint64_t job_token;

    StockfishEvalResult result;

    bool cache_ready;
    char cache_fen[256];
    int cache_think_time_ms;
    StockfishEvalResult cache_result;
    int64_t cache_time_ms;
} StockfishEvalWorkerImpl;

static int64_t now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

static void trim_trailing_whitespace(char *s) {
    if (s == NULL) {
        return;
    }

    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[len - 1] = '\0';
        len -= 1;
    }
}

static bool resolve_stockfish_binary(char out_path[512]) {
    if (out_path == NULL) {
        return false;
    }

    const char *env_path = getenv("CHESS_STOCKFISH_BIN");
    if (env_path != NULL && env_path[0] != '\0') {
        strncpy(out_path, env_path, 511);
        out_path[511] = '\0';
        return true;
    }

    static const char *k_candidates[] = {
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    };

    for (size_t i = 0; i < sizeof(k_candidates) / sizeof(k_candidates[0]); ++i) {
        if (access(k_candidates[i], X_OK) == 0) {
            strncpy(out_path, k_candidates[i], 511);
            out_path[511] = '\0';
            return true;
        }
    }

    FILE *fp = popen("command -v stockfish 2>/dev/null", "r");
    if (fp == NULL) {
        return false;
    }

    char line[512];
    bool ok = false;
    if (fgets(line, sizeof(line), fp) != NULL) {
        trim_trailing_whitespace(line);
        if (line[0] != '\0') {
            strncpy(out_path, line, 511);
            out_path[511] = '\0';
            ok = true;
        }
    }

    pclose(fp);
    return ok;
}

static bool parse_info_line(const char *line, int *out_depth, bool *out_mate, int *out_score_value) {
    if (line == NULL || out_depth == NULL || out_mate == NULL || out_score_value == NULL) {
        return false;
    }
    if (strncmp(line, "info ", 5) != 0) {
        return false;
    }

    char copy[STOCKFISH_LINE_MAX];
    strncpy(copy, line, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';

    int depth = -1;
    bool has_score = false;
    bool is_mate = false;
    int score_value = 0;

    char *saveptr = NULL;
    char *tok = strtok_r(copy, " \t\r\n", &saveptr);
    while (tok != NULL) {
        if (strcmp(tok, "depth") == 0) {
            char *v = strtok_r(NULL, " \t\r\n", &saveptr);
            if (v != NULL) {
                depth = atoi(v);
            }
        } else if (strcmp(tok, "score") == 0) {
            char *kind = strtok_r(NULL, " \t\r\n", &saveptr);
            char *value = strtok_r(NULL, " \t\r\n", &saveptr);
            if (kind != NULL && value != NULL) {
                if (strcmp(kind, "cp") == 0) {
                    has_score = true;
                    is_mate = false;
                    score_value = atoi(value);
                } else if (strcmp(kind, "mate") == 0) {
                    has_score = true;
                    is_mate = true;
                    score_value = atoi(value);
                }
            }
        }
        tok = strtok_r(NULL, " \t\r\n", &saveptr);
    }

    if (!has_score) {
        return false;
    }

    *out_depth = depth;
    *out_mate = is_mate;
    *out_score_value = score_value;
    return true;
}

static bool run_stockfish_eval(const char *binary_path,
                               const char *fen,
                               int side_to_move,
                               int think_time_ms,
                               uint64_t token,
                               StockfishEvalResult *out) {
    if (binary_path == NULL || fen == NULL || out == NULL) {
        return false;
    }

    int in_pipe[2] = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) != 0) {
        return false;
    }
    if (pipe(out_pipe) != 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        return false;
    }

    if (pid == 0) {
        (void)dup2(in_pipe[0], STDIN_FILENO);
        (void)dup2(out_pipe[1], STDOUT_FILENO);
        (void)dup2(out_pipe[1], STDERR_FILENO);

        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);

        execlp(binary_path, binary_path, (char *)NULL);
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);

    FILE *sf_in = fdopen(in_pipe[1], "w");
    FILE *sf_out = fdopen(out_pipe[0], "r");
    if (sf_in == NULL || sf_out == NULL) {
        if (sf_in != NULL) {
            fclose(sf_in);
        } else {
            close(in_pipe[1]);
        }
        if (sf_out != NULL) {
            fclose(sf_out);
        } else {
            close(out_pipe[0]);
        }
        (void)waitpid(pid, NULL, 0);
        return false;
    }

    int bounded_time = think_time_ms;
    if (bounded_time < 20) {
        bounded_time = 20;
    }
    if (bounded_time > 5000) {
        bounded_time = 5000;
    }

    int64_t started = now_ms();

    fprintf(sf_in, "uci\n");
    fprintf(sf_in, "setoption name Threads value 1\n");
    fprintf(sf_in, "setoption name Hash value 8\n");
    fprintf(sf_in, "isready\n");
    fprintf(sf_in, "position fen %s\n", fen);
    fprintf(sf_in, "go movetime %d\n", bounded_time);
    fflush(sf_in);

    bool have_score = false;
    bool score_is_mate = false;
    int score_value = 0;
    int best_depth = -1;

    char line[STOCKFISH_LINE_MAX];
    while (fgets(line, sizeof(line), sf_out) != NULL) {
        int line_depth = -1;
        bool line_is_mate = false;
        int line_score = 0;

        if (parse_info_line(line, &line_depth, &line_is_mate, &line_score)) {
            if (!have_score || line_depth >= best_depth) {
                have_score = true;
                best_depth = line_depth;
                score_is_mate = line_is_mate;
                score_value = line_score;
            }
        }

        if (strncmp(line, "bestmove ", 9) == 0) {
            break;
        }
    }

    fprintf(sf_in, "quit\n");
    fflush(sf_in);
    fclose(sf_in);
    fclose(sf_out);

    int wait_status = 0;
    (void)waitpid(pid, &wait_status, 0);

    if (!have_score) {
        return false;
    }

    int white_cp = 0;
    int white_mate = 0;
    if (score_is_mate) {
        int mate_plies = score_value;
        int mate_sign = (mate_plies > 0) ? 1 : -1;
        int mate_dist = abs(mate_plies);
        if (mate_dist == 0) {
            mate_dist = 1;
        }
        int mapped = STOCKFISH_MATE_CP - (mate_dist - 1) * 100;
        if (mapped < 1000) {
            mapped = 1000;
        }
        int stm_cp = mate_sign * mapped;
        if (side_to_move == PIECE_WHITE) {
            white_cp = stm_cp;
            white_mate = mate_plies;
        } else {
            white_cp = -stm_cp;
            white_mate = -mate_plies;
        }
    } else {
        int stm_cp = score_value;
        white_cp = (side_to_move == PIECE_WHITE) ? stm_cp : -stm_cp;
        white_mate = 0;
    }

    memset(out, 0, sizeof(*out));
    out->has_score = true;
    out->is_mate = score_is_mate;
    out->score_cp = white_cp;
    out->mate_in = white_mate;
    out->depth = best_depth;
    out->elapsed_ms = (int)(now_ms() - started);
    out->token = token;
    return true;
}

static void *stockfish_worker_thread_main(void *arg) {
    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)arg;

    for (;;) {
        pthread_mutex_lock(&impl->mutex);
        while (impl->running && !impl->has_job) {
            pthread_cond_wait(&impl->cond, &impl->mutex);
        }

        if (!impl->running) {
            pthread_mutex_unlock(&impl->mutex);
            break;
        }

        char fen[256];
        int side = impl->job_side_to_move;
        int think_time_ms = impl->job_think_time_ms;
        uint64_t token = impl->job_token;

        strncpy(fen, impl->job_fen, sizeof(fen) - 1);
        fen[sizeof(fen) - 1] = '\0';

        impl->has_job = false;
        impl->busy = true;
        impl->result_ready = false;
        pthread_mutex_unlock(&impl->mutex);

        StockfishEvalResult result;
        bool ok = run_stockfish_eval(impl->binary_path, fen, side, think_time_ms, token, &result);

        pthread_mutex_lock(&impl->mutex);
        if (!ok) {
            impl->consecutive_failures += 1;
            if (impl->consecutive_failures >= 3) {
                impl->available = false;
                impl->retry_after_ms = now_ms() + 5000;
            }
            impl->busy = false;
            impl->result_ready = false;
            pthread_mutex_unlock(&impl->mutex);
            continue;
        }

        impl->consecutive_failures = 0;
        impl->available = true;
        impl->retry_after_ms = 0;
        impl->result = result;
        impl->cache_ready = true;
        strncpy(impl->cache_fen, fen, sizeof(impl->cache_fen) - 1);
        impl->cache_fen[sizeof(impl->cache_fen) - 1] = '\0';
        impl->cache_think_time_ms = think_time_ms;
        impl->cache_result = result;
        impl->cache_time_ms = now_ms();
        impl->busy = false;
        impl->result_ready = true;
        pthread_mutex_unlock(&impl->mutex);
    }

    return NULL;
}

bool stockfish_eval_worker_init(StockfishEvalWorker *worker) {
    if (worker == NULL) {
        return false;
    }

    memset(worker, 0, sizeof(*worker));

    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)calloc(1, sizeof(*impl));
    if (impl == NULL) {
        return false;
    }

    if (!resolve_stockfish_binary(impl->binary_path)) {
        free(impl);
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
    impl->available = true;

    if (pthread_create(&impl->thread, NULL, stockfish_worker_thread_main, impl) != 0) {
        pthread_cond_destroy(&impl->cond);
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return false;
    }

    worker->impl = impl;
    worker->initialized = true;
    return true;
}

void stockfish_eval_worker_shutdown(StockfishEvalWorker *worker) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return;
    }

    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)worker->impl;

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

bool stockfish_eval_worker_request(StockfishEvalWorker *worker,
                                   const GameState *state,
                                   const StockfishEvalConfig *cfg,
                                   uint64_t token) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL || state == NULL) {
        return false;
    }

    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)worker->impl;

    char fen[256];
    chess_export_fen(state, fen, sizeof(fen));
    int requested_think_ms = (cfg != NULL && cfg->think_time_ms > 0) ? cfg->think_time_ms : 80;

    pthread_mutex_lock(&impl->mutex);
    if (!impl->running) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    // Fast path: repeated requests for the same position can reuse a very
    // recent completed score instead of spawning another Stockfish process.
    int64_t now = now_ms();
    if (!impl->busy && !impl->has_job && impl->cache_ready &&
        strcmp(impl->cache_fen, fen) == 0 &&
        impl->cache_think_time_ms == requested_think_ms &&
        (now - impl->cache_time_ms) <= 3000) {
        impl->result = impl->cache_result;
        impl->result.token = token;
        impl->result_ready = true;
        pthread_mutex_unlock(&impl->mutex);
        return true;
    }

    if (!impl->available && impl->retry_after_ms > 0 && now_ms() >= impl->retry_after_ms) {
        impl->available = true;
        impl->consecutive_failures = 0;
        impl->retry_after_ms = 0;
    }
    if (!impl->available || impl->busy || impl->has_job) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    strncpy(impl->job_fen, fen, sizeof(impl->job_fen) - 1);
    impl->job_fen[sizeof(impl->job_fen) - 1] = '\0';
    impl->job_side_to_move = state->side_to_move;
    impl->job_think_time_ms = requested_think_ms;
    impl->job_token = token;
    impl->result_ready = false;
    impl->has_job = true;
    pthread_cond_signal(&impl->cond);
    pthread_mutex_unlock(&impl->mutex);

    return true;
}

bool stockfish_eval_worker_poll(StockfishEvalWorker *worker, StockfishEvalResult *out_result) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return false;
    }

    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)worker->impl;

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

bool stockfish_eval_worker_is_busy(StockfishEvalWorker *worker) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return false;
    }

    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)worker->impl;
    pthread_mutex_lock(&impl->mutex);
    bool busy = impl->busy || impl->has_job;
    pthread_mutex_unlock(&impl->mutex);
    return busy;
}

bool stockfish_eval_worker_is_available(const StockfishEvalWorker *worker) {
    if (worker == NULL || !worker->initialized || worker->impl == NULL) {
        return false;
    }

    StockfishEvalWorkerImpl *impl = (StockfishEvalWorkerImpl *)worker->impl;
    pthread_mutex_lock(&impl->mutex);
    if (!impl->available && impl->retry_after_ms > 0 && now_ms() >= impl->retry_after_ms) {
        impl->available = true;
        impl->consecutive_failures = 0;
        impl->retry_after_ms = 0;
    }
    bool available = impl->available;
    pthread_mutex_unlock(&impl->mutex);
    return available;
}
