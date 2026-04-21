#include "ai_test_runner.h"

#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "chess_ai.h"
#include "chess_io.h"
#include "chess_rules.h"

#define AI_TEST_MAX_POSITIONS 2048
#define AI_TEST_FEN_MAX 256
#define AI_TEST_MAX_PLY_FALLBACK 180
#define AI_TEST_STATUS_UPDATE_PLY_MASK 31
#define AI_TEST_STATUS_UPDATE_MIN_MS 250

static const int k_stockfish_elo_ladder[] = {1400, 1600, 1800, 2000, 2200, 2400, 2600};

typedef struct StockfishProc {
    pid_t pid;
    FILE *in;
    FILE *out;
    bool ready;
    bool configured;
    int configured_skill_level;
    bool configured_limit_elo_mode;
    int configured_elo;
} StockfishProc;

typedef struct AiTestRunnerImpl {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    bool running;
    bool has_job;
    bool stop_requested;
    bool dirty;

    AiTestConfig job_cfg;
    AiTestStatus status;
} AiTestRunnerImpl;

static const char *k_default_equal_fens[] = {
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 4 6",
    "r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 4 6",
    "r1bq1rk1/pppn1ppp/3bpn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 9",
    "r2q1rk1/ppp2ppp/2npbn2/3Np3/2B1P3/2N1B3/PPP2PPP/R2Q1RK1 b - - 0 10",
    "r1bq1rk1/pp3ppp/2np1n2/2p1p3/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 b - - 0 9",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 b - - 1 8",
    "r2q1rk1/pp1n1ppp/2pbpn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 2 9",
    "r1bq1rk1/pp2bppp/2np1n2/2p1p3/2B1P3/2NP1N1P/PPP2PP1/R1BQ1RK1 w - - 1 10",
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8",
    "r2q1rk1/pppb1ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 2 9",
    "r1bq1rk1/ppp2ppp/2n5/3np3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 8",
    "r2q1rk1/ppp2ppp/2n5/3np3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 b - - 3 8",
    "r1bq1rk1/pp1n1ppp/2p1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 9",
    "r1bq1rk1/ppp2ppp/2n2n2/2bp4/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 8",
    "r2q1rk1/ppp2ppp/2n2n2/2bp4/4P3/2NP1N2/PPP1QPPP/R1B2RK1 b - - 3 8",
    "r1bq1rk1/pppn1ppp/3b1n2/3pp3/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 8",
    "r1bq1rk1/pppn1ppp/3b1n2/3pp3/3P4/2NBPN2/PPQ1BPPP/R4RK1 b - - 1 8",
    "r2q1rk1/pppn1ppp/3b1n2/3pp3/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - - 2 9",
    "r1bq1rk1/pppn1ppp/3bpn2/3p4/3P4/2NBPN2/PPQ2PPP/2KR1B1R w - - 6 10",
    "r1bq1rk1/pppn1ppp/3bpn2/3p4/3P4/2NBPN2/PPQ2PPP/2KR1B1R b - - 7 10",
    "r1bq1rk1/pppn1ppp/3bpn2/3p4/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - - 8 11",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 9",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B1R1K1 b - - 1 9",
    "r2q1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2NBPN2/PPQ2PPP/R1B1R1K1 w - - 2 10",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 8",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3P4/2NBPN2/PPQ1BPPP/R4RK1 b - - 1 8",
    "r2q1rk1/ppp2ppp/2n2n2/3pp3/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - - 2 9",
    "r1bq1rk1/pp3ppp/2n1pn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 9",
    "r2q1rk1/pp3ppp/2n1pn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B1R1K1 b - - 1 9",
    "r2q1rk1/pp3ppp/2n1pn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B1R1K1 w - - 2 10",
    "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 9",
    "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R4RK1 b - - 1 9",
    "r2q1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - - 2 10",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3P4/2NBPN2/PPQ2PPP/R1B2RK1 b - - 1 8",
    "r2q1rk1/ppp2ppp/2n2n2/3pp3/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 2 9",
};

static int64_t now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

static void trim_ascii(char *s) {
    if (s == NULL) {
        return;
    }
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[len - 1] = '\0';
        len -= 1;
    }
    size_t lead = 0;
    while (s[lead] != '\0' && isspace((unsigned char)s[lead])) {
        lead += 1;
    }
    if (lead > 0) {
        memmove(s, s + lead, strlen(s + lead) + 1);
    }
}

static int count_fen_fields(const char *s) {
    if (s == NULL) {
        return 0;
    }
    int fields = 0;
    bool in_field = false;
    for (const char *p = s; *p != '\0'; ++p) {
        if (isspace((unsigned char)*p)) {
            in_field = false;
            continue;
        }
        if (!in_field) {
            fields += 1;
            in_field = true;
        }
    }
    return fields;
}

static void normalize_fen_for_loader(const char *line, char out[AI_TEST_FEN_MAX]) {
    if (out == NULL) {
        return;
    }
    out[0] = '\0';
    if (line == NULL) {
        return;
    }

    snprintf(out, AI_TEST_FEN_MAX, "%s", line);
    trim_ascii(out);

    // Accept EPD-style 4-field rows by appending default halfmove/fullmove.
    if (count_fen_fields(out) == 4) {
        size_t len = strlen(out);
        if (len + 4 < AI_TEST_FEN_MAX) {
            memcpy(out + len, " 0 1", 5);
        }
    }
}

static double expected_score(double our_elo, double opp_elo) {
    return 1.0 / (1.0 + pow(10.0, (opp_elo - our_elo) / 400.0));
}

static int estimate_elo_delta_single_bracket_smoothed(const AiTestStatus *st) {
    int played = st->wins + st->draws + st->losses;
    if (played <= 0) {
        return 0;
    }

    // Laplace smoothing to avoid unstable early extreme deltas.
    double score_points = (double)st->wins + 0.5 * (double)st->draws;
    double smoothed_score = (score_points + 1.0) / ((double)played + 2.0);
    if (smoothed_score < 0.01) {
        smoothed_score = 0.01;
    } else if (smoothed_score > 0.99) {
        smoothed_score = 0.99;
    }

    double elo = -400.0 * log10((1.0 / smoothed_score) - 1.0);
    if (elo > 1200.0) {
        elo = 1200.0;
    } else if (elo < -1200.0) {
        elo = -1200.0;
    }
    return (int)llround(elo);
}

static int estimate_elo_from_brackets(const AiTestStatus *st) {
    double lo = 600.0;
    double hi = 3400.0;

    bool has_games = false;
    for (int i = 0; i < st->bracket_count; ++i) {
        if (st->bracket_games[i] > 0) {
            has_games = true;
            break;
        }
    }
    if (!has_games) {
        return 0;
    }

    for (int iter = 0; iter < 50; ++iter) {
        double mid = 0.5 * (lo + hi);
        double f = 0.0;
        for (int i = 0; i < st->bracket_count; ++i) {
            int n = st->bracket_games[i];
            if (n <= 0) {
                continue;
            }
            double s = (double)st->bracket_wins[i] + 0.5 * (double)st->bracket_draws[i];
            double e = expected_score(mid, (double)st->bracket_elo[i]);
            f += s - (double)n * e;
        }
        if (f > 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return (int)llround(0.5 * (lo + hi));
}

static void update_progress_fields(AiTestStatus *st) {
    int played = st->wins + st->draws + st->losses;
    if (played > 0) {
        double score = (double)st->wins + 0.5 * (double)st->draws;
        st->score_pct = score / (double)played;

        if (st->bracket_count > 1) {
            st->estimated_elo = estimate_elo_from_brackets(st);
            st->estimated_elo_is_delta = false;
        } else {
            st->estimated_elo_is_delta = true;
            st->estimated_elo = estimate_elo_delta_single_bracket_smoothed(st);
        }
    } else {
        st->score_pct = 0.0;
        st->estimated_elo = 0;
        st->estimated_elo_is_delta = true;
    }

    if (st->games_completed > 0 && st->total_games > st->games_completed) {
        double avg_game_ms = (double)st->elapsed_ms / (double)st->games_completed;
        int remain = st->total_games - st->games_completed;
        st->eta_ms = (int)(avg_game_ms * (double)remain);
    } else {
        st->eta_ms = 0;
    }
}

static void store_status(AiTestRunnerImpl *impl, const AiTestStatus *status) {
    pthread_mutex_lock(&impl->mutex);
    impl->status = *status;
    impl->dirty = true;
    pthread_mutex_unlock(&impl->mutex);
}

static bool should_stop_job(AiTestRunnerImpl *impl) {
    pthread_mutex_lock(&impl->mutex);
    bool stop = impl->stop_requested || !impl->running;
    pthread_mutex_unlock(&impl->mutex);
    return stop;
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
        trim_ascii(line);
        if (line[0] != '\0') {
            strncpy(out_path, line, 511);
            out_path[511] = '\0';
            ok = true;
        }
    }
    pclose(fp);
    return ok;
}

static bool stockfish_wait_ready(StockfishProc *proc, char *err, size_t err_sz) {
    if (proc == NULL || proc->out == NULL) {
        snprintf(err, err_sz, "Stockfish stream unavailable");
        return false;
    }

    char line[1024];
    bool saw_ready = false;
    while (fgets(line, sizeof(line), proc->out) != NULL) {
        if (strncmp(line, "readyok", 7) == 0) {
            saw_ready = true;
            break;
        }
    }
    if (!saw_ready) {
        snprintf(err, err_sz, "Stockfish init failed (readyok not received)");
        return false;
    }
    return true;
}

static bool stockfish_configure(StockfishProc *proc,
                                int skill_level,
                                bool limit_elo_mode,
                                int elo,
                                char *err,
                                size_t err_sz) {
    if (proc == NULL || proc->in == NULL) {
        snprintf(err, err_sz, "Stockfish is not ready");
        return false;
    }

    if (skill_level < 0) {
        skill_level = 0;
    }
    if (skill_level > 20) {
        skill_level = 20;
    }
    if (elo < 1320) {
        elo = 1320;
    }
    if (elo > 3190) {
        elo = 3190;
    }

    bool changed = !proc->configured;
    if (!changed) {
        if (proc->configured_skill_level != skill_level ||
            proc->configured_limit_elo_mode != limit_elo_mode ||
            (limit_elo_mode && proc->configured_elo != elo)) {
            changed = true;
        }
    }
    if (!changed) {
        return true;
    }

    if (!proc->configured) {
        fprintf(proc->in, "setoption name Threads value 1\n");
        fprintf(proc->in, "setoption name Hash value 32\n");
    }
    if (!proc->configured || proc->configured_skill_level != skill_level) {
        fprintf(proc->in, "setoption name Skill Level value %d\n", skill_level);
    }
    if (!proc->configured || proc->configured_limit_elo_mode != limit_elo_mode) {
        fprintf(proc->in, "setoption name UCI_LimitStrength value %s\n", limit_elo_mode ? "true" : "false");
    }
    if (limit_elo_mode && (!proc->configured || proc->configured_elo != elo)) {
        fprintf(proc->in, "setoption name UCI_Elo value %d\n", elo);
    }
    fprintf(proc->in, "isready\n");
    fflush(proc->in);
    if (!stockfish_wait_ready(proc, err, err_sz)) {
        return false;
    }

    proc->configured = true;
    proc->configured_skill_level = skill_level;
    proc->configured_limit_elo_mode = limit_elo_mode;
    proc->configured_elo = elo;
    return true;
}

static void stockfish_stop(StockfishProc *proc);

static bool stockfish_start(StockfishProc *proc, int skill_level, bool limit_elo_mode, int elo, char *err, size_t err_sz) {
    memset(proc, 0, sizeof(*proc));

    char path[512];
    if (!resolve_stockfish_binary(path)) {
        snprintf(err, err_sz, "Stockfish binary not found");
        return false;
    }

    int in_pipe[2] = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
        snprintf(err, err_sz, "Failed to create pipes");
        if (in_pipe[0] >= 0) {
            close(in_pipe[0]);
            close(in_pipe[1]);
        }
        if (out_pipe[0] >= 0) {
            close(out_pipe[0]);
            close(out_pipe[1]);
        }
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        snprintf(err, err_sz, "Failed to fork stockfish");
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
        execlp(path, path, (char *)NULL);
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);

    proc->pid = pid;
    proc->in = fdopen(in_pipe[1], "w");
    proc->out = fdopen(out_pipe[0], "r");
    if (proc->in == NULL || proc->out == NULL) {
        if (proc->in != NULL) {
            fclose(proc->in);
        } else {
            close(in_pipe[1]);
        }
        if (proc->out != NULL) {
            fclose(proc->out);
        } else {
            close(out_pipe[0]);
        }
        (void)waitpid(pid, NULL, 0);
        snprintf(err, err_sz, "Failed to open stockfish streams");
        return false;
    }

    fprintf(proc->in, "uci\n");
    fflush(proc->in);

    char line[1024];
    bool saw_uciok = false;
    while (fgets(line, sizeof(line), proc->out) != NULL) {
        if (strncmp(line, "uciok", 5) == 0) {
            saw_uciok = true;
            break;
        }
    }
    if (!saw_uciok) {
        snprintf(err, err_sz, "Stockfish init failed (uciok not received)");
        stockfish_stop(proc);
        return false;
    }

    if (!stockfish_configure(proc, skill_level, limit_elo_mode, elo, err, err_sz)) {
        stockfish_stop(proc);
        return false;
    }

    proc->ready = true;
    return true;
}

static void stockfish_stop(StockfishProc *proc) {
    if (proc == NULL) {
        return;
    }

    if (proc->in != NULL) {
        fprintf(proc->in, "quit\n");
        fflush(proc->in);
        fclose(proc->in);
        proc->in = NULL;
    }
    if (proc->out != NULL) {
        fclose(proc->out);
        proc->out = NULL;
    }
    if (proc->pid > 0) {
        (void)waitpid(proc->pid, NULL, 0);
        proc->pid = 0;
    }
    proc->ready = false;
}

static bool stockfish_pick_move(StockfishProc *proc,
                                const GameState *state,
                                int movetime_ms,
                                Move *out_move) {
    if (proc == NULL || !proc->ready || state == NULL || out_move == NULL) {
        return false;
    }

    if (movetime_ms < 20) {
        movetime_ms = 20;
    }
    if (movetime_ms > 10000) {
        movetime_ms = 10000;
    }

    char fen[256];
    chess_export_fen(state, fen, sizeof(fen));

    fprintf(proc->in, "position fen %s\n", fen);
    fprintf(proc->in, "go movetime %d\n", movetime_ms);
    fflush(proc->in);

    char line[1024];
    while (fgets(line, sizeof(line), proc->out) != NULL) {
        if (strncmp(line, "bestmove ", 9) != 0) {
            continue;
        }

        char uci[16] = {0};
        if (sscanf(line + 9, "%15s", uci) != 1) {
            return false;
        }
        if (strcmp(uci, "(none)") == 0) {
            return false;
        }

        Move parsed = 0;
        if (!chess_move_from_uci(state, uci, &parsed)) {
            return false;
        }
        *out_move = parsed;
        return true;
    }

    return false;
}

static int load_positions_from_file(const char *path, char (*out_fens)[AI_TEST_FEN_MAX], int cap) {
    if (path == NULL || path[0] == '\0' || out_fens == NULL || cap <= 0) {
        return 0;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };
    GameState tmp;
    chess_init(&tmp, &cfg);

    int count = 0;
    char line[1024];
    while (count < cap && fgets(line, sizeof(line), fp) != NULL) {
        trim_ascii(line);
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }

        char fen_line[AI_TEST_FEN_MAX];
        normalize_fen_for_loader(line, fen_line);
        if (fen_line[0] == '\0') {
            continue;
        }

        char err[128];
        if (!chess_load_fen(&tmp, fen_line, err, sizeof(err))) {
            continue;
        }
        strncpy(out_fens[count], fen_line, AI_TEST_FEN_MAX - 1);
        out_fens[count][AI_TEST_FEN_MAX - 1] = '\0';
        count += 1;
    }

    fclose(fp);
    return count;
}

static int load_positions(const AiTestConfig *cfg, char (*out_fens)[AI_TEST_FEN_MAX], int cap) {
    int count = 0;
    if (cfg->use_lichess_positions) {
        count = load_positions_from_file(cfg->positions_path, out_fens, cap);
    }
    if (count > 0) {
        return count;
    }

    int fallback = (int)(sizeof(k_default_equal_fens) / sizeof(k_default_equal_fens[0]));
    if (fallback > cap) {
        fallback = cap;
    }
    for (int i = 0; i < fallback; ++i) {
        strncpy(out_fens[i], k_default_equal_fens[i], AI_TEST_FEN_MAX - 1);
        out_fens[i][AI_TEST_FEN_MAX - 1] = '\0';
    }
    return fallback;
}

static int score_game_for_side(const GameState *state, int side) {
    switch (state->result) {
        case GAME_RESULT_WHITE_WIN:
        case GAME_RESULT_WHITE_WIN_RESIGN:
        case GAME_RESULT_WIN_TIMEOUT:
            if (side == PIECE_WHITE) {
                return 2;
            }
            return 0;
        case GAME_RESULT_BLACK_WIN:
        case GAME_RESULT_BLACK_WIN_RESIGN:
            if (side == PIECE_BLACK) {
                return 2;
            }
            return 0;
        case GAME_RESULT_DRAW_STALEMATE:
        case GAME_RESULT_DRAW_REPETITION:
        case GAME_RESULT_DRAW_50:
        case GAME_RESULT_DRAW_75:
        case GAME_RESULT_DRAW_INSUFFICIENT:
        case GAME_RESULT_DRAW_AGREED:
            return 1;
        case GAME_RESULT_ONGOING:
        default:
            return 1;
    }
}

static int find_or_add_bracket(AiTestStatus *st, int elo) {
    for (int i = 0; i < st->bracket_count; ++i) {
        if (st->bracket_elo[i] == elo) {
            return i;
        }
    }
    if (st->bracket_count >= AI_TEST_MAX_BRACKETS) {
        return st->bracket_count - 1;
    }
    int idx = st->bracket_count++;
    st->bracket_elo[idx] = elo;
    st->bracket_games[idx] = 0;
    st->bracket_wins[idx] = 0;
    st->bracket_draws[idx] = 0;
    st->bracket_losses[idx] = 0;
    return idx;
}

static int default_elo_for_skill(int skill_level) {
    if (skill_level < 0) {
        skill_level = 0;
    }
    if (skill_level > 20) {
        skill_level = 20;
    }
    int elo = 1320 + (int)llround((double)skill_level * ((3190.0 - 1320.0) / 20.0));
    if (elo < 1320) {
        elo = 1320;
    }
    if (elo > 3190) {
        elo = 3190;
    }
    return elo;
}

static int pick_ladder_elo(int game_idx, uint64_t sequence) {
    int n = (int)(sizeof(k_stockfish_elo_ladder) / sizeof(k_stockfish_elo_ladder[0]));
    if (n <= 0) {
        return 2000;
    }
    int offset = (int)(sequence % (uint64_t)n);
    if (offset < 0) {
        offset = 0;
    }
    int idx = (game_idx + offset) % n;
    return k_stockfish_elo_ladder[idx];
}

static bool pick_engine_move_with_backend(GameState *game,
                                          ChessAiBackend backend,
                                          int think_ms,
                                          Move *out_move,
                                          AiSearchResult *out_result);

static bool pick_engine_move_with_backend(GameState *game,
                                          ChessAiBackend backend,
                                          int think_ms,
                                          Move *out_move,
                                          AiSearchResult *out_result) {
    if (game == NULL || out_move == NULL) {
        return false;
    }
    if (!chess_ai_set_backend(backend)) {
        return false;
    }
    AiSearchConfig ai_cfg = {
        .think_time_ms = think_ms,
        .max_depth = 16,
    };
    AiSearchResult ai_result;
    if (!chess_ai_pick_move(game, &ai_cfg, &ai_result) || !ai_result.found_move) {
        return false;
    }
    *out_move = ai_result.best_move;
    if (out_result != NULL) {
        *out_result = ai_result;
    }
    return true;
}

static void accumulate_search_metrics(AiTestStatus *st, bool our_side_move, const AiSearchResult *res) {
    if (st == NULL || res == NULL || !res->found_move) {
        return;
    }

    int elapsed_ms = res->elapsed_ms;
    if (elapsed_ms < 0) {
        elapsed_ms = 0;
    }

    if (our_side_move) {
        st->our_search_moves += 1;
        st->our_search_depth_sum += res->depth_reached;
        st->our_search_nodes_sum += res->nodes;
        st->our_search_time_ms_sum += (int64_t)elapsed_ms;
    } else {
        st->opp_search_moves += 1;
        st->opp_search_depth_sum += res->depth_reached;
        st->opp_search_nodes_sum += res->nodes;
        st->opp_search_time_ms_sum += (int64_t)elapsed_ms;
    }
}

static void run_test_job(AiTestRunnerImpl *impl, const AiTestConfig *cfg) {
    AiTestStatus st;
    memset(&st, 0, sizeof(st));
    st.running = true;
    st.mode = cfg->mode;
    st.our_backend = cfg->our_backend;
    st.opponent_backend = cfg->opponent_backend;
    st.total_games = cfg->total_games;
    st.seed_used = cfg->random_seed;
    st.estimated_elo_is_delta = !cfg->stockfish_ladder_enabled;
    snprintf(st.message, sizeof(st.message), "Preparing test");
    store_status(impl, &st);

    ChessAiBackend saved_backend = chess_ai_get_backend();
    if ((cfg->our_backend == CHESS_AI_BACKEND_NN || cfg->opponent_backend == CHESS_AI_BACKEND_NN) &&
        cfg->nn_model_path[0] != '\0' &&
        !chess_ai_nn_model_is_loaded()) {
        (void)chess_ai_set_nn_model_path(cfg->nn_model_path);
    }
    if ((cfg->our_backend == CHESS_AI_BACKEND_NN || cfg->opponent_backend == CHESS_AI_BACKEND_NN) &&
        !chess_ai_nn_model_is_loaded()) {
        st.running = false;
        st.failed = true;
        snprintf(st.message, sizeof(st.message), "NN model unavailable for AI test");
        store_status(impl, &st);
        return;
    }

    char (*fens)[AI_TEST_FEN_MAX] = (char(*)[AI_TEST_FEN_MAX])calloc(AI_TEST_MAX_POSITIONS, AI_TEST_FEN_MAX);
    if (fens == NULL) {
        st.running = false;
        st.failed = true;
        snprintf(st.message, sizeof(st.message), "Out of memory for FEN list");
        store_status(impl, &st);
        (void)chess_ai_set_backend(saved_backend);
        return;
    }

    int fen_count = load_positions(cfg, fens, AI_TEST_MAX_POSITIONS);
    st.positions_loaded = fen_count;
    if (fen_count <= 0) {
        st.running = false;
        st.failed = true;
        snprintf(st.message, sizeof(st.message), "No valid test positions found");
        store_status(impl, &st);
        free(fens);
        (void)chess_ai_set_backend(saved_backend);
        return;
    }

    int base_elo = 0;
    if (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) {
        base_elo = default_elo_for_skill(cfg->stockfish_skill_level);
        if (cfg->stockfish_ladder_enabled) {
            int ladder_n = (int)(sizeof(k_stockfish_elo_ladder) / sizeof(k_stockfish_elo_ladder[0]));
            for (int i = 0; i < ladder_n && i < AI_TEST_MAX_BRACKETS; ++i) {
                (void)find_or_add_bracket(&st, k_stockfish_elo_ladder[i]);
            }
        } else {
            (void)find_or_add_bracket(&st, base_elo);
        }
    }

    StockfishProc sf;
    memset(&sf, 0, sizeof(sf));
    bool stockfish_ready = false;
    char sf_err[160];
    if (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) {
        int initial_elo = cfg->stockfish_ladder_enabled ? k_stockfish_elo_ladder[0] : base_elo;
        if (!stockfish_start(&sf, cfg->stockfish_skill_level, cfg->stockfish_ladder_enabled, initial_elo, sf_err, sizeof(sf_err))) {
            st.running = false;
            st.failed = true;
            snprintf(st.message, sizeof(st.message), "Stockfish unavailable: %s", sf_err);
            store_status(impl, &st);
            free(fens);
            (void)chess_ai_set_backend(saved_backend);
            return;
        }
        stockfish_ready = true;
    }

    int64_t started_ms = now_ms();
    int64_t last_status_update_ms = started_ms;
    uint64_t sequence = cfg->random_seed;
    if (sequence == 0ULL) {
        sequence = (uint64_t)started_ms ^ 0x9E3779B97F4A7C15ULL;
    }
    if (sequence == 0ULL) {
        sequence = 0xA5A5A5A5A5A5A5A5ULL;
    }
    st.seed_used = sequence;

    MatchConfig match_cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    int max_plies = cfg->max_plies_per_game;
    if (max_plies <= 0) {
        max_plies = AI_TEST_MAX_PLY_FALLBACK;
    }

    for (int game_idx = 0; game_idx < cfg->total_games; ++game_idx) {
        if (should_stop_job(impl)) {
            st.running = false;
            st.canceled = true;
            snprintf(st.message, sizeof(st.message), "Test canceled");
            break;
        }

        sequence ^= sequence << 7;
        sequence ^= sequence >> 9;
        sequence ^= sequence << 8;

        int opp_elo = 0;
        if (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) {
            opp_elo = cfg->stockfish_ladder_enabled ? pick_ladder_elo(game_idx, sequence) : base_elo;
            if (!stockfish_configure(&sf, cfg->stockfish_skill_level, cfg->stockfish_ladder_enabled, opp_elo, sf_err, sizeof(sf_err))) {
                st.running = false;
                st.failed = true;
                snprintf(st.message, sizeof(st.message), "Stockfish config failed: %s", sf_err);
                break;
            }
        }

        int fen_idx = (int)(sequence % (uint64_t)fen_count);
        if (fen_idx < 0) {
            fen_idx = 0;
        }

        GameState game;
        chess_init(&game, &match_cfg);
        char err[128];
        if (!chess_load_fen(&game, fens[fen_idx], err, sizeof(err))) {
            chess_set_result(&game, GAME_RESULT_DRAW_AGREED);
        }
        game.config = match_cfg;

        int our_side = (game_idx & 1) ? PIECE_BLACK : PIECE_WHITE;
        st.current_game = game_idx + 1;
        st.current_ply = 0;
        st.current_our_side = our_side;
        st.current_opponent_elo = opp_elo;
        st.current_opponent_skill = (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) ? cfg->stockfish_skill_level : 0;

        for (int ply = 0; ply < max_plies && game.result == GAME_RESULT_ONGOING; ++ply) {
            if (should_stop_job(impl)) {
                st.running = false;
                st.canceled = true;
                snprintf(st.message, sizeof(st.message), "Test canceled");
                break;
            }

            st.current_ply = ply + 1;
            bool should_publish_status = ((ply & AI_TEST_STATUS_UPDATE_PLY_MASK) == 0);
            int64_t status_now_ms = 0;
            if (!should_publish_status) {
                status_now_ms = now_ms();
                if ((status_now_ms - last_status_update_ms) >= AI_TEST_STATUS_UPDATE_MIN_MS) {
                    should_publish_status = true;
                }
            }
            if (should_publish_status) {
                if (status_now_ms == 0) {
                    status_now_ms = now_ms();
                }
                st.elapsed_ms = (int)(status_now_ms - started_ms);
                update_progress_fields(&st);
                snprintf(st.message, sizeof(st.message), "Game %d/%d on ply %d (%s white)",
                         st.current_game,
                         st.total_games,
                         st.current_ply,
                         (our_side == PIECE_WHITE) ? chess_ai_backend_name(cfg->our_backend)
                                                   : (cfg->mode == AI_TEST_MODE_VS_STOCKFISH ? "stockfish"
                                                                                             : chess_ai_backend_name(cfg->opponent_backend)));
                store_status(impl, &st);
                last_status_update_ms = status_now_ms;
            }

            Move m = 0;
            AiSearchResult ai_move_result;
            memset(&ai_move_result, 0, sizeof(ai_move_result));
            if (game.side_to_move == our_side) {
                if (!pick_engine_move_with_backend(&game, cfg->our_backend, cfg->our_think_ms, &m, &ai_move_result)) {
                    (void)chess_update_result(&game);
                    break;
                }
                accumulate_search_metrics(&st, true, &ai_move_result);
            } else {
                if (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) {
                    if (!stockfish_pick_move(&sf, &game, cfg->stockfish_think_ms, &m)) {
                        (void)chess_update_result(&game);
                        break;
                    }
                } else {
                    if (!pick_engine_move_with_backend(&game, cfg->opponent_backend, cfg->stockfish_think_ms, &m, &ai_move_result)) {
                        (void)chess_update_result(&game);
                        break;
                    }
                    accumulate_search_metrics(&st, false, &ai_move_result);
                }
            }

            if (!chess_make_move(&game, m)) {
                (void)chess_update_result(&game);
                break;
            }
        }

        if (!st.canceled && game.result == GAME_RESULT_ONGOING) {
            chess_set_result(&game, GAME_RESULT_DRAW_AGREED);
        }

        if (st.canceled) {
            break;
        }

        int outcome = score_game_for_side(&game, our_side);
        int bracket_idx = -1;
        if (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) {
            bracket_idx = find_or_add_bracket(&st, opp_elo);
            st.bracket_games[bracket_idx] += 1;
        }
        if (outcome == 2) {
            st.wins += 1;
            if (bracket_idx >= 0) {
                st.bracket_wins[bracket_idx] += 1;
            }
        } else if (outcome == 1) {
            st.draws += 1;
            if (bracket_idx >= 0) {
                st.bracket_draws[bracket_idx] += 1;
            }
        } else {
            st.losses += 1;
            if (bracket_idx >= 0) {
                st.bracket_losses[bracket_idx] += 1;
            }
        }

        st.games_completed = game_idx + 1;
        st.elapsed_ms = (int)(now_ms() - started_ms);
        update_progress_fields(&st);
        snprintf(st.message, sizeof(st.message), "Completed %d/%d games",
                 st.games_completed,
                 st.total_games);
        store_status(impl, &st);
    }

    if (stockfish_ready) {
        stockfish_stop(&sf);
    }
    free(fens);
    (void)chess_ai_set_backend(saved_backend);

    if (!st.canceled && !st.failed) {
        st.running = false;
        st.completed = true;
        st.elapsed_ms = (int)(now_ms() - started_ms);
        update_progress_fields(&st);
        snprintf(st.message, sizeof(st.message), "Test complete");
    }

    store_status(impl, &st);
}

static void *runner_thread_main(void *arg) {
    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)arg;

    for (;;) {
        pthread_mutex_lock(&impl->mutex);
        while (impl->running && !impl->has_job) {
            pthread_cond_wait(&impl->cond, &impl->mutex);
        }

        if (!impl->running) {
            pthread_mutex_unlock(&impl->mutex);
            break;
        }

        AiTestConfig cfg = impl->job_cfg;
        impl->has_job = false;
        impl->stop_requested = false;
        pthread_mutex_unlock(&impl->mutex);

        run_test_job(impl, &cfg);

        pthread_mutex_lock(&impl->mutex);
        impl->stop_requested = false;
        pthread_mutex_unlock(&impl->mutex);
    }

    return NULL;
}

bool ai_test_runner_init(AiTestRunner *runner) {
    if (runner == NULL) {
        return false;
    }

    memset(runner, 0, sizeof(*runner));
    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)calloc(1, sizeof(*impl));
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
        size_t stack_size = 16u * 1024u * 1024u;
        (void)pthread_attr_setstacksize(&attr, stack_size);
    }

    int create_rc = pthread_create(&impl->thread, attr_ok ? &attr : NULL, runner_thread_main, impl);
    if (attr_ok) {
        pthread_attr_destroy(&attr);
    }

    if (create_rc != 0) {
        pthread_cond_destroy(&impl->cond);
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return false;
    }

    runner->impl = impl;
    runner->initialized = true;
    return true;
}

void ai_test_runner_shutdown(AiTestRunner *runner) {
    if (runner == NULL || !runner->initialized || runner->impl == NULL) {
        return;
    }

    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)runner->impl;
    pthread_mutex_lock(&impl->mutex);
    impl->running = false;
    impl->stop_requested = true;
    impl->has_job = false;
    pthread_cond_signal(&impl->cond);
    pthread_mutex_unlock(&impl->mutex);

    pthread_join(impl->thread, NULL);
    pthread_cond_destroy(&impl->cond);
    pthread_mutex_destroy(&impl->mutex);
    free(impl);

    runner->impl = NULL;
    runner->initialized = false;
}

bool ai_test_runner_start(AiTestRunner *runner, const AiTestConfig *cfg) {
    if (runner == NULL || !runner->initialized || runner->impl == NULL || cfg == NULL) {
        return false;
    }

    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)runner->impl;

    pthread_mutex_lock(&impl->mutex);
    if (!impl->running || impl->has_job || impl->status.running) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    impl->job_cfg = *cfg;
    impl->has_job = true;
    impl->stop_requested = false;

    memset(&impl->status, 0, sizeof(impl->status));
    impl->status.mode = cfg->mode;
    impl->status.our_backend = cfg->our_backend;
    impl->status.opponent_backend = cfg->opponent_backend;
    impl->status.seed_used = cfg->random_seed;
    impl->status.total_games = cfg->total_games;
    impl->status.estimated_elo_is_delta = !cfg->stockfish_ladder_enabled;
    impl->status.current_opponent_skill = cfg->stockfish_skill_level;
    snprintf(impl->status.message, sizeof(impl->status.message), "Queued");
    impl->dirty = true;

    pthread_cond_signal(&impl->cond);
    pthread_mutex_unlock(&impl->mutex);
    return true;
}

void ai_test_runner_stop(AiTestRunner *runner) {
    if (runner == NULL || !runner->initialized || runner->impl == NULL) {
        return;
    }

    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)runner->impl;
    pthread_mutex_lock(&impl->mutex);
    impl->stop_requested = true;
    pthread_mutex_unlock(&impl->mutex);
}

bool ai_test_runner_poll(AiTestRunner *runner, AiTestStatus *out_status) {
    if (runner == NULL || !runner->initialized || runner->impl == NULL) {
        return false;
    }

    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)runner->impl;
    pthread_mutex_lock(&impl->mutex);
    if (!impl->dirty) {
        pthread_mutex_unlock(&impl->mutex);
        return false;
    }

    if (out_status != NULL) {
        *out_status = impl->status;
    }
    impl->dirty = false;
    pthread_mutex_unlock(&impl->mutex);
    return true;
}

bool ai_test_runner_get_status(AiTestRunner *runner, AiTestStatus *out_status) {
    if (runner == NULL || !runner->initialized || runner->impl == NULL || out_status == NULL) {
        return false;
    }

    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)runner->impl;
    pthread_mutex_lock(&impl->mutex);
    *out_status = impl->status;
    pthread_mutex_unlock(&impl->mutex);
    return true;
}

bool ai_test_runner_is_running(AiTestRunner *runner) {
    if (runner == NULL || !runner->initialized || runner->impl == NULL) {
        return false;
    }

    AiTestRunnerImpl *impl = (AiTestRunnerImpl *)runner->impl;
    pthread_mutex_lock(&impl->mutex);
    bool active = impl->status.running || impl->has_job;
    pthread_mutex_unlock(&impl->mutex);
    return active;
}
