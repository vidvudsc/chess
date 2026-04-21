#include "ai_test_lab_cli.h"

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

#include "ai_test_runner.h"

#define ARRAY_LEN(x) ((int)(sizeof(x) / sizeof((x)[0])))

static volatile sig_atomic_t g_ai_test_cli_stop_requested = 0;

typedef struct CliMatchupSelection {
    const char *name;
    AiTestMode mode;
    ChessAiBackend our_backend;
    ChessAiBackend opponent_backend;
} CliMatchupSelection;

static const CliMatchupSelection k_cli_matchups[] = {
    {"classic-vs-stockfish", AI_TEST_MODE_VS_STOCKFISH, CHESS_AI_BACKEND_CLASSIC, CHESS_AI_BACKEND_CLASSIC},
    {"nn-vs-stockfish", AI_TEST_MODE_VS_STOCKFISH, CHESS_AI_BACKEND_NN, CHESS_AI_BACKEND_CLASSIC},
    {"classic-vs-nn", AI_TEST_MODE_INTERNAL, CHESS_AI_BACKEND_CLASSIC, CHESS_AI_BACKEND_NN},
    {"nn-vs-classic", AI_TEST_MODE_INTERNAL, CHESS_AI_BACKEND_NN, CHESS_AI_BACKEND_CLASSIC},
};

static void ai_test_cli_on_signal(int signum) {
    (void)signum;
    g_ai_test_cli_stop_requested = 1;
}

static bool cli_file_exists(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return false;
    }
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool resolve_default_nn_model_path(char out_path[1024]) {
    const char *env_path = getenv("CHESS_NN_MODEL");
    if (env_path != NULL && env_path[0] != '\0' && cli_file_exists(env_path)) {
        snprintf(out_path, 1024, "%s", env_path);
        return true;
    }

    static const char *k_candidates[] = {
        "src/core/bot/nn/model/nn_eval.bin",
        "./src/core/bot/nn/model/nn_eval.bin",
        "current/nn_eval.bin",
        "./current/nn_eval.bin",
        "nn_eval.bin",
    };
    for (int i = 0; i < ARRAY_LEN(k_candidates); ++i) {
        if (cli_file_exists(k_candidates[i])) {
            snprintf(out_path, 1024, "%s", k_candidates[i]);
            return true;
        }
    }

    out_path[0] = '\0';
    return false;
}

static void ai_test_cli_print_usage(FILE *stream, const char *argv0) {
    const char *prog = (argv0 != NULL && argv0[0] != '\0') ? argv0 : "ai_test_lab";
    fprintf(stream,
            "Usage:\n"
            "  %s [options]\n"
            "  bin/chess --ai-test-lab [options]\n"
            "\n"
            "Options:\n"
            "  --matchup <name>          classic-vs-stockfish|nn-vs-stockfish|classic-vs-nn|nn-vs-classic\n"
            "  --games <n>               Total games to run (default: 200)\n"
            "  --our-think-ms <n>        Our side think time in ms (default: 350)\n"
            "  --opp-think-ms <n>        Opponent think time in ms (default: 350)\n"
            "  --stockfish-skill <n>     Stockfish skill 0-20 (default: 10)\n"
            "  --stockfish-mode <mode>   ladder|fixed (default: ladder)\n"
            "  --positions <mode>        lichess|built-in (default: lichess)\n"
            "  --positions-path <path>   Override the lichess/file position path\n"
            "  --nn-model <path>         Override NN model path when an NN matchup is used\n"
            "  --seed <n>                Fixed seed for reproducible position/order selection\n"
            "  --max-plies <n>           Max plies per game before adjudicating draw (default: 180)\n"
            "  --help                    Show this help\n",
            prog);
}

static bool parse_positive_int(const char *text, int *out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return false;
    }
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 1000000000L) {
        return false;
    }
    *out = (int)value;
    return true;
}

static bool parse_u64(const char *text, uint64_t *out) {
    if (text == NULL || out == NULL || text[0] == '\0') {
        return false;
    }
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    *out = (uint64_t)value;
    return true;
}

static const char *backend_label(ChessAiBackend backend) {
    return (backend == CHESS_AI_BACKEND_NN) ? "NN" : "Classic";
}

static const char *cli_matchup_name(const AiTestConfig *cfg) {
    if (cfg == NULL) {
        return "unknown";
    }
    for (int i = 0; i < ARRAY_LEN(k_cli_matchups); ++i) {
        if (k_cli_matchups[i].mode == cfg->mode &&
            k_cli_matchups[i].our_backend == cfg->our_backend &&
            k_cli_matchups[i].opponent_backend == cfg->opponent_backend) {
            return k_cli_matchups[i].name;
        }
    }
    return "custom";
}

static bool parse_matchup_name(const char *text, AiTestConfig *cfg) {
    if (text == NULL || cfg == NULL) {
        return false;
    }
    for (int i = 0; i < ARRAY_LEN(k_cli_matchups); ++i) {
        if (strcmp(text, k_cli_matchups[i].name) == 0) {
            cfg->mode = k_cli_matchups[i].mode;
            cfg->our_backend = k_cli_matchups[i].our_backend;
            cfg->opponent_backend = k_cli_matchups[i].opponent_backend;
            return true;
        }
    }
    return false;
}

static void ai_test_cli_print_config(const AiTestConfig *cfg) {
    const char *positions_mode = cfg->use_lichess_positions ? "lichess" : "built-in";
    const char *stockfish_mode = cfg->stockfish_ladder_enabled ? "ladder" : "fixed";
    fprintf(stdout,
            "Starting AI Test Lab\n"
            "matchup=%s\n"
            "games=%d\n"
            "our_backend=%s\n"
            "opponent=%s\n"
            "our_think_ms=%d\n"
            "opp_think_ms=%d\n"
            "stockfish_skill=%d\n"
            "stockfish_mode=%s\n"
            "positions=%s\n"
            "positions_path=%s\n"
            "seed=%llu\n"
            "max_plies=%d\n",
            cli_matchup_name(cfg),
            cfg->total_games,
            backend_label(cfg->our_backend),
            (cfg->mode == AI_TEST_MODE_VS_STOCKFISH) ? "Stockfish" : backend_label(cfg->opponent_backend),
            cfg->our_think_ms,
            cfg->stockfish_think_ms,
            cfg->stockfish_skill_level,
            stockfish_mode,
            positions_mode,
            cfg->positions_path[0] != '\0' ? cfg->positions_path : "(default)",
            (unsigned long long)cfg->random_seed,
            cfg->max_plies_per_game);
    if (cfg->nn_model_path[0] != '\0') {
        fprintf(stdout, "nn_model=%s\n", cfg->nn_model_path);
    }
    fflush(stdout);
}

typedef struct AiTestStatsSummary {
    bool valid;
    int games;
    double mean_score;
    double score_ci_low;
    double score_ci_high;
    double elo;
    double elo_ci_low;
    double elo_ci_high;
    double z_score;
    double probability_better;
} AiTestStatsSummary;

static double score_to_elo_clamped(double score) {
    if (score <= 0.0) {
        score = 1e-6;
    } else if (score >= 1.0) {
        score = 1.0 - 1e-6;
    }
    return -400.0 * log10((1.0 / score) - 1.0);
}

static double normal_cdf(double z) {
    return 0.5 * (1.0 + erf(z / sqrt(2.0)));
}

static AiTestStatsSummary ai_test_cli_compute_stats(const AiTestStatus *st) {
    AiTestStatsSummary stats;
    memset(&stats, 0, sizeof(stats));
    if (st == NULL) {
        return stats;
    }

    int games = st->wins + st->draws + st->losses;
    if (games <= 0) {
        return stats;
    }

    double wins = (double)st->wins;
    double draws = (double)st->draws;
    double mean = (wins + 0.5 * draws) / (double)games;
    double sum_sq = wins + 0.25 * draws;

    stats.valid = true;
    stats.games = games;
    stats.mean_score = mean;
    stats.elo = score_to_elo_clamped(mean);
    stats.score_ci_low = mean;
    stats.score_ci_high = mean;
    stats.elo_ci_low = stats.elo;
    stats.elo_ci_high = stats.elo;
    stats.probability_better = (mean > 0.5) ? 1.0 : ((mean < 0.5) ? 0.0 : 0.5);

    if (games >= 2) {
        double sample_var = (sum_sq - (double)games * mean * mean) / (double)(games - 1);
        if (sample_var < 0.0) {
            sample_var = 0.0;
        }
        double se = sqrt(sample_var / (double)games);
        if (se > 0.0) {
            double low = mean - 1.96 * se;
            double high = mean + 1.96 * se;
            if (low < 1e-6) {
                low = 1e-6;
            }
            if (high > 1.0 - 1e-6) {
                high = 1.0 - 1e-6;
            }
            stats.score_ci_low = low;
            stats.score_ci_high = high;
            stats.elo_ci_low = score_to_elo_clamped(low);
            stats.elo_ci_high = score_to_elo_clamped(high);
            stats.z_score = (mean - 0.5) / se;
            stats.probability_better = normal_cdf(stats.z_score);
        }
    }

    return stats;
}

static void ai_test_cli_print_status(const AiTestStatus *st) {
    if (st == NULL) {
        return;
    }
    fprintf(stdout,
            "status: %s | games %d/%d | W-D-L %d-%d-%d | score %.1f%% | elo %s%d | pos %d | elapsed %.1fs | eta %.1fs\n",
            st->message[0] != '\0' ? st->message : "Idle",
            st->games_completed,
            st->total_games,
            st->wins,
            st->draws,
            st->losses,
            st->score_pct * 100.0,
            st->estimated_elo_is_delta ? "delta " : "",
            st->estimated_elo,
            st->positions_loaded,
            (double)st->elapsed_ms / 1000.0,
            (double)st->eta_ms / 1000.0);
    fflush(stdout);
}

static void ai_test_cli_print_summary(const AiTestStatus *st) {
    AiTestStatsSummary stats = ai_test_cli_compute_stats(st);
    if (st == NULL) {
        return;
    }
    fprintf(stdout,
            "Final result\n"
            "message=%s\n"
            "seed_used=%llu\n"
            "games=%d/%d\n"
            "wins=%d\n"
            "draws=%d\n"
            "losses=%d\n"
            "score_pct=%.2f\n"
            "estimated_elo=%d\n"
            "estimated_elo_kind=%s\n"
            "elapsed_sec=%.2f\n",
            st->message[0] != '\0' ? st->message : "Idle",
            (unsigned long long)st->seed_used,
            st->games_completed,
            st->total_games,
            st->wins,
            st->draws,
            st->losses,
            st->score_pct * 100.0,
            st->estimated_elo,
            st->estimated_elo_is_delta ? "delta" : "absolute",
            (double)st->elapsed_ms / 1000.0);
    if (stats.valid) {
        fprintf(stdout,
                "score_ci_95_pct=%.2f..%.2f\n"
                "elo_estimate_logistic=%.1f\n"
                "elo_ci_95=%.1f..%.1f\n"
                "probability_better_than_50pct=%.4f\n",
                stats.score_ci_low * 100.0,
                stats.score_ci_high * 100.0,
                stats.elo,
                stats.elo_ci_low,
                stats.elo_ci_high,
                stats.probability_better);
    }
    fflush(stdout);
}

static int run_ai_test_job(const AiTestConfig *cfg) {
    AiTestRunner runner;
    memset(&runner, 0, sizeof(runner));
    if (!ai_test_runner_init(&runner)) {
        fprintf(stderr, "Failed to initialize AI test runner\n");
        return 1;
    }

    if (!ai_test_runner_start(&runner, cfg)) {
        AiTestStatus status;
        if (ai_test_runner_get_status(&runner, &status) && status.message[0] != '\0') {
            fprintf(stderr, "Failed to start AI test lab: %s\n", status.message);
        } else {
            fprintf(stderr, "Failed to start AI test lab\n");
        }
        ai_test_runner_shutdown(&runner);
        return 1;
    }

    bool stop_sent = false;
    AiTestStatus final_status;
    memset(&final_status, 0, sizeof(final_status));
    while (ai_test_runner_is_running(&runner)) {
        if (g_ai_test_cli_stop_requested && !stop_sent) {
            fprintf(stdout, "Stop requested, canceling AI test lab...\n");
            fflush(stdout);
            ai_test_runner_stop(&runner);
            stop_sent = true;
        }

        AiTestStatus latest;
        if (ai_test_runner_poll(&runner, &latest)) {
            final_status = latest;
            ai_test_cli_print_status(&latest);
        }
        usleep(50000);
    }

    if (!ai_test_runner_get_status(&runner, &final_status)) {
        fprintf(stderr, "Failed to retrieve final AI test status\n");
        ai_test_runner_shutdown(&runner);
        return 1;
    }
    ai_test_runner_shutdown(&runner);

    ai_test_cli_print_summary(&final_status);
    if (final_status.failed) {
        return 1;
    }
    if (final_status.canceled) {
        return 130;
    }
    return 0;
}

int chess_run_ai_test_cli(int argc, char **argv) {
    AiTestConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = AI_TEST_MODE_VS_STOCKFISH;
    cfg.our_backend = CHESS_AI_BACKEND_CLASSIC;
    cfg.opponent_backend = CHESS_AI_BACKEND_CLASSIC;
    cfg.total_games = 200;
    cfg.our_think_ms = 350;
    cfg.stockfish_think_ms = 350;
    cfg.stockfish_skill_level = 10;
    cfg.stockfish_ladder_enabled = true;
    cfg.max_plies_per_game = 180;
    cfg.use_lichess_positions = true;
    cfg.random_seed = 0;
    snprintf(cfg.positions_path, sizeof(cfg.positions_path), "%s", "data/positions/lichess_equal_positions.fen");
    (void)resolve_default_nn_model_path(cfg.nn_model_path);

    int argi = 1;
    if (argc > 1 && argv != NULL && argv[1] != NULL &&
        (strcmp(argv[1], "--ai-test-lab") == 0 || strcmp(argv[1], "ai-test-lab") == 0)) {
        argi = 2;
    }

    for (int i = argi; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            ai_test_cli_print_usage(stdout, argv[0]);
            return 0;
        }
        if (strcmp(arg, "--matchup") == 0) {
            if (i + 1 >= argc || !parse_matchup_name(argv[++i], &cfg)) {
                fprintf(stderr, "Invalid --matchup value\n");
                ai_test_cli_print_usage(stderr, argv[0]);
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--games") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg.total_games)) {
                fprintf(stderr, "Invalid --games value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--our-think-ms") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg.our_think_ms)) {
                fprintf(stderr, "Invalid --our-think-ms value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--opp-think-ms") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg.stockfish_think_ms)) {
                fprintf(stderr, "Invalid --opp-think-ms value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--stockfish-skill") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg.stockfish_skill_level) || cfg.stockfish_skill_level > 20) {
                fprintf(stderr, "Invalid --stockfish-skill value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--stockfish-mode") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing --stockfish-mode value\n");
                return 2;
            }
            const char *mode = argv[++i];
            if (strcmp(mode, "ladder") == 0) {
                cfg.stockfish_ladder_enabled = true;
            } else if (strcmp(mode, "fixed") == 0) {
                cfg.stockfish_ladder_enabled = false;
            } else {
                fprintf(stderr, "Invalid --stockfish-mode value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--positions") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing --positions value\n");
                return 2;
            }
            const char *mode = argv[++i];
            if (strcmp(mode, "lichess") == 0) {
                cfg.use_lichess_positions = true;
            } else if (strcmp(mode, "built-in") == 0) {
                cfg.use_lichess_positions = false;
            } else {
                fprintf(stderr, "Invalid --positions value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--positions-path") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing --positions-path value\n");
                return 2;
            }
            cfg.use_lichess_positions = true;
            snprintf(cfg.positions_path, sizeof(cfg.positions_path), "%s", argv[++i]);
            continue;
        }
        if (strcmp(arg, "--nn-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing --nn-model value\n");
                return 2;
            }
            snprintf(cfg.nn_model_path, sizeof(cfg.nn_model_path), "%s", argv[++i]);
            continue;
        }
        if (strcmp(arg, "--seed") == 0) {
            if (i + 1 >= argc || !parse_u64(argv[++i], &cfg.random_seed)) {
                fprintf(stderr, "Invalid --seed value\n");
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--max-plies") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg.max_plies_per_game)) {
                fprintf(stderr, "Invalid --max-plies value\n");
                return 2;
            }
            continue;
        }

        fprintf(stderr, "Unknown option: %s\n", arg);
        ai_test_cli_print_usage(stderr, argv[0]);
        return 2;
    }

    if (!cfg.use_lichess_positions) {
        cfg.positions_path[0] = '\0';
    }

    struct sigaction sa;
    struct sigaction old_int;
    struct sigaction old_term;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = ai_test_cli_on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, &old_int);
    sigaction(SIGTERM, &sa, &old_term);

    ai_test_cli_print_config(&cfg);
    int exit_code = run_ai_test_job(&cfg);

    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGTERM, &old_term, NULL);
    return exit_code;
}
