#include "ui.h"

#include <dirent.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <raylib.h>

#include "assets.h"
#include "ai_test_runner.h"
#include "chess_ai.h"
#include "chess_ai_worker.h"
#include "chess_io.h"
#include "chess_rules.h"
#include "game_log.h"
#include "stockfish_eval_worker.h"

#define ARRAY_LEN(x) ((int)(sizeof(x) / sizeof((x)[0])))

typedef struct TimePreset {
    const char *label;
    int initial_ms;
    int increment_ms;
} TimePreset;

typedef struct AiThinkPreset {
    const char *label;
    int ms;
} AiThinkPreset;

typedef enum AiGameMode {
    AI_MODE_HUMAN_VS_HUMAN = 0,
    AI_MODE_WHITE_VS_AI,
    AI_MODE_BLACK_VS_AI,
    AI_MODE_AI_VS_AI,
    AI_MODE_COUNT,
} AiGameMode;

typedef enum SettingsModalAction {
    SETTINGS_ACT_TIMER = 0,
    SETTINGS_ACT_PRESET,
    SETTINGS_ACT_EVAL_BAR,
    SETTINGS_ACT_STOCKFISH_BAR,
    SETTINGS_ACT_COPY_FEN,
    SETTINGS_ACT_LOAD_FEN,
    SETTINGS_ACT_OPEN_LOGS,
    SETTINGS_ACT_AI_TEST_LAB,
    SETTINGS_ACT_CLOSE,
    SETTINGS_ACT_COUNT,
} SettingsModalAction;

typedef enum AiModalAction {
    AI_ACT_HVH = 0,
    AI_ACT_WHITE_VS_AI,
    AI_ACT_BLACK_VS_AI,
    AI_ACT_AI_VS_AI,
    AI_ACT_BACKEND,
    AI_ACT_THINK_TIME,
    AI_ACT_CLOSE,
    AI_ACT_COUNT,
} AiModalAction;

typedef enum AiTestModalAction {
    AI_TEST_ACT_MATCHUP = 0,
    AI_TEST_ACT_POSITIONS,
    AI_TEST_ACT_GAMES,
    AI_TEST_ACT_STOCKFISH_SKILL,
    AI_TEST_ACT_OPPONENT_MODE,
    AI_TEST_ACT_OUR_THINK,
    AI_TEST_ACT_OPP_THINK,
    AI_TEST_ACT_START_STOP,
    AI_TEST_ACT_CLOSE,
    AI_TEST_ACT_COUNT,
} AiTestModalAction;

typedef enum AiTestMatchup {
    AI_TEST_MATCHUP_CLASSIC_VS_STOCKFISH = 0,
    AI_TEST_MATCHUP_NN_VS_STOCKFISH,
    AI_TEST_MATCHUP_CLASSIC_VS_NN,
    AI_TEST_MATCHUP_NN_VS_CLASSIC,
    AI_TEST_MATCHUP_COUNT,
} AiTestMatchup;

typedef enum UiButtonId {
    BTN_NEW_GAME = 0,
    BTN_UNDO,
    BTN_RESIGN,
    BTN_AI,
    BTN_SETTINGS,
    BTN_COUNT,
} UiButtonId;

typedef struct UiButton {
    UiButtonId id;
    Rectangle rect;
    char label[64];
    bool enabled;
} UiButton;

typedef struct UiLayout {
    Rectangle sidebar;
    Rectangle board;
    UiButton buttons[BTN_COUNT];
} UiLayout;

typedef struct ChessUi {
    GameState game;
    ChessAssets assets;
    GameLog log;

    bool board_flipped;
    int selected_sq;

    Move legal_moves[CHESS_MAX_MOVES];
    int legal_count;

    int press_sq;
    Vector2 press_mouse;
    int drag_from_sq;
    bool drag_started;

    bool timer_setting_enabled;
    int preset_index;
    bool clock_started;

    int think_ms_accum[2];

    bool promotion_modal;
    Move pending_promo_moves[8];
    int pending_promo_count;

    bool fen_modal;
    char fen_input[1024];
    char fen_error[128];

    bool ai_modal;
    int ai_mode;
    int ai_think_index;
    ChessAiBackend ai_backend;
    bool nn_backend_available;
    char nn_model_path[1024];
    AiSearchResult live_ai_result;
    bool ai_live_had_result;
    AiSearchResult last_ai_result;
    bool ai_last_had_result;
    AiWorker ai_worker;
    bool ai_worker_ready;
    bool ai_request_active;
    uint64_t ai_request_serial;
    int ai_request_side;
    uint64_t ai_request_hash;
    int ai_request_ply;
    int ai_request_started_ms;
    uint64_t position_serial;

    StockfishEvalWorker stockfish_worker;
    bool stockfish_worker_ready;
    bool stockfish_request_active;
    uint64_t stockfish_request_serial;
    int stockfish_request_side;
    uint64_t stockfish_request_hash;
    int stockfish_request_ply;

    bool settings_modal;
    bool eval_bar_enabled;
    float eval_bar_norm;
    int eval_bar_target_cp;
    int eval_bar_sample_accum_ms;
    bool stockfish_bar_enabled;
    float stockfish_bar_norm;
    int stockfish_bar_target_cp;
    int stockfish_bar_sample_accum_ms;
    StockfishEvalResult stockfish_last_result;
    bool stockfish_has_result;

    bool ai_test_modal;
    AiTestRunner ai_test_runner;
    bool ai_test_runner_ready;
    AiTestStatus ai_test_status;
    bool ai_test_use_lichess_positions;
    int ai_test_matchup_index;
    bool ai_test_stockfish_ladder;
    int ai_test_games_index;
    int ai_test_stockfish_level_index;
    int ai_test_our_time_index;
    int ai_test_stockfish_time_index;

    bool result_modal_visible;
    GameResult prev_result;
} ChessUi;

static const TimePreset k_presets[] = {
    {"Untimed", 0, 0},
    {"1+0", 60 * 1000, 0},
    {"3+2", 3 * 60 * 1000, 2 * 1000},
    {"5+0", 5 * 60 * 1000, 0},
    {"10+0", 10 * 60 * 1000, 0},
    {"15+10", 15 * 60 * 1000, 10 * 1000},
};

static const AiThinkPreset k_ai_think_presets[] = {
    {"0.1s", 100},
    {"0.25s", 250},
    {"0.5s", 500},
    {"1.0s", 1000},
    {"2.0s", 2000},
    {"3.0s", 3000},
    {"5.0s", 5000},
};

static const int k_ai_test_game_counts[] = {50, 100, 200, 500, 1000};
static const int k_ai_test_stockfish_levels[] = {1, 5, 10, 15, 20};
static const int k_ai_test_time_options_ms[] = {100, 200, 350, 500, 1000};
static const char *k_ai_test_matchup_labels[AI_TEST_MATCHUP_COUNT] = {
    "Classic vs Stockfish",
    "NN vs Stockfish",
    "Classic vs NN",
    "NN vs Classic",
};

static const char *k_ai_mode_labels[AI_MODE_COUNT] = {
    "Human vs Human",
    "White vs AI",
    "Black vs AI",
    "AI vs AI",
};

static const Color COLOR_BG = {0x1E, 0x24, 0x33, 0xFF};
static const Color COLOR_FRAME = {0x61, 0x60, 0x5E, 0xFF};
static const Color COLOR_LIGHT = {0xD2, 0xB8, 0x96, 0xFF};
static const Color COLOR_DARK = {0x8F, 0x6F, 0x55, 0xFF};
static const Color COLOR_SELECTED = {0xC9, 0x96, 0x00, 0xCC};
static const Color COLOR_LAST = {0xB3, 0x8A, 0x39, 0x88};
static const Color COLOR_LEGAL_DOT = {0x1B, 0x1B, 0x1B, 0x9A};
static const Color COLOR_CAPTURE_RING = {0xC4, 0x2B, 0x2B, 0xCC};
static const Color COLOR_CHECK = {0xDB, 0x2F, 0x2F, 0xDD};
static const Color COLOR_EVAL_LIGHT = {0xE7, 0xE7, 0xE7, 0xFF};
static const Color COLOR_EVAL_DARK = {0x1B, 0x1B, 0x1B, 0xFF};
static const Color COLOR_EVAL_STOCKFISH_LIGHT = {0x97, 0xD7, 0x9A, 0xFF};

static float eval_cp_to_norm(int eval_cp);

static bool point_in_rect(Vector2 p, Rectangle r) {
    return p.x >= r.x && p.y >= r.y && p.x <= (r.x + r.width) && p.y <= (r.y + r.height);
}

static float clampf_ui(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static int ui_now_ms(void) {
    return (int)llround(GetTime() * 1000.0);
}

static bool file_exists_ui(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return false;
    }
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static bool resolve_default_nn_model_path(char out_path[1024]) {
    const char *env_path = getenv("CHESS_NN_MODEL");
    if (env_path != NULL && env_path[0] != '\0' && file_exists_ui(env_path)) {
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
    for (size_t i = 0; i < sizeof(k_candidates) / sizeof(k_candidates[0]); ++i) {
        if (file_exists_ui(k_candidates[i])) {
            snprintf(out_path, 1024, "%s", k_candidates[i]);
            return true;
        }
    }

    out_path[0] = '\0';
    return false;
}

static ChessAiBackend next_available_ai_backend(const ChessUi *ui, ChessAiBackend current) {
    ChessAiBackend order[] = {
        CHESS_AI_BACKEND_CLASSIC,
        CHESS_AI_BACKEND_NN,
    };
    int current_index = 0;
    for (int i = 0; i < ARRAY_LEN(order); ++i) {
        if (order[i] == current) {
            current_index = i;
            break;
        }
    }

    for (int step = 1; step <= ARRAY_LEN(order); ++step) {
        ChessAiBackend candidate = order[(current_index + step) % ARRAY_LEN(order)];
        if (candidate == CHESS_AI_BACKEND_CLASSIC) {
            return candidate;
        }
        if (candidate == CHESS_AI_BACKEND_NN && ui->nn_backend_available) {
            return candidate;
        }
    }
    return CHESS_AI_BACKEND_CLASSIC;
}

static void refresh_eval_bars(ChessUi *ui) {
    ui->eval_bar_target_cp = chess_ai_eval_fast_cp(&ui->game);
    ui->eval_bar_norm = eval_cp_to_norm(ui->eval_bar_target_cp);
    ui->eval_bar_sample_accum_ms = 0;
    ui->stockfish_bar_target_cp = ui->eval_bar_target_cp;
    ui->stockfish_bar_norm = eval_cp_to_norm(ui->stockfish_bar_target_cp);
    ui->stockfish_bar_sample_accum_ms = 0;
}

static const char *ui_backend_label(ChessAiBackend backend) {
    return (backend == CHESS_AI_BACKEND_NN) ? "NN" : "Classic";
}

static void format_ai_result_summary(const char *prefix,
                                     const AiSearchResult *result,
                                     char out[96]) {
    if (out == NULL) {
        return;
    }
    if (prefix == NULL) {
        prefix = "";
    }
    if (result == NULL) {
        snprintf(out, 96, "%sNo result", prefix);
        return;
    }
    if (result->used_opening_book) {
        snprintf(out, 96, "%sBook move", prefix);
        return;
    }
    snprintf(out, 96, "%sd%d %dms n=%llu",
             prefix,
             result->depth_reached,
             result->elapsed_ms,
             (unsigned long long)result->nodes);
}

static void apply_ai_backend(ChessUi *ui) {
    if (ui->ai_backend == CHESS_AI_BACKEND_NN) {
        if (ui->nn_model_path[0] != '\0' && (chess_ai_nn_model_is_loaded() || chess_ai_set_nn_model_path(ui->nn_model_path))) {
            if (chess_ai_set_backend(CHESS_AI_BACKEND_NN)) {
                ui->nn_backend_available = true;
                return;
            }
        }
        ui->ai_backend = CHESS_AI_BACKEND_CLASSIC;
    }
    (void)chess_ai_set_backend(CHESS_AI_BACKEND_CLASSIC);
}

static Rectangle square_rect(Rectangle board, int sq, bool flipped) {
    int file = square_file(sq);
    int rank = square_rank(sq);
    int col = flipped ? (7 - file) : file;
    int row = flipped ? rank : (7 - rank);
    float cell = board.width / 8.0f;
    return (Rectangle){
        .x = board.x + (float)col * cell,
        .y = board.y + (float)row * cell,
        .width = cell,
        .height = cell,
    };
}

static int mouse_to_square(Vector2 mouse, Rectangle board, bool flipped) {
    if (!point_in_rect(mouse, board)) {
        return -1;
    }

    float cell = board.width / 8.0f;
    int col = (int)((mouse.x - board.x) / cell);
    int row = (int)((mouse.y - board.y) / cell);
    if (col < 0 || col > 7 || row < 0 || row > 7) {
        return -1;
    }

    int file = flipped ? (7 - col) : col;
    int rank = flipped ? row : (7 - row);
    return make_square(file, rank);
}

static void format_clock_mmss(int ms, char out[32]) {
    if (ms < 0) {
        ms = 0;
    }

    int total = ms / 1000;
    int minutes = total / 60;
    int seconds = total % 60;
    snprintf(out, 32, "%02d:%02d", minutes, seconds);
}

static void format_duration_hhmmss(int ms, char out[32]) {
    if (ms < 0) {
        ms = 0;
    }
    int total = ms / 1000;
    int h = total / 3600;
    int m = (total % 3600) / 60;
    int s = total % 60;
    if (h > 0) {
        snprintf(out, 32, "%d:%02d:%02d", h, m, s);
    } else {
        snprintf(out, 32, "%02d:%02d", m, s);
    }
}

static void format_ai_test_search_avg_line(char *out,
                                           size_t out_sz,
                                           const char *label,
                                           int moves,
                                           int depth_sum,
                                           uint64_t nodes_sum,
                                           int64_t time_ms_sum) {
    if (out == NULL || out_sz == 0) {
        return;
    }
    if (label == NULL) {
        label = "AI";
    }
    if (moves <= 0 || time_ms_sum <= 0) {
        snprintf(out, out_sz, "%s avg: no internal-search data yet", label);
        return;
    }

    double avg_depth = (double)depth_sum / (double)moves;
    double avg_time_ms = (double)time_ms_sum / (double)moves;
    uint64_t nps = (uint64_t)((nodes_sum * 1000ULL) / (uint64_t)((time_ms_sum > 0) ? time_ms_sum : 1));

    snprintf(out, out_sz, "%s avg: d=%.2f  nps=%llu  t=%.1fms  m=%d",
             label,
             avg_depth,
             (unsigned long long)nps,
             avg_time_ms,
             moves);
}

static const char *ai_test_positions_path(void) {
    return "data/positions/lichess_equal_positions.fen";
}

static AiTestMatchup current_ai_test_matchup(const ChessUi *ui) {
    if (ui == NULL || ui->ai_test_matchup_index < 0 || ui->ai_test_matchup_index >= AI_TEST_MATCHUP_COUNT) {
        return AI_TEST_MATCHUP_CLASSIC_VS_STOCKFISH;
    }
    return (AiTestMatchup)ui->ai_test_matchup_index;
}

static bool ai_test_matchup_uses_stockfish(AiTestMatchup matchup) {
    return matchup == AI_TEST_MATCHUP_CLASSIC_VS_STOCKFISH || matchup == AI_TEST_MATCHUP_NN_VS_STOCKFISH;
}

static ChessAiBackend ai_test_matchup_our_backend(AiTestMatchup matchup) {
    switch (matchup) {
        case AI_TEST_MATCHUP_NN_VS_STOCKFISH:
        case AI_TEST_MATCHUP_NN_VS_CLASSIC:
            return CHESS_AI_BACKEND_NN;
        case AI_TEST_MATCHUP_CLASSIC_VS_STOCKFISH:
        case AI_TEST_MATCHUP_CLASSIC_VS_NN:
        default:
            return CHESS_AI_BACKEND_CLASSIC;
    }
}

static ChessAiBackend ai_test_matchup_opponent_backend(AiTestMatchup matchup) {
    switch (matchup) {
        case AI_TEST_MATCHUP_CLASSIC_VS_NN:
            return CHESS_AI_BACKEND_NN;
        case AI_TEST_MATCHUP_NN_VS_CLASSIC:
            return CHESS_AI_BACKEND_CLASSIC;
        case AI_TEST_MATCHUP_CLASSIC_VS_STOCKFISH:
        case AI_TEST_MATCHUP_NN_VS_STOCKFISH:
        default:
            return CHESS_AI_BACKEND_CLASSIC;
    }
}

static const char *ai_test_competitor_label(AiTestMode mode, ChessAiBackend backend, bool stockfish_side) {
    if (mode == AI_TEST_MODE_VS_STOCKFISH && stockfish_side) {
        return "Stockfish";
    }
    return (backend == CHESS_AI_BACKEND_NN) ? "NN" : "Classic";
}

static void poll_ai_test_status(ChessUi *ui) {
    if (!ui->ai_test_runner_ready) {
        return;
    }

    AiTestStatus latest;
    if (ai_test_runner_poll(&ui->ai_test_runner, &latest)) {
        ui->ai_test_status = latest;
    }
}

static const char *result_text(const ChessUi *ui, char out[64]) {
    switch (ui->game.result) {
        case GAME_RESULT_WHITE_WIN: return "Checkmate: White wins";
        case GAME_RESULT_BLACK_WIN: return "Checkmate: Black wins";
        case GAME_RESULT_WHITE_WIN_RESIGN: return "White wins by resignation";
        case GAME_RESULT_BLACK_WIN_RESIGN: return "Black wins by resignation";
        case GAME_RESULT_DRAW_STALEMATE: return "Draw by stalemate";
        case GAME_RESULT_DRAW_REPETITION: return "Draw by repetition";
        case GAME_RESULT_DRAW_50: return "Draw by 50-move rule";
        case GAME_RESULT_DRAW_75: return "Draw by 75-move rule";
        case GAME_RESULT_DRAW_INSUFFICIENT: return "Draw by insufficient material";
        case GAME_RESULT_DRAW_AGREED: return "Draw agreed";
        case GAME_RESULT_WIN_TIMEOUT: {
            const char *winner = (ui->game.side_to_move == PIECE_WHITE) ? "Black" : "White";
            snprintf(out, 64, "Time out: %s wins", winner);
            return out;
        }
        case GAME_RESULT_ONGOING:
        default:
            return "";
    }
}

static bool ai_mode_side_is_ai(int ai_mode, int side) {
    switch (ai_mode) {
        case AI_MODE_WHITE_VS_AI:
            return side == PIECE_BLACK;
        case AI_MODE_BLACK_VS_AI:
            return side == PIECE_WHITE;
        case AI_MODE_AI_VS_AI:
            return true;
        case AI_MODE_HUMAN_VS_HUMAN:
        default:
            return false;
    }
}

static bool ui_side_is_ai(const ChessUi *ui, int side) {
    return ai_mode_side_is_ai(ui->ai_mode, side);
}

static void apply_player_kind_config(ChessUi *ui) {
    ui->game.config.white_kind = ui_side_is_ai(ui, PIECE_WHITE) ? PLAYER_BOT_STUB : PLAYER_LOCAL_HUMAN;
    ui->game.config.black_kind = ui_side_is_ai(ui, PIECE_BLACK) ? PLAYER_BOT_STUB : PLAYER_LOCAL_HUMAN;
}

static void invalidate_ai_request(ChessUi *ui) {
    ui->ai_request_active = false;
    ui->ai_request_serial = 0;
    ui->ai_request_side = -1;
    ui->ai_request_hash = 0;
    ui->ai_request_ply = -1;
    ui->ai_request_started_ms = 0;
    ui->ai_live_had_result = false;
    memset(&ui->live_ai_result, 0, sizeof(ui->live_ai_result));
}

static void invalidate_stockfish_request(ChessUi *ui) {
    ui->stockfish_request_active = false;
    ui->stockfish_request_serial = 0;
    ui->stockfish_request_side = -1;
    ui->stockfish_request_hash = 0;
    ui->stockfish_request_ply = -1;
}

static void bump_position_serial(ChessUi *ui) {
    ui->position_serial += 1;
    invalidate_ai_request(ui);
    invalidate_stockfish_request(ui);
}

static MatchConfig make_match_config(const ChessUi *ui) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = ui_side_is_ai(ui, PIECE_WHITE) ? PLAYER_BOT_STUB : PLAYER_LOCAL_HUMAN,
        .black_kind = ui_side_is_ai(ui, PIECE_BLACK) ? PLAYER_BOT_STUB : PLAYER_LOCAL_HUMAN,
    };

    TimePreset preset = k_presets[ui->preset_index];
    if (ui->timer_setting_enabled && preset.initial_ms > 0) {
        cfg.clock_enabled = true;
        cfg.initial_ms = preset.initial_ms;
        cfg.increment_ms = preset.increment_ms;
    }

    return cfg;
}

static void refresh_legal(ChessUi *ui) {
    ui->legal_count = chess_generate_legal_moves_mut(&ui->game, ui->legal_moves);
}

static void start_new_game(ChessUi *ui) {
    MatchConfig cfg = make_match_config(ui);
    chess_init(&ui->game, &cfg);
    apply_player_kind_config(ui);

    ui->selected_sq = -1;
    ui->press_sq = -1;
    ui->drag_from_sq = -1;
    ui->drag_started = false;

    ui->promotion_modal = false;
    ui->pending_promo_count = 0;

    ui->fen_modal = false;
    ui->fen_error[0] = '\0';
    ui->ai_modal = false;
    ui->settings_modal = false;
    ui->ai_test_modal = false;
    ui->ai_last_had_result = false;
    memset(&ui->last_ai_result, 0, sizeof(ui->last_ai_result));

    ui->clock_started = false;
    ui->think_ms_accum[PIECE_WHITE] = 0;
    ui->think_ms_accum[PIECE_BLACK] = 0;
    refresh_eval_bars(ui);
    ui->stockfish_has_result = false;
    memset(&ui->stockfish_last_result, 0, sizeof(ui->stockfish_last_result));

    ui->result_modal_visible = false;
    ui->prev_result = ui->game.result;
    bump_position_serial(ui);

    refresh_legal(ui);
    game_log_begin(&ui->log, &ui->game);
}

static bool find_pending_promo_move(const ChessUi *ui, int promo_piece, Move *out) {
    for (int i = 0; i < ui->pending_promo_count; ++i) {
        Move m = ui->pending_promo_moves[i];
        if (move_promo(m) == promo_piece) {
            if (out != NULL) {
                *out = m;
            }
            return true;
        }
    }
    return false;
}

static bool commit_chosen_move(ChessUi *ui, Move chosen, int think_ms) {
    int mover = ui->game.side_to_move;
    bool mover_ai = ui_side_is_ai(ui, mover);

    int ai_depth = 0;
    uint64_t ai_nodes = 0;
    int ai_score_cp = 0;
    if (mover_ai && ui->ai_last_had_result) {
        ai_depth = ui->last_ai_result.depth_reached;
        ai_nodes = ui->last_ai_result.nodes;
        ai_score_cp = ui->last_ai_result.score_cp;
    }

    if (!chess_make_move(&ui->game, chosen)) {
        return false;
    }
    bump_position_serial(ui);

    ui->clock_started = true;
    ui->think_ms_accum[mover] = 0;

    ui->selected_sq = -1;
    ui->promotion_modal = false;
    ui->pending_promo_count = 0;
    ui->eval_bar_sample_accum_ms = 9999;
    ui->stockfish_bar_sample_accum_ms = 9999;

    refresh_legal(ui);

    game_log_record_move(&ui->log, &ui->game, chosen, mover, think_ms, mover_ai, ai_depth, ai_nodes, ai_score_cp);
    game_log_set_result(&ui->log, ui->game.result);
    game_log_set_elapsed(&ui->log, ui->game.clock_ms[PIECE_WHITE], ui->game.clock_ms[PIECE_BLACK]);
    game_log_flush(&ui->log);
    return true;
}

static bool try_commit_move(ChessUi *ui, int from, int to, int forced_promo_piece) {
    Move candidates[8];
    int candidate_count = 0;

    for (int i = 0; i < ui->legal_count; ++i) {
        Move m = ui->legal_moves[i];
        if (move_from(m) == from && move_to(m) == to) {
            if (candidate_count < ARRAY_LEN(candidates)) {
                candidates[candidate_count++] = m;
            }
        }
    }

    if (candidate_count == 0) {
        return false;
    }

    Move chosen = candidates[0];
    if (candidate_count > 1) {
        if (forced_promo_piece == CHESS_PROMO_NONE) {
            ui->promotion_modal = true;
            ui->pending_promo_count = candidate_count;
            for (int i = 0; i < candidate_count; ++i) {
                ui->pending_promo_moves[i] = candidates[i];
            }
            return true;
        }

        bool found = false;
        for (int i = 0; i < candidate_count; ++i) {
            if (move_promo(candidates[i]) == forced_promo_piece) {
                chosen = candidates[i];
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    int mover = ui->game.side_to_move;
    return commit_chosen_move(ui, chosen, ui->think_ms_accum[mover]);
}

static void build_layout(int screen_w, int screen_h, const ChessUi *ui, UiLayout *layout) {
    const float outer = 18.0f;
    const float gap = 18.0f;
    bool stacked = screen_w < 1040;

    float board_slot_x;
    float board_slot_y;
    float board_slot_w;
    float board_slot_h;

    if (!stacked) {
        float panel_w = clampf_ui((float)screen_w * 0.25f, 230.0f, 320.0f);
        layout->sidebar = (Rectangle){outer, outer, panel_w, (float)screen_h - 2.0f * outer};

        board_slot_x = layout->sidebar.x + layout->sidebar.width + gap;
        board_slot_y = outer;
        board_slot_w = (float)screen_w - board_slot_x - outer;
        board_slot_h = (float)screen_h - 2.0f * outer;
    } else {
        float panel_h = clampf_ui((float)screen_h * 0.36f, 250.0f, 330.0f);
        layout->sidebar = (Rectangle){outer, outer, (float)screen_w - 2.0f * outer, panel_h};

        board_slot_x = outer;
        board_slot_y = layout->sidebar.y + layout->sidebar.height + gap;
        board_slot_w = (float)screen_w - 2.0f * outer;
        board_slot_h = (float)screen_h - board_slot_y - outer;
    }

    float board_size = floorf(fminf(board_slot_w, board_slot_h) - 28.0f);
    if (board_size < 160.0f) {
        board_size = floorf(fminf(board_slot_w, board_slot_h));
    }
    board_size = floorf(board_size);
    board_size -= fmodf(board_size, 8.0f);
    if (board_size < 120.0f) {
        board_size = 120.0f;
    }

    float board_x = board_slot_x + floorf((board_slot_w - board_size) * 0.5f);
    float board_y = board_slot_y + floorf((board_slot_h - board_size) * 0.5f);
    layout->board = (Rectangle){board_x, board_y, board_size, board_size};

    float clock_fs = (layout->sidebar.width < 300.0f) ? 20.0f : 24.0f;
    float clock1_y = layout->sidebar.y + ((layout->sidebar.width < 300.0f) ? 58.0f : 62.0f);
    float clock2_y = clock1_y + clock_fs + 2.0f;
    float clock_block_bottom = clock2_y + clock_fs;
    bool ai_enabled = (ui->ai_mode != AI_MODE_HUMAN_VS_HUMAN);
    bool ai_thinking = ui->ai_request_active && ui_side_is_ai(ui, ui->game.side_to_move);
    float extra_info_h = 38.0f;
    if (ai_enabled) {
        extra_info_h = 78.0f;
        if (ui->ai_last_had_result || ai_thinking) {
            extra_info_h = 98.0f;
        }
        if (ai_thinking && ui->ai_live_had_result) {
            extra_info_h = 116.0f;
        }
    }
    float btn_start = clock_block_bottom + extra_info_h;

    float btn_x = layout->sidebar.x + 14.0f;
    float btn_w = layout->sidebar.width - 28.0f;
    float btn_bottom_limit = layout->sidebar.y + layout->sidebar.height - 44.0f;
    float avail = btn_bottom_limit - btn_start;
    float btn_gap = 7.0f;
    float btn_h = (avail - 4.0f * btn_gap) / 5.0f;
    if (btn_h < 28.0f) {
        btn_gap = 4.0f;
        btn_h = (avail - 4.0f * btn_gap) / 5.0f;
    }
    btn_h = clampf_ui(btn_h, 24.0f, 44.0f);
    float y = btn_start;

    layout->buttons[BTN_NEW_GAME] = (UiButton){BTN_NEW_GAME, {btn_x, y, btn_w, btn_h}, "New Game", true};
    y += btn_h + btn_gap;
    layout->buttons[BTN_UNDO] = (UiButton){BTN_UNDO, {btn_x, y, btn_w, btn_h}, "Undo", true};
    y += btn_h + btn_gap;
    layout->buttons[BTN_RESIGN] = (UiButton){BTN_RESIGN, {btn_x, y, btn_w, btn_h}, "Resign", true};
    y += btn_h + btn_gap;
    layout->buttons[BTN_AI] = (UiButton){BTN_AI, {btn_x, y, btn_w, btn_h}, "AI", true};
    y += btn_h + btn_gap;
    layout->buttons[BTN_SETTINGS] = (UiButton){BTN_SETTINGS, {btn_x, y, btn_w, btn_h}, "Settings", true};
}

static void draw_button(const UiButton *button, Vector2 mouse) {
    Color base = (Color){56, 65, 82, 255};
    if (button->enabled && point_in_rect(mouse, button->rect)) {
        base = (Color){79, 90, 112, 255};
    }
    if (!button->enabled) {
        base = (Color){42, 46, 57, 255};
    }

    DrawRectangleRounded(button->rect, 0.14f, 8, base);
    DrawRectangleRoundedLinesEx(button->rect, 0.14f, 8, 2.0f, (Color){98, 108, 130, 255});

    int fs = (int)(button->rect.height * 0.48f);
    if (fs > 22) {
        fs = 22;
    }
    if (fs < 16) {
        fs = 16;
    }
    int tw = MeasureText(button->label, fs);
    while (tw > (int)button->rect.width - 20 && fs > 13) {
        fs -= 1;
        tw = MeasureText(button->label, fs);
    }
    DrawText(button->label,
             (int)(button->rect.x + button->rect.width * 0.5f - tw * 0.5f),
             (int)(button->rect.y + button->rect.height * 0.5f - fs * 0.5f),
             fs,
             button->enabled ? RAYWHITE : (Color){165, 165, 165, 255});
}

static void draw_wrapped_text(Rectangle box, const char *text, int font_size, Color color) {
    const float pad = 8.0f;
    float x = box.x + pad;
    float y = box.y + pad;
    float max_w = box.width - 2.0f * pad;
    float line_h = (float)font_size + 5.0f;

    char line[1024] = {0};
    int len = 0;

    BeginScissorMode((int)box.x + 1, (int)box.y + 1, (int)box.width - 2, (int)box.height - 2);

    for (size_t i = 0;; ++i) {
        char c = text[i];
        bool end = (c == '\0');

        if (!end) {
            if (len < (int)sizeof(line) - 2) {
                line[len++] = c;
                line[len] = '\0';
            }

            if (MeasureText(line, font_size) > max_w && len > 1) {
                line[len - 1] = '\0';
                DrawText(line, (int)x, (int)y, font_size, color);
                y += line_h;
                len = 0;

                if (c != ' ') {
                    line[len++] = c;
                    line[len] = '\0';
                }
            }
        }

        if (end) {
            if (len > 0) {
                DrawText(line, (int)x, (int)y, font_size, color);
            }
            break;
        }
    }

    EndScissorMode();
}

static void draw_piece(const ChessUi *ui, int sq, Rectangle dst) {
    int color = -1;
    int piece = chess_piece_on_square(&ui->game, sq, &color);
    if (piece == PIECE_NONE) {
        return;
    }

    DrawTexturePro(ui->assets.pieces_texture,
                   ui->assets.piece_src[color][piece],
                   dst,
                   (Vector2){0, 0},
                   0.0f,
                   RAYWHITE);
}

static float eval_cp_to_norm(int eval_cp) {
    return 0.5f + 0.5f * clampf_ui((float)eval_cp / 1400.0f, -1.0f, 1.0f);
}

static void update_stockfish_eval_bar_state(ChessUi *ui, int delta_ms) {
    if (!ui->stockfish_bar_enabled) {
        return;
    }
    if (!ui->stockfish_worker_ready || !stockfish_eval_worker_is_available(&ui->stockfish_worker)) {
        return;
    }
    if (ui->game.result != GAME_RESULT_ONGOING) {
        return;
    }

    if (delta_ms < 0) {
        delta_ms = 0;
    }

    if (ui->stockfish_request_active) {
        if (ui->stockfish_request_serial != ui->position_serial ||
            ui->stockfish_request_side != ui->game.side_to_move ||
            ui->stockfish_request_hash != ui->game.zobrist_hash ||
            ui->stockfish_request_ply != ui->game.ply) {
            invalidate_stockfish_request(ui);
        }
    }

    if (ui->stockfish_request_active) {
        StockfishEvalResult sf_result;
        if (stockfish_eval_worker_poll(&ui->stockfish_worker, &sf_result)) {
            bool same_position = ui->stockfish_request_serial == ui->position_serial &&
                                 ui->stockfish_request_side == ui->game.side_to_move &&
                                 ui->stockfish_request_hash == ui->game.zobrist_hash &&
                                 ui->stockfish_request_ply == ui->game.ply &&
                                 ui->game.result == GAME_RESULT_ONGOING;
            invalidate_stockfish_request(ui);
            if (same_position && sf_result.has_score) {
                ui->stockfish_last_result = sf_result;
                ui->stockfish_has_result = true;
                ui->stockfish_bar_target_cp = sf_result.score_cp;
            }
        }
    }

    if (!ui->stockfish_request_active) {
        ui->stockfish_bar_sample_accum_ms += delta_ms;
        if (ui->stockfish_bar_sample_accum_ms >= 380 || ui->stockfish_bar_sample_accum_ms < 0) {
            ui->stockfish_bar_sample_accum_ms = 0;
            StockfishEvalConfig cfg = {
                .think_time_ms = 90,
            };
            if (stockfish_eval_worker_request(&ui->stockfish_worker, &ui->game, &cfg, ui->position_serial)) {
                ui->stockfish_request_active = true;
                ui->stockfish_request_serial = ui->position_serial;
                ui->stockfish_request_side = ui->game.side_to_move;
                ui->stockfish_request_hash = ui->game.zobrist_hash;
                ui->stockfish_request_ply = ui->game.ply;
            }
        }
    }

    float target = eval_cp_to_norm(ui->stockfish_bar_target_cp);
    float t = clampf_ui(GetFrameTime() * 6.0f, 0.0f, 1.0f);
    ui->stockfish_bar_norm = ui->stockfish_bar_norm + (target - ui->stockfish_bar_norm) * t;
}

static void update_eval_bar_state(ChessUi *ui, int delta_ms) {
    if (!ui->eval_bar_enabled) {
        update_stockfish_eval_bar_state(ui, delta_ms);
        return;
    }

    if (delta_ms < 0) {
        delta_ms = 0;
    }
    if (!ui->drag_started) {
        ui->eval_bar_sample_accum_ms += delta_ms;
        if (ui->eval_bar_sample_accum_ms >= 180 || ui->eval_bar_sample_accum_ms < 0) {
            ui->eval_bar_sample_accum_ms = 0;
            ui->eval_bar_target_cp = chess_ai_eval_fast_cp(&ui->game);
        }
    }

    float target = eval_cp_to_norm(ui->eval_bar_target_cp);
    float t = clampf_ui(GetFrameTime() * 6.0f, 0.0f, 1.0f);
    ui->eval_bar_norm = ui->eval_bar_norm + (target - ui->eval_bar_norm) * t;
    update_stockfish_eval_bar_state(ui, delta_ms);
}

static void draw_one_eval_bar(Rectangle outer, float norm, Color light_color, const char *label) {
    norm = clampf_ui(norm, 0.0f, 1.0f);

    DrawRectangleRounded(outer, 0.25f, 8, (Color){40, 48, 62, 255});
    DrawRectangleRoundedLinesEx(outer, 0.25f, 8, 2.0f, (Color){98, 108, 130, 255});

    Rectangle inner = {
        .x = outer.x + 2.0f,
        .y = outer.y + 2.0f,
        .width = outer.width - 4.0f,
        .height = outer.height - 4.0f,
    };

    DrawRectangleRec(inner, COLOR_EVAL_DARK);

    float light_h = inner.height * norm;
    Rectangle light_rect = {
        .x = inner.x,
        .y = inner.y + inner.height - light_h,
        .width = inner.width,
        .height = light_h,
    };
    DrawRectangleRec(light_rect, light_color);

    float mid_y = inner.y + inner.height * 0.5f;
    DrawRectangle((int)inner.x, (int)mid_y - 1, (int)inner.width, 2, (Color){118, 124, 136, 210});

    if (label != NULL && label[0] != '\0') {
        int fs = 11;
        int tw = MeasureText(label, fs);
        DrawText(label,
                 (int)(outer.x + outer.width * 0.5f - tw * 0.5f),
                 (int)(outer.y - 14.0f),
                 fs,
                 (Color){190, 201, 222, 255});
    }
}

static void draw_eval_bars(const ChessUi *ui, Rectangle board) {
    bool draw_internal = ui->eval_bar_enabled;
    bool draw_stockfish = ui->stockfish_bar_enabled &&
                          ui->stockfish_worker_ready &&
                          stockfish_eval_worker_is_available(&ui->stockfish_worker);
    if (!draw_internal && !draw_stockfish) {
        return;
    }

    float frame_pad = 16.0f;
    float bar_w = clampf_ui(board.width * 0.024f, 14.0f, 22.0f);
    float gap = 6.0f;
    int count = (draw_internal ? 1 : 0) + (draw_stockfish ? 1 : 0);
    float total_w = (float)count * bar_w + (float)(count - 1) * gap;
    float left_x = board.x - frame_pad - total_w - 8.0f;
    if (left_x < 6.0f) {
        left_x = 6.0f;
    }

    int idx = 0;
    if (draw_stockfish) {
        Rectangle outer = {
            .x = left_x + (bar_w + gap) * (float)idx,
            .y = board.y - frame_pad,
            .width = bar_w,
            .height = board.height + frame_pad * 2.0f,
        };
        draw_one_eval_bar(outer, ui->stockfish_bar_norm, COLOR_EVAL_STOCKFISH_LIGHT, "SF");
        idx += 1;
    }
    if (draw_internal) {
        Rectangle outer = {
            .x = left_x + (bar_w + gap) * (float)idx,
            .y = board.y - frame_pad,
            .width = bar_w,
            .height = board.height + frame_pad * 2.0f,
        };
        draw_one_eval_bar(outer, ui->eval_bar_norm, COLOR_EVAL_LIGHT, "HCE");
    }
}

static void draw_board_and_pieces(ChessUi *ui, Rectangle board, Vector2 mouse) {
    float cell = board.width / 8.0f;

    draw_eval_bars(ui, board);
    DrawRectangle((int)(board.x - 16), (int)(board.y - 16), (int)(board.width + 32), (int)(board.height + 32), COLOR_FRAME);

    for (int sq = 0; sq < 64; ++sq) {
        Rectangle r = square_rect(board, sq, ui->board_flipped);
        bool light = ((square_file(sq) + square_rank(sq)) & 1) == 0;
        DrawRectangleRec(r, light ? COLOR_LIGHT : COLOR_DARK);
    }

    if (ui->game.has_last_move) {
        Rectangle from = square_rect(board, move_from(ui->game.last_move), ui->board_flipped);
        Rectangle to = square_rect(board, move_to(ui->game.last_move), ui->board_flipped);
        DrawRectangleRec(from, COLOR_LAST);
        DrawRectangleRec(to, COLOR_LAST);
    }

    if (ui->selected_sq >= 0) {
        Rectangle selected = square_rect(board, ui->selected_sq, ui->board_flipped);
        DrawRectangleRec(selected, COLOR_SELECTED);

        for (int i = 0; i < ui->legal_count; ++i) {
            Move m = ui->legal_moves[i];
            if (move_from(m) != ui->selected_sq) {
                continue;
            }

            int to = move_to(m);
            Rectangle tr = square_rect(board, to, ui->board_flipped);
            Vector2 center = {tr.x + tr.width * 0.5f, tr.y + tr.height * 0.5f};

            if (move_has_flag(m, MOVE_FLAG_CAPTURE)) {
                DrawCircleLines((int)center.x, (int)center.y, tr.width * 0.37f, COLOR_CAPTURE_RING);
                DrawCircleLines((int)center.x, (int)center.y, tr.width * 0.35f, COLOR_CAPTURE_RING);
            } else {
                DrawCircleV(center, tr.width * 0.12f, COLOR_LEGAL_DOT);
            }
        }
    }

    int drag_piece = PIECE_NONE;
    int drag_color = -1;
    if (ui->drag_started && ui->drag_from_sq >= 0) {
        drag_piece = chess_piece_on_square(&ui->game, ui->drag_from_sq, &drag_color);
    }

    for (int sq = 0; sq < 64; ++sq) {
        if (ui->drag_started && sq == ui->drag_from_sq) {
            continue;
        }

        Rectangle sr = square_rect(board, sq, ui->board_flipped);
        float pad = sr.width * 0.05f;
        Rectangle dst = {sr.x + pad, sr.y + pad, sr.width - 2.0f * pad, sr.height - 2.0f * pad};
        draw_piece(ui, sq, dst);
    }

    if (ui->drag_started && drag_piece != PIECE_NONE) {
        float size = cell * 0.90f;
        Rectangle dst = {mouse.x - size * 0.5f, mouse.y - size * 0.5f, size, size};
        DrawTexturePro(ui->assets.pieces_texture,
                       ui->assets.piece_src[drag_color][drag_piece],
                       dst,
                       (Vector2){0, 0},
                       0.0f,
                       RAYWHITE);
    }

    for (int side = 0; side < 2; ++side) {
        if (!chess_in_check(&ui->game, side)) {
            continue;
        }
        int king_sq = chess_find_king_square(&ui->game, side);
        if (king_sq < 0) {
            continue;
        }
        Rectangle kr = square_rect(board, king_sq, ui->board_flipped);
        DrawRectangleLinesEx(kr, 4.0f, COLOR_CHECK);
    }
}

static void draw_sidebar(const ChessUi *ui, const UiLayout *layout, Vector2 mouse) {
    Rectangle panel = layout->sidebar;
    DrawRectangleRounded(panel, 0.05f, 8, (Color){34, 40, 54, 255});
    DrawRectangleRoundedLinesEx(panel, 0.05f, 8, 2.0f, (Color){89, 101, 123, 255});

    int title_fs = (panel.width < 300.0f) ? 34 : 42;
    DrawText("Chess", (int)panel.x + 14, (int)panel.y + 14, title_fs, RAYWHITE);

    char white[32];
    char black[32];
    format_clock_mmss(ui->game.clock_ms[PIECE_WHITE], white);
    format_clock_mmss(ui->game.clock_ms[PIECE_BLACK], black);

    int clock_fs = (panel.width < 300.0f) ? 20 : 24;
    int clock_y = (panel.width < 300.0f) ? 58 : 62;
    DrawText(TextFormat("White  %s", white), (int)panel.x + 14, (int)panel.y + clock_y, clock_fs,
             ui->game.side_to_move == PIECE_WHITE ? (Color){242, 242, 242, 255} : (Color){196, 196, 196, 255});
    DrawText(TextFormat("Black  %s", black), (int)panel.x + 14, (int)panel.y + clock_y + clock_fs + 2, clock_fs,
             ui->game.side_to_move == PIECE_BLACK ? (Color){242, 242, 242, 255} : (Color){196, 196, 196, 255});

    DrawText(TextFormat("Mode: %s", k_ai_mode_labels[ui->ai_mode]), (int)panel.x + 14, (int)panel.y + clock_y + clock_fs * 2 + 12, 16,
             (Color){200, 208, 225, 255});
    bool ai_enabled = (ui->ai_mode != AI_MODE_HUMAN_VS_HUMAN);
    if (ai_enabled) {
        char result_line[96];
        int info_y = (int)panel.y + clock_y + clock_fs * 2 + 30;
        DrawText(TextFormat("Engine: %s", ui_backend_label(ui->ai_backend)), (int)panel.x + 14, info_y, 16,
                 (Color){200, 208, 225, 255});
        int detail_y = info_y + 18;
        DrawText(TextFormat("AI budget: %s", k_ai_think_presets[ui->ai_think_index].label), (int)panel.x + 14, detail_y, 16,
                 (Color){200, 208, 225, 255});
        if (ui->ai_request_active && ui_side_is_ai(ui, ui->game.side_to_move)) {
            int elapsed_ms = ui_now_ms() - ui->ai_request_started_ms;
            if (elapsed_ms < 0) {
                elapsed_ms = 0;
            }
            DrawText(TextFormat("AI thinking: %dms", elapsed_ms),
                     (int)panel.x + 14,
                     detail_y + 18,
                     15,
                     (Color){186, 199, 220, 255});
            if (ui->ai_live_had_result) {
                format_ai_result_summary("Live ", &ui->live_ai_result, result_line);
                DrawText(result_line,
                         (int)panel.x + 14,
                         detail_y + 36,
                         15,
                         (Color){173, 188, 210, 255});
            }
        } else if (ui->ai_last_had_result) {
            format_ai_result_summary("Last ", &ui->last_ai_result, result_line);
            DrawText(result_line,
                     (int)panel.x + 14,
                     detail_y + 18,
                     15,
                     (Color){173, 188, 210, 255});
        }
    }

    DrawText("Press B to flip board", (int)panel.x + 14, (int)panel.y + (int)panel.height - 24, 16, (Color){169, 180, 206, 255});

    for (int i = 0; i < BTN_COUNT; ++i) {
        draw_button(&layout->buttons[i], mouse);
    }
}

static void draw_promotion_modal(const ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 170});
    Rectangle modal = {(float)screen_w * 0.5f - 250.0f, (float)screen_h * 0.5f - 120.0f, 500.0f, 240.0f};
    DrawRectangleRounded(modal, 0.10f, 10, (Color){33, 39, 51, 255});
    DrawRectangleRoundedLinesEx(modal, 0.10f, 10, 2.0f, (Color){96, 108, 131, 255});
    DrawText("Choose Promotion", (int)modal.x + 130, (int)modal.y + 20, 30, RAYWHITE);

    const int order[4] = {PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT};
    const float bw = 98.0f;
    const float gap = 14.0f;
    const float start_x = modal.x + (modal.width - (4.0f * bw + 3.0f * gap)) * 0.5f;

    int color = ui->game.side_to_move;
    for (int i = 0; i < 4; ++i) {
        Move promo_move;
        if (!find_pending_promo_move(ui, order[i], &promo_move)) {
            continue;
        }

        Rectangle br = {start_x + i * (bw + gap), modal.y + 94.0f, bw, 102.0f};
        Color fill = point_in_rect(mouse, br) ? (Color){81, 92, 113, 255} : (Color){58, 67, 83, 255};
        DrawRectangleRounded(br, 0.10f, 8, fill);
        DrawRectangleRoundedLinesEx(br, 0.10f, 8, 2.0f, (Color){104, 117, 142, 255});

        Rectangle src = ui->assets.piece_src[color][order[i]];
        Rectangle dst = {br.x + 8.0f, br.y + 8.0f, br.width - 16.0f, br.height - 16.0f};
        DrawTexturePro(ui->assets.pieces_texture, src, dst, (Vector2){0, 0}, 0.0f, RAYWHITE);
    }
}

static void draw_fen_modal(const ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 180});

    Rectangle modal = {(float)screen_w * 0.5f - 390.0f, (float)screen_h * 0.5f - 170.0f, 780.0f, 340.0f};
    DrawRectangleRounded(modal, 0.07f, 10, (Color){30, 35, 47, 255});
    DrawRectangleRoundedLinesEx(modal, 0.07f, 10, 2.0f, (Color){93, 106, 129, 255});

    DrawText("Load FEN", (int)modal.x + 20, (int)modal.y + 16, 34, RAYWHITE);
    DrawText("Paste FEN and press Enter or Load", (int)modal.x + 20, (int)modal.y + 56, 21, (Color){194, 202, 220, 255});

    Rectangle input = {modal.x + 20.0f, modal.y + 94.0f, modal.width - 40.0f, 170.0f};
    DrawRectangleRounded(input, 0.05f, 8, (Color){18, 23, 32, 255});
    DrawRectangleRoundedLinesEx(input, 0.05f, 8, 2.0f, (Color){84, 95, 116, 255});
    draw_wrapped_text(input, ui->fen_input, 22, RAYWHITE);

    if (ui->fen_error[0] != '\0') {
        DrawText(ui->fen_error, (int)modal.x + 22, (int)modal.y + 272, 18, (Color){233, 104, 104, 255});
    }

    Rectangle load_btn = {modal.x + modal.width - 220.0f, modal.y + modal.height - 56.0f, 90.0f, 38.0f};
    Rectangle cancel_btn = {modal.x + modal.width - 110.0f, modal.y + modal.height - 56.0f, 90.0f, 38.0f};

    UiButton load = {0, load_btn, "Load", true};
    UiButton cancel = {0, cancel_btn, "Cancel", true};
    draw_button(&load, mouse);
    draw_button(&cancel, mouse);
}

static int build_settings_modal_widgets(const ChessUi *ui, int screen_w, int screen_h, Rectangle *out_modal, UiButton out[SETTINGS_ACT_COUNT]) {
    float modal_w = clampf_ui((float)screen_w * 0.60f, 420.0f, 700.0f);
    float modal_h = clampf_ui((float)screen_h * 0.78f, 560.0f, 740.0f);
    Rectangle modal = {
        .x = floorf((float)screen_w * 0.5f - modal_w * 0.5f),
        .y = floorf((float)screen_h * 0.5f - modal_h * 0.5f),
        .width = modal_w,
        .height = modal_h,
    };
    *out_modal = modal;

    float btn_x = modal.x + 20.0f;
    float btn_w = modal.width - 40.0f;
    float btn_h = 34.0f;
    float gap = 6.0f;
    float y = modal.y + 126.0f;

    snprintf(out[SETTINGS_ACT_TIMER].label, sizeof(out[SETTINGS_ACT_TIMER].label), "Timer: %s", ui->timer_setting_enabled ? "On" : "Off");
    out[SETTINGS_ACT_TIMER].id = BTN_SETTINGS;
    out[SETTINGS_ACT_TIMER].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_TIMER].enabled = true;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_PRESET].label, sizeof(out[SETTINGS_ACT_PRESET].label), "Preset: %s", k_presets[ui->preset_index].label);
    out[SETTINGS_ACT_PRESET].id = BTN_SETTINGS;
    out[SETTINGS_ACT_PRESET].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_PRESET].enabled = true;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_EVAL_BAR].label, sizeof(out[SETTINGS_ACT_EVAL_BAR].label), "Eval Bar: %s", ui->eval_bar_enabled ? "On" : "Off");
    out[SETTINGS_ACT_EVAL_BAR].id = BTN_SETTINGS;
    out[SETTINGS_ACT_EVAL_BAR].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_EVAL_BAR].enabled = true;
    y += btn_h + gap;

    bool sf_available = ui->stockfish_worker_ready &&
                        stockfish_eval_worker_is_available(&ui->stockfish_worker);
    if (sf_available) {
        snprintf(out[SETTINGS_ACT_STOCKFISH_BAR].label, sizeof(out[SETTINGS_ACT_STOCKFISH_BAR].label), "Stockfish Debug Bar: %s", ui->stockfish_bar_enabled ? "On" : "Off");
    } else {
        snprintf(out[SETTINGS_ACT_STOCKFISH_BAR].label, sizeof(out[SETTINGS_ACT_STOCKFISH_BAR].label), "Stockfish Debug Bar: Unavailable");
    }
    out[SETTINGS_ACT_STOCKFISH_BAR].id = BTN_SETTINGS;
    out[SETTINGS_ACT_STOCKFISH_BAR].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_STOCKFISH_BAR].enabled = sf_available;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_COPY_FEN].label, sizeof(out[SETTINGS_ACT_COPY_FEN].label), "Copy FEN");
    out[SETTINGS_ACT_COPY_FEN].id = BTN_SETTINGS;
    out[SETTINGS_ACT_COPY_FEN].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_COPY_FEN].enabled = true;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_LOAD_FEN].label, sizeof(out[SETTINGS_ACT_LOAD_FEN].label), "Load FEN");
    out[SETTINGS_ACT_LOAD_FEN].id = BTN_SETTINGS;
    out[SETTINGS_ACT_LOAD_FEN].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_LOAD_FEN].enabled = true;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_OPEN_LOGS].label, sizeof(out[SETTINGS_ACT_OPEN_LOGS].label), "Open Logs Folder");
    out[SETTINGS_ACT_OPEN_LOGS].id = BTN_SETTINGS;
    out[SETTINGS_ACT_OPEN_LOGS].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_OPEN_LOGS].enabled = true;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_AI_TEST_LAB].label, sizeof(out[SETTINGS_ACT_AI_TEST_LAB].label), "AI Test Lab");
    out[SETTINGS_ACT_AI_TEST_LAB].id = BTN_SETTINGS;
    out[SETTINGS_ACT_AI_TEST_LAB].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_AI_TEST_LAB].enabled = true;
    y += btn_h + gap;

    snprintf(out[SETTINGS_ACT_CLOSE].label, sizeof(out[SETTINGS_ACT_CLOSE].label), "Close");
    out[SETTINGS_ACT_CLOSE].id = BTN_SETTINGS;
    out[SETTINGS_ACT_CLOSE].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[SETTINGS_ACT_CLOSE].enabled = true;

    return SETTINGS_ACT_COUNT;
}

static void draw_settings_modal(const ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 180});

    Rectangle modal;
    UiButton buttons[SETTINGS_ACT_COUNT];
    int count = build_settings_modal_widgets(ui, screen_w, screen_h, &modal, buttons);

    DrawRectangleRounded(modal, 0.08f, 10, (Color){30, 35, 47, 255});
    DrawRectangleRoundedLinesEx(modal, 0.08f, 10, 2.0f, (Color){93, 106, 129, 255});

    DrawText("Settings", (int)modal.x + 20, (int)modal.y + 18, 34, RAYWHITE);
    DrawText("Clock, eval bars, FEN, logs, and test lab", (int)modal.x + 20, (int)modal.y + 56, 20, (Color){194, 202, 220, 255});
    DrawText(TextFormat("Engine: %s",
                        chess_ai_backend_name(ui->ai_backend)),
             (int)modal.x + 20,
             (int)modal.y + 80,
             14,
             (Color){168, 184, 208, 255});

    for (int i = 0; i < count; ++i) {
        draw_button(&buttons[i], mouse);
    }
}

static int build_ai_modal_widgets(const ChessUi *ui, int screen_w, int screen_h, Rectangle *out_modal, UiButton out[AI_ACT_COUNT]) {
    float modal_w = clampf_ui((float)screen_w * 0.62f, 440.0f, 760.0f);
    float modal_h = clampf_ui((float)screen_h * 0.74f, 420.0f, 620.0f);
    Rectangle modal = {
        .x = floorf((float)screen_w * 0.5f - modal_w * 0.5f),
        .y = floorf((float)screen_h * 0.5f - modal_h * 0.5f),
        .width = modal_w,
        .height = modal_h,
    };
    *out_modal = modal;

    float btn_x = modal.x + 20.0f;
    float btn_w = modal.width - 40.0f;
    float btn_h = 44.0f;
    float gap = 8.0f;
    float y = modal.y + 86.0f;

    snprintf(out[AI_ACT_HVH].label, sizeof(out[AI_ACT_HVH].label), "%s%s", ui->ai_mode == AI_MODE_HUMAN_VS_HUMAN ? "[x] " : "[ ] ", k_ai_mode_labels[AI_MODE_HUMAN_VS_HUMAN]);
    out[AI_ACT_HVH].id = BTN_AI;
    out[AI_ACT_HVH].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_HVH].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_ACT_WHITE_VS_AI].label, sizeof(out[AI_ACT_WHITE_VS_AI].label), "%s%s", ui->ai_mode == AI_MODE_WHITE_VS_AI ? "[x] " : "[ ] ", k_ai_mode_labels[AI_MODE_WHITE_VS_AI]);
    out[AI_ACT_WHITE_VS_AI].id = BTN_AI;
    out[AI_ACT_WHITE_VS_AI].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_WHITE_VS_AI].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_ACT_BLACK_VS_AI].label, sizeof(out[AI_ACT_BLACK_VS_AI].label), "%s%s", ui->ai_mode == AI_MODE_BLACK_VS_AI ? "[x] " : "[ ] ", k_ai_mode_labels[AI_MODE_BLACK_VS_AI]);
    out[AI_ACT_BLACK_VS_AI].id = BTN_AI;
    out[AI_ACT_BLACK_VS_AI].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_BLACK_VS_AI].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_ACT_AI_VS_AI].label, sizeof(out[AI_ACT_AI_VS_AI].label), "%s%s", ui->ai_mode == AI_MODE_AI_VS_AI ? "[x] " : "[ ] ", k_ai_mode_labels[AI_MODE_AI_VS_AI]);
    out[AI_ACT_AI_VS_AI].id = BTN_AI;
    out[AI_ACT_AI_VS_AI].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_AI_VS_AI].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_ACT_BACKEND].label,
             sizeof(out[AI_ACT_BACKEND].label),
             "Engine backend: %s%s",
             ui_backend_label(ui->ai_backend),
             (!ui->nn_backend_available && ui->ai_backend == CHESS_AI_BACKEND_CLASSIC)
                 ? " (nn unavailable)"
                 : "");
    out[AI_ACT_BACKEND].id = BTN_AI;
    out[AI_ACT_BACKEND].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_BACKEND].enabled = ui->nn_backend_available;
    y += btn_h + gap;

    snprintf(out[AI_ACT_THINK_TIME].label, sizeof(out[AI_ACT_THINK_TIME].label), "AI think time: %s", k_ai_think_presets[ui->ai_think_index].label);
    out[AI_ACT_THINK_TIME].id = BTN_AI;
    out[AI_ACT_THINK_TIME].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_THINK_TIME].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_ACT_CLOSE].label, sizeof(out[AI_ACT_CLOSE].label), "Close");
    out[AI_ACT_CLOSE].id = BTN_AI;
    out[AI_ACT_CLOSE].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_ACT_CLOSE].enabled = true;

    return AI_ACT_COUNT;
}

static void draw_ai_modal(const ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 180});

    Rectangle modal;
    UiButton buttons[AI_ACT_COUNT];
    int count = build_ai_modal_widgets(ui, screen_w, screen_h, &modal, buttons);

    DrawRectangleRounded(modal, 0.08f, 10, (Color){30, 35, 47, 255});
    DrawRectangleRoundedLinesEx(modal, 0.08f, 10, 2.0f, (Color){93, 106, 129, 255});

    DrawText("AI Setup", (int)modal.x + 20, (int)modal.y + 18, 34, RAYWHITE);
    DrawText("Pick mode, backend, and think time", (int)modal.x + 20, (int)modal.y + 56, 20, (Color){194, 202, 220, 255});

    for (int i = 0; i < count; ++i) {
        if (buttons[i].rect.width < 2.0f || buttons[i].rect.height < 2.0f) {
            continue;
        }
        draw_button(&buttons[i], mouse);
    }
}

static int build_ai_test_modal_widgets(const ChessUi *ui, int screen_w, int screen_h, Rectangle *out_modal, UiButton out[AI_TEST_ACT_COUNT]) {
    float modal_w = clampf_ui((float)screen_w * 0.66f, 520.0f, 860.0f);
    float modal_h = clampf_ui((float)screen_h * 0.82f, 560.0f, 760.0f);
    Rectangle modal = {
        .x = floorf((float)screen_w * 0.5f - modal_w * 0.5f),
        .y = floorf((float)screen_h * 0.5f - modal_h * 0.5f),
        .width = modal_w,
        .height = modal_h,
    };
    *out_modal = modal;

    float btn_x = modal.x + 20.0f;
    float btn_w = modal.width - 40.0f;
    float gap = 6.0f;
    float y = modal.y + 120.0f;
    // Keep button area clear of the AI test info panel (which now has extra metric rows).
    float info_top = modal.y + modal.height - 214.0f;
    int button_count = AI_TEST_ACT_COUNT;
    float btn_h = (info_top - y - (float)(button_count - 1) * gap - 8.0f) / (float)button_count;
    btn_h = clampf_ui(btn_h, 20.0f, 40.0f);

    AiTestMatchup matchup = current_ai_test_matchup(ui);
    bool uses_stockfish = ai_test_matchup_uses_stockfish(matchup);
    bool lichess_exists = FileExists(ai_test_positions_path());
    const char *our_label = ai_test_competitor_label(uses_stockfish ? AI_TEST_MODE_VS_STOCKFISH : AI_TEST_MODE_INTERNAL,
                                                     ai_test_matchup_our_backend(matchup),
                                                     false);
    const char *opp_label = ai_test_competitor_label(uses_stockfish ? AI_TEST_MODE_VS_STOCKFISH : AI_TEST_MODE_INTERNAL,
                                                     ai_test_matchup_opponent_backend(matchup),
                                                     uses_stockfish);

    snprintf(out[AI_TEST_ACT_MATCHUP].label, sizeof(out[AI_TEST_ACT_MATCHUP].label), "Matchup: %s", k_ai_test_matchup_labels[matchup]);
    out[AI_TEST_ACT_MATCHUP].id = BTN_SETTINGS;
    out[AI_TEST_ACT_MATCHUP].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_MATCHUP].enabled = !ui->ai_test_status.running;
    y += btn_h + gap;

    if (ui->ai_test_use_lichess_positions) {
        snprintf(out[AI_TEST_ACT_POSITIONS].label, sizeof(out[AI_TEST_ACT_POSITIONS].label), "Positions: Lichess file (%s)", lichess_exists ? "found" : "missing");
    } else {
        snprintf(out[AI_TEST_ACT_POSITIONS].label, sizeof(out[AI_TEST_ACT_POSITIONS].label), "Positions: Built-in equal set");
    }
    out[AI_TEST_ACT_POSITIONS].id = BTN_SETTINGS;
    out[AI_TEST_ACT_POSITIONS].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_POSITIONS].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_TEST_ACT_GAMES].label, sizeof(out[AI_TEST_ACT_GAMES].label), "Games: %d", k_ai_test_game_counts[ui->ai_test_games_index]);
    out[AI_TEST_ACT_GAMES].id = BTN_SETTINGS;
    out[AI_TEST_ACT_GAMES].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_GAMES].enabled = true;
    y += btn_h + gap;

    if (uses_stockfish) {
        snprintf(out[AI_TEST_ACT_STOCKFISH_SKILL].label, sizeof(out[AI_TEST_ACT_STOCKFISH_SKILL].label), "Stockfish skill: %d", k_ai_test_stockfish_levels[ui->ai_test_stockfish_level_index]);
    } else {
        snprintf(out[AI_TEST_ACT_STOCKFISH_SKILL].label, sizeof(out[AI_TEST_ACT_STOCKFISH_SKILL].label), "Stockfish skill: N/A");
    }
    out[AI_TEST_ACT_STOCKFISH_SKILL].enabled = uses_stockfish && !ui->ai_test_status.running;
    out[AI_TEST_ACT_STOCKFISH_SKILL].id = BTN_SETTINGS;
    out[AI_TEST_ACT_STOCKFISH_SKILL].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    y += btn_h + gap;

    if (uses_stockfish) {
        snprintf(out[AI_TEST_ACT_OPPONENT_MODE].label, sizeof(out[AI_TEST_ACT_OPPONENT_MODE].label), "Opponent mode: %s", ui->ai_test_stockfish_ladder ? "Elo Ladder Sweep" : "Fixed Opponent");
    } else {
        snprintf(out[AI_TEST_ACT_OPPONENT_MODE].label, sizeof(out[AI_TEST_ACT_OPPONENT_MODE].label), "Opponent mode: Internal H2H");
    }
    out[AI_TEST_ACT_OPPONENT_MODE].enabled = uses_stockfish && !ui->ai_test_status.running;
    out[AI_TEST_ACT_OPPONENT_MODE].id = BTN_SETTINGS;
    out[AI_TEST_ACT_OPPONENT_MODE].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    y += btn_h + gap;

    snprintf(out[AI_TEST_ACT_OUR_THINK].label,
             sizeof(out[AI_TEST_ACT_OUR_THINK].label),
             "%s think time: %dms",
             our_label,
             k_ai_test_time_options_ms[ui->ai_test_our_time_index]);
    out[AI_TEST_ACT_OUR_THINK].id = BTN_SETTINGS;
    out[AI_TEST_ACT_OUR_THINK].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_OUR_THINK].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_TEST_ACT_OPP_THINK].label, sizeof(out[AI_TEST_ACT_OPP_THINK].label), "%s think time: %dms", opp_label, k_ai_test_time_options_ms[ui->ai_test_stockfish_time_index]);
    out[AI_TEST_ACT_OPP_THINK].id = BTN_SETTINGS;
    out[AI_TEST_ACT_OPP_THINK].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_OPP_THINK].enabled = true;
    y += btn_h + gap;

    snprintf(out[AI_TEST_ACT_START_STOP].label, sizeof(out[AI_TEST_ACT_START_STOP].label), "%s", ui->ai_test_status.running ? "Stop Test" : "Start Test");
    out[AI_TEST_ACT_START_STOP].id = BTN_SETTINGS;
    out[AI_TEST_ACT_START_STOP].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_START_STOP].enabled = ui->ai_test_runner_ready;
    y += btn_h + gap;

    snprintf(out[AI_TEST_ACT_CLOSE].label, sizeof(out[AI_TEST_ACT_CLOSE].label), "Close");
    out[AI_TEST_ACT_CLOSE].id = BTN_SETTINGS;
    out[AI_TEST_ACT_CLOSE].rect = (Rectangle){btn_x, y, btn_w, btn_h};
    out[AI_TEST_ACT_CLOSE].enabled = true;

    return AI_TEST_ACT_COUNT;
}

static void draw_ai_test_modal(const ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 185});

    Rectangle modal;
    UiButton buttons[AI_TEST_ACT_COUNT];
    int count = build_ai_test_modal_widgets(ui, screen_w, screen_h, &modal, buttons);

    AiTestMode mode = ui->ai_test_status.mode;
    if (!ui->ai_test_status.running && !ui->ai_test_status.completed && !ui->ai_test_status.failed && !ui->ai_test_status.canceled) {
        mode = ai_test_matchup_uses_stockfish(current_ai_test_matchup(ui)) ? AI_TEST_MODE_VS_STOCKFISH : AI_TEST_MODE_INTERNAL;
    }
    ChessAiBackend our_backend = ui->ai_test_status.our_backend;
    ChessAiBackend opp_backend = ui->ai_test_status.opponent_backend;
    if (!ui->ai_test_status.running && !ui->ai_test_status.completed && !ui->ai_test_status.failed && !ui->ai_test_status.canceled) {
        AiTestMatchup matchup = current_ai_test_matchup(ui);
        our_backend = ai_test_matchup_our_backend(matchup);
        opp_backend = ai_test_matchup_opponent_backend(matchup);
    }
    const char *our_label = ai_test_competitor_label(mode, our_backend, false);
    const char *opp_label = ai_test_competitor_label(mode, opp_backend, mode == AI_TEST_MODE_VS_STOCKFISH);

    DrawRectangleRounded(modal, 0.08f, 10, (Color){30, 35, 47, 255});
    DrawRectangleRoundedLinesEx(modal, 0.08f, 10, 2.0f, (Color){93, 106, 129, 255});

    DrawText("AI Test Lab", (int)modal.x + 20, (int)modal.y + 18, 34, RAYWHITE);
    DrawText("Run backend matchups from balanced non-start positions",
             (int)modal.x + 20,
             (int)modal.y + 56,
             20,
             (Color){194, 202, 220, 255});
    DrawText(TextFormat("Lichess path: %s", ai_test_positions_path()), (int)modal.x + 20, (int)modal.y + 82, 14, (Color){167, 183, 206, 255});

    for (int i = 0; i < count; ++i) {
        draw_button(&buttons[i], mouse);
    }

    Rectangle info = {modal.x + 20.0f, modal.y + modal.height - 214.0f, modal.width - 40.0f, 188.0f};
    DrawRectangleRounded(info, 0.05f, 8, (Color){18, 23, 32, 255});
    DrawRectangleRoundedLinesEx(info, 0.05f, 8, 2.0f, (Color){84, 95, 116, 255});

    const AiTestStatus *st = &ui->ai_test_status;
    char eta[32];
    char bracket_line[160] = {0};
    char search_line_our[192];
    char search_line_opp[192];
    format_duration_hhmmss(st->eta_ms, eta);
    search_line_our[0] = '\0';
    search_line_opp[0] = '\0';

    const char *our_search_label = our_label;
    const char *opp_search_label = opp_label;

    format_ai_test_search_avg_line(search_line_our, sizeof(search_line_our),
                                   our_search_label,
                                   st->our_search_moves,
                                   st->our_search_depth_sum,
                                   st->our_search_nodes_sum,
                                   st->our_search_time_ms_sum);
    if (mode == AI_TEST_MODE_VS_STOCKFISH) {
        snprintf(search_line_opp, sizeof(search_line_opp), "%s avg: external engine (not tracked here)", opp_search_label);
    } else {
        format_ai_test_search_avg_line(search_line_opp, sizeof(search_line_opp),
                                       opp_search_label,
                                       st->opp_search_moves,
                                       st->opp_search_depth_sum,
                                       st->opp_search_nodes_sum,
                                       st->opp_search_time_ms_sum);
    }

    int off = 0;
    for (int i = 0; i < st->bracket_count; ++i) {
        int n = st->bracket_games[i];
        if (n <= 0) {
            continue;
        }
        double pct = ((double)st->bracket_wins[i] + 0.5 * (double)st->bracket_draws[i]) * 100.0 / (double)n;
        char part[48];
        snprintf(part, sizeof(part), "%d:%.0f%% ", st->bracket_elo[i], pct);
        int part_len = (int)strlen(part);
        if (off + part_len + 1 >= (int)sizeof(bracket_line)) {
            break;
        }
        memcpy(bracket_line + off, part, (size_t)part_len);
        off += part_len;
        bracket_line[off] = '\0';
    }

    DrawText(TextFormat("Status: %s", st->message[0] ? st->message : (ui->ai_test_runner_ready ? "Idle" : "Runner unavailable")),
             (int)info.x + 10,
             (int)info.y + 10,
             18,
             (Color){219, 225, 239, 255});
    DrawText(TextFormat("Games: %d/%d  %s record W:%d D:%d L:%d",
                        st->games_completed,
                        st->total_games,
                        our_label,
                        st->wins,
                        st->draws,
                        st->losses),
             (int)info.x + 10,
             (int)info.y + 36,
             17,
             (Color){190, 202, 223, 255});
    const bool elo_provisional = st->games_completed < 20;
    if (st->estimated_elo_is_delta) {
        DrawText(TextFormat("Score: %.1f%%   Elo Delta (%s minus %s): %+d%s",
                            st->score_pct * 100.0,
                            our_label,
                            opp_label,
                            st->estimated_elo,
                            elo_provisional ? " (provisional)" : ""),
                 (int)info.x + 10,
                 (int)info.y + 60,
                 17,
                 (Color){190, 202, 223, 255});
    } else {
        DrawText(TextFormat("Score: %.1f%%   Estimated Elo: %d", st->score_pct * 100.0, st->estimated_elo),
                 (int)info.x + 10,
                 (int)info.y + 60,
                 17,
                 (Color){190, 202, 223, 255});
    }
    DrawText(TextFormat("Matchup: %s vs %s   ETA: %s",
                        our_label,
                        opp_label,
                        eta),
             (int)info.x + 10,
             (int)info.y + 84,
             17,
             (Color){190, 202, 223, 255});
    DrawText(TextFormat("Current game ply: %d   Positions loaded: %d",
                        st->current_ply,
                        st->positions_loaded),
             (int)info.x + 10,
             (int)info.y + 108,
             17,
             (Color){190, 202, 223, 255});
    {
        const bool our_is_white = (st->current_our_side == PIECE_WHITE);
        DrawText(TextFormat("This game colors: %s=%s  %s=%s",
                            our_label,
                            our_is_white ? "White" : "Black",
                            opp_label,
                            our_is_white ? "Black" : "White"),
                 (int)info.x + 10,
                 (int)info.y + 126,
                 16,
                 (Color){170, 185, 210, 255});
    }
    DrawText(search_line_our,
             (int)info.x + 10,
             (int)info.y + 144,
             15,
             (Color){170, 185, 210, 255});
    DrawText(search_line_opp,
             (int)info.x + 10,
             (int)info.y + 162,
             15,
             (Color){170, 185, 210, 255});
    if (bracket_line[0] != '\0') {
        DrawText(TextFormat("Bracket scores: %s", bracket_line),
                 (int)info.x + 10,
                 (int)info.y + 178,
                 16,
                 (Color){170, 185, 210, 255});
    }
}

static void draw_result_modal(const ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    if (!ui->result_modal_visible || ui->game.result == GAME_RESULT_ONGOING) {
        return;
    }

    DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 175});
    Rectangle modal = {(float)screen_w * 0.5f - 250.0f, (float)screen_h * 0.5f - 120.0f, 500.0f, 240.0f};
    DrawRectangleRounded(modal, 0.10f, 10, (Color){31, 37, 50, 255});
    DrawRectangleRoundedLinesEx(modal, 0.10f, 10, 2.0f, (Color){98, 111, 135, 255});

    char text_buf[64];
    DrawText(result_text(ui, text_buf), (int)modal.x + 42, (int)modal.y + 42, 34, RAYWHITE);

    UiButton new_game = {0, {modal.x + 72.0f, modal.y + 154.0f, 150.0f, 44.0f}, "New Game", true};
    UiButton close = {0, {modal.x + 280.0f, modal.y + 154.0f, 150.0f, 44.0f}, "Close", true};
    draw_button(&new_game, mouse);
    draw_button(&close, mouse);
}

static void handle_fen_text_input(ChessUi *ui) {
    int c = GetCharPressed();
    while (c > 0) {
        if (c >= 32 && c <= 126) {
            size_t len = strlen(ui->fen_input);
            if (len + 1 < sizeof(ui->fen_input)) {
                ui->fen_input[len] = (char)c;
                ui->fen_input[len + 1] = '\0';
            }
        }
        c = GetCharPressed();
    }

    if (IsKeyPressed(KEY_BACKSPACE)) {
        size_t len = strlen(ui->fen_input);
        if (len > 0) {
            ui->fen_input[len - 1] = '\0';
        }
    }

    bool paste_mod = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL) ||
                     IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_RIGHT_SUPER);
    if (paste_mod && IsKeyPressed(KEY_V)) {
        const char *clip = GetClipboardText();
        if (clip != NULL) {
            size_t len = strlen(ui->fen_input);
            for (size_t i = 0; clip[i] != '\0' && len + 1 < sizeof(ui->fen_input); ++i) {
                unsigned char ch = (unsigned char)clip[i];
                if (ch >= 32 && ch <= 126) {
                    ui->fen_input[len++] = (char)ch;
                }
            }
            ui->fen_input[len] = '\0';
        }
    }
}

static void apply_loaded_fen(ChessUi *ui) {
    ui->game.config = make_match_config(ui);
    apply_player_kind_config(ui);
    chess_reset_clocks(&ui->game);
    ui->clock_started = false;
    ui->think_ms_accum[PIECE_WHITE] = 0;
    ui->think_ms_accum[PIECE_BLACK] = 0;
    refresh_eval_bars(ui);
    ui->stockfish_has_result = false;
    memset(&ui->stockfish_last_result, 0, sizeof(ui->stockfish_last_result));
    ui->ai_last_had_result = false;
    memset(&ui->last_ai_result, 0, sizeof(ui->last_ai_result));
    ui->selected_sq = -1;
    refresh_legal(ui);

    ui->result_modal_visible = (ui->game.result != GAME_RESULT_ONGOING);
    ui->prev_result = ui->game.result;
    bump_position_serial(ui);

    game_log_begin(&ui->log, &ui->game);
}

static void on_result_transition(ChessUi *ui) {
    if (ui->game.result != GAME_RESULT_ONGOING && ui->prev_result == GAME_RESULT_ONGOING) {
        bump_position_serial(ui);
        ui->result_modal_visible = true;
        game_log_set_result(&ui->log, ui->game.result);
        game_log_set_elapsed(&ui->log, ui->game.clock_ms[PIECE_WHITE], ui->game.clock_ms[PIECE_BLACK]);
        game_log_flush(&ui->log);
    }
    ui->prev_result = ui->game.result;
}

static void maybe_play_ai_move(ChessUi *ui) {
    if (ui->game.result != GAME_RESULT_ONGOING) {
        return;
    }

    int side = ui->game.side_to_move;
    if (!ui_side_is_ai(ui, side)) {
        invalidate_ai_request(ui);
        return;
    }

    if (ui->promotion_modal || ui->fen_modal || ui->settings_modal || ui->ai_modal || ui->ai_test_modal || ui->result_modal_visible) {
        return;
    }

    AiSearchConfig cfg = {
        .think_time_ms = k_ai_think_presets[ui->ai_think_index].ms,
        .max_depth = 16,
    };

    if (!ui->ai_worker_ready) {
        AiSearchResult sync_result;
        if (!chess_ai_pick_move(&ui->game, &cfg, &sync_result) || !sync_result.found_move) {
            return;
        }

        if (sync_result.elapsed_ms > 0) {
            chess_tick_clock(&ui->game, sync_result.elapsed_ms);
            ui->think_ms_accum[side] += sync_result.elapsed_ms;
            game_log_set_elapsed(&ui->log, ui->game.clock_ms[PIECE_WHITE], ui->game.clock_ms[PIECE_BLACK]);
        }
        if (ui->game.result != GAME_RESULT_ONGOING) {
            return;
        }

        ui->last_ai_result = sync_result;
        ui->ai_last_had_result = true;
        (void)commit_chosen_move(ui, sync_result.best_move, ui->think_ms_accum[side]);
        return;
    }

    if (ui->ai_request_active) {
        AiSearchResult live_result;
        if (ai_worker_get_progress(&ui->ai_worker, &live_result)) {
            ui->live_ai_result = live_result;
            ui->ai_live_had_result = live_result.found_move;
        }

        if (ui->ai_request_serial != ui->position_serial ||
            ui->ai_request_side != ui->game.side_to_move ||
            ui->ai_request_hash != ui->game.zobrist_hash ||
            ui->ai_request_ply != ui->game.ply) {
            invalidate_ai_request(ui);
        }
    }

    if (ui->ai_request_active) {
        AiSearchResult async_result;
        if (!ai_worker_poll(&ui->ai_worker, &async_result)) {
            return;
        }

        bool still_same_position = ui->ai_request_serial == ui->position_serial &&
                                   ui->ai_request_side == ui->game.side_to_move &&
                                   ui->ai_request_hash == ui->game.zobrist_hash &&
                                   ui->ai_request_ply == ui->game.ply &&
                                   ui_side_is_ai(ui, ui->game.side_to_move) &&
                                   ui->game.result == GAME_RESULT_ONGOING;

        invalidate_ai_request(ui);
        if (!still_same_position || !async_result.found_move) {
            return;
        }

        ui->last_ai_result = async_result;
        ui->ai_last_had_result = true;
        (void)commit_chosen_move(ui, async_result.best_move, ui->think_ms_accum[side]);
        return;
    }

    if (ai_worker_request(&ui->ai_worker, &ui->game, &cfg)) {
        ui->ai_request_active = true;
        ui->ai_request_serial = ui->position_serial;
        ui->ai_request_side = ui->game.side_to_move;
        ui->ai_request_hash = ui->game.zobrist_hash;
        ui->ai_request_ply = ui->game.ply;
        ui->ai_request_started_ms = ui_now_ms();
        ui->ai_live_had_result = false;
        memset(&ui->live_ai_result, 0, sizeof(ui->live_ai_result));
    }
}

static void handle_button_action(ChessUi *ui, UiButtonId id) {
    switch (id) {
        case BTN_NEW_GAME:
            start_new_game(ui);
            break;
        case BTN_UNDO:
            if (chess_undo_move(&ui->game)) {
                ui->selected_sq = -1;
                ui->promotion_modal = false;
                ui->pending_promo_count = 0;
                ui->think_ms_accum[PIECE_WHITE] = 0;
                ui->think_ms_accum[PIECE_BLACK] = 0;
                ui->clock_started = (ui->game.ply > 0);
                bump_position_serial(ui);
                refresh_legal(ui);
                ui->ai_last_had_result = false;

                game_log_truncate(&ui->log, ui->log.move_count - 1);
                game_log_set_result(&ui->log, ui->game.result);
                game_log_set_elapsed(&ui->log, ui->game.clock_ms[PIECE_WHITE], ui->game.clock_ms[PIECE_BLACK]);
                game_log_flush(&ui->log);
            }
            break;
        case BTN_RESIGN:
            if (ui->game.result == GAME_RESULT_ONGOING) {
                ui->game.result = (ui->game.side_to_move == PIECE_WHITE) ? GAME_RESULT_BLACK_WIN_RESIGN : GAME_RESULT_WHITE_WIN_RESIGN;
                bump_position_serial(ui);
                on_result_transition(ui);
            }
            break;
        case BTN_AI:
            ui->ai_modal = true;
            break;
        case BTN_SETTINGS:
            ui->settings_modal = true;
            break;
        default:
            break;
    }
}

static void handle_main_input(ChessUi *ui, const UiLayout *layout, Vector2 mouse) {
    bool left_pressed = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
    bool left_down = IsMouseButtonDown(MOUSE_LEFT_BUTTON);
    bool left_released = IsMouseButtonReleased(MOUSE_LEFT_BUTTON);

    if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
        ui->selected_sq = -1;
    }

    if (left_pressed) {
        for (int i = 0; i < BTN_COUNT; ++i) {
            const UiButton *button = &layout->buttons[i];
            if (button->enabled && point_in_rect(mouse, button->rect)) {
                handle_button_action(ui, button->id);
                return;
            }
        }

        if (ui->game.result == GAME_RESULT_ONGOING && ui_side_is_ai(ui, ui->game.side_to_move)) {
            return;
        }

        int sq = mouse_to_square(mouse, layout->board, ui->board_flipped);
        ui->press_sq = sq;
        ui->press_mouse = mouse;
        ui->drag_started = false;

        if (sq >= 0 && ui->game.result == GAME_RESULT_ONGOING) {
            int color = -1;
            int piece = chess_piece_on_square(&ui->game, sq, &color);
            if (piece != PIECE_NONE && color == ui->game.side_to_move) {
                ui->drag_from_sq = sq;
            } else {
                ui->drag_from_sq = -1;
            }
        } else {
            ui->drag_from_sq = -1;
        }
    }

    if (left_down && ui->drag_from_sq >= 0 && !ui->drag_started) {
        float dx = mouse.x - ui->press_mouse.x;
        float dy = mouse.y - ui->press_mouse.y;
        if ((dx * dx + dy * dy) >= 16.0f) {
            ui->drag_started = true;
            ui->selected_sq = ui->drag_from_sq;
        }
    }

    if (left_released) {
        int release_sq = mouse_to_square(mouse, layout->board, ui->board_flipped);

        if (ui->drag_from_sq >= 0 && ui->drag_started) {
            if (release_sq >= 0) {
                (void)try_commit_move(ui, ui->drag_from_sq, release_sq, CHESS_PROMO_NONE);
            }
            ui->press_sq = -1;
            ui->drag_from_sq = -1;
            ui->drag_started = false;
            return;
        }

        if (release_sq >= 0 && ui->game.result == GAME_RESULT_ONGOING) {
            if (ui->selected_sq >= 0 && ui->selected_sq != release_sq) {
                if (try_commit_move(ui, ui->selected_sq, release_sq, CHESS_PROMO_NONE)) {
                    ui->press_sq = -1;
                    ui->drag_from_sq = -1;
                    ui->drag_started = false;
                    return;
                }
            }

            int color = -1;
            int piece = chess_piece_on_square(&ui->game, release_sq, &color);
            if (piece != PIECE_NONE && color == ui->game.side_to_move) {
                if (ui->selected_sq == release_sq) {
                    ui->selected_sq = -1;
                } else {
                    ui->selected_sq = release_sq;
                }
            } else if (ui->selected_sq == release_sq || ui->press_sq == release_sq) {
                ui->selected_sq = -1;
            }
        }

        ui->press_sq = -1;
        ui->drag_from_sq = -1;
        ui->drag_started = false;
    }
}

static void handle_promotion_modal_input(ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    if (IsKeyPressed(KEY_ESCAPE)) {
        ui->promotion_modal = false;
        ui->pending_promo_count = 0;
        return;
    }

    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return;
    }

    Rectangle modal = {(float)screen_w * 0.5f - 250.0f, (float)screen_h * 0.5f - 120.0f, 500.0f, 240.0f};
    const int order[4] = {PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT};
    const float bw = 98.0f;
    const float gap = 14.0f;
    const float start_x = modal.x + (modal.width - (4.0f * bw + 3.0f * gap)) * 0.5f;

    for (int i = 0; i < 4; ++i) {
        Move promo_move;
        if (!find_pending_promo_move(ui, order[i], &promo_move)) {
            continue;
        }

        Rectangle br = {start_x + i * (bw + gap), modal.y + 94.0f, bw, 102.0f};
        if (point_in_rect(mouse, br)) {
            try_commit_move(ui, move_from(promo_move), move_to(promo_move), order[i]);
            break;
        }
    }
}

static void handle_fen_modal_input(ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    handle_fen_text_input(ui);

    if (IsKeyPressed(KEY_ESCAPE)) {
        ui->fen_modal = false;
        return;
    }

    if (IsKeyPressed(KEY_ENTER)) {
        if (chess_load_fen(&ui->game, ui->fen_input, ui->fen_error, sizeof(ui->fen_error))) {
            ui->fen_modal = false;
            apply_loaded_fen(ui);
        }
    }

    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return;
    }

    Rectangle modal = {(float)screen_w * 0.5f - 390.0f, (float)screen_h * 0.5f - 170.0f, 780.0f, 340.0f};
    Rectangle load_btn = {modal.x + modal.width - 220.0f, modal.y + modal.height - 56.0f, 90.0f, 38.0f};
    Rectangle cancel_btn = {modal.x + modal.width - 110.0f, modal.y + modal.height - 56.0f, 90.0f, 38.0f};

    if (point_in_rect(mouse, load_btn)) {
        if (chess_load_fen(&ui->game, ui->fen_input, ui->fen_error, sizeof(ui->fen_error))) {
            ui->fen_modal = false;
            apply_loaded_fen(ui);
        }
    } else if (point_in_rect(mouse, cancel_btn)) {
        ui->fen_modal = false;
    }
}

static void handle_settings_modal_input(ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    if (IsKeyPressed(KEY_ESCAPE)) {
        ui->settings_modal = false;
        return;
    }

    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return;
    }

    Rectangle modal;
    UiButton buttons[SETTINGS_ACT_COUNT];
    int count = build_settings_modal_widgets(ui, screen_w, screen_h, &modal, buttons);

    for (int i = 0; i < count; ++i) {
        if (!buttons[i].enabled || buttons[i].rect.width < 2.0f || buttons[i].rect.height < 2.0f) {
            continue;
        }
        if (!point_in_rect(mouse, buttons[i].rect)) {
            continue;
        }

        switch (i) {
            case SETTINGS_ACT_TIMER:
                ui->timer_setting_enabled = !ui->timer_setting_enabled;
                return;
            case SETTINGS_ACT_PRESET:
                ui->preset_index = (ui->preset_index + 1) % ARRAY_LEN(k_presets);
                return;
            case SETTINGS_ACT_EVAL_BAR:
                ui->eval_bar_enabled = !ui->eval_bar_enabled;
                if (ui->eval_bar_enabled) {
                    ui->eval_bar_target_cp = chess_ai_eval_fast_cp(&ui->game);
                    ui->eval_bar_sample_accum_ms = 0;
                }
                return;
            case SETTINGS_ACT_STOCKFISH_BAR:
                if (ui->stockfish_worker_ready && stockfish_eval_worker_is_available(&ui->stockfish_worker)) {
                    ui->stockfish_bar_enabled = !ui->stockfish_bar_enabled;
                    if (ui->stockfish_bar_enabled) {
                        ui->stockfish_bar_target_cp = chess_ai_eval_fast_cp(&ui->game);
                        ui->stockfish_bar_norm = eval_cp_to_norm(ui->stockfish_bar_target_cp);
                        ui->stockfish_bar_sample_accum_ms = 9999;
                    } else {
                        invalidate_stockfish_request(ui);
                    }
                }
                return;
            case SETTINGS_ACT_COPY_FEN: {
                char fen[256];
                chess_export_fen(&ui->game, fen, sizeof(fen));
                SetClipboardText(fen);
                return;
            }
            case SETTINGS_ACT_LOAD_FEN:
                ui->settings_modal = false;
                ui->fen_modal = true;
                ui->fen_error[0] = '\0';
                chess_export_fen(&ui->game, ui->fen_input, sizeof(ui->fen_input));
                return;
            case SETTINGS_ACT_OPEN_LOGS:
            {
                char url[768];
                snprintf(url, sizeof(url), "file://%s", game_log_directory());
                OpenURL(url);
                return;
            }
            case SETTINGS_ACT_AI_TEST_LAB:
                ui->settings_modal = false;
                ui->ai_test_modal = true;
                return;
            case SETTINGS_ACT_CLOSE:
                ui->settings_modal = false;
                return;
            default:
                return;
        }
    }
}

static void handle_ai_modal_input(ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    if (IsKeyPressed(KEY_ESCAPE)) {
        ui->ai_modal = false;
        return;
    }

    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return;
    }

    Rectangle modal;
    UiButton buttons[AI_ACT_COUNT];
    int count = build_ai_modal_widgets(ui, screen_w, screen_h, &modal, buttons);

    for (int i = 0; i < count; ++i) {
        if (!buttons[i].enabled || buttons[i].rect.width < 2.0f || buttons[i].rect.height < 2.0f) {
            continue;
        }
        if (!point_in_rect(mouse, buttons[i].rect)) {
            continue;
        }

        switch (i) {
            case AI_ACT_HVH:
                if (ui->ai_mode != AI_MODE_HUMAN_VS_HUMAN) {
                    ui->ai_mode = AI_MODE_HUMAN_VS_HUMAN;
                    start_new_game(ui);
                } else {
                    ui->ai_modal = false;
                }
                return;
            case AI_ACT_WHITE_VS_AI:
                if (ui->ai_mode != AI_MODE_WHITE_VS_AI) {
                    ui->ai_mode = AI_MODE_WHITE_VS_AI;
                    start_new_game(ui);
                } else {
                    ui->ai_modal = false;
                }
                return;
            case AI_ACT_BLACK_VS_AI:
                if (ui->ai_mode != AI_MODE_BLACK_VS_AI) {
                    ui->ai_mode = AI_MODE_BLACK_VS_AI;
                    start_new_game(ui);
                } else {
                    ui->ai_modal = false;
                }
                return;
            case AI_ACT_AI_VS_AI:
                if (ui->ai_mode != AI_MODE_AI_VS_AI) {
                    ui->ai_mode = AI_MODE_AI_VS_AI;
                    start_new_game(ui);
                } else {
                    ui->ai_modal = false;
                }
                return;
            case AI_ACT_BACKEND:
                if (!ui->nn_backend_available) {
                    return;
                }
                if (ui->ai_worker_ready && ai_worker_is_busy(&ui->ai_worker)) {
                    return;
                }
                ui->ai_backend = next_available_ai_backend(ui, ui->ai_backend);
                apply_ai_backend(ui);
                invalidate_ai_request(ui);
                refresh_eval_bars(ui);
                return;
            case AI_ACT_THINK_TIME:
                ui->ai_think_index = (ui->ai_think_index + 1) % ARRAY_LEN(k_ai_think_presets);
                invalidate_ai_request(ui);
                return;
            case AI_ACT_CLOSE:
                ui->ai_modal = false;
                return;
            default:
                return;
        }
    }
}

static void handle_ai_test_modal_input(ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    if (IsKeyPressed(KEY_ESCAPE)) {
        ui->ai_test_modal = false;
        return;
    }

    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return;
    }

    Rectangle modal;
    UiButton buttons[AI_TEST_ACT_COUNT];
    int count = build_ai_test_modal_widgets(ui, screen_w, screen_h, &modal, buttons);

    for (int i = 0; i < count; ++i) {
        if (!buttons[i].enabled || buttons[i].rect.width < 2.0f || buttons[i].rect.height < 2.0f) {
            continue;
        }
        if (!point_in_rect(mouse, buttons[i].rect)) {
            continue;
        }

        switch (i) {
            case AI_TEST_ACT_MATCHUP:
                if (!ui->ai_test_status.running) {
                    ui->ai_test_matchup_index = (ui->ai_test_matchup_index + 1) % AI_TEST_MATCHUP_COUNT;
                    ui->ai_test_status.estimated_elo_is_delta =
                        !ai_test_matchup_uses_stockfish(current_ai_test_matchup(ui)) || !ui->ai_test_stockfish_ladder;
                }
                return;
            case AI_TEST_ACT_POSITIONS:
                ui->ai_test_use_lichess_positions = !ui->ai_test_use_lichess_positions;
                return;
            case AI_TEST_ACT_GAMES:
                ui->ai_test_games_index = (ui->ai_test_games_index + 1) % ARRAY_LEN(k_ai_test_game_counts);
                return;
            case AI_TEST_ACT_STOCKFISH_SKILL:
                ui->ai_test_stockfish_level_index = (ui->ai_test_stockfish_level_index + 1) % ARRAY_LEN(k_ai_test_stockfish_levels);
                return;
            case AI_TEST_ACT_OPPONENT_MODE:
                if (!ui->ai_test_status.running) {
                    ui->ai_test_stockfish_ladder = !ui->ai_test_stockfish_ladder;
                    ui->ai_test_status.estimated_elo_is_delta = !ui->ai_test_stockfish_ladder;
                }
                return;
            case AI_TEST_ACT_OUR_THINK:
                ui->ai_test_our_time_index = (ui->ai_test_our_time_index + 1) % ARRAY_LEN(k_ai_test_time_options_ms);
                return;
            case AI_TEST_ACT_OPP_THINK:
                ui->ai_test_stockfish_time_index = (ui->ai_test_stockfish_time_index + 1) % ARRAY_LEN(k_ai_test_time_options_ms);
                return;
            case AI_TEST_ACT_START_STOP:
                if (!ui->ai_test_runner_ready) {
                    snprintf(ui->ai_test_status.message, sizeof(ui->ai_test_status.message), "AI test runner unavailable");
                    return;
                }
                if (ui->ai_test_status.running) {
                    ai_test_runner_stop(&ui->ai_test_runner);
                    return;
                } else {
                    AiTestMatchup matchup = current_ai_test_matchup(ui);
                    ChessAiBackend our_backend = ai_test_matchup_our_backend(matchup);
                    ChessAiBackend opp_backend = ai_test_matchup_opponent_backend(matchup);
                    bool uses_stockfish = ai_test_matchup_uses_stockfish(matchup);
                    AiTestConfig cfg;
                    memset(&cfg, 0, sizeof(cfg));
                    cfg.mode = uses_stockfish ? AI_TEST_MODE_VS_STOCKFISH : AI_TEST_MODE_INTERNAL;
                    cfg.our_backend = our_backend;
                    cfg.opponent_backend = opp_backend;
                    cfg.total_games = k_ai_test_game_counts[ui->ai_test_games_index];
                    cfg.our_think_ms = k_ai_test_time_options_ms[ui->ai_test_our_time_index];
                    cfg.stockfish_think_ms = k_ai_test_time_options_ms[ui->ai_test_stockfish_time_index];
                    cfg.stockfish_skill_level = k_ai_test_stockfish_levels[ui->ai_test_stockfish_level_index];
                    cfg.stockfish_ladder_enabled = uses_stockfish ? ui->ai_test_stockfish_ladder : false;
                    cfg.max_plies_per_game = 180;
                    cfg.use_lichess_positions = ui->ai_test_use_lichess_positions;
                    strncpy(cfg.positions_path, ai_test_positions_path(), sizeof(cfg.positions_path) - 1);
                    cfg.positions_path[sizeof(cfg.positions_path) - 1] = '\0';
                    if (ui->nn_model_path[0] != '\0') {
                        strncpy(cfg.nn_model_path, ui->nn_model_path, sizeof(cfg.nn_model_path) - 1);
                        cfg.nn_model_path[sizeof(cfg.nn_model_path) - 1] = '\0';
                    }

                    if (!ai_test_runner_start(&ui->ai_test_runner, &cfg)) {
                        AiTestStatus st;
                        if (ai_test_runner_get_status(&ui->ai_test_runner, &st)) {
                            ui->ai_test_status = st;
                        } else {
                            snprintf(ui->ai_test_status.message, sizeof(ui->ai_test_status.message), "Failed to start test");
                        }
                    }
                    return;
                }
            case AI_TEST_ACT_CLOSE:
                ui->ai_test_modal = false;
                return;
            default:
                return;
        }
    }
}

static void handle_result_modal_input(ChessUi *ui, Vector2 mouse, int screen_w, int screen_h) {
    if (!ui->result_modal_visible || ui->game.result == GAME_RESULT_ONGOING) {
        return;
    }

    if (IsKeyPressed(KEY_ESCAPE)) {
        ui->result_modal_visible = false;
        return;
    }

    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        return;
    }

    Rectangle modal = {(float)screen_w * 0.5f - 250.0f, (float)screen_h * 0.5f - 120.0f, 500.0f, 240.0f};
    Rectangle new_game_btn = {modal.x + 72.0f, modal.y + 154.0f, 150.0f, 44.0f};
    Rectangle close_btn = {modal.x + 280.0f, modal.y + 154.0f, 150.0f, 44.0f};

    if (point_in_rect(mouse, new_game_btn)) {
        start_new_game(ui);
    } else if (point_in_rect(mouse, close_btn)) {
        ui->result_modal_visible = false;
    }
}

int chess_run_ui(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(1280, 900, "Chess");
    SetExitKey(KEY_NULL);
    SetWindowMinSize(920, 680);

    ChessUi ui;
    memset(&ui, 0, sizeof(ui));
    ui.selected_sq = -1;
    ui.press_sq = -1;
    ui.drag_from_sq = -1;

    ui.timer_setting_enabled = false;
    ui.preset_index = 0;
    ui.clock_started = false;
    ui.ai_mode = AI_MODE_HUMAN_VS_HUMAN;
    ui.ai_think_index = 3;
    ui.ai_backend = CHESS_AI_BACKEND_CLASSIC;
    ui.ai_modal = false;
    ui.ai_test_modal = false;
    ui.ai_worker_ready = false;
    ui.stockfish_worker_ready = false;
    ui.ai_test_runner_ready = false;
    ui.position_serial = 1;
    ui.eval_bar_enabled = false;
    ui.eval_bar_norm = 0.5f;
    ui.eval_bar_target_cp = 0;
    ui.eval_bar_sample_accum_ms = 0;
    ui.stockfish_bar_enabled = false;
    ui.stockfish_bar_norm = 0.5f;
    ui.stockfish_bar_target_cp = 0;
    ui.stockfish_bar_sample_accum_ms = 0;
    ui.stockfish_has_result = false;
    ui.ai_test_use_lichess_positions = true;
    ui.ai_test_matchup_index = AI_TEST_MATCHUP_CLASSIC_VS_STOCKFISH;
    ui.ai_test_stockfish_ladder = true;
    ui.ai_test_games_index = 2;
    ui.ai_test_stockfish_level_index = 2;
    ui.ai_test_our_time_index = 2;
    ui.ai_test_stockfish_time_index = 2;
    memset(&ui.ai_test_status, 0, sizeof(ui.ai_test_status));
    ui.ai_test_status.mode = AI_TEST_MODE_VS_STOCKFISH;
    ui.ai_test_status.our_backend = CHESS_AI_BACKEND_CLASSIC;
    ui.ai_test_status.opponent_backend = CHESS_AI_BACKEND_CLASSIC;
    ui.ai_test_status.estimated_elo_is_delta = !ui.ai_test_stockfish_ladder;
    snprintf(ui.ai_test_status.message, sizeof(ui.ai_test_status.message), "Idle");
    ui.nn_backend_available = resolve_default_nn_model_path(ui.nn_model_path);
    if (ui.nn_backend_available) {
        ui.nn_backend_available = chess_ai_set_nn_model_path(ui.nn_model_path);
        if (!ui.nn_backend_available) {
            ui.nn_model_path[0] = '\0';
        }
    }
    invalidate_ai_request(&ui);
    invalidate_stockfish_request(&ui);

    if (!assets_load(&ui.assets)) {
        CloseWindow();
        return 1;
    }

    ui.ai_worker_ready = ai_worker_init(&ui.ai_worker);
    ui.stockfish_worker_ready = stockfish_eval_worker_init(&ui.stockfish_worker);
    ui.ai_test_runner_ready = ai_test_runner_init(&ui.ai_test_runner);
    apply_ai_backend(&ui);

    assets_set_window_icon(&ui.assets);
    start_new_game(&ui);

    while (!WindowShouldClose()) {
        int screen_w = GetScreenWidth();
        int screen_h = GetScreenHeight();
        Vector2 mouse = GetMousePosition();

        UiLayout layout;
        build_layout(screen_w, screen_h, &ui, &layout);

        int delta_ms = (int)roundf(GetFrameTime() * 1000.0f);
        if (ui.game.result == GAME_RESULT_ONGOING) {
            if (ui.clock_started && delta_ms > 0) {
                int side = ui.game.side_to_move;
                chess_tick_clock(&ui.game, delta_ms);
                ui.think_ms_accum[side] += delta_ms;
                game_log_set_elapsed(&ui.log, ui.game.clock_ms[PIECE_WHITE], ui.game.clock_ms[PIECE_BLACK]);
            }
        }
        update_eval_bar_state(&ui, delta_ms);
        poll_ai_test_status(&ui);

        on_result_transition(&ui);

        bool modal_open = ui.promotion_modal || ui.fen_modal || ui.settings_modal || ui.ai_modal || ui.ai_test_modal ||
                          (ui.result_modal_visible && ui.game.result != GAME_RESULT_ONGOING);
        if (!modal_open && IsKeyPressed(KEY_B)) {
            ui.board_flipped = !ui.board_flipped;
        }

        if (ui.promotion_modal) {
            handle_promotion_modal_input(&ui, mouse, screen_w, screen_h);
        } else if (ui.fen_modal) {
            handle_fen_modal_input(&ui, mouse, screen_w, screen_h);
        } else if (ui.settings_modal) {
            handle_settings_modal_input(&ui, mouse, screen_w, screen_h);
        } else if (ui.ai_modal) {
            handle_ai_modal_input(&ui, mouse, screen_w, screen_h);
        } else if (ui.ai_test_modal) {
            handle_ai_test_modal_input(&ui, mouse, screen_w, screen_h);
        } else if (ui.result_modal_visible && ui.game.result != GAME_RESULT_ONGOING) {
            handle_result_modal_input(&ui, mouse, screen_w, screen_h);
        } else {
            handle_main_input(&ui, &layout, mouse);
        }

        maybe_play_ai_move(&ui);
        on_result_transition(&ui);

        BeginDrawing();
        ClearBackground(COLOR_BG);

        draw_board_and_pieces(&ui, layout.board, mouse);
        draw_sidebar(&ui, &layout, mouse);

        if (ui.fen_modal) {
            draw_fen_modal(&ui, mouse, screen_w, screen_h);
        }
        if (ui.settings_modal) {
            draw_settings_modal(&ui, mouse, screen_w, screen_h);
        }
        if (ui.ai_modal) {
            draw_ai_modal(&ui, mouse, screen_w, screen_h);
        }
        if (ui.ai_test_modal) {
            draw_ai_test_modal(&ui, mouse, screen_w, screen_h);
        }
        if (ui.promotion_modal) {
            draw_promotion_modal(&ui, mouse, screen_w, screen_h);
        }
        if (ui.result_modal_visible && ui.game.result != GAME_RESULT_ONGOING) {
            draw_result_modal(&ui, mouse, screen_w, screen_h);
        }

        EndDrawing();
    }

    if (ui.ai_worker_ready) {
        ai_worker_shutdown(&ui.ai_worker);
    }
    if (ui.stockfish_worker_ready) {
        stockfish_eval_worker_shutdown(&ui.stockfish_worker);
    }
    if (ui.ai_test_runner_ready) {
        ai_test_runner_stop(&ui.ai_test_runner);
        ai_test_runner_shutdown(&ui.ai_test_runner);
    }
    assets_unload(&ui.assets);
    CloseWindow();
    return 0;
}
