#include "hce_internal.h"
#include "chess_hash.h"
#include "nn_eval.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HCE_TT_BITS 20
#define HCE_TT_SIZE (1u << HCE_TT_BITS)
#define HCE_TT_MASK (HCE_TT_SIZE - 1u)
#define HCE_NN_EVAL_CACHE_BITS 16
#define HCE_NN_EVAL_CACHE_SIZE (1u << HCE_NN_EVAL_CACHE_BITS)
#define HCE_NN_EVAL_CACHE_MASK (HCE_NN_EVAL_CACHE_SIZE - 1u)

typedef enum HceTtBound {
    HCE_TT_NONE = 0,
    HCE_TT_EXACT = 1,
    HCE_TT_LOWER = 2,
    HCE_TT_UPPER = 3,
} HceTtBound;

typedef struct HceTtEntry {
    uint64_t key;
    Move move;
    int16_t score;
    int8_t depth;
    uint8_t bound;
    uint8_t age;
} HceTtEntry;

typedef struct HceEvalCacheEntry {
    uint64_t key;
    int score;
    bool valid;
} HceEvalCacheEntry;

typedef struct HceSearchProfile {
    const char *name;
    int eval_scale_permille;
    int qsearch_delta_margin;
    int static_prune_margin_per_depth;
    int null_move_base_reduction;
    int lmr_base_reduction;
    int lmr_depth_bonus_threshold;
    int lmr_late_move_threshold;
    int lmr_good_history_threshold;
    int lmr_bad_history_threshold;
    int lmr_backend_adjust;
    int lmp_max_depth;
    int lmp_base_moves;
    int futility_max_depth;
    int futility_margin_per_depth;
    int see_prune_max_depth;
    int see_prune_margin_per_depth;
    int aspiration_base;
    int aspiration_depth_scale;
    int twofold_draw;
} HceSearchProfile;

typedef struct HceSearchContext {
    int64_t start_ms;
    int64_t deadline_ms;
    int64_t hard_deadline_ms;
    bool timed_out;
    uint64_t nodes;
    int max_depth;
    Move policy_root_moves[CHESS_MAX_MOVES];
    int policy_root_count;
    int policy_root_bonus;
    const HceSearchProfile *profile;
    Move killer[HCE_MAX_PLY][2];
    int history[PIECE_COLOR_COUNT][64][64];
    NnAccumulatorFrame nn_frames[HCE_MAX_PLY];
    HceEvalCacheEntry nn_eval_cache[HCE_NN_EVAL_CACHE_SIZE];
} HceSearchContext;

static const HceSearchProfile HCE_SEARCH_PROFILE_CLASSIC = {
    .name = "classic",
    .eval_scale_permille = 1000,
    .qsearch_delta_margin = 120,
    .static_prune_margin_per_depth = 90,
    .null_move_base_reduction = 2,
    .lmr_base_reduction = 1,
    .lmr_depth_bonus_threshold = 6,
    .lmr_late_move_threshold = 6,
    .lmr_good_history_threshold = 12000,
    .lmr_bad_history_threshold = -8000,
    .lmr_backend_adjust = 0,
    .lmp_max_depth = 0,
    .lmp_base_moves = 4,
    .futility_max_depth = 0,
    .futility_margin_per_depth = 100,
    .see_prune_max_depth = 0,
    .see_prune_margin_per_depth = 40,
    .aspiration_base = 24,
    .aspiration_depth_scale = 6,
    .twofold_draw = 0,
};

static const HceSearchProfile HCE_SEARCH_PROFILE_NN_DEFAULT = {
    .name = "nn",
    .eval_scale_permille = 800,
    .qsearch_delta_margin = 120,
    .static_prune_margin_per_depth = 80,
    .null_move_base_reduction = 1,
    .lmr_base_reduction = 1,
    .lmr_depth_bonus_threshold = 6,
    .lmr_late_move_threshold = 6,
    .lmr_good_history_threshold = 12000,
    .lmr_bad_history_threshold = -8000,
    .lmr_backend_adjust = 0,
    .lmp_max_depth = 2,
    .lmp_base_moves = 4,
    .futility_max_depth = 0,
    .futility_margin_per_depth = 100,
    .see_prune_max_depth = 0,
    .see_prune_margin_per_depth = 40,
    .aspiration_base = 40,
    .aspiration_depth_scale = 10,
    .twofold_draw = 1,
};

static HceSearchProfile g_hce_search_profile_nn = {
    .name = "nn",
    .eval_scale_permille = 800,
    .qsearch_delta_margin = 120,
    .static_prune_margin_per_depth = 80,
    .null_move_base_reduction = 1,
    .lmr_base_reduction = 1,
    .lmr_depth_bonus_threshold = 6,
    .lmr_late_move_threshold = 6,
    .lmr_good_history_threshold = 12000,
    .lmr_bad_history_threshold = -8000,
    .lmr_backend_adjust = 0,
    .lmp_max_depth = 2,
    .lmp_base_moves = 4,
    .futility_max_depth = 0,
    .futility_margin_per_depth = 100,
    .see_prune_max_depth = 0,
    .see_prune_margin_per_depth = 40,
    .aspiration_base = 40,
    .aspiration_depth_scale = 10,
    .twofold_draw = 1,
};

static HceTtEntry g_hce_tt[HCE_TT_SIZE];
static uint8_t g_hce_tt_generation = 0;
static atomic_flag g_hce_lock = ATOMIC_FLAG_INIT;
static FILE *g_nn_leaf_log_fp = NULL;
static char g_nn_leaf_log_path[512] = {0};
static int g_nn_leaf_log_limit = 0;
static int g_nn_leaf_log_count = 0;

bool hce_nn_leaf_log_set_path(const char *path) {
    if (g_nn_leaf_log_fp != NULL) {
        fclose(g_nn_leaf_log_fp);
        g_nn_leaf_log_fp = NULL;
    }
    g_nn_leaf_log_path[0] = '\0';
    g_nn_leaf_log_count = 0;

    if (path == NULL || path[0] == '\0' || strcmp(path, "off") == 0 || strcmp(path, "none") == 0) {
        return true;
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return false;
    }
    g_nn_leaf_log_fp = fp;
    snprintf(g_nn_leaf_log_path, sizeof(g_nn_leaf_log_path), "%s", path);
    return true;
}

const char *hce_nn_leaf_log_path(void) {
    return g_nn_leaf_log_path[0] != '\0' ? g_nn_leaf_log_path : NULL;
}

void hce_nn_leaf_log_set_limit(int limit) {
    g_nn_leaf_log_limit = limit > 0 ? limit : 0;
}

int hce_nn_leaf_log_limit(void) {
    return g_nn_leaf_log_limit;
}

int hce_nn_leaf_log_count(void) {
    return g_nn_leaf_log_count;
}

static bool hce_option_ieq(const char *a, const char *b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    while (*a != '\0' && *b != '\0') {
        char ca = *a;
        char cb = *b;
        if (ca >= 'A' && ca <= 'Z') {
            ca = (char)(ca - 'A' + 'a');
        }
        if (cb >= 'A' && cb <= 'Z') {
            cb = (char)(cb - 'A' + 'a');
        }
        if (ca != cb) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static bool hce_nn_search_option_ref(const char *name, int **out) {
    if (name == NULL || out == NULL) {
        return false;
    }
    if (hce_option_ieq(name, "NNEvalScale") || hce_option_ieq(name, "EvalScale")) {
        *out = &g_hce_search_profile_nn.eval_scale_permille;
        return true;
    }
    if (hce_option_ieq(name, "NNQDeltaMargin") || hce_option_ieq(name, "QDeltaMargin")) {
        *out = &g_hce_search_profile_nn.qsearch_delta_margin;
        return true;
    }
    if (hce_option_ieq(name, "NNStaticPruneMargin") || hce_option_ieq(name, "StaticPruneMargin")) {
        *out = &g_hce_search_profile_nn.static_prune_margin_per_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNNullMoveBaseReduction") || hce_option_ieq(name, "NullMoveBaseReduction")) {
        *out = &g_hce_search_profile_nn.null_move_base_reduction;
        return true;
    }
    if (hce_option_ieq(name, "NNLmrBackendAdjust") || hce_option_ieq(name, "LmrBackendAdjust")) {
        *out = &g_hce_search_profile_nn.lmr_backend_adjust;
        return true;
    }
    if (hce_option_ieq(name, "NNLmpMaxDepth") || hce_option_ieq(name, "LmpMaxDepth")) {
        *out = &g_hce_search_profile_nn.lmp_max_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNLmpBaseMoves") || hce_option_ieq(name, "LmpBaseMoves")) {
        *out = &g_hce_search_profile_nn.lmp_base_moves;
        return true;
    }
    if (hce_option_ieq(name, "NNFutilityMaxDepth") || hce_option_ieq(name, "FutilityMaxDepth")) {
        *out = &g_hce_search_profile_nn.futility_max_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNFutilityMargin") || hce_option_ieq(name, "FutilityMargin")) {
        *out = &g_hce_search_profile_nn.futility_margin_per_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNSeePruneMaxDepth") || hce_option_ieq(name, "SeePruneMaxDepth")) {
        *out = &g_hce_search_profile_nn.see_prune_max_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNSeePruneMargin") || hce_option_ieq(name, "SeePruneMargin")) {
        *out = &g_hce_search_profile_nn.see_prune_margin_per_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNAspirationBase") || hce_option_ieq(name, "AspirationBase")) {
        *out = &g_hce_search_profile_nn.aspiration_base;
        return true;
    }
    if (hce_option_ieq(name, "NNAspirationDepthScale") || hce_option_ieq(name, "AspirationDepthScale")) {
        *out = &g_hce_search_profile_nn.aspiration_depth_scale;
        return true;
    }
    if (hce_option_ieq(name, "NNTwofoldDraw") || hce_option_ieq(name, "TwofoldDraw")) {
        *out = &g_hce_search_profile_nn.twofold_draw;
        return true;
    }
    return false;
}

bool hce_nn_search_set_option(const char *name, int value) {
    int *field = NULL;
    if (!hce_nn_search_option_ref(name, &field)) {
        return false;
    }
    if (field == &g_hce_search_profile_nn.eval_scale_permille) {
        if (value < 100 || value > 3000) {
            return false;
        }
    } else if (field == &g_hce_search_profile_nn.lmr_backend_adjust) {
        if (value < -4 || value > 4) {
            return false;
        }
    } else if (field == &g_hce_search_profile_nn.twofold_draw) {
        if (value < 0 || value > 1) {
            return false;
        }
    } else if (value < 0 || value > 10000) {
        return false;
    }
    *field = value;
    return true;
}

int hce_nn_search_get_option(const char *name) {
    int *field = NULL;
    if (!hce_nn_search_option_ref(name, &field)) {
        return 0;
    }
    return *field;
}

void hce_nn_search_reset_options(void) {
    g_hce_search_profile_nn = HCE_SEARCH_PROFILE_NN_DEFAULT;
}

static int64_t now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

static void hce_lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_hce_lock, memory_order_acquire)) {
    }
}

static void hce_unlock(void) {
    atomic_flag_clear_explicit(&g_hce_lock, memory_order_release);
}

static bool search_is_insufficient_material(const GameState *s) {
    if (s == NULL) {
        return false;
    }
    return !chess_has_mating_material(s, PIECE_WHITE) &&
           !chess_has_mating_material(s, PIECE_BLACK);
}

static bool search_is_repetition_draw(const GameState *s, int required_occurrences) {
    if (s == NULL || s->hash_history_count <= 0) {
        return false;
    }
    if (required_occurrences < 2) {
        required_occurrences = 2;
    }

    uint64_t current = s->zobrist_hash;
    int begin = s->irreversible_ply;
    if (begin < 0) {
        begin = 0;
    }

    // Virtual index of the current position; history may or may not have it
    // appended depending on how the search state was built. Count the current
    // position exactly once, then require two previous same-side occurrences.
    int cur_index = s->hash_history_count;
    if (s->hash_history[cur_index - 1] == current) {
        cur_index -= 1;
    }

    int repetitions = 1;
    // Same-side positions sit an even number of plies back, and the side-to-
    // move zobrist key makes other parities unable to match anyway.
    for (int i = cur_index - 2; i >= begin; i -= 2) {
        if (s->hash_history[i] == current) {
            ++repetitions;
            if (repetitions >= required_occurrences) {
                return true;
            }
        }
    }
    return false;
}

int hce_score_search_draw_stm(const GameState *s) {
    if (s == NULL) {
        return INT_MIN;
    }
    if (search_is_insufficient_material(s)) {
        return 0;
    }
    if (s->halfmove_clock >= 100) {
        return 0;
    }
    if (search_is_repetition_draw(s, 3)) {
        return 0;
    }
    return INT_MIN;
}

static int score_terminal_stm(const GameState *s, int ply, const HceSearchContext *ctx) {
    if (s == NULL) {
        return 0;
    }
    switch (s->result) {
        case GAME_RESULT_WHITE_WIN:
        case GAME_RESULT_WHITE_WIN_RESIGN:
        case GAME_RESULT_WIN_TIMEOUT:
            return (s->side_to_move == PIECE_WHITE) ? (HCE_MATE - ply) : (-HCE_MATE + ply);
        case GAME_RESULT_BLACK_WIN:
        case GAME_RESULT_BLACK_WIN_RESIGN:
            return (s->side_to_move == PIECE_BLACK) ? (HCE_MATE - ply) : (-HCE_MATE + ply);
        case GAME_RESULT_DRAW_STALEMATE:
        case GAME_RESULT_DRAW_REPETITION:
        case GAME_RESULT_DRAW_50:
        case GAME_RESULT_DRAW_75:
        case GAME_RESULT_DRAW_INSUFFICIENT:
        case GAME_RESULT_DRAW_AGREED:
            return 0;
        case GAME_RESULT_ONGOING:
            if (ctx != NULL && ctx->profile != NULL && ctx->profile->twofold_draw != 0 &&
                search_is_repetition_draw(s, 2)) {
                return 0;
            }
            return hce_score_search_draw_stm(s);
        default:
            if (ctx != NULL && ctx->profile != NULL && ctx->profile->twofold_draw != 0 &&
                search_is_repetition_draw(s, 2)) {
                return 0;
            }
            return hce_score_search_draw_stm(s);
    }
}

static int tt_score_to_store(int score, int ply) {
    if (score > HCE_MATE_THRESHOLD) {
        return score + ply;
    }
    if (score < -HCE_MATE_THRESHOLD) {
        return score - ply;
    }
    return score;
}

static int tt_score_from_store(int score, int ply) {
    if (score > HCE_MATE_THRESHOLD) {
        return score - ply;
    }
    if (score < -HCE_MATE_THRESHOLD) {
        return score + ply;
    }
    return score;
}

static HceTtEntry *tt_entry(uint64_t key) {
    return &g_hce_tt[key & HCE_TT_MASK];
}

static bool tt_probe(uint64_t key, int depth, int ply, int alpha, int beta, Move *move_out, int *score_out) {
    HceTtEntry *entry = tt_entry(key);
    if (entry->bound == HCE_TT_NONE || entry->key != key) {
        return false;
    }
    if (move_out != NULL) {
        *move_out = entry->move;
    }
    if (entry->depth < depth || score_out == NULL) {
        return false;
    }

    int score = tt_score_from_store(entry->score, ply);
    if (entry->bound == HCE_TT_EXACT) {
        *score_out = score;
        return true;
    }
    if (entry->bound == HCE_TT_LOWER && score >= beta) {
        *score_out = score;
        return true;
    }
    if (entry->bound == HCE_TT_UPPER && score <= alpha) {
        *score_out = score;
        return true;
    }
    return false;
}

static void tt_store(uint64_t key, int depth, int ply, int score, HceTtBound bound, Move move) {
    HceTtEntry *entry = tt_entry(key);
    // Keep deeper data for the same position within the current search
    // generation; entries from older searches are always replaceable.
    if (entry->bound != HCE_TT_NONE &&
        entry->key == key &&
        entry->age == g_hce_tt_generation &&
        entry->depth > depth &&
        bound != HCE_TT_EXACT) {
        return;
    }
    entry->key = key;
    entry->move = move;
    entry->depth = (int8_t)depth;
    entry->bound = (uint8_t)bound;
    entry->age = g_hce_tt_generation;
    entry->score = (int16_t)tt_score_to_store(score, ply);
}

static bool should_stop(HceSearchContext *ctx) {
    if (ctx == NULL || ctx->deadline_ms <= 0) {
        return false;
    }
    if ((ctx->nodes & 2047ULL) != 0ULL) {
        return false;
    }
    if (now_ms() >= ctx->deadline_ms) {
        ctx->timed_out = true;
        return true;
    }
    return false;
}

static bool search_uses_nn_backend(void) {
    return chess_ai_get_backend() == CHESS_AI_BACKEND_NN && nn_eval_is_loaded();
}

static const HceSearchProfile *search_profile_for_current_backend(void) {
    return search_uses_nn_backend() ? &g_hce_search_profile_nn : &HCE_SEARCH_PROFILE_CLASSIC;
}

static const HceSearchProfile *search_profile(const HceSearchContext *ctx) {
    if (ctx != NULL && ctx->profile != NULL) {
        return ctx->profile;
    }
    return search_profile_for_current_backend();
}

static void search_log_nn_leaf(const GameState *s,
                               const HceSearchContext *ctx,
                               int ply,
                               int depth,
                               const char *phase,
                               int score) {
    if (g_nn_leaf_log_fp == NULL || s == NULL) {
        return;
    }
    if (g_nn_leaf_log_limit > 0 && g_nn_leaf_log_count >= g_nn_leaf_log_limit) {
        return;
    }

    char fen[256];
    chess_export_fen(s, fen, sizeof(fen));
    fprintf(g_nn_leaf_log_fp,
            "{\"fen\":\"%s\",\"score_cp\":%d,\"ply\":%d,\"depth\":%d,"
            "\"phase\":\"%s\",\"nodes\":%llu,\"side_to_move\":\"%c\"}\n",
            fen,
            score,
            ply,
            depth,
            phase != NULL ? phase : "search",
            (unsigned long long)(ctx != NULL ? ctx->nodes : 0ULL),
            s->side_to_move == PIECE_WHITE ? 'w' : 'b');
    g_nn_leaf_log_count += 1;
    if ((g_nn_leaf_log_count & 1023) == 0) {
        fflush(g_nn_leaf_log_fp);
    }
}

static bool search_nn_eval_cache_probe(HceSearchContext *ctx, uint64_t key, int *score_out) {
    if (ctx == NULL || score_out == NULL) {
        return false;
    }
    HceEvalCacheEntry *entry = &ctx->nn_eval_cache[key & HCE_NN_EVAL_CACHE_MASK];
    if (!entry->valid || entry->key != key) {
        return false;
    }
    *score_out = entry->score;
    return true;
}

static void search_nn_eval_cache_store(HceSearchContext *ctx, uint64_t key, int score) {
    if (ctx == NULL) {
        return;
    }
    HceEvalCacheEntry *entry = &ctx->nn_eval_cache[key & HCE_NN_EVAL_CACHE_MASK];
    entry->key = key;
    entry->score = score;
    entry->valid = true;
}

static int search_scale_eval_for_profile(const HceSearchContext *ctx, int score) {
    const HceSearchProfile *profile = search_profile(ctx);
    int scale = profile != NULL ? profile->eval_scale_permille : 1000;
    if (scale == 1000 || score > HCE_MATE_THRESHOLD || score < -HCE_MATE_THRESHOLD) {
        return score;
    }
    int64_t scaled = ((int64_t)score * (int64_t)scale) / 1000;
    if (scaled > HCE_MATE_THRESHOLD - 1) {
        return HCE_MATE_THRESHOLD - 1;
    }
    if (scaled < -HCE_MATE_THRESHOLD + 1) {
        return -HCE_MATE_THRESHOLD + 1;
    }
    return (int)scaled;
}

static bool search_ensure_nn_frame(const GameState *s, HceSearchContext *ctx, int ply) {
    if (s == NULL ||
        ctx == NULL ||
        chess_ai_get_backend() != CHESS_AI_BACKEND_NN ||
        !nn_eval_is_loaded() ||
        ply < 0 ||
        ply >= HCE_MAX_PLY) {
        return false;
    }

    NnAccumulatorFrame *frame = &ctx->nn_frames[ply];
    if (frame->valid && frame->key == s->zobrist_hash) {
        return true;
    }

    bool ok = false;
    if (ply > 0 && s->ply > 0) {
        const UndoRecord *undo = &s->undo_stack[s->ply - 1];
        const NnAccumulatorFrame *parent = &ctx->nn_frames[ply - 1];
        if (parent->valid && parent->key == undo->hash_prev) {
            ok = nn_eval_update_frame(s, undo, parent, frame);
        }
    }
    if (!ok) {
        ok = nn_eval_build_frame(s, frame);
    }
    if (!ok) {
        frame->valid = false;
    }
    return ok;
}

static void search_prepare_nn_child_frame(const GameState *child,
                                          HceSearchContext *ctx,
                                          int parent_ply,
                                          int child_ply) {
    if (child == NULL ||
        ctx == NULL ||
        chess_ai_get_backend() != CHESS_AI_BACKEND_NN ||
        !nn_eval_is_loaded() ||
        parent_ply < 0 ||
        parent_ply >= HCE_MAX_PLY ||
        child_ply < 0 ||
        child_ply >= HCE_MAX_PLY ||
        child->ply <= 0) {
        return;
    }

    NnAccumulatorFrame *child_frame = &ctx->nn_frames[child_ply];
    const NnAccumulatorFrame *parent = &ctx->nn_frames[parent_ply];
    const UndoRecord *undo = &child->undo_stack[child->ply - 1];
    if (!parent->valid || parent->key != undo->hash_prev) {
        child_frame->valid = false;
        return;
    }
    if (!nn_eval_update_frame(child, undo, parent, child_frame)) {
        child_frame->valid = false;
    }
}

static void search_prepare_nn_null_frame(const GameState *child,
                                         HceSearchContext *ctx,
                                         int parent_ply,
                                         int child_ply) {
    if (child == NULL ||
        ctx == NULL ||
        chess_ai_get_backend() != CHESS_AI_BACKEND_NN ||
        !nn_eval_is_loaded() ||
        parent_ply < 0 ||
        parent_ply >= HCE_MAX_PLY ||
        child_ply < 0 ||
        child_ply >= HCE_MAX_PLY) {
        return;
    }

    const NnAccumulatorFrame *parent = &ctx->nn_frames[parent_ply];
    NnAccumulatorFrame *child_frame = &ctx->nn_frames[child_ply];
    if (!parent->valid) {
        child_frame->valid = false;
        return;
    }
    if (!nn_eval_copy_frame(child, parent, child_frame)) {
        child_frame->valid = false;
    }
}

static int search_eval_cp_stm(const GameState *s,
                              HceSearchContext *ctx,
                              int ply,
                              int depth,
                              const char *phase) {
    if (s == NULL) {
        return 0;
    }
    if (chess_ai_get_backend() != CHESS_AI_BACKEND_NN || !nn_eval_is_loaded()) {
        return engine_eval_cp_stm(s);
    }
    if (ply < 0 || ply >= HCE_MAX_PLY) {
        int score = search_scale_eval_for_profile(ctx, nn_eval_cp_stm(s));
        search_log_nn_leaf(s, ctx, ply, depth, phase, score);
        return score;
    }

    NnAccumulatorFrame *frame = &ctx->nn_frames[ply];
    if (!search_ensure_nn_frame(s, ctx, ply)) {
        int score = search_scale_eval_for_profile(ctx, nn_eval_cp_stm(s));
        search_log_nn_leaf(s, ctx, ply, depth, phase, score);
        return score;
    }

    if (frame->valid && frame->key == s->zobrist_hash) {
        int cached_score = 0;
        if (search_nn_eval_cache_probe(ctx, s->zobrist_hash, &cached_score)) {
            search_log_nn_leaf(s, ctx, ply, depth, phase, cached_score);
            return cached_score;
        }
        int score = search_scale_eval_for_profile(ctx, nn_eval_cp_stm_from_frame(s, frame));
        search_nn_eval_cache_store(ctx, s->zobrist_hash, score);
        search_log_nn_leaf(s, ctx, ply, depth, phase, score);
        return score;
    }
    return search_scale_eval_for_profile(ctx, nn_eval_cp_stm(s));
}

static int captured_piece_for_move(const GameState *s, Move m) {
    if (!move_has_flag(m, MOVE_FLAG_CAPTURE)) {
        return PIECE_NONE;
    }
    int target_sq = move_to(m);
    if (move_has_flag(m, MOVE_FLAG_EN_PASSANT)) {
        target_sq = (s->side_to_move == PIECE_WHITE) ? (target_sq - 8) : (target_sq + 8);
    }
    if (target_sq < 0 || target_sq >= 64) {
        return PIECE_NONE;
    }
    return s->sq_piece[target_sq];
}

static int search_move_extension(const GameState *s_after_move, Move m, int depth) {
    if (depth <= 2) {
        return 0;
    }
    if (chess_in_check(s_after_move, s_after_move->side_to_move)) {
        return 1;
    }
    if (move_has_flag(m, MOVE_FLAG_PROMOTION)) {
        return 1;
    }
    return 0;
}

static int qsearch_move_gain_cp(const GameState *s, Move m) {
    int gain = 0;
    int captured = captured_piece_for_move(s, m);
    if (captured != PIECE_NONE) {
        gain += hce_piece_value[captured];
    }
    if (move_has_flag(m, MOVE_FLAG_PROMOTION)) {
        int promo = move_promo(m);
        if (promo != PIECE_NONE && promo != CHESS_PROMO_NONE) {
            gain += hce_piece_value[promo] - hce_piece_value[PIECE_PAWN];
        }
    }
    return gain;
}

static uint64_t attackers_to_square_local(const uint64_t bb[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT],
                                          uint64_t occ_all,
                                          int sq,
                                          int side) {
    uint64_t attackers = 0;
    attackers |= bb[side][PIECE_PAWN] & hce_pawn_attacks(side ^ 1, sq);
    attackers |= bb[side][PIECE_KNIGHT] & hce_knight_attacks(sq);
    attackers |= bb[side][PIECE_KING] & hce_king_attacks(sq);

    uint64_t bishop_like = bb[side][PIECE_BISHOP] | bb[side][PIECE_QUEEN];
    uint64_t rook_like = bb[side][PIECE_ROOK] | bb[side][PIECE_QUEEN];
    attackers |= bishop_like & hce_bishop_attacks(sq, occ_all);
    attackers |= rook_like & hce_rook_attacks(sq, occ_all);
    return attackers;
}

static bool select_least_valuable_attacker(const uint64_t bb[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT],
                                           uint64_t attackers,
                                           int side,
                                           int *piece_out,
                                           int *sq_out) {
    static const int k_piece_order[] = {
        PIECE_PAWN,
        PIECE_KNIGHT,
        PIECE_BISHOP,
        PIECE_ROOK,
        PIECE_QUEEN,
        PIECE_KING,
    };

    if (piece_out == NULL || sq_out == NULL) {
        return false;
    }

    for (size_t i = 0; i < sizeof(k_piece_order) / sizeof(k_piece_order[0]); ++i) {
        int piece = k_piece_order[i];
        uint64_t set = attackers & bb[side][piece];
        if (set == 0) {
            continue;
        }
        *piece_out = piece;
        *sq_out = chess_pop_lsb(&set);
        return true;
    }
    return false;
}

// Lightweight SEE for qsearch pruning. Promotions and king moves are handled
// outside this path to keep the pruning conservative.
static int static_exchange_eval(const GameState *s, Move m) {
    if (s == NULL || !move_has_flag(m, MOVE_FLAG_CAPTURE)) {
        return 0;
    }
    if (move_has_flag(m, MOVE_FLAG_PROMOTION) || move_has_flag(m, MOVE_FLAG_CASTLE)) {
        return qsearch_move_gain_cp(s, m);
    }

    int side = s->side_to_move;
    int opp = side ^ 1;
    int from = move_from(m);
    int to = move_to(m);
    int moving_piece = move_piece(m);
    int captured_piece = captured_piece_for_move(s, m);
    if (from < 0 || from >= 64 || to < 0 || to >= 64 || captured_piece == PIECE_NONE) {
        return 0;
    }

    uint64_t bb[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT];
    memcpy(bb, s->bb, sizeof(bb));
    uint64_t occ[PIECE_COLOR_COUNT] = {s->occ[PIECE_WHITE], s->occ[PIECE_BLACK]};
    uint64_t occ_all = s->occ_all;

    int captured_sq = to;
    if (move_has_flag(m, MOVE_FLAG_EN_PASSANT)) {
        captured_sq = (side == PIECE_WHITE) ? (to - 8) : (to + 8);
    }
    if (captured_sq < 0 || captured_sq >= 64) {
        return 0;
    }

    uint64_t from_mask = 1ULL << from;
    uint64_t to_mask = 1ULL << to;
    uint64_t captured_mask = 1ULL << captured_sq;

    bb[side][moving_piece] &= ~from_mask;
    occ[side] &= ~from_mask;
    occ_all &= ~from_mask;

    bb[opp][captured_piece] &= ~captured_mask;
    occ[opp] &= ~captured_mask;
    occ_all &= ~captured_mask;

    bb[side][moving_piece] |= to_mask;
    occ[side] |= to_mask;
    occ_all |= to_mask;

    int gain[32];
    int depth = 0;
    gain[0] = hce_piece_value[captured_piece];

    int target_side = side;
    int target_piece = moving_piece;
    int stm = opp;

    for (;;) {
        uint64_t attackers = attackers_to_square_local(bb, occ_all, to, stm);
        int attacker_piece = PIECE_NONE;
        int attacker_sq = CHESS_NO_SQUARE;
        if (!select_least_valuable_attacker(bb, attackers, stm, &attacker_piece, &attacker_sq)) {
            break;
        }

        uint64_t attacker_mask = 1ULL << attacker_sq;
        if (attacker_piece == PIECE_KING) {
            uint64_t occ_without_from = occ_all & ~attacker_mask;
            if (attackers_to_square_local(bb, occ_without_from, to, stm ^ 1) != 0) {
                break;
            }
        }

        bb[target_side][target_piece] &= ~to_mask;
        bb[stm][attacker_piece] &= ~attacker_mask;
        occ[stm] &= ~attacker_mask;
        occ_all &= ~attacker_mask;

        bb[stm][attacker_piece] |= to_mask;
        occ[stm] |= to_mask;
        occ_all |= to_mask;

        int next_depth = depth + 1;
        gain[next_depth] = hce_piece_value[target_piece] - gain[depth];
        depth = next_depth;
        target_side = stm;
        target_piece = attacker_piece;
        stm ^= 1;

        if (depth >= (int)(sizeof(gain) / sizeof(gain[0])) - 1) {
            break;
        }
    }

    while (depth > 0) {
        gain[depth - 1] = -((gain[depth - 1] > -gain[depth]) ? gain[depth - 1] : -gain[depth]);
        depth -= 1;
    }

    return gain[0];
}

static bool has_non_pawn_material(const GameState *s, int side) {
    return (s->bb[side][PIECE_QUEEN] |
            s->bb[side][PIECE_ROOK] |
            s->bb[side][PIECE_BISHOP] |
            s->bb[side][PIECE_KNIGHT]) != 0;
}

typedef struct NullMoveUndo {
    uint64_t hash;
    int ep_square;
    int halfmove_clock;
    int fullmove_number;
    Move last_move;
    bool has_last_move;
} NullMoveUndo;

static void make_null_move(GameState *s, NullMoveUndo *u) {
    u->hash = s->zobrist_hash;
    u->ep_square = s->ep_square;
    u->halfmove_clock = s->halfmove_clock;
    u->fullmove_number = s->fullmove_number;
    u->last_move = s->last_move;
    u->has_last_move = s->has_last_move;

    s->zobrist_hash ^= chess_hash_side_key() ^ chess_hash_ep_key(s->ep_square);
    if (s->side_to_move == PIECE_BLACK) {
        s->fullmove_number += 1;
    }
    s->side_to_move ^= 1;
    s->ep_square = CHESS_NO_SQUARE;
    s->halfmove_clock += 1;
    s->has_last_move = false;
    s->last_move = 0;
}

static void undo_null_move(GameState *s, const NullMoveUndo *u) {
    s->side_to_move ^= 1;
    s->zobrist_hash = u->hash;
    s->ep_square = u->ep_square;
    s->halfmove_clock = u->halfmove_clock;
    s->fullmove_number = u->fullmove_number;
    s->last_move = u->last_move;
    s->has_last_move = u->has_last_move;
}

static bool is_quiet_move(Move m) {
    return !move_has_flag(m, MOVE_FLAG_CAPTURE) &&
           !move_has_flag(m, MOVE_FLAG_PROMOTION);
}

static bool is_recapture_move(const GameState *s, Move m) {
    if (s == NULL || !s->has_last_move || !move_has_flag(m, MOVE_FLAG_CAPTURE)) {
        return false;
    }
    return move_to(m) == move_to(s->last_move);
}

static int move_score(const GameState *s, Move m, Move tt_move, HceSearchContext *ctx, int ply) {
    if (m == tt_move) {
        return 200000000;
    }

    int score = 0;
    if (move_has_flag(m, MOVE_FLAG_CAPTURE)) {
        int victim = captured_piece_for_move(s, m);
        int attacker = move_piece(m);
        if (victim == PIECE_NONE) {
            victim = PIECE_PAWN;
        }
        score += 1000000 + hce_piece_value[victim] * 16 - hce_piece_value[attacker];
    } else {
        if (ply >= 0 && ply < HCE_MAX_PLY) {
            if (ctx->killer[ply][0] == m) {
                score += 900000;
            } else if (ctx->killer[ply][1] == m) {
                score += 850000;
            }
        }
        score += ctx->history[s->side_to_move][move_from(m)][move_to(m)];
    }
    if (move_has_flag(m, MOVE_FLAG_PROMOTION)) {
        score += 700000 + hce_piece_value[move_promo(m)] * 8;
    }
    if (move_has_flag(m, MOVE_FLAG_CASTLE)) {
        score += 2000;
    }
    return score;
}

static void score_moves(const GameState *s,
                         int scores[CHESS_MAX_MOVES],
                         const Move moves[CHESS_MAX_MOVES],
                         int n,
                         Move tt_move,
                         HceSearchContext *ctx,
                         int ply) {
    for (int i = 0; i < n; ++i) {
        scores[i] = move_score(s, moves[i], tt_move, ctx, ply);
    }
}

static void apply_root_policy_scores(int scores[CHESS_MAX_MOVES],
                                     const Move moves[CHESS_MAX_MOVES],
                                     int n,
                                     const HceSearchContext *ctx) {
    if (ctx == NULL || ctx->policy_root_count <= 0 || ctx->policy_root_bonus <= 0) {
        return;
    }
    int count = ctx->policy_root_count;
    if (count > CHESS_MAX_MOVES) {
        count = CHESS_MAX_MOVES;
    }
    for (int rank = 0; rank < count; ++rank) {
        Move hinted = ctx->policy_root_moves[rank];
        int bonus = ctx->policy_root_bonus * (count - rank);
        if (bonus <= 0) {
            bonus = 1;
        }
        for (int i = 0; i < n; ++i) {
            if (moves[i] == hinted) {
                scores[i] += bonus;
                break;
            }
        }
    }
}

static Move pick_next_move(Move moves[CHESS_MAX_MOVES],
                           int scores[CHESS_MAX_MOVES],
                           int start,
                           int n) {
    int best = start;
    for (int i = start + 1; i < n; ++i) {
        if (scores[i] > scores[best]) {
            best = i;
        }
    }
    if (best != start) {
        int score_tmp = scores[start];
        scores[start] = scores[best];
        scores[best] = score_tmp;
        Move move_tmp = moves[start];
        moves[start] = moves[best];
        moves[best] = move_tmp;
    }
    return moves[start];
}

static void update_killer(HceSearchContext *ctx, int ply, Move move) {
    if (ply < 0 || ply >= HCE_MAX_PLY || move_has_flag(move, MOVE_FLAG_CAPTURE)) {
        return;
    }
    if (ctx->killer[ply][0] == move) {
        return;
    }
    ctx->killer[ply][1] = ctx->killer[ply][0];
    ctx->killer[ply][0] = move;
}

static int history_bonus(int depth) {
    if (depth < 1) {
        depth = 1;
    }
    return depth * depth;
}

static void history_update_delta(HceSearchContext *ctx, int side, Move move, int delta) {
    if (ctx == NULL || side < 0 || side >= PIECE_COLOR_COUNT || move_has_flag(move, MOVE_FLAG_CAPTURE)) {
        return;
    }
    int *hist = &ctx->history[side][move_from(move)][move_to(move)];
    *hist += delta;
    if (*hist > 240000) {
        *hist = 240000;
    } else if (*hist < -240000) {
        *hist = -240000;
    }
}

static void update_history(HceSearchContext *ctx, int side, Move move, int depth) {
    history_update_delta(ctx, side, move, history_bonus(depth));
}

static void penalize_quiet_history(HceSearchContext *ctx,
                                   int side,
                                   const Move quiets[CHESS_MAX_MOVES],
                                   int quiet_count,
                                   int depth) {
    int malus = history_bonus(depth) / 2;
    if (malus < 1) {
        malus = 1;
    }
    for (int i = 0; i < quiet_count; ++i) {
        history_update_delta(ctx, side, quiets[i], -malus);
    }
}


static int quiescence(GameState *s, int alpha, int beta, int ply, HceSearchContext *ctx) {
    if (should_stop(ctx)) {
        return search_eval_cp_stm(s, ctx, ply, 0, "qsearch_timeout");
    }

    int term = score_terminal_stm(s, ply, ctx);
    if (term != INT_MIN) {
        return term;
    }

    bool in_check = chess_in_check(s, s->side_to_move);
    int stand_pat = search_eval_cp_stm(s, ctx, ply, 0, "qsearch");
    if (!in_check) {
        if (stand_pat >= beta) {
            return beta;
        }
        if (stand_pat > alpha) {
            alpha = stand_pat;
        }
    }

    Move tactical_moves[CHESS_MAX_MOVES];
    int tactical_n;
    if (in_check) {
        tactical_n = chess_generate_legal_moves_mut(s, tactical_moves);
        if (tactical_n <= 0) {
            return -HCE_MATE + ply;
        }
    } else {
        tactical_n = chess_generate_tactical_moves_mut(s, tactical_moves);
        if (tactical_n <= 0) {
            return alpha;
        }
    }
    int tactical_scores[CHESS_MAX_MOVES];
    score_moves(s, tactical_scores, tactical_moves, tactical_n, 0, ctx, ply);

    for (int i = 0; i < tactical_n; ++i) {
        Move m = pick_next_move(tactical_moves, tactical_scores, i, tactical_n);
        if (!in_check &&
            alpha > -HCE_MATE_THRESHOLD &&
            beta < HCE_MATE_THRESHOLD) {
            int gain = qsearch_move_gain_cp(s, m);
            int delta_margin = search_profile(ctx)->qsearch_delta_margin;
            if (!move_has_flag(m, MOVE_FLAG_PROMOTION) &&
                stand_pat + gain + delta_margin <= alpha) {
                continue;
            }
            if (move_has_flag(m, MOVE_FLAG_CAPTURE) &&
                !move_has_flag(m, MOVE_FLAG_PROMOTION) &&
                !move_has_flag(m, MOVE_FLAG_EN_PASSANT) &&
                move_piece(m) != PIECE_KING &&
                static_exchange_eval(s, m) < 0) {
                continue;
            }
        }
        if (!chess_make_move_trusted(s, m)) {
            continue;
        }
        ctx->nodes += 1;
        search_prepare_nn_child_frame(s, ctx, ply, ply + 1);
        int score = -quiescence(s, -beta, -alpha, ply + 1, ctx);
        chess_undo_move(s);
        if (ctx->timed_out) {
            return alpha;
        }
        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

static int negamax(GameState *s,
                   int depth,
                   int alpha,
                   int beta,
                   int ply,
                   HceSearchContext *ctx,
                   Move *best_move_out) {
    if (should_stop(ctx)) {
        return search_eval_cp_stm(s, ctx, ply, depth, "search_timeout");
    }

    int term = score_terminal_stm(s, ply, ctx);
    if (term != INT_MIN) {
        return term;
    }
    if (ply >= HCE_MAX_PLY - 1) {
        return search_eval_cp_stm(s, ctx, ply, depth, "max_ply");
    }

    int mate_alpha = -HCE_MATE + ply;
    int mate_beta = HCE_MATE - ply - 1;
    if (alpha < mate_alpha) {
        alpha = mate_alpha;
    }
    if (beta > mate_beta) {
        beta = mate_beta;
    }
    if (alpha >= beta) {
        return alpha;
    }

    if (depth <= 0) {
        return quiescence(s, alpha, beta, ply, ctx);
    }

    Move tt_move = 0;
    int tt_score = 0;
    if (tt_probe(s->zobrist_hash, depth, ply, alpha, beta, &tt_move, &tt_score)) {
        return tt_score;
    }

    bool in_check = chess_in_check(s, s->side_to_move);
    const HceSearchProfile *profile = search_profile(ctx);
    bool static_eval_valid = false;
    int static_eval = 0;
    if (!in_check &&
        (depth <= 3 || (profile->futility_max_depth > 0 && depth <= profile->futility_max_depth))) {
        static_eval = search_eval_cp_stm(s, ctx, ply, depth, "static_eval");
        static_eval_valid = true;
    }
    if (!in_check && depth <= 3 && beta < HCE_MATE_THRESHOLD) {
        int margin = profile->static_prune_margin_per_depth * depth;
        if (static_eval >= beta + margin) {
            return beta;
        }
    }

    if (ply > 0 &&
        depth >= 3 &&
        !in_check &&
        beta < HCE_MATE_THRESHOLD &&
        has_non_pawn_material(s, s->side_to_move)) {
        int reduction = search_profile(ctx)->null_move_base_reduction + depth / 4;
        if (reduction > depth - 1) {
            reduction = depth - 1;
        }
        if (reduction > 0) {
            ctx->nodes += 1;
            NullMoveUndo null_undo;
            make_null_move(s, &null_undo);
            search_prepare_nn_null_frame(s, ctx, ply, ply + 1);
            int score = -negamax(s,
                                 depth - 1 - reduction,
                                 -beta,
                                 -beta + 1,
                                 ply + 1,
                                 ctx,
                                 NULL);
            undo_null_move(s, &null_undo);
            if (ctx->timed_out) {
                return alpha;
            }
            if (score >= beta) {
                return beta;
            }
        }
    }

    Move moves[CHESS_MAX_MOVES];
    int n = chess_generate_legal_moves_mut(s, moves);
    if (n <= 0) {
        if (in_check) {
            return -HCE_MATE + ply;
        }
        return 0;
    }
    int move_scores[CHESS_MAX_MOVES];
    score_moves(s, move_scores, moves, n, tt_move, ctx, ply);

    Move best_move = moves[0];
    int best_score = -HCE_INF;
    int alpha_orig = alpha;
    int side = s->side_to_move;
    int searched = 0;
    Move failed_quiets[CHESS_MAX_MOVES];
    int failed_quiet_count = 0;

    for (int i = 0; i < n; ++i) {
        Move m = pick_next_move(moves, move_scores, i, n);
        bool quiet = is_quiet_move(m);
        bool recapture = is_recapture_move(s, m);
        if (quiet && !in_check && profile->lmp_max_depth > 0 &&
            depth <= profile->lmp_max_depth &&
            searched >= profile->lmp_base_moves + depth * 3) {
            continue;
        }
        if (!quiet && searched > 0 && !in_check &&
            move_has_flag(m, MOVE_FLAG_CAPTURE) &&
            !move_has_flag(m, MOVE_FLAG_PROMOTION) &&
            profile->see_prune_max_depth > 0 && depth <= profile->see_prune_max_depth &&
            static_exchange_eval(s, m) < -profile->see_prune_margin_per_depth * depth) {
            continue;
        }
        if (!chess_make_move_trusted(s, m)) {
            continue;
        }
        if (quiet && !recapture && searched > 0 && static_eval_valid &&
            profile->futility_max_depth > 0 && depth <= profile->futility_max_depth &&
            alpha < HCE_MATE_THRESHOLD &&
            static_eval + profile->futility_margin_per_depth * depth <= alpha &&
            !chess_in_check(s, s->side_to_move)) {
            chess_undo_move(s);
            continue;
        }
        ctx->nodes += 1;
        search_prepare_nn_child_frame(s, ctx, ply, ply + 1);

        int extension = search_move_extension(s, m, depth);

        int score;
        int next_depth = depth - 1 + extension;
        if (searched == 0) {
            score = -negamax(s, next_depth, -beta, -alpha, ply + 1, ctx, NULL);
        } else {
            int reduction = 0;
            if (!in_check &&
                quiet &&
                depth >= 3 &&
                searched >= 2) {
                reduction = profile->lmr_base_reduction;
                if (depth >= profile->lmr_depth_bonus_threshold) {
                    reduction += 1;
                }
                if (searched >= profile->lmr_late_move_threshold) {
                    reduction += 1;
                }
                int hist = ctx->history[side][move_from(m)][move_to(m)];
                if (hist > profile->lmr_good_history_threshold) {
                    reduction -= 1;
                } else if (hist < profile->lmr_bad_history_threshold) {
                    reduction += 1;
                }
                if (recapture) {
                    reduction -= 1;
                }
                reduction += profile->lmr_backend_adjust;
                if (reduction < 0) {
                    reduction = 0;
                }
                if (reduction > next_depth - 1) {
                    reduction = next_depth - 1;
                }
            }
            if (reduction > 0) {
                score = -negamax(s, next_depth - reduction, -alpha - 1, -alpha, ply + 1, ctx, NULL);
                if (score > alpha) {
                    score = -negamax(s, next_depth, -alpha - 1, -alpha, ply + 1, ctx, NULL);
                }
            } else {
                score = -negamax(s, next_depth, -alpha - 1, -alpha, ply + 1, ctx, NULL);
            }
            if (score > alpha && score < beta) {
                score = -negamax(s, next_depth, -beta, -alpha, ply + 1, ctx, NULL);
            }
        }
        chess_undo_move(s);
        if (ctx->timed_out) {
            break;
        }

        searched += 1;
        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
            if (alpha >= beta) {
                update_killer(ctx, ply, m);
                update_history(ctx, side, m, depth);
                if (has_non_pawn_material(s, side)) {
                    penalize_quiet_history(ctx, side, failed_quiets, failed_quiet_count, depth);
                }
                tt_store(s->zobrist_hash, depth, ply, beta, HCE_TT_LOWER, m);
                if (best_move_out != NULL) {
                    *best_move_out = m;
                }
                return beta;
            }
        }
        if (quiet && failed_quiet_count < CHESS_MAX_MOVES) {
            failed_quiets[failed_quiet_count++] = m;
        }
    }

    if (best_score == -HCE_INF) {
        best_score = in_check ? 0 : search_eval_cp_stm(s, ctx, ply, depth, "search_fallback");
    }

    HceTtBound bound = HCE_TT_EXACT;
    if (best_score <= alpha_orig) {
        bound = HCE_TT_UPPER;
    } else if (best_score >= beta) {
        bound = HCE_TT_LOWER;
    }
    if (!ctx->timed_out) {
        tt_store(s->zobrist_hash, depth, ply, best_score, bound, best_move);
    }
    if (best_move_out != NULL) {
        *best_move_out = best_move;
    }
    return best_score;
}

static int search_root(GameState *root,
                       int depth,
                       int alpha,
                       int beta,
                       HceSearchContext *ctx,
                       Move *best_move_out) {
    Move tt_move = 0;
    int tt_score = 0;
    Move moves[CHESS_MAX_MOVES];
    int n = chess_generate_legal_moves_mut(root, moves);
    int alpha_orig = alpha;
    int best_score = -HCE_INF;
    Move best_move = (n > 0) ? moves[0] : 0;
    int side = root->side_to_move;
    int searched = 0;

    if (n <= 0) {
        if (best_move_out != NULL) {
            *best_move_out = 0;
        }
        return score_terminal_stm(root, 0, ctx);
    }

    (void)tt_probe(root->zobrist_hash, depth, 0, alpha, beta, &tt_move, &tt_score);
    search_ensure_nn_frame(root, ctx, 0);
    int root_scores[CHESS_MAX_MOVES];
    score_moves(root, root_scores, moves, n, tt_move, ctx, 0);
    apply_root_policy_scores(root_scores, moves, n, ctx);

    for (int i = 0; i < n; ++i) {
        Move m = pick_next_move(moves, root_scores, i, n);
        GameState child = *root;
        int score = -HCE_INF;

        if (!chess_make_move_trusted(&child, m)) {
            continue;
        }
        ctx->nodes += 1;
        search_prepare_nn_child_frame(&child, ctx, 0, 1);

        int extension = search_move_extension(&child, m, depth);
        int next_depth = depth - 1 + extension;

        if (searched == 0) {
            score = -negamax(&child, next_depth, -beta, -alpha, 1, ctx, NULL);
        } else {
            score = -negamax(&child, next_depth, -alpha - 1, -alpha, 1, ctx, NULL);
            if (score > alpha && score < beta) {
                score = -negamax(&child, next_depth, -beta, -alpha, 1, ctx, NULL);
            }
        }

        if (ctx->timed_out) {
            break;
        }

        searched += 1;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
            if (alpha >= beta) {
                update_killer(ctx, 0, m);
                update_history(ctx, side, m, depth);
                tt_store(root->zobrist_hash, depth, 0, beta, HCE_TT_LOWER, m);
                if (best_move_out != NULL) {
                    *best_move_out = m;
                }
                return beta;
            }
        }
    }

    if (best_score == -HCE_INF) {
        best_score = search_eval_cp_stm(root, ctx, 0, depth, "root_fallback");
    }

    HceTtBound bound = HCE_TT_EXACT;
    if (best_score <= alpha_orig) {
        bound = HCE_TT_UPPER;
    } else if (best_score >= beta) {
        bound = HCE_TT_LOWER;
    }
    if (!ctx->timed_out) {
        tt_store(root->zobrist_hash, depth, 0, best_score, bound, best_move);
    }
    if (best_move_out != NULL) {
        *best_move_out = best_move;
    }
    return best_score;
}

static bool build_move_history_key(const GameState *state, char *out, size_t out_sz) {
    if (state == NULL || out == NULL || out_sz == 0) {
        return false;
    }
    out[0] = '\0';
    size_t used = 0;
    for (int i = 0; i < state->ply; ++i) {
        char uci[6] = {0};
        if (!chess_move_to_uci(state->undo_stack[i].move, uci)) {
            return false;
        }
        size_t len = strlen(uci);
        size_t need = len + ((used > 0) ? 1 : 0);
        if (used + need + 1 > out_sz) {
            return false;
        }
        if (used > 0) {
            out[used++] = ' ';
        }
        memcpy(out + used, uci, len);
        used += len;
        out[used] = '\0';
    }
    return true;
}

static bool state_matches_start_history(const GameState *state) {
    if (state == NULL) {
        return false;
    }

    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };
    GameState replay;
    chess_init(&replay, &cfg);

    for (int i = 0; i < state->ply; ++i) {
        if (!chess_make_move(&replay, state->undo_stack[i].move)) {
            return false;
        }
    }

    char state_fen[256];
    char replay_fen[256];
    chess_export_fen(state, state_fen, sizeof(state_fen));
    chess_export_fen(&replay, replay_fen, sizeof(replay_fen));
    return strcmp(state_fen, replay_fen) == 0;
}

bool hce_pick_opening_move(const GameState *s, Move *out_move) {
    if (s == NULL || out_move == NULL || !chess_opening_book_is_loaded()) {
        return false;
    }
    if (s->ply > 24) {
        return false;
    }
    if (!state_matches_start_history(s)) {
        return false;
    }

    Move legal[CHESS_MAX_MOVES];
    int legal_n = chess_generate_legal_moves(s, legal);
    if (legal_n <= 0) {
        return false;
    }

    char history_key[768];
    if (!build_move_history_key(s, history_key, sizeof(history_key))) {
        return false;
    }

    const ChessOpeningBookMove *moves = NULL;
    int move_count = 0;
    if (!chess_opening_book_lookup(history_key, &moves, &move_count) || move_count <= 0) {
        return false;
    }

    uint64_t total = 0;
    for (int i = 0; i < move_count; ++i) {
        total += (moves[i].weight > 0) ? moves[i].weight : 1U;
    }
    if (total == 0) {
        return false;
    }

    uint64_t pick = (s->zobrist_hash ^ ((uint64_t)s->fullmove_number << 21) ^ (uint64_t)now_ms()) % total;
    for (int i = 0; i < move_count; ++i) {
        Move parsed = 0;
        if (!chess_move_from_uci(s, moves[i].uci, &parsed)) {
            continue;
        }
        bool legal_move = false;
        for (int j = 0; j < legal_n; ++j) {
            if (legal[j] == parsed) {
                legal_move = true;
                break;
            }
        }
        if (!legal_move) {
            continue;
        }
        uint64_t weight = (moves[i].weight > 0) ? moves[i].weight : 1U;
        if (pick < weight) {
            *out_move = parsed;
            return true;
        }
        pick -= weight;
    }
    return false;
}

static bool run_search(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out, int override_depth, int override_ms) {
    if (state == NULL || out == NULL) {
        return false;
    }
    memset(out, 0, sizeof(*out));

    GameState root = *state;
    root.config.clock_enabled = false;
    root.config.increment_ms = 0;

    Move legal[CHESS_MAX_MOVES];
    int legal_n = chess_generate_legal_moves_mut(&root, legal);
    if (legal_n <= 0) {
        return false;
    }

    if (hce_pick_opening_move(&root, &out->best_move)) {
        out->found_move = true;
        out->used_opening_book = true;
        out->score_cp = 20;
        return true;
    }

    HceSearchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.profile = search_profile_for_current_backend();
    ctx.start_ms = now_ms();
    int think_ms = (override_ms > 0) ? override_ms : ((cfg != NULL && cfg->think_time_ms > 0) ? cfg->think_time_ms : 120);
    int hard_ms = think_ms;
    if (override_ms <= 0 && cfg != NULL && cfg->hard_time_ms > think_ms) {
        hard_ms = cfg->hard_time_ms;
    }
    ctx.hard_deadline_ms = ctx.start_ms + hard_ms;
    ctx.deadline_ms = ctx.hard_deadline_ms;
    ctx.max_depth = (override_depth > 0) ? override_depth : ((cfg != NULL && cfg->max_depth > 0) ? cfg->max_depth : 10);
    if (cfg != NULL && cfg->policy_root_count > 0 && cfg->policy_root_bonus > 0) {
        ctx.policy_root_count = cfg->policy_root_count;
        if (ctx.policy_root_count > CHESS_MAX_MOVES) {
            ctx.policy_root_count = CHESS_MAX_MOVES;
        }
        ctx.policy_root_bonus = cfg->policy_root_bonus;
        memcpy(ctx.policy_root_moves,
               cfg->policy_root_moves,
               (size_t)ctx.policy_root_count * sizeof(ctx.policy_root_moves[0]));
    }
    if (ctx.max_depth > HCE_MAX_DEPTH) {
        ctx.max_depth = HCE_MAX_DEPTH;
    }
    Move best_move = legal[0];
    int best_score = -HCE_INF;
    int depth_reached = 0;
    bool score_unstable = false;
    int last_iter_score = 0;
    bool have_iter_score = false;

    for (int depth = 1; depth <= ctx.max_depth; ++depth) {
        if (should_stop(&ctx)) {
            break;
        }
        if (depth >= 2 && depth_reached >= 1) {
            // Don't start an iteration that is unlikely to finish: each depth
            // costs roughly as much as all previous ones combined. A falling
            // score extends the soft budget toward the hard cap instead.
            int64_t elapsed = now_ms() - ctx.start_ms;
            int64_t soft_budget = think_ms;
            if (score_unstable && hard_ms > think_ms) {
                soft_budget = (int64_t)think_ms * 2;
                if (soft_budget > hard_ms) {
                    soft_budget = hard_ms;
                }
            }
            if (elapsed * 100 >= soft_budget * 55) {
                break;
            }
        }
        Move iter_best = best_move;
        int score = 0;
        int window = ctx.profile->aspiration_base + depth * ctx.profile->aspiration_depth_scale;
        int alpha = -HCE_INF;
        int beta = HCE_INF;
        if (depth >= 2 && best_score > -HCE_INF / 2 && best_score < HCE_INF / 2) {
            alpha = best_score - window;
            beta = best_score + window;
        }

        for (;;) {
            GameState iter = root;
            score = search_root(&iter, depth, alpha, beta, &ctx, &iter_best);
            if (ctx.timed_out) {
                break;
            }
            if (score <= alpha && alpha > -HCE_INF) {
                alpha = (score - window > -HCE_INF) ? (score - window) : -HCE_INF;
                window *= 2;
                continue;
            }
            if (score >= beta && beta < HCE_INF) {
                beta = (score + window < HCE_INF) ? (score + window) : HCE_INF;
                window *= 2;
                continue;
            }
            break;
        }
        if (ctx.timed_out) {
            break;
        }
        best_move = iter_best;
        best_score = score;
        depth_reached = depth;
        if (have_iter_score) {
            score_unstable = (score <= last_iter_score - 60);
        }
        last_iter_score = score;
        have_iter_score = true;
        if (cfg != NULL && cfg->info_callback != NULL) {
            cfg->info_callback(depth_reached,
                               best_score,
                               best_move,
                               ctx.nodes,
                               (int)(now_ms() - ctx.start_ms),
                               cfg->info_user_data);
        }
        if (best_score >= HCE_MATE_THRESHOLD || best_score <= -HCE_MATE_THRESHOLD) {
            break;
        }
    }

    out->best_move = best_move;
    out->found_move = true;
    out->score_cp = best_score;
    out->depth_reached = depth_reached;
    out->nodes = ctx.nodes;
    out->elapsed_ms = (int)(now_ms() - ctx.start_ms);
    if (g_nn_leaf_log_fp != NULL) {
        fflush(g_nn_leaf_log_fp);
    }
    return true;
}

bool hce_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out) {
    bool ok;
    hce_lock();
    hce_init_tables();
    g_hce_tt_generation += 1;
    ok = run_search(state, cfg, out, 0, 0);
    hce_unlock();
    return ok;
}

int hce_probe_deep_eval_cp_stm(const GameState *state) {
    if (state == NULL) {
        return 0;
    }
    AiSearchConfig cfg = {
        .think_time_ms = 35,
        .max_depth = 4,
        .info_callback = NULL,
        .info_user_data = NULL,
    };
    AiSearchResult result;
    hce_lock();
    hce_init_tables();
    bool ok = run_search(state, &cfg, &result, 4, 35);
    hce_unlock();
    if (ok && result.found_move && result.depth_reached > 0) {
        return result.score_cp;
    }
    return engine_eval_cp_stm(state);
}
