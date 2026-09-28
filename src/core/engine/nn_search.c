#define hce_nn_leaf_log_set_path nn_search_leaf_log_set_path
#define hce_nn_leaf_log_path nn_search_leaf_log_path
#define hce_nn_leaf_log_set_limit nn_search_leaf_log_set_limit
#define hce_nn_leaf_log_limit nn_search_leaf_log_limit
#define hce_nn_leaf_log_count nn_search_leaf_log_count
#define hce_nn_search_set_option nn_search_set_option
#define hce_nn_search_get_option nn_search_get_option
#define hce_nn_search_reset_options nn_search_reset_options
#define hce_score_search_draw_stm nn_search_score_search_draw_stm
#define hce_pick_opening_move nn_search_pick_opening_move
#define hce_pick_move nn_search_pick_move
#define hce_probe_deep_eval_cp_stm nn_search_probe_deep_eval_cp_stm
#define hce_piece_value nn_search_piece_value
#define hce_bishop_attacks nn_search_bishop_attacks
#define hce_rook_attacks nn_search_rook_attacks
#define hce_init_tables nn_search_init_tables
#define hce_knight_attacks nn_search_knight_attacks
#define hce_king_attacks nn_search_king_attacks
#define hce_pawn_attacks nn_search_pawn_attacks

#include "hce_internal.h"
#include "chess_hash.h"
#include "nn_eval.h"
#include "hce_tb.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
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
#define HCE_NN_PAWN_CORRECTION_BITS 14
#define HCE_NN_PAWN_CORRECTION_SIZE (1u << HCE_NN_PAWN_CORRECTION_BITS)
#define HCE_NN_PAWN_CORRECTION_MASK (HCE_NN_PAWN_CORRECTION_SIZE - 1u)
#define HCE_NN_PAWN_CORRECTION_GRAIN 64

#define HCE_CONT_KEYS (PIECE_COLOR_COUNT * PIECE_TYPE_COUNT * 64)

typedef enum HceTtBound {
    HCE_TT_NONE = 0,
    HCE_TT_EXACT = 1,
    HCE_TT_LOWER = 2,
    HCE_TT_UPPER = 3,
} HceTtBound;

// Lock-free TT entry shared by lazy-SMP threads: a key word plus one packed
// payload word, written under a key lock so readers never see a torn entry.
typedef struct HceTtEntry {
    atomic_uint_fast64_t key;
    atomic_uint_fast64_t payload;
} HceTtEntry;

#define HCE_TT_WRITE_LOCK UINT64_MAX
#define HCE_TT_MOVE_MASK ((1ULL << 26) - 1ULL)
#define HCE_TT_SCORE_SHIFT 26u
#define HCE_TT_DEPTH_SHIFT 42u
#define HCE_TT_BOUND_SHIFT 50u
#define HCE_TT_AGE_SHIFT 52u

typedef struct HceEvalCacheEntry {
    uint64_t key;
    int score;
    bool valid;
} HceEvalCacheEntry;

typedef struct HceSearchProfile {
    const char *name;
    int eval_scale_permille;
    int qsearch_delta_margin;
    int capture_see_ordering;
    int check_extensions;
    int countermove_ordering;
    int iteration_start_percent;
    int static_prune_margin_per_depth;
    int null_move_base_reduction;
    int null_move_eval_gate;
    int lmr_base_reduction;
    int lmr_depth_bonus_threshold;
    int lmr_late_move_threshold;
    int lmr_good_history_threshold;
    int lmr_bad_history_threshold;
    int lmr_backend_adjust;
    int lmr_log;
    int qsearch_tt;
    int improving;
    int singular;
    int history_gravity;
    int internal_reduction;
    int probcut_min_depth;
    int probcut_margin;
    int lmp_max_depth;
    int lmp_base_moves;
    int futility_max_depth;
    int futility_margin_per_depth;
    int see_prune_max_depth;
    int see_prune_margin_per_depth;
    int aspiration_base;
    int aspiration_depth_scale;
    int twofold_draw;
    int pawn_correction_weight_permille;
    int structure_correction_weight_permille;
    int cont_hist;            // continuation history (1- and 2-ply) in quiet ordering
    int rfp_max_depth;        // reverse futility pruning depth limit
    int null_eval_reduction;  // extra null-move reduction from (eval - beta)
    int multi_cut;            // singular search failing high above beta cuts the node
    int tt_eval;              // refine static eval with a bound-compatible TT score
} HceSearchProfile;

typedef struct HceSearchContext {
    int64_t start_ms;
    int64_t deadline_ms;
    int64_t hard_deadline_ms;
    bool timed_out;
    // The first iteration always completes so stop can never yield depth 0.
    bool in_first_iteration;
    uint64_t nodes;
    int max_depth;
    Move policy_root_moves[CHESS_MAX_MOVES];
    int policy_root_count;
    int policy_root_bonus;
    const HceSearchProfile *profile;
    Move killer[HCE_MAX_PLY][2];
    Move countermove[64][64];
    int history[PIECE_COLOR_COUNT][64][64];
    // Static eval per ply for the improving heuristic (INT_MIN = none).
    int eval_stack[HCE_MAX_PLY];
    // Move excluded at this ply by a singular-extension verification search.
    Move excluded[HCE_MAX_PLY];
    // Continuation history: [previous piece-to][current piece-to], heap
    // allocated only when NNContHist is on. cont_key[ply] is the piece-to of
    // the move made at ply (-1 for a null move).
    int16_t (*cont_hist)[HCE_CONT_KEYS];
    int cont_key[HCE_MAX_PLY];
    NnAccumulatorFrame nn_frames[HCE_MAX_PLY];
    HceEvalCacheEntry nn_eval_cache[HCE_NN_EVAL_CACHE_SIZE];
} HceSearchContext;

static const HceSearchProfile HCE_SEARCH_PROFILE_CLASSIC = {
    .name = "classic",
    .eval_scale_permille = 1000,
    .qsearch_delta_margin = 120,
    .capture_see_ordering = 0,
    .check_extensions = 1,
    .countermove_ordering = 0,
    .iteration_start_percent = 55,
    .static_prune_margin_per_depth = 90,
    .null_move_base_reduction = 2,
    .null_move_eval_gate = 0,
    .lmr_base_reduction = 1,
    .lmr_depth_bonus_threshold = 6,
    .lmr_late_move_threshold = 6,
    .lmr_good_history_threshold = 12000,
    .lmr_bad_history_threshold = -8000,
    .lmr_backend_adjust = 0,
    .lmr_log = 0,
    .qsearch_tt = 0,
    .improving = 0,
    .singular = 0,
    .history_gravity = 0,
    .internal_reduction = 0,
    .probcut_min_depth = 0,
    .probcut_margin = 200,
    .lmp_max_depth = 0,
    .lmp_base_moves = 4,
    .futility_max_depth = 0,
    .futility_margin_per_depth = 100,
    .see_prune_max_depth = 0,
    .see_prune_margin_per_depth = 40,
    .aspiration_base = 24,
    .aspiration_depth_scale = 6,
    .twofold_draw = 0,
    .pawn_correction_weight_permille = 0,
    .structure_correction_weight_permille = 0,
    .cont_hist = 0,
    .rfp_max_depth = 3,
    .null_eval_reduction = 0,
    .multi_cut = 0,
    .tt_eval = 0,
};

static const HceSearchProfile HCE_SEARCH_PROFILE_NN_DEFAULT = {
    .name = "nn",
    .eval_scale_permille = 800,
    .qsearch_delta_margin = 300,
    .capture_see_ordering = 0,
    .check_extensions = 1,
    .countermove_ordering = 0,
    .iteration_start_percent = 85,
    .static_prune_margin_per_depth = 80,
    .null_move_base_reduction = 2,
    .null_move_eval_gate = 0,
    .lmr_base_reduction = 1,
    .lmr_depth_bonus_threshold = 6,
    .lmr_late_move_threshold = 6,
    .lmr_good_history_threshold = 12000,
    .lmr_bad_history_threshold = -8000,
    .lmr_backend_adjust = 0,
    .lmr_log = 0,
    .qsearch_tt = 1,  // +47 Elo [+11, +85] vs off, 200 games at 10+0.1 (2026-09-28)
    .improving = 0,
    .singular = 1,  // +33 Elo [-0.3, +67], P=97%, 200 games at 10+0.1 (2026-09-28)
    .history_gravity = 0,
    .internal_reduction = 1,
    .probcut_min_depth = 0,
    .probcut_margin = 200,
    .lmp_max_depth = 2,
    .lmp_base_moves = 4,
    .futility_max_depth = 2,
    .futility_margin_per_depth = 100,
    .see_prune_max_depth = 0,
    .see_prune_margin_per_depth = 40,
    .aspiration_base = 40,
    .aspiration_depth_scale = 10,
    .twofold_draw = 1,
    .pawn_correction_weight_permille = 0,
    .structure_correction_weight_permille = 0,
    .cont_hist = 0,
    .rfp_max_depth = 3,
    .null_eval_reduction = 0,
    .multi_cut = 0,
    .tt_eval = 0,
};

static HceSearchProfile g_hce_search_profile_nn = {
    .name = "nn",
    .eval_scale_permille = 800,
    .qsearch_delta_margin = 300,
    .capture_see_ordering = 0,
    .check_extensions = 1,
    .countermove_ordering = 0,
    .iteration_start_percent = 85,
    .static_prune_margin_per_depth = 80,
    .null_move_base_reduction = 2,
    .null_move_eval_gate = 0,
    .lmr_base_reduction = 1,
    .lmr_depth_bonus_threshold = 6,
    .lmr_late_move_threshold = 6,
    .lmr_good_history_threshold = 12000,
    .lmr_bad_history_threshold = -8000,
    .lmr_backend_adjust = 0,
    .lmr_log = 0,
    .qsearch_tt = 1,
    .improving = 0,
    .singular = 1,
    .history_gravity = 0,
    .internal_reduction = 1,
    .probcut_min_depth = 0,
    .probcut_margin = 200,
    .lmp_max_depth = 2,
    .lmp_base_moves = 4,
    .futility_max_depth = 2,
    .futility_margin_per_depth = 100,
    .see_prune_max_depth = 0,
    .see_prune_margin_per_depth = 40,
    .aspiration_base = 40,
    .aspiration_depth_scale = 10,
    .twofold_draw = 1,
    .pawn_correction_weight_permille = 0,
    .structure_correction_weight_permille = 0,
    .cont_hist = 0,
    .rfp_max_depth = 3,
    .null_eval_reduction = 0,
    .multi_cut = 0,
    .tt_eval = 0,
};

static HceTtEntry g_hce_tt_default[HCE_TT_SIZE];
static HceTtEntry *g_hce_tt = g_hce_tt_default;
static uint64_t g_hce_tt_mask = HCE_TT_MASK;
static uint8_t g_hce_tt_generation = 0;
static int16_t g_nn_pawn_correction[PIECE_COLOR_COUNT][HCE_NN_PAWN_CORRECTION_SIZE];
static int16_t g_nn_minor_correction[PIECE_COLOR_COUNT][HCE_NN_PAWN_CORRECTION_SIZE];
static int16_t
    g_nn_nonpawn_correction[PIECE_COLOR_COUNT][PIECE_COLOR_COUNT][HCE_NN_PAWN_CORRECTION_SIZE];
static char g_nn_pawn_correction_model_path[1024];
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
    if (hce_option_ieq(name, "NNCaptureSeeOrdering") ||
        hce_option_ieq(name, "CaptureSeeOrdering")) {
        *out = &g_hce_search_profile_nn.capture_see_ordering;
        return true;
    }
    if (hce_option_ieq(name, "NNCheckExtensions") ||
        hce_option_ieq(name, "CheckExtensions")) {
        *out = &g_hce_search_profile_nn.check_extensions;
        return true;
    }
    if (hce_option_ieq(name, "NNCountermoveOrdering") ||
        hce_option_ieq(name, "CountermoveOrdering")) {
        *out = &g_hce_search_profile_nn.countermove_ordering;
        return true;
    }
    if (hce_option_ieq(name, "NNIterationStartPercent") ||
        hce_option_ieq(name, "IterationStartPercent")) {
        *out = &g_hce_search_profile_nn.iteration_start_percent;
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
    if (hce_option_ieq(name, "NNNullMoveEvalGate") ||
        hce_option_ieq(name, "NullMoveEvalGate")) {
        *out = &g_hce_search_profile_nn.null_move_eval_gate;
        return true;
    }
    if (hce_option_ieq(name, "NNContHist")) {
        *out = &g_hce_search_profile_nn.cont_hist;
        return true;
    }
    if (hce_option_ieq(name, "NNRfpDepth")) {
        *out = &g_hce_search_profile_nn.rfp_max_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNNullEvalRed")) {
        *out = &g_hce_search_profile_nn.null_eval_reduction;
        return true;
    }
    if (hce_option_ieq(name, "NNMultiCut")) {
        *out = &g_hce_search_profile_nn.multi_cut;
        return true;
    }
    if (hce_option_ieq(name, "NNTTEval")) {
        *out = &g_hce_search_profile_nn.tt_eval;
        return true;
    }
    if (hce_option_ieq(name, "NNSingular")) {
        *out = &g_hce_search_profile_nn.singular;
        return true;
    }
    if (hce_option_ieq(name, "NNImproving")) {
        *out = &g_hce_search_profile_nn.improving;
        return true;
    }
    if (hce_option_ieq(name, "NNQsearchTT")) {
        *out = &g_hce_search_profile_nn.qsearch_tt;
        return true;
    }
    if (hce_option_ieq(name, "NNLmrLog")) {
        *out = &g_hce_search_profile_nn.lmr_log;
        return true;
    }
    if (hce_option_ieq(name, "NNLmrBackendAdjust") || hce_option_ieq(name, "LmrBackendAdjust")) {
        *out = &g_hce_search_profile_nn.lmr_backend_adjust;
        return true;
    }
    if (hce_option_ieq(name, "NNHistoryGravity") ||
        hce_option_ieq(name, "HistoryGravity")) {
        *out = &g_hce_search_profile_nn.history_gravity;
        return true;
    }
    if (hce_option_ieq(name, "NNInternalReduction") ||
        hce_option_ieq(name, "InternalReduction")) {
        *out = &g_hce_search_profile_nn.internal_reduction;
        return true;
    }
    if (hce_option_ieq(name, "NNProbCutMinDepth") ||
        hce_option_ieq(name, "ProbCutMinDepth")) {
        *out = &g_hce_search_profile_nn.probcut_min_depth;
        return true;
    }
    if (hce_option_ieq(name, "NNProbCutMargin") ||
        hce_option_ieq(name, "ProbCutMargin")) {
        *out = &g_hce_search_profile_nn.probcut_margin;
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
    if (hce_option_ieq(name, "NNPawnCorrectionWeight") ||
        hce_option_ieq(name, "PawnCorrectionWeight")) {
        *out = &g_hce_search_profile_nn.pawn_correction_weight_permille;
        return true;
    }
    if (hce_option_ieq(name, "NNStructureCorrectionWeight") ||
        hce_option_ieq(name, "StructureCorrectionWeight")) {
        *out = &g_hce_search_profile_nn.structure_correction_weight_permille;
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
    } else if (field == &g_hce_search_profile_nn.lmr_log ||
               field == &g_hce_search_profile_nn.qsearch_tt ||
               field == &g_hce_search_profile_nn.improving ||
               field == &g_hce_search_profile_nn.singular ||
               field == &g_hce_search_profile_nn.cont_hist ||
               field == &g_hce_search_profile_nn.null_eval_reduction ||
               field == &g_hce_search_profile_nn.multi_cut ||
               field == &g_hce_search_profile_nn.tt_eval ||
               field == &g_hce_search_profile_nn.twofold_draw ||
               field == &g_hce_search_profile_nn.check_extensions ||
               field == &g_hce_search_profile_nn.countermove_ordering ||
               field == &g_hce_search_profile_nn.null_move_eval_gate ||
               field == &g_hce_search_profile_nn.history_gravity ||
               field == &g_hce_search_profile_nn.internal_reduction) {
        if (value < 0 || value > 1) {
            return false;
        }
    } else if (field == &g_hce_search_profile_nn.rfp_max_depth) {
        if (value < 0 || value > 12) {
            return false;
        }
    } else if (field == &g_hce_search_profile_nn.capture_see_ordering) {
        if (value < 0 || value > 2) {
            return false;
        }
    } else if (field == &g_hce_search_profile_nn.iteration_start_percent) {
        if (value < 1 || value > 100) {
            return false;
        }
    } else if (field == &g_hce_search_profile_nn.pawn_correction_weight_permille ||
               field == &g_hce_search_profile_nn.structure_correction_weight_permille) {
        if (value < 0 || value > 2000) {
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
    memset(g_nn_pawn_correction, 0, sizeof(g_nn_pawn_correction));
    memset(g_nn_minor_correction, 0, sizeof(g_nn_minor_correction));
    memset(g_nn_nonpawn_correction, 0, sizeof(g_nn_nonpawn_correction));
    g_nn_pawn_correction_model_path[0] = '\0';
}

static int64_t now_ms(void) {
    struct timespec ts;
#if defined(_WIN32)
    // msvcrt-based mingw has no timespec_get; winpthreads supplies clock_gettime.
    clock_gettime(CLOCK_REALTIME, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
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
    return &g_hce_tt[key & g_hce_tt_mask];
}

static uint64_t tt_pack_payload(Move move, int score, int depth, HceTtBound bound, uint8_t age) {
    return ((uint64_t)move & HCE_TT_MOVE_MASK) |
           ((uint64_t)(uint16_t)(int16_t)score << HCE_TT_SCORE_SHIFT) |
           ((uint64_t)(uint8_t)(int8_t)depth << HCE_TT_DEPTH_SHIFT) |
           ((uint64_t)bound << HCE_TT_BOUND_SHIFT) |
           ((uint64_t)age << HCE_TT_AGE_SHIFT);
}

static Move tt_payload_move(uint64_t payload) {
    return (Move)(payload & HCE_TT_MOVE_MASK);
}

static int tt_payload_score(uint64_t payload) {
    return (int)(int16_t)((payload >> HCE_TT_SCORE_SHIFT) & 0xFFFFULL);
}

static int tt_payload_depth(uint64_t payload) {
    return (int)(int8_t)((payload >> HCE_TT_DEPTH_SHIFT) & 0xFFULL);
}

static HceTtBound tt_payload_bound(uint64_t payload) {
    return (HceTtBound)((payload >> HCE_TT_BOUND_SHIFT) & 0x3ULL);
}

static uint8_t tt_payload_age(uint64_t payload) {
    return (uint8_t)((payload >> HCE_TT_AGE_SHIFT) & 0xFFULL);
}

static bool tt_probe(uint64_t key, int depth, int ply, int alpha, int beta, Move *move_out, int *score_out) {
    HceTtEntry *entry = tt_entry(key);
    uint64_t key_before = atomic_load_explicit(&entry->key, memory_order_acquire);
    if (key_before != key || key_before == HCE_TT_WRITE_LOCK) {
        return false;
    }
    uint64_t payload = atomic_load_explicit(&entry->payload, memory_order_relaxed);
    uint64_t key_after = atomic_load_explicit(&entry->key, memory_order_acquire);
    HceTtBound bound = tt_payload_bound(payload);
    if (key_after != key_before || bound == HCE_TT_NONE) {
        return false;
    }
    if (move_out != NULL) {
        *move_out = tt_payload_move(payload);
    }
    if (tt_payload_depth(payload) < depth || score_out == NULL) {
        return false;
    }

    int score = tt_score_from_store(tt_payload_score(payload), ply);
    if (bound == HCE_TT_EXACT) {
        *score_out = score;
        return true;
    }
    if (bound == HCE_TT_LOWER && score >= beta) {
        *score_out = score;
        return true;
    }
    if (bound == HCE_TT_UPPER && score <= alpha) {
        *score_out = score;
        return true;
    }
    return false;
}

// Read a TT entry without cutoff logic (singular extensions need its depth,
// bound and score).
static bool tt_peek(uint64_t key, int ply, Move *move, int *score, int *depth, HceTtBound *bound) {
    HceTtEntry *entry = tt_entry(key);
    uint64_t key_before = atomic_load_explicit(&entry->key, memory_order_acquire);
    if (key_before != key || key_before == HCE_TT_WRITE_LOCK) {
        return false;
    }
    uint64_t payload = atomic_load_explicit(&entry->payload, memory_order_relaxed);
    if (atomic_load_explicit(&entry->key, memory_order_acquire) != key_before) {
        return false;
    }
    *bound = tt_payload_bound(payload);
    if (*bound == HCE_TT_NONE) {
        return false;
    }
    *move = tt_payload_move(payload);
    *score = tt_score_from_store(tt_payload_score(payload), ply);
    *depth = tt_payload_depth(payload);
    return true;
}

static void tt_store(uint64_t key, int depth, int ply, int score, HceTtBound bound, Move move) {
    HceTtEntry *entry = tt_entry(key);
    uint64_t current_key = atomic_load_explicit(&entry->key, memory_order_acquire);
    if (current_key == HCE_TT_WRITE_LOCK) {
        return;
    }
    uint64_t current_payload = atomic_load_explicit(&entry->payload, memory_order_relaxed);
    // Keep deeper data for the same position within the current search
    // generation; entries from older searches are always replaceable.
    if (tt_payload_bound(current_payload) != HCE_TT_NONE &&
        current_key == key &&
        tt_payload_age(current_payload) == g_hce_tt_generation &&
        tt_payload_depth(current_payload) > depth &&
        bound != HCE_TT_EXACT) {
        return;
    }
    if (!atomic_compare_exchange_strong_explicit(&entry->key,
                                                  &current_key,
                                                  HCE_TT_WRITE_LOCK,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
        return;
    }
    atomic_store_explicit(&entry->payload,
                          tt_pack_payload(move, tt_score_to_store(score, ply), depth, bound,
                                          g_hce_tt_generation),
                          memory_order_relaxed);
    atomic_store_explicit(&entry->key, key, memory_order_release);
}

// UCI Hash for the NN search TT: largest power of two that fits. Takes the
// search lock so it never races a search; 16 MB keeps the built-in table.
int nn_search_set_hash_mb(int mb) {
    if (mb < 1) {
        mb = 1;
    }
    uint64_t entries = 1;
    while (entries * 2 * sizeof(HceTtEntry) <= (uint64_t)mb * 1024u * 1024u) {
        entries *= 2;
    }
    hce_lock();
    HceTtEntry *next = g_hce_tt_default;
    if (entries != HCE_TT_SIZE) {
        next = calloc((size_t)entries, sizeof(HceTtEntry));
        if (next == NULL) {
            hce_unlock();
            return (int)((g_hce_tt_mask + 1) * sizeof(HceTtEntry) / (1024u * 1024u));
        }
    } else {
        memset(g_hce_tt_default, 0, sizeof(g_hce_tt_default));
    }
    if (g_hce_tt != g_hce_tt_default) {
        free(g_hce_tt);
    }
    g_hce_tt = next;
    g_hce_tt_mask = entries - 1;
    hce_unlock();
    return (int)(entries * sizeof(HceTtEntry) / (1024u * 1024u));
}

static bool should_stop(HceSearchContext *ctx) {
    if (ctx == NULL || ctx->in_first_iteration) {
        return false;
    }
    if ((ctx->nodes & 2047ULL) != 0ULL) {
        return false;
    }
    // UCI "stop" (and the end of the main thread for SMP helpers). Without
    // this a pondering bot could not interrupt a long NN search.
    if (hce_search_stop_requested()) {
        ctx->timed_out = true;
        return true;
    }
    if (ctx->deadline_ms <= 0) {
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

    if (nn_eval_uses_full_threats()) {
        /*
         * Full Threats updates must collect and diff both attack sets.  Keep
         * them lazy: search_ensure_nn_frame() applies the incremental update
         * only if this child reaches an evaluation, avoiding work on TT hits,
         * pruned nodes, and immediate cutoffs.
         */
        ctx->nn_frames[child_ply].valid = false;
        return;
    }

    NnAccumulatorFrame *child_frame = &ctx->nn_frames[child_ply];
    const NnAccumulatorFrame *parent = &ctx->nn_frames[parent_ply];
    const UndoRecord *undo = &child->undo_stack[child->ply - 1];
    if (!parent->valid || parent->key != undo->hash_prev ||
        !nn_eval_update_frame(child, undo, parent, child_frame)) {
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

    int cached_score = 0;
    if (search_nn_eval_cache_probe(ctx, s->zobrist_hash, &cached_score)) {
        search_log_nn_leaf(s, ctx, ply, depth, phase, cached_score);
        return cached_score;
    }

    NnAccumulatorFrame *frame = &ctx->nn_frames[ply];
    if (!search_ensure_nn_frame(s, ctx, ply)) {
        int score = search_scale_eval_for_profile(ctx, nn_eval_cp_stm(s));
        search_log_nn_leaf(s, ctx, ply, depth, phase, score);
        return score;
    }

    if (frame->valid && frame->key == s->zobrist_hash) {
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

static int search_move_extension(const GameState *s_after_move,
                                 Move m,
                                 int depth,
                                 const HceSearchContext *ctx) {
    if (depth <= 2) {
        return 0;
    }
    if (search_profile(ctx)->check_extensions != 0 &&
        chess_in_check(s_after_move, s_after_move->side_to_move)) {
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
    attackers |= bb[side][PIECE_PAWN] & g_chess_pawn_attacks[side ^ 1][sq];
    attackers |= bb[side][PIECE_KNIGHT] & g_chess_knight_attacks[sq];
    attackers |= bb[side][PIECE_KING] & g_chess_king_attacks[sq];

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

static inline int cont_key_of(int side, Move m) {
    return (side * PIECE_TYPE_COUNT + move_piece(m)) * 64 + move_to(m);
}

static inline void cont_key_set(HceSearchContext *ctx, int ply, int key) {
    if (ply >= 0 && ply < HCE_MAX_PLY) {
        ctx->cont_key[ply] = key;
    }
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
        int material_order = hce_piece_value[victim] * 16 - hce_piece_value[attacker];
        if (search_profile(ctx)->capture_see_ordering != 0) {
            int see = static_exchange_eval(s, m);
            /*
             * Mode 1 only improves ordering within the capture stage. Mode 2
             * additionally demotes losing captures behind strong quiet moves.
             */
            int capture_stage =
                search_profile(ctx)->capture_see_ordering >= 2 && see < 0
                    ? 100000
                    : 1000000;
            score += capture_stage + material_order + see * 8;
        } else {
            score += 1000000 + material_order;
        }
    } else {
        if (ply >= 0 && ply < HCE_MAX_PLY) {
            if (ctx->killer[ply][0] == m) {
                score += 900000;
            } else if (ctx->killer[ply][1] == m) {
                score += 850000;
            }
        }
        if (search_profile(ctx)->countermove_ordering != 0 &&
            s->has_last_move &&
            ctx->countermove[move_from(s->last_move)][move_to(s->last_move)] == m) {
            score += 800000;
        }
        score += ctx->history[s->side_to_move][move_from(m)][move_to(m)];
        if (ctx->cont_hist != NULL && ply >= 1 && ply < HCE_MAX_PLY) {
            int cur = cont_key_of(s->side_to_move, m);
            int k1 = ctx->cont_key[ply - 1];
            if (k1 >= 0) {
                score += ctx->cont_hist[k1][cur];
            }
            if (ply >= 2 && ctx->cont_key[ply - 2] >= 0) {
                score += ctx->cont_hist[ctx->cont_key[ply - 2]][cur];
            }
        }
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

static void update_countermove(HceSearchContext *ctx,
                               const GameState *s,
                               Move move) {
    if (ctx == NULL ||
        s == NULL ||
        !s->has_last_move ||
        !is_quiet_move(move) ||
        search_profile(ctx)->countermove_ordering == 0) {
        return;
    }
    ctx->countermove[move_from(s->last_move)][move_to(s->last_move)] = move;
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
    if (search_profile(ctx)->history_gravity != 0) {
        /*
         * Search contexts are rebuilt for every played move, so the old
         * depth-squared bonus almost never reached the LMR history bands.
         * Scale it into a bounded gravity update: new evidence matters
         * immediately, while repeated updates asymptotically saturate.
         */
        const int limit = 16384;
        /*
         * A 32x scale reacted too sharply in short searches.  The 16x scale
         * still reaches the LMR history bands after repeated evidence while
         * retaining more of the previous ordering signal.
         */
        int scaled = delta * 16;
        if (scaled > limit) {
            scaled = limit;
        } else if (scaled < -limit) {
            scaled = -limit;
        }
        int magnitude = scaled >= 0 ? scaled : -scaled;
        *hist += scaled - (*hist * magnitude) / limit;
        if (*hist > limit) {
            *hist = limit;
        } else if (*hist < -limit) {
            *hist = -limit;
        }
        return;
    }
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

static void cont_hist_update(HceSearchContext *ctx, int ply, int side, Move m, int bonus) {
    int cur = cont_key_of(side, m);
    for (int back = 1; back <= 2 && ply - back >= 0; ++back) {
        int k = ctx->cont_key[ply - back];
        if (k < 0) {
            continue;
        }
        int16_t *h = &ctx->cont_hist[k][cur];
        int magnitude = bonus >= 0 ? bonus : -bonus;
        int v = *h + bonus - (*h * magnitude) / 16384;
        *h = (int16_t)(v > 16384 ? 16384 : (v < -16384 ? -16384 : v));
    }
}

// Quiet beta cutoff: reward the cutoff move, punish the quiets tried before it.
static void cont_hist_on_cutoff(HceSearchContext *ctx,
                                int ply,
                                int side,
                                Move best,
                                const Move quiets[CHESS_MAX_MOVES],
                                int quiet_count,
                                int depth) {
    if (ctx->cont_hist == NULL || ply < 1 || ply >= HCE_MAX_PLY) {
        return;
    }
    int bonus = 120 * depth - 80;
    if (bonus > 1600) {
        bonus = 1600;
    } else if (bonus < 40) {
        bonus = 40;
    }
    cont_hist_update(ctx, ply, side, best, bonus);
    for (int i = 0; i < quiet_count; ++i) {
        cont_hist_update(ctx, ply, side, quiets[i], -bonus);
    }
}

static uint32_t pawn_correction_index(const GameState *s) {
    uint64_t white = s->bb[PIECE_WHITE][PIECE_PAWN];
    uint64_t black = s->bb[PIECE_BLACK][PIECE_PAWN];
    uint64_t mixed = white * UINT64_C(0x9E3779B185EBCA87);
    mixed ^= (black << 23) | (black >> 41);
    mixed ^= mixed >> 29;
    mixed *= UINT64_C(0xC2B2AE3D27D4EB4F);
    return (uint32_t)(mixed & HCE_NN_PAWN_CORRECTION_MASK);
}

static uint64_t correction_mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xBF58476D1CE4E5B9);
    value ^= value >> 27;
    value *= UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

static uint64_t correction_rotate64(uint64_t value, unsigned shift) {
    return (value << shift) | (value >> (64U - shift));
}

static uint32_t minor_correction_index(const GameState *s) {
    uint64_t mixed = s->bb[PIECE_WHITE][PIECE_KNIGHT];
    mixed ^= correction_rotate64(s->bb[PIECE_WHITE][PIECE_BISHOP], 13);
    mixed ^= correction_rotate64(s->bb[PIECE_BLACK][PIECE_KNIGHT], 29);
    mixed ^= correction_rotate64(s->bb[PIECE_BLACK][PIECE_BISHOP], 47);
    return (uint32_t)(correction_mix64(mixed) & HCE_NN_PAWN_CORRECTION_MASK);
}

static uint32_t nonpawn_correction_index(const GameState *s, int color) {
    uint64_t mixed = s->bb[color][PIECE_KNIGHT];
    mixed ^= correction_rotate64(s->bb[color][PIECE_BISHOP], 11);
    mixed ^= correction_rotate64(s->bb[color][PIECE_ROOK], 23);
    mixed ^= correction_rotate64(s->bb[color][PIECE_QUEEN], 37);
    mixed ^= correction_rotate64(s->bb[color][PIECE_KING], 53);
    mixed ^= color == PIECE_WHITE
                 ? UINT64_C(0x9E3779B97F4A7C15)
                 : UINT64_C(0xD1B54A32D192ED03);
    return (uint32_t)(correction_mix64(mixed) & HCE_NN_PAWN_CORRECTION_MASK);
}

static int apply_eval_correction(const GameState *s,
                                 const HceSearchContext *ctx,
                                 int raw_eval) {
    const HceSearchProfile *profile = search_profile(ctx);
    if (profile->pawn_correction_weight_permille <= 0 &&
        profile->structure_correction_weight_permille <= 0) {
        return raw_eval;
    }
    int side = s->side_to_move;
    uint32_t pawn_index = pawn_correction_index(s);
    int pawn_stored = g_nn_pawn_correction[side][pawn_index];
    int adjusted = raw_eval;
    if (profile->pawn_correction_weight_permille > 0) {
        adjusted += (pawn_stored / HCE_NN_PAWN_CORRECTION_GRAIN) *
                    profile->pawn_correction_weight_permille / 1000;
    }
    if (profile->structure_correction_weight_permille > 0) {
        int minor_stored =
            g_nn_minor_correction[side][minor_correction_index(s)];
        int white_stored =
            g_nn_nonpawn_correction[side][PIECE_WHITE][
                nonpawn_correction_index(s, PIECE_WHITE)];
        int black_stored =
            g_nn_nonpawn_correction[side][PIECE_BLACK][
                nonpawn_correction_index(s, PIECE_BLACK)];
        /*
         * Keep the current Stockfish relative mix, normalized back to one
         * correction estimate so independently learned tables cannot add the
         * same error four times.
         */
        int64_t combined =
            INT64_C(15341) * pawn_stored +
            INT64_C(10569) * minor_stored +
            INT64_C(12906) * ((int64_t)white_stored + black_stored);
        const int64_t denominator =
            INT64_C(15341) + INT64_C(10569) + INT64_C(12906) * 2;
        int correction_cp =
            (int)(combined / denominator / HCE_NN_PAWN_CORRECTION_GRAIN);
        adjusted += correction_cp *
                    profile->structure_correction_weight_permille / 1000;
    }
    if (adjusted >= HCE_MATE_THRESHOLD) {
        adjusted = HCE_MATE_THRESHOLD - 1;
    } else if (adjusted <= -HCE_MATE_THRESHOLD) {
        adjusted = -HCE_MATE_THRESHOLD + 1;
    }
    return adjusted;
}

static void update_correction_entry(int16_t *entry,
                                    int target,
                                    int learning) {
    if (learning < 1) {
        learning = 1;
    } else if (learning > 96) {
        learning = 96;
    }
    int updated = (int)*entry + (target - (int)*entry) * learning / 256;
    if (updated > INT16_MAX) {
        updated = INT16_MAX;
    } else if (updated < INT16_MIN) {
        updated = INT16_MIN;
    }
    *entry = (int16_t)updated;
}

static void update_eval_correction(const GameState *s,
                                   HceSearchContext *ctx,
                                   int depth,
                                   int raw_eval,
                                   int searched_score) {
    const HceSearchProfile *profile = search_profile(ctx);
    if ((profile->pawn_correction_weight_permille <= 0 &&
         profile->structure_correction_weight_permille <= 0) ||
        searched_score >= HCE_MATE_THRESHOLD ||
        searched_score <= -HCE_MATE_THRESHOLD) {
        return;
    }
    int error = searched_score - raw_eval;
    if (error > 256) {
        error = 256;
    } else if (error < -256) {
        error = -256;
    }
    int side = s->side_to_move;
    int target = error * HCE_NN_PAWN_CORRECTION_GRAIN;
    int learning = depth * 8;
    if (learning < 8) {
        learning = 8;
    } else if (learning > 64) {
        learning = 64;
    }
    update_correction_entry(
        &g_nn_pawn_correction[side][pawn_correction_index(s)],
        target,
        learning);
    if (profile->structure_correction_weight_permille > 0) {
        update_correction_entry(
            &g_nn_minor_correction[side][minor_correction_index(s)],
            target,
            learning * 150 / 128);
        for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
            update_correction_entry(
                &g_nn_nonpawn_correction[side][color][
                    nonpawn_correction_index(s, color)],
                target,
                learning * 186 / 128);
        }
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

    // Optional TT use at depth 0: any stored entry (qsearch or full search)
    // may cut, and the result is stored back as a depth-0 bound.
    const bool use_tt = search_profile(ctx)->qsearch_tt != 0;
    const int alpha_orig = alpha;
    if (use_tt) {
        int tt_score = 0;
        Move tt_move_q = 0;
        if (tt_probe(s->zobrist_hash, 0, ply, alpha, beta, &tt_move_q, &tt_score)) {
            return tt_score;
        }
    }

    bool in_check = chess_in_check(s, s->side_to_move);
    int stand_pat = 0;
    if (!in_check) {
        stand_pat = search_eval_cp_stm(s, ctx, ply, 0, "qsearch");
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
        cont_key_set(ctx, ply, cont_key_of(s->side_to_move ^ 1, m));
        search_prepare_nn_child_frame(s, ctx, ply, ply + 1);
        int score = -quiescence(s, -beta, -alpha, ply + 1, ctx);
        chess_undo_move(s);
        if (ctx->timed_out) {
            return alpha;
        }
        if (score >= beta) {
            if (use_tt) {
                tt_store(s->zobrist_hash, 0, ply, beta, HCE_TT_LOWER, m);
            }
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    if (use_tt && !ctx->timed_out) {
        tt_store(s->zobrist_hash, 0, ply, alpha,
                 alpha > alpha_orig ? HCE_TT_EXACT : HCE_TT_UPPER, 0);
    }
    return alpha;
}

static int nn_lmr_log_reduction(int depth, int move_number) {
    static int table[64][64];
    static atomic_int ready;
    if (!atomic_load_explicit(&ready, memory_order_acquire)) {
        for (int d = 0; d < 64; ++d) {
            for (int m = 0; m < 64; ++m) {
                table[d][m] = (d == 0 || m == 0)
                                  ? 0
                                  : (int)(0.75 + log((double)d) * log((double)m) / 2.25);
            }
        }
        atomic_store_explicit(&ready, 1, memory_order_release);
    }
    int d = depth < 0 ? 0 : (depth > 63 ? 63 : depth);
    int m = move_number < 0 ? 0 : (move_number > 63 ? 63 : move_number);
    return table[d][m];
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
    const Move excluded = (ply >= 0 && ply < HCE_MAX_PLY) ? ctx->excluded[ply] : 0;
    if (excluded == 0 && tt_probe(s->zobrist_hash, depth, ply, alpha, beta, &tt_move, &tt_score)) {
        return tt_score;
    }
    if (excluded != 0) {
        Move peek_move = 0;
        int peek_score = 0, peek_depth = 0;
        HceTtBound peek_bound = HCE_TT_NONE;
        if (tt_peek(s->zobrist_hash, ply, &peek_move, &peek_score, &peek_depth, &peek_bound)) {
            tt_move = peek_move;
        }
    }

    // Syzygy WDL right after a capture or pawn move; cursed wins and blessed
    // losses are draws under the fifty-move rule.
    if (ply > 0 && hce_tb_largest() > 0) {
        int wdl = 0;
        if (hce_tb_probe_wdl(s, &wdl)) {
            int tb_score = wdl;
            if (wdl == 2) {
                tb_score = HCE_TB_WIN - ply;
            } else if (wdl == -2) {
                tb_score = -HCE_TB_WIN + ply;
            }
            tt_store(s->zobrist_hash, depth + 6, ply, tb_score, HCE_TT_EXACT, 0);
            return tb_score;
        }
    }

    bool in_check = chess_in_check(s, s->side_to_move);
    const HceSearchProfile *profile = search_profile(ctx);
    if (profile->internal_reduction != 0 &&
        depth >= 5 &&
        !in_check &&
        tt_move == 0) {
        depth -= 1;
    }
    bool static_eval_valid = false;
    int raw_static_eval = 0;
    int static_eval = 0;
    bool null_move_candidate =
        excluded == 0 &&
        ply > 0 &&
        depth >= 3 &&
        !in_check &&
        beta < HCE_MATE_THRESHOLD &&
        has_non_pawn_material(s, s->side_to_move);
    if (!in_check &&
        (depth <= profile->rfp_max_depth || profile->improving != 0 ||
         (profile->null_eval_reduction != 0 && null_move_candidate) ||
         (profile->futility_max_depth > 0 && depth <= profile->futility_max_depth) ||
         (profile->null_move_eval_gate != 0 && null_move_candidate) ||
         (profile->probcut_min_depth > 0 && depth >= profile->probcut_min_depth))) {
        raw_static_eval = search_eval_cp_stm(s, ctx, ply, depth, "static_eval");
        static_eval = apply_eval_correction(s, ctx, raw_static_eval);
        static_eval_valid = true;
        if (profile->tt_eval != 0) {
            // A stored search score bounds the true value better than the
            // static eval whenever its bound points the right way.
            Move te_move = 0;
            int te_score = 0, te_depth = 0;
            HceTtBound te_bound = HCE_TT_NONE;
            if (tt_peek(s->zobrist_hash, ply, &te_move, &te_score, &te_depth, &te_bound) &&
                te_score > -HCE_MATE_THRESHOLD && te_score < HCE_MATE_THRESHOLD &&
                (te_bound == HCE_TT_EXACT ||
                 (te_bound == HCE_TT_LOWER && te_score > static_eval) ||
                 (te_bound == HCE_TT_UPPER && te_score < static_eval))) {
                static_eval = te_score;
            }
        }
    }
    // Improving: this side's static eval beats its value two plies ago.
    bool improving = false;
    if (profile->improving != 0 && ply >= 0 && ply < HCE_MAX_PLY) {
        ctx->eval_stack[ply] = static_eval_valid ? static_eval : INT_MIN;
        improving = static_eval_valid && ply >= 2 && ctx->eval_stack[ply - 2] != INT_MIN &&
                    static_eval > ctx->eval_stack[ply - 2];
    }
    if (!in_check && depth <= profile->rfp_max_depth && beta < HCE_MATE_THRESHOLD) {
        int margin = profile->static_prune_margin_per_depth * (depth - (improving ? 1 : 0));
        if (static_eval >= beta + margin) {
            return beta;
        }
    }

    if (null_move_candidate &&
        (profile->null_move_eval_gate == 0 ||
         (static_eval_valid && static_eval >= beta))) {
        int reduction = search_profile(ctx)->null_move_base_reduction + depth / 4;
        if (profile->null_eval_reduction != 0 && static_eval_valid && static_eval > beta) {
            int extra = (static_eval - beta) / 200;
            reduction += extra > 3 ? 3 : extra;
        }
        if (reduction > depth - 1) {
            reduction = depth - 1;
        }
        if (reduction > 0) {
            ctx->nodes += 1;
            NullMoveUndo null_undo;
            if (nn_eval_uses_full_threats()) {
                (void)search_ensure_nn_frame(s, ctx, ply);
            }
            make_null_move(s, &null_undo);
            cont_key_set(ctx, ply, -1);
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
    if (!in_check &&
        profile->probcut_min_depth > 0 &&
        depth >= profile->probcut_min_depth &&
        beta < HCE_MATE_THRESHOLD - profile->probcut_margin) {
        int probcut_beta = beta + profile->probcut_margin;
        int see_threshold = probcut_beta - static_eval;
        if (nn_eval_uses_full_threats()) {
            (void)search_ensure_nn_frame(s, ctx, ply);
        }
        for (int i = 0; i < n; ++i) {
            Move m = moves[i];
            if ((!move_has_flag(m, MOVE_FLAG_CAPTURE) &&
                 !move_has_flag(m, MOVE_FLAG_PROMOTION)) ||
                static_exchange_eval(s, m) < see_threshold) {
                continue;
            }
            if (!chess_make_move_trusted(s, m)) {
                continue;
            }
            ctx->nodes += 1;
            cont_key_set(ctx, ply, cont_key_of(s->side_to_move ^ 1, m));
            search_prepare_nn_child_frame(s, ctx, ply, ply + 1);
            int score = -quiescence(
                s, -probcut_beta, -probcut_beta + 1, ply + 1, ctx);
            int probcut_depth = depth - 4;
            if (!ctx->timed_out && score >= probcut_beta && probcut_depth > 0) {
                score = -negamax(
                    s,
                    probcut_depth,
                    -probcut_beta,
                    -probcut_beta + 1,
                    ply + 1,
                    ctx,
                    NULL);
            }
            chess_undo_move(s);
            if (ctx->timed_out) {
                return alpha;
            }
            if (score >= probcut_beta) {
                tt_store(
                    s->zobrist_hash,
                    probcut_depth + 1,
                    ply,
                    beta,
                    HCE_TT_LOWER,
                    m);
                return beta;
            }
        }
    }
    int move_scores[CHESS_MAX_MOVES];
    score_moves(s, move_scores, moves, n, tt_move, ctx, ply);
    if (nn_eval_uses_full_threats()) {
        (void)search_ensure_nn_frame(s, ctx, ply);
    }

    // Singular extension: if every move but the TT move fails well below the
    // TT score in a reduced search that excludes it, extend the TT move.
    int singular_extension = 0;
    if (profile->singular != 0 && excluded == 0 && tt_move != 0 && ply > 0 &&
        depth >= 8 && ply < HCE_MAX_PLY - 1) {
        Move se_move = 0;
        int se_score = 0, se_depth = 0;
        HceTtBound se_bound = HCE_TT_NONE;
        if (tt_peek(s->zobrist_hash, ply, &se_move, &se_score, &se_depth, &se_bound) &&
            se_move == tt_move && se_depth >= depth - 3 &&
            (se_bound == HCE_TT_LOWER || se_bound == HCE_TT_EXACT) &&
            se_score > -HCE_MATE_THRESHOLD && se_score < HCE_MATE_THRESHOLD) {
            int singular_beta = se_score - 2 * depth;
            ctx->excluded[ply] = tt_move;
            int v = negamax(s, (depth - 1) / 2, singular_beta - 1, singular_beta, ply, ctx, NULL);
            ctx->excluded[ply] = 0;
            if (ctx->timed_out) {
                return alpha;
            }
            if (v < singular_beta) {
                singular_extension = 1;
            } else if (profile->multi_cut != 0 && singular_beta >= beta) {
                // Another move also beats beta at reduced depth: cut.
                return beta;
            }
        }
    }

    Move best_move = moves[0];
    int best_score = -HCE_INF;
    int alpha_orig = alpha;
    int side = s->side_to_move;
    int searched = 0;
    Move failed_quiets[CHESS_MAX_MOVES];
    int failed_quiet_count = 0;

    for (int i = 0; i < n; ++i) {
        Move m = pick_next_move(moves, move_scores, i, n);
        if (m == excluded) {
            continue;
        }
        bool quiet = is_quiet_move(m);
        bool recapture = is_recapture_move(s, m);
        if (quiet && !in_check && profile->lmp_max_depth > 0 &&
            depth <= profile->lmp_max_depth &&
            searched >= profile->lmp_base_moves + depth * 3) {
            // Every later quiet is skipped too; stop once no non-quiet move
            // remains instead of selection-picking quiets just to skip them.
            bool tactical_left = false;
            for (int j = i + 1; j < n; ++j) {
                if (!is_quiet_move(moves[j]) && moves[j] != excluded) {
                    tactical_left = true;
                    break;
                }
            }
            if (!tactical_left) {
                break;
            }
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
        cont_key_set(ctx, ply, cont_key_of(side, m));
        search_prepare_nn_child_frame(s, ctx, ply, ply + 1);

        int extension = search_move_extension(s, m, depth, ctx);
        if (singular_extension != 0 && m == tt_move && extension == 0) {
            extension = 1;
        }

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
                if (profile->lmr_log) {
                    // Classic-search table: 0.75 + ln(depth) * ln(move) / 2.25.
                    reduction = nn_lmr_log_reduction(depth, searched);
                } else {
                    reduction = profile->lmr_base_reduction;
                    if (depth >= profile->lmr_depth_bonus_threshold) {
                        reduction += 1;
                    }
                    if (searched >= profile->lmr_late_move_threshold) {
                        reduction += 1;
                    }
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
                if (profile->improving != 0 && !improving) {
                    reduction += 1;
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
                update_countermove(ctx, s, m);
                update_history(ctx, side, m, depth);
                if (has_non_pawn_material(s, side)) {
                    penalize_quiet_history(ctx, side, failed_quiets, failed_quiet_count, depth);
                }
                if (quiet) {
                    cont_hist_on_cutoff(ctx, ply, side, m, failed_quiets, failed_quiet_count, depth);
                }
                if (quiet && static_eval_valid && beta > raw_static_eval) {
                    update_eval_correction(s, ctx, depth, raw_static_eval, beta);
                }
                if (excluded == 0) {
                    tt_store(s->zobrist_hash, depth, ply, beta, HCE_TT_LOWER, m);
                }
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
    if (!ctx->timed_out && excluded == 0) {
        if (static_eval_valid && is_quiet_move(best_move)) {
            if (bound == HCE_TT_EXACT ||
                (bound == HCE_TT_UPPER && best_score < raw_static_eval)) {
                update_eval_correction(s, ctx, depth, raw_static_eval, best_score);
            }
        }
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
        cont_key_set(ctx, 0, cont_key_of(side, m));
        search_prepare_nn_child_frame(&child, ctx, 0, 1);

        int extension = search_move_extension(&child, m, depth, ctx);
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
    if (ctx.profile->cont_hist != 0) {
        ctx.cont_hist = calloc(HCE_CONT_KEYS, sizeof(*ctx.cont_hist));
    }
    if (ctx.profile->pawn_correction_weight_permille > 0 ||
        ctx.profile->structure_correction_weight_permille > 0) {
        const char *model_path = nn_eval_model_path();
        if (model_path != NULL &&
            strcmp(model_path, g_nn_pawn_correction_model_path) != 0) {
            memset(g_nn_pawn_correction, 0, sizeof(g_nn_pawn_correction));
            memset(g_nn_minor_correction, 0, sizeof(g_nn_minor_correction));
            memset(g_nn_nonpawn_correction, 0, sizeof(g_nn_nonpawn_correction));
            snprintf(g_nn_pawn_correction_model_path,
                     sizeof(g_nn_pawn_correction_model_path),
                     "%s",
                     model_path);
        }
    }
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
        ctx.in_first_iteration = (depth == 1);
        if (depth > 1 && should_stop(&ctx)) {
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
            if (elapsed * 100 >=
                soft_budget * ctx.profile->iteration_start_percent) {
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
    free(ctx.cont_hist);
    return true;
}

// Lazy SMP: helpers search the same root with their own contexts and share
// only the TT; the main thread's result is the one played.
#define NN_MAX_THREADS 8

typedef struct NnSmpHelperArgs {
    GameState root;
    AiSearchConfig cfg;
} NnSmpHelperArgs;

static void *nn_smp_helper_main(void *arg) {
    NnSmpHelperArgs *a = (NnSmpHelperArgs *)arg;
    AiSearchResult scratch;
    run_search(&a->root, &a->cfg, &scratch, 0, 0);
    return NULL;
}

bool hce_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out) {
    bool ok;
    hce_lock();
    hce_init_tables();
    g_hce_tt_generation += 1;

    Move tb_move = 0;
    int tb_score = 0;
    if (state != NULL && out != NULL && hce_tb_largest() > 0 &&
        hce_tb_probe_root(state, &tb_move, &tb_score)) {
        memset(out, 0, sizeof(*out));
        out->best_move = tb_move;
        out->found_move = true;
        out->score_cp = tb_score;
        out->depth_reached = 1;
        hce_unlock();
        return true;
    }

    int threads = (cfg != NULL) ? cfg->threads : 1;
    if (threads > NN_MAX_THREADS) {
        threads = NN_MAX_THREADS;
    }
    pthread_t helper_threads[NN_MAX_THREADS];
    static NnSmpHelperArgs helper_args[NN_MAX_THREADS];
    int helpers_started = 0;
    if (threads > 1 && state != NULL && cfg != NULL) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 64u * 1024u * 1024u);
        for (int i = 0; i < threads - 1; ++i) {
            NnSmpHelperArgs *a = &helper_args[i];
            a->root = *state;
            a->cfg = *cfg;
            a->cfg.threads = 1;
            a->cfg.info_callback = NULL;
            a->cfg.info_user_data = NULL;
            if (a->cfg.hard_time_ms > a->cfg.think_time_ms) {
                a->cfg.think_time_ms = a->cfg.hard_time_ms;
            }
            if (pthread_create(&helper_threads[helpers_started], &attr,
                               nn_smp_helper_main, a) != 0) {
                break;
            }
            helpers_started += 1;
        }
        pthread_attr_destroy(&attr);
    }

    ok = run_search(state, cfg, out, 0, 0);

    if (helpers_started > 0) {
        hce_search_request_stop();
        for (int i = 0; i < helpers_started; ++i) {
            pthread_join(helper_threads[i], NULL);
        }
    }
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
