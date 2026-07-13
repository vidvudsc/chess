#include "chess_ai.h"

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "hce_internal.h"
#include "nn_eval.h"

static ChessAiBackend g_chess_ai_backend = CHESS_AI_BACKEND_CLASSIC;

#define NN_EVAL_CACHE_BITS 18u
#define NN_EVAL_CACHE_SIZE (1u << NN_EVAL_CACHE_BITS)
#define NN_EVAL_CACHE_MASK (NN_EVAL_CACHE_SIZE - 1u)

#define EVAL_CACHE_TAG_BITS (64u - NN_EVAL_CACHE_BITS)
#define EVAL_CACHE_TAG_MASK ((1ULL << EVAL_CACHE_TAG_BITS) - 1ULL)
#define EVAL_CACHE_SCORE_SHIFT EVAL_CACHE_TAG_BITS
#define EVAL_CACHE_VALID (1ULL << 62)

typedef atomic_uint_fast64_t EvalCacheEntry;

static EvalCacheEntry g_nn_eval_cache[NN_EVAL_CACHE_SIZE];

static uint64_t eval_cache_pack(uint64_t key, int score_cp_stm) {
    return EVAL_CACHE_VALID |
           ((key >> NN_EVAL_CACHE_BITS) & EVAL_CACHE_TAG_MASK) |
           ((uint64_t)(uint16_t)(int16_t)score_cp_stm << EVAL_CACHE_SCORE_SHIFT);
}

static bool eval_cache_probe(const EvalCacheEntry *entry, uint64_t key, int *score_out) {
    uint64_t packed = atomic_load_explicit(entry, memory_order_relaxed);
    uint64_t tag = (key >> NN_EVAL_CACHE_BITS) & EVAL_CACHE_TAG_MASK;
    if ((packed & EVAL_CACHE_VALID) == 0 ||
        (packed & EVAL_CACHE_TAG_MASK) != tag) {
        return false;
    }
    if (score_out != NULL) {
        *score_out = (int)(int16_t)((packed >> EVAL_CACHE_SCORE_SHIFT) & 0xFFFFULL);
    }
    return true;
}

static void eval_cache_store(EvalCacheEntry *entry, uint64_t key, int score_cp_stm) {
    atomic_store_explicit(entry, eval_cache_pack(key, score_cp_stm), memory_order_relaxed);
}

static void nn_eval_cache_clear(void) {
    for (size_t i = 0; i < NN_EVAL_CACHE_SIZE; ++i) {
        atomic_store_explicit(&g_nn_eval_cache[i], 0, memory_order_relaxed);
    }
}

// Same idea for the classic HCE eval: it is a pure function of the position,
// so transposed/re-visited nodes can reuse the score.
static EvalCacheEntry g_hce_eval_cache[NN_EVAL_CACHE_SIZE];

void chess_ai_warmup(void) {
    hce_init_tables();
}

int engine_eval_cp_stm(const GameState *state) {
    if (state == NULL) {
        return 0;
    }
    if (g_chess_ai_backend == CHESS_AI_BACKEND_NN && nn_eval_is_loaded()) {
        EvalCacheEntry *entry = &g_nn_eval_cache[state->zobrist_hash & NN_EVAL_CACHE_MASK];
        int cached = 0;
        if (eval_cache_probe(entry, state->zobrist_hash, &cached)) {
            return cached;
        }
        int score = nn_eval_cp_stm(state);
        eval_cache_store(entry, state->zobrist_hash, score);
        return score;
    }
    if (g_chess_ai_backend == CHESS_AI_BACKEND_EXPERIMENTAL) {
        return hce_experimental_eval_cp_stm(state);
    }
    EvalCacheEntry *entry = &g_hce_eval_cache[state->zobrist_hash & NN_EVAL_CACHE_MASK];
    int cached = 0;
    if (eval_cache_probe(entry, state->zobrist_hash, &cached)) {
        return cached;
    }
    int score = hce_eval_cp_stm(state);
    eval_cache_store(entry, state->zobrist_hash, score);
    return score;
}

int chess_ai_eval_cp(const GameState *state) {
    if (state == NULL) {
        return 0;
    }
    int score_stm = hce_probe_deep_eval_cp_stm(state);
    return (state->side_to_move == PIECE_WHITE) ? score_stm : -score_stm;
}

int chess_ai_eval_fast_cp(const GameState *state) {
    if (state == NULL) {
        return 0;
    }
    int score_stm = engine_eval_cp_stm(state);
    return (state->side_to_move == PIECE_WHITE) ? score_stm : -score_stm;
}

bool chess_ai_eval_breakdown(const GameState *state, ChessEvalBreakdown *out) {
    if (state == NULL || out == NULL) {
        return false;
    }
    return hce_eval_breakdown(state, out);
}

bool chess_ai_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out) {
    AiSearchConfig local_cfg = {
        .think_time_ms = 120,
        .max_depth = 0,
        .info_callback = NULL,
        .info_user_data = NULL,
    };
    if (cfg != NULL) {
        local_cfg = *cfg;
    }
    return hce_pick_move(state, &local_cfg, out);
}

bool chess_engine_query(const GameState *state, const ChessEngineRequest *req, ChessEngineResponse *out) {
    if (state == NULL || out == NULL) {
        return false;
    }
    memset(out, 0, sizeof(*out));

    ChessEngineRequest request = {
        .kind = CHESS_ENGINE_REQUEST_BEST_MOVE,
        .think_time_ms = 120,
        .max_depth = 10,
    };
    if (req != NULL) {
        request = *req;
    }

    switch (request.kind) {
        case CHESS_ENGINE_REQUEST_BEST_MOVE: {
            AiSearchConfig cfg = {
                .think_time_ms = request.think_time_ms,
                .max_depth = request.max_depth,
                .info_callback = NULL,
                .info_user_data = NULL,
            };
            AiSearchResult result;
            if (!chess_ai_pick_move(state, &cfg, &result)) {
                return false;
            }
            out->ok = true;
            out->best_move = result.best_move;
            out->found_move = result.found_move;
            out->used_opening_book = result.used_opening_book;
            out->score_cp_stm = result.score_cp;
            out->score_cp_white = (state->side_to_move == PIECE_WHITE) ? result.score_cp : -result.score_cp;
            out->depth_reached = result.depth_reached;
            out->nodes = result.nodes;
            out->elapsed_ms = result.elapsed_ms;
            return true;
        }
        case CHESS_ENGINE_REQUEST_EVAL_FAST: {
            int white_cp = chess_ai_eval_fast_cp(state);
            out->ok = true;
            out->score_cp_white = white_cp;
            out->score_cp_stm = (state->side_to_move == PIECE_WHITE) ? white_cp : -white_cp;
            return true;
        }
        case CHESS_ENGINE_REQUEST_EVAL_DEEP: {
            int white_cp = chess_ai_eval_cp(state);
            out->ok = true;
            out->score_cp_white = white_cp;
            out->score_cp_stm = (state->side_to_move == PIECE_WHITE) ? white_cp : -white_cp;
            return true;
        }
        default:
            return false;
    }
}

bool chess_ai_opening_book_set_path(const char *path) {
    return chess_opening_book_set_path(path);
}

bool chess_ai_opening_book_is_loaded(void) {
    return chess_opening_book_is_loaded();
}

const char *chess_ai_opening_book_path(void) {
    return chess_opening_book_loaded_path();
}

bool chess_ai_set_backend(ChessAiBackend backend) {
    if (backend != CHESS_AI_BACKEND_CLASSIC &&
        backend != CHESS_AI_BACKEND_NN &&
        backend != CHESS_AI_BACKEND_EXPERIMENTAL) {
        return false;
    }
    if (backend == CHESS_AI_BACKEND_NN && !nn_eval_is_loaded()) {
        return false;
    }
    g_chess_ai_backend = backend;
    return true;
}

ChessAiBackend chess_ai_get_backend(void) {
    return g_chess_ai_backend;
}

const char *chess_ai_backend_name(ChessAiBackend backend) {
    switch (backend) {
        case CHESS_AI_BACKEND_NN:
            return "nn";
        case CHESS_AI_BACKEND_EXPERIMENTAL:
            return "experimental";
        case CHESS_AI_BACKEND_CLASSIC:
        default:
            return "classic";
    }
}

bool chess_ai_set_nn_model_path(const char *path) {
    bool ok = nn_eval_load_model(path);
    if (ok) {
        nn_eval_cache_clear();
    }
    return ok;
}

bool chess_ai_nn_model_is_loaded(void) {
    return nn_eval_is_loaded();
}

const char *chess_ai_nn_model_path(void) {
    return nn_eval_model_path();
}
