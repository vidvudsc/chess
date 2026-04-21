#include "hce_internal.h"
#include "chess_hash.h"
#include "nn_eval.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HCE_TT_BITS 18
#define HCE_TT_SIZE (1u << HCE_TT_BITS)
#define HCE_TT_MASK (HCE_TT_SIZE - 1u)

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
} HceTtEntry;

typedef struct HceSearchContext {
    int64_t start_ms;
    int64_t deadline_ms;
    int64_t hard_deadline_ms;
    bool timed_out;
    uint64_t nodes;
    int max_depth;
    int verification_plies;
    Move killer[HCE_MAX_PLY][2];
    int history[PIECE_COLOR_COUNT][64][64];
    Move pv_move[HCE_MAX_PLY];
    GameState *null_scratch[HCE_MAX_PLY];
    NnAccumulatorFrame nn_frames[HCE_MAX_PLY];
} HceSearchContext;

typedef struct HceRootMoveInfo {
    Move move;
    int score;
    bool searched;
} HceRootMoveInfo;

static int previous_root_order_bonus(const HceRootMoveInfo *prev_moves,
                                     int prev_count,
                                     Move move) {
    if (prev_moves == NULL || prev_count <= 0) {
        return 0;
    }
    for (int i = 0; i < prev_count; ++i) {
        if (!prev_moves[i].searched || prev_moves[i].move != move) {
            continue;
        }
        // Keep root ordering stable across iterative deepening so low-time
        // searches spend nodes first on the best moves from the previous pass.
        return 4000000 + prev_moves[i].score;
    }
    return 0;
}

static HceTtEntry g_hce_tt[HCE_TT_SIZE];
static atomic_flag g_hce_lock = ATOMIC_FLAG_INIT;

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

static void free_null_scratch(HceSearchContext *ctx) {
    if (ctx == NULL) {
        return;
    }
    for (int i = 0; i < HCE_MAX_PLY; ++i) {
        free(ctx->null_scratch[i]);
        ctx->null_scratch[i] = NULL;
    }
}

static bool search_is_insufficient_material(const GameState *s) {
    if (s == NULL) {
        return false;
    }
    return !chess_has_mating_material(s, PIECE_WHITE) &&
           !chess_has_mating_material(s, PIECE_BLACK);
}

static bool search_is_repetition_draw(const GameState *s) {
    if (s == NULL || s->hash_history_count <= 0) {
        return false;
    }

    int repetitions = 0;
    uint64_t current = s->zobrist_hash;
    int begin = s->irreversible_ply;
    if (begin < 0) {
        begin = 0;
    }
    for (int i = s->hash_history_count - 1; i >= begin; --i) {
        if (s->hash_history[i] == current) {
            ++repetitions;
        }
    }

    // Be robust to search-only states that may not have appended the current hash.
    if (s->hash_history[s->hash_history_count - 1] != current) {
        ++repetitions;
    }
    return repetitions >= 3;
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
    if (search_is_repetition_draw(s)) {
        return 0;
    }
    return INT_MIN;
}

static int score_terminal_stm(const GameState *s, int ply) {
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
            return hce_score_search_draw_stm(s);
        default:
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
    if (entry->bound != HCE_TT_NONE && entry->key == key && entry->depth > depth && bound != HCE_TT_EXACT) {
        return;
    }
    entry->key = key;
    entry->move = move;
    entry->depth = (int8_t)depth;
    entry->bound = (uint8_t)bound;
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

static int search_eval_cp_stm(const GameState *s, HceSearchContext *ctx, int ply) {
    if (s == NULL) {
        return 0;
    }
    if (chess_ai_get_backend() != CHESS_AI_BACKEND_NN || !nn_eval_is_loaded()) {
        return engine_eval_cp_stm(s);
    }
    if (ply < 0 || ply >= HCE_MAX_PLY) {
        return nn_eval_cp_stm(s);
    }

    NnAccumulatorFrame *frame = &ctx->nn_frames[ply];
    if (frame->valid && frame->key == s->zobrist_hash) {
        return nn_eval_cp_stm_from_frame(s, frame);
    }

    bool ok = false;
    if (ply == 0 || s->ply == 0) {
        ok = nn_eval_build_frame(s, frame);
    } else {
        const NnAccumulatorFrame *parent = &ctx->nn_frames[ply - 1];
        const UndoRecord *undo = &s->undo_stack[s->ply - 1];
        if (parent->valid && parent->key == undo->hash_prev) {
            ok = nn_eval_update_frame(s, undo, parent, frame);
        }
        if (!ok) {
            ok = nn_eval_build_frame(s, frame);
        }
    }
    if (!ok) {
        return nn_eval_cp_stm(s);
    }
    return nn_eval_cp_stm_from_frame(s, frame);
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

static int search_move_extension(const GameState *s_after_move, Move m, int depth, bool recapture) {
    if (depth <= 2) {
        return 0;
    }
    if (chess_in_check(s_after_move, s_after_move->side_to_move)) {
        return 1;
    }
    if (move_has_flag(m, MOVE_FLAG_PROMOTION)) {
        return 1;
    }
    if (recapture) {
        return 1;
    }
    return 0;
}

static bool search_in_verification(const HceSearchContext *ctx, int ply) {
    return ctx != NULL && ctx->verification_plies > ply;
}

static bool is_qsearch_tactical_move(Move m) {
    return move_has_flag(m, MOVE_FLAG_CAPTURE) ||
           move_has_flag(m, MOVE_FLAG_PROMOTION);
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

static void apply_null_move(GameState *dst, const GameState *src) {
    *dst = *src;
    dst->side_to_move ^= 1;
    dst->ep_square = CHESS_NO_SQUARE;
    dst->halfmove_clock += 1;
    dst->has_last_move = false;
    dst->last_move = 0;
    if (src->side_to_move == PIECE_BLACK) {
        dst->fullmove_number += 1;
    }
    dst->zobrist_hash = chess_hash_full(dst);
}

static GameState *null_scratch_for_ply(HceSearchContext *ctx, int ply) {
    if (ctx == NULL || ply < 0 || ply >= HCE_MAX_PLY) {
        return NULL;
    }
    if (ctx->null_scratch[ply] == NULL) {
        ctx->null_scratch[ply] = (GameState *)malloc(sizeof(GameState));
    }
    return ctx->null_scratch[ply];
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
        return search_eval_cp_stm(s, ctx, ply);
    }

    int term = score_terminal_stm(s, ply);
    if (term != INT_MIN) {
        return term;
    }

    bool in_check = chess_in_check(s, s->side_to_move);
    int stand_pat = search_eval_cp_stm(s, ctx, ply);
    if (!in_check) {
        if (stand_pat >= beta) {
            return beta;
        }
        if (stand_pat > alpha) {
            alpha = stand_pat;
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

    Move tactical_moves[CHESS_MAX_MOVES];
    int tactical_n = 0;
    if (in_check) {
        memcpy(tactical_moves, moves, (size_t)n * sizeof(Move));
        tactical_n = n;
    } else {
        for (int i = 0; i < n; ++i) {
            if (is_qsearch_tactical_move(moves[i])) {
                tactical_moves[tactical_n++] = moves[i];
            }
        }
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
            int delta_margin = search_uses_nn_backend() ? 160 : 120;
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
        return search_eval_cp_stm(s, ctx, ply);
    }

    int term = score_terminal_stm(s, ply);
    if (term != INT_MIN) {
        return term;
    }
    if (ply >= HCE_MAX_PLY - 1) {
        return search_eval_cp_stm(s, ctx, ply);
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
    int static_eval = in_check ? 0 : search_eval_cp_stm(s, ctx, ply);

    if (!search_in_verification(ctx, ply) && !in_check && depth <= 3 && beta < HCE_MATE_THRESHOLD) {
        int margin = search_uses_nn_backend() ? (110 * depth) : (90 * depth);
        if (static_eval >= beta + margin) {
            return beta;
        }
    }

    if (!search_in_verification(ctx, ply) &&
        ply > 0 &&
        depth >= 3 &&
        !in_check &&
        beta < HCE_MATE_THRESHOLD &&
        has_non_pawn_material(s, s->side_to_move)) {
        GameState *null_state = null_scratch_for_ply(ctx, ply);
        if (null_state != NULL) {
            apply_null_move(null_state, s);
            int reduction = (search_uses_nn_backend() ? 1 : 2) + depth / 4;
            if (reduction > depth - 1) {
                reduction = depth - 1;
            }
            if (reduction > 0) {
                ctx->nodes += 1;
                int score = -negamax(null_state,
                                     depth - 1 - reduction,
                                     -beta,
                                     -beta + 1,
                                     ply + 1,
                                     ctx,
                                     NULL);
                if (ctx->timed_out) {
                    return alpha;
                }
                if (score >= beta) {
                    return beta;
                }
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
        if (!search_in_verification(ctx, ply) && !in_check && quiet && depth <= 3 && searched >= 4) {
            int futility_margin = search_uses_nn_backend()
                                      ? (160 * depth + searched * 10)
                                      : (120 * depth + searched * 12);
            if (static_eval + futility_margin <= alpha) {
                continue;
            }
        }
        if (!chess_make_move_trusted(s, m)) {
            continue;
        }
        ctx->nodes += 1;

        int extension = search_move_extension(s, m, depth, recapture);

        int score;
        int next_depth = depth - 1 + extension;
        if (searched == 0) {
            score = -negamax(s, next_depth, -beta, -alpha, ply + 1, ctx, NULL);
        } else {
            int reduction = 0;
            if (!search_in_verification(ctx, ply) &&
                !in_check &&
                quiet &&
                depth >= 3 &&
                searched >= 2) {
                reduction = 1;
                if (depth >= 6) {
                    reduction += 1;
                }
                if (searched >= 6) {
                    reduction += 1;
                }
                int hist = ctx->history[side][move_from(m)][move_to(m)];
                if (hist > 12000) {
                    reduction -= 1;
                } else if (hist < -8000) {
                    reduction += 1;
                }
                if (recapture) {
                    reduction -= 1;
                }
                if (search_uses_nn_backend()) {
                    reduction -= 1;
                }
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
            if (ply < HCE_MAX_PLY) {
                ctx->pv_move[ply] = m;
            }
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
        best_score = static_eval;
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
                       Move *best_move_out,
                       const HceRootMoveInfo *prev_root_moves,
                       int prev_root_move_count,
                       HceRootMoveInfo root_moves[CHESS_MAX_MOVES],
                       int *root_move_count_out) {
    Move tt_move = 0;
    int tt_score = 0;
    Move moves[CHESS_MAX_MOVES];
    int n = chess_generate_legal_moves_mut(root, moves);
    int alpha_orig = alpha;
    int best_score = -HCE_INF;
    Move best_move = (n > 0) ? moves[0] : 0;
    int side = root->side_to_move;
    int searched = 0;

    if (root_move_count_out != NULL) {
        *root_move_count_out = 0;
    }
    if (n <= 0) {
        if (best_move_out != NULL) {
            *best_move_out = 0;
        }
        return score_terminal_stm(root, 0);
    }

    (void)tt_probe(root->zobrist_hash, depth, 0, alpha, beta, &tt_move, &tt_score);
    int root_scores[CHESS_MAX_MOVES];
    score_moves(root, root_scores, moves, n, tt_move, ctx, 0);
    for (int i = 0; i < n; ++i) {
        root_scores[i] += previous_root_order_bonus(prev_root_moves, prev_root_move_count, moves[i]);
    }

    for (int i = 0; i < n; ++i) {
        Move m = pick_next_move(moves, root_scores, i, n);
        GameState child = *root;
        bool recapture = is_recapture_move(root, m);
        int score = -HCE_INF;

        if (!chess_make_move_trusted(&child, m)) {
            continue;
        }
        ctx->nodes += 1;

        int extension = search_move_extension(&child, m, depth, recapture);
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

        if (root_moves != NULL && searched < CHESS_MAX_MOVES) {
            root_moves[searched].move = m;
            root_moves[searched].score = score;
            root_moves[searched].searched = true;
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
                if (root_move_count_out != NULL) {
                    *root_move_count_out = searched;
                }
                return beta;
            }
        }
    }

    if (best_score == -HCE_INF) {
        best_score = search_eval_cp_stm(root, ctx, 0);
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
    if (root_move_count_out != NULL) {
        *root_move_count_out = searched;
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
    ctx.start_ms = now_ms();
    int think_ms = (override_ms > 0) ? override_ms : ((cfg != NULL && cfg->think_time_ms > 0) ? cfg->think_time_ms : 120);
    ctx.hard_deadline_ms = ctx.start_ms + think_ms;
    ctx.deadline_ms = ctx.hard_deadline_ms;
    ctx.max_depth = (override_depth > 0) ? override_depth : ((cfg != NULL && cfg->max_depth > 0) ? cfg->max_depth : 10);
    if (ctx.max_depth > HCE_MAX_DEPTH) {
        ctx.max_depth = HCE_MAX_DEPTH;
    }
    Move best_move = legal[0];
    int best_score = -HCE_INF;
    int depth_reached = 0;
    HceRootMoveInfo prev_root_moves[CHESS_MAX_MOVES];
    HceRootMoveInfo iter_root_moves[CHESS_MAX_MOVES];
    int prev_root_move_count = 0;
    memset(prev_root_moves, 0, sizeof(prev_root_moves));

    for (int depth = 1; depth <= ctx.max_depth; ++depth) {
        if (should_stop(&ctx)) {
            break;
        }
        Move iter_best = best_move;
        int score = 0;
        int iter_root_move_count = 0;
        int window = search_uses_nn_backend() ? (32 + depth * 8) : (24 + depth * 6);
        int alpha = -HCE_INF;
        int beta = HCE_INF;
        if (depth >= 2 && best_score > -HCE_INF / 2 && best_score < HCE_INF / 2) {
            alpha = best_score - window;
            beta = best_score + window;
        }

        for (;;) {
            GameState iter = root;
            memset(iter_root_moves, 0, sizeof(iter_root_moves));
            iter_root_move_count = 0;
            score = search_root(&iter,
                                depth,
                                alpha,
                                beta,
                                &ctx,
                                &iter_best,
                                prev_root_moves,
                                prev_root_move_count,
                                iter_root_moves,
                                &iter_root_move_count);
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
        memcpy(prev_root_moves, iter_root_moves, sizeof(prev_root_moves));
        prev_root_move_count = iter_root_move_count;
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
    free_null_scratch(&ctx);
    return true;
}

bool hce_pick_move(const GameState *state, const AiSearchConfig *cfg, AiSearchResult *out) {
    bool ok;
    hce_lock();
    hce_init_tables();
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
