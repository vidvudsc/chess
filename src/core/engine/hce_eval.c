#include "hce_internal.h"
#include "chess_rules.h"

#include <string.h>

const int hce_piece_value[PIECE_TYPE_COUNT] = {
    0,
    1367,
    435,
    455,
    678,
    100,
};

const int hce_phase_inc[PIECE_TYPE_COUNT] = {
    0,
    4,
    1,
    1,
    2,
    0,
};

static bool g_hce_tables_ready = false;
static uint64_t g_knight_attacks[64];
static uint64_t g_king_attacks[64];
static uint64_t g_pawn_attacks[PIECE_COLOR_COUNT][64];
static uint64_t g_file_masks[8];
static uint64_t g_neighbor_file_masks[8];
static uint64_t g_passed_masks[PIECE_COLOR_COUNT][64];

static const int k_pawn_pst[64] = {
       0,    0,    0,    0,    0,    0,    0,    0,
      44,   83,   33,   37,   61,   38,    5,   59,
       8,   17,   42,   40,   67,   42,   29,   39,
       4,    8,   41,   71,   53,   68,   13,    2,
     -13,  -10,   11,   33,   35,   30,    3,  -13,
     -17,  -17,   -9,   -4,    8,   15,   25,  -14,
     -30,  -15,   -4,  -14,  -10,   30,   19,  -21,
       0,    0,    0,    0,    0,    0,    0,    0,
};
static const int k_pawn_pst_eg[64] = {
       0,    0,    0,    0,    0,    0,    0,    0,
      27,   14,   37,   13,   38,   24,   19,   40,
      34,   11,    4,   23,   12,   17,   23,   23,
      27,   28,   13,    7,    9,    5,   29,   17,
      16,   17,   15,   10,   10,    1,   18,   18,
      12,   11,   11,   16,   16,   12,   -1,    9,
      18,   11,   17,   18,   19,    6,    2,   16,
       0,    0,    0,    0,    0,    0,    0,    0,
};

static const int k_knight_pst[64] = {
     -84,  -43,  -32,  -35,  -20,  -29,  -46,  -63,
     -69,  -17,  -12,   10,   11,    2,  -28,  -42,
     -23,    0,    4,   14,   17,   17,   13,  -23,
      -4,  -18,   19,   25,   31,   25,   -4,    3,
     -17,   -2,   24,   45,   35,   18,   -4,   27,
     -12,    4,   19,   50,   41,   14,   35,  -25,
     -43,  -26,   -4,   10,    7,   10,  -38,  -42,
    -206,  -35,  -38,  -23,  -23,  -49,  -46,  -89,
};
static const int k_knight_pst_eg[64] = {
     -38,  -39,  -38,  -23,  -34,  -28,  -44,  -32,
     -30,  -25,   -9,  -17,  -17,  -16,  -24,  -25,
     -34,  -13,    3,   11,   -1,   -4,  -10,  -17,
     -15,   -1,   14,   21,   26,   12,    6,  -23,
     -11,   14,   24,   27,   23,   22,   13,  -15,
     -12,    1,    3,   17,   11,   13,    0,  -11,
     -25,  -19,   -9,   16,    9,    9,  -13,  -33,
     -55,  -21,  -25,  -14,   -7,  -25,  -17,  -32,
};

static const int k_bishop_pst[64] = {
     -22,   -1,  -12,  -20,    0,  -18,   -5,  -22,
     -16,   28,   14,   -3,   14,    3,   39,   -5,
     -11,   16,   21,   -8,    5,   26,   24,    5,
      11,  -12,  -19,   19,   12,   -8,  -10,   10,
       3,   -7,  -13,   25,   16,   -9,   -3,   -8,
      -7,   -1,   10,   20,   14,    4,    6,   15,
      -5,   -1,  -23,    5,    4,   -8,    9,  -40,
     -31,   -4,  -13,   -8,   -9,  -17,   -6,  -22,
};
static const int k_bishop_pst_eg[64] = {
     -20,   -6,  -17,   -6,   -9,   -4,    3,   -5,
      -3,  -29,   -9,   -6,  -13,  -13,  -27,  -12,
      -8,    3,  -11,   14,    3,   -2,  -13,  -18,
      -9,   -3,    8,    4,    9,    7,    5,   -5,
      -5,   18,    8,    8,    4,   -3,   -7,  -15,
     -15,   13,    5,    5,   -6,    7,   10,    7,
     -11,    0,   -7,    7,   10,    5,   15,    3,
     -14,    2,    3,    2,   -4,   -9,    4,  -12,
};

static const int k_rook_pst[64] = {
     -13,   -5,   -6,    5,    5,   12,  -23,  -20,
     -40,  -32,  -10,  -13,  -14,   -8,  -25,  -57,
     -29,  -24,  -30,   -7,  -12,   -6,  -16,  -28,
     -17,  -16,   -3,  -15,   -2,  -21,    2,  -16,
       6,    2,    2,   -4,   -6,   -2,    7,  -10,
       4,   13,    6,   -4,   16,   13,   10,   -2,
      13,   28,   20,   13,   24,   -1,   11,   13,
      17,    9,   15,    0,   -3,    4,    2,    1,
};
static const int k_rook_pst_eg[64] = {
      -8,  -16,  -18,  -27,  -26,  -27,   -5,  -10,
     -16,  -13,   -8,  -12,  -20,  -15,  -16,   -7,
       1,    6,    1,   -1,   -4,   -7,    2,    2,
       8,   11,    2,    5,   -5,    7,    1,   -3,
      19,   14,   16,    5,   11,   10,   -3,    9,
      37,   23,   20,   20,   20,   20,   16,    9,
      23,   14,   25,   20,   17,   27,   20,   24,
      14,   19,   10,   12,    7,   23,   13,   13,
};

static const int k_queen_pst[64] = {
     -32,    2,    0,   -1,    5,   -4,    3,   -9,
     -24,  -31,   -2,   -8,    8,    9,   15,   -3,
     -18,   -2,    4,    7,   17,   13,   16,    4,
     -22,   -9,   -2,    7,   10,   -2,   -9,    1,
       0,  -20,  -13,  -10,    1,   -4,   -4,  -27,
     -26,   -3,    9,    6,    4,   12,   12,   -1,
     -16,   16,   15,   12,   25,   10,   -3,   -6,
     -20,  -13,  -14,   -4,  -25,  -10,   -9,  -18,
};
static const int k_queen_pst_eg[64] = {
     -12,    7,    3,    3,    6,    0,    5,   -5,
      -1,    6,    2,   -4,    9,    7,    6,   -2,
      -5,   -5,   -8,    3,    7,    8,    5,    9,
      -9,   -8,   -2,    0,   -1,    9,   -3,   -2,
       4,    3,    8,   14,    1,    6,   -7,    5,
      -7,   -7,   19,   -7,    6,    6,    8,   -6,
     -12,    0,  -21,  -18,  -12,   -4,   -6,   -4,
     -17,  -12,  -10,  -26,   -9,  -11,   -5,  -11,
};

static const int k_king_mid_pst[64] = {
      -4,   21,   47,  -12,   25,  -14,   43,   10,
       3,   -8,    6,   -9,   -5,    2,    3,    7,
       2,    0,  -24,  -13,  -34,   -7,   -5,  -20,
       2,    0,   -5,   -6,   -8,  -15,   -1,  -17,
       5,    2,   -2,    2,   -2,   -8,   -4,    0,
       2,    6,    0,    4,    4,    4,    6,    1,
       0,    5,    1,    2,    1,    1,    2,    1,
      -1,    3,    0,    0,    1,    1,    1,    0,
};

static const int k_king_end_pst[64] = {
     -31,  -28,   -5,    0,  -15,   -6,  -42,  -47,
     -27,   -5,   -9,  -10,  -11,  -13,   -1,  -28,
     -12,    6,  -16,  -23,  -19,  -17,   -8,  -11,
       8,   10,  -11,   -7,  -11,  -21,   17,   -4,
      14,   37,    7,    7,   11,    3,   34,   19,
      25,   28,    4,   23,   17,   15,   22,   28,
      -4,   20,    7,   17,    9,   25,   21,    9,
     -14,    2,   -1,    3,    1,    5,    5,   -3,
};

static inline int mirror_sq(int sq) {
    return sq ^ 56;
}

// Endgame-knowledge switches (UCI HcePawnPstFix/HceKingPst/HceKsFade/
// HcePasser/HceScale), on by default: together +115 Elo on the endgame suite
// and +108 on equal openings vs the all-off engine (10+0.1, 2026-09-27).
// All zero reproduces the pre-switch evaluation.
static int g_opt_pawn_pst_fix = 1;
static int g_opt_king_pst = 1;
static int g_opt_ks_fade = 1;
static int g_opt_passer = 1;
static int g_opt_scale = 1;

bool hce_eval_set_option(const char *name, int value) {
    int *slot = NULL;
    if (strcmp(name, "HcePawnPstFix") == 0) {
        slot = &g_opt_pawn_pst_fix;
    } else if (strcmp(name, "HceKingPst") == 0) {
        slot = &g_opt_king_pst;
    } else if (strcmp(name, "HceKsFade") == 0) {
        slot = &g_opt_ks_fade;
    } else if (strcmp(name, "HcePasser") == 0) {
        slot = &g_opt_passer;
    } else if (strcmp(name, "HceScale") == 0) {
        slot = &g_opt_scale;
    }
    if (slot == NULL) {
        return false;
    }
    *slot = value;
    return true;
}

int hce_eval_get_option(const char *name) {
    if (strcmp(name, "HcePawnPstFix") == 0) return g_opt_pawn_pst_fix;
    if (strcmp(name, "HceKingPst") == 0) return g_opt_king_pst;
    if (strcmp(name, "HceKsFade") == 0) return g_opt_ks_fade;
    if (strcmp(name, "HcePasser") == 0) return g_opt_passer;
    if (strcmp(name, "HceScale") == 0) return g_opt_scale;
    return -1;
}

// The pawn and queen tables are written rank 8 first, but squares count from
// a1, so the side-relative view has to be flipped to read them as drawn.
static inline int pawn_pst_index(int view) {
    return g_opt_pawn_pst_fix ? (view ^ 56) : view;
}

// King tables, drawn rank 8 first (index with view ^ 56). Middlegame values
// are half the classic table because shield/file terms already exist.
static const int k_king_mg_visual[64] = {
     -15,  -20,  -20,  -25,  -25,  -20,  -20,  -15,
     -15,  -20,  -20,  -25,  -25,  -20,  -20,  -15,
     -15,  -20,  -20,  -25,  -25,  -20,  -20,  -15,
     -15,  -20,  -20,  -25,  -25,  -20,  -20,  -15,
     -10,  -15,  -15,  -20,  -20,  -15,  -15,  -10,
      -5,  -10,  -10,  -10,  -10,  -10,  -10,   -5,
      10,   10,    0,    0,    0,    0,   10,   10,
      10,   15,    5,    0,    0,    5,   15,   10,
};
static const int k_king_eg_visual[64] = {
     -50,  -40,  -30,  -20,  -20,  -30,  -40,  -50,
     -30,  -20,  -10,    0,    0,  -10,  -20,  -30,
     -30,  -10,   20,   30,   30,   20,  -10,  -30,
     -30,  -10,   30,   40,   40,   30,  -10,  -30,
     -30,  -10,   30,   40,   40,   30,  -10,  -30,
     -30,  -10,   20,   30,   30,   20,  -10,  -30,
     -30,  -30,    0,    0,    0,    0,  -30,  -30,
     -50,  -30,  -30,  -30,  -30,  -30,  -30,  -50,
};

static inline int square_distance(int a, int b) {
    int df = square_file(a) - square_file(b);
    int dr = square_rank(a) - square_rank(b);
    if (df < 0) df = -df;
    if (dr < 0) dr = -dr;
    return df > dr ? df : dr;
}

static uint64_t knight_attacks_mask(int sq) {
    int f = square_file(sq);
    int r = square_rank(sq);
    static const int df[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    static const int dr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    uint64_t mask = 0;
    for (int i = 0; i < 8; ++i) {
        int nf = f + df[i];
        int nr = r + dr[i];
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) {
            continue;
        }
        mask |= 1ULL << make_square(nf, nr);
    }
    return mask;
}

static uint64_t king_attacks_mask(int sq) {
    int f = square_file(sq);
    int r = square_rank(sq);
    uint64_t mask = 0;
    for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
            if (df == 0 && dr == 0) {
                continue;
            }
            int nf = f + df;
            int nr = r + dr;
            if (nf < 0 || nf > 7 || nr < 0 || nr > 7) {
                continue;
            }
            mask |= 1ULL << make_square(nf, nr);
        }
    }
    return mask;
}

static uint64_t pawn_attacks_mask(int side, int sq) {
    int f = square_file(sq);
    int r = square_rank(sq);
    int step = (side == PIECE_WHITE) ? 1 : -1;
    int nr = r + step;
    if (nr < 0 || nr > 7) {
        return 0;
    }

    uint64_t mask = 0;
    if (f > 0) {
        mask |= 1ULL << make_square(f - 1, nr);
    }
    if (f < 7) {
        mask |= 1ULL << make_square(f + 1, nr);
    }
    return mask;
}






void hce_init_tables(void) {
    if (g_hce_tables_ready) {
        return;
    }
    chess_attack_tables_init();

    for (int sq = 0; sq < 64; ++sq) {
        g_knight_attacks[sq] = knight_attacks_mask(sq);
        g_king_attacks[sq] = king_attacks_mask(sq);
        g_pawn_attacks[PIECE_WHITE][sq] = pawn_attacks_mask(PIECE_WHITE, sq);
        g_pawn_attacks[PIECE_BLACK][sq] = pawn_attacks_mask(PIECE_BLACK, sq);
    }

    for (int file = 0; file < 8; ++file) {
        uint64_t file_mask = 0;
        for (int rank = 0; rank < 8; ++rank) {
            file_mask |= 1ULL << make_square(file, rank);
        }
        g_file_masks[file] = file_mask;
        uint64_t neighbor = 0;
        if (file > 0) {
            neighbor |= g_file_masks[file - 1];
        }
        if (file < 7) {
            neighbor |= file_mask << 1;
        }
        g_neighbor_file_masks[file] = neighbor;
    }

    for (int side = PIECE_WHITE; side <= PIECE_BLACK; ++side) {
        for (int sq = 0; sq < 64; ++sq) {
            int file = square_file(sq);
            int rank = square_rank(sq);
            uint64_t mask = g_file_masks[file] | g_neighbor_file_masks[file];
            uint64_t ranks = 0;
            if (side == PIECE_WHITE) {
                for (int r = rank + 1; r < 8; ++r) {
                    ranks |= 0xFFULL << (r * 8);
                }
            } else {
                for (int r = 0; r < rank; ++r) {
                    ranks |= 0xFFULL << (r * 8);
                }
            }
            g_passed_masks[side][sq] = mask & ranks;
        }
    }


    g_hce_tables_ready = true;
}

uint64_t hce_knight_attacks(int sq) {
    hce_init_tables();
    return g_knight_attacks[sq];
}

uint64_t hce_king_attacks(int sq) {
    hce_init_tables();
    return g_king_attacks[sq];
}

uint64_t hce_pawn_attacks(int side, int sq) {
    hce_init_tables();
    return g_pawn_attacks[side][sq];
}

// Only called from inside the evaluation, after hce_init_tables().
uint64_t hce_attackers_to_square(const GameState *s, int sq, int side) {
    uint64_t occ = s->occ_all;
    uint64_t attackers = 0;
    attackers |= s->bb[side][PIECE_PAWN] & g_pawn_attacks[side ^ 1][sq];
    attackers |= s->bb[side][PIECE_KNIGHT] & g_knight_attacks[sq];
    attackers |= s->bb[side][PIECE_KING] & g_king_attacks[sq];
    uint64_t bishop_like = s->bb[side][PIECE_BISHOP] | s->bb[side][PIECE_QUEEN];
    uint64_t rook_like = s->bb[side][PIECE_ROOK] | s->bb[side][PIECE_QUEEN];
    attackers |= bishop_like & hce_bishop_attacks(sq, occ);
    attackers |= rook_like & hce_rook_attacks(sq, occ);
    return attackers;
}

static bool is_passed_pawn(const GameState *s, int side, int sq) {
    return (g_passed_masks[side][sq] & s->bb[side ^ 1][PIECE_PAWN]) == 0;
}

static bool is_isolated_pawn(const GameState *s, int side, int sq) {
    int file = square_file(sq);
    return (g_neighbor_file_masks[file] & s->bb[side][PIECE_PAWN]) == 0;
}

static bool is_doubled_pawn(const GameState *s, int side, int sq) {
    int file = square_file(sq);
    uint64_t same_file = s->bb[side][PIECE_PAWN] & g_file_masks[file];
    return chess_count_bits(same_file) > 1;
}

static bool square_supported_by_pawn(const GameState *s, int side, int sq) {
    // A pawn of `side` defends sq iff sq is a pawn-attack target from the
    // opposite direction.
    return (s->bb[side][PIECE_PAWN] & g_pawn_attacks[side ^ 1][sq]) != 0;
}

// Union of all squares attacked by `side`, with and without the king's
// attacks. Computed once per eval_side call and reused by the tactical
// penalty terms so they only pay for full attacker sets on attacked squares.
typedef struct AttackUnions {
    uint64_t all[PIECE_COLOR_COUNT];
    uint64_t non_king[PIECE_COLOR_COUNT];
    uint64_t pawn[PIECE_COLOR_COUNT];
    // Per-piece-type attack unions (knight/bishop/rook/queen only; the rest
    // stay zero). Used for safe-check counting in king safety.
    uint64_t piece_atk[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT];
    int mobility[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT];
    // Mobility restricted to squares the enemy's pawns do not attack.
    int mobility_safe[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT];
    int king_attack_units[PIECE_COLOR_COUNT];
} AttackUnions;

static void compute_attack_unions(const GameState *s, AttackUnions *out) {
    memset(out, 0, sizeof(*out));
    // Pawn attack maps for both sides first: each side's safe mobility needs
    // the other side's pawn attacks.
    for (int side = PIECE_WHITE; side <= PIECE_BLACK; ++side) {
        uint64_t pawns = s->bb[side][PIECE_PAWN];
        out->pawn[side] = (side == PIECE_WHITE)
                              ? (((pawns & ~g_file_masks[0]) << 7) | ((pawns & ~g_file_masks[7]) << 9))
                              : (((pawns & ~g_file_masks[0]) >> 9) | ((pawns & ~g_file_masks[7]) >> 7));
    }
    for (int side = PIECE_WHITE; side <= PIECE_BLACK; ++side) {
        int enemy_king_sq = chess_find_king_square(s, side ^ 1);
        uint64_t enemy_king_zone = enemy_king_sq >= 0
                                       ? (g_king_attacks[enemy_king_sq] | (1ULL << enemy_king_sq))
                                       : 0;
        uint64_t safe_from_pawns = ~out->pawn[side ^ 1];
        uint64_t pawns = s->bb[side][PIECE_PAWN];
        uint64_t a = out->pawn[side];
        uint64_t pawn_scan = pawns;
        while (pawn_scan != 0) {
            int sq = chess_pop_lsb(&pawn_scan);
            if ((g_pawn_attacks[side][sq] & enemy_king_zone) != 0) {
                out->king_attack_units[side] += 1;
            }
        }
        uint64_t knights = s->bb[side][PIECE_KNIGHT];
        while (knights != 0) {
            int sq = chess_pop_lsb(&knights);
            uint64_t attacks = g_knight_attacks[sq];
            a |= attacks;
            out->piece_atk[side][PIECE_KNIGHT] |= attacks;
            uint64_t open = attacks & ~s->occ[side];
            out->mobility[side][PIECE_KNIGHT] += chess_count_bits(open);
            out->mobility_safe[side][PIECE_KNIGHT] += chess_count_bits(open & safe_from_pawns);
            if ((attacks & enemy_king_zone) != 0) {
                out->king_attack_units[side] += 2;
            }
        }
        uint64_t bishops = s->bb[side][PIECE_BISHOP];
        while (bishops != 0) {
            int sq = chess_pop_lsb(&bishops);
            uint64_t attacks = hce_bishop_attacks(sq, s->occ_all);
            a |= attacks;
            out->piece_atk[side][PIECE_BISHOP] |= attacks;
            uint64_t open = attacks & ~s->occ[side];
            out->mobility[side][PIECE_BISHOP] += chess_count_bits(open);
            out->mobility_safe[side][PIECE_BISHOP] += chess_count_bits(open & safe_from_pawns);
            if ((attacks & enemy_king_zone) != 0) {
                out->king_attack_units[side] += 2;
            }
        }
        uint64_t rooks = s->bb[side][PIECE_ROOK];
        while (rooks != 0) {
            int sq = chess_pop_lsb(&rooks);
            uint64_t attacks = hce_rook_attacks(sq, s->occ_all);
            a |= attacks;
            out->piece_atk[side][PIECE_ROOK] |= attacks;
            uint64_t open = attacks & ~s->occ[side];
            out->mobility[side][PIECE_ROOK] += chess_count_bits(open);
            out->mobility_safe[side][PIECE_ROOK] += chess_count_bits(open & safe_from_pawns);
            if ((attacks & enemy_king_zone) != 0) {
                out->king_attack_units[side] += 3;
            }
        }
        uint64_t queens = s->bb[side][PIECE_QUEEN];
        while (queens != 0) {
            int sq = chess_pop_lsb(&queens);
            uint64_t attacks = hce_bishop_attacks(sq, s->occ_all) |
                               hce_rook_attacks(sq, s->occ_all);
            a |= attacks;
            out->piece_atk[side][PIECE_QUEEN] |= attacks;
            uint64_t open = attacks & ~s->occ[side];
            out->mobility[side][PIECE_QUEEN] += chess_count_bits(open);
            out->mobility_safe[side][PIECE_QUEEN] += chess_count_bits(open & safe_from_pawns);
            if ((attacks & enemy_king_zone) != 0) {
                out->king_attack_units[side] += 5;
            }
        }
        out->non_king[side] = a;
        uint64_t king = s->bb[side][PIECE_KING];
        while (king != 0) {
            a |= g_king_attacks[chess_pop_lsb(&king)];
        }
        out->all[side] = a;
    }
}

static int king_shield_penalty(const GameState *s, int side) {
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return 0;
    }

    int penalty = 0;
    int file = square_file(king_sq);
    int rank = square_rank(king_sq);
    int forward = (side == PIECE_WHITE) ? 1 : -1;
    for (int df = -1; df <= 1; ++df) {
        int nf = file + df;
        if (nf < 0 || nf > 7) {
            continue;
        }
        int nr = rank + forward;
        bool found = false;
        if (nr >= 0 && nr < 8) {
            int nsq = make_square(nf, nr);
            found = (s->bb[side][PIECE_PAWN] & (1ULL << nsq)) != 0;
        }
        if (!found) {
            penalty += 10;
        }
    }
    return penalty;
}

static int king_file_pressure_penalty(const GameState *s, int side) {
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return 0;
    }

    int enemy = side ^ 1;
    int king_file = square_file(king_sq);
    int penalty = 0;
    for (int df = -1; df <= 1; ++df) {
        int file = king_file + df;
        if (file < 0 || file > 7) {
            continue;
        }
        bool own_pawn = (g_file_masks[file] & s->bb[side][PIECE_PAWN]) != 0;
        bool enemy_pawn = (g_file_masks[file] & s->bb[enemy][PIECE_PAWN]) != 0;
        bool enemy_heavy = (g_file_masks[file] & (s->bb[enemy][PIECE_ROOK] | s->bb[enemy][PIECE_QUEEN])) != 0;
        if (!own_pawn && enemy_heavy) {
            penalty += enemy_pawn ? 12 : 22;
        } else if (!own_pawn) {
            penalty += enemy_pawn ? 4 : 8;
        }
    }
    return penalty;
}

static int least_attacker_value(const GameState *s, uint64_t attackers, int side) {
    if (s == NULL || attackers == 0) {
        return HCE_INF;
    }
    if ((attackers & s->bb[side][PIECE_PAWN]) != 0) {
        return hce_piece_value[PIECE_PAWN];
    }
    if ((attackers & s->bb[side][PIECE_KNIGHT]) != 0) {
        return hce_piece_value[PIECE_KNIGHT];
    }
    if ((attackers & s->bb[side][PIECE_BISHOP]) != 0) {
        return hce_piece_value[PIECE_BISHOP];
    }
    if ((attackers & s->bb[side][PIECE_ROOK]) != 0) {
        return hce_piece_value[PIECE_ROOK];
    }
    if ((attackers & s->bb[side][PIECE_QUEEN]) != 0) {
        return hce_piece_value[PIECE_QUEEN];
    }
    if ((attackers & s->bb[side][PIECE_KING]) != 0) {
        return 2000;
    }
    return HCE_INF;
}

// Count enemy pieces able to give check from squares this side does not
// defend and the enemy does not occupy. out = {knight, bishop, rook, queen}.
static void safe_check_counts(const GameState *s,
                              int side,
                              const AttackUnions *atk,
                              int out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0;
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return;
    }
    int enemy = side ^ 1;
    uint64_t safe = ~atk->all[side] & ~s->occ[enemy];
    uint64_t n_check = g_knight_attacks[king_sq];
    uint64_t b_check = hce_bishop_attacks(king_sq, s->occ_all);
    uint64_t r_check = hce_rook_attacks(king_sq, s->occ_all);
    out[0] = chess_count_bits(n_check & atk->piece_atk[enemy][PIECE_KNIGHT] & safe);
    out[1] = chess_count_bits(b_check & atk->piece_atk[enemy][PIECE_BISHOP] & safe);
    out[2] = chess_count_bits(r_check & atk->piece_atk[enemy][PIECE_ROOK] & safe);
    out[3] = chess_count_bits((b_check | r_check) & atk->piece_atk[enemy][PIECE_QUEEN] & safe);
}

static int king_safety_penalty(const GameState *s, int side, const AttackUnions *atk) {
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return 0;
    }

    int enemy = side ^ 1;
    int enemy_queen_count = chess_count_bits(s->bb[enemy][PIECE_QUEEN]);
    int enemy_rook_count = chess_count_bits(s->bb[enemy][PIECE_ROOK]);
    int attack_units = atk->king_attack_units[enemy];
    int shield = king_shield_penalty(s, side);
    int file_pressure = king_file_pressure_penalty(s, side);
    int home_distance = (side == PIECE_WHITE) ? square_rank(king_sq) : (7 - square_rank(king_sq));
    int penalty = 0;

    if (home_distance > 1) {
        shield = (shield * 2) / 3;
    }
    if (home_distance > 2) {
        shield /= 2;
    }
    if (enemy_queen_count == 0 && enemy_rook_count == 0) {
        file_pressure /= 2;
    }

    penalty += (attack_units > 0) ? shield : (shield / 2);
    penalty += file_pressure;

    if (attack_units > 0) {
        int attack_scale = (enemy_queen_count > 0) ? 5 : 4;
        penalty += attack_units * attack_scale;
        if (attack_units >= 6) {
            penalty += shield / 2;
        }
    }

    if (enemy_queen_count == 0 && enemy_rook_count <= 1 && attack_units <= 2) {
        penalty = (penalty * 3) / 4;
    }

    return penalty;
}

static int hanging_piece_penalty(const GameState *s, int side, const AttackUnions *atk) {
    int penalty = 0;
    static const int scan_order[] = {PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT};
    for (size_t i = 0; i < sizeof(scan_order) / sizeof(scan_order[0]); ++i) {
        int piece = scan_order[i];
        uint64_t bb = s->bb[side][piece] & atk->all[side ^ 1];
        while (bb != 0) {
            int sq = chess_pop_lsb(&bb);
            uint64_t attackers = hce_attackers_to_square(s, sq, side ^ 1);
            if (attackers == 0) {
                continue;
            }
            uint64_t defenders = hce_attackers_to_square(s, sq, side);
            int atk_count = chess_count_bits(attackers);
            int def_count = chess_count_bits(defenders);
            int least_atk = least_attacker_value(s, attackers, side ^ 1);
            int least_def = least_attacker_value(s, defenders, side);
            int base_penalty = (piece == PIECE_QUEEN) ? 36 :
                               (piece == PIECE_ROOK) ? 18 :
                               14;
            if (defenders == 0) {
                int multiplier = (least_atk <= hce_piece_value[piece]) ? 2 : 1;
                if (piece == PIECE_QUEEN && least_atk <= hce_piece_value[piece]) {
                    multiplier = 4;
                } else if (piece == PIECE_ROOK && least_atk <= hce_piece_value[piece]) {
                    multiplier = 3;
                }
                penalty += base_penalty * multiplier;
                continue;
            }
            bool overloaded = atk_count > def_count;
            bool cheap_pressure = (least_atk + 60) < least_def;
            if (overloaded) {
                penalty += base_penalty;
            } else if (cheap_pressure) {
                penalty += base_penalty / 2;
            }
        }
    }
    return penalty;
}

static int queen_trap_penalty(const GameState *s, int side, const AttackUnions *atk) {
    if (s == NULL) {
        return 0;
    }

    int enemy = side ^ 1;
    int penalty = 0;
    uint64_t queens = s->bb[side][PIECE_QUEEN] & atk->all[enemy];
    while (queens != 0) {
        int sq = chess_pop_lsb(&queens);
        uint64_t attackers = hce_attackers_to_square(s, sq, enemy);
        if (attackers == 0) {
            continue;
        }

        uint64_t defenders = hce_attackers_to_square(s, sq, side);
        uint64_t mobility = (hce_rook_attacks(sq, s->occ_all) |
                             hce_bishop_attacks(sq, s->occ_all)) &
                            ~s->occ[side];
        int safe_escapes = chess_count_bits(mobility & ~atk->non_king[enemy]);

        int trap = 0;
        if (safe_escapes <= 1) {
            trap += 260;
        } else if (safe_escapes == 2) {
            trap += 170;
        } else if (safe_escapes <= 4) {
            trap += 90;
        } else if (safe_escapes <= 6) {
            trap += 40;
        }

        int least_atk = least_attacker_value(s, attackers, enemy);
        if (defenders == 0 && least_atk <= hce_piece_value[PIECE_QUEEN]) {
            trap += 60;
        }
        if (s->side_to_move == enemy && least_atk <= hce_piece_value[PIECE_QUEEN]) {
            int atk_count = chess_count_bits(attackers);
            int def_count = chess_count_bits(defenders);
            if (defenders == 0) {
                trap += 300;
            } else if (least_atk + 80 < least_attacker_value(s, defenders, side)) {
                trap += 160;
            } else if (atk_count > def_count) {
                trap += 100;
            }
        }

        int file = square_file(sq);
        int rank = square_rank(sq);
        if ((file == 0 || file == 7 || rank == 0 || rank == 7) && safe_escapes <= 2) {
            trap += 30;
        }

        penalty += trap;
    }
    return penalty;
}

static int phase_value(const GameState *s) {
    int phase = 0;
    for (int side = PIECE_WHITE; side <= PIECE_BLACK; ++side) {
        for (int piece = PIECE_QUEEN; piece <= PIECE_PAWN; ++piece) {
            phase += chess_count_bits(s->bb[side][piece]) * hce_phase_inc[piece];
        }
    }
    if (phase > 24) {
        phase = 24;
    }
    return phase;
}

typedef struct EvalTermPair {
    int mg;
    int eg;
} EvalTermPair;

typedef struct EvalSideTerms {
    EvalTermPair material;
    EvalTermPair piece_square;
    EvalTermPair pawn_structure;
    EvalTermPair passed_pawns;
    EvalTermPair rook_files;
    EvalTermPair mobility;
    EvalTermPair pawn_activity;
    EvalTermPair king_safety_penalty;
    EvalTermPair hanging_penalty;
    EvalTermPair queen_trap_penalty;
    EvalTermPair endgame_extra;
} EvalSideTerms;

static const int k_passed_mg_scale = -23;
static const int k_passed_eg_scale = 88;
static const int k_king_mg_scale = -119;
static const int k_king_eg_scale = -236;
static const int k_hanging_mg_scale = -22;
static const int k_hanging_eg_scale = -74;
static const int k_queen_mg_scale = -14;
static const int k_queen_eg_scale = -60;
static const int k_pawn_push_mg = 18;
static const int k_pawn_push_eg = 11;
static const int k_pawn_threat_minor_mg = 61;
static const int k_pawn_threat_minor_eg = 10;
static const int k_pawn_threat_major_mg = 2;
static const int k_pawn_threat_major_eg = 18;
// Named copies of eval literals that appear in both the cached pawn path and
// the feature-dump path, so texel_apply_tune.py can patch one definition.
static const int k_iso_mg = -18;
static const int k_iso_eg = -19;
static const int k_dbl_mg = -12;
static const int k_dbl_eg = -2;
static const int k_mob_n_mg = -11;
static const int k_mob_n_eg = -3;
static const int k_mob_b_mg = 2;
static const int k_mob_b_eg = 6;
static const int k_mob_r_mg = -6;
static const int k_mob_r_eg = 5;
static const int k_mob_q_mg = -9;
static const int k_mob_q_eg = 17;
static const int k_rook_open_mg = 59;
static const int k_rook_open_eg = -11;
static const int k_rook_semi_mg = 30;
static const int k_rook_semi_eg = 8;
// Stage-B feature weights: zero until texel-fitted, so the engine plays
// identically to the pre-stage-B build while the tunedump exposes the counts.
static const int k_safe_check_n_mg = -58;
static const int k_safe_check_n_eg = 2;
static const int k_safe_check_b_mg = -22;
static const int k_safe_check_b_eg = -19;
static const int k_safe_check_r_mg = -58;
static const int k_safe_check_r_eg = -12;
static const int k_safe_check_q_mg = -36;
static const int k_safe_check_q_eg = -14;
static const int k_bishop_pair_mg = 53;
static const int k_bishop_pair_eg = 55;
static const int k_mob_safe_n_mg = 21;
static const int k_mob_safe_n_eg = 2;
static const int k_mob_safe_b_mg = 10;
static const int k_mob_safe_b_eg = -2;
static const int k_mob_safe_r_mg = 10;
static const int k_mob_safe_r_eg = 4;
static const int k_mob_safe_q_mg = 10;
static const int k_mob_safe_q_eg = -13;
static const int k_passer_rank_mg[6] = {-31, -20, -24, 1, 22, 38};
static const int k_passer_rank_eg[6] = {2, -12, -9, -14, 19, 24};

#define HCE_PAWN_CACHE_BITS 16u
#define HCE_PAWN_CACHE_SIZE (1u << HCE_PAWN_CACHE_BITS)
#define HCE_PAWN_CACHE_MASK (HCE_PAWN_CACHE_SIZE - 1u)

typedef struct PawnEvalTerms {
    uint64_t passers;
    EvalTermPair material;
    EvalTermPair piece_square;
    EvalTermPair pawn_structure;
    EvalTermPair passed_pawns;
    EvalTermPair pawn_activity;
} PawnEvalTerms;

typedef struct PawnEvalCacheEntry {
    uint64_t white_pawns;
    uint64_t black_pawns;
    PawnEvalTerms side[PIECE_COLOR_COUNT];
    int flags;
    bool valid;
} PawnEvalCacheEntry;

// Thread-local: lazy-SMP helper threads each get their own pawn cache, so a
// torn concurrent write can never hand one thread another position's terms.
static _Thread_local PawnEvalCacheEntry g_pawn_eval_cache[HCE_PAWN_CACHE_SIZE];

static inline void eval_term_add(EvalTermPair *term, int mg, int eg) {
    if (term == NULL) {
        return;
    }
    term->mg += mg;
    term->eg += eg;
}

static inline int eval_term_blend(EvalTermPair term, int phase) {
    return (term.mg * phase + term.eg * (24 - phase)) / 24;
}

static void compute_pawn_eval_terms(const GameState *s, int side, PawnEvalTerms *out) {
    memset(out, 0, sizeof(*out));
    int enemy = side ^ 1;
    int enemy_pawn_min_file = 8;
    int enemy_pawn_max_file = -1;
    uint64_t enemy_pawns = s->bb[enemy][PIECE_PAWN];
    while (enemy_pawns != 0) {
        int file = square_file(chess_pop_lsb(&enemy_pawns));
        if (file < enemy_pawn_min_file) {
            enemy_pawn_min_file = file;
        }
        if (file > enemy_pawn_max_file) {
            enemy_pawn_max_file = file;
        }
    }

    uint64_t pawns = s->bb[side][PIECE_PAWN];
    while (pawns != 0) {
        int sq = chess_pop_lsb(&pawns);
        int view = (side == PIECE_WHITE) ? sq : mirror_sq(sq);
        eval_term_add(&out->material,
                      hce_piece_value[PIECE_PAWN],
                      hce_piece_value[PIECE_PAWN]);
        int pidx = pawn_pst_index(view);
        eval_term_add(&out->piece_square, k_pawn_pst[pidx], k_pawn_pst_eg[pidx]);
        // The free-push bonus depends on every piece, not just pawns, so it is
        // added per position in eval_side rather than cached here.
        if (is_isolated_pawn(s, side, sq)) {
            eval_term_add(&out->pawn_structure, k_iso_mg, k_iso_eg);
        }
        if (is_doubled_pawn(s, side, sq)) {
            eval_term_add(&out->pawn_structure, k_dbl_mg, k_dbl_eg);
        }
        if (!is_passed_pawn(s, side, sq)) {
            continue;
        }
        out->passers |= 1ULL << sq;

        int file = square_file(sq);
        int advance = (side == PIECE_WHITE) ? square_rank(sq) : (7 - square_rank(sq));
        int passer_mg = 18 + advance * 5;
        int passer_eg = 28 + advance * 8;
        if (enemy_pawn_max_file >= 0 &&
            (file <= enemy_pawn_min_file - 3 || file >= enemy_pawn_max_file + 3)) {
            passer_mg += 10;
            passer_eg += 28;
        }
        int front_sq = sq + ((side == PIECE_WHITE) ? 8 : -8);
        if (front_sq >= 0 && front_sq < 64 && square_supported_by_pawn(s, side, front_sq)) {
            passer_mg += 4;
            passer_eg += 8;
        }
        int mg_cap = 30 + advance * 6;
        int eg_cap = 44 + advance * 10;
        if (passer_mg > mg_cap) {
            passer_mg = mg_cap;
        }
        if (passer_eg > eg_cap) {
            passer_eg = eg_cap;
        }
        eval_term_add(&out->passed_pawns, passer_mg, passer_eg);
        int adv_idx = advance - 1;
        if (adv_idx >= 0 && adv_idx < 6) {
            // Outside passed_pawns so the passed-pawn scale does not touch
            // it: texel_tune.py models these as plain linear terms.
            eval_term_add(&out->pawn_activity,
                          k_passer_rank_mg[adv_idx],
                          k_passer_rank_eg[adv_idx]);
        }
    }
}

static const PawnEvalTerms *probe_pawn_eval_terms(const GameState *s, int side) {
    uint64_t white = s->bb[PIECE_WHITE][PIECE_PAWN];
    uint64_t black = s->bb[PIECE_BLACK][PIECE_PAWN];
    uint64_t mixed = white ^ ((black << 1) | (black >> 63));
    mixed ^= mixed >> 32;
    mixed *= 0x9e3779b97f4a7c15ULL;
    // Index by the product's high bits: its low 16 bits depend only on the
    // low 16 bits of `mixed` (ranks 1-2 and 5-6), so structures differing on
    // ranks 3-4 or 7-8 all shared one slot.
    PawnEvalCacheEntry *entry = &g_pawn_eval_cache[mixed >> (64u - HCE_PAWN_CACHE_BITS)];
    if (!entry->valid || entry->white_pawns != white || entry->black_pawns != black ||
        entry->flags != g_opt_pawn_pst_fix) {
        entry->flags = g_opt_pawn_pst_fix;
        entry->white_pawns = white;
        entry->black_pawns = black;
        compute_pawn_eval_terms(s, PIECE_WHITE, &entry->side[PIECE_WHITE]);
        compute_pawn_eval_terms(s, PIECE_BLACK, &entry->side[PIECE_BLACK]);
        entry->valid = true;
    }
    return &entry->side[side];
}

static int side_phase_material(const GameState *s, int side) {
    return 4 * chess_count_bits(s->bb[side][PIECE_QUEEN]) +
           2 * chess_count_bits(s->bb[side][PIECE_ROOK]) +
           chess_count_bits(s->bb[side][PIECE_BISHOP]) +
           chess_count_bits(s->bb[side][PIECE_KNIGHT]);
}

// King-danger multiplier in percent: full with an enemy queen, otherwise
// shrinking with the enemy's remaining pieces (a lone rook gives 25%).
static int king_danger_fade_pct(const GameState *s, int side) {
    int enemy = side ^ 1;
    if (!g_opt_ks_fade || s->bb[enemy][PIECE_QUEEN] != 0) {
        return 100;
    }
    int ph = side_phase_material(s, enemy);
    return ph >= 8 ? 100 : ph * 100 / 8;
}

// Passed-pawn knowledge that depends on kings, pieces and attacks, so it
// cannot live in the pawn cache: king proximity to the stop square, free
// path, pawn support, rook behind the passer, and the rule of the square.
static void passer_extra_terms(const GameState *s,
                               int side,
                               uint64_t passers,
                               const AttackUnions *atk,
                               EvalTermPair *out) {
    int enemy = side ^ 1;
    int own_king = chess_find_king_square(s, side);
    int enemy_king = chess_find_king_square(s, enemy);
    if (own_king < 0 || enemy_king < 0) {
        return;
    }
    int up = (side == PIECE_WHITE) ? 8 : -8;
    bool enemy_pawns_only = side_phase_material(s, enemy) == 0;
    int best_unstoppable = 0;
    while (passers != 0) {
        int sq = chess_pop_lsb(&passers);
        int file = square_file(sq);
        int r = (side == PIECE_WHITE) ? square_rank(sq) : (7 - square_rank(sq));
        int block = sq + up;
        if (block < 0 || block >= 64) {
            continue;
        }
        int promo = make_square(file, side == PIECE_WHITE ? 7 : 0);
        if (r >= 3) {
            int w = 5 * r - 13;
            int their_d = square_distance(enemy_king, block);
            int our_d = square_distance(own_king, block);
            if (their_d > 5) their_d = 5;
            if (our_d > 5) our_d = 5;
            int prox = (their_d * 19 / 4 - our_d * 2) * w;
            if (r < 6) {
                int next = block + up;
                int d2 = square_distance(own_king, next);
                prox -= (d2 > 5 ? 5 : d2) * w;
            }
            eval_term_add(out, 0, prox / 2);
            if (s->sq_piece[block] == PIECE_NONE) {
                uint64_t path = 0;
                for (int t = block; t >= 0 && t < 64; t += up) {
                    path |= 1ULL << t;
                }
                int k = 0;
                if ((path & (atk->all[enemy] | s->occ_all)) == 0) {
                    k = 18;
                } else if (((1ULL << block) & atk->all[enemy]) == 0) {
                    k = 6;
                }
                if (((1ULL << block) & atk->all[side]) != 0) {
                    k += 3;
                }
                eval_term_add(out, k * w / 3, k * w);
            }
        }
        bool supported = (s->bb[side][PIECE_PAWN] & g_pawn_attacks[enemy][sq]) != 0;
        uint64_t phalanx = s->bb[side][PIECE_PAWN] & g_neighbor_file_masks[file] &
                           (0xFFULL << (square_rank(sq) * 8));
        if (supported || phalanx != 0) {
            eval_term_add(out, 2 * r, 6 * r);
        }
        uint64_t file_mask = g_file_masks[file];
        uint64_t rooks_behind = file_mask & (side == PIECE_WHITE
                                                 ? ((1ULL << sq) - 1ULL)
                                                 : ~((1ULL << sq) | ((1ULL << sq) - 1ULL)));
        uint64_t own_rooks = s->bb[side][PIECE_ROOK] & rooks_behind;
        uint64_t enemy_rooks = s->bb[enemy][PIECE_ROOK] & rooks_behind;
        uint64_t pawn_sees = hce_rook_attacks(sq, s->occ_all);
        if ((own_rooks & pawn_sees) != 0) {
            eval_term_add(out, 5, 12 + 2 * r);
        }
        if ((enemy_rooks & pawn_sees) != 0) {
            eval_term_add(out, -5, -(10 + 2 * r));
        }
        if (enemy_pawns_only) {
            int pawn_d = 7 - r;
            if (r == 1) {
                pawn_d -= 1;
            }
            int king_d = square_distance(enemy_king, promo) - (s->side_to_move == enemy ? 1 : 0);
            uint64_t ahead = 0;
            for (int t = block; t >= 0 && t < 64; t += up) {
                ahead |= 1ULL << t;
            }
            bool own_blocks = (ahead & s->occ[side]) != 0;
            if (king_d > pawn_d && !own_blocks) {
                int v = 500 + 20 * r;
                if (v > best_unstoppable) {
                    best_unstoppable = v;
                }
            }
        }
    }
    eval_term_add(out, 0, best_unstoppable);
}

// Scale (out of 64) for endings the raw material count misjudges.
static int endgame_scale64(const GameState *s, int strong) {
    int weak = strong ^ 1;
    int strong_pawns = chess_count_bits(s->bb[strong][PIECE_PAWN]);
    int weak_pawns = chess_count_bits(s->bb[weak][PIECE_PAWN]);
    int npm[2];
    for (int c = 0; c < 2; ++c) {
        npm[c] = chess_count_bits(s->bb[c][PIECE_QUEEN]) * hce_piece_value[PIECE_QUEEN] +
                 chess_count_bits(s->bb[c][PIECE_ROOK]) * hce_piece_value[PIECE_ROOK] +
                 chess_count_bits(s->bb[c][PIECE_BISHOP]) * hce_piece_value[PIECE_BISHOP] +
                 chess_count_bits(s->bb[c][PIECE_KNIGHT]) * hce_piece_value[PIECE_KNIGHT];
    }
    if (strong_pawns == 0) {
        uint64_t strong_minors_only = s->bb[strong][PIECE_QUEEN] | s->bb[strong][PIECE_ROOK] |
                                      s->bb[strong][PIECE_BISHOP];
        if (npm[weak] == 0 && strong_minors_only == 0 &&
            chess_count_bits(s->bb[strong][PIECE_KNIGHT]) <= 2) {
            return 0;
        }
        if (npm[strong] - npm[weak] <= hce_piece_value[PIECE_BISHOP]) {
            return npm[strong] < hce_piece_value[PIECE_ROOK] ? 0 : 6;
        }
    }
    int sc = 64;
    if (chess_count_bits(s->bb[strong][PIECE_BISHOP]) == 1 &&
        chess_count_bits(s->bb[weak][PIECE_BISHOP]) == 1) {
        static const uint64_t k_dark = 0xAA55AA55AA55AA55ULL;
        bool sd = (s->bb[strong][PIECE_BISHOP] & k_dark) != 0;
        bool wd = (s->bb[weak][PIECE_BISHOP] & k_dark) != 0;
        if (sd != wd) {
            bool bishops_only = npm[strong] == hce_piece_value[PIECE_BISHOP] &&
                                npm[weak] == hce_piece_value[PIECE_BISHOP];
            sc = bishops_only ? 24 : 48;
        }
    }
    if (strong_pawns == 1 && npm[strong] <= npm[weak] && npm[weak] > 0 && sc > 40) {
        sc = 40;
    }
    if (strong_pawns - weak_pawns == 1 && weak_pawns > 0 && sc == 64 &&
        npm[strong] == hce_piece_value[PIECE_ROOK] && npm[weak] == hce_piece_value[PIECE_ROOK]) {
        uint64_t all = s->bb[PIECE_WHITE][PIECE_PAWN] | s->bb[PIECE_BLACK][PIECE_PAWN];
        static const uint64_t k_queenside = 0x0F0F0F0F0F0F0F0FULL;
        if ((all & k_queenside) == 0 || (all & ~k_queenside) == 0) {
            sc = 44;
        }
    }
    return sc;
}

static int eval_side(const GameState *s,
                     int side,
                     int phase,
                     const AttackUnions *attack_unions,
                     ChessEvalSideBreakdown *out_breakdown,
                     HceTuneFeatures *feat) {
    EvalSideTerms terms;
    memset(&terms, 0, sizeof(terms));
    int enemy = side ^ 1;
    bool use_pawn_cache = feat == NULL;
    uint64_t passers = 0;
    int enemy_pawn_min_file = 8;
    int enemy_pawn_max_file = -1;
    if (use_pawn_cache) {
        const PawnEvalTerms *pawn_terms = probe_pawn_eval_terms(s, side);
        terms.material = pawn_terms->material;
        terms.piece_square = pawn_terms->piece_square;
        terms.pawn_structure = pawn_terms->pawn_structure;
        terms.passed_pawns = pawn_terms->passed_pawns;
        terms.pawn_activity = pawn_terms->pawn_activity;
        // Pawns whose push square is empty (outside the pawn-structure cache:
        // it was keyed by pawns only, so pieces moving in front left it stale).
        uint64_t own_pawns = s->bb[side][PIECE_PAWN];
        uint64_t pushable = side == PIECE_WHITE ? ((own_pawns << 8) & ~s->occ_all)
                                                : ((own_pawns >> 8) & ~s->occ_all);
        int pushes = chess_count_bits(pushable);
        eval_term_add(&terms.pawn_activity, pushes * k_pawn_push_mg, pushes * k_pawn_push_eg);
        passers = pawn_terms->passers;
    } else {
        uint64_t scan = s->bb[side][PIECE_PAWN];
        while (scan != 0) {
            int psq = chess_pop_lsb(&scan);
            if (is_passed_pawn(s, side, psq)) {
                passers |= 1ULL << psq;
            }
        }
        uint64_t enemy_pawns_scan = s->bb[enemy][PIECE_PAWN];
        while (enemy_pawns_scan != 0) {
            int sq = chess_pop_lsb(&enemy_pawns_scan);
            int file = square_file(sq);
            if (file < enemy_pawn_min_file) {
                enemy_pawn_min_file = file;
            }
            if (file > enemy_pawn_max_file) {
                enemy_pawn_max_file = file;
            }
        }
    }

    for (int piece = PIECE_KING; piece <= PIECE_PAWN; ++piece) {
        if (piece == PIECE_PAWN && use_pawn_cache) {
            continue;
        }
        uint64_t bb = s->bb[side][piece];
        while (bb != 0) {
            int sq = chess_pop_lsb(&bb);
            int view = (side == PIECE_WHITE) ? sq : mirror_sq(sq);
            eval_term_add(&terms.material, hce_piece_value[piece], hce_piece_value[piece]);
            if (feat != NULL) {
                feat->mat[piece] += 1;
                feat->pst[piece][(piece == PIECE_PAWN || piece == PIECE_QUEEN) ? pawn_pst_index(view) : view] += 1;
            }
            switch (piece) {
                case PIECE_PAWN:
                    eval_term_add(&terms.piece_square,
                                  k_pawn_pst[pawn_pst_index(view)],
                                  k_pawn_pst_eg[pawn_pst_index(view)]);
                    {
                        int front_sq = sq + ((side == PIECE_WHITE) ? 8 : -8);
                        if (front_sq >= 0 && front_sq < 64 &&
                            s->sq_piece[front_sq] == PIECE_NONE) {
                            eval_term_add(&terms.pawn_activity,
                                          k_pawn_push_mg,
                                          k_pawn_push_eg);
                            if (feat != NULL) {
                                feat->pawn_pushes += 1;
                            }
                        }
                    }
                    if (is_isolated_pawn(s, side, sq)) {
                        eval_term_add(&terms.pawn_structure, k_iso_mg, k_iso_eg);
                    if (feat != NULL) {
                        feat->isolated += 1;
                        }
                    }
                    if (is_doubled_pawn(s, side, sq)) {
                        eval_term_add(&terms.pawn_structure, k_dbl_mg, k_dbl_eg);
                    if (feat != NULL) {
                        feat->doubled += 1;
                        }
                    }
                    if (is_passed_pawn(s, side, sq)) {
                        int file = square_file(sq);
                        int advance = (side == PIECE_WHITE) ? square_rank(sq) : (7 - square_rank(sq));
                        int passer_mg = 18 + advance * 5;
                        int passer_eg = 28 + advance * 8;
                        if (enemy_pawn_max_file >= 0 &&
                            (file <= enemy_pawn_min_file - 3 || file >= enemy_pawn_max_file + 3)) {
                            passer_mg += 10;
                            passer_eg += 28;
                        }
                        int front_sq = sq + ((side == PIECE_WHITE) ? 8 : -8);
                        if (front_sq >= 0 && front_sq < 64) {
                            if (square_supported_by_pawn(s, side, front_sq)) {
                                passer_mg += 4;
                                passer_eg += 8;
                            }
                        }
                        int mg_cap = 30 + advance * 6;
                        int eg_cap = 44 + advance * 10;
                        if (passer_mg > mg_cap) {
                            passer_mg = mg_cap;
                        }
                        if (passer_eg > eg_cap) {
                            passer_eg = eg_cap;
                        }
                        eval_term_add(&terms.passed_pawns, passer_mg, passer_eg);
                        if (feat != NULL) {
                            feat->passed_mg += passer_mg;
                            feat->passed_eg += passer_eg;
                        }
                        int adv_idx = advance - 1;
                        if (adv_idx >= 0 && adv_idx < 6) {
                            eval_term_add(&terms.pawn_activity,
                                          k_passer_rank_mg[adv_idx],
                                          k_passer_rank_eg[adv_idx]);
                            if (feat != NULL) {
                                feat->passer_rank[adv_idx] += 1;
                            }
                        }
                    }
                    break;
                case PIECE_KNIGHT: {
                    eval_term_add(&terms.piece_square, k_knight_pst[view], k_knight_pst_eg[view]);
                    break;
                }
                case PIECE_BISHOP: {
                    eval_term_add(&terms.piece_square, k_bishop_pst[view], k_bishop_pst_eg[view]);
                    break;
                }
                case PIECE_ROOK: {
                    eval_term_add(&terms.piece_square, k_rook_pst[view], k_rook_pst_eg[view]);
                    int file = square_file(sq);
                    bool own_pawn = (g_file_masks[file] & s->bb[side][PIECE_PAWN]) != 0;
                    bool enemy_pawn = (g_file_masks[file] & s->bb[enemy][PIECE_PAWN]) != 0;
                    if (!own_pawn && !enemy_pawn) {
                        eval_term_add(&terms.rook_files, k_rook_open_mg, k_rook_open_eg);
                        if (feat != NULL) {
                            feat->rook_open += 1;
                        }
                    } else if (!own_pawn) {
                        eval_term_add(&terms.rook_files, k_rook_semi_mg, k_rook_semi_eg);
                        if (feat != NULL) {
                            feat->rook_semi += 1;
                        }
                    }
                    break;
                }
                case PIECE_QUEEN:
                    eval_term_add(&terms.piece_square,
                                  k_queen_pst[pawn_pst_index(view)],
                                  k_queen_pst_eg[pawn_pst_index(view)]);
                    break;
                case PIECE_KING:
                    eval_term_add(&terms.piece_square, k_king_mid_pst[view], k_king_end_pst[view]);
                    if (g_opt_king_pst) {
                        eval_term_add(&terms.endgame_extra,
                                      k_king_mg_visual[view ^ 56],
                                      k_king_eg_visual[view ^ 56]);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    int knight_mob = attack_unions->mobility[side][PIECE_KNIGHT];
    int bishop_mob = attack_unions->mobility[side][PIECE_BISHOP];
    int rook_mob = attack_unions->mobility[side][PIECE_ROOK];
    int queen_mob = attack_unions->mobility[side][PIECE_QUEEN];
    eval_term_add(&terms.mobility,
                  knight_mob * k_mob_n_mg + bishop_mob * k_mob_b_mg +
                      rook_mob * k_mob_r_mg + queen_mob * k_mob_q_mg,
                  knight_mob * k_mob_n_eg + bishop_mob * k_mob_b_eg +
                      rook_mob * k_mob_r_eg + queen_mob * k_mob_q_eg);
    if (feat != NULL) {
        feat->mob_n += knight_mob;
        feat->mob_b += bishop_mob;
        feat->mob_r += rook_mob;
        feat->mob_q += queen_mob;
    }

    // Stage-B features (weights zero until fitted; see k_safe_check_* etc.).
    int mob_safe_n = attack_unions->mobility_safe[side][PIECE_KNIGHT];
    int mob_safe_b = attack_unions->mobility_safe[side][PIECE_BISHOP];
    int mob_safe_r = attack_unions->mobility_safe[side][PIECE_ROOK];
    int mob_safe_q = attack_unions->mobility_safe[side][PIECE_QUEEN];
    eval_term_add(&terms.mobility,
                  mob_safe_n * k_mob_safe_n_mg + mob_safe_b * k_mob_safe_b_mg +
                      mob_safe_r * k_mob_safe_r_mg + mob_safe_q * k_mob_safe_q_mg,
                  mob_safe_n * k_mob_safe_n_eg + mob_safe_b * k_mob_safe_b_eg +
                      mob_safe_r * k_mob_safe_r_eg + mob_safe_q * k_mob_safe_q_eg);
    int safe_checks[4];
    safe_check_counts(s, side, attack_unions, safe_checks);
    eval_term_add(&terms.king_safety_penalty,
                  safe_checks[0] * k_safe_check_n_mg + safe_checks[1] * k_safe_check_b_mg +
                      safe_checks[2] * k_safe_check_r_mg + safe_checks[3] * k_safe_check_q_mg,
                  safe_checks[0] * k_safe_check_n_eg + safe_checks[1] * k_safe_check_b_eg +
                      safe_checks[2] * k_safe_check_r_eg + safe_checks[3] * k_safe_check_q_eg);
    int bishop_pair = (chess_count_bits(s->bb[side][PIECE_BISHOP]) >= 2) ? 1 : 0;
    eval_term_add(&terms.material,
                  bishop_pair * k_bishop_pair_mg,
                  bishop_pair * k_bishop_pair_eg);
    if (feat != NULL) {
        feat->mob_safe_n += mob_safe_n;
        feat->mob_safe_b += mob_safe_b;
        feat->mob_safe_r += mob_safe_r;
        feat->mob_safe_q += mob_safe_q;
        feat->safe_check_n += safe_checks[0];
        feat->safe_check_b += safe_checks[1];
        feat->safe_check_r += safe_checks[2];
        feat->safe_check_q += safe_checks[3];
        feat->bishop_pair += bishop_pair;
    }

    uint64_t enemy_minors = s->bb[enemy][PIECE_BISHOP] | s->bb[enemy][PIECE_KNIGHT];
    uint64_t enemy_majors = s->bb[enemy][PIECE_ROOK] | s->bb[enemy][PIECE_QUEEN];
    int pawn_threat_minor = chess_count_bits(attack_unions->pawn[side] & enemy_minors);
    int pawn_threat_major = chess_count_bits(attack_unions->pawn[side] & enemy_majors);
    eval_term_add(&terms.pawn_activity,
                  pawn_threat_minor * k_pawn_threat_minor_mg +
                      pawn_threat_major * k_pawn_threat_major_mg,
                  pawn_threat_minor * k_pawn_threat_minor_eg +
                      pawn_threat_major * k_pawn_threat_major_eg);
    if (feat != NULL) {
        feat->pawn_threat_minor = pawn_threat_minor;
        feat->pawn_threat_major = pawn_threat_major;
    }

    int king_danger = king_safety_penalty(s, side, attack_unions) *
                      king_danger_fade_pct(s, side) / 100;
    if (g_opt_passer && passers != 0) {
        passer_extra_terms(s, side, passers, attack_unions, &terms.endgame_extra);
    }
    int hanging = hanging_piece_penalty(s, side, attack_unions);
    int queen_trap = queen_trap_penalty(s, side, attack_unions);
    terms.passed_pawns.mg = terms.passed_pawns.mg * k_passed_mg_scale / 100;
    terms.passed_pawns.eg = terms.passed_pawns.eg * k_passed_eg_scale / 100;
    eval_term_add(&terms.king_safety_penalty,
                  king_danger * k_king_mg_scale / 100,
                  (king_danger / 4) * k_king_eg_scale / 100);
    eval_term_add(&terms.hanging_penalty,
                  hanging * k_hanging_mg_scale / 100,
                  hanging * k_hanging_eg_scale / 100);
    eval_term_add(&terms.queen_trap_penalty,
                  queen_trap * k_queen_mg_scale / 100,
                  (queen_trap / 2) * k_queen_eg_scale / 100);
    if (feat != NULL) {
        feat->king_mg = king_danger;
        feat->king_eg = king_danger / 4;
        feat->hanging = hanging;
        feat->queen_mg = queen_trap;
        feat->queen_eg = queen_trap / 2;
    }

    if (out_breakdown != NULL) {
        out_breakdown->material = eval_term_blend(terms.material, phase);
        out_breakdown->piece_square = eval_term_blend(terms.piece_square, phase);
        out_breakdown->pawn_structure = eval_term_blend(terms.pawn_structure, phase);
        out_breakdown->passed_pawns = eval_term_blend(terms.passed_pawns, phase);
        out_breakdown->rook_files = eval_term_blend(terms.rook_files, phase);
        out_breakdown->mobility = eval_term_blend(terms.mobility, phase);
        out_breakdown->pawn_activity = eval_term_blend(terms.pawn_activity, phase);
        out_breakdown->king_safety_penalty = -eval_term_blend(terms.king_safety_penalty, phase);
        out_breakdown->hanging_penalty = -eval_term_blend(terms.hanging_penalty, phase);
        out_breakdown->queen_trap_penalty = -eval_term_blend(terms.queen_trap_penalty, phase);
        out_breakdown->outposts = eval_term_blend(terms.endgame_extra, phase);
    }

    int total_mg = terms.material.mg +
                   terms.piece_square.mg +
                   terms.pawn_structure.mg +
                   terms.passed_pawns.mg +
                   terms.rook_files.mg +
                   terms.mobility.mg +
                   terms.pawn_activity.mg +
                   terms.king_safety_penalty.mg +
                   terms.hanging_penalty.mg +
                   terms.queen_trap_penalty.mg +
                   terms.endgame_extra.mg;
    int total_eg = terms.material.eg +
                   terms.piece_square.eg +
                   terms.pawn_structure.eg +
                   terms.passed_pawns.eg +
                   terms.rook_files.eg +
                   terms.mobility.eg +
                   terms.pawn_activity.eg +
                   terms.king_safety_penalty.eg +
                   terms.hanging_penalty.eg +
                   terms.queen_trap_penalty.eg +
                   terms.endgame_extra.eg;
    if (feat != NULL) {
        // Residual = everything not linearly reconstructed from the captured
        // counts (material, piece squares, pawn structure, passed pawns,
        // rook files, mobility, king danger, hanging pieces, and queen traps).
        // Keep it in pre-blend mg/eg form so the Python side can sum then blend
        // once, matching the engine's single integer division exactly.
        int tuned_mg = terms.material.mg + terms.piece_square.mg +
                       terms.pawn_structure.mg + terms.rook_files.mg +
                       terms.mobility.mg + terms.passed_pawns.mg +
                       terms.pawn_activity.mg +
                       terms.king_safety_penalty.mg + terms.hanging_penalty.mg +
                       terms.queen_trap_penalty.mg;
        int tuned_eg = terms.material.eg + terms.piece_square.eg +
                       terms.pawn_structure.eg + terms.rook_files.eg +
                       terms.mobility.eg + terms.passed_pawns.eg +
                       terms.pawn_activity.eg +
                       terms.king_safety_penalty.eg + terms.hanging_penalty.eg +
                       terms.queen_trap_penalty.eg;
        feat->residual_mg = total_mg - tuned_mg;
        feat->residual_eg = total_eg - tuned_eg;
    }
    return (total_mg * phase + total_eg * (24 - phase)) / 24;
}

int hce_eval_cp_stm(const GameState *s) {
    if (s == NULL) {
        return 0;
    }
    hce_init_tables();

    int phase = phase_value(s);
    AttackUnions attack_unions;
    compute_attack_unions(s, &attack_unions);
    int white = eval_side(s, PIECE_WHITE, phase, &attack_unions, NULL, NULL);
    int black = eval_side(s, PIECE_BLACK, phase, &attack_unions, NULL, NULL);
    int cp_white = white - black;
    if (g_opt_scale && cp_white != 0) {
        cp_white = cp_white * endgame_scale64(s, cp_white > 0 ? PIECE_WHITE : PIECE_BLACK) / 64;
    }
    int cp_stm = (s->side_to_move == PIECE_WHITE) ? cp_white : -cp_white;
    // Tempo bonus: small advantage for having the move
    cp_stm += 12;
    return cp_stm;
}

int hce_eval_tune_features(const GameState *s,
                           HceTuneFeatures *white_out,
                           HceTuneFeatures *black_out,
                           int *phase_out) {
    if (s == NULL || white_out == NULL || black_out == NULL) {
        return 0;
    }
    hce_init_tables();
    memset(white_out, 0, sizeof(*white_out));
    memset(black_out, 0, sizeof(*black_out));
    int phase = phase_value(s);
    if (phase_out != NULL) {
        *phase_out = phase;
    }
    AttackUnions attack_unions;
    compute_attack_unions(s, &attack_unions);
    int white = eval_side(s, PIECE_WHITE, phase, &attack_unions, NULL, white_out);
    int black = eval_side(s, PIECE_BLACK, phase, &attack_unions, NULL, black_out);
    int cp_white = white - black;
    int cp_stm = (s->side_to_move == PIECE_WHITE) ? cp_white : -cp_white;
    cp_stm += 12;
    return cp_stm;
}

int hce_experimental_eval_cp_stm(const GameState *s) {
    return hce_eval_cp_stm(s);
}

bool hce_eval_breakdown(const GameState *s, ChessEvalBreakdown *out) {
    if (s == NULL || out == NULL) {
        return false;
    }

    hce_init_tables();
    memset(out, 0, sizeof(*out));
    out->phase = phase_value(s);
    AttackUnions attack_unions;
    compute_attack_unions(s, &attack_unions);
    out->white.total = eval_side(s, PIECE_WHITE, out->phase, &attack_unions, &out->white, NULL);
    out->black.total = eval_side(s, PIECE_BLACK, out->phase, &attack_unions, &out->black, NULL);
    out->score_cp_white = out->white.total - out->black.total;
    if (g_opt_scale && out->score_cp_white != 0) {
        out->score_cp_white = out->score_cp_white *
                              endgame_scale64(s, out->score_cp_white > 0 ? PIECE_WHITE : PIECE_BLACK) / 64;
    }
    out->score_cp_stm = (s->side_to_move == PIECE_WHITE) ? out->score_cp_white : -out->score_cp_white;
    out->score_cp_stm += 12;
    out->score_cp_white = (s->side_to_move == PIECE_WHITE) ? out->score_cp_stm : -out->score_cp_stm;
    return true;
}
