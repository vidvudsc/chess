#include "hce_internal.h"

#include <string.h>

const int hce_piece_value[PIECE_TYPE_COUNT] = {
    0,
    900,
    335,
    320,
    500,
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
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
};

static const int k_knight_pst[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
};

static const int k_bishop_pst[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
};

static const int k_rook_pst[64] = {
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
};

static const int k_queen_pst[64] = {
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
};

static const int k_king_mid_pst[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
};

static const int k_king_end_pst[64] = {
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
};

static inline int mirror_sq(int sq) {
    return sq ^ 56;
}

static inline bool is_light_square(int sq) {
    return ((square_file(sq) + square_rank(sq)) & 1) == 0;
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

uint64_t hce_bishop_attacks(int sq, uint64_t occ) {
    int f = square_file(sq);
    int r = square_rank(sq);
    uint64_t attacks = 0;

    for (int nf = f + 1, nr = r + 1; nf <= 7 && nr <= 7; ++nf, ++nr) {
        int nsq = make_square(nf, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nf = f - 1, nr = r + 1; nf >= 0 && nr <= 7; --nf, ++nr) {
        int nsq = make_square(nf, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nf = f + 1, nr = r - 1; nf <= 7 && nr >= 0; ++nf, --nr) {
        int nsq = make_square(nf, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nf = f - 1, nr = r - 1; nf >= 0 && nr >= 0; --nf, --nr) {
        int nsq = make_square(nf, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }

    return attacks;
}

uint64_t hce_rook_attacks(int sq, uint64_t occ) {
    int f = square_file(sq);
    int r = square_rank(sq);
    uint64_t attacks = 0;

    for (int nf = f + 1; nf <= 7; ++nf) {
        int nsq = make_square(nf, r);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nf = f - 1; nf >= 0; --nf) {
        int nsq = make_square(nf, r);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nr = r + 1; nr <= 7; ++nr) {
        int nsq = make_square(f, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nr = r - 1; nr >= 0; --nr) {
        int nsq = make_square(f, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }

    return attacks;
}

void hce_init_tables(void) {
    if (g_hce_tables_ready) {
        return;
    }

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

uint64_t hce_attackers_to_square(const GameState *s, int sq, int side) {
    hce_init_tables();
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
    return (hce_attackers_to_square(s, sq, side) & s->bb[side][PIECE_PAWN]) != 0;
}

static bool is_true_outpost_square(int side, int sq) {
    int rank = square_rank(sq);
    if (side == PIECE_WHITE) {
        return rank >= 4;
    }
    return rank <= 3;
}

static bool enemy_pawn_can_challenge_square(const GameState *s, int side, int sq) {
    if (s == NULL) {
        return false;
    }

    int enemy = side ^ 1;
    int file = square_file(sq);
    int rank = square_rank(sq);
    uint64_t candidate_files = g_neighbor_file_masks[file];
    uint64_t candidate_ranks = 0;

    if (side == PIECE_WHITE) {
        for (int r = rank + 1; r < 8; ++r) {
            candidate_ranks |= 0xFFULL << (r * 8);
        }
    } else {
        for (int r = 0; r < rank; ++r) {
            candidate_ranks |= 0xFFULL << (r * 8);
        }
    }

    return (s->bb[enemy][PIECE_PAWN] & candidate_files & candidate_ranks) != 0;
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

static int king_attack_units(const GameState *s, int attacker_side, int king_sq) {
    if (king_sq < 0) {
        return 0;
    }

    uint64_t zone = hce_king_attacks(king_sq) | (1ULL << king_sq);
    uint64_t occ = s->occ_all;
    int units = 0;

    uint64_t pawns = s->bb[attacker_side][PIECE_PAWN];
    while (pawns != 0) {
        int sq = chess_pop_lsb(&pawns);
        if ((hce_pawn_attacks(attacker_side, sq) & zone) != 0) {
            units += 1;
        }
    }

    uint64_t knights = s->bb[attacker_side][PIECE_KNIGHT];
    while (knights != 0) {
        int sq = chess_pop_lsb(&knights);
        if ((hce_knight_attacks(sq) & zone) != 0) {
            units += 2;
        }
    }

    uint64_t bishops = s->bb[attacker_side][PIECE_BISHOP];
    while (bishops != 0) {
        int sq = chess_pop_lsb(&bishops);
        if ((hce_bishop_attacks(sq, occ) & zone) != 0) {
            units += 2;
        }
    }

    uint64_t rooks = s->bb[attacker_side][PIECE_ROOK];
    while (rooks != 0) {
        int sq = chess_pop_lsb(&rooks);
        if ((hce_rook_attacks(sq, occ) & zone) != 0) {
            units += 3;
        }
    }

    uint64_t queens = s->bb[attacker_side][PIECE_QUEEN];
    while (queens != 0) {
        int sq = chess_pop_lsb(&queens);
        if (((hce_bishop_attacks(sq, occ) | hce_rook_attacks(sq, occ)) & zone) != 0) {
            units += 5;
        }
    }

    return units;
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

static int king_safety_penalty(const GameState *s, int side) {
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return 0;
    }

    int enemy = side ^ 1;
    int enemy_queen_count = chess_count_bits(s->bb[enemy][PIECE_QUEEN]);
    int enemy_rook_count = chess_count_bits(s->bb[enemy][PIECE_ROOK]);
    int attack_units = king_attack_units(s, enemy, king_sq);
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

static int bishop_quality_adjustment(const GameState *s, int side, int sq) {
    bool light = is_light_square(sq);
    int same_color_pawns = 0;
    int blocked_same_color_pawns = 0;
    uint64_t pawns = s->bb[side][PIECE_PAWN];

    while (pawns != 0) {
        int psq = chess_pop_lsb(&pawns);
        if (is_light_square(psq) != light) {
            continue;
        }
        same_color_pawns += 1;
        int step = (side == PIECE_WHITE) ? 8 : -8;
        int front_sq = psq + step;
        if (front_sq >= 0 && front_sq < 64 && (s->occ_all & (1ULL << front_sq)) != 0) {
            blocked_same_color_pawns += 1;
        }
    }

    return -same_color_pawns * 6 - blocked_same_color_pawns * 3;
}

static bool is_connected_passed_pawn(const GameState *s, int side, int sq) {
    int file = square_file(sq);
    int rank = square_rank(sq);
    for (int df = -1; df <= 1; df += 2) {
        int nf = file + df;
        if (nf < 0 || nf > 7) {
            continue;
        }
        for (int dr = -1; dr <= 1; ++dr) {
            int nr = rank + dr;
            if (nr < 0 || nr > 7) {
                continue;
            }
            int nsq = make_square(nf, nr);
            if ((s->bb[side][PIECE_PAWN] & (1ULL << nsq)) != 0 && is_passed_pawn(s, side, nsq)) {
                return true;
            }
        }
    }
    return false;
}

static int hanging_piece_penalty(const GameState *s, int side) {
    int penalty = 0;
    static const int scan_order[] = {PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT};
    for (size_t i = 0; i < sizeof(scan_order) / sizeof(scan_order[0]); ++i) {
        int piece = scan_order[i];
        uint64_t bb = s->bb[side][piece];
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

static int queen_trap_penalty(const GameState *s, int side) {
    if (s == NULL) {
        return 0;
    }

    int enemy = side ^ 1;
    int penalty = 0;
    uint64_t queens = s->bb[side][PIECE_QUEEN];
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
        int safe_escapes = 0;
        while (mobility != 0) {
            int to = chess_pop_lsb(&mobility);
            uint64_t enemy_attacks = hce_attackers_to_square(s, to, enemy);
            if ((enemy_attacks & ~s->bb[enemy][PIECE_KING]) == 0) {
                safe_escapes += 1;
            }
        }

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
    EvalTermPair outposts;
    EvalTermPair bishop_quality;
    EvalTermPair rook_files;
    EvalTermPair mobility;
    EvalTermPair bishop_pair;
    EvalTermPair king_safety_penalty;
    EvalTermPair hanging_penalty;
    EvalTermPair queen_trap_penalty;
} EvalSideTerms;

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

static int eval_side(const GameState *s, int side, int phase, ChessEvalSideBreakdown *out_breakdown) {
    EvalSideTerms terms;
    memset(&terms, 0, sizeof(terms));
    int enemy = side ^ 1;
    int enemy_pawn_min_file = 8;
    int enemy_pawn_max_file = -1;
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

    for (int piece = PIECE_QUEEN; piece <= PIECE_PAWN; ++piece) {
        uint64_t bb = s->bb[side][piece];
        while (bb != 0) {
            int sq = chess_pop_lsb(&bb);
            int view = (side == PIECE_WHITE) ? sq : mirror_sq(sq);
            eval_term_add(&terms.material, hce_piece_value[piece], hce_piece_value[piece]);
            switch (piece) {
                case PIECE_PAWN:
                    eval_term_add(&terms.piece_square, k_pawn_pst[view], k_pawn_pst[view] / 2);
                    if (is_isolated_pawn(s, side, sq)) {
                        eval_term_add(&terms.pawn_structure, -10, -15);
                    }
                    if (is_doubled_pawn(s, side, sq)) {
                        eval_term_add(&terms.pawn_structure, -12, -15);
                    }
                    if (is_passed_pawn(s, side, sq)) {
                        int file = square_file(sq);
                        int advance = (side == PIECE_WHITE) ? square_rank(sq) : (7 - square_rank(sq));
                        int passer_mg = 18 + advance * 5;
                        int passer_eg = 28 + advance * 8;
                        if (is_connected_passed_pawn(s, side, sq)) {
                            passer_mg += 8 + advance * 2;
                            passer_eg += 18 + advance * 4;
                        }
                        if (enemy_pawn_max_file >= 0 &&
                            (file <= enemy_pawn_min_file - 3 || file >= enemy_pawn_max_file + 3)) {
                            passer_mg += 10;
                            passer_eg += 28;
                        }
                        int front_sq = sq + ((side == PIECE_WHITE) ? 8 : -8);
                        if (front_sq >= 0 && front_sq < 64) {
                            if ((hce_attackers_to_square(s, front_sq, side) & s->bb[side][PIECE_PAWN]) != 0) {
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
                    }
                    break;
                case PIECE_KNIGHT:
                    eval_term_add(&terms.piece_square, k_knight_pst[view], k_knight_pst[view] / 2);
                    if (is_true_outpost_square(side, sq) &&
                        square_supported_by_pawn(s, side, sq) &&
                        !enemy_pawn_can_challenge_square(s, side, sq)) {
                        eval_term_add(&terms.outposts, 42, 22);
                    }
                    eval_term_add(&terms.mobility,
                                  chess_count_bits(hce_knight_attacks(sq) & ~s->occ[side]) * 3,
                                  chess_count_bits(hce_knight_attacks(sq) & ~s->occ[side]) * 2);
                    break;
                case PIECE_BISHOP:
                    eval_term_add(&terms.piece_square, k_bishop_pst[view], k_bishop_pst[view] / 2);
                    eval_term_add(&terms.mobility,
                                  chess_count_bits(hce_bishop_attacks(sq, s->occ_all) & ~s->occ[side]) * 2,
                                  chess_count_bits(hce_bishop_attacks(sq, s->occ_all) & ~s->occ[side]) * 3);
                    {
                        int bishop_quality = bishop_quality_adjustment(s, side, sq);
                        eval_term_add(&terms.bishop_quality, bishop_quality, bishop_quality / 2);
                    }
                    break;
                case PIECE_ROOK: {
                    eval_term_add(&terms.piece_square, k_rook_pst[view], k_rook_pst[view] / 2);
                    int file = square_file(sq);
                    bool own_pawn = (g_file_masks[file] & s->bb[side][PIECE_PAWN]) != 0;
                    bool enemy_pawn = (g_file_masks[file] & s->bb[enemy][PIECE_PAWN]) != 0;
                    if (!own_pawn && !enemy_pawn) {
                        eval_term_add(&terms.rook_files, 18, 12);
                    } else if (!own_pawn) {
                        eval_term_add(&terms.rook_files, 10, 6);
                    }
                    {
                        int rook_mob = chess_count_bits(hce_rook_attacks(sq, s->occ_all) & ~s->occ[side]);
                        eval_term_add(&terms.mobility, rook_mob, rook_mob * 2);
                    }
                    break;
                }
                case PIECE_QUEEN:
                    eval_term_add(&terms.piece_square, k_queen_pst[view], k_queen_pst[view] / 2);
                    {
                        int queen_mob = chess_count_bits((hce_rook_attacks(sq, s->occ_all) |
                                                          hce_bishop_attacks(sq, s->occ_all)) &
                                                         ~s->occ[side]);
                        eval_term_add(&terms.mobility, queen_mob, queen_mob);
                    }
                    break;
                case PIECE_KING:
                    eval_term_add(&terms.piece_square, k_king_mid_pst[view], k_king_end_pst[view]);
                    break;
                default:
                    break;
            }
        }
    }

    if (chess_count_bits(s->bb[side][PIECE_BISHOP]) >= 2) {
        eval_term_add(&terms.bishop_pair, 24, 28);
    }

    int king_danger = king_safety_penalty(s, side);
    int hanging = hanging_piece_penalty(s, side);
    int queen_trap = queen_trap_penalty(s, side);
    eval_term_add(&terms.king_safety_penalty, -king_danger, -(king_danger / 4));
    eval_term_add(&terms.hanging_penalty, -hanging, -hanging);
    eval_term_add(&terms.queen_trap_penalty, -queen_trap, -(queen_trap / 2));

    if (out_breakdown != NULL) {
        out_breakdown->material = eval_term_blend(terms.material, phase);
        out_breakdown->piece_square = eval_term_blend(terms.piece_square, phase);
        out_breakdown->pawn_structure = eval_term_blend(terms.pawn_structure, phase);
        out_breakdown->passed_pawns = eval_term_blend(terms.passed_pawns, phase);
        out_breakdown->outposts = eval_term_blend(terms.outposts, phase);
        out_breakdown->bishop_quality = eval_term_blend(terms.bishop_quality, phase);
        out_breakdown->rook_files = eval_term_blend(terms.rook_files, phase);
        out_breakdown->mobility = eval_term_blend(terms.mobility, phase);
        out_breakdown->bishop_pair = eval_term_blend(terms.bishop_pair, phase);
        out_breakdown->king_safety_penalty = -eval_term_blend(terms.king_safety_penalty, phase);
        out_breakdown->hanging_penalty = -eval_term_blend(terms.hanging_penalty, phase);
        out_breakdown->queen_trap_penalty = -eval_term_blend(terms.queen_trap_penalty, phase);
    }

    int total_mg = terms.material.mg +
                   terms.piece_square.mg +
                   terms.pawn_structure.mg +
                   terms.passed_pawns.mg +
                   terms.outposts.mg +
                   terms.bishop_quality.mg +
                   terms.rook_files.mg +
                   terms.mobility.mg +
                   terms.bishop_pair.mg +
                   terms.king_safety_penalty.mg +
                   terms.hanging_penalty.mg +
                   terms.queen_trap_penalty.mg;
    int total_eg = terms.material.eg +
                   terms.piece_square.eg +
                   terms.pawn_structure.eg +
                   terms.passed_pawns.eg +
                   terms.outposts.eg +
                   terms.bishop_quality.eg +
                   terms.rook_files.eg +
                   terms.mobility.eg +
                   terms.bishop_pair.eg +
                   terms.king_safety_penalty.eg +
                   terms.hanging_penalty.eg +
                   terms.queen_trap_penalty.eg;
    return (total_mg * phase + total_eg * (24 - phase)) / 24;
}

int hce_eval_cp_stm(const GameState *s) {
    if (s == NULL) {
        return 0;
    }
    hce_init_tables();

    int phase = phase_value(s);
    int white = eval_side(s, PIECE_WHITE, phase, NULL);
    int black = eval_side(s, PIECE_BLACK, phase, NULL);
    int cp_white = white - black;
    int cp_stm = (s->side_to_move == PIECE_WHITE) ? cp_white : -cp_white;
    // Tempo bonus: small advantage for having the move
    cp_stm += 12;
    return cp_stm;
}

bool hce_eval_breakdown(const GameState *s, ChessEvalBreakdown *out) {
    if (s == NULL || out == NULL) {
        return false;
    }

    hce_init_tables();
    memset(out, 0, sizeof(*out));
    out->phase = phase_value(s);
    out->white.total = eval_side(s, PIECE_WHITE, out->phase, &out->white);
    out->black.total = eval_side(s, PIECE_BLACK, out->phase, &out->black);
    out->score_cp_white = out->white.total - out->black.total;
    out->score_cp_stm = (s->side_to_move == PIECE_WHITE) ? out->score_cp_white : -out->score_cp_white;
    return true;
}
