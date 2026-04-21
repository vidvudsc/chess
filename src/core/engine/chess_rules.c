#include "chess_rules.h"

#include <stdlib.h>
#include <string.h>

#include "chess_hash.h"

static uint64_t g_knight_attacks[64];
static uint64_t g_king_attacks[64];
static uint64_t g_pawn_attacks[PIECE_COLOR_COUNT][64];
static bool g_attacks_ready = false;

static void init_attack_tables(void);

static void ensure_engine_ready(void) {
    init_attack_tables();
    chess_hash_init();
}

static void init_attack_tables(void) {
    if (g_attacks_ready) {
        return;
    }

    for (int sq = 0; sq < 64; ++sq) {
        int f = square_file(sq);
        int r = square_rank(sq);

        uint64_t knight = 0;
        const int knight_df[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        const int knight_dr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        for (int i = 0; i < 8; ++i) {
            int nf = f + knight_df[i];
            int nr = r + knight_dr[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                knight |= 1ULL << make_square(nf, nr);
            }
        }
        g_knight_attacks[sq] = knight;

        uint64_t king = 0;
        for (int df = -1; df <= 1; ++df) {
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) {
                    continue;
                }
                int nf = f + df;
                int nr = r + dr;
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                    king |= 1ULL << make_square(nf, nr);
                }
            }
        }
        g_king_attacks[sq] = king;

        uint64_t white_pawn = 0;
        uint64_t black_pawn = 0;
        if (f > 0 && r < 7) {
            white_pawn |= 1ULL << make_square(f - 1, r + 1);
        }
        if (f < 7 && r < 7) {
            white_pawn |= 1ULL << make_square(f + 1, r + 1);
        }
        if (f > 0 && r > 0) {
            black_pawn |= 1ULL << make_square(f - 1, r - 1);
        }
        if (f < 7 && r > 0) {
            black_pawn |= 1ULL << make_square(f + 1, r - 1);
        }
        g_pawn_attacks[PIECE_WHITE][sq] = white_pawn;
        g_pawn_attacks[PIECE_BLACK][sq] = black_pawn;
    }

    g_attacks_ready = true;
}

static uint64_t rook_attacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int f = square_file(sq);
    int r = square_rank(sq);

    for (int nr = r + 1; nr < 8; ++nr) {
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
    for (int nf = f + 1; nf < 8; ++nf) {
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

    return attacks;
}

static uint64_t bishop_attacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int f = square_file(sq);
    int r = square_rank(sq);

    for (int nf = f + 1, nr = r + 1; nf < 8 && nr < 8; ++nf, ++nr) {
        int nsq = make_square(nf, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nf = f - 1, nr = r + 1; nf >= 0 && nr < 8; --nf, ++nr) {
        int nsq = make_square(nf, nr);
        attacks |= 1ULL << nsq;
        if ((occ & (1ULL << nsq)) != 0) {
            break;
        }
    }
    for (int nf = f + 1, nr = r - 1; nf < 8 && nr >= 0; ++nf, --nr) {
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

static int piece_for_color_on_square(const GameState *s, int color, int sq) {
    if (sq < 0 || sq >= CHESS_BOARD_SQUARES) {
        return PIECE_NONE;
    }
    if (s->sq_color[sq] != color) {
        return PIECE_NONE;
    }
    int piece = s->sq_piece[sq];
    if (piece < PIECE_KING || piece >= PIECE_TYPE_COUNT) {
        return PIECE_NONE;
    }
    return piece;
}

static bool square_attacked(const GameState *s, int sq, int by_color) {
    uint64_t pawns = s->bb[by_color][PIECE_PAWN];
    if ((pawns & g_pawn_attacks[by_color ^ 1][sq]) != 0) {
        return true;
    }

    if ((s->bb[by_color][PIECE_KNIGHT] & g_knight_attacks[sq]) != 0) {
        return true;
    }
    if ((s->bb[by_color][PIECE_KING] & g_king_attacks[sq]) != 0) {
        return true;
    }

    int f = square_file(sq);
    int r = square_rank(sq);

    for (int nf = f + 1, nr = r + 1; nf < 8 && nr < 8; ++nf, ++nr) {
        int nsq = make_square(nf, nr);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_BISHOP || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }
    for (int nf = f - 1, nr = r + 1; nf >= 0 && nr < 8; --nf, ++nr) {
        int nsq = make_square(nf, nr);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_BISHOP || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }
    for (int nf = f + 1, nr = r - 1; nf < 8 && nr >= 0; ++nf, --nr) {
        int nsq = make_square(nf, nr);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_BISHOP || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }
    for (int nf = f - 1, nr = r - 1; nf >= 0 && nr >= 0; --nf, --nr) {
        int nsq = make_square(nf, nr);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_BISHOP || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }

    for (int nr = r + 1; nr < 8; ++nr) {
        int nsq = make_square(f, nr);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_ROOK || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }
    for (int nr = r - 1; nr >= 0; --nr) {
        int nsq = make_square(f, nr);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_ROOK || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }
    for (int nf = f + 1; nf < 8; ++nf) {
        int nsq = make_square(nf, r);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_ROOK || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }
    for (int nf = f - 1; nf >= 0; --nf) {
        int nsq = make_square(nf, r);
        int color = s->sq_color[nsq];
        if (color < 0) {
            continue;
        }
        if (color == by_color) {
            int piece = s->sq_piece[nsq];
            if (piece == PIECE_ROOK || piece == PIECE_QUEEN) {
                return true;
            }
        }
        break;
    }

    return false;
}

static void add_move(Move out[CHESS_MAX_MOVES], int *count, Move m) {
    if (*count >= CHESS_MAX_MOVES) {
        return;
    }
    out[*count] = m;
    ++(*count);
}

static void add_pawn_promotions(Move out[CHESS_MAX_MOVES], int *count, int from, int to, uint32_t base_flags) {
    add_move(out, count, move_pack(from, to, PIECE_PAWN, PIECE_QUEEN, base_flags | MOVE_FLAG_PROMOTION));
    add_move(out, count, move_pack(from, to, PIECE_PAWN, PIECE_ROOK, base_flags | MOVE_FLAG_PROMOTION));
    add_move(out, count, move_pack(from, to, PIECE_PAWN, PIECE_BISHOP, base_flags | MOVE_FLAG_PROMOTION));
    add_move(out, count, move_pack(from, to, PIECE_PAWN, PIECE_KNIGHT, base_flags | MOVE_FLAG_PROMOTION));
}

static bool apply_move_internal(GameState *s, Move m, UndoRecord *u, bool record_for_undo) {
    if (u == NULL) {
        return false;
    }

    int side = s->side_to_move;
    int opp = side ^ 1;
    int from = move_from(m);
    int to = move_to(m);
    int piece = move_piece(m);
    int promo = move_promo(m);

    uint64_t from_mask = 1ULL << from;
    uint64_t to_mask = 1ULL << to;

    if ((s->bb[side][piece] & from_mask) == 0) {
        return false;
    }

    u->move = m;
    u->captured_piece = PIECE_NONE;
    u->captured_square = to;
    u->castling_rights_prev = s->castling_rights;
    u->ep_square_prev = s->ep_square;
    u->halfmove_prev = s->halfmove_clock;
    u->fullmove_prev = s->fullmove_number;
    u->hash_prev = s->zobrist_hash;
    u->result_prev = s->result;
    u->extended_state_saved = record_for_undo;
    if (record_for_undo) {
        u->clock_prev[PIECE_WHITE] = s->clock_ms[PIECE_WHITE];
        u->clock_prev[PIECE_BLACK] = s->clock_ms[PIECE_BLACK];
        u->irreversible_ply_prev = s->irreversible_ply;
        u->last_move_prev = s->last_move;
        u->had_last_move_prev = s->has_last_move;
        u->hash_history_count_prev = s->hash_history_count;
    }

    bool is_castle = move_has_flag(m, MOVE_FLAG_CASTLE);
    bool is_en_passant = move_has_flag(m, MOVE_FLAG_EN_PASSANT);
    bool is_promotion = move_has_flag(m, MOVE_FLAG_PROMOTION);

    int captured_square = to;
    if (is_en_passant) {
        captured_square = to + ((side == PIECE_WHITE) ? -8 : 8);
    }
    uint64_t captured_mask = 1ULL << captured_square;
    uint64_t side_occ = s->occ[side];
    uint64_t opp_occ = s->occ[opp];

    if (s->sq_color[captured_square] == opp) {
        u->captured_piece = piece_for_color_on_square(s, opp, captured_square);
        u->captured_square = captured_square;
        if (u->captured_piece == PIECE_NONE) {
            return false;
        }
    }

    int placed_piece = piece;
    if (is_promotion) {
        bool valid_promo = (promo == PIECE_QUEEN) ||
                           (promo == PIECE_ROOK) ||
                           (promo == PIECE_BISHOP) ||
                           (promo == PIECE_KNIGHT);
        if (piece != PIECE_PAWN || !valid_promo) {
            return false;
        }
        placed_piece = promo;
    }

    uint8_t rights_old = s->castling_rights;
    uint8_t rights_new = rights_old;

    if (piece == PIECE_KING) {
        if (side == PIECE_WHITE) {
            rights_new &= (uint8_t)~(CASTLE_WHITE_KING | CASTLE_WHITE_QUEEN);
        } else {
            rights_new &= (uint8_t)~(CASTLE_BLACK_KING | CASTLE_BLACK_QUEEN);
        }
    }

    if (piece == PIECE_ROOK) {
        if (from == 0) {
            rights_new &= (uint8_t)~CASTLE_WHITE_QUEEN;
        } else if (from == 7) {
            rights_new &= (uint8_t)~CASTLE_WHITE_KING;
        } else if (from == 56) {
            rights_new &= (uint8_t)~CASTLE_BLACK_QUEEN;
        } else if (from == 63) {
            rights_new &= (uint8_t)~CASTLE_BLACK_KING;
        }
    }

    if (u->captured_piece == PIECE_ROOK) {
        if (captured_square == 0) {
            rights_new &= (uint8_t)~CASTLE_WHITE_QUEEN;
        } else if (captured_square == 7) {
            rights_new &= (uint8_t)~CASTLE_WHITE_KING;
        } else if (captured_square == 56) {
            rights_new &= (uint8_t)~CASTLE_BLACK_QUEEN;
        } else if (captured_square == 63) {
            rights_new &= (uint8_t)~CASTLE_BLACK_KING;
        }
    }

    int ep_new = CHESS_NO_SQUARE;
    if (piece == PIECE_PAWN && abs(to - from) == 16) {
        ep_new = (from + to) / 2;
    }

    uint64_t hash = s->zobrist_hash;
    hash ^= chess_hash_side_key();
    hash ^= chess_hash_castle_key(rights_old);
    if (s->ep_square != CHESS_NO_SQUARE) {
        hash ^= chess_hash_ep_key(s->ep_square);
    }

    hash ^= chess_hash_piece_key(side, piece, from);

    s->bb[side][piece] &= ~from_mask;
    side_occ &= ~from_mask;
    s->sq_piece[from] = PIECE_NONE;
    s->sq_color[from] = -1;

    if (u->captured_piece != PIECE_NONE) {
        s->bb[opp][u->captured_piece] &= ~captured_mask;
        hash ^= chess_hash_piece_key(opp, u->captured_piece, captured_square);
        opp_occ &= ~captured_mask;
        s->sq_piece[captured_square] = PIECE_NONE;
        s->sq_color[captured_square] = -1;
    }

    if (is_castle && piece == PIECE_KING) {
        int rook_from = -1;
        int rook_to = -1;
        if (side == PIECE_WHITE) {
            if (to == 6) {
                rook_from = 7;
                rook_to = 5;
            } else if (to == 2) {
                rook_from = 0;
                rook_to = 3;
            }
        } else {
            if (to == 62) {
                rook_from = 63;
                rook_to = 61;
            } else if (to == 58) {
                rook_from = 56;
                rook_to = 59;
            }
        }

        if (rook_from >= 0) {
            uint64_t rook_from_mask = 1ULL << rook_from;
            uint64_t rook_to_mask = 1ULL << rook_to;
            s->bb[side][PIECE_ROOK] &= ~rook_from_mask;
            s->bb[side][PIECE_ROOK] |= rook_to_mask;
            hash ^= chess_hash_piece_key(side, PIECE_ROOK, rook_from);
            hash ^= chess_hash_piece_key(side, PIECE_ROOK, rook_to);
            side_occ &= ~rook_from_mask;
            side_occ |= rook_to_mask;
            s->sq_piece[rook_from] = PIECE_NONE;
            s->sq_color[rook_from] = -1;
            s->sq_piece[rook_to] = PIECE_ROOK;
            s->sq_color[rook_to] = (int8_t)side;
        }
    }

    s->bb[side][placed_piece] |= to_mask;
    side_occ |= to_mask;
    hash ^= chess_hash_piece_key(side, placed_piece, to);
    s->sq_piece[to] = (int8_t)placed_piece;
    s->sq_color[to] = (int8_t)side;

    s->castling_rights = rights_new;
    s->ep_square = ep_new;
    if (piece == PIECE_PAWN || u->captured_piece != PIECE_NONE) {
        s->halfmove_clock = 0;
    } else {
        s->halfmove_clock += 1;
    }

    if (side == PIECE_BLACK) {
        s->fullmove_number += 1;
    }

    s->side_to_move = opp;

    hash ^= chess_hash_castle_key(rights_new);
    if (ep_new != CHESS_NO_SQUARE) {
        hash ^= chess_hash_ep_key(ep_new);
    }

    s->zobrist_hash = hash;
    s->occ[side] = side_occ;
    s->occ[opp] = opp_occ;
    s->occ_all = side_occ | opp_occ;

    if (record_for_undo) {
        if (s->ply >= CHESS_MAX_GAME_PLY) {
            return false;
        }
        s->undo_stack[s->ply++] = *u;

        if (s->hash_history_count < CHESS_MAX_GAME_PLY + 1) {
            s->hash_history[s->hash_history_count++] = s->zobrist_hash;
        }

        bool irreversible = (piece == PIECE_PAWN) || (u->captured_piece != PIECE_NONE) || (rights_new != rights_old);
        if (irreversible) {
            s->irreversible_ply = s->hash_history_count - 1;
        }

        s->last_move = m;
        s->has_last_move = true;
    }

    return true;
}

static void undo_move_internal(GameState *s, const UndoRecord *u) {
    Move m = u->move;
    int side = s->side_to_move ^ 1;
    int opp = side ^ 1;
    int from = move_from(m);
    int to = move_to(m);
    int piece = move_piece(m);
    int moved_piece = move_has_flag(m, MOVE_FLAG_PROMOTION) ? move_promo(m) : piece;

    uint64_t from_mask = 1ULL << from;
    uint64_t to_mask = 1ULL << to;
    uint64_t side_occ = s->occ[side];
    uint64_t opp_occ = s->occ[opp];

    s->bb[side][moved_piece] &= ~to_mask;
    s->bb[side][piece] |= from_mask;
    side_occ &= ~to_mask;
    side_occ |= from_mask;
    s->sq_piece[to] = PIECE_NONE;
    s->sq_color[to] = -1;
    s->sq_piece[from] = (int8_t)piece;
    s->sq_color[from] = (int8_t)side;

    if (move_has_flag(m, MOVE_FLAG_CASTLE) && piece == PIECE_KING) {
        int rook_from = -1;
        int rook_to = -1;
        if (side == PIECE_WHITE) {
            if (to == 6) {
                rook_from = 7;
                rook_to = 5;
            } else if (to == 2) {
                rook_from = 0;
                rook_to = 3;
            }
        } else {
            if (to == 62) {
                rook_from = 63;
                rook_to = 61;
            } else if (to == 58) {
                rook_from = 56;
                rook_to = 59;
            }
        }

        if (rook_from >= 0) {
            s->bb[side][PIECE_ROOK] &= ~(1ULL << rook_to);
            s->bb[side][PIECE_ROOK] |= (1ULL << rook_from);
            side_occ &= ~(1ULL << rook_to);
            side_occ |= (1ULL << rook_from);
            s->sq_piece[rook_to] = PIECE_NONE;
            s->sq_color[rook_to] = -1;
            s->sq_piece[rook_from] = PIECE_ROOK;
            s->sq_color[rook_from] = (int8_t)side;
        }
    }

    if (u->captured_piece != PIECE_NONE) {
        s->bb[opp][u->captured_piece] |= (1ULL << u->captured_square);
        opp_occ |= (1ULL << u->captured_square);
        s->sq_piece[u->captured_square] = (int8_t)u->captured_piece;
        s->sq_color[u->captured_square] = (int8_t)opp;
    }

    s->side_to_move = side;
    s->castling_rights = u->castling_rights_prev;
    s->ep_square = u->ep_square_prev;
    s->halfmove_clock = u->halfmove_prev;
    s->fullmove_number = u->fullmove_prev;
    s->zobrist_hash = u->hash_prev;
    s->result = u->result_prev;
    if (u->extended_state_saved) {
        s->clock_ms[PIECE_WHITE] = u->clock_prev[PIECE_WHITE];
        s->clock_ms[PIECE_BLACK] = u->clock_prev[PIECE_BLACK];
        s->irreversible_ply = u->irreversible_ply_prev;
        s->last_move = u->last_move_prev;
        s->has_last_move = u->had_last_move_prev;
        s->hash_history_count = u->hash_history_count_prev;
    }
    s->occ[side] = side_occ;
    s->occ[opp] = opp_occ;
    s->occ_all = side_occ | opp_occ;
}

static int sign_int(int v) {
    if (v > 0) {
        return 1;
    }
    if (v < 0) {
        return -1;
    }
    return 0;
}

static uint64_t line_segment_mask(int from, int to) {
    int ff = square_file(from);
    int fr = square_rank(from);
    int tf = square_file(to);
    int tr = square_rank(to);

    int df = sign_int(tf - ff);
    int dr = sign_int(tr - fr);
    int af = tf - ff;
    if (af < 0) {
        af = -af;
    }
    int ar = tr - fr;
    if (ar < 0) {
        ar = -ar;
    }

    if (!((df == 0 && dr != 0) || (df != 0 && dr == 0) || (af == ar && af != 0))) {
        return 0;
    }

    uint64_t mask = 1ULL << from;
    int f = ff;
    int r = fr;
    while (f != tf || r != tr) {
        f += df;
        r += dr;
        if (f < 0 || f > 7 || r < 0 || r > 7) {
            return 0;
        }
        mask |= 1ULL << make_square(f, r);
    }

    return mask;
}

static void add_move_targets(Move out[CHESS_MAX_MOVES],
                             int *count,
                             int from,
                             int piece,
                             uint64_t targets,
                             uint64_t opp_occ) {
    while (targets != 0) {
        int to = chess_pop_lsb(&targets);
        uint32_t flags = ((opp_occ & (1ULL << to)) != 0) ? MOVE_FLAG_CAPTURE : MOVE_FLAG_NONE;
        add_move(out, count, move_pack(from, to, piece, CHESS_PROMO_NONE, flags));
    }
}

static void add_move_if_legal_after_apply(GameState *s,
                                          Move out[CHESS_MAX_MOVES],
                                          int *count,
                                          Move m,
                                          int king_sq_after) {
    UndoRecord u;
    if (!apply_move_internal(s, m, &u, false)) {
        return;
    }

    bool illegal = (king_sq_after < 0) || square_attacked(s, king_sq_after, s->side_to_move);
    undo_move_internal(s, &u);
    if (!illegal) {
        add_move(out, count, m);
    }
}

static void compute_checkers_and_pins(const GameState *s,
                                      int side,
                                      int king_sq,
                                      uint64_t *checkers_out,
                                      uint64_t pin_masks[64]) {
    static const int dir_df[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    static const int dir_dr[8] = {1, -1, 0, 0, 1, 1, -1, -1};

    uint64_t checkers = 0;
    memset(pin_masks, 0, sizeof(uint64_t) * 64);

    int opp = side ^ 1;
    checkers |= s->bb[opp][PIECE_PAWN] & g_pawn_attacks[side][king_sq];
    checkers |= s->bb[opp][PIECE_KNIGHT] & g_knight_attacks[king_sq];
    checkers |= s->bb[opp][PIECE_KING] & g_king_attacks[king_sq];

    int kf = square_file(king_sq);
    int kr = square_rank(king_sq);
    for (int dir = 0; dir < 8; ++dir) {
        int df = dir_df[dir];
        int dr = dir_dr[dir];
        int f = kf + df;
        int r = kr + dr;
        int blocker_sq = CHESS_NO_SQUARE;
        uint64_t ray_mask = 1ULL << king_sq;

        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int sq = make_square(f, r);
            ray_mask |= 1ULL << sq;

            int color = s->sq_color[sq];
            if (color < 0) {
                f += df;
                r += dr;
                continue;
            }

            if (color == side) {
                if (blocker_sq != CHESS_NO_SQUARE) {
                    break;
                }
                blocker_sq = sq;
                f += df;
                r += dr;
                continue;
            }

            int piece = s->sq_piece[sq];
            bool slider = false;
            if (dir < 4) {
                slider = (piece == PIECE_ROOK || piece == PIECE_QUEEN);
            } else {
                slider = (piece == PIECE_BISHOP || piece == PIECE_QUEEN);
            }

            if (slider) {
                if (blocker_sq == CHESS_NO_SQUARE) {
                    checkers |= 1ULL << sq;
                } else {
                    pin_masks[blocker_sq] = ray_mask;
                }
            }
            break;
        }
    }

    *checkers_out = checkers;
}

static int generate_legal_moves_mutable(GameState *s, Move out[CHESS_MAX_MOVES]) {
    int legal_count = 0;
    int side = s->side_to_move;
    int opp = side ^ 1;
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return 0;
    }

    uint64_t own_occ = s->occ[side];
    uint64_t opp_occ = s->occ[opp];
    uint64_t all_occ = s->occ_all;
    uint64_t checkers = 0;
    uint64_t pin_masks[64];
    compute_checkers_and_pins(s, side, king_sq, &checkers, pin_masks);

    int check_count = chess_count_bits(checkers);
    uint64_t check_mask = ~0ULL;
    if (check_count == 1) {
        int checker_sq = chess_pop_lsb(&checkers);
        int checker_piece = s->sq_piece[checker_sq];
        if (checker_piece == PIECE_ROOK || checker_piece == PIECE_BISHOP || checker_piece == PIECE_QUEEN) {
            uint64_t ray = line_segment_mask(king_sq, checker_sq);
            check_mask = (ray != 0) ? ray : (1ULL << checker_sq);
        } else {
            check_mask = 1ULL << checker_sq;
        }
    } else if (check_count >= 2) {
        check_mask = 0;
    }

    if (check_count < 2) {
        uint64_t pawns = s->bb[side][PIECE_PAWN];
        while (pawns != 0) {
            int from = chess_pop_lsb(&pawns);
            uint64_t pin_mask = pin_masks[from];
            int file = square_file(from);
            int rank = square_rank(from);
            int step = (side == PIECE_WHITE) ? 8 : -8;
            int promote_rank = (side == PIECE_WHITE) ? 7 : 0;

            int to = from + step;
            if (to >= 0 && to < 64 && (all_occ & (1ULL << to)) == 0) {
                uint64_t to_bit = 1ULL << to;
                if ((to_bit & check_mask) != 0 && (pin_mask == 0 || (to_bit & pin_mask) != 0)) {
                    if (square_rank(to) == promote_rank) {
                        add_pawn_promotions(out, &legal_count, from, to, MOVE_FLAG_NONE);
                    } else {
                        add_move(out, &legal_count, move_pack(from, to, PIECE_PAWN, CHESS_PROMO_NONE, MOVE_FLAG_NONE));
                    }
                }

                if (((side == PIECE_WHITE && rank == 1) || (side == PIECE_BLACK && rank == 6))) {
                    int to2 = from + 2 * step;
                    if ((all_occ & (1ULL << to2)) == 0) {
                        uint64_t to2_bit = 1ULL << to2;
                        if ((to2_bit & check_mask) != 0 && (pin_mask == 0 || (to2_bit & pin_mask) != 0)) {
                            add_move(out, &legal_count, move_pack(from, to2, PIECE_PAWN, CHESS_PROMO_NONE, MOVE_FLAG_DOUBLE_PUSH));
                        }
                    }
                }
            }

            int cap_left = (side == PIECE_WHITE) ? (from + 7) : (from - 9);
            int cap_right = (side == PIECE_WHITE) ? (from + 9) : (from - 7);

            if (file > 0 && cap_left >= 0 && cap_left < 64) {
                uint64_t to_bit = 1ULL << cap_left;
                if ((opp_occ & to_bit) != 0 && (to_bit & check_mask) != 0 && (pin_mask == 0 || (to_bit & pin_mask) != 0)) {
                    uint32_t flags = MOVE_FLAG_CAPTURE;
                    if (square_rank(cap_left) == promote_rank) {
                        add_pawn_promotions(out, &legal_count, from, cap_left, flags);
                    } else {
                        add_move(out, &legal_count, move_pack(from, cap_left, PIECE_PAWN, CHESS_PROMO_NONE, flags));
                    }
                } else if (s->ep_square == cap_left) {
                    Move ep = move_pack(from, cap_left, PIECE_PAWN, CHESS_PROMO_NONE, MOVE_FLAG_CAPTURE | MOVE_FLAG_EN_PASSANT);
                    add_move_if_legal_after_apply(s, out, &legal_count, ep, king_sq);
                }
            }

            if (file < 7 && cap_right >= 0 && cap_right < 64) {
                uint64_t to_bit = 1ULL << cap_right;
                if ((opp_occ & to_bit) != 0 && (to_bit & check_mask) != 0 && (pin_mask == 0 || (to_bit & pin_mask) != 0)) {
                    uint32_t flags = MOVE_FLAG_CAPTURE;
                    if (square_rank(cap_right) == promote_rank) {
                        add_pawn_promotions(out, &legal_count, from, cap_right, flags);
                    } else {
                        add_move(out, &legal_count, move_pack(from, cap_right, PIECE_PAWN, CHESS_PROMO_NONE, flags));
                    }
                } else if (s->ep_square == cap_right) {
                    Move ep = move_pack(from, cap_right, PIECE_PAWN, CHESS_PROMO_NONE, MOVE_FLAG_CAPTURE | MOVE_FLAG_EN_PASSANT);
                    add_move_if_legal_after_apply(s, out, &legal_count, ep, king_sq);
                }
            }
        }

        uint64_t knights = s->bb[side][PIECE_KNIGHT];
        while (knights != 0) {
            int from = chess_pop_lsb(&knights);
            uint64_t targets = g_knight_attacks[from] & ~own_occ & check_mask;
            if (pin_masks[from] != 0) {
                targets &= pin_masks[from];
            }
            add_move_targets(out, &legal_count, from, PIECE_KNIGHT, targets, opp_occ);
        }

        uint64_t bishops = s->bb[side][PIECE_BISHOP];
        while (bishops != 0) {
            int from = chess_pop_lsb(&bishops);
            uint64_t targets = bishop_attacks(from, all_occ) & ~own_occ & check_mask;
            if (pin_masks[from] != 0) {
                targets &= pin_masks[from];
            }
            add_move_targets(out, &legal_count, from, PIECE_BISHOP, targets, opp_occ);
        }

        uint64_t rooks = s->bb[side][PIECE_ROOK];
        while (rooks != 0) {
            int from = chess_pop_lsb(&rooks);
            uint64_t targets = rook_attacks(from, all_occ) & ~own_occ & check_mask;
            if (pin_masks[from] != 0) {
                targets &= pin_masks[from];
            }
            add_move_targets(out, &legal_count, from, PIECE_ROOK, targets, opp_occ);
        }

        uint64_t queens = s->bb[side][PIECE_QUEEN];
        while (queens != 0) {
            int from = chess_pop_lsb(&queens);
            uint64_t targets = (rook_attacks(from, all_occ) | bishop_attacks(from, all_occ)) & ~own_occ & check_mask;
            if (pin_masks[from] != 0) {
                targets &= pin_masks[from];
            }
            add_move_targets(out, &legal_count, from, PIECE_QUEEN, targets, opp_occ);
        }
    }

    uint64_t king_targets = g_king_attacks[king_sq] & ~own_occ;
    while (king_targets != 0) {
        int to = chess_pop_lsb(&king_targets);
        uint32_t flags = ((opp_occ & (1ULL << to)) != 0) ? MOVE_FLAG_CAPTURE : MOVE_FLAG_NONE;
        Move m = move_pack(king_sq, to, PIECE_KING, CHESS_PROMO_NONE, flags);
        add_move_if_legal_after_apply(s, out, &legal_count, m, to);
    }

    if (check_count == 0) {
        if (side == PIECE_WHITE && king_sq == 4) {
            bool king_safe = !square_attacked(s, 4, PIECE_BLACK);
            if ((s->castling_rights & CASTLE_WHITE_KING) != 0 &&
                (s->occ_all & ((1ULL << 5) | (1ULL << 6))) == 0 &&
                (s->bb[PIECE_WHITE][PIECE_ROOK] & (1ULL << 7)) != 0 &&
                king_safe && !square_attacked(s, 5, PIECE_BLACK) && !square_attacked(s, 6, PIECE_BLACK)) {
                add_move(out, &legal_count, move_pack(4, 6, PIECE_KING, CHESS_PROMO_NONE, MOVE_FLAG_CASTLE));
            }
            if ((s->castling_rights & CASTLE_WHITE_QUEEN) != 0 &&
                (s->occ_all & ((1ULL << 1) | (1ULL << 2) | (1ULL << 3))) == 0 &&
                (s->bb[PIECE_WHITE][PIECE_ROOK] & (1ULL << 0)) != 0 &&
                king_safe && !square_attacked(s, 3, PIECE_BLACK) && !square_attacked(s, 2, PIECE_BLACK)) {
                add_move(out, &legal_count, move_pack(4, 2, PIECE_KING, CHESS_PROMO_NONE, MOVE_FLAG_CASTLE));
            }
        } else if (side == PIECE_BLACK && king_sq == 60) {
            bool king_safe = !square_attacked(s, 60, PIECE_WHITE);
            if ((s->castling_rights & CASTLE_BLACK_KING) != 0 &&
                (s->occ_all & ((1ULL << 61) | (1ULL << 62))) == 0 &&
                (s->bb[PIECE_BLACK][PIECE_ROOK] & (1ULL << 63)) != 0 &&
                king_safe && !square_attacked(s, 61, PIECE_WHITE) && !square_attacked(s, 62, PIECE_WHITE)) {
                add_move(out, &legal_count, move_pack(60, 62, PIECE_KING, CHESS_PROMO_NONE, MOVE_FLAG_CASTLE));
            }
            if ((s->castling_rights & CASTLE_BLACK_QUEEN) != 0 &&
                (s->occ_all & ((1ULL << 57) | (1ULL << 58) | (1ULL << 59))) == 0 &&
                (s->bb[PIECE_BLACK][PIECE_ROOK] & (1ULL << 56)) != 0 &&
                king_safe && !square_attacked(s, 59, PIECE_WHITE) && !square_attacked(s, 58, PIECE_WHITE)) {
                add_move(out, &legal_count, move_pack(60, 58, PIECE_KING, CHESS_PROMO_NONE, MOVE_FLAG_CASTLE));
            }
        }
    }

    return legal_count;
}

void chess_init(GameState *s, const MatchConfig *cfg) {
    ensure_engine_ready();

    chess_state_clear(s);
    if (cfg != NULL) {
        s->config = *cfg;
    }

    s->bb[PIECE_WHITE][PIECE_KING] = 0x0000000000000010ULL;
    s->bb[PIECE_WHITE][PIECE_QUEEN] = 0x0000000000000008ULL;
    s->bb[PIECE_WHITE][PIECE_BISHOP] = 0x0000000000000024ULL;
    s->bb[PIECE_WHITE][PIECE_KNIGHT] = 0x0000000000000042ULL;
    s->bb[PIECE_WHITE][PIECE_ROOK] = 0x0000000000000081ULL;
    s->bb[PIECE_WHITE][PIECE_PAWN] = 0x000000000000FF00ULL;

    s->bb[PIECE_BLACK][PIECE_KING] = 0x1000000000000000ULL;
    s->bb[PIECE_BLACK][PIECE_QUEEN] = 0x0800000000000000ULL;
    s->bb[PIECE_BLACK][PIECE_BISHOP] = 0x2400000000000000ULL;
    s->bb[PIECE_BLACK][PIECE_KNIGHT] = 0x4200000000000000ULL;
    s->bb[PIECE_BLACK][PIECE_ROOK] = 0x8100000000000000ULL;
    s->bb[PIECE_BLACK][PIECE_PAWN] = 0x00FF000000000000ULL;

    s->side_to_move = PIECE_WHITE;
    s->castling_rights = CASTLE_WHITE_KING | CASTLE_WHITE_QUEEN | CASTLE_BLACK_KING | CASTLE_BLACK_QUEEN;
    s->ep_square = CHESS_NO_SQUARE;
    s->halfmove_clock = 0;
    s->fullmove_number = 1;
    s->result = GAME_RESULT_ONGOING;
    s->ply = 0;
    s->has_last_move = false;

    chess_state_rebuild_occupancy(s);
    s->zobrist_hash = chess_hash_full(s);
    s->hash_history_count = 1;
    s->hash_history[0] = s->zobrist_hash;
    s->irreversible_ply = 0;

    chess_reset_clocks(s);
}

int chess_generate_legal_moves(const GameState *s, Move out[CHESS_MAX_MOVES]) {
    ensure_engine_ready();
    GameState copy;
    memcpy(copy.bb, s->bb, sizeof(copy.bb));
    copy.occ[PIECE_WHITE] = s->occ[PIECE_WHITE];
    copy.occ[PIECE_BLACK] = s->occ[PIECE_BLACK];
    copy.occ_all = s->occ_all;

    copy.side_to_move = s->side_to_move;
    copy.castling_rights = s->castling_rights;
    copy.ep_square = s->ep_square;
    copy.halfmove_clock = s->halfmove_clock;
    copy.fullmove_number = s->fullmove_number;
    copy.zobrist_hash = s->zobrist_hash;
    copy.result = s->result;

    copy.config = s->config;
    copy.clock_ms[PIECE_WHITE] = s->clock_ms[PIECE_WHITE];
    copy.clock_ms[PIECE_BLACK] = s->clock_ms[PIECE_BLACK];

    copy.ply = s->ply;
    copy.hash_history_count = s->hash_history_count;
    copy.irreversible_ply = s->irreversible_ply;
    copy.last_move = s->last_move;
    copy.has_last_move = s->has_last_move;
    memcpy(copy.sq_piece, s->sq_piece, sizeof(copy.sq_piece));
    memcpy(copy.sq_color, s->sq_color, sizeof(copy.sq_color));

    return generate_legal_moves_mutable(&copy, out);
}

int chess_generate_legal_moves_mut(GameState *s, Move out[CHESS_MAX_MOVES]) {
    ensure_engine_ready();
    return generate_legal_moves_mutable(s, out);
}

bool chess_in_check(const GameState *s, int side) {
    ensure_engine_ready();
    int king_sq = chess_find_king_square(s, side);
    if (king_sq < 0) {
        return false;
    }
    return square_attacked(s, king_sq, side ^ 1);
}

bool chess_make_move(GameState *s, Move m) {
    ensure_engine_ready();
    if (s->result != GAME_RESULT_ONGOING) {
        return false;
    }

    Move legal_moves[CHESS_MAX_MOVES];
    int legal_count = generate_legal_moves_mutable(s, legal_moves);

    Move chosen = 0;
    bool found = false;
    for (int i = 0; i < legal_count; ++i) {
        Move candidate = legal_moves[i];
        if (move_from(candidate) == move_from(m) &&
            move_to(candidate) == move_to(m) &&
            move_piece(candidate) == move_piece(m) &&
            move_promo(candidate) == move_promo(m)) {
            chosen = candidate;
            found = true;
            break;
        }
    }
    if (!found) {
        return false;
    }

    UndoRecord u;
    int moving_side = s->side_to_move;
    if (!apply_move_internal(s, chosen, &u, true)) {
        return false;
    }

    if (s->config.clock_enabled && s->config.increment_ms > 0) {
        s->clock_ms[moving_side] += s->config.increment_ms;
    }

    chess_update_result(s);
    return true;
}

bool chess_make_move_trusted(GameState *s, Move legal_move) {
    ensure_engine_ready();
    if (s == NULL || s->result != GAME_RESULT_ONGOING) {
        return false;
    }

    UndoRecord u;
    int moving_side = s->side_to_move;
    if (!apply_move_internal(s, legal_move, &u, true)) {
        return false;
    }

    if (s->config.clock_enabled && s->config.increment_ms > 0) {
        s->clock_ms[moving_side] += s->config.increment_ms;
    }

    // Search callers generate legal moves at each node and evaluate terminal
    // conditions explicitly, so avoid expensive full result recomputation here.
    s->result = GAME_RESULT_ONGOING;
    return true;
}

bool chess_undo_move(GameState *s) {
    if (s->ply <= 0) {
        return false;
    }

    UndoRecord u = s->undo_stack[s->ply - 1];
    s->ply -= 1;
    undo_move_internal(s, &u);
    return true;
}

bool chess_is_move_legal(const GameState *s, Move m) {
    Move legal[CHESS_MAX_MOVES];
    int n = chess_generate_legal_moves(s, legal);
    for (int i = 0; i < n; ++i) {
        if (move_from(legal[i]) == move_from(m) &&
            move_to(legal[i]) == move_to(m) &&
            move_piece(legal[i]) == move_piece(m) &&
            move_promo(legal[i]) == move_promo(m)) {
            return true;
        }
    }
    return false;
}

bool chess_has_mating_material(const GameState *s, int side) {
    uint64_t pawns = s->bb[side][PIECE_PAWN];
    uint64_t rooks = s->bb[side][PIECE_ROOK];
    uint64_t queens = s->bb[side][PIECE_QUEEN];
    uint64_t bishops = s->bb[side][PIECE_BISHOP];
    uint64_t knights = s->bb[side][PIECE_KNIGHT];

    if (pawns != 0 || rooks != 0 || queens != 0) {
        return true;
    }

    int bishop_count = chess_count_bits(bishops);
    int knight_count = chess_count_bits(knights);

    if (bishop_count >= 1 && knight_count >= 1) {
        return true;
    }
    if (knight_count >= 3) {
        return true;
    }

    if (bishop_count >= 2) {
        bool light = false;
        bool dark = false;
        uint64_t b = bishops;
        while (b != 0) {
            int sq = chess_pop_lsb(&b);
            if (((square_file(sq) + square_rank(sq)) & 1) == 0) {
                light = true;
            } else {
                dark = true;
            }
        }
        if (light && dark) {
            return true;
        }
    }

    return false;
}

static bool insufficient_material(const GameState *s) {
    return !chess_has_mating_material(s, PIECE_WHITE) && !chess_has_mating_material(s, PIECE_BLACK);
}

GameResult chess_update_result(GameState *s) {
    ensure_engine_ready();
    if (s->result != GAME_RESULT_ONGOING) {
        return s->result;
    }

    if (insufficient_material(s)) {
        s->result = GAME_RESULT_DRAW_INSUFFICIENT;
        return s->result;
    }

    if (s->halfmove_clock >= 150) {
        s->result = GAME_RESULT_DRAW_75;
        return s->result;
    }
    if (s->halfmove_clock >= 100) {
        s->result = GAME_RESULT_DRAW_50;
        return s->result;
    }

    int repetitions = 0;
    uint64_t current = s->zobrist_hash;
    for (int i = s->hash_history_count - 1; i >= s->irreversible_ply; --i) {
        if (s->hash_history[i] == current) {
            ++repetitions;
        }
    }
    if (repetitions >= 3) {
        s->result = GAME_RESULT_DRAW_REPETITION;
        return s->result;
    }

    Move legal[CHESS_MAX_MOVES];
    int legal_count = generate_legal_moves_mutable(s, legal);
    if (legal_count == 0) {
        if (chess_in_check(s, s->side_to_move)) {
            s->result = (s->side_to_move == PIECE_WHITE) ? GAME_RESULT_BLACK_WIN : GAME_RESULT_WHITE_WIN;
        } else {
            s->result = GAME_RESULT_DRAW_STALEMATE;
        }
    }

    return s->result;
}

void chess_set_result(GameState *s, GameResult result) {
    s->result = result;
}

void chess_tick_clock(GameState *s, int delta_ms) {
    if (s->result != GAME_RESULT_ONGOING || delta_ms <= 0) {
        return;
    }

    int side = s->side_to_move;
    if (!s->config.clock_enabled) {
        s->clock_ms[side] += delta_ms;
        return;
    }

    s->clock_ms[side] -= delta_ms;
    if (s->clock_ms[side] > 0) {
        return;
    }

    s->clock_ms[side] = 0;
    if (!chess_has_mating_material(s, side ^ 1)) {
        s->result = GAME_RESULT_DRAW_INSUFFICIENT;
    } else {
        s->result = GAME_RESULT_WIN_TIMEOUT;
    }
}

uint64_t chess_perft(GameState *s, int depth) {
    ensure_engine_ready();
    if (depth <= 0) {
        return 1;
    }

    Move legal[CHESS_MAX_MOVES];
    int n = generate_legal_moves_mutable(s, legal);
    if (depth == 1) {
        return (uint64_t)n;
    }

    uint64_t nodes = 0;
    for (int i = 0; i < n; ++i) {
        UndoRecord u;
        if (!apply_move_internal(s, legal[i], &u, false)) {
            continue;
        }
        nodes += chess_perft(s, depth - 1);
        undo_move_internal(s, &u);
    }

    return nodes;
}
