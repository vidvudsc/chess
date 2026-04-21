#include "chess_io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess_hash.h"

static void set_err(char *err, size_t err_sz, const char *msg) {
    if (err != NULL && err_sz > 0) {
        snprintf(err, err_sz, "%s", msg);
    }
}

static int piece_from_fen_char(char c, int *out_color) {
    int color = isupper((unsigned char)c) ? PIECE_WHITE : PIECE_BLACK;
    char lower = (char)tolower((unsigned char)c);
    int piece = PIECE_NONE;

    switch (lower) {
        case 'k': piece = PIECE_KING; break;
        case 'q': piece = PIECE_QUEEN; break;
        case 'b': piece = PIECE_BISHOP; break;
        case 'n': piece = PIECE_KNIGHT; break;
        case 'r': piece = PIECE_ROOK; break;
        case 'p': piece = PIECE_PAWN; break;
        default: return PIECE_NONE;
    }

    if (out_color != NULL) {
        *out_color = color;
    }
    return piece;
}

static char piece_to_fen_char(int color, int piece) {
    static const char white_chars[PIECE_TYPE_COUNT] = {'K', 'Q', 'B', 'N', 'R', 'P'};
    static const char black_chars[PIECE_TYPE_COUNT] = {'k', 'q', 'b', 'n', 'r', 'p'};
    if (piece < 0 || piece >= PIECE_TYPE_COUNT) {
        return '?';
    }
    return (color == PIECE_WHITE) ? white_chars[piece] : black_chars[piece];
}

static bool square_to_alg(int sq, char out[3]) {
    if (sq < 0 || sq >= 64 || out == NULL) {
        return false;
    }
    out[0] = (char)('a' + square_file(sq));
    out[1] = (char)('1' + square_rank(sq));
    out[2] = '\0';
    return true;
}

static bool alg_to_square(char file, char rank, int *out_sq) {
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8') {
        return false;
    }
    int f = file - 'a';
    int r = rank - '1';
    if (out_sq != NULL) {
        *out_sq = make_square(f, r);
    }
    return true;
}

bool chess_load_fen(GameState *s, const char *fen, char *err, size_t err_sz) {
    if (s == NULL || fen == NULL) {
        set_err(err, err_sz, "Null argument");
        return false;
    }

    GameState t;
    chess_state_clear(&t);
    t.config = s->config;

    const char *p = fen;
    int rank = 7;
    int file = 0;

    while (*p != '\0' && *p != ' ') {
        char c = *p;
        if (c == '/') {
            if (file != 8 || rank == 0) {
                set_err(err, err_sz, "Invalid board rows");
                return false;
            }
            rank -= 1;
            file = 0;
            p += 1;
            continue;
        }

        if (isdigit((unsigned char)c)) {
            int empty = c - '0';
            if (empty < 1 || empty > 8 || file + empty > 8) {
                set_err(err, err_sz, "Invalid empty count");
                return false;
            }
            file += empty;
            p += 1;
            continue;
        }

        int color = -1;
        int piece = piece_from_fen_char(c, &color);
        if (piece == PIECE_NONE) {
            set_err(err, err_sz, "Invalid piece character");
            return false;
        }
        if (file >= 8) {
            set_err(err, err_sz, "Too many squares in rank");
            return false;
        }

        int sq = make_square(file, rank);
        t.bb[color][piece] |= 1ULL << sq;
        file += 1;
        p += 1;
    }

    if (rank != 0 || file != 8) {
        set_err(err, err_sz, "Incomplete board description");
        return false;
    }
    if (*p != ' ') {
        set_err(err, err_sz, "Missing active color");
        return false;
    }
    p += 1;

    if (*p == 'w') {
        t.side_to_move = PIECE_WHITE;
    } else if (*p == 'b') {
        t.side_to_move = PIECE_BLACK;
    } else {
        set_err(err, err_sz, "Invalid active color");
        return false;
    }
    p += 1;

    if (*p != ' ') {
        set_err(err, err_sz, "Missing castling rights");
        return false;
    }
    p += 1;

    t.castling_rights = 0;
    if (*p == '-') {
        p += 1;
    } else {
        bool seen[4] = {false, false, false, false};
        while (*p != '\0' && *p != ' ') {
            switch (*p) {
                case 'K':
                    if (seen[0]) { set_err(err, err_sz, "Duplicate castling right"); return false; }
                    seen[0] = true;
                    t.castling_rights |= CASTLE_WHITE_KING;
                    break;
                case 'Q':
                    if (seen[1]) { set_err(err, err_sz, "Duplicate castling right"); return false; }
                    seen[1] = true;
                    t.castling_rights |= CASTLE_WHITE_QUEEN;
                    break;
                case 'k':
                    if (seen[2]) { set_err(err, err_sz, "Duplicate castling right"); return false; }
                    seen[2] = true;
                    t.castling_rights |= CASTLE_BLACK_KING;
                    break;
                case 'q':
                    if (seen[3]) { set_err(err, err_sz, "Duplicate castling right"); return false; }
                    seen[3] = true;
                    t.castling_rights |= CASTLE_BLACK_QUEEN;
                    break;
                default:
                    set_err(err, err_sz, "Invalid castling rights");
                    return false;
            }
            p += 1;
        }
    }

    if (*p != ' ') {
        set_err(err, err_sz, "Missing en passant square");
        return false;
    }
    p += 1;

    t.ep_square = CHESS_NO_SQUARE;
    if (*p == '-') {
        p += 1;
    } else {
        int ep = CHESS_NO_SQUARE;
        if (!alg_to_square(p[0], p[1], &ep)) {
            set_err(err, err_sz, "Invalid en passant square");
            return false;
        }
        int ep_rank = square_rank(ep);
        if ((t.side_to_move == PIECE_WHITE && ep_rank != 5) ||
            (t.side_to_move == PIECE_BLACK && ep_rank != 2)) {
            set_err(err, err_sz, "Invalid en passant rank");
            return false;
        }
        t.ep_square = ep;
        p += 2;
    }

    if (*p != ' ') {
        set_err(err, err_sz, "Missing halfmove clock");
        return false;
    }
    p += 1;

    char *endptr = NULL;
    long halfmove = strtol(p, &endptr, 10);
    if (endptr == p || halfmove < 0 || halfmove > 10000) {
        set_err(err, err_sz, "Invalid halfmove clock");
        return false;
    }
    t.halfmove_clock = (int)halfmove;
    p = endptr;

    if (*p != ' ') {
        set_err(err, err_sz, "Missing fullmove number");
        return false;
    }
    p += 1;

    long fullmove = strtol(p, &endptr, 10);
    if (endptr == p || fullmove <= 0 || fullmove > 10000) {
        set_err(err, err_sz, "Invalid fullmove number");
        return false;
    }
    t.fullmove_number = (int)fullmove;
    p = endptr;

    while (*p == ' ') {
        p += 1;
    }
    if (*p != '\0') {
        set_err(err, err_sz, "Unexpected trailing FEN data");
        return false;
    }

    chess_state_rebuild_occupancy(&t);

    if (chess_count_bits(t.bb[PIECE_WHITE][PIECE_KING]) != 1 ||
        chess_count_bits(t.bb[PIECE_BLACK][PIECE_KING]) != 1) {
        set_err(err, err_sz, "Position must have one king per side");
        return false;
    }

    int white_king = chess_find_king_square(&t, PIECE_WHITE);
    int black_king = chess_find_king_square(&t, PIECE_BLACK);
    if (white_king < 0 || black_king < 0) {
        set_err(err, err_sz, "Missing king");
        return false;
    }

    if ((abs(square_file(white_king) - square_file(black_king)) <= 1) &&
        (abs(square_rank(white_king) - square_rank(black_king)) <= 1)) {
        set_err(err, err_sz, "Kings cannot be adjacent");
        return false;
    }

    if (chess_in_check(&t, PIECE_WHITE) && chess_in_check(&t, PIECE_BLACK)) {
        set_err(err, err_sz, "Both kings cannot be in check");
        return false;
    }

    t.result = GAME_RESULT_ONGOING;
    t.ply = 0;
    t.has_last_move = false;
    chess_reset_clocks(&t);

    chess_hash_init();
    t.zobrist_hash = chess_hash_full(&t);
    t.hash_history_count = 1;
    t.hash_history[0] = t.zobrist_hash;
    t.irreversible_ply = 0;

    *s = t;
    set_err(err, err_sz, "");
    return true;
}

void chess_export_fen(const GameState *s, char *out, size_t out_sz) {
    if (s == NULL || out == NULL || out_sz == 0) {
        return;
    }

    size_t used = 0;
    out[0] = '\0';

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = make_square(file, rank);
            int color = -1;
            int piece = chess_piece_on_square(s, sq, &color);
            if (piece == PIECE_NONE) {
                empty += 1;
            } else {
                if (empty > 0) {
                    used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "%d", empty);
                    empty = 0;
                }
                used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "%c", piece_to_fen_char(color, piece));
            }
        }
        if (empty > 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "%d", empty);
        }
        if (rank > 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "/");
        }
    }

    used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, " %c ", s->side_to_move == PIECE_WHITE ? 'w' : 'b');

    if (s->castling_rights == 0) {
        used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "-");
    } else {
        if ((s->castling_rights & CASTLE_WHITE_KING) != 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "K");
        }
        if ((s->castling_rights & CASTLE_WHITE_QUEEN) != 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "Q");
        }
        if ((s->castling_rights & CASTLE_BLACK_KING) != 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "k");
        }
        if ((s->castling_rights & CASTLE_BLACK_QUEEN) != 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "q");
        }
    }

    used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, " ");
    if (s->ep_square == CHESS_NO_SQUARE) {
        used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "-");
    } else {
        char sqbuf[3];
        square_to_alg(s->ep_square, sqbuf);
        used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "%s", sqbuf);
    }

    snprintf(out + used, (used < out_sz) ? out_sz - used : 0, " %d %d", s->halfmove_clock, s->fullmove_number);
}

bool chess_move_to_uci(Move m, char out[6]) {
    if (out == NULL) {
        return false;
    }

    int from = move_from(m);
    int to = move_to(m);
    if (from < 0 || from >= 64 || to < 0 || to >= 64) {
        return false;
    }

    out[0] = (char)('a' + square_file(from));
    out[1] = (char)('1' + square_rank(from));
    out[2] = (char)('a' + square_file(to));
    out[3] = (char)('1' + square_rank(to));

    if (move_has_flag(m, MOVE_FLAG_PROMOTION)) {
        int promo = move_promo(m);
        char p = 'q';
        switch (promo) {
            case PIECE_QUEEN: p = 'q'; break;
            case PIECE_ROOK: p = 'r'; break;
            case PIECE_BISHOP: p = 'b'; break;
            case PIECE_KNIGHT: p = 'n'; break;
            default: return false;
        }
        out[4] = p;
        out[5] = '\0';
    } else {
        out[4] = '\0';
    }

    return true;
}

bool chess_move_from_uci(const GameState *s, const char *uci, Move *out) {
    if (s == NULL || uci == NULL || out == NULL) {
        return false;
    }

    size_t len = strlen(uci);
    if (len != 4 && len != 5) {
        return false;
    }

    int from = -1;
    int to = -1;
    if (!alg_to_square((char)tolower((unsigned char)uci[0]), uci[1], &from) ||
        !alg_to_square((char)tolower((unsigned char)uci[2]), uci[3], &to)) {
        return false;
    }

    int promo = CHESS_PROMO_NONE;
    if (len == 5) {
        switch ((char)tolower((unsigned char)uci[4])) {
            case 'q': promo = PIECE_QUEEN; break;
            case 'r': promo = PIECE_ROOK; break;
            case 'b': promo = PIECE_BISHOP; break;
            case 'n': promo = PIECE_KNIGHT; break;
            default: return false;
        }
    }

    Move legal[CHESS_MAX_MOVES];
    int n = chess_generate_legal_moves(s, legal);
    for (int i = 0; i < n; ++i) {
        if (move_from(legal[i]) == from && move_to(legal[i]) == to) {
            if (!move_has_flag(legal[i], MOVE_FLAG_PROMOTION) && promo == CHESS_PROMO_NONE) {
                *out = legal[i];
                return true;
            }
            if (move_has_flag(legal[i], MOVE_FLAG_PROMOTION) && move_promo(legal[i]) == promo) {
                *out = legal[i];
                return true;
            }
        }
    }

    return false;
}
