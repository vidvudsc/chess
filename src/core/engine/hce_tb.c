#include "hce_tb.h"

#include <string.h>

#include "chess_rules.h"
#include "fathom/tbprobe.h"

static int g_tb_largest = 0;

int hce_tb_init(const char *path) {
    if (path == NULL || path[0] == '\0' || strcmp(path, "<empty>") == 0) {
        tb_free();
        g_tb_largest = 0;
        return 0;
    }
    if (!tb_init(path)) {
        g_tb_largest = 0;
        return 0;
    }
    g_tb_largest = (int)TB_LARGEST;
    return g_tb_largest;
}

int hce_tb_largest(void) {
    return g_tb_largest;
}

typedef struct TbPosition {
    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool turn;
} TbPosition;

static bool tb_position(const GameState *s, TbPosition *p) {
    if (g_tb_largest <= 0 || s->castling_rights != 0 ||
        chess_count_bits(s->occ_all) > g_tb_largest) {
        return false;
    }
    p->white = s->occ[PIECE_WHITE];
    p->black = s->occ[PIECE_BLACK];
    p->kings = s->bb[PIECE_WHITE][PIECE_KING] | s->bb[PIECE_BLACK][PIECE_KING];
    p->queens = s->bb[PIECE_WHITE][PIECE_QUEEN] | s->bb[PIECE_BLACK][PIECE_QUEEN];
    p->rooks = s->bb[PIECE_WHITE][PIECE_ROOK] | s->bb[PIECE_BLACK][PIECE_ROOK];
    p->bishops = s->bb[PIECE_WHITE][PIECE_BISHOP] | s->bb[PIECE_BLACK][PIECE_BISHOP];
    p->knights = s->bb[PIECE_WHITE][PIECE_KNIGHT] | s->bb[PIECE_BLACK][PIECE_KNIGHT];
    p->pawns = s->bb[PIECE_WHITE][PIECE_PAWN] | s->bb[PIECE_BLACK][PIECE_PAWN];
    p->ep = (s->ep_square >= 0 && s->ep_square < 64) ? (unsigned)s->ep_square : 0u;
    p->turn = s->side_to_move == PIECE_WHITE;
    return true;
}

bool hce_tb_probe_wdl(const GameState *s, int *wdl) {
    TbPosition p;
    if (s->halfmove_clock != 0 || !tb_position(s, &p)) {
        return false;
    }
    unsigned r = tb_probe_wdl(p.white, p.black, p.kings, p.queens, p.rooks, p.bishops,
                              p.knights, p.pawns, 0, 0, p.ep, p.turn);
    if (r == TB_RESULT_FAILED) {
        return false;
    }
    *wdl = (int)r - 2;
    return true;
}

bool hce_tb_probe_root(const GameState *s, Move *move_out, int *score_out) {
    TbPosition p;
    if (!tb_position(s, &p)) {
        return false;
    }
    unsigned r = tb_probe_root(p.white, p.black, p.kings, p.queens, p.rooks, p.bishops,
                               p.knights, p.pawns, (unsigned)s->halfmove_clock, 0, p.ep,
                               p.turn, NULL);
    if (r == TB_RESULT_FAILED || r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE) {
        return false;
    }
    int from = (int)TB_GET_FROM(r);
    int to = (int)TB_GET_TO(r);
    int promotes = (int)TB_GET_PROMOTES(r);
    static const int k_promo_piece[] = {PIECE_NONE, PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT};
    Move legal[CHESS_MAX_MOVES];
    int n = chess_generate_legal_moves(s, legal);
    for (int i = 0; i < n; ++i) {
        Move m = legal[i];
        if (move_from(m) != from || move_to(m) != to) {
            continue;
        }
        if (move_has_flag(m, MOVE_FLAG_PROMOTION)) {
            if (promotes == TB_PROMOTES_NONE || move_promo(m) != k_promo_piece[promotes]) {
                continue;
            }
        } else if (promotes != TB_PROMOTES_NONE) {
            continue;
        }
        int wdl = (int)TB_GET_WDL(r) - 2;
        int dtz = (int)TB_GET_DTZ(r);
        int score = 0;
        if (wdl == 2) {
            score = HCE_TB_WIN - dtz;
        } else if (wdl == -2) {
            score = -HCE_TB_WIN + dtz;
        }
        *move_out = m;
        *score_out = score;
        return true;
    }
    return false;
}
