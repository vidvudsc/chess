#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "chess_io.h"

static void must(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        assert(cond);
    }
}

static bool has_uci(const GameState *s, const char *uci) {
    Move m;
    return chess_move_from_uci(s, uci, &m);
}

int main(void) {
    GameState s;
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    chess_init(&s, &cfg);

    Move legal[CHESS_MAX_MOVES];
    int legal_count = chess_generate_legal_moves(&s, legal);
    must(legal_count == 20, "Initial position must have 20 legal moves");

    must(chess_perft(&s, 1) == 20ULL, "Perft depth 1 mismatch");
    must(chess_perft(&s, 2) == 400ULL, "Perft depth 2 mismatch");
    must(chess_perft(&s, 3) == 8902ULL, "Perft depth 3 mismatch");
    must(chess_perft(&s, 4) == 197281ULL, "Perft depth 4 mismatch");

    char fen_err[128];
    must(chess_load_fen(&s, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", fen_err, sizeof(fen_err)), "Load castling FEN");
    must(has_uci(&s, "e1g1"), "White king-side castle should be legal");
    must(has_uci(&s, "e1c1"), "White queen-side castle should be legal");

    must(chess_load_fen(&s, "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", fen_err, sizeof(fen_err)), "Load en passant FEN");
    must(has_uci(&s, "e5d6"), "En passant capture should be legal");

    must(chess_load_fen(&s, "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", fen_err, sizeof(fen_err)), "Load promotion FEN");
    must(has_uci(&s, "a7a8q"), "Promotion to queen should be legal");
    must(has_uci(&s, "a7a8n"), "Promotion to knight should be legal");

    must(chess_load_fen(&s, "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", fen_err, sizeof(fen_err)), "Load mate FEN");
    chess_update_result(&s);
    must(s.result == GAME_RESULT_WHITE_WIN, "Checkmate should result in white win");

    must(chess_load_fen(&s, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", fen_err, sizeof(fen_err)), "Load stalemate FEN");
    chess_update_result(&s);
    must(s.result == GAME_RESULT_DRAW_STALEMATE, "Stalemate should be draw");

    chess_init(&s, &cfg);
    const char *rep_seq[] = {
        "g1f3", "g8f6", "f3g1", "f6g8",
        "g1f3", "g8f6", "f3g1", "f6g8",
    };
    for (int i = 0; i < 8; ++i) {
        Move m;
        must(chess_move_from_uci(&s, rep_seq[i], &m), "Repetition move parse");
        must(chess_make_move(&s, m), "Repetition move legal");
    }
    must(s.result == GAME_RESULT_DRAW_REPETITION, "Threefold repetition should auto-draw");

    chess_init(&s, &cfg);
    char start_fen[256];
    chess_export_fen(&s, start_fen, sizeof(start_fen));
    uint64_t start_hash = s.zobrist_hash;

    Move m;
    must(chess_move_from_uci(&s, "e2e4", &m), "Parse e2e4");
    must(chess_make_move(&s, m), "Apply e2e4");
    must(chess_undo_move(&s), "Undo e2e4");

    char restored_fen[256];
    chess_export_fen(&s, restored_fen, sizeof(restored_fen));
    must(strcmp(start_fen, restored_fen) == 0, "Undo must restore full FEN");
    must(start_hash == s.zobrist_hash, "Undo must restore hash");

    printf("test_rules: OK\n");
    return 0;
}
