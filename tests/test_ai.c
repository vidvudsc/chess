#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "chess_ai.h"
#include "chess_io.h"
#include "hce_internal.h"

static void must(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        assert(cond);
    }
}

static void write_tiny_quant_nn_model(const char *path) {
    enum {
        halfkp_dim = 64 * 10 * 64,
        dummy_index = halfkp_dim,
        acc_rows = halfkp_dim + 1,
    };
    FILE *fp = fopen(path, "wb");
    must(fp != NULL, "Create tiny quantized NN model");

    const char magic[8] = {'C', 'H', 'N', 'N', 'U', 'E', '1', '\0'};
    uint32_t version = 2;
    uint32_t halfkp = halfkp_dim;
    uint32_t acc_dim = 1;
    uint32_t hidden_dim = 1;
    uint32_t dummy = dummy_index;
    float cp_scale = 100.0f;
    float acc_scale = 1.0f;
    float fc1_scale = 0.1f;
    float fc2_scale = 1.0f;
    float out_scale = 1.0f;

    must(fwrite(magic, 1, sizeof(magic), fp) == sizeof(magic), "Write NN magic");
    must(fwrite(&version, sizeof(version), 1, fp) == 1, "Write NN version");
    must(fwrite(&halfkp, sizeof(halfkp), 1, fp) == 1, "Write NN halfkp dim");
    must(fwrite(&acc_dim, sizeof(acc_dim), 1, fp) == 1, "Write NN accumulator dim");
    must(fwrite(&hidden_dim, sizeof(hidden_dim), 1, fp) == 1, "Write NN hidden dim");
    must(fwrite(&dummy, sizeof(dummy), 1, fp) == 1, "Write NN dummy index");
    must(fwrite(&cp_scale, sizeof(cp_scale), 1, fp) == 1, "Write NN cp scale");
    must(fwrite(&acc_scale, sizeof(acc_scale), 1, fp) == 1, "Write NN acc scale");
    must(fwrite(&fc1_scale, sizeof(fc1_scale), 1, fp) == 1, "Write NN fc1 scale");
    must(fwrite(&fc2_scale, sizeof(fc2_scale), 1, fp) == 1, "Write NN fc2 scale");
    must(fwrite(&out_scale, sizeof(out_scale), 1, fp) == 1, "Write NN out scale");

    int16_t *acc_weight = (int16_t *)calloc((size_t)acc_rows, sizeof(*acc_weight));
    must(acc_weight != NULL, "Allocate tiny NN accumulator");

    int white_king_e1 = 4;
    int white_queen_f1 = 5;
    int own_queen_plane = 4;
    int queen_feature = (white_king_e1 * 10 + own_queen_plane) * 64 + white_queen_f1;
    acc_weight[queen_feature] = 10;

    int8_t fc1_weight[2] = {1, 0};
    float fc1_bias[1] = {0.0f};
    int8_t fc2_weight[1] = {1};
    float fc2_bias[1] = {0.0f};
    int8_t out_weight[1] = {1};
    float out_bias = 0.0f;

    must(fwrite(acc_weight, sizeof(*acc_weight), (size_t)acc_rows, fp) == (size_t)acc_rows,
         "Write NN accumulator");
    must(fwrite(fc1_weight, sizeof(fc1_weight), 1, fp) == 1, "Write NN fc1 weights");
    must(fwrite(fc1_bias, sizeof(fc1_bias), 1, fp) == 1, "Write NN fc1 bias");
    must(fwrite(fc2_weight, sizeof(fc2_weight), 1, fp) == 1, "Write NN fc2 weights");
    must(fwrite(fc2_bias, sizeof(fc2_bias), 1, fp) == 1, "Write NN fc2 bias");
    must(fwrite(out_weight, sizeof(out_weight), 1, fp) == 1, "Write NN output weights");
    must(fwrite(&out_bias, sizeof(out_bias), 1, fp) == 1, "Write NN output bias");

    free(acc_weight);
    must(fclose(fp) == 0, "Close tiny quantized NN model");
}

int main(void) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };

    GameState s;
    chess_init(&s, &cfg);

    AiSearchConfig ai_cfg = {
        .think_time_ms = 150,
        .max_depth = 6,
    };
    AiSearchResult result;

    char temp_book_path[256];
    snprintf(temp_book_path, sizeof(temp_book_path), "/tmp/chess_test_book_%ld.txt", (long)getpid());
    FILE *book_fp = fopen(temp_book_path, "w");
    must(book_fp != NULL, "Create temporary opening book");
    must(fputs("Test Opening|e2e4 e7e5 g1f3\n", book_fp) >= 0, "Write temporary curated opening line");
    must(fclose(book_fp) == 0, "Close temporary opening book");
    must(chess_ai_opening_book_set_path(temp_book_path), "Load temporary opening book");
    must(chess_ai_opening_book_is_loaded(), "Temporary opening book should be loaded");

    must(chess_ai_pick_move(&s, &ai_cfg, &result), "AI should produce a move from start position");
    must(result.found_move, "AI move should be flagged as found");
    must(result.used_opening_book, "AI should use opening book from start position");
    must(chess_is_move_legal(&s, result.best_move), "AI move must be legal");
    char verify_uci[6] = {0};
    must(chess_move_to_uci(result.best_move, verify_uci), "AI book move should export to UCI");
    must(strcmp(verify_uci, "e2e4") == 0, "AI should follow temporary curated opening move e2e4");

    ChessEngineRequest engine_req = {
        .kind = CHESS_ENGINE_REQUEST_BEST_MOVE,
        .think_time_ms = 120,
        .max_depth = 6,
    };
    ChessEngineResponse engine_res;
    char err[128] = {0};
    must(chess_engine_query(&s, &engine_req, &engine_res), "Unified engine API should return best move");
    must(engine_res.ok, "Unified engine API best-move response should be ok");
    must(engine_res.found_move, "Unified engine API best-move should mark found_move");
    must(engine_res.used_opening_book, "Unified engine API should report opening-book move");
    must(chess_is_move_legal(&s, engine_res.best_move), "Unified engine API best move must be legal");

    engine_req.kind = CHESS_ENGINE_REQUEST_EVAL_FAST;
    must(chess_engine_query(&s, &engine_req, &engine_res), "Unified engine API should return fast eval");
    must(engine_res.ok, "Unified engine API fast eval response should be ok");
    must(engine_res.score_cp_white == chess_ai_eval_fast_cp(&s),
         "Unified engine API fast eval should match direct fast eval");
    ChessEvalBreakdown breakdown;
    must(chess_ai_eval_breakdown(&s, &breakdown), "Eval breakdown should be available");
    must(breakdown.score_cp_stm == chess_ai_eval_fast_cp(&s),
         "Eval breakdown STM score should include tempo and match fast eval");
    must(chess_load_fen(&s, "4k3/8/8/8/8/8/8/4K3 b - - 0 1", err, sizeof(err)),
         "Load black-to-move eval breakdown FEN");
    must(chess_ai_eval_breakdown(&s, &breakdown), "Black-to-move eval breakdown should be available");
    must(breakdown.score_cp_white == chess_ai_eval_fast_cp(&s),
         "Eval breakdown white score should include tempo with the correct side-to-move sign");

    engine_req.kind = CHESS_ENGINE_REQUEST_EVAL_DEEP;
    must(chess_engine_query(&s, &engine_req, &engine_res), "Unified engine API should return deep eval");
    must(engine_res.ok, "Unified engine API deep eval response should be ok");
    must(engine_res.score_cp_white == chess_ai_eval_cp(&s),
         "Unified engine API deep eval should match direct deep eval");

    char temp_nn_path[256];
    snprintf(temp_nn_path, sizeof(temp_nn_path), "/tmp/chess_test_nn_%ld.bin", (long)getpid());
    write_tiny_quant_nn_model(temp_nn_path);
    must(chess_ai_set_nn_model_path(temp_nn_path), "Load tiny quantized NN model");
    must(chess_ai_set_backend(CHESS_AI_BACKEND_NN), "NN backend should be selectable with loaded model");
    must(chess_load_fen(&s, "4k3/8/8/8/8/8/8/4K3 w - - 0 1", err, sizeof(err)),
         "Load bare-kings NN regression FEN");
    int nn_bare_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", err, sizeof(err)),
         "Load queen NN regression FEN");
    int nn_queen_eval = chess_ai_eval_fast_cp(&s);
    must(nn_queen_eval > nn_bare_eval + 50,
         "Quantized NN accumulator should include non-king piece features");
    must(chess_ai_set_backend(CHESS_AI_BACKEND_CLASSIC), "Classic backend should be restored after NN regression");

    must(chess_load_fen(&s, "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - - 2 10", err, sizeof(err)),
         "Load experimental backend regression FEN");
    int classic_fast_eval = chess_ai_eval_fast_cp(&s);
    must(chess_ai_set_backend(CHESS_AI_BACKEND_EXPERIMENTAL), "Experimental backend should be selectable");
    must(strcmp(chess_ai_backend_name(CHESS_AI_BACKEND_EXPERIMENTAL), "experimental") == 0,
         "Experimental backend should have a stable UCI name");
    int experimental_fast_eval = chess_ai_eval_fast_cp(&s);
    must(experimental_fast_eval == classic_fast_eval,
         "Experimental backend should mirror Classic eval until experiments are enabled");
    ai_cfg.think_time_ms = 50;
    ai_cfg.max_depth = 4;
    must(chess_ai_pick_move(&s, &ai_cfg, &result), "Experimental backend should search");
    must(result.found_move, "Experimental backend should return a move");
    must(chess_is_move_legal(&s, result.best_move), "Experimental backend move must be legal");
    must(chess_ai_set_backend(CHESS_AI_BACKEND_CLASSIC), "Classic backend should be restored after experimental regression");

    must(chess_load_fen(&s, "6k1/5ppp/8/8/8/8/6PP/6K1 w - - 0 1", err, sizeof(err)), "Load simple endgame FEN");
    must(chess_ai_pick_move(&s, &ai_cfg, &result), "AI should produce move in endgame");
    must(chess_is_move_legal(&s, result.best_move), "AI endgame move must be legal");

    char temp_nonstart_book_path[256];
    snprintf(temp_nonstart_book_path, sizeof(temp_nonstart_book_path), "/tmp/chess_test_nonstart_book_%ld.txt", (long)getpid());
    book_fp = fopen(temp_nonstart_book_path, "w");
    must(book_fp != NULL, "Create temporary non-start opening book");
    must(fputs("Problem Line|d2d4 d7d5\n", book_fp) >= 0, "Write temporary non-start opening line");
    must(fclose(book_fp) == 0, "Close temporary non-start opening book");
    must(chess_ai_opening_book_set_path(temp_nonstart_book_path), "Load temporary non-start opening book");
    must(chess_ai_opening_book_is_loaded(), "Temporary non-start opening book should be loaded");

    must(chess_load_fen(&s, "8/2r5/1k6/8/4PP2/PP5P/1K1R4/8 w - - 5 43", err, sizeof(err)),
         "Load arbitrary non-start FEN that previously hit the opening book");
    ai_cfg.think_time_ms = 40;
    ai_cfg.max_depth = 8;
    must(chess_ai_pick_move(&s, &ai_cfg, &result), "AI should search arbitrary non-start FEN");
    must(!result.used_opening_book, "AI should not use opening book on arbitrary non-start FEN");
    must(result.depth_reached > 0 || result.nodes > 0,
         "AI should search arbitrary non-start FEN instead of returning a book move");

    must(chess_load_fen(&s, "r3k2r/ppp2ppp/8/3N4/4P3/8/PPP2PPP/R3K2R w KQkq - 0 1", err, sizeof(err)),
         "Load outpost-strength FEN");
    int outpost_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "r3k2r/ppp2ppp/8/8/4P3/5N2/PPP2PPP/R3K2R w KQkq - 0 1", err, sizeof(err)),
         "Load passive-knight FEN");
    int passive_eval = chess_ai_eval_fast_cp(&s);
    must(outpost_eval > passive_eval + 10,
         "Supported knight outpost should score above a passive knight setup");

    must(chess_load_fen(&s, "4k3/8/2q5/3P4/8/8/4Q3/4K3 w - - 0 1", err, sizeof(err)),
         "Load loose-queen threat FEN");
    int loose_queen_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "8/3k4/2q5/3P4/8/8/4Q3/4K3 w - - 0 1", err, sizeof(err)),
         "Load defended-queen threat FEN");
    int defended_queen_eval = chess_ai_eval_fast_cp(&s);
    must(loose_queen_eval > defended_queen_eval + 20,
         "Loose attacked queen should evaluate worse for the defending side");

    must(chess_load_fen(&s, "4k2r/Q1p1n2p/p1nqbp2/4p3/3p4/P2P1N2/1PP1PPPP/R3KB1R b KQk - 0 15", err, sizeof(err)),
         "Load trapped-queen regression FEN");
    int trapped_queen_eval = chess_ai_eval_fast_cp(&s);
    must(trapped_queen_eval < 0,
         "Fast eval should not call a trapped, immediately capturable queen position favorable for white");

    must(chess_load_fen(&s, "6k1/6r1/8/6q1/8/8/5PPP/4R1K1 w - - 0 1", err, sizeof(err)),
         "Load sheltered king-safety FEN");
    int sheltered_king_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "6k1/6r1/8/6q1/8/8/PPP5/4R1K1 w - - 0 1", err, sizeof(err)),
         "Load exposed king-safety FEN");
    int exposed_king_eval = chess_ai_eval_fast_cp(&s);
    must(sheltered_king_eval > exposed_king_eval + 15,
         "Sheltered king should evaluate better than an exposed king under file pressure");

    must(chess_load_fen(&s, "4k3/8/8/8/8/P1P1P1P1/8/2B1K3 w - - 0 1", err, sizeof(err)),
         "Load bad-bishop FEN");
    int bad_bishop_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "4k3/8/8/8/8/1P1P1P1P/8/2B1K3 w - - 0 1", err, sizeof(err)),
         "Load good-bishop FEN");
    int good_bishop_eval = chess_ai_eval_fast_cp(&s);
    must(good_bishop_eval > bad_bishop_eval + 10,
         "Bishop with opposite-color pawns should evaluate better than a congested bad bishop");

    must(chess_load_fen(&s, "4k3/p6p/8/3PP3/4K3/8/8/8 w - - 0 1", err, sizeof(err)),
         "Load connected passers FEN");
    int connected_passers_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "4k3/p6p/8/2P2P2/4K3/8/8/8 w - - 0 1", err, sizeof(err)),
         "Load separated passers FEN");
    int separated_passers_eval = chess_ai_eval_fast_cp(&s);
    must(connected_passers_eval > separated_passers_eval + 10,
         "Connected passers should evaluate better than separated passers");

    must(chess_load_fen(&s, "6k1/8/8/P4P2/2Kpp3/8/8/8 w - - 0 1", err, sizeof(err)),
         "Load outside passer FEN");
    int outside_passer_eval = chess_ai_eval_fast_cp(&s);
    must(chess_load_fen(&s, "6k1/8/8/3P1P2/2Kpp3/8/8/8 w - - 0 1", err, sizeof(err)),
         "Load central passer FEN");
    int central_passer_eval = chess_ai_eval_fast_cp(&s);
    must(outside_passer_eval > central_passer_eval + 8,
         "Outside passer should evaluate better than a similar central passer");

    must(chess_load_fen(&s, "7k/5KQ1/8/8/8/8/8/8 w - - 0 1", err, sizeof(err)), "Load mate-in-one FEN");
    ai_cfg.think_time_ms = 120;
    ai_cfg.max_depth = 8;
    must(chess_ai_pick_move(&s, &ai_cfg, &result), "AI should produce move in mate-in-one FEN");
    must(chess_is_move_legal(&s, result.best_move), "AI mate-in-one move must be legal");
    must(chess_make_move(&s, result.best_move), "AI mate-in-one move should apply");
    must(s.result == GAME_RESULT_WHITE_WIN, "AI should find checkmate in one when available");

    must(chess_load_fen(&s, "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R4RK1 w - - 2 10", err, sizeof(err)),
         "Load balanced middlegame FEN");
    ai_cfg.think_time_ms = 40;
    ai_cfg.max_depth = 10;
    for (int ply = 0; ply < 80 && s.result == GAME_RESULT_ONGOING; ++ply) {
        must(chess_ai_pick_move(&s, &ai_cfg, &result), "AI selfplay move generation in stress loop");
        must(result.found_move, "AI stress move should be found");
        must(chess_is_move_legal(&s, result.best_move), "AI stress move must be legal");
        must(chess_make_move(&s, result.best_move), "AI stress move should apply");
    }

    must(chess_ai_set_backend(CHESS_AI_BACKEND_CLASSIC), "Classic backend should be selectable");
    must(chess_load_fen(&s, "3q3k/5Q1p/5pp1/2B1R3/1P6/P5PP/2pr1P1K/8 b - - 0 1", err, sizeof(err)),
         "Load classic crash regression FEN");
    ai_cfg.think_time_ms = 200;
    ai_cfg.max_depth = 12;
    must(chess_ai_pick_move(&s, &ai_cfg, &result), "Classic backend should survive crash regression FEN");
    must(result.found_move, "Classic backend crash regression FEN should return a move");
    must(chess_is_move_legal(&s, result.best_move), "Classic backend crash regression move must be legal");

    must(chess_load_fen(&s, "3q4/3b3k/7p/6p1/4Qp2/1r1P3N/4PPPP/3K1B1R b - - 5 35", err, sizeof(err)),
         "Load repetition-trap regression FEN");
    const char *repetition_trap_setup[] = {
        "h7h8", "e4d4", "h8h7", "d4e4", "h7h8", "e4d4", "h8h7",
    };
    for (size_t i = 0; i < sizeof(repetition_trap_setup) / sizeof(repetition_trap_setup[0]); ++i) {
        Move prep = 0;
        must(chess_move_from_uci(&s, repetition_trap_setup[i], &prep), "Parse repetition-trap setup move");
        must(chess_make_move(&s, prep), "Apply repetition-trap setup move");
    }
    Move repetition_draw_move = 0;
    must(chess_move_from_uci(&s, "d4e4", &repetition_draw_move), "Parse repetition-draw move");
    GameState draw_probe = s;
    must(chess_make_move(&draw_probe, repetition_draw_move), "Apply repetition-draw move");
    must(draw_probe.result == GAME_RESULT_DRAW_REPETITION,
         "Repetition-draw move should immediately end the game as a draw");

    ai_cfg.think_time_ms = 200;
    ai_cfg.max_depth = 12;
    must(chess_ai_pick_move(&s, &ai_cfg, &result), "Classic backend should search repetition-trap position");
    must(result.found_move, "Classic backend repetition-trap position should return a move");
    must(chess_move_to_uci(result.best_move, verify_uci), "Classic backend repetition-trap move should export to UCI");
    must(strcmp(verify_uci, "d4e4") != 0,
         "Classic backend should avoid a repetition move that immediately forces a draw");

    must(chess_load_fen(&s, "6k1/8/8/8/8/8/5K2/4R3 w - - 0 1", err, sizeof(err)),
         "Load twofold-repetition search regression FEN");
    const char *twofold_cycle[] = {"e1e2", "g8h8", "e2e1", "h8g8"};
    for (size_t i = 0; i < sizeof(twofold_cycle) / sizeof(twofold_cycle[0]); ++i) {
        Move cycle_move = 0;
        must(chess_move_from_uci(&s, twofold_cycle[i], &cycle_move), "Parse twofold-repetition regression move");
        must(chess_make_move(&s, cycle_move), "Apply twofold-repetition regression move");
    }
    must(s.result == GAME_RESULT_ONGOING,
         "Second occurrence should remain an ongoing game under the real rules");
    must(hce_score_search_draw_stm(&s) == INT_MIN,
         "Search draw detection should not flatten a mere twofold repetition to a draw");

    for (size_t i = 0; i < sizeof(twofold_cycle) / sizeof(twofold_cycle[0]); ++i) {
        Move cycle_move = 0;
        must(chess_move_from_uci(&s, twofold_cycle[i], &cycle_move), "Parse threefold-repetition regression move");
        must(chess_make_move(&s, cycle_move), "Apply threefold-repetition regression move");
    }
    must(s.result == GAME_RESULT_DRAW_REPETITION,
         "Third occurrence should still be recognized as a draw");
    must(hce_score_search_draw_stm(&s) == 0,
         "Search draw detection should still score true threefold repetition as a draw");

    unlink(temp_book_path);
    unlink(temp_nonstart_book_path);
    unlink(temp_nn_path);

    printf("test_ai: OK\n");
    return 0;
}
