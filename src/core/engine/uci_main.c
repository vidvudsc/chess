#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess_ai.h"
#include "chess_io.h"
#include "chess_rules.h"
#include "hce_internal.h"

typedef struct UciOptions {
    int think_time_ms;
    int max_depth;
    int hce_rfp_margin_per_depth;
    int hce_null_base_reduction;
    int hce_null_depth_divisor;
    int hce_lmr_base_reduction;
    int hce_lmr_depth_bonus_at;
    int hce_lmr_move_bonus_at;
    ChessAiBackend backend;
    char book_file_path[512];
    char nn_model_path[512];
} UciOptions;

static void uci_search_info_callback(int depth,
                                     int score_cp,
                                     Move best_move,
                                     uint64_t nodes,
                                     int elapsed_ms,
                                     void *user_data);

static void trim_in_place(char *s) {
    if (s == NULL) {
        return;
    }
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }
    size_t start = 0;
    while (s[start] != '\0' && isspace((unsigned char)s[start])) {
        ++start;
    }
    if (start > 0) {
        memmove(s, s + start, len - start + 1);
    }
}

static bool starts_with(const char *s, const char *prefix) {
    if (s == NULL || prefix == NULL) {
        return false;
    }
    size_t p = strlen(prefix);
    return strncmp(s, prefix, p) == 0;
}

static bool str_ieq(const char *a, const char *b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static bool parse_int_token(const char *s, int *out) {
    if (s == NULL || out == NULL || *s == '\0') {
        return false;
    }
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    if (v < -1000000000L || v > 1000000000L) {
        return false;
    }
    *out = (int)v;
    return true;
}

static char *next_token(char **cursor) {
    if (cursor == NULL || *cursor == NULL) {
        return NULL;
    }

    char *p = *cursor;
    while (*p != '\0' && isspace((unsigned char)*p)) {
        ++p;
    }
    if (*p == '\0') {
        *cursor = p;
        return NULL;
    }

    char *start = p;
    while (*p != '\0' && !isspace((unsigned char)*p)) {
        ++p;
    }
    if (*p != '\0') {
        *p = '\0';
        ++p;
    }
    *cursor = p;
    return start;
}

static bool set_spin_option(const char *name_buf,
                            const char *value_buf,
                            const char *option_name,
                            int min_value,
                            int max_value,
                            int *target) {
    if (!str_ieq(name_buf, option_name) || target == NULL) {
        return false;
    }
    int parsed = 0;
    if (parse_int_token(value_buf, &parsed) && parsed >= min_value && parsed <= max_value) {
        *target = parsed;
        printf("info string %s set to %d\n", option_name, *target);
        fflush(stdout);
    }
    return true;
}

static MatchConfig make_headless_match_config(void) {
    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_BOT_STUB,
        .black_kind = PLAYER_BOT_STUB,
    };
    return cfg;
}

static void reset_start_position(GameState *state) {
    MatchConfig cfg = make_headless_match_config();
    chess_init(state, &cfg);
}

static void parse_setoption(const char *line, UciOptions *opt) {
    if (line == NULL || opt == NULL) {
        return;
    }

    const char *name_tok = strstr(line, "name ");
    if (name_tok == NULL) {
        return;
    }
    name_tok += 5;

    const char *value_tok = strstr(name_tok, " value ");
    char name_buf[128];
    char value_buf[768];
    name_buf[0] = '\0';
    value_buf[0] = '\0';

    if (value_tok != NULL) {
        size_t name_len = (size_t)(value_tok - name_tok);
        if (name_len >= sizeof(name_buf)) {
            name_len = sizeof(name_buf) - 1;
        }
        memcpy(name_buf, name_tok, name_len);
        name_buf[name_len] = '\0';
        snprintf(value_buf, sizeof(value_buf), "%s", value_tok + 7);
    } else {
        snprintf(name_buf, sizeof(name_buf), "%s", name_tok);
    }

    trim_in_place(name_buf);
    trim_in_place(value_buf);

    if (str_ieq(name_buf, "MoveTime")) {
        int ms = 0;
        if (parse_int_token(value_buf, &ms) && ms > 0) {
            opt->think_time_ms = ms;
            printf("info string movetime set to %d\n", opt->think_time_ms);
            fflush(stdout);
        }
        return;
    }

    if (str_ieq(name_buf, "MaxDepth")) {
        int depth = 0;
        if (parse_int_token(value_buf, &depth) && depth > 0) {
            opt->max_depth = depth;
            printf("info string max depth set to %d\n", opt->max_depth);
            fflush(stdout);
        }
        return;
    }

    if (set_spin_option(name_buf, value_buf, "HceRfpMargin", 0, 500, &opt->hce_rfp_margin_per_depth) ||
        set_spin_option(name_buf, value_buf, "HceNullBase", 0, 6, &opt->hce_null_base_reduction) ||
        set_spin_option(name_buf, value_buf, "HceNullDepthDivisor", 0, 12, &opt->hce_null_depth_divisor) ||
        set_spin_option(name_buf, value_buf, "HceLmrBase", 0, 4, &opt->hce_lmr_base_reduction) ||
        set_spin_option(name_buf, value_buf, "HceLmrDepthBonusAt", 0, 16, &opt->hce_lmr_depth_bonus_at) ||
        set_spin_option(name_buf, value_buf, "HceLmrMoveBonusAt", 0, 32, &opt->hce_lmr_move_bonus_at)) {
        return;
    }

    if (str_ieq(name_buf, "BookFile") || str_ieq(name_buf, "OpeningBook")) {
        if (value_buf[0] != '\0') {
            if (chess_ai_opening_book_set_path(value_buf)) {
                snprintf(opt->book_file_path, sizeof(opt->book_file_path), "%s", value_buf);
                printf("info string opening book path set to %s\n", opt->book_file_path);
                fflush(stdout);
            } else {
                printf("info string failed to set opening book path: %s\n", value_buf);
                fflush(stdout);
            }
        }
        return;
    }

    if (str_ieq(name_buf, "Backend")) {
        ChessAiBackend backend = CHESS_AI_BACKEND_CLASSIC;
        if (str_ieq(value_buf, "nn")) {
            backend = CHESS_AI_BACKEND_NN;
        } else if (str_ieq(value_buf, "experimental")) {
            backend = CHESS_AI_BACKEND_EXPERIMENTAL;
        }
        if (chess_ai_set_backend(backend)) {
            opt->backend = backend;
            printf("info string backend set to %s\n", chess_ai_backend_name(opt->backend));
        } else {
            printf("info string failed to set backend to %s\n", value_buf[0] ? value_buf : "classic");
        }
        fflush(stdout);
        return;
    }

    if (str_ieq(name_buf, "NNModel")) {
        if (value_buf[0] != '\0') {
            if (chess_ai_set_nn_model_path(value_buf)) {
                snprintf(opt->nn_model_path, sizeof(opt->nn_model_path), "%s", value_buf);
                printf("info string nn model path set to %s\n", opt->nn_model_path);
            } else {
                printf("info string failed to set nn model path: %s\n", value_buf);
            }
            fflush(stdout);
        }
    }
}

static void apply_moves_uci(GameState *state, char *moves) {
    if (state == NULL || moves == NULL) {
        return;
    }

    char *cursor = moves;
    char *tok = NULL;
    while ((tok = next_token(&cursor)) != NULL) {
        // Adjudication (e.g. our stricter insufficient-material rule) is the
        // GUI's job, not ours: the internal engine may consider a position
        // drawn (KN vs KN) that Lichess keeps playing. Refusing history moves
        // here desyncs the board and produces illegal bestmoves.
        if (state->result != GAME_RESULT_ONGOING) {
            chess_set_result(state, GAME_RESULT_ONGOING);
        }
        Move mv;
        if (!chess_move_from_uci(state, tok, &mv) || !chess_make_move(state, mv)) {
            printf("info string invalid move in position command: %s\n", tok);
            fflush(stdout);
            return;
        }
    }
}

static void handle_position(GameState *state, const char *line) {
    if (state == NULL || line == NULL) {
        return;
    }

    char buf[2048];
    snprintf(buf, sizeof(buf), "%s", line);
    trim_in_place(buf);

    char *p = buf;
    if (starts_with(p, "position")) {
        p += 8;
        while (*p != '\0' && isspace((unsigned char)*p)) {
            ++p;
        }
    }

    if (starts_with(p, "startpos")) {
        reset_start_position(state);
        p += 8;
        while (*p != '\0' && isspace((unsigned char)*p)) {
            ++p;
        }
        if (starts_with(p, "moves")) {
            p += 5;
            while (*p != '\0' && isspace((unsigned char)*p)) {
                ++p;
            }
            apply_moves_uci(state, p);
        }
        return;
    }

    if (starts_with(p, "fen")) {
        p += 3;
        while (*p != '\0' && isspace((unsigned char)*p)) {
            ++p;
        }

        char *moves_tok = strstr(p, " moves ");
        char fen[256];
        fen[0] = '\0';
        if (moves_tok != NULL) {
            size_t fen_len = (size_t)(moves_tok - p);
            while (fen_len > 0 && isspace((unsigned char)p[fen_len - 1])) {
                --fen_len;
            }
            if (fen_len >= sizeof(fen)) {
                fen_len = sizeof(fen) - 1;
            }
            memcpy(fen, p, fen_len);
            fen[fen_len] = '\0';
            moves_tok += 7;
        } else {
            snprintf(fen, sizeof(fen), "%s", p);
            trim_in_place(fen);
        }

        if (fen[0] == '\0') {
            printf("info string empty FEN in position command\n");
            fflush(stdout);
            return;
        }

        char err[160];
        if (!chess_load_fen(state, fen, err, sizeof(err))) {
            printf("info string invalid FEN: %s\n", err);
            fflush(stdout);
            return;
        }

        if (moves_tok != NULL) {
            apply_moves_uci(state, moves_tok);
        }
    }
}

static void handle_go(GameState *state, const char *line, const UciOptions *opt) {
    if (state == NULL || line == NULL || opt == NULL) {
        printf("bestmove 0000\n");
        fflush(stdout);
        return;
    }

    int think_ms = opt->think_time_ms;
    int max_depth = opt->max_depth;
    bool saw_movetime = false;
    int wtime = -1, btime = -1, winc = 0, binc = 0;

    char buf[2048];
    snprintf(buf, sizeof(buf), "%s", line);
    trim_in_place(buf);
    char *p = buf;
    if (starts_with(p, "go")) {
        p += 2;
    }
    while (*p != '\0' && isspace((unsigned char)*p)) {
        ++p;
    }

    char *cursor = p;
    char *tok = NULL;
    while ((tok = next_token(&cursor)) != NULL) {
        if (strcmp(tok, "movetime") == 0) {
            char *val = next_token(&cursor);
            int parsed = 0;
            if (parse_int_token(val, &parsed) && parsed > 0) {
                think_ms = parsed;
                saw_movetime = true;
            }
        } else if (strcmp(tok, "depth") == 0) {
            char *val = next_token(&cursor);
            int parsed = 0;
            if (parse_int_token(val, &parsed) && parsed > 0) {
                max_depth = parsed;
            }
        } else if (strcmp(tok, "wtime") == 0) {
            char *val = next_token(&cursor);
            parse_int_token(val, &wtime);
        } else if (strcmp(tok, "btime") == 0) {
            char *val = next_token(&cursor);
            parse_int_token(val, &btime);
        } else if (strcmp(tok, "winc") == 0) {
            char *val = next_token(&cursor);
            parse_int_token(val, &winc);
        } else if (strcmp(tok, "binc") == 0) {
            char *val = next_token(&cursor);
            parse_int_token(val, &binc);
        } else if (strcmp(tok, "infinite") == 0) {
            think_ms = 2000;
        }
    }

    int hard_ms = 0;
    if (!saw_movetime) {
        int remaining = (state->side_to_move == PIECE_WHITE) ? wtime : btime;
        int inc = (state->side_to_move == PIECE_WHITE) ? winc : binc;
        if (remaining > 0) {
            int budget = remaining / 25 + inc / 2;
            if (remaining < 2000) {
                budget = remaining / 8 + inc / 3;
            }
            if (budget < 20) {
                budget = 20;
            }
            if (budget > 5000) {
                budget = 5000;
            }
            think_ms = budget;

            // With a real clock the budget is a soft target: allow unstable
            // searches to run up to 2x, but never more than a safe slice of
            // the remaining clock.
            hard_ms = budget * 2;
            int clock_cap = remaining / 6;
            if (hard_ms > clock_cap) {
                hard_ms = clock_cap;
            }
            if (hard_ms < think_ms) {
                hard_ms = 0;
            }
        }
    }

    if (max_depth < 1) {
        max_depth = 1;
    }
    if (max_depth > 32) {
        max_depth = 32;
    }
    if (think_ms < 1) {
        think_ms = 1;
    }

    AiSearchConfig cfg = {
        .think_time_ms = think_ms,
        .hard_time_ms = hard_ms,
        .max_depth = max_depth,
        .hce_rfp_margin_per_depth = opt->hce_rfp_margin_per_depth,
        .hce_null_base_reduction = opt->hce_null_base_reduction,
        .hce_null_depth_divisor = opt->hce_null_depth_divisor,
        .hce_lmr_base_reduction = opt->hce_lmr_base_reduction,
        .hce_lmr_depth_bonus_at = opt->hce_lmr_depth_bonus_at,
        .hce_lmr_move_bonus_at = opt->hce_lmr_move_bonus_at,
        .info_callback = uci_search_info_callback,
        .info_user_data = NULL,
    };
    AiSearchResult res;
    if (!chess_ai_pick_move(state, &cfg, &res) || !res.found_move) {
        printf("bestmove 0000\n");
        fflush(stdout);
        return;
    }

    char best_uci[6];
    if (!chess_move_to_uci(res.best_move, best_uci)) {
        printf("bestmove 0000\n");
        fflush(stdout);
        return;
    }

    int nps = 0;
    if (res.elapsed_ms > 0) {
        nps = (int)((res.nodes * 1000ULL) / (uint64_t)res.elapsed_ms);
    }
    if (res.used_opening_book) {
        const char *book_path = chess_ai_opening_book_path();
        if (book_path != NULL) {
            printf("info string book move from %s\n", book_path);
        } else {
            printf("info string book move\n");
        }
    }
    printf("info depth %d score cp %d nodes %llu time %d nps %d pv %s\n",
           res.depth_reached,
           res.score_cp,
           (unsigned long long)res.nodes,
           res.elapsed_ms,
           nps,
           best_uci);
    printf("bestmove %s\n", best_uci);
    fflush(stdout);
}

static void uci_search_info_callback(int depth,
                                     int score_cp,
                                     Move best_move,
                                     uint64_t nodes,
                                     int elapsed_ms,
                                     void *user_data) {
    (void)user_data;

    char best_uci[6];
    if (!chess_move_to_uci(best_move, best_uci)) {
        snprintf(best_uci, sizeof(best_uci), "0000");
    }

    int nps = 0;
    if (elapsed_ms > 0) {
        nps = (int)((nodes * 1000ULL) / (uint64_t)elapsed_ms);
    }

    printf("info depth %d score cp %d nodes %llu time %d nps %d pv %s\n",
           depth,
           score_cp,
           (unsigned long long)nodes,
           elapsed_ms,
           nps,
           best_uci);
    fflush(stdout);
}

static void print_uci_intro(const UciOptions *opt) {
    printf("id name Chess HCE Engine\n");
    printf("id author Codex + vidvudscalitis\n");
    printf("option name Backend type combo default classic var classic var nn var experimental\n");
    printf("option name MoveTime type spin default %d min 1 max 10000\n", opt->think_time_ms);
    printf("option name MaxDepth type spin default %d min 1 max 32\n", opt->max_depth);
    printf("option name HceRfpMargin type spin default 0 min 0 max 500\n");
    printf("option name HceNullBase type spin default 0 min 0 max 6\n");
    printf("option name HceNullDepthDivisor type spin default 0 min 0 max 12\n");
    printf("option name HceLmrBase type spin default 0 min 0 max 4\n");
    printf("option name HceLmrDepthBonusAt type spin default 0 min 0 max 16\n");
    printf("option name HceLmrMoveBonusAt type spin default 0 min 0 max 32\n");
    printf("option name NNModel type string default auto\n");
    printf("option name BookFile type string default auto\n");
    printf("uciok\n");
    fflush(stdout);
}

// Texel feature dump: reads "<fen>;<label>" lines from infile and writes, per
// position, "<label> <phase> <eval_true>" followed by white's then black's
// feature block (5 material counts q/n/b/r/p, isolated, doubled, 4 mobility
// counts n/b/r/q, rook_open, rook_semi, residual_mg, residual_eg). Lets the
// tuner reconstruct and optimize the linear eval weights offline.
static int run_tune_dump(const char *infile, const char *outfile) {
    FILE *fin = fopen(infile, "r");
    if (fin == NULL) {
        printf("info string tunedump: cannot open %s\n", infile);
        return 1;
    }
    FILE *fout = fopen(outfile, "w");
    if (fout == NULL) {
        fclose(fin);
        printf("info string tunedump: cannot open %s\n", outfile);
        return 1;
    }

    MatchConfig cfg = {
        .clock_enabled = false,
        .initial_ms = 0,
        .increment_ms = 0,
        .white_kind = PLAYER_LOCAL_HUMAN,
        .black_kind = PLAYER_LOCAL_HUMAN,
    };
    GameState st;
    chess_init(&st, &cfg);

    char line[512];
    long ok = 0, bad = 0;
    while (fgets(line, sizeof(line), fin) != NULL) {
        char *semi = strchr(line, ';');
        if (semi == NULL) {
            continue;
        }
        *semi = '\0';
        double label = atof(semi + 1);
        char err[160];
        if (!chess_load_fen(&st, line, err, sizeof(err))) {
            bad += 1;
            continue;
        }
        HceTuneFeatures w, b;
        int phase = 0;
        int eval_true = hce_eval_tune_features(&st, &w, &b, &phase);
        fprintf(fout,
                "%.1f %d %d "
                "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d "
                "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                label, phase, eval_true,
                w.mat[PIECE_QUEEN], w.mat[PIECE_KNIGHT], w.mat[PIECE_BISHOP],
                w.mat[PIECE_ROOK], w.mat[PIECE_PAWN], w.isolated, w.doubled,
                w.mob_n, w.mob_b, w.mob_r, w.mob_q, w.rook_open, w.rook_semi,
                w.residual_mg, w.residual_eg,
                b.mat[PIECE_QUEEN], b.mat[PIECE_KNIGHT], b.mat[PIECE_BISHOP],
                b.mat[PIECE_ROOK], b.mat[PIECE_PAWN], b.isolated, b.doubled,
                b.mob_n, b.mob_b, b.mob_r, b.mob_q, b.rook_open, b.rook_semi,
                b.residual_mg, b.residual_eg);
        ok += 1;
    }
    fclose(fin);
    fclose(fout);
    printf("info string tunedump: wrote %ld positions (%ld bad FEN)\n", ok, bad);
    fflush(stdout);
    return 0;
}

int main(void) {
    GameState state;
    UciOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.think_time_ms = 350;
    opt.max_depth = 16;
    opt.backend = CHESS_AI_BACKEND_CLASSIC;
    opt.book_file_path[0] = '\0';
    opt.nn_model_path[0] = '\0';

    reset_start_position(&state);
    chess_ai_warmup();

    char line[2048];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        trim_in_place(line);
        if (line[0] == '\0') {
            continue;
        }

        if (strcmp(line, "uci") == 0) {
            print_uci_intro(&opt);
            continue;
        }
        if (strcmp(line, "isready") == 0) {
            printf("readyok\n");
            fflush(stdout);
            continue;
        }
        if (starts_with(line, "setoption ")) {
            parse_setoption(line, &opt);
            continue;
        }
        if (strcmp(line, "ucinewgame") == 0) {
            reset_start_position(&state);
            continue;
        }
        if (starts_with(line, "position")) {
            handle_position(&state, line);
            continue;
        }
        if (starts_with(line, "go")) {
            handle_go(&state, line, &opt);
            continue;
        }
        if (strcmp(line, "stop") == 0) {
            continue;
        }
        if (strcmp(line, "d") == 0) {
            char fen[256];
            chess_export_fen(&state, fen, sizeof(fen));
            printf("info string fen %s\n", fen);
            fflush(stdout);
            continue;
        }
        if (starts_with(line, "tunedump ")) {
            char inpath[256] = {0};
            char outpath[256] = {0};
            if (sscanf(line + 9, "%255s %255s", inpath, outpath) == 2) {
                run_tune_dump(inpath, outpath);
            } else {
                printf("info string usage: tunedump <infile> <outfile>\n");
                fflush(stdout);
            }
            continue;
        }
        if (strcmp(line, "quit") == 0) {
            break;
        }
    }

    return 0;
}
