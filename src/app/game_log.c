#include "game_log.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static int g_log_counter = 0;
static const char *k_log_dir = "/tmp/chess_logs";

static const char *result_name(GameResult result) {
    switch (result) {
        case GAME_RESULT_ONGOING: return "ONGOING";
        case GAME_RESULT_WHITE_WIN: return "WHITE_WIN";
        case GAME_RESULT_BLACK_WIN: return "BLACK_WIN";
        case GAME_RESULT_WHITE_WIN_RESIGN: return "WHITE_WIN_RESIGN";
        case GAME_RESULT_BLACK_WIN_RESIGN: return "BLACK_WIN_RESIGN";
        case GAME_RESULT_DRAW_STALEMATE: return "DRAW_STALEMATE";
        case GAME_RESULT_DRAW_REPETITION: return "DRAW_REPETITION";
        case GAME_RESULT_DRAW_50: return "DRAW_50";
        case GAME_RESULT_DRAW_75: return "DRAW_75";
        case GAME_RESULT_DRAW_INSUFFICIENT: return "DRAW_INSUFFICIENT";
        case GAME_RESULT_DRAW_AGREED: return "DRAW_AGREED";
        case GAME_RESULT_WIN_TIMEOUT: return "WIN_TIMEOUT";
        default: return "UNKNOWN";
    }
}

static const char *side_name(int side) {
    return (side == PIECE_WHITE) ? "white" : "black";
}

static void build_movetext_uci(const GameLog *log, char *out, size_t out_sz) {
    if (out_sz == 0) {
        return;
    }
    out[0] = '\0';

    size_t used = 0;
    for (int i = 0; i < log->move_count; ++i) {
        if ((i % 2) == 0) {
            used += (size_t)snprintf(out + used, (used < out_sz) ? out_sz - used : 0, "%d. ", i / 2 + 1);
        }
        used += (size_t)snprintf(out + used,
                                 (used < out_sz) ? out_sz - used : 0,
                                 "%s%s",
                                 log->moves[i].uci,
                                 (i + 1 < log->move_count) ? " " : "");
    }
}

static bool ensure_log_directory(void) {
    if (mkdir(k_log_dir, 0755) == 0) {
        return true;
    }
    return errno == EEXIST;
}

void game_log_begin(GameLog *log, const GameState *state) {
    if (log == NULL || state == NULL) {
        return;
    }

    memset(log, 0, sizeof(*log));

    time_t now = time(NULL);
    pid_t pid = getpid();
    int id = g_log_counter++;
    const char *dir = ensure_log_directory() ? k_log_dir : "/tmp";
    snprintf(log->path, sizeof(log->path), "%s/chess_game_%ld_%d_%d.json", dir, (long)now, (int)pid, id);

    chess_export_fen(state, log->start_fen, sizeof(log->start_fen));
    log->result = state->result;
    log->elapsed_white_ms = 0;
    log->elapsed_black_ms = 0;
    log->move_count = 0;
    log->active = true;

    game_log_flush(log);
}

void game_log_record_move(GameLog *log,
                          const GameState *state_after,
                          Move move,
                          int mover_side,
                          int think_ms,
                          bool ai_move,
                          int ai_depth,
                          uint64_t ai_nodes,
                          int ai_score_cp) {
    if (log == NULL || !log->active || state_after == NULL) {
        return;
    }
    if (log->move_count >= CHESS_MAX_GAME_PLY) {
        return;
    }

    GameLogMove *entry = &log->moves[log->move_count];
    memset(entry, 0, sizeof(*entry));

    entry->ply = log->move_count + 1;
    entry->side = mover_side;
    entry->think_ms = think_ms;
    entry->ai_move = ai_move;
    entry->ai_depth = ai_depth;
    entry->ai_nodes = ai_nodes;
    entry->ai_score_cp = ai_score_cp;
    entry->clock_white_ms = state_after->clock_ms[PIECE_WHITE];
    entry->clock_black_ms = state_after->clock_ms[PIECE_BLACK];
    chess_move_to_uci(move, entry->uci);
    chess_export_fen(state_after, entry->fen_after, sizeof(entry->fen_after));

    log->move_count += 1;
    log->result = state_after->result;

    game_log_flush(log);
}

void game_log_truncate(GameLog *log, int move_count) {
    if (log == NULL || !log->active) {
        return;
    }

    if (move_count < 0) {
        move_count = 0;
    }
    if (move_count > log->move_count) {
        move_count = log->move_count;
    }

    log->move_count = move_count;
    game_log_flush(log);
}

void game_log_set_elapsed(GameLog *log, int white_ms, int black_ms) {
    if (log == NULL || !log->active) {
        return;
    }

    log->elapsed_white_ms = white_ms;
    log->elapsed_black_ms = black_ms;
}

void game_log_set_result(GameLog *log, GameResult result) {
    if (log == NULL || !log->active) {
        return;
    }

    log->result = result;
}

bool game_log_flush(const GameLog *log) {
    if (log == NULL || !log->active || log->path[0] == '\0') {
        return false;
    }

    FILE *fp = fopen(log->path, "wb");
    if (fp == NULL) {
        return false;
    }

    char movetext[8192];
    build_movetext_uci(log, movetext, sizeof(movetext));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"format\": \"chess-game-log/v1\",\n");
    fprintf(fp, "  \"path\": \"%s\",\n", log->path);
    fprintf(fp, "  \"start_fen\": \"%s\",\n", log->start_fen);
    fprintf(fp, "  \"result\": \"%s\",\n", result_name(log->result));
    fprintf(fp, "  \"elapsed_ms\": {\"white\": %d, \"black\": %d},\n", log->elapsed_white_ms, log->elapsed_black_ms);
    fprintf(fp, "  \"movetext_uci\": \"%s\",\n", movetext);
    fprintf(fp, "  \"moves\": [\n");

    for (int i = 0; i < log->move_count; ++i) {
        const GameLogMove *m = &log->moves[i];
        fprintf(fp,
                "    {\"ply\": %d, \"side\": \"%s\", \"uci\": \"%s\", \"think_ms\": %d, \"ai_move\": %s, \"ai_depth\": %d, \"ai_nodes\": %llu, \"ai_score_cp\": %d, \"clock_white_ms\": %d, \"clock_black_ms\": %d, \"fen_after\": \"%s\"}%s\n",
                m->ply,
                side_name(m->side),
                m->uci,
                m->think_ms,
                m->ai_move ? "true" : "false",
                m->ai_depth,
                (unsigned long long)m->ai_nodes,
                m->ai_score_cp,
                m->clock_white_ms,
                m->clock_black_ms,
                m->fen_after,
                (i + 1 < log->move_count) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return true;
}

const char *game_log_path(const GameLog *log) {
    if (log == NULL) {
        return "";
    }
    return log->path;
}

const char *game_log_directory(void) {
    return k_log_dir;
}
