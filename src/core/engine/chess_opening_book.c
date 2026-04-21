#include "chess_opening_book.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct BookEntry {
    char *history_key;
    int first_move;
    int move_count;
} BookEntry;

typedef struct RawRow {
    char *history_key;
    char uci[6];
    uint32_t weight;
} RawRow;

static BookEntry *g_entries = NULL;
static int g_entry_count = 0;
static ChessOpeningBookMove *g_moves = NULL;
static int g_move_count = 0;
static bool g_load_attempted = false;
static bool g_loaded = false;
static char g_loaded_path[1024];
static char g_forced_path[1024];

static char *dup_cstr(const char *s) {
    if (s == NULL) {
        return NULL;
    }
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n + 1);
    return out;
}

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

static bool valid_uci_move(const char *s) {
    if (s == NULL) {
        return false;
    }
    size_t n = strlen(s);
    if (n != 4 && n != 5) {
        return false;
    }
    if (s[0] < 'a' || s[0] > 'h' || s[2] < 'a' || s[2] > 'h') {
        return false;
    }
    if (s[1] < '1' || s[1] > '8' || s[3] < '1' || s[3] > '8') {
        return false;
    }
    if (n == 5) {
        char p = (char)tolower((unsigned char)s[4]);
        if (p != 'q' && p != 'r' && p != 'b' && p != 'n') {
            return false;
        }
    }
    return true;
}

static int raw_row_cmp(const void *a, const void *b) {
    const RawRow *ra = (const RawRow *)a;
    const RawRow *rb = (const RawRow *)b;
    int c = strcmp(ra->history_key, rb->history_key);
    if (c != 0) {
        return c;
    }
    return strcmp(ra->uci, rb->uci);
}

static void clear_loaded_data(void) {
    if (g_entries != NULL) {
        for (int i = 0; i < g_entry_count; ++i) {
            free(g_entries[i].history_key);
        }
        free(g_entries);
    }
    free(g_moves);

    g_entries = NULL;
    g_entry_count = 0;
    g_moves = NULL;
    g_move_count = 0;
    g_loaded = false;
    g_loaded_path[0] = '\0';
}

void chess_opening_book_reset(void) {
    clear_loaded_data();
    g_load_attempted = false;
}

static bool parse_uint32(const char *s, uint32_t *out) {
    if (s == NULL || out == NULL || *s == '\0') {
        return false;
    }
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    if (v == 0 || v > UINT_MAX) {
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

static bool append_raw_row(RawRow **rows, int *count, int *cap, const RawRow *row) {
    if (*count >= *cap) {
        int next_cap = (*cap <= 0) ? 4096 : (*cap * 2);
        RawRow *next = (RawRow *)realloc(*rows, (size_t)next_cap * sizeof(**rows));
        if (next == NULL) {
            return false;
        }
        *rows = next;
        *cap = next_cap;
    }
    (*rows)[*count] = *row;
    (*count)++;
    return true;
}

static bool append_book_edge(RawRow **rows,
                             int *count,
                             int *cap,
                             const char *history,
                             const char *move,
                             uint32_t weight) {
    RawRow row;
    row.history_key = dup_cstr((history != NULL) ? history : "");
    if (row.history_key == NULL) {
        return false;
    }
    snprintf(row.uci, sizeof(row.uci), "%s", move);
    for (int i = 0; row.uci[i] != '\0'; ++i) {
        row.uci[i] = (char)tolower((unsigned char)row.uci[i]);
    }
    row.weight = weight;
    if (!append_raw_row(rows, count, cap, &row)) {
        free(row.history_key);
        return false;
    }
    return true;
}

static bool append_curated_game_rows(RawRow **rows,
                                     int *count,
                                     int *cap,
                                     const char *sequence,
                                     uint32_t weight) {
    if (sequence == NULL || sequence[0] == '\0') {
        return false;
    }

    char seq_buf[4096];
    char history_buf[4096];
    snprintf(seq_buf, sizeof(seq_buf), "%s", sequence);
    history_buf[0] = '\0';

    bool saw_move = false;
    char *cursor = seq_buf;
    while (*cursor != '\0') {
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        char *tok = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (*cursor != '\0') {
            *cursor = '\0';
            ++cursor;
        }

        if (!valid_uci_move(tok)) {
            return false;
        }
        if (!append_book_edge(rows, count, cap, history_buf, tok, weight)) {
            return false;
        }
        size_t history_len = strlen(history_buf);
        size_t tok_len = strlen(tok);
        size_t needed = tok_len + ((history_len > 0) ? 1U : 0U);
        if (history_len + needed + 1U > sizeof(history_buf)) {
            return false;
        }
        if (history_len > 0) {
            history_buf[history_len++] = ' ';
        }
        memcpy(history_buf + history_len, tok, tok_len + 1U);
        saw_move = true;
    }

    return saw_move;
}

static bool load_from_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return false;
    }

    RawRow *rows = NULL;
    int row_count = 0;
    int row_cap = 0;
    bool ok = true;
    char line[4096];

    while (fgets(line, sizeof(line), fp) != NULL) {
        trim_in_place(line);
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }

        char *sep1 = strchr(line, '|');
        if (sep1 == NULL) {
            continue;
        }
        *sep1 = '\0';
        char *history = line;
        char *move = sep1 + 1;
        char *weight_str = NULL;

        char *sep2 = strchr(move, '|');
        if (sep2 != NULL) {
            *sep2 = '\0';
            weight_str = sep2 + 1;
        }

        trim_in_place(history);
        trim_in_place(move);
        if (weight_str != NULL) {
            trim_in_place(weight_str);
        }

        uint32_t weight = 1;
        if (weight_str != NULL && weight_str[0] != '\0') {
            if (!parse_uint32(weight_str, &weight)) {
                continue;
            }
        }

        if (strcmp(history, "-") == 0) {
            history[0] = '\0';
        }

        if (valid_uci_move(move)) {
            if (!append_book_edge(&rows, &row_count, &row_cap, history, move, weight)) {
                ok = false;
                break;
            }
            continue;
        }

        if (!append_curated_game_rows(&rows, &row_count, &row_cap, move, weight)) {
            continue;
        }
    }
    fclose(fp);

    if (!ok || row_count <= 0) {
        if (rows != NULL) {
            for (int i = 0; i < row_count; ++i) {
                free(rows[i].history_key);
            }
            free(rows);
        }
        return false;
    }

    qsort(rows, (size_t)row_count, sizeof(rows[0]), raw_row_cmp);

    RawRow *dedup = (RawRow *)calloc((size_t)row_count, sizeof(*dedup));
    if (dedup == NULL) {
        for (int i = 0; i < row_count; ++i) {
            free(rows[i].history_key);
        }
        free(rows);
        return false;
    }

    int dedup_count = 0;
    for (int i = 0; i < row_count; ++i) {
        if (dedup_count > 0 &&
            strcmp(dedup[dedup_count - 1].history_key, rows[i].history_key) == 0 &&
            strcmp(dedup[dedup_count - 1].uci, rows[i].uci) == 0) {
            uint64_t sum = (uint64_t)dedup[dedup_count - 1].weight + (uint64_t)rows[i].weight;
            dedup[dedup_count - 1].weight = (sum > UINT_MAX) ? UINT_MAX : (uint32_t)sum;
            continue;
        }

        dedup[dedup_count].history_key = dup_cstr(rows[i].history_key);
        if (dedup[dedup_count].history_key == NULL) {
            for (int j = 0; j < row_count; ++j) {
                free(rows[j].history_key);
            }
            free(rows);
            for (int j = 0; j < dedup_count; ++j) {
                free(dedup[j].history_key);
            }
            free(dedup);
            return false;
        }
        snprintf(dedup[dedup_count].uci, sizeof(dedup[dedup_count].uci), "%s", rows[i].uci);
        dedup[dedup_count].weight = rows[i].weight;
        dedup_count++;
    }

    for (int i = 0; i < row_count; ++i) {
        free(rows[i].history_key);
    }
    free(rows);
    rows = dedup;
    row_count = dedup_count;

    int entry_count = 0;
    for (int i = 0; i < row_count; ++i) {
        if (i == 0 || strcmp(rows[i].history_key, rows[i - 1].history_key) != 0) {
            entry_count++;
        }
    }

    BookEntry *entries = (BookEntry *)calloc((size_t)entry_count, sizeof(*entries));
    ChessOpeningBookMove *moves = (ChessOpeningBookMove *)calloc((size_t)row_count, sizeof(*moves));
    if (entries == NULL || moves == NULL) {
        free(entries);
        free(moves);
        for (int i = 0; i < row_count; ++i) {
            free(rows[i].history_key);
        }
        free(rows);
        return false;
    }

    int e = -1;
    int m = 0;
    for (int i = 0; i < row_count; ++i) {
        if (i == 0 || strcmp(rows[i].history_key, rows[i - 1].history_key) != 0) {
            ++e;
            entries[e].history_key = dup_cstr(rows[i].history_key);
            if (entries[e].history_key == NULL) {
                for (int j = 0; j <= e; ++j) {
                    free(entries[j].history_key);
                }
                free(entries);
                free(moves);
                for (int j = 0; j < row_count; ++j) {
                    free(rows[j].history_key);
                }
                free(rows);
                return false;
            }
            entries[e].first_move = m;
            entries[e].move_count = 0;
        }
        snprintf(moves[m].uci, sizeof(moves[m].uci), "%s", rows[i].uci);
        moves[m].weight = rows[i].weight;
        entries[e].move_count++;
        ++m;
    }
    for (int i = 0; i < row_count; ++i) {
        free(rows[i].history_key);
    }
    free(rows);

    clear_loaded_data();
    g_entries = entries;
    g_entry_count = entry_count;
    g_moves = moves;
    g_move_count = row_count;
    g_loaded = true;
    snprintf(g_loaded_path, sizeof(g_loaded_path), "%s", path);
    return true;
}

static bool try_load_from_candidates(void) {
    if (g_forced_path[0] != '\0') {
        return load_from_file(g_forced_path);
    }

    const char *env_candidates[] = {
        getenv("CHESS_OPENING_BOOK"),
        getenv("OPENING_BOOK_PATH"),
    };
    for (size_t i = 0; i < sizeof(env_candidates) / sizeof(env_candidates[0]); ++i) {
        if (env_candidates[i] != NULL && env_candidates[i][0] != '\0') {
            if (load_from_file(env_candidates[i])) {
                return true;
            }
        }
    }

    const char *paths[] = {
        "data/openings/opening_book.txt",
        "data/openings/opening_games_100.txt",
        "data/openings/book.txt",
        "src/core/bot/opening_book.txt",
        "src/core/bot/opening_games_100.txt",
        "opening_book.txt",
        "engine/opening_book.txt",
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) {
        if (load_from_file(paths[i])) {
            return true;
        }
    }
    return false;
}

static void ensure_loaded_once(void) {
    if (g_load_attempted) {
        return;
    }
    g_load_attempted = true;
    (void)try_load_from_candidates();
}

bool chess_opening_book_set_path(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return false;
    }
    snprintf(g_forced_path, sizeof(g_forced_path), "%s", path);
    chess_opening_book_reset();
    g_load_attempted = true;
    if (load_from_file(g_forced_path)) {
        return true;
    }
    g_load_attempted = false;
    return false;
}

bool chess_opening_book_lookup(const char *history_key,
                               const ChessOpeningBookMove **moves_out,
                               int *move_count_out) {
    if (moves_out == NULL || move_count_out == NULL) {
        return false;
    }

    *moves_out = NULL;
    *move_count_out = 0;
    ensure_loaded_once();
    if (!g_loaded || g_entry_count <= 0 || g_move_count <= 0) {
        return false;
    }

    const char *key = (history_key != NULL) ? history_key : "";

    int lo = 0;
    int hi = g_entry_count - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) / 2);
        int c = strcmp(key, g_entries[mid].history_key);
        if (c == 0) {
            *moves_out = &g_moves[g_entries[mid].first_move];
            *move_count_out = g_entries[mid].move_count;
            return true;
        }
        if (c < 0) {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    return false;
}

bool chess_opening_book_is_loaded(void) {
    ensure_loaded_once();
    return g_loaded;
}

const char *chess_opening_book_loaded_path(void) {
    ensure_loaded_once();
    return g_loaded ? g_loaded_path : NULL;
}
