#ifndef CHESS_OPENING_BOOK_H
#define CHESS_OPENING_BOOK_H

#include <stdbool.h>
#include <stdint.h>

typedef struct ChessOpeningBookMove {
    char uci[6];
    uint32_t weight;
} ChessOpeningBookMove;

bool chess_opening_book_set_path(const char *path);
bool chess_opening_book_lookup(const char *history_key,
                               const ChessOpeningBookMove **moves_out,
                               int *move_count_out);
bool chess_opening_book_is_loaded(void);
const char *chess_opening_book_loaded_path(void);
void chess_opening_book_reset(void);

#endif
