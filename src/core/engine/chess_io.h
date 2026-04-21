#ifndef CHESS_IO_H
#define CHESS_IO_H

#include "chess_rules.h"

bool chess_load_fen(GameState *s, const char *fen, char *err, size_t err_sz);
void chess_export_fen(const GameState *s, char *out, size_t out_sz);
bool chess_move_to_uci(Move m, char out[6]);
bool chess_move_from_uci(const GameState *s, const char *uci, Move *out);

#endif
