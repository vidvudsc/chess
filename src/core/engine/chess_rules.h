#ifndef CHESS_RULES_H
#define CHESS_RULES_H

#include "chess_state.h"

void chess_init(GameState *s, const MatchConfig *cfg);
int chess_generate_legal_moves(const GameState *s, Move out[CHESS_MAX_MOVES]);
int chess_generate_legal_moves_mut(GameState *s, Move out[CHESS_MAX_MOVES]);
bool chess_make_move(GameState *s, Move m);
bool chess_make_move_trusted(GameState *s, Move legal_move);
bool chess_undo_move(GameState *s);
bool chess_in_check(const GameState *s, int side);
GameResult chess_update_result(GameState *s);
uint64_t chess_perft(GameState *s, int depth);
bool chess_is_move_legal(const GameState *s, Move m);
bool chess_has_mating_material(const GameState *s, int side);
void chess_set_result(GameState *s, GameResult result);
void chess_tick_clock(GameState *s, int delta_ms);

#endif
