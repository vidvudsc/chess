#ifndef HCE_TB_H
#define HCE_TB_H

#include <stdbool.h>

#include "chess_state.h"

// Syzygy tablebase access (Fathom). Scores for tablebase wins sit below mate
// scores and above anything the static eval produces.
#define HCE_TB_WIN 20000

// Loads tables from a path list (':' separated, ';' on Windows). An empty
// string or "<empty>" unloads. Returns the largest piece count available.
int hce_tb_init(const char *path);
int hce_tb_largest(void);

// WDL probe for search. Only succeeds right after a capture or pawn move
// (halfmove clock zero) with no castling rights. *wdl is -2..2 from the side
// to move's view: -2 loss, -1 blessed loss, 0 draw, 1 cursed win, 2 win.
bool hce_tb_probe_wdl(const GameState *s, int *wdl);

// Root DTZ probe: returns the tablebase-optimal move and a search-style score.
// Not thread safe; call once per search from the searching thread.
bool hce_tb_probe_root(const GameState *s, Move *move_out, int *score_out);

#endif
