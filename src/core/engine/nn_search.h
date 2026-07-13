#ifndef NN_SEARCH_H
#define NN_SEARCH_H

#include "chess_ai.h"

bool nn_search_pick_move(const GameState *state,
                         const AiSearchConfig *cfg,
                         AiSearchResult *out);
int nn_search_probe_deep_eval_cp_stm(const GameState *state);

bool nn_search_leaf_log_set_path(const char *path);
const char *nn_search_leaf_log_path(void);
void nn_search_leaf_log_set_limit(int limit);
int nn_search_leaf_log_limit(void);
int nn_search_leaf_log_count(void);

bool nn_search_set_option(const char *name, int value);
int nn_search_get_option(const char *name);
void nn_search_reset_options(void);

#endif
