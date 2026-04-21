#ifndef NN_EVAL_H
#define NN_EVAL_H

#include <stdbool.h>
#include <stdint.h>

#include "chess_state.h"

#define NN_MAX_ACC_DIM 256u

typedef struct NnAccumulatorFrame {
    bool valid;
    uint64_t key;
    uint16_t non_king_piece_count;
    int32_t white_acc[NN_MAX_ACC_DIM];
    int32_t black_acc[NN_MAX_ACC_DIM];
} NnAccumulatorFrame;

bool nn_eval_load_model(const char *path);
bool nn_eval_is_loaded(void);
const char *nn_eval_model_path(void);
int nn_eval_cp_stm(const GameState *state);
bool nn_eval_build_frame(const GameState *state, NnAccumulatorFrame *frame);
bool nn_eval_update_frame(const GameState *state,
                          const UndoRecord *undo,
                          const NnAccumulatorFrame *parent,
                          NnAccumulatorFrame *frame);
int nn_eval_cp_stm_from_frame(const GameState *state, const NnAccumulatorFrame *frame);

#endif
