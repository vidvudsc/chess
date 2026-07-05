#include "nn_eval.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess_types.h"

#define NN_MAGIC_BYTES 8
#define NN_MAGIC "CHNNUE1\0"
#define NN_VERSION_FLOAT 1u
#define NN_VERSION_QUANT 2u
#define NN_MAX_HIDDEN_DIM 128u
#define NN_EXPECTED_HALFKP_DIM (64u * 10u * 64u)
#define NN_CP_FALLBACK 0
#define NN_ACT_QMAX 32767
#define NN_W_QMAX 127

#define NN_HEADER_PREFIX_SIZE (sizeof(uint32_t) * 5u + sizeof(float))

typedef struct NnEvalHeader {
    uint32_t version;
    uint32_t halfkp_dim;
    uint32_t accumulator_dim;
    uint32_t hidden_dim;
    uint32_t dummy_index;
    float cp_scale;
    float acc_scale;
    float fc1_scale;
    float fc2_scale;
    float out_scale;
} NnEvalHeader;

typedef enum NnEvalModelKind {
    NN_MODEL_KIND_NONE = 0,
    NN_MODEL_KIND_QUANT = 1,
} NnEvalModelKind;

typedef struct NnEvalModel {
    bool loaded;
    NnEvalModelKind kind;
    char path[1024];
    NnEvalHeader header;
    int16_t *acc_weight;
    int8_t *fc1_weight;
    float *fc1_bias;
    int8_t *fc2_weight;
    float *fc2_bias;
    int8_t *out_weight;
    float out_bias;
} NnEvalModel;

static NnEvalModel g_nn_model;

static void nn_eval_free_model(NnEvalModel *model) {
    if (model == NULL) {
        return;
    }
    free(model->acc_weight);
    free(model->fc1_weight);
    free(model->fc1_bias);
    free(model->fc2_weight);
    free(model->fc2_bias);
    free(model->out_weight);
    memset(model, 0, sizeof(*model));
}

static bool read_exact(FILE *fp, void *dst, size_t bytes) {
    return fp != NULL && dst != NULL && fread(dst, 1, bytes, fp) == bytes;
}

static bool read_prefix_header(FILE *fp, NnEvalHeader *header) {
    if (fp == NULL || header == NULL) {
        return false;
    }
    memset(header, 0, sizeof(*header));
    if (!read_exact(fp, header, NN_HEADER_PREFIX_SIZE)) {
        return false;
    }
    if (header->version == NN_VERSION_QUANT) {
        if (!read_exact(fp, &header->acc_scale, sizeof(float)) ||
            !read_exact(fp, &header->fc1_scale, sizeof(float)) ||
            !read_exact(fp, &header->fc2_scale, sizeof(float)) ||
            !read_exact(fp, &header->out_scale, sizeof(float))) {
            return false;
        }
    }
    return true;
}

static bool is_white_king_perspective(int perspective) {
    return perspective == PIECE_WHITE;
}

static int mirror_sq(int sq) {
    return sq ^ 56;
}

static int orient_square(int sq, int perspective) {
    return is_white_king_perspective(perspective) ? sq : mirror_sq(sq);
}

static int piece_plane(int piece, int color, int perspective) {
    bool own = color == perspective;
    int base = 0;
    switch (piece) {
        case PIECE_PAWN:
            base = 0;
            break;
        case PIECE_KNIGHT:
            base = 1;
            break;
        case PIECE_BISHOP:
            base = 2;
            break;
        case PIECE_ROOK:
            base = 3;
            break;
        case PIECE_QUEEN:
            base = 4;
            break;
        default:
            return -1;
    }
    return own ? base : (base + 5);
}

static int halfkp_index(int king_sq, int plane, int piece_sq, int perspective) {
    int oriented_king = orient_square(king_sq, perspective);
    int oriented_piece = orient_square(piece_sq, perspective);
    return (oriented_king * 10 + plane) * 64 + oriented_piece;
}

static float quant_scale_from_max(float max_abs, int qmax) {
    if (max_abs <= 1e-12f) {
        return 1.0f;
    }
    return max_abs / (float)qmax;
}

static int16_t quantize_to_i16(float value, float scale) {
    if (scale <= 1e-12f) {
        return 0;
    }
    long q = lroundf(value / scale);
    if (q > 32767L) {
        q = 32767L;
    } else if (q < -32767L) {
        q = -32767L;
    }
    return (int16_t)q;
}

static int8_t quantize_to_i8(float value, float scale) {
    if (scale <= 1e-12f) {
        return 0;
    }
    long q = lroundf(value / scale);
    if (q > 127L) {
        q = 127L;
    } else if (q < -127L) {
        q = -127L;
    }
    return (int8_t)q;
}

static bool allocate_quantized_model(NnEvalModel *model, const NnEvalHeader *header) {
    if (model == NULL || header == NULL) {
        return false;
    }
    size_t acc_rows = (size_t)header->halfkp_dim + 1u;
    size_t acc_count = acc_rows * (size_t)header->accumulator_dim;
    size_t fc1_count = (size_t)header->accumulator_dim * ((size_t)header->accumulator_dim * 2u);
    size_t fc2_count = (size_t)header->hidden_dim * (size_t)header->accumulator_dim;

    model->acc_weight = (int16_t *)malloc(acc_count * sizeof(int16_t));
    model->fc1_weight = (int8_t *)malloc(fc1_count * sizeof(int8_t));
    model->fc1_bias = (float *)malloc((size_t)header->accumulator_dim * sizeof(float));
    model->fc2_weight = (int8_t *)malloc(fc2_count * sizeof(int8_t));
    model->fc2_bias = (float *)malloc((size_t)header->hidden_dim * sizeof(float));
    model->out_weight = (int8_t *)malloc((size_t)header->hidden_dim * sizeof(int8_t));
    if (model->acc_weight == NULL || model->fc1_weight == NULL || model->fc1_bias == NULL ||
        model->fc2_weight == NULL || model->fc2_bias == NULL || model->out_weight == NULL) {
        nn_eval_free_model(model);
        return false;
    }
    return true;
}

static bool quantize_float_model(NnEvalModel *model,
                                 const NnEvalHeader *header,
                                 const float *acc_weight,
                                 const float *fc1_weight,
                                 const float *fc1_bias,
                                 const float *fc2_weight,
                                 const float *fc2_bias,
                                 const float *out_weight,
                                 float out_bias) {
    if (model == NULL || header == NULL || acc_weight == NULL || fc1_weight == NULL || fc1_bias == NULL ||
        fc2_weight == NULL || fc2_bias == NULL || out_weight == NULL) {
        return false;
    }

    NnEvalHeader qh = *header;
    size_t acc_rows = (size_t)header->halfkp_dim + 1u;
    size_t acc_count = acc_rows * (size_t)header->accumulator_dim;
    size_t fc1_count = (size_t)header->accumulator_dim * ((size_t)header->accumulator_dim * 2u);
    size_t fc2_count = (size_t)header->hidden_dim * (size_t)header->accumulator_dim;

    float acc_max = 0.0f;
    for (size_t i = 0; i < acc_count; ++i) {
        float a = fabsf(acc_weight[i]);
        if (a > acc_max) {
            acc_max = a;
        }
    }
    float fc1_max = 0.0f;
    for (size_t i = 0; i < fc1_count; ++i) {
        float a = fabsf(fc1_weight[i]);
        if (a > fc1_max) {
            fc1_max = a;
        }
    }
    float fc2_max = 0.0f;
    for (size_t i = 0; i < fc2_count; ++i) {
        float a = fabsf(fc2_weight[i]);
        if (a > fc2_max) {
            fc2_max = a;
        }
    }
    float out_max = 0.0f;
    for (size_t i = 0; i < (size_t)header->hidden_dim; ++i) {
        float a = fabsf(out_weight[i]);
        if (a > out_max) {
            out_max = a;
        }
    }

    qh.version = NN_VERSION_QUANT;
    qh.acc_scale = quant_scale_from_max(acc_max, 32767);
    qh.fc1_scale = quant_scale_from_max(fc1_max, NN_W_QMAX);
    qh.fc2_scale = quant_scale_from_max(fc2_max, NN_W_QMAX);
    qh.out_scale = quant_scale_from_max(out_max, NN_W_QMAX);

    if (!allocate_quantized_model(model, &qh)) {
        return false;
    }
    model->header = qh;

    for (size_t i = 0; i < acc_count; ++i) {
        model->acc_weight[i] = quantize_to_i16(acc_weight[i], qh.acc_scale);
    }
    for (size_t i = 0; i < fc1_count; ++i) {
        model->fc1_weight[i] = quantize_to_i8(fc1_weight[i], qh.fc1_scale);
    }
    memcpy(model->fc1_bias, fc1_bias, (size_t)header->accumulator_dim * sizeof(float));
    for (size_t i = 0; i < fc2_count; ++i) {
        model->fc2_weight[i] = quantize_to_i8(fc2_weight[i], qh.fc2_scale);
    }
    memcpy(model->fc2_bias, fc2_bias, (size_t)header->hidden_dim * sizeof(float));
    for (size_t i = 0; i < (size_t)header->hidden_dim; ++i) {
        model->out_weight[i] = quantize_to_i8(out_weight[i], qh.out_scale);
    }
    model->out_bias = out_bias;
    return true;
}

static int nn_output_to_cp(float value, float cp_scale) {
    const float eps = 1e-6f;
    float clamped = value;
    if (clamped > 1.0f - eps) {
        clamped = 1.0f - eps;
    } else if (clamped < -1.0f + eps) {
        clamped = -1.0f + eps;
    }
    float cp = 0.5f * logf((1.0f + clamped) / (1.0f - clamped)) * cp_scale;
    if (cp > 32000.0f) {
        cp = 32000.0f;
    } else if (cp < -32000.0f) {
        cp = -32000.0f;
    }
    return (int)lroundf(cp);
}

static void accumulate_perspective(const GameState *state,
                                   const NnEvalModel *model,
                                   int perspective,
                                   int32_t *out_acc,
                                   uint16_t *feature_count_out) {
    const uint32_t acc_dim = model->header.accumulator_dim;
    for (uint32_t i = 0; i < acc_dim; ++i) {
        out_acc[i] = 0;
    }

    int king_sq = chess_find_king_square(state, perspective);
    if (king_sq < 0 || king_sq >= 64) {
        return;
    }

    bool had_feature = false;
    uint16_t feature_count = 0;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        for (int piece = PIECE_QUEEN; piece <= PIECE_PAWN; ++piece) {
            int plane = piece_plane(piece, color, perspective);
            if (plane < 0) {
                continue;
            }
            uint64_t bb = state->bb[color][piece];
            while (bb != 0ULL) {
                int sq = chess_pop_lsb(&bb);
                int idx = halfkp_index(king_sq, plane, sq, perspective);
                if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
                    continue;
                }
                const int16_t *row = model->acc_weight + ((size_t)idx * acc_dim);
                for (uint32_t i = 0; i < acc_dim; ++i) {
                    out_acc[i] += (int32_t)row[i];
                }
                had_feature = true;
                feature_count += 1;
            }
        }
    }

    if (!had_feature) {
        const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * acc_dim);
        for (uint32_t i = 0; i < acc_dim; ++i) {
            out_acc[i] += (int32_t)row[i];
        }
    }
    if (feature_count_out != NULL) {
        *feature_count_out = feature_count;
    }
}

static void add_row(int32_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
    for (uint32_t i = 0; i < acc_dim; ++i) {
        acc[i] += sign * (int32_t)row[i];
    }
}

static bool update_piece_feature(int32_t *acc,
                                 const NnEvalModel *model,
                                 int perspective,
                                 int king_sq,
                                 int color,
                                 int piece,
                                 int sq,
                                 int sign) {
    int plane = piece_plane(piece, color, perspective);
    if (plane < 0) {
        return true;
    }
    int idx = halfkp_index(king_sq, plane, sq, perspective);
    if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
        return false;
    }
    const int16_t *row = model->acc_weight + ((size_t)idx * model->header.accumulator_dim);
    add_row(acc, row, model->header.accumulator_dim, sign);
    return true;
}

static void add_dummy_if_needed(int32_t *acc, const NnEvalModel *model) {
    const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * model->header.accumulator_dim);
    add_row(acc, row, model->header.accumulator_dim, 1);
}

static void remove_dummy_if_needed(int32_t *acc, const NnEvalModel *model) {
    const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * model->header.accumulator_dim);
    add_row(acc, row, model->header.accumulator_dim, -1);
}

static bool rebuild_frame(const GameState *state, NnAccumulatorFrame *frame) {
    if (state == NULL || frame == NULL || !g_nn_model.loaded || g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return false;
    }
    const NnEvalModel *model = &g_nn_model;
    uint16_t white_count = 0;
    accumulate_perspective(state, model, PIECE_WHITE, frame->white_acc, &white_count);
    accumulate_perspective(state, model, PIECE_BLACK, frame->black_acc, NULL);
    frame->valid = true;
    frame->key = state->zobrist_hash;
    frame->non_king_piece_count = white_count;
    return true;
}

static float quantize_activation_relu(const float *src, uint32_t count, int16_t *dst) {
    float max_val = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        float v = src[i] > 0.0f ? src[i] : 0.0f;
        if (v > max_val) {
            max_val = v;
        }
    }
    if (max_val <= 1e-12f) {
        for (uint32_t i = 0; i < count; ++i) {
            dst[i] = 0;
        }
        return 1.0f;
    }
    float scale = max_val / (float)NN_ACT_QMAX;
    for (uint32_t i = 0; i < count; ++i) {
        float v = src[i] > 0.0f ? src[i] : 0.0f;
        dst[i] = quantize_to_i16(v, scale);
    }
    return scale;
}

static int evaluate_from_frame(const GameState *state, const NnEvalModel *model, const NnAccumulatorFrame *frame) {
    if (state == NULL || model == NULL || frame == NULL || !frame->valid) {
        return NN_CP_FALLBACK;
    }
    const uint32_t acc_dim = model->header.accumulator_dim;
    const uint32_t hidden_dim = model->header.hidden_dim;
    const int32_t *front_acc = (state->side_to_move == PIECE_WHITE) ? frame->white_acc : frame->black_acc;
    const int32_t *back_acc = (state->side_to_move == PIECE_WHITE) ? frame->black_acc : frame->white_acc;
    float hidden1_f[NN_MAX_ACC_DIM];
    int16_t hidden1_q[NN_MAX_ACC_DIM];
    float hidden2_f[NN_MAX_HIDDEN_DIM];
    int16_t hidden2_q[NN_MAX_HIDDEN_DIM];

    float fc1_factor = model->header.acc_scale * model->header.fc1_scale;
    for (uint32_t out = 0; out < acc_dim; ++out) {
        int64_t sum = 0;
        const int8_t *w = model->fc1_weight + ((size_t)out * (size_t)acc_dim * 2u);
        for (uint32_t i = 0; i < acc_dim; ++i) {
            sum += (int64_t)w[i] * (int64_t)front_acc[i];
        }
        for (uint32_t i = 0; i < acc_dim; ++i) {
            sum += (int64_t)w[acc_dim + i] * (int64_t)back_acc[i];
        }
        hidden1_f[out] = (float)sum * fc1_factor + model->fc1_bias[out];
    }
    float hidden1_scale = quantize_activation_relu(hidden1_f, acc_dim, hidden1_q);

    float fc2_factor = hidden1_scale * model->header.fc2_scale;
    for (uint32_t out = 0; out < hidden_dim; ++out) {
        int64_t sum = 0;
        const int8_t *w = model->fc2_weight + ((size_t)out * acc_dim);
        for (uint32_t i = 0; i < acc_dim; ++i) {
            sum += (int64_t)w[i] * (int64_t)hidden1_q[i];
        }
        hidden2_f[out] = (float)sum * fc2_factor + model->fc2_bias[out];
    }
    float hidden2_scale = quantize_activation_relu(hidden2_f, hidden_dim, hidden2_q);

    int64_t out_sum = 0;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        out_sum += (int64_t)model->out_weight[i] * (int64_t)hidden2_q[i];
    }
    float out_value = (float)out_sum * (hidden2_scale * model->header.out_scale) + model->out_bias;
    return nn_output_to_cp(tanhf(out_value), model->header.cp_scale);
}

bool nn_eval_load_model(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return false;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return false;
    }

    NnEvalModel model;
    memset(&model, 0, sizeof(model));
    bool ok = false;

    char magic8[NN_MAGIC_BYTES];
    if (!read_exact(fp, magic8, sizeof(magic8)) || memcmp(magic8, NN_MAGIC, sizeof(magic8)) != 0) {
        fclose(fp);
        return false;
    }

    NnEvalHeader header;
    if (!read_prefix_header(fp, &header)) {
        fclose(fp);
        return false;
    }
    if ((header.version != NN_VERSION_FLOAT && header.version != NN_VERSION_QUANT) ||
        header.halfkp_dim != NN_EXPECTED_HALFKP_DIM ||
        header.accumulator_dim == 0 || header.accumulator_dim > NN_MAX_ACC_DIM ||
        header.hidden_dim == 0 || header.hidden_dim > NN_MAX_HIDDEN_DIM ||
        header.dummy_index != header.halfkp_dim) {
        fclose(fp);
        return false;
    }

    size_t acc_rows = (size_t)header.halfkp_dim + 1u;
    size_t acc_count = acc_rows * (size_t)header.accumulator_dim;
    size_t fc1_count = (size_t)header.accumulator_dim * ((size_t)header.accumulator_dim * 2u);
    size_t fc2_count = (size_t)header.hidden_dim * (size_t)header.accumulator_dim;

    if (header.version == NN_VERSION_QUANT) {
        if (!allocate_quantized_model(&model, &header)) {
            fclose(fp);
            nn_eval_free_model(&model);
            return false;
        }
        model.header = header;
        model.kind = NN_MODEL_KIND_QUANT;
        ok = read_exact(fp, model.acc_weight, acc_count * sizeof(int16_t)) &&
             read_exact(fp, model.fc1_weight, fc1_count * sizeof(int8_t)) &&
             read_exact(fp, model.fc1_bias, (size_t)header.accumulator_dim * sizeof(float)) &&
             read_exact(fp, model.fc2_weight, fc2_count * sizeof(int8_t)) &&
             read_exact(fp, model.fc2_bias, (size_t)header.hidden_dim * sizeof(float)) &&
             read_exact(fp, model.out_weight, (size_t)header.hidden_dim * sizeof(int8_t)) &&
             read_exact(fp, &model.out_bias, sizeof(float));
    } else {
        float *acc_weight = (float *)malloc(acc_count * sizeof(float));
        float *fc1_weight = (float *)malloc(fc1_count * sizeof(float));
        float *fc1_bias = (float *)malloc((size_t)header.accumulator_dim * sizeof(float));
        float *fc2_weight = (float *)malloc(fc2_count * sizeof(float));
        float *fc2_bias = (float *)malloc((size_t)header.hidden_dim * sizeof(float));
        float *out_weight = (float *)malloc((size_t)header.hidden_dim * sizeof(float));
        float out_bias = 0.0f;
        if (acc_weight == NULL || fc1_weight == NULL || fc1_bias == NULL ||
            fc2_weight == NULL || fc2_bias == NULL || out_weight == NULL) {
            free(acc_weight);
            free(fc1_weight);
            free(fc1_bias);
            free(fc2_weight);
            free(fc2_bias);
            free(out_weight);
            fclose(fp);
            return false;
        }
        ok = read_exact(fp, acc_weight, acc_count * sizeof(float)) &&
             read_exact(fp, fc1_weight, fc1_count * sizeof(float)) &&
             read_exact(fp, fc1_bias, (size_t)header.accumulator_dim * sizeof(float)) &&
             read_exact(fp, fc2_weight, fc2_count * sizeof(float)) &&
             read_exact(fp, fc2_bias, (size_t)header.hidden_dim * sizeof(float)) &&
             read_exact(fp, out_weight, (size_t)header.hidden_dim * sizeof(float)) &&
             read_exact(fp, &out_bias, sizeof(float));
        if (ok) {
            ok = quantize_float_model(&model,
                                      &header,
                                      acc_weight,
                                      fc1_weight,
                                      fc1_bias,
                                      fc2_weight,
                                      fc2_bias,
                                      out_weight,
                                      out_bias);
            if (ok) {
                model.kind = NN_MODEL_KIND_QUANT;
            }
        }
        free(acc_weight);
        free(fc1_weight);
        free(fc1_bias);
        free(fc2_weight);
        free(fc2_bias);
        free(out_weight);
    }
    fclose(fp);
    if (!ok) {
        nn_eval_free_model(&model);
        return false;
    }

    model.loaded = true;
    snprintf(model.path, sizeof(model.path), "%s", path);

    nn_eval_free_model(&g_nn_model);
    g_nn_model = model;
    return true;
}

bool nn_eval_is_loaded(void) {
    return g_nn_model.loaded;
}

const char *nn_eval_model_path(void) {
    return g_nn_model.loaded ? g_nn_model.path : "";
}

int nn_eval_cp_stm(const GameState *state) {
    if (state == NULL || !g_nn_model.loaded) {
        return NN_CP_FALLBACK;
    }
    NnAccumulatorFrame frame;
    if (!rebuild_frame(state, &frame)) {
        return NN_CP_FALLBACK;
    }
    return evaluate_from_frame(state, &g_nn_model, &frame);
}

bool nn_eval_build_frame(const GameState *state, NnAccumulatorFrame *frame) {
    if (!g_nn_model.loaded || g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return false;
    }
    return rebuild_frame(state, frame);
}

bool nn_eval_update_frame(const GameState *state,
                          const UndoRecord *undo,
                          const NnAccumulatorFrame *parent,
                          NnAccumulatorFrame *frame) {
    if (state == NULL || undo == NULL || parent == NULL || frame == NULL || !g_nn_model.loaded || !parent->valid) {
        return false;
    }
    if (g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return false;
    }

    int mover = state->side_to_move ^ 1;
    int from = move_from(undo->move);
    int to = move_to(undo->move);
    int piece = move_piece(undo->move);
    int placed_piece = move_has_flag(undo->move, MOVE_FLAG_PROMOTION) ? move_promo(undo->move) : piece;

    *frame = *parent;
    const NnEvalModel *model = &g_nn_model;
    int white_king_sq = chess_find_king_square(state, PIECE_WHITE);
    int black_king_sq = chess_find_king_square(state, PIECE_BLACK);
    if (white_king_sq < 0 || black_king_sq < 0) {
        return rebuild_frame(state, frame);
    }

    if (piece == PIECE_KING) {
        if (move_has_flag(undo->move, MOVE_FLAG_CASTLE)) {
            return rebuild_frame(state, frame);
        }
        if (mover == PIECE_WHITE) {
            accumulate_perspective(state, model, PIECE_WHITE, frame->white_acc, NULL);
        } else {
            accumulate_perspective(state, model, PIECE_BLACK, frame->black_acc, NULL);
        }
        frame->valid = true;
        frame->key = state->zobrist_hash;
        return true;
    }

    if (parent->non_king_piece_count == 0 && undo->captured_piece == PIECE_NONE) {
        frame->key = state->zobrist_hash;
        return true;
    }

    if (parent->non_king_piece_count == 0) {
        remove_dummy_if_needed(frame->white_acc, model);
        remove_dummy_if_needed(frame->black_acc, model);
    }

    if (!update_piece_feature(frame->white_acc, model, PIECE_WHITE, white_king_sq, mover, piece, from, -1) ||
        !update_piece_feature(frame->black_acc, model, PIECE_BLACK, black_king_sq, mover, piece, from, -1) ||
        !update_piece_feature(frame->white_acc, model, PIECE_WHITE, white_king_sq, mover, placed_piece, to, 1) ||
        !update_piece_feature(frame->black_acc, model, PIECE_BLACK, black_king_sq, mover, placed_piece, to, 1)) {
        return rebuild_frame(state, frame);
    }

    if (undo->captured_piece != PIECE_NONE) {
        int victim = state->side_to_move;
        if (!update_piece_feature(frame->white_acc, model, PIECE_WHITE, white_king_sq, victim, undo->captured_piece, undo->captured_square, -1) ||
            !update_piece_feature(frame->black_acc, model, PIECE_BLACK, black_king_sq, victim, undo->captured_piece, undo->captured_square, -1)) {
            return rebuild_frame(state, frame);
        }
        if (undo->captured_piece != PIECE_KING && frame->non_king_piece_count > 0) {
            frame->non_king_piece_count -= 1;
        }
    }

    if (frame->non_king_piece_count == 0) {
        add_dummy_if_needed(frame->white_acc, model);
        add_dummy_if_needed(frame->black_acc, model);
    }
    frame->valid = true;
    frame->key = state->zobrist_hash;
    return true;
}

int nn_eval_cp_stm_from_frame(const GameState *state, const NnAccumulatorFrame *frame) {
    if (!g_nn_model.loaded) {
        return NN_CP_FALLBACK;
    }
    if (g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return NN_CP_FALLBACK;
    }
    return evaluate_from_frame(state, &g_nn_model, frame);
}
