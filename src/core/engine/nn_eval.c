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
#define NN_LEGACY_MAGIC_BYTES 4
#define NN_LEGACY_MAGIC "CHNN"
#define NN_LEGACY_RUN1_VERSION 3u
#define NN_LEGACY_RUN1_FLAG_USE_PHASE (1u << 0)
#define NN_LEGACY_RUN1_FLAG_PERSPECTIVE_NORM (1u << 1)
#define NN_LEGACY_MAX_HIDDEN 1024u
#define NN_LEGACY_MAX_HEAD1 1024u
#define NN_LEGACY_MAX_HEAD2 1024u
#define NN_LEGACY_MAX_PHASE_DIM 32u
#define NN_LEGACY_MAX_ACTIVE 30
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

typedef struct NnLegacyRun1Header {
    char magic[NN_LEGACY_MAGIC_BYTES];
    uint32_t version;
    uint32_t num_features;
    uint32_t hidden;
    uint32_t head1;
    uint32_t head2;
    uint32_t flags;
    uint32_t king_buckets;
    uint32_t phase_dim;
    uint32_t reserved;
    float tanh_scale;
    float cp_clip;
} NnLegacyRun1Header;

typedef enum NnEvalModelKind {
    NN_MODEL_KIND_NONE = 0,
    NN_MODEL_KIND_QUANT = 1,
    NN_MODEL_KIND_LEGACY_RUN1 = 2,
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

    float *legacy_blob;
    uint32_t legacy_num_features;
    uint32_t legacy_hidden;
    uint32_t legacy_head1;
    uint32_t legacy_head2;
    uint32_t legacy_flags;
    uint32_t legacy_king_buckets;
    uint32_t legacy_phase_dim;
    float legacy_tanh_scale;
    float legacy_cp_clip;
    float *legacy_emb;
    float *legacy_phase_emb;
    float *legacy_fc1_w;
    float *legacy_fc1_b;
    float *legacy_fc2_w;
    float *legacy_fc2_b;
    float *legacy_fc3_w;
    float legacy_fc3_b;
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
    free(model->legacy_blob);
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

static bool safe_mul_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > ((size_t)-1) / b) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool safe_add_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) {
        return false;
    }
    if (a > ((size_t)-1) - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static int clamp_cp_int(int cp) {
    if (cp > 32000) {
        return 32000;
    }
    if (cp < -32000) {
        return -32000;
    }
    return cp;
}

static int legacy_transform_sq_rotate180(int sq) {
    if (sq < 0 || sq >= 64) {
        return sq;
    }
    return 63 - sq;
}

static int legacy_run1_piece_slot(int is_us_piece, int piece) {
    int base = is_us_piece ? 0 : 5;
    switch (piece) {
        case PIECE_PAWN:
            return base + 0;
        case PIECE_KNIGHT:
            return base + 1;
        case PIECE_BISHOP:
            return base + 2;
        case PIECE_ROOK:
            return base + 3;
        case PIECE_QUEEN:
            return base + 4;
        default:
            return -1;
    }
}

static int legacy_run1_king_bucket(int king_sq, uint32_t buckets) {
    if (king_sq < 0) {
        return 0;
    }
    if (buckets == 0u) {
        return king_sq;
    }

    int f = king_sq % 8;
    int r = king_sq / 8;
    if (buckets == 32u) {
        return (r / 2) * 8 + f;
    }
    if (buckets == 16u) {
        return (r / 2) * 4 + (f / 2);
    }

    uint32_t bucket = (uint32_t)(((uint32_t)king_sq * buckets) / 64u);
    if (bucket >= buckets) {
        bucket = buckets - 1u;
    }
    return (int)bucket;
}

static int legacy_estimate_phase_id_from_state(const GameState *state) {
    if (state == NULL) {
        return 3;
    }

    int ply_est = state->ply;
    if (state->fullmove_number > 0) {
        int from_fullmove = (state->fullmove_number - 1) * 2 + (state->side_to_move == PIECE_BLACK ? 1 : 0);
        if (from_fullmove > ply_est) {
            ply_est = from_fullmove;
        }
    }

    if (ply_est <= 16) {
        return 0;
    }

    int total_npm = 0;
    total_npm += 320 * chess_count_bits(state->bb[PIECE_WHITE][PIECE_KNIGHT]);
    total_npm += 320 * chess_count_bits(state->bb[PIECE_BLACK][PIECE_KNIGHT]);
    total_npm += 330 * chess_count_bits(state->bb[PIECE_WHITE][PIECE_BISHOP]);
    total_npm += 330 * chess_count_bits(state->bb[PIECE_BLACK][PIECE_BISHOP]);
    total_npm += 500 * chess_count_bits(state->bb[PIECE_WHITE][PIECE_ROOK]);
    total_npm += 500 * chess_count_bits(state->bb[PIECE_BLACK][PIECE_ROOK]);
    int queens = chess_count_bits(state->bb[PIECE_WHITE][PIECE_QUEEN]) +
                 chess_count_bits(state->bb[PIECE_BLACK][PIECE_QUEEN]);
    total_npm += 900 * queens;

    if (total_npm <= 1800 || (queens == 0 && total_npm <= 2600)) {
        return 2;
    }
    return 1;
}

static int legacy_encode_run1_indices(const GameState *state,
                                      bool perspective_norm,
                                      uint32_t king_buckets,
                                      int out_idx_us[NN_LEGACY_MAX_ACTIVE],
                                      int out_idx_th[NN_LEGACY_MAX_ACTIVE]) {
    if (state == NULL || out_idx_us == NULL || out_idx_th == NULL) {
        return 0;
    }

    int w_king_sq = chess_find_king_square(state, PIECE_WHITE);
    int b_king_sq = chess_find_king_square(state, PIECE_BLACK);
    if (w_king_sq < 0 || b_king_sq < 0) {
        return 0;
    }

    bool stm_white = (state->side_to_move == PIECE_WHITE);
    bool rotate = perspective_norm && !stm_white;

    int us_king_sq = stm_white ? w_king_sq : b_king_sq;
    int th_king_sq = stm_white ? b_king_sq : w_king_sq;
    if (perspective_norm && rotate) {
        us_king_sq = legacy_transform_sq_rotate180(us_king_sq);
        th_king_sq = legacy_transform_sq_rotate180(th_king_sq);
    } else if (!perspective_norm) {
        us_king_sq = w_king_sq;
        th_king_sq = b_king_sq;
    }

    int ku = legacy_run1_king_bucket(us_king_sq, king_buckets);
    int kt = legacy_run1_king_bucket(th_king_sq, king_buckets);

    int count = 0;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        for (int piece = PIECE_QUEEN; piece <= PIECE_PAWN; ++piece) {
            uint64_t bb = state->bb[color][piece];
            while (bb != 0ULL) {
                int sq = chess_pop_lsb(&bb);
                int sq_norm = rotate ? legacy_transform_sq_rotate180(sq) : sq;
                int is_us_piece = perspective_norm
                                      ? ((color == state->side_to_move) ? 1 : 0)
                                      : ((color == PIECE_WHITE) ? 1 : 0);
                int slot = legacy_run1_piece_slot(is_us_piece, piece);
                if (slot < 0) {
                    continue;
                }
                if (count < NN_LEGACY_MAX_ACTIVE) {
                    out_idx_us[count] = ((ku * 10 + slot) * 64) + sq_norm;
                    out_idx_th[count] = ((kt * 10 + slot) * 64) + sq_norm;
                    count += 1;
                }
            }
        }
    }
    return count;
}

static bool load_legacy_run1_model(FILE *fp,
                                   const char *path,
                                   const NnLegacyRun1Header *header,
                                   NnEvalModel *model) {
    if (fp == NULL || path == NULL || header == NULL || model == NULL) {
        return false;
    }

    uint32_t buckets = (header->king_buckets == 0u) ? 64u : header->king_buckets;
    if (header->hidden == 0u || header->head1 == 0u || header->head2 == 0u ||
        header->hidden > NN_LEGACY_MAX_HIDDEN ||
        header->head1 > NN_LEGACY_MAX_HEAD1 ||
        header->head2 > NN_LEGACY_MAX_HEAD2 ||
        header->phase_dim > NN_LEGACY_MAX_PHASE_DIM ||
        buckets > 64u ||
        !(header->tanh_scale > 0.0f) ||
        !(header->cp_clip > 0.0f)) {
        return false;
    }
    if ((header->flags & NN_LEGACY_RUN1_FLAG_USE_PHASE) == 0u && header->phase_dim != 0u) {
        return false;
    }
    if ((header->flags & NN_LEGACY_RUN1_FLAG_USE_PHASE) != 0u && header->phase_dim == 0u) {
        return false;
    }
    if (header->num_features != buckets * 10u * 64u) {
        return false;
    }

    size_t emb_count = 0;
    size_t phase_count = 0;
    size_t fc1_in = 0;
    size_t fc1_w_count = 0;
    size_t fc1_b_count = (size_t)header->head1;
    size_t fc2_w_count = 0;
    size_t fc2_b_count = (size_t)header->head2;
    size_t fc3_w_count = (size_t)header->head2;
    if (!safe_mul_size((size_t)header->num_features, (size_t)header->hidden, &emb_count) ||
        !safe_mul_size(4u, (size_t)header->phase_dim, &phase_count) ||
        !safe_mul_size(2u, (size_t)header->hidden, &fc1_in) ||
        !safe_add_size(fc1_in, (size_t)header->phase_dim, &fc1_in) ||
        !safe_mul_size((size_t)header->head1, fc1_in, &fc1_w_count) ||
        !safe_mul_size((size_t)header->head2, (size_t)header->head1, &fc2_w_count)) {
        return false;
    }

    size_t total_floats = 0;
    if (!safe_add_size(total_floats, emb_count, &total_floats) ||
        !safe_add_size(total_floats, phase_count, &total_floats) ||
        !safe_add_size(total_floats, fc1_w_count, &total_floats) ||
        !safe_add_size(total_floats, fc1_b_count, &total_floats) ||
        !safe_add_size(total_floats, fc2_w_count, &total_floats) ||
        !safe_add_size(total_floats, fc2_b_count, &total_floats) ||
        !safe_add_size(total_floats, fc3_w_count, &total_floats) ||
        !safe_add_size(total_floats, 1u, &total_floats)) {
        return false;
    }

    float *blob = (float *)malloc(total_floats * sizeof(float));
    if (blob == NULL) {
        return false;
    }

    float *ptr = blob;
    float *emb = ptr; ptr += emb_count;
    float *phase_emb = ptr; ptr += phase_count;
    float *fc1_w = ptr; ptr += fc1_w_count;
    float *fc1_b = ptr; ptr += fc1_b_count;
    float *fc2_w = ptr; ptr += fc2_w_count;
    float *fc2_b = ptr; ptr += fc2_b_count;
    float *fc3_w = ptr; ptr += fc3_w_count;
    float *fc3_b_ptr = ptr;

    bool ok = read_exact(fp, emb, emb_count * sizeof(float)) &&
              read_exact(fp, phase_emb, phase_count * sizeof(float)) &&
              read_exact(fp, fc1_w, fc1_w_count * sizeof(float)) &&
              read_exact(fp, fc1_b, fc1_b_count * sizeof(float)) &&
              read_exact(fp, fc2_w, fc2_w_count * sizeof(float)) &&
              read_exact(fp, fc2_b, fc2_b_count * sizeof(float)) &&
              read_exact(fp, fc3_w, fc3_w_count * sizeof(float)) &&
              read_exact(fp, fc3_b_ptr, sizeof(float));
    if (!ok) {
        free(blob);
        return false;
    }

    memset(model, 0, sizeof(*model));
    model->loaded = true;
    model->kind = NN_MODEL_KIND_LEGACY_RUN1;
    snprintf(model->path, sizeof(model->path), "%s", path);
    model->legacy_blob = blob;
    model->legacy_num_features = header->num_features;
    model->legacy_hidden = header->hidden;
    model->legacy_head1 = header->head1;
    model->legacy_head2 = header->head2;
    model->legacy_flags = header->flags;
    model->legacy_king_buckets = header->king_buckets;
    model->legacy_phase_dim = header->phase_dim;
    model->legacy_tanh_scale = header->tanh_scale;
    model->legacy_cp_clip = header->cp_clip;
    model->legacy_emb = emb;
    model->legacy_phase_emb = phase_emb;
    model->legacy_fc1_w = fc1_w;
    model->legacy_fc1_b = fc1_b;
    model->legacy_fc2_w = fc2_w;
    model->legacy_fc2_b = fc2_b;
    model->legacy_fc3_w = fc3_w;
    model->legacy_fc3_b = *fc3_b_ptr;
    return true;
}

static int legacy_eval_cp_stm(const GameState *state, const NnEvalModel *model) {
    if (state == NULL || model == NULL || model->kind != NN_MODEL_KIND_LEGACY_RUN1) {
        return NN_CP_FALLBACK;
    }

    int idx_us[NN_LEGACY_MAX_ACTIVE];
    int idx_th[NN_LEGACY_MAX_ACTIVE];
    bool perspective_norm = (model->legacy_flags & NN_LEGACY_RUN1_FLAG_PERSPECTIVE_NORM) != 0u;
    int active = legacy_encode_run1_indices(
        state,
        perspective_norm,
        model->legacy_king_buckets,
        idx_us,
        idx_th
    );
    if (active <= 0) {
        return NN_CP_FALLBACK;
    }
    if (active > NN_LEGACY_MAX_ACTIVE) {
        active = NN_LEGACY_MAX_ACTIVE;
    }

    uint32_t h = model->legacy_hidden;
    uint32_t head1 = model->legacy_head1;
    uint32_t head2 = model->legacy_head2;
    uint32_t phase_dim = model->legacy_phase_dim;

    float acc_us[NN_LEGACY_MAX_HIDDEN];
    float acc_th[NN_LEGACY_MAX_HIDDEN];
    float x0[(NN_LEGACY_MAX_HIDDEN * 2) + NN_LEGACY_MAX_PHASE_DIM];
    float l1[NN_LEGACY_MAX_HEAD1];
    float l2[NN_LEGACY_MAX_HEAD2];

    for (uint32_t i = 0; i < h; ++i) {
        acc_us[i] = 0.0f;
        acc_th[i] = 0.0f;
    }
    for (int k = 0; k < active; ++k) {
        int fu = idx_us[k];
        int ft = idx_th[k];
        if (fu < 0 || ft < 0 ||
            (uint32_t)fu >= model->legacy_num_features ||
            (uint32_t)ft >= model->legacy_num_features) {
            continue;
        }
        const float *row_u = model->legacy_emb + ((size_t)fu * (size_t)h);
        const float *row_t = model->legacy_emb + ((size_t)ft * (size_t)h);
        for (uint32_t i = 0; i < h; ++i) {
            acc_us[i] += row_u[i];
            acc_th[i] += row_t[i];
        }
    }

    for (uint32_t i = 0; i < h; ++i) {
        x0[i] = acc_us[i];
        x0[h + i] = acc_th[i];
    }

    size_t x0_size = (size_t)h * 2u;
    if ((model->legacy_flags & NN_LEGACY_RUN1_FLAG_USE_PHASE) != 0u && phase_dim > 0u) {
        int phase_id = legacy_estimate_phase_id_from_state(state);
        if (phase_id < 0 || phase_id > 3) {
            phase_id = 3;
        }
        const float *prow = model->legacy_phase_emb + ((size_t)phase_id * (size_t)phase_dim);
        for (uint32_t i = 0; i < phase_dim; ++i) {
            x0[x0_size + i] = prow[i];
        }
        x0_size += phase_dim;
    }

    for (uint32_t o = 0; o < head1; ++o) {
        const float *w = model->legacy_fc1_w + ((size_t)o * x0_size);
        float sum = model->legacy_fc1_b[o];
        for (size_t i = 0; i < x0_size; ++i) {
            sum += w[i] * x0[i];
        }
        l1[o] = (sum > 0.0f) ? sum : 0.0f;
    }

    for (uint32_t o = 0; o < head2; ++o) {
        const float *w = model->legacy_fc2_w + ((size_t)o * (size_t)head1);
        float sum = model->legacy_fc2_b[o];
        for (uint32_t i = 0; i < head1; ++i) {
            sum += w[i] * l1[i];
        }
        l2[o] = (sum > 0.0f) ? sum : 0.0f;
    }

    float out = model->legacy_fc3_b;
    for (uint32_t i = 0; i < head2; ++i) {
        out += model->legacy_fc3_w[i] * l2[i];
    }

    float y = tanhf(out);
    if (y > 0.999999f) {
        y = 0.999999f;
    } else if (y < -0.999999f) {
        y = -0.999999f;
    }

    int cp_stm = (int)lroundf(atanhf(y) * model->legacy_tanh_scale);
    int clip_cp = (int)lroundf(model->legacy_cp_clip);
    if (clip_cp > 0) {
        if (cp_stm > clip_cp) {
            cp_stm = clip_cp;
        } else if (cp_stm < -clip_cp) {
            cp_stm = -clip_cp;
        }
    }
    return clamp_cp_int(cp_stm);
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
        for (int piece = PIECE_PAWN; piece <= PIECE_QUEEN; ++piece) {
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
    if (read_exact(fp, magic8, sizeof(magic8)) && memcmp(magic8, NN_MAGIC, sizeof(magic8)) == 0) {
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
    } else {
        if (fseek(fp, 0L, SEEK_SET) != 0) {
            fclose(fp);
            return false;
        }
        NnLegacyRun1Header legacy_header;
        if (read_exact(fp, &legacy_header, sizeof(legacy_header)) &&
            memcmp(legacy_header.magic, NN_LEGACY_MAGIC, NN_LEGACY_MAGIC_BYTES) == 0 &&
            legacy_header.version == NN_LEGACY_RUN1_VERSION) {
            ok = load_legacy_run1_model(fp, path, &legacy_header, &model);
        }
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
    if (g_nn_model.kind == NN_MODEL_KIND_LEGACY_RUN1) {
        return legacy_eval_cp_stm(state, &g_nn_model);
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
        return legacy_eval_cp_stm(state, &g_nn_model);
    }
    return evaluate_from_frame(state, &g_nn_model, frame);
}
