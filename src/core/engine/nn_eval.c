#include "nn_eval.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define NN_HAS_NEON 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define NN_HAS_X86_64 1
#endif

#include "chess_types.h"

#if defined(__GNUC__) || defined(__clang__)
#define NN_MAYBE_UNUSED __attribute__((unused))
#else
#define NN_MAYBE_UNUSED
#endif

#define NN_MAGIC_BYTES 8
#define NN_MAGIC "CHNNUE1\0"
#define NN_VERSION_FLOAT 1u
#define NN_VERSION_QUANT 2u
#define NN_VERSION_LINEAR_HEAD_QUANT 3u
#define NN_VERSION_BOTTLENECK_HEAD_QUANT 4u
#define NN_VERSION_LINEAR_HEAD_FIXED_ACT 5u
#define NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT 6u
#define NN_VERSION_LINEAR_HEAD_SCRELU 7u
#define NN_VERSION_BOTTLENECK_HEAD_SCRELU 8u
#define NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC 9u
#define NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS 10u
#define NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT 11u
#define NN_MAX_OUTPUT_BUCKETS 32u
#define NN_MAX_HIDDEN_DIM 128u
#define NN_EXPECTED_HALFKP_DIM (64u * 10u * 64u)
#define NN_EXPECTED_HALFKP_HM_DIM (32u * 10u * 64u)
#define NN_EXPECTED_HALFKA_HM_DIM (32u * 11u * 64u)
#define NN_MAX_TRANSFORM_DIM (NN_MAX_ACC_DIM * 2u)
#define NN_CP_FALLBACK 0
#define NN_ACT_QMAX 32767
#define NN_FIXED_ACT_QMAX 127
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
    uint32_t bottleneck_dim;
    float act0_scale;
    float act1_scale;
    float act2_scale;
    uint32_t num_buckets;
    float psqt_scale;
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
    float *out_bias_buckets;
    int16_t *psqt_weight;
    int32_t acc_act_multiplier;
    uint8_t acc_act_shift;
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
    free(model->out_bias_buckets);
    free(model->psqt_weight);
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
    if (header->version == NN_VERSION_QUANT ||
        header->version == NN_VERSION_LINEAR_HEAD_QUANT ||
        header->version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
        header->version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
        header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
        header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
        if (!read_exact(fp, &header->acc_scale, sizeof(float)) ||
            !read_exact(fp, &header->fc1_scale, sizeof(float)) ||
            !read_exact(fp, &header->fc2_scale, sizeof(float)) ||
            !read_exact(fp, &header->out_scale, sizeof(float))) {
            return false;
        }
        if ((header->version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
             header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
             header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
             header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
             header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
             header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
             header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) &&
            (!read_exact(fp, &header->act0_scale, sizeof(float)) ||
             !read_exact(fp, &header->act1_scale, sizeof(float)) ||
             !read_exact(fp, &header->act2_scale, sizeof(float)))) {
            return false;
        }
        if ((header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
             header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) &&
            !read_exact(fp, &header->num_buckets, sizeof(uint32_t))) {
            return false;
        }
        if (header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT &&
            !read_exact(fp, &header->psqt_scale, sizeof(float))) {
            return false;
        }
        if ((header->version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
             header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
             header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU) &&
            !read_exact(fp, &header->bottleneck_dim, sizeof(uint32_t))) {
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
        case PIECE_KING:
            return 10;
        default:
            return -1;
    }
    return own ? base : (base + 5);
}

static int halfkp_index(int king_sq,
                        int plane,
                        int piece_sq,
                        int perspective,
                        uint32_t feature_dim) {
    int oriented_king = orient_square(king_sq, perspective);
    int oriented_piece = orient_square(piece_sq, perspective);
    if (feature_dim == NN_EXPECTED_HALFKP_HM_DIM ||
        feature_dim == NN_EXPECTED_HALFKA_HM_DIM) {
        int planes = feature_dim == NN_EXPECTED_HALFKA_HM_DIM ? 11 : 10;
        if ((oriented_king & 7) < 4) {
            oriented_king ^= 7;
            oriented_piece ^= 7;
        }
        int king_bucket = (oriented_king >> 3) * 4 + (oriented_king & 7) - 4;
        return (king_bucket * planes + plane) * 64 + oriented_piece;
    }
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

static bool header_uses_fc2(const NnEvalHeader *header) {
    return header != NULL &&
           header->version != NN_VERSION_LINEAR_HEAD_QUANT &&
           header->version != NN_VERSION_LINEAR_HEAD_FIXED_ACT &&
           header->version != NN_VERSION_LINEAR_HEAD_SCRELU &&
           header->version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC &&
           header->version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS &&
           header->version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT;
}

static bool header_uses_fixed_activation(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
            header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
            header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT);
}

static bool header_uses_squared_clipped_relu(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
            header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT);
}

static bool header_uses_i16_accumulator(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT);
}

static uint32_t header_output_buckets(const NnEvalHeader *header) {
    if (header == NULL ||
        (header->version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS &&
         header->version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT)) {
        return 1u;
    }
    return header->num_buckets;
}

static NN_MAYBE_UNUSED int64_t dot_i8_i32_scalar(const int8_t *weights, const int32_t *values, uint32_t count) {
    int64_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += (int64_t)weights[i] * (int64_t)values[i];
    }
    return sum;
}

static NN_MAYBE_UNUSED int64_t dot_i8_i16_scalar(const int8_t *weights, const int16_t *values, uint32_t count) {
    int64_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += (int64_t)weights[i] * (int64_t)values[i];
    }
    return sum;
}

static NN_MAYBE_UNUSED void add_row_scalar(int32_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
    for (uint32_t i = 0; i < acc_dim; ++i) {
        acc[i] += sign * (int32_t)row[i];
    }
}

#if defined(NN_HAS_NEON)
static int64_t horizontal_add_s64x2(int64x2_t value) {
    return vgetq_lane_s64(value, 0) + vgetq_lane_s64(value, 1);
}

static int64_t dot_i8_i32_neon(const int8_t *weights, const int32_t *values, uint32_t count) {
    int64x2_t acc64 = vdupq_n_s64(0);
    uint32_t i = 0;
    for (; i + 8u <= count; i += 8u) {
        int8x8_t w8 = vld1_s8(weights + i);
        int16x8_t w16 = vmovl_s8(w8);
        int32x4_t w0 = vmovl_s16(vget_low_s16(w16));
        int32x4_t w1 = vmovl_s16(vget_high_s16(w16));
        int32x4_t v0 = vld1q_s32(values + i);
        int32x4_t v1 = vld1q_s32(values + i + 4u);
        acc64 = vpadalq_s32(acc64, vmulq_s32(w0, v0));
        acc64 = vpadalq_s32(acc64, vmulq_s32(w1, v1));
    }
    int64_t sum = horizontal_add_s64x2(acc64);
    for (; i < count; ++i) {
        sum += (int64_t)weights[i] * (int64_t)values[i];
    }
    return sum;
}

static int64_t dot_i8_i16_neon(const int8_t *weights, const int16_t *values, uint32_t count) {
    int64x2_t acc64 = vdupq_n_s64(0);
    uint32_t i = 0;
    for (; i + 8u <= count; i += 8u) {
        int8x8_t w8 = vld1_s8(weights + i);
        int16x8_t w16 = vmovl_s8(w8);
        int16x8_t v16 = vld1q_s16(values + i);
        acc64 = vpadalq_s32(acc64, vmull_s16(vget_low_s16(w16), vget_low_s16(v16)));
        acc64 = vpadalq_s32(acc64, vmull_s16(vget_high_s16(w16), vget_high_s16(v16)));
    }
    int64_t sum = horizontal_add_s64x2(acc64);
    for (; i < count; ++i) {
        sum += (int64_t)weights[i] * (int64_t)values[i];
    }
    return sum;
}

static void add_row_neon(int32_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
    uint32_t i = 0;
    for (; i + 8u <= acc_dim; i += 8u) {
        int16x8_t r16 = vld1q_s16(row + i);
        int32x4_t r0 = vmovl_s16(vget_low_s16(r16));
        int32x4_t r1 = vmovl_s16(vget_high_s16(r16));
        int32x4_t a0 = vld1q_s32(acc + i);
        int32x4_t a1 = vld1q_s32(acc + i + 4u);
        if (sign >= 0) {
            a0 = vaddq_s32(a0, r0);
            a1 = vaddq_s32(a1, r1);
        } else {
            a0 = vsubq_s32(a0, r0);
            a1 = vsubq_s32(a1, r1);
        }
        vst1q_s32(acc + i, a0);
        vst1q_s32(acc + i + 4u, a1);
    }
    for (; i < acc_dim; ++i) {
        acc[i] += sign * (int32_t)row[i];
    }
}
#endif

#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static int64_t horizontal_add_s64x4_avx2(__m256i value) {
    int64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

__attribute__((target("avx2")))
static int64_t dot_i8_i32_avx2(const int8_t *weights, const int32_t *values, uint32_t count) {
    __m256i acc64 = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; i + 8u <= count; i += 8u) {
        __m128i w8 = _mm_loadl_epi64((const __m128i *)(const void *)(weights + i));
        __m256i w32 = _mm256_cvtepi8_epi32(w8);
        __m256i v32 = _mm256_loadu_si256((const __m256i *)(const void *)(values + i));
        __m256i prod32 = _mm256_mullo_epi32(w32, v32);
        __m128i lo = _mm256_castsi256_si128(prod32);
        __m128i hi = _mm256_extracti128_si256(prod32, 1);
        acc64 = _mm256_add_epi64(acc64, _mm256_cvtepi32_epi64(lo));
        acc64 = _mm256_add_epi64(acc64, _mm256_cvtepi32_epi64(hi));
    }
    int64_t sum = horizontal_add_s64x4_avx2(acc64);
    for (; i < count; ++i) {
        sum += (int64_t)weights[i] * (int64_t)values[i];
    }
    return sum;
}

__attribute__((target("avx2")))
static int64_t dot_i8_i16_avx2(const int8_t *weights, const int16_t *values, uint32_t count) {
    __m256i acc64 = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; i + 16u <= count; i += 16u) {
        __m128i w8 = _mm_loadu_si128((const __m128i *)(const void *)(weights + i));
        __m256i w16 = _mm256_cvtepi8_epi16(w8);
        __m256i v16 = _mm256_loadu_si256((const __m256i *)(const void *)(values + i));
        __m256i pair32 = _mm256_madd_epi16(w16, v16);
        __m128i lo = _mm256_castsi256_si128(pair32);
        __m128i hi = _mm256_extracti128_si256(pair32, 1);
        acc64 = _mm256_add_epi64(acc64, _mm256_cvtepi32_epi64(lo));
        acc64 = _mm256_add_epi64(acc64, _mm256_cvtepi32_epi64(hi));
    }
    int64_t sum = horizontal_add_s64x4_avx2(acc64);
    for (; i < count; ++i) {
        sum += (int64_t)weights[i] * (int64_t)values[i];
    }
    return sum;
}

__attribute__((target("avx2")))
static void add_row_avx2(int32_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
    uint32_t i = 0;
    for (; i + 16u <= acc_dim; i += 16u) {
        __m128i r16 = _mm_loadu_si128((const __m128i *)(const void *)(row + i));
        __m256i r0 = _mm256_cvtepi16_epi32(r16);
        __m256i r1 = _mm256_cvtepi16_epi32(_mm_srli_si128(r16, 8));
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)(acc + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(const void *)(acc + i + 8u));
        if (sign >= 0) {
            a0 = _mm256_add_epi32(a0, r0);
            a1 = _mm256_add_epi32(a1, r1);
        } else {
            a0 = _mm256_sub_epi32(a0, r0);
            a1 = _mm256_sub_epi32(a1, r1);
        }
        _mm256_storeu_si256((__m256i *)(void *)(acc + i), a0);
        _mm256_storeu_si256((__m256i *)(void *)(acc + i + 8u), a1);
    }
    for (; i < acc_dim; ++i) {
        acc[i] += sign * (int32_t)row[i];
    }
}

static bool cpu_supports_avx2(void) {
    static int cached = -1;
    if (cached < 0) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") ? 1 : 0;
#else
        cached = 0;
#endif
    }
    return cached != 0;
}
#endif

static int64_t dot_i8_i32(const int8_t *weights, const int32_t *values, uint32_t count) {
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        return dot_i8_i32_avx2(weights, values, count);
    }
#endif
#if defined(NN_HAS_NEON)
    return dot_i8_i32_neon(weights, values, count);
#else
    return dot_i8_i32_scalar(weights, values, count);
#endif
}

static int64_t dot_i8_i16(const int8_t *weights, const int16_t *values, uint32_t count) {
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        return dot_i8_i16_avx2(weights, values, count);
    }
#endif
#if defined(NN_HAS_NEON)
    return dot_i8_i16_neon(weights, values, count);
#else
    return dot_i8_i16_scalar(weights, values, count);
#endif
}

static void add_row_fast(int32_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        add_row_avx2(acc, row, acc_dim, sign);
        return;
    }
#endif
#if defined(NN_HAS_NEON)
    add_row_neon(acc, row, acc_dim, sign);
#else
    add_row_scalar(acc, row, acc_dim, sign);
#endif
}

static NN_MAYBE_UNUSED void add_row_i16_scalar(int16_t *acc,
                                                const int16_t *row,
                                                uint32_t acc_dim,
                                                int sign) {
    for (uint32_t i = 0; i < acc_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + sign * row[i]);
    }
}

#if defined(NN_HAS_NEON)
static void add_row_i16_neon(int16_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
    uint32_t i = 0;
    for (; i + 8u <= acc_dim; i += 8u) {
        int16x8_t a = vld1q_s16(acc + i);
        int16x8_t r = vld1q_s16(row + i);
        vst1q_s16(acc + i, sign >= 0 ? vaddq_s16(a, r) : vsubq_s16(a, r));
    }
    for (; i < acc_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + sign * row[i]);
    }
}
#endif

#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static void add_row_i16_avx2(int16_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
    uint32_t i = 0;
    for (; i + 16u <= acc_dim; i += 16u) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)(acc + i));
        __m256i r = _mm256_loadu_si256((const __m256i *)(const void *)(row + i));
        __m256i result = sign >= 0 ? _mm256_add_epi16(a, r) : _mm256_sub_epi16(a, r);
        _mm256_storeu_si256((__m256i *)(void *)(acc + i), result);
    }
    for (; i < acc_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + sign * row[i]);
    }
}
#endif

static void add_row_i16_fast(int16_t *acc, const int16_t *row, uint32_t acc_dim, int sign) {
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        add_row_i16_avx2(acc, row, acc_dim, sign);
        return;
    }
#endif
#if defined(NN_HAS_NEON)
    add_row_i16_neon(acc, row, acc_dim, sign);
#else
    add_row_i16_scalar(acc, row, acc_dim, sign);
#endif
}

static size_t header_fc1_out_dim(const NnEvalHeader *header) {
    if (header == NULL) {
        return 0u;
    }
    if (header->version == NN_VERSION_LINEAR_HEAD_QUANT ||
        header->version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
        return (size_t)header->hidden_dim;
    }
    if (header->version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
        header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
        header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU) {
        return (size_t)header->bottleneck_dim;
    }
    return (size_t)header->accumulator_dim;
}

static bool allocate_quantized_model(NnEvalModel *model, const NnEvalHeader *header) {
    if (model == NULL || header == NULL) {
        return false;
    }
    size_t acc_rows = (size_t)header->halfkp_dim + 1u;
    size_t acc_count = acc_rows * (size_t)header->accumulator_dim;
    size_t buckets = (size_t)header_output_buckets(header);
    size_t fc1_out = header_fc1_out_dim(header);
    size_t fc1_count = buckets * fc1_out * ((size_t)header->accumulator_dim * 2u);
    size_t fc2_count = (size_t)header->hidden_dim * fc1_out;

    model->acc_weight = (int16_t *)malloc(acc_count * sizeof(int16_t));
    model->fc1_weight = (int8_t *)malloc(fc1_count * sizeof(int8_t));
    model->fc1_bias = (float *)malloc(buckets * fc1_out * sizeof(float));
    if (!header_uses_fc2(header)) {
        model->fc2_weight = NULL;
        model->fc2_bias = NULL;
    } else {
        model->fc2_weight = (int8_t *)malloc(fc2_count * sizeof(int8_t));
        model->fc2_bias = (float *)malloc((size_t)header->hidden_dim * sizeof(float));
    }
    model->out_weight = (int8_t *)malloc(buckets * (size_t)header->hidden_dim * sizeof(int8_t));
    if (header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
        header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
        model->out_bias_buckets = (float *)malloc(buckets * sizeof(float));
    } else {
        model->out_bias_buckets = NULL;
    }
    model->psqt_weight = header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT
                             ? (int16_t *)malloc(acc_rows * buckets * sizeof(int16_t))
                             : NULL;
    if (model->acc_weight == NULL || model->fc1_weight == NULL || model->fc1_bias == NULL ||
        model->out_weight == NULL ||
        ((header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
          header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) &&
         model->out_bias_buckets == NULL) ||
        (header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT &&
         model->psqt_weight == NULL) ||
        (header_uses_fc2(header) && (model->fc2_weight == NULL || model->fc2_bias == NULL))) {
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

static int nn_logit_to_cp(float logit, float cp_scale) {
    float cp = tanhf(logit) * cp_scale;
    if (cp > 32000.0f) {
        cp = 32000.0f;
    } else if (cp < -32000.0f) {
        cp = -32000.0f;
    }
    return (int)lroundf(cp);
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

static void quantize_activation_fixed_relu(const float *src, uint32_t count, float scale, int16_t *dst) {
    if (scale <= 1e-12f) {
        scale = 1.0f / (float)NN_FIXED_ACT_QMAX;
    }
    for (uint32_t i = 0; i < count; ++i) {
        float v = src[i] > 0.0f ? src[i] : 0.0f;
        long q = lroundf(v / scale);
        if (q < 0L) {
            q = 0L;
        } else if (q > (long)NN_FIXED_ACT_QMAX) {
            q = (long)NN_FIXED_ACT_QMAX;
        }
        dst[i] = (int16_t)q;
    }
}

static void square_quantized_activation(int16_t *values, uint32_t count) {
    uint32_t i = 0;
#if defined(NN_HAS_NEON)
    for (; i + 8u <= count; i += 8u) {
        int16x8_t value = vld1q_s16(values + i);
        vst1q_s16(values + i, vmulq_s16(value, value));
    }
#endif
    for (; i < count; ++i) {
        int32_t v = values[i];
        values[i] = (int16_t)(v * v);
    }
}

static void quantize_accumulator_fixed_relu(const int32_t *src,
                                            uint32_t count,
                                            float acc_scale,
                                            float act_scale,
                                            int16_t *dst) {
    if (act_scale <= 1e-12f) {
        act_scale = 1.0f / (float)NN_FIXED_ACT_QMAX;
    }
    float factor = acc_scale / act_scale;
    for (uint32_t i = 0; i < count; ++i) {
        long q = lroundf((float)src[i] * factor);
        if (q < 0L) {
            q = 0L;
        } else if (q > (long)NN_FIXED_ACT_QMAX) {
            q = (long)NN_FIXED_ACT_QMAX;
        }
        dst[i] = (int16_t)q;
    }
}

static void quantize_accumulator_i16_relu(const int16_t *src,
                                           uint32_t count,
                                           int32_t multiplier,
                                           uint8_t shift,
                                           int16_t *dst) {
    const int64_t rounding = shift > 0 ? (INT64_C(1) << (shift - 1u)) : 0;
    uint32_t i = 0;
#if defined(NN_HAS_NEON)
    if (shift == 16u && multiplier <= INT16_MAX) {
        const int16x8_t zero = vdupq_n_s16(0);
        const int16x8_t max_value = vdupq_n_s16(NN_FIXED_ACT_QMAX);
        const int32x4_t round_value = vdupq_n_s32(1 << 15);
        const int16_t multiplier16 = (int16_t)multiplier;
        for (; i + 8u <= count; i += 8u) {
            int16x8_t value = vmaxq_s16(vld1q_s16(src + i), zero);
            int32x4_t lo = vmull_n_s16(vget_low_s16(value), multiplier16);
            int32x4_t hi = vmull_n_s16(vget_high_s16(value), multiplier16);
            lo = vshrq_n_s32(vaddq_s32(lo, round_value), 16);
            hi = vshrq_n_s32(vaddq_s32(hi, round_value), 16);
            int16x8_t packed = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
            vst1q_s16(dst + i, vminq_s16(packed, max_value));
        }
    }
#endif
    for (; i < count; ++i) {
        int32_t value = src[i];
        if (value <= 0) {
            dst[i] = 0;
            continue;
        }
        int64_t scaled = (int64_t)value * multiplier + rounding;
        int64_t q = shift > 0 ? (scaled >> shift) : scaled;
        if (q > NN_FIXED_ACT_QMAX) {
            q = NN_FIXED_ACT_QMAX;
        }
        dst[i] = (int16_t)q;
    }
}

static void accumulate_perspective(const GameState *state,
                                   const NnEvalModel *model,
                                   int perspective,
                                   int32_t *out_acc,
                                   uint16_t *feature_count_out) {
    const uint32_t acc_dim = model->header.accumulator_dim;
    memset(out_acc, 0, (size_t)acc_dim * sizeof(out_acc[0]));

    int king_sq = chess_find_king_square(state, perspective);
    if (king_sq < 0 || king_sq >= 64) {
        return;
    }

    bool had_feature = false;
    uint16_t feature_count = 0;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        int first_piece = model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM
                              ? PIECE_KING : PIECE_QUEEN;
        for (int piece = first_piece; piece <= PIECE_PAWN; ++piece) {
            int plane = piece_plane(piece, color, perspective);
            if (plane < 0) {
                continue;
            }
            uint64_t bb = state->bb[color][piece];
            while (bb != 0ULL) {
                int sq = chess_pop_lsb(&bb);
                int idx = halfkp_index(king_sq, plane, sq, perspective, model->header.halfkp_dim);
                if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
                    continue;
                }
                const int16_t *row = model->acc_weight + ((size_t)idx * acc_dim);
                add_row_fast(out_acc, row, acc_dim, 1);
                had_feature = true;
                feature_count += piece != PIECE_KING;
            }
        }
    }

    if (!had_feature) {
        const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * acc_dim);
        add_row_fast(out_acc, row, acc_dim, 1);
    }
    if (feature_count_out != NULL) {
        *feature_count_out = feature_count;
    }
}

static void accumulate_perspective_i16(const GameState *state,
                                       const NnEvalModel *model,
                                       int perspective,
                                       int16_t *out_acc,
                                       uint16_t *feature_count_out) {
    const uint32_t acc_dim = model->header.accumulator_dim;
    memset(out_acc, 0, (size_t)acc_dim * sizeof(out_acc[0]));

    int king_sq = chess_find_king_square(state, perspective);
    if (king_sq < 0 || king_sq >= 64) {
        return;
    }

    bool had_feature = false;
    uint16_t feature_count = 0;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        int first_piece = model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM
                              ? PIECE_KING : PIECE_QUEEN;
        for (int piece = first_piece; piece <= PIECE_PAWN; ++piece) {
            int plane = piece_plane(piece, color, perspective);
            if (plane < 0) {
                continue;
            }
            uint64_t bb = state->bb[color][piece];
            while (bb != 0ULL) {
                int sq = chess_pop_lsb(&bb);
                int idx = halfkp_index(king_sq, plane, sq, perspective, model->header.halfkp_dim);
                if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
                    continue;
                }
                const int16_t *row = model->acc_weight + ((size_t)idx * acc_dim);
                add_row_i16_fast(out_acc, row, acc_dim, 1);
                had_feature = true;
                feature_count += piece != PIECE_KING;
            }
        }
    }

    if (!had_feature) {
        const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * acc_dim);
        add_row_i16_fast(out_acc, row, acc_dim, 1);
    }
    if (feature_count_out != NULL) {
        *feature_count_out = feature_count;
    }
}

static void accumulate_psqt_perspective(const GameState *state,
                                        const NnEvalModel *model,
                                        int perspective,
                                        int32_t out_psqt[NN_MAX_OUTPUT_BUCKETS]) {
    memset(out_psqt, 0, NN_MAX_OUTPUT_BUCKETS * sizeof(out_psqt[0]));
    if (model->psqt_weight == NULL) return;
    int king_sq = chess_find_king_square(state, perspective);
    if (king_sq < 0 || king_sq >= 64) return;
    uint32_t buckets = header_output_buckets(&model->header);
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        int first_piece = model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM
                              ? PIECE_KING : PIECE_QUEEN;
        for (int piece = first_piece; piece <= PIECE_PAWN; ++piece) {
            int plane = piece_plane(piece, color, perspective);
            uint64_t bb = state->bb[color][piece];
            while (bb != 0ULL) {
                int sq = chess_pop_lsb(&bb);
                int idx = halfkp_index(king_sq, plane, sq, perspective,
                                       model->header.halfkp_dim);
                if (idx < 0 || (uint32_t)idx >= model->header.dummy_index) continue;
                const int16_t *row = model->psqt_weight + (size_t)idx * buckets;
                for (uint32_t bucket = 0; bucket < buckets; ++bucket) {
                    out_psqt[bucket] += row[bucket];
                }
            }
        }
    }
}

static bool update_piece_psqt(int32_t psqt[NN_MAX_OUTPUT_BUCKETS],
                              const NnEvalModel *model,
                              int perspective,
                              int king_sq,
                              int color,
                              int piece,
                              int sq,
                              int sign) {
    if (model->psqt_weight == NULL) return true;
    int plane = piece_plane(piece, color, perspective);
    if (plane < 0) return true;
    int idx = halfkp_index(king_sq, plane, sq, perspective, model->header.halfkp_dim);
    if (idx < 0 || (uint32_t)idx >= model->header.dummy_index) return false;
    uint32_t buckets = header_output_buckets(&model->header);
    const int16_t *row = model->psqt_weight + (size_t)idx * buckets;
    for (uint32_t bucket = 0; bucket < buckets; ++bucket) {
        psqt[bucket] += sign * row[bucket];
    }
    return true;
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
    int idx = halfkp_index(king_sq, plane, sq, perspective, model->header.halfkp_dim);
    if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
        return false;
    }
    const int16_t *row = model->acc_weight + ((size_t)idx * model->header.accumulator_dim);
    add_row_fast(acc, row, model->header.accumulator_dim, sign);
    return true;
}

static void add_dummy_if_needed(int32_t *acc, const NnEvalModel *model) {
    const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * model->header.accumulator_dim);
    add_row_fast(acc, row, model->header.accumulator_dim, 1);
}

static void remove_dummy_if_needed(int32_t *acc, const NnEvalModel *model) {
    const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * model->header.accumulator_dim);
    add_row_fast(acc, row, model->header.accumulator_dim, -1);
}

static bool update_piece_feature_i16(int16_t *acc,
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
    int idx = halfkp_index(king_sq, plane, sq, perspective, model->header.halfkp_dim);
    if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
        return false;
    }
    const int16_t *row = model->acc_weight + ((size_t)idx * model->header.accumulator_dim);
    add_row_i16_fast(acc, row, model->header.accumulator_dim, sign);
    return true;
}

static void update_dummy_i16(int16_t *acc, const NnEvalModel *model, int sign) {
    const int16_t *row = model->acc_weight + ((size_t)model->header.dummy_index * model->header.accumulator_dim);
    add_row_i16_fast(acc, row, model->header.accumulator_dim, sign);
}

static bool rebuild_frame(const GameState *state, NnAccumulatorFrame *frame) {
    if (state == NULL || frame == NULL || !g_nn_model.loaded || g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return false;
    }
    const NnEvalModel *model = &g_nn_model;
    uint16_t white_count = 0;
    if (header_uses_i16_accumulator(&model->header)) {
        accumulate_perspective_i16(state, model, PIECE_WHITE, frame->white_acc16, &white_count);
        accumulate_perspective_i16(state, model, PIECE_BLACK, frame->black_acc16, NULL);
    } else {
        accumulate_perspective(state, model, PIECE_WHITE, frame->white_acc, &white_count);
        accumulate_perspective(state, model, PIECE_BLACK, frame->black_acc, NULL);
    }
    accumulate_psqt_perspective(state, model, PIECE_WHITE, frame->white_psqt);
    accumulate_psqt_perspective(state, model, PIECE_BLACK, frame->black_psqt);
    frame->valid = true;
    frame->key = state->zobrist_hash;
    frame->non_king_piece_count = white_count;
    return true;
}

static int evaluate_from_frame(const GameState *state, const NnEvalModel *model, const NnAccumulatorFrame *frame) {
    if (state == NULL || model == NULL || frame == NULL || !frame->valid) {
        return NN_CP_FALLBACK;
    }
    const uint32_t acc_dim = model->header.accumulator_dim;
    const uint32_t fc1_out_dim = (uint32_t)header_fc1_out_dim(&model->header);
    const uint32_t hidden_dim = model->header.hidden_dim;
    const int32_t *front_acc = (state->side_to_move == PIECE_WHITE) ? frame->white_acc : frame->black_acc;
    const int32_t *back_acc = (state->side_to_move == PIECE_WHITE) ? frame->black_acc : frame->white_acc;
    float hidden1_f[NN_MAX_ACC_DIM] = {0};
    int16_t hidden1_q[NN_MAX_ACC_DIM] = {0};
    float hidden2_f[NN_MAX_HIDDEN_DIM] = {0};
    int16_t hidden2_q[NN_MAX_HIDDEN_DIM] = {0};

    float fc1_factor = model->header.acc_scale * model->header.fc1_scale;
    for (uint32_t out = 0; out < fc1_out_dim; ++out) {
        const int8_t *w = model->fc1_weight + ((size_t)out * (size_t)acc_dim * 2u);
        int64_t sum = dot_i8_i32(w, front_acc, acc_dim) + dot_i8_i32(w + acc_dim, back_acc, acc_dim);
        hidden1_f[out] = (float)sum * fc1_factor + model->fc1_bias[out];
    }
    float hidden1_scale = quantize_activation_relu(hidden1_f, fc1_out_dim, hidden1_q);

    float fc2_factor = hidden1_scale * model->header.fc2_scale;
    for (uint32_t out = 0; out < hidden_dim; ++out) {
        const int8_t *w = model->fc2_weight + ((size_t)out * fc1_out_dim);
        int64_t sum = dot_i8_i16(w, hidden1_q, fc1_out_dim);
        hidden2_f[out] = (float)sum * fc2_factor + model->fc2_bias[out];
    }
    float hidden2_scale = quantize_activation_relu(hidden2_f, hidden_dim, hidden2_q);

    int64_t out_sum = dot_i8_i16(model->out_weight, hidden2_q, hidden_dim);
    float out_value = (float)out_sum * (hidden2_scale * model->header.out_scale) + model->out_bias;
    return nn_logit_to_cp(out_value, model->header.cp_scale);
}

static int evaluate_fixed_from_frame(const GameState *state, const NnEvalModel *model, const NnAccumulatorFrame *frame) {
    if (state == NULL || model == NULL || frame == NULL || !frame->valid) {
        return NN_CP_FALLBACK;
    }
    const uint32_t acc_dim = model->header.accumulator_dim;
    const uint32_t fc1_out_dim = (uint32_t)header_fc1_out_dim(&model->header);
    const uint32_t hidden_dim = model->header.hidden_dim;
    const int32_t *front_acc = (state->side_to_move == PIECE_WHITE) ? frame->white_acc : frame->black_acc;
    const int32_t *back_acc = (state->side_to_move == PIECE_WHITE) ? frame->black_acc : frame->white_acc;
    int16_t transformed[NN_MAX_TRANSFORM_DIM] = {0};
    float hidden1_f[NN_MAX_ACC_DIM] = {0};
    int16_t hidden1_q[NN_MAX_ACC_DIM] = {0};
    float hidden2_f[NN_MAX_HIDDEN_DIM] = {0};
    int16_t hidden2_q[NN_MAX_HIDDEN_DIM] = {0};

    quantize_accumulator_fixed_relu(front_acc, acc_dim, model->header.acc_scale, model->header.act0_scale, transformed);
    quantize_accumulator_fixed_relu(back_acc,
                                    acc_dim,
                                    model->header.acc_scale,
                                    model->header.act0_scale,
                                    transformed + acc_dim);

    bool screlu = header_uses_squared_clipped_relu(&model->header);
    if (screlu) {
        square_quantized_activation(transformed, acc_dim * 2u);
    }

    float fc1_factor = (screlu ? (model->header.act0_scale * model->header.act0_scale) : model->header.act0_scale) *
                       model->header.fc1_scale;
    for (uint32_t out = 0; out < fc1_out_dim; ++out) {
        const int8_t *w = model->fc1_weight + ((size_t)out * (size_t)acc_dim * 2u);
        int64_t sum = dot_i8_i16(w, transformed, acc_dim * 2u);
        hidden1_f[out] = (float)sum * fc1_factor + model->fc1_bias[out];
    }
    quantize_activation_fixed_relu(hidden1_f, fc1_out_dim, model->header.act1_scale, hidden1_q);
    if (screlu) {
        square_quantized_activation(hidden1_q, fc1_out_dim);
    }

    if (model->header.version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
        model->header.version == NN_VERSION_LINEAR_HEAD_SCRELU) {
        int64_t out_sum = dot_i8_i16(model->out_weight, hidden1_q, hidden_dim);
        float out_factor = (screlu ? (model->header.act1_scale * model->header.act1_scale) : model->header.act1_scale) *
                           model->header.out_scale;
        float out_value = (float)out_sum * out_factor + model->out_bias;
        return nn_logit_to_cp(out_value, model->header.cp_scale);
    }

    float fc2_factor = (screlu ? (model->header.act1_scale * model->header.act1_scale) : model->header.act1_scale) *
                       model->header.fc2_scale;
    for (uint32_t out = 0; out < hidden_dim; ++out) {
        const int8_t *w = model->fc2_weight + ((size_t)out * fc1_out_dim);
        int64_t sum = dot_i8_i16(w, hidden1_q, fc1_out_dim);
        hidden2_f[out] = (float)sum * fc2_factor + model->fc2_bias[out];
    }
    quantize_activation_fixed_relu(hidden2_f, hidden_dim, model->header.act2_scale, hidden2_q);
    if (screlu) {
        square_quantized_activation(hidden2_q, hidden_dim);
    }

    int64_t out_sum = dot_i8_i16(model->out_weight, hidden2_q, hidden_dim);
    float out_factor = (screlu ? (model->header.act2_scale * model->header.act2_scale) : model->header.act2_scale) *
                       model->header.out_scale;
    float out_value = (float)out_sum * out_factor + model->out_bias;
    return nn_logit_to_cp(out_value, model->header.cp_scale);
}

static uint32_t nn_material_bucket(const GameState *state, uint32_t num_buckets) {
    uint32_t total = 0;
    for (int color = PIECE_WHITE; color < PIECE_COLOR_COUNT; ++color) {
        for (int piece = PIECE_KING; piece < PIECE_TYPE_COUNT; ++piece) {
            total += (uint32_t)__builtin_popcountll(state->bb[color][piece]);
        }
    }
    if (total == 0) {
        return 0;
    }
    uint32_t bucket = (total - 1u) / 4u;
    return bucket < num_buckets ? bucket : num_buckets - 1u;
}

static int evaluate_i16_screlu_from_frame(const GameState *state,
                                          const NnEvalModel *model,
                                          const NnAccumulatorFrame *frame) {
    if (state == NULL || model == NULL || frame == NULL || !frame->valid) {
        return NN_CP_FALLBACK;
    }
    const uint32_t acc_dim = model->header.accumulator_dim;
    const uint32_t hidden_dim = model->header.hidden_dim;
    const int16_t *front_acc =
        state->side_to_move == PIECE_WHITE ? frame->white_acc16 : frame->black_acc16;
    const int16_t *back_acc =
        state->side_to_move == PIECE_WHITE ? frame->black_acc16 : frame->white_acc16;
    int16_t transformed[NN_MAX_TRANSFORM_DIM] = {0};
    float hidden_f[NN_MAX_HIDDEN_DIM] = {0};
    int16_t hidden_q[NN_MAX_HIDDEN_DIM] = {0};

    quantize_accumulator_i16_relu(front_acc,
                                  acc_dim,
                                  model->acc_act_multiplier,
                                  model->acc_act_shift,
                                  transformed);
    quantize_accumulator_i16_relu(back_acc,
                                  acc_dim,
                                  model->acc_act_multiplier,
                                  model->acc_act_shift,
                                  transformed + acc_dim);
    square_quantized_activation(transformed, acc_dim * 2u);

    const uint32_t num_buckets = header_output_buckets(&model->header);
    uint32_t bucket = num_buckets > 1u ? nn_material_bucket(state, num_buckets) : 0u;
    const int8_t *fc1_weight =
        model->fc1_weight + (size_t)bucket * (size_t)hidden_dim * (size_t)acc_dim * 2u;
    const float *fc1_bias = model->fc1_bias + (size_t)bucket * (size_t)hidden_dim;
    const int8_t *out_weight = model->out_weight + (size_t)bucket * (size_t)hidden_dim;
    const float out_bias =
        model->out_bias_buckets != NULL ? model->out_bias_buckets[bucket] : model->out_bias;

    const float fc1_factor = model->header.act0_scale * model->header.act0_scale *
                             model->header.fc1_scale;
    for (uint32_t out = 0; out < hidden_dim; ++out) {
        const int8_t *w = fc1_weight + ((size_t)out * (size_t)acc_dim * 2u);
        int64_t sum = dot_i8_i16(w, transformed, acc_dim * 2u);
        hidden_f[out] = (float)sum * fc1_factor + fc1_bias[out];
    }
    quantize_activation_fixed_relu(hidden_f, hidden_dim, model->header.act1_scale, hidden_q);
    square_quantized_activation(hidden_q, hidden_dim);

    int64_t out_sum = dot_i8_i16(out_weight, hidden_q, hidden_dim);
    const float out_factor = model->header.act1_scale * model->header.act1_scale *
                             model->header.out_scale;
    float out_value = (float)out_sum * out_factor + out_bias;
    if (model->header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
        const int32_t *front_psqt = state->side_to_move == PIECE_WHITE
                                        ? frame->white_psqt
                                        : frame->black_psqt;
        out_value += (float)front_psqt[bucket] * model->header.psqt_scale;
    }
    return nn_logit_to_cp(out_value, model->header.cp_scale);
}

static int evaluate_linear_head_from_frame(const GameState *state, const NnEvalModel *model, const NnAccumulatorFrame *frame) {
    if (state == NULL || model == NULL || frame == NULL || !frame->valid) {
        return NN_CP_FALLBACK;
    }
    const uint32_t acc_dim = model->header.accumulator_dim;
    const uint32_t hidden_dim = model->header.hidden_dim;
    const int32_t *front_acc = (state->side_to_move == PIECE_WHITE) ? frame->white_acc : frame->black_acc;
    const int32_t *back_acc = (state->side_to_move == PIECE_WHITE) ? frame->black_acc : frame->white_acc;
    float hidden_f[NN_MAX_HIDDEN_DIM] = {0};
    int16_t hidden_q[NN_MAX_HIDDEN_DIM] = {0};

    float fc1_factor = model->header.acc_scale * model->header.fc1_scale;
    for (uint32_t out = 0; out < hidden_dim; ++out) {
        const int8_t *w = model->fc1_weight + ((size_t)out * (size_t)acc_dim * 2u);
        int64_t sum = dot_i8_i32(w, front_acc, acc_dim) + dot_i8_i32(w + acc_dim, back_acc, acc_dim);
        hidden_f[out] = (float)sum * fc1_factor + model->fc1_bias[out];
    }
    float hidden_scale = quantize_activation_relu(hidden_f, hidden_dim, hidden_q);

    int64_t out_sum = dot_i8_i16(model->out_weight, hidden_q, hidden_dim);
    float out_value = (float)out_sum * (hidden_scale * model->header.out_scale) + model->out_bias;
    return nn_logit_to_cp(out_value, model->header.cp_scale);
}

static int evaluate_loaded_from_frame(const GameState *state, const NnEvalModel *model, const NnAccumulatorFrame *frame) {
    if (model != NULL && header_uses_i16_accumulator(&model->header)) {
        return evaluate_i16_screlu_from_frame(state, model, frame);
    }
    if (model != NULL && header_uses_fixed_activation(&model->header)) {
        return evaluate_fixed_from_frame(state, model, frame);
    }
    if (model != NULL && model->header.version == NN_VERSION_LINEAR_HEAD_QUANT) {
        return evaluate_linear_head_from_frame(state, model, frame);
    }
    return evaluate_from_frame(state, model, frame);
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
    if ((header.version != NN_VERSION_FLOAT &&
         header.version != NN_VERSION_QUANT &&
         header.version != NN_VERSION_LINEAR_HEAD_QUANT &&
         header.version != NN_VERSION_BOTTLENECK_HEAD_QUANT &&
         header.version != NN_VERSION_LINEAR_HEAD_FIXED_ACT &&
         header.version != NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT &&
         header.version != NN_VERSION_LINEAR_HEAD_SCRELU &&
         header.version != NN_VERSION_BOTTLENECK_HEAD_SCRELU &&
         header.version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC &&
         header.version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS &&
         header.version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) ||
        (header.halfkp_dim != NN_EXPECTED_HALFKP_DIM &&
         header.halfkp_dim != NN_EXPECTED_HALFKP_HM_DIM &&
         header.halfkp_dim != NN_EXPECTED_HALFKA_HM_DIM) ||
        header.accumulator_dim == 0 || header.accumulator_dim > NN_MAX_ACC_DIM ||
        header.hidden_dim == 0 || header.hidden_dim > NN_MAX_HIDDEN_DIM ||
        header.dummy_index != header.halfkp_dim) {
        fclose(fp);
        return false;
    }
    if ((header.version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
         header.version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
         header.version == NN_VERSION_BOTTLENECK_HEAD_SCRELU) &&
        (header.bottleneck_dim == 0 || header.bottleneck_dim > NN_MAX_ACC_DIM)) {
        fclose(fp);
        return false;
    }
    if ((header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
         header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) &&
        (header.num_buckets == 0 || header.num_buckets > NN_MAX_OUTPUT_BUCKETS)) {
        fclose(fp);
        return false;
    }
    if (header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT &&
        header.psqt_scale <= 1e-12f) {
        fclose(fp);
        return false;
    }
    if (header_uses_fixed_activation(&header) &&
        (header.act0_scale <= 1e-12f || header.act1_scale <= 1e-12f || header.act2_scale <= 1e-12f)) {
        fclose(fp);
        return false;
    }

    size_t acc_rows = (size_t)header.halfkp_dim + 1u;
    size_t acc_count = acc_rows * (size_t)header.accumulator_dim;
    size_t buckets = (size_t)header_output_buckets(&header);
    size_t fc1_out = header_fc1_out_dim(&header);
    size_t fc1_count = buckets * fc1_out * ((size_t)header.accumulator_dim * 2u);
    size_t fc2_count = (size_t)header.hidden_dim * fc1_out;

    if (header.version == NN_VERSION_QUANT ||
        header.version == NN_VERSION_LINEAR_HEAD_QUANT ||
        header.version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
        header.version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
        header.version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU ||
        header.version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
        if (!allocate_quantized_model(&model, &header)) {
            fclose(fp);
            nn_eval_free_model(&model);
            return false;
        }
        model.header = header;
        model.kind = NN_MODEL_KIND_QUANT;
        ok = read_exact(fp, model.acc_weight, acc_count * sizeof(int16_t)) &&
             read_exact(fp, model.fc1_weight, fc1_count * sizeof(int8_t)) &&
             read_exact(fp, model.fc1_bias, buckets * fc1_out * sizeof(float));
        if (ok && header_uses_fc2(&header)) {
            ok = read_exact(fp, model.fc2_weight, fc2_count * sizeof(int8_t)) &&
                 read_exact(fp, model.fc2_bias, (size_t)header.hidden_dim * sizeof(float));
        }
        ok = ok &&
             read_exact(fp, model.out_weight, buckets * (size_t)header.hidden_dim * sizeof(int8_t));
        if (ok) {
            if (header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
                header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
                ok = read_exact(fp, model.out_bias_buckets, buckets * sizeof(float));
            } else {
                ok = read_exact(fp, &model.out_bias, sizeof(float));
            }
        }
        if (ok && header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT) {
            ok = read_exact(fp, model.psqt_weight,
                            acc_rows * buckets * sizeof(int16_t));
        }
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
    if (header_uses_i16_accumulator(&model.header)) {
        model.acc_act_shift = 16u;
        double factor = (double)model.header.acc_scale / (double)model.header.act0_scale;
        model.acc_act_multiplier = (int32_t)llround(factor * (double)(1u << model.acc_act_shift));
        if (model.acc_act_multiplier < 1) {
            model.acc_act_multiplier = 1;
        }
    }
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
    return evaluate_loaded_from_frame(state, &g_nn_model, &frame);
}

bool nn_eval_build_frame(const GameState *state, NnAccumulatorFrame *frame) {
    if (!g_nn_model.loaded || g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return false;
    }
    return rebuild_frame(state, frame);
}

static bool update_frame_i16(const GameState *state,
                             const UndoRecord *undo,
                             const NnAccumulatorFrame *parent,
                             NnAccumulatorFrame *frame) {
    int mover = state->side_to_move ^ 1;
    int from = move_from(undo->move);
    int to = move_to(undo->move);
    int piece = move_piece(undo->move);
    int placed_piece = move_has_flag(undo->move, MOVE_FLAG_PROMOTION) ? move_promo(undo->move) : piece;
    const NnEvalModel *model = &g_nn_model;
    const size_t acc_bytes = (size_t)model->header.accumulator_dim * sizeof(frame->white_acc16[0]);

    frame->valid = parent->valid;
    frame->key = parent->key;
    frame->non_king_piece_count = parent->non_king_piece_count;
    memcpy(frame->white_acc16, parent->white_acc16, acc_bytes);
    memcpy(frame->black_acc16, parent->black_acc16, acc_bytes);
    memcpy(frame->white_psqt, parent->white_psqt, sizeof(frame->white_psqt));
    memcpy(frame->black_psqt, parent->black_psqt, sizeof(frame->black_psqt));

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
            accumulate_perspective_i16(state, model, PIECE_WHITE, frame->white_acc16, NULL);
            accumulate_psqt_perspective(state, model, PIECE_WHITE, frame->white_psqt);
            if (model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM &&
                (!update_piece_feature_i16(frame->black_acc16, model, PIECE_BLACK,
                                           black_king_sq, mover, PIECE_KING, from, -1) ||
                 !update_piece_feature_i16(frame->black_acc16, model, PIECE_BLACK,
                                           black_king_sq, mover, PIECE_KING, to, 1) ||
                 !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK,
                                    black_king_sq, mover, PIECE_KING, from, -1) ||
                 !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK,
                                    black_king_sq, mover, PIECE_KING, to, 1))) {
                return rebuild_frame(state, frame);
            }
        } else {
            accumulate_perspective_i16(state, model, PIECE_BLACK, frame->black_acc16, NULL);
            accumulate_psqt_perspective(state, model, PIECE_BLACK, frame->black_psqt);
            if (model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM &&
                (!update_piece_feature_i16(frame->white_acc16, model, PIECE_WHITE,
                                           white_king_sq, mover, PIECE_KING, from, -1) ||
                 !update_piece_feature_i16(frame->white_acc16, model, PIECE_WHITE,
                                           white_king_sq, mover, PIECE_KING, to, 1) ||
                 !update_piece_psqt(frame->white_psqt, model, PIECE_WHITE,
                                    white_king_sq, mover, PIECE_KING, from, -1) ||
                 !update_piece_psqt(frame->white_psqt, model, PIECE_WHITE,
                                    white_king_sq, mover, PIECE_KING, to, 1))) {
                return rebuild_frame(state, frame);
            }
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
        update_dummy_i16(frame->white_acc16, model, -1);
        update_dummy_i16(frame->black_acc16, model, -1);
    }
    if (!update_piece_feature_i16(frame->white_acc16, model, PIECE_WHITE, white_king_sq, mover, piece, from, -1) ||
        !update_piece_feature_i16(frame->black_acc16, model, PIECE_BLACK, black_king_sq, mover, piece, from, -1) ||
        !update_piece_feature_i16(frame->white_acc16, model, PIECE_WHITE, white_king_sq, mover, placed_piece, to, 1) ||
        !update_piece_feature_i16(frame->black_acc16, model, PIECE_BLACK, black_king_sq, mover, placed_piece, to, 1)) {
        return rebuild_frame(state, frame);
    }
    if (!update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                           mover, piece, from, -1) ||
        !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                           mover, piece, from, -1) ||
        !update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                           mover, placed_piece, to, 1) ||
        !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                           mover, placed_piece, to, 1)) {
        return rebuild_frame(state, frame);
    }
    if (undo->captured_piece != PIECE_NONE) {
        int victim = state->side_to_move;
        if (!update_piece_feature_i16(frame->white_acc16,
                                      model,
                                      PIECE_WHITE,
                                      white_king_sq,
                                      victim,
                                      undo->captured_piece,
                                      undo->captured_square,
                                      -1) ||
            !update_piece_feature_i16(frame->black_acc16,
                                      model,
                                      PIECE_BLACK,
                                      black_king_sq,
                                      victim,
                                      undo->captured_piece,
                                      undo->captured_square,
                                      -1)) {
            return rebuild_frame(state, frame);
        }
        if (!update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                               victim, undo->captured_piece, undo->captured_square, -1) ||
            !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                               victim, undo->captured_piece, undo->captured_square, -1)) {
            return rebuild_frame(state, frame);
        }
        if (undo->captured_piece != PIECE_KING && frame->non_king_piece_count > 0) {
            frame->non_king_piece_count -= 1;
        }
    }
    if (frame->non_king_piece_count == 0) {
        update_dummy_i16(frame->white_acc16, model, 1);
        update_dummy_i16(frame->black_acc16, model, 1);
    }
    frame->valid = true;
    frame->key = state->zobrist_hash;
    return true;
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
    if (header_uses_i16_accumulator(&g_nn_model.header)) {
        return update_frame_i16(state, undo, parent, frame);
    }

    int mover = state->side_to_move ^ 1;
    int from = move_from(undo->move);
    int to = move_to(undo->move);
    int piece = move_piece(undo->move);
    int placed_piece = move_has_flag(undo->move, MOVE_FLAG_PROMOTION) ? move_promo(undo->move) : piece;

    const NnEvalModel *model = &g_nn_model;
    const size_t acc_bytes = (size_t)model->header.accumulator_dim * sizeof(frame->white_acc[0]);
    frame->valid = parent->valid;
    frame->key = parent->key;
    frame->non_king_piece_count = parent->non_king_piece_count;
    memcpy(frame->white_acc, parent->white_acc, acc_bytes);
    memcpy(frame->black_acc, parent->black_acc, acc_bytes);
    memcpy(frame->white_psqt, parent->white_psqt, sizeof(frame->white_psqt));
    memcpy(frame->black_psqt, parent->black_psqt, sizeof(frame->black_psqt));
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
            accumulate_psqt_perspective(state, model, PIECE_WHITE, frame->white_psqt);
            if (model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM &&
                (!update_piece_feature(frame->black_acc, model, PIECE_BLACK, black_king_sq,
                                       mover, PIECE_KING, from, -1) ||
                 !update_piece_feature(frame->black_acc, model, PIECE_BLACK, black_king_sq,
                                       mover, PIECE_KING, to, 1) ||
                 !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                                    mover, PIECE_KING, from, -1) ||
                 !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                                    mover, PIECE_KING, to, 1))) {
                return rebuild_frame(state, frame);
            }
        } else {
            accumulate_perspective(state, model, PIECE_BLACK, frame->black_acc, NULL);
            accumulate_psqt_perspective(state, model, PIECE_BLACK, frame->black_psqt);
            if (model->header.halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM &&
                (!update_piece_feature(frame->white_acc, model, PIECE_WHITE, white_king_sq,
                                       mover, PIECE_KING, from, -1) ||
                 !update_piece_feature(frame->white_acc, model, PIECE_WHITE, white_king_sq,
                                       mover, PIECE_KING, to, 1) ||
                 !update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                                    mover, PIECE_KING, from, -1) ||
                 !update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                                    mover, PIECE_KING, to, 1))) {
                return rebuild_frame(state, frame);
            }
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
    if (!update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                           mover, piece, from, -1) ||
        !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                           mover, piece, from, -1) ||
        !update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                           mover, placed_piece, to, 1) ||
        !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                           mover, placed_piece, to, 1)) {
        return rebuild_frame(state, frame);
    }

    if (undo->captured_piece != PIECE_NONE) {
        int victim = state->side_to_move;
        if (!update_piece_feature(frame->white_acc, model, PIECE_WHITE, white_king_sq, victim, undo->captured_piece, undo->captured_square, -1) ||
            !update_piece_feature(frame->black_acc, model, PIECE_BLACK, black_king_sq, victim, undo->captured_piece, undo->captured_square, -1)) {
            return rebuild_frame(state, frame);
        }
        if (!update_piece_psqt(frame->white_psqt, model, PIECE_WHITE, white_king_sq,
                               victim, undo->captured_piece, undo->captured_square, -1) ||
            !update_piece_psqt(frame->black_psqt, model, PIECE_BLACK, black_king_sq,
                               victim, undo->captured_piece, undo->captured_square, -1)) {
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

bool nn_eval_copy_frame(const GameState *state,
                        const NnAccumulatorFrame *source,
                        NnAccumulatorFrame *frame) {
    if (state == NULL || source == NULL || frame == NULL ||
        !g_nn_model.loaded || g_nn_model.kind != NN_MODEL_KIND_QUANT || !source->valid) {
        return false;
    }
    frame->valid = true;
    frame->key = state->zobrist_hash;
    frame->non_king_piece_count = source->non_king_piece_count;
    memcpy(frame->white_psqt, source->white_psqt, sizeof(frame->white_psqt));
    memcpy(frame->black_psqt, source->black_psqt, sizeof(frame->black_psqt));
    if (header_uses_i16_accumulator(&g_nn_model.header)) {
        const size_t acc_bytes =
            (size_t)g_nn_model.header.accumulator_dim * sizeof(frame->white_acc16[0]);
        memcpy(frame->white_acc16, source->white_acc16, acc_bytes);
        memcpy(frame->black_acc16, source->black_acc16, acc_bytes);
    } else {
        const size_t acc_bytes =
            (size_t)g_nn_model.header.accumulator_dim * sizeof(frame->white_acc[0]);
        memcpy(frame->white_acc, source->white_acc, acc_bytes);
        memcpy(frame->black_acc, source->black_acc, acc_bytes);
    }
    return true;
}

int nn_eval_cp_stm_from_frame(const GameState *state, const NnAccumulatorFrame *frame) {
    if (!g_nn_model.loaded) {
        return NN_CP_FALLBACK;
    }
    if (g_nn_model.kind != NN_MODEL_KIND_QUANT) {
        return NN_CP_FALLBACK;
    }
    return evaluate_loaded_from_frame(state, &g_nn_model, frame);
}
