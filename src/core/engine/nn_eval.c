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
#include "hce_internal.h"

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
#define NN_VERSION_STOCKFISH_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT 12u
#define NN_VERSION_STOCKFISH_THREATS_SCRELU_I16_ACC_BUCKETS_PSQT 13u
#define NN_VERSION_LINEAR_THREATS_SCRELU_I16_ACC_BUCKETS_PSQT 14u
#define NN_VERSION_LINEAR_THREATS_I8_SCRELU_I16_ACC_BUCKETS_PSQT 15u
#define NN_VERSION_LINEAR_THREATS_ALL_I8_SCRELU_I16_ACC_BUCKETS_PSQT 16u
#define NN_VERSION_LINEAR_THREATS_PER_ROW_I8_SCRELU_I16_ACC_BUCKETS_PSQT 17u
#define NN_VERSION_IS_LINEAR_BUCKETED_PSQT(v) \
    ((v) == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT || \
     (v) == NN_VERSION_LINEAR_THREATS_SCRELU_I16_ACC_BUCKETS_PSQT || \
     (v) == NN_VERSION_LINEAR_THREATS_I8_SCRELU_I16_ACC_BUCKETS_PSQT || \
     (v) == NN_VERSION_LINEAR_THREATS_ALL_I8_SCRELU_I16_ACC_BUCKETS_PSQT || \
     (v) == NN_VERSION_LINEAR_THREATS_PER_ROW_I8_SCRELU_I16_ACC_BUCKETS_PSQT)
#define NN_VERSION_IS_STOCKFISH_HEAD(v) \
    ((v) == NN_VERSION_STOCKFISH_HEAD_SCRELU_I16_ACC_BUCKETS_PSQT || \
     (v) == NN_VERSION_STOCKFISH_THREATS_SCRELU_I16_ACC_BUCKETS_PSQT)
#define NN_MAX_OUTPUT_BUCKETS 32u
#define NN_MAX_HIDDEN_DIM 128u
#define NN_EXPECTED_HALFKP_DIM (64u * 10u * 64u)
#define NN_EXPECTED_HALFKP_HM_DIM (32u * 10u * 64u)
#define NN_EXPECTED_HALFKA_HM_DIM (32u * 11u * 64u)
#define NN_FULL_THREATS_DIM 59808u
#define NN_EXPECTED_HALFKA_THREATS_HM_DIM (NN_EXPECTED_HALFKA_HM_DIM + NN_FULL_THREATS_DIM)
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
    int8_t *acc_weight_i8;
    int8_t *threat_weight;
    int8_t *fc1_weight;
    // fc1 widened to int16 at load (linear screlu head only): saves the
    // per-eval sign extension; products and sums are unchanged.
    int16_t *fc1_weight16;
    float *fc1_row_scales;
    float *fc1_bias;
    int8_t *fc2_weight;
    float *fc2_bias;
    int8_t *out_weight;
    float *out_row_scales;
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
    free(model->acc_weight_i8);
    free(model->threat_weight);
    free(model->fc1_weight);
    free(model->fc1_weight16);
    free(model->fc1_row_scales);
    free(model->fc1_bias);
    free(model->fc2_weight);
    free(model->fc2_bias);
    free(model->out_weight);
    free(model->out_row_scales);
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
        NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
        NN_VERSION_IS_STOCKFISH_HEAD(header->version)) {
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
             NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
             NN_VERSION_IS_STOCKFISH_HEAD(header->version)) &&
            (!read_exact(fp, &header->act0_scale, sizeof(float)) ||
             !read_exact(fp, &header->act1_scale, sizeof(float)) ||
             !read_exact(fp, &header->act2_scale, sizeof(float)))) {
            return false;
        }
        if ((header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
             NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
             NN_VERSION_IS_STOCKFISH_HEAD(header->version)) &&
            !read_exact(fp, &header->num_buckets, sizeof(uint32_t))) {
            return false;
        }
        if ((NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
             NN_VERSION_IS_STOCKFISH_HEAD(header->version)) &&
            !read_exact(fp, &header->psqt_scale, sizeof(float))) {
            return false;
        }
        if ((header->version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
             header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
             header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU) &&
            !read_exact(fp, &header->bottleneck_dim, sizeof(uint32_t))) {
            return false;
        }
        if (NN_VERSION_IS_STOCKFISH_HEAD(header->version) &&
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
        feature_dim == NN_EXPECTED_HALFKA_HM_DIM ||
        feature_dim == NN_EXPECTED_HALFKA_THREATS_HM_DIM) {
        int planes = (feature_dim == NN_EXPECTED_HALFKA_HM_DIM ||
                      feature_dim == NN_EXPECTED_HALFKA_THREATS_HM_DIM) ? 11 : 10;
        if ((oriented_king & 7) < 4) {
            oriented_king ^= 7;
            oriented_piece ^= 7;
        }
        int king_bucket = (oriented_king >> 3) * 4 + (oriented_king & 7) - 4;
        return (king_bucket * planes + plane) * 64 + oriented_piece;
    }
    return (oriented_king * 10 + plane) * 64 + oriented_piece;
}

static bool header_uses_halfka(const NnEvalHeader *header) {
    return header != NULL &&
           (header->halfkp_dim == NN_EXPECTED_HALFKA_HM_DIM ||
            header->halfkp_dim == NN_EXPECTED_HALFKA_THREATS_HM_DIM);
}

static bool header_uses_full_threats(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_STOCKFISH_THREATS_SCRELU_I16_ACC_BUCKETS_PSQT ||
            header->version == NN_VERSION_LINEAR_THREATS_SCRELU_I16_ACC_BUCKETS_PSQT ||
            header->version == NN_VERSION_LINEAR_THREATS_I8_SCRELU_I16_ACC_BUCKETS_PSQT ||
            header->version == NN_VERSION_LINEAR_THREATS_ALL_I8_SCRELU_I16_ACC_BUCKETS_PSQT ||
            header->version ==
                NN_VERSION_LINEAR_THREATS_PER_ROW_I8_SCRELU_I16_ACC_BUCKETS_PSQT);
}

static bool header_uses_i8_threat_weights(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_THREATS_I8_SCRELU_I16_ACC_BUCKETS_PSQT ||
            header->version ==
                NN_VERSION_LINEAR_THREATS_PER_ROW_I8_SCRELU_I16_ACC_BUCKETS_PSQT);
}

static bool header_uses_per_row_head_scales(const NnEvalHeader *header) {
    return header != NULL &&
           header->version ==
               NN_VERSION_LINEAR_THREATS_PER_ROW_I8_SCRELU_I16_ACC_BUCKETS_PSQT;
}

static bool header_uses_all_i8_acc_weights(const NnEvalHeader *header) {
    return header != NULL &&
           header->version == NN_VERSION_LINEAR_THREATS_ALL_I8_SCRELU_I16_ACC_BUCKETS_PSQT;
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
           !NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version);
}

static bool header_uses_fixed_activation(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
            header->version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
            header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
            NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
            NN_VERSION_IS_STOCKFISH_HEAD(header->version));
}

static bool header_uses_squared_clipped_relu(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_HEAD_SCRELU ||
            header->version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
            NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
            NN_VERSION_IS_STOCKFISH_HEAD(header->version));
}

static bool header_uses_i16_accumulator(const NnEvalHeader *header) {
    return header != NULL &&
           (header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
            header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
            NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
            NN_VERSION_IS_STOCKFISH_HEAD(header->version));
}

static uint32_t header_output_buckets(const NnEvalHeader *header) {
    if (header == NULL ||
        (header->version != NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS &&
         !NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) &&
         !NN_VERSION_IS_STOCKFISH_HEAD(header->version))) {
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

// Eight int32 lanes each take count/8 products of |w| <= 128, |v| <= 32768,
// so they cannot overflow for count <= 4096 (NN_MAX_TRANSFORM_DIM is 1024);
// widening once at the end gives the same sum as per-step 64-bit adds.
static int64_t dot_i8_i16_neon(const int8_t *weights, const int16_t *values, uint32_t count) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    uint32_t i = 0;
    // Four independent chains hide the multiply-accumulate latency.
    for (; i + 16u <= count; i += 16u) {
        int8x16_t w8 = vld1q_s8(weights + i);
        int16x8_t wa = vmovl_s8(vget_low_s8(w8));
        int16x8_t wb = vmovl_s8(vget_high_s8(w8));
        int16x8_t va = vld1q_s16(values + i);
        int16x8_t vb = vld1q_s16(values + i + 8u);
        acc0 = vmlal_s16(acc0, vget_low_s16(wa), vget_low_s16(va));
        acc1 = vmlal_s16(acc1, vget_high_s16(wa), vget_high_s16(va));
        acc2 = vmlal_s16(acc2, vget_low_s16(wb), vget_low_s16(vb));
        acc3 = vmlal_s16(acc3, vget_high_s16(wb), vget_high_s16(vb));
    }
    for (; i + 8u <= count; i += 8u) {
        int16x8_t w16 = vmovl_s8(vld1_s8(weights + i));
        int16x8_t v16 = vld1q_s16(values + i);
        acc0 = vmlal_s16(acc0, vget_low_s16(w16), vget_low_s16(v16));
        acc1 = vmlal_s16(acc1, vget_high_s16(w16), vget_high_s16(v16));
    }
    int64_t sum = (int64_t)vaddlvq_s32(acc0) + vaddlvq_s32(acc1) + vaddlvq_s32(acc2) + vaddlvq_s32(acc3);
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
    // Each int32 lane gets count/16 madd pairs (each |pair| <= 2^23), which
    // cannot overflow for count <= 4096; widen once at the end.
    __m256i acc32 = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; i + 16u <= count; i += 16u) {
        __m128i w8 = _mm_loadu_si128((const __m128i *)(const void *)(weights + i));
        __m256i w16 = _mm256_cvtepi8_epi16(w8);
        __m256i v16 = _mm256_loadu_si256((const __m256i *)(const void *)(values + i));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(w16, v16));
    }
    __m256i acc64 = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc32)),
                                     _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc32, 1)));
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

// Four fc1 rows against one activation vector. Per-lane int32 sums stay
// exact for count <= 4096 (|w| <= 128, |v| <= 32768), as in dot_i8_i16.
static void dot4_i16_i16(const int16_t *w0, const int16_t *w1, const int16_t *w2, const int16_t *w3,
                         const int16_t *values, uint32_t count, int64_t out[4]) {
#if defined(NN_HAS_NEON)
    int32x4_t a0 = vdupq_n_s32(0), b0 = vdupq_n_s32(0);
    int32x4_t a1 = vdupq_n_s32(0), b1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), b2 = vdupq_n_s32(0);
    int32x4_t a3 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);
    uint32_t i = 0;
    for (; i + 8u <= count; i += 8u) {
        int16x8_t v = vld1q_s16(values + i);
        int16x4_t vl = vget_low_s16(v), vh = vget_high_s16(v);
        int16x8_t x0 = vld1q_s16(w0 + i), x1 = vld1q_s16(w1 + i);
        int16x8_t x2 = vld1q_s16(w2 + i), x3 = vld1q_s16(w3 + i);
        a0 = vmlal_s16(a0, vget_low_s16(x0), vl); b0 = vmlal_s16(b0, vget_high_s16(x0), vh);
        a1 = vmlal_s16(a1, vget_low_s16(x1), vl); b1 = vmlal_s16(b1, vget_high_s16(x1), vh);
        a2 = vmlal_s16(a2, vget_low_s16(x2), vl); b2 = vmlal_s16(b2, vget_high_s16(x2), vh);
        a3 = vmlal_s16(a3, vget_low_s16(x3), vl); b3 = vmlal_s16(b3, vget_high_s16(x3), vh);
    }
    out[0] = (int64_t)vaddlvq_s32(a0) + vaddlvq_s32(b0);
    out[1] = (int64_t)vaddlvq_s32(a1) + vaddlvq_s32(b1);
    out[2] = (int64_t)vaddlvq_s32(a2) + vaddlvq_s32(b2);
    out[3] = (int64_t)vaddlvq_s32(a3) + vaddlvq_s32(b3);
#else
    uint32_t i = 0;
    out[0] = out[1] = out[2] = out[3] = 0;
#endif
    for (; i < count; ++i) {
        out[0] += (int64_t)w0[i] * values[i];
        out[1] += (int64_t)w1[i] * values[i];
        out[2] += (int64_t)w2[i] * values[i];
        out[3] += (int64_t)w3[i] * values[i];
    }
}

#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static void dot4_i16_i16_avx2(const int16_t *w0, const int16_t *w1, const int16_t *w2, const int16_t *w3,
                              const int16_t *values, uint32_t count, int64_t out[4]) {
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; i + 16u <= count; i += 16u) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(values + i));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(w0 + i)), v));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(w1 + i)), v));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(w2 + i)), v));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(w3 + i)), v));
    }
    __m256i acc[4] = {a0, a1, a2, a3};
    const int16_t *ws[4] = {w0, w1, w2, w3};
    for (int k = 0; k < 4; ++k) {
        __m256i wide = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc[k])),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc[k], 1)));
        int64_t sum = horizontal_add_s64x4_avx2(wide);
        for (uint32_t j = i; j < count; ++j) {
            sum += (int64_t)ws[k][j] * values[j];
        }
        out[k] = sum;
    }
}
#endif

static void dot4_i16(const int16_t *w0, const int16_t *w1, const int16_t *w2, const int16_t *w3,
                     const int16_t *values, uint32_t count, int64_t out[4]) {
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        dot4_i16_i16_avx2(w0, w1, w2, w3, values, count, out);
        return;
    }
#endif
    dot4_i16_i16(w0, w1, w2, w3, values, count, out);
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

static NN_MAYBE_UNUSED void add_row_i8_to_i16_scalar(int16_t *acc,
                                                      const int8_t *row,
                                                      uint32_t acc_dim,
                                                      int sign) {
    for (uint32_t i = 0; i < acc_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + sign * (int16_t)row[i]);
    }
}

#if defined(NN_HAS_NEON)
static void add_row_i8_to_i16_neon(int16_t *acc,
                                   const int8_t *row,
                                   uint32_t acc_dim,
                                   int sign) {
    uint32_t i = 0;
    for (; i + 16u <= acc_dim; i += 16u) {
        int8x16_t packed = vld1q_s8(row + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(packed));
        int16x8_t hi = vmovl_s8(vget_high_s8(packed));
        int16x8_t cur_lo = vld1q_s16(acc + i);
        int16x8_t cur_hi = vld1q_s16(acc + i + 8u);
        vst1q_s16(acc + i, sign > 0 ? vaddq_s16(cur_lo, lo) : vsubq_s16(cur_lo, lo));
        vst1q_s16(acc + i + 8u,
                  sign > 0 ? vaddq_s16(cur_hi, hi) : vsubq_s16(cur_hi, hi));
    }
    for (; i < acc_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + sign * (int16_t)row[i]);
    }
}
#endif

#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static void add_row_i8_to_i16_avx2(int16_t *acc,
                                   const int8_t *row,
                                   uint32_t acc_dim,
                                   int sign) {
    uint32_t i = 0;
    for (; i + 16u <= acc_dim; i += 16u) {
        __m128i packed = _mm_loadu_si128((const __m128i *)(const void *)(row + i));
        __m256i values = _mm256_cvtepi8_epi16(packed);
        __m256i current = _mm256_loadu_si256((const __m256i *)(const void *)(acc + i));
        current = sign > 0 ? _mm256_add_epi16(current, values)
                           : _mm256_sub_epi16(current, values);
        _mm256_storeu_si256((__m256i *)(void *)(acc + i), current);
    }
    for (; i < acc_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + sign * (int16_t)row[i]);
    }
}
#endif

static void add_row_i8_to_i16_fast(int16_t *acc,
                                   const int8_t *row,
                                   uint32_t acc_dim,
                                   int sign) {
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        add_row_i8_to_i16_avx2(acc, row, acc_dim, sign);
        return;
    }
#endif
#if defined(NN_HAS_NEON)
    add_row_i8_to_i16_neon(acc, row, acc_dim, sign);
#else
    add_row_i8_to_i16_scalar(acc, row, acc_dim, sign);
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
        NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version)) {
        return (size_t)header->hidden_dim;
    }
    if (NN_VERSION_IS_STOCKFISH_HEAD(header->version)) {
        return (size_t)header->bottleneck_dim;
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
    size_t fc2_buckets = NN_VERSION_IS_STOCKFISH_HEAD(header->version)
                             ? buckets : 1u;
    size_t fc2_count = fc2_buckets * (size_t)header->hidden_dim * fc1_out;

    model->acc_weight = header_uses_all_i8_acc_weights(header)
                            ? NULL
                            : (int16_t *)malloc(acc_count * sizeof(int16_t));
    model->acc_weight_i8 = header_uses_all_i8_acc_weights(header)
                               ? (int8_t *)malloc(acc_count * sizeof(int8_t))
                               : NULL;
    model->threat_weight = header_uses_i8_threat_weights(header)
                               ? (int8_t *)malloc((size_t)NN_FULL_THREATS_DIM *
                                                 (size_t)header->accumulator_dim)
                               : NULL;
    model->fc1_weight = (int8_t *)malloc(fc1_count * sizeof(int8_t));
    model->fc1_row_scales =
        header_uses_per_row_head_scales(header)
            ? (float *)malloc(buckets * fc1_out * sizeof(float))
            : NULL;
    model->fc1_bias = (float *)malloc(buckets * fc1_out * sizeof(float));
    if (!header_uses_fc2(header)) {
        model->fc2_weight = NULL;
        model->fc2_bias = NULL;
    } else {
        model->fc2_weight = (int8_t *)malloc(fc2_count * sizeof(int8_t));
        model->fc2_bias = (float *)malloc(fc2_buckets * (size_t)header->hidden_dim * sizeof(float));
    }
    model->out_weight = (int8_t *)malloc(buckets * (size_t)header->hidden_dim * sizeof(int8_t));
    model->out_row_scales =
        header_uses_per_row_head_scales(header)
            ? (float *)malloc(buckets * sizeof(float))
            : NULL;
    if (header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
        NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
        NN_VERSION_IS_STOCKFISH_HEAD(header->version)) {
        model->out_bias_buckets = (float *)malloc(buckets * sizeof(float));
    } else {
        model->out_bias_buckets = NULL;
    }
    model->psqt_weight = (NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
                          NN_VERSION_IS_STOCKFISH_HEAD(header->version))
                             ? (int16_t *)malloc(acc_rows * buckets * sizeof(int16_t))
                             : NULL;
    if ((!header_uses_all_i8_acc_weights(header) && model->acc_weight == NULL) ||
        (header_uses_all_i8_acc_weights(header) && model->acc_weight_i8 == NULL) ||
        model->fc1_weight == NULL || model->fc1_bias == NULL ||
        model->out_weight == NULL ||
        (header_uses_per_row_head_scales(header) &&
         (model->fc1_row_scales == NULL || model->out_row_scales == NULL)) ||
        (header_uses_i8_threat_weights(header) && model->threat_weight == NULL) ||
        ((header->version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
          NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
          NN_VERSION_IS_STOCKFISH_HEAD(header->version)) &&
         model->out_bias_buckets == NULL) ||
        ((NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header->version) ||
          NN_VERSION_IS_STOCKFISH_HEAD(header->version)) &&
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

#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
// (v * m + 2^15) >> 16 for 0 <= v <= 32767, 0 < m <= 65535, from the unsigned
// 16x16 product halves: the high half plus the rounding bit of the low half.
__attribute__((target("avx2")))
static uint32_t quantize_accumulator_i16_relu_avx2(const int16_t *src,
                                                   uint32_t count,
                                                   uint16_t multiplier16,
                                                   int16_t *dst) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max_value = _mm256_set1_epi16(NN_FIXED_ACT_QMAX);
    const __m256i mult = _mm256_set1_epi16((short)multiplier16);
    uint32_t i = 0;
    for (; i + 16u <= count; i += 16u) {
        __m256i v = _mm256_max_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(src + i)), zero);
        __m256i hi = _mm256_mulhi_epu16(v, mult);
        __m256i lo = _mm256_mullo_epi16(v, mult);
        __m256i q = _mm256_add_epi16(hi, _mm256_srli_epi16(lo, 15));
        _mm256_storeu_si256((__m256i *)(void *)(dst + i), _mm256_min_epu16(q, max_value));
    }
    return i;
}
#endif

static void quantize_accumulator_i16_relu(const int16_t *src,
                                           uint32_t count,
                                           int32_t multiplier,
                                           uint8_t shift,
                                           int16_t *dst) {
    const int64_t rounding = shift > 0 ? (INT64_C(1) << (shift - 1u)) : 0;
    uint32_t i = 0;
    // Models load with shift 16 and multipliers around 36-37k, above the old
    // signed-16 fast-path limit, so every eval used the int64 scalar loop.
    // Unsigned products keep the exact same rounding for m <= 65535.
    const bool simd_ok = shift == 16u && multiplier > 0 && multiplier <= 65535;
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (simd_ok && cpu_supports_avx2()) {
        i = quantize_accumulator_i16_relu_avx2(src, count, (uint16_t)multiplier, dst);
    }
#endif
#if defined(NN_HAS_NEON)
    if (simd_ok) {
        const int16x8_t zero = vdupq_n_s16(0);
        const uint16x8_t max_value = vdupq_n_u16(NN_FIXED_ACT_QMAX);
        const uint32x4_t round_value = vdupq_n_u32(1u << 15);
        const uint16_t multiplier16 = (uint16_t)multiplier;
        for (; i + 8u <= count; i += 8u) {
            uint16x8_t value = vreinterpretq_u16_s16(vmaxq_s16(vld1q_s16(src + i), zero));
            uint32x4_t lo = vmull_n_u16(vget_low_u16(value), multiplier16);
            uint32x4_t hi = vmull_n_u16(vget_high_u16(value), multiplier16);
            lo = vshrq_n_u32(vaddq_u32(lo, round_value), 16);
            hi = vshrq_n_u32(vaddq_u32(hi, round_value), 16);
            uint16x8_t packed = vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi));
            vst1q_s16(dst + i, vreinterpretq_s16_u16(vminq_u16(packed, max_value)));
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

static bool g_threat_tables_ready;
static uint16_t g_threat_from_offset[12][64];
static uint8_t g_threat_target_rank[12][64][64];
static uint16_t g_threat_geometry_index[12][64][64];
static uint16_t g_threat_geometry_size[12];
static uint16_t g_threat_index_base[2][2][5][2][5][2];
static uint64_t g_threat_knight_attacks[64];
static const uint32_t k_threat_piece_offset[12] = {
    0, 336, 3696, 8176, 15344, 29904, 29904, 30240, 33600, 38080, 45248, 59808,
};
static const uint8_t k_threat_valid_targets[12] = {4, 10, 8, 8, 10, 0, 4, 10, 8, 8, 10, 0};
static const int8_t k_threat_target_map[6][6] = {
    {-1, 0, -1, 1, -1, -1},
    {0, 1, 2, 3, 4, -1},
    {0, 1, 2, 3, -1, -1},
    {0, 1, 2, 3, -1, -1},
    {0, 1, 2, 3, 4, -1},
    {-1, -1, -1, -1, -1, -1},
};

static int threat_piece_type(int piece) {
    static const int8_t map[PIECE_TYPE_COUNT] = {5, 4, 2, 1, 3, 0};
    return piece >= 0 && piece < PIECE_TYPE_COUNT ? map[piece] : -1;
}

static uint64_t threat_pseudo_attacks(int type, int color, int sq) {
    if (type == 0) {
        int rank = square_rank(sq);
        return rank >= 1 && rank <= 6 ? hce_pawn_attacks(color, sq) : 0ULL;
    }
    if (type == 1) return hce_knight_attacks(sq);
    if (type == 2) return hce_bishop_attacks(sq, 0ULL);
    if (type == 3) return hce_rook_attacks(sq, 0ULL);
    if (type == 4) return hce_bishop_attacks(sq, 0ULL) | hce_rook_attacks(sq, 0ULL);
    return hce_king_attacks(sq);
}

static void init_threat_tables(void) {
    if (g_threat_tables_ready) return;
    hce_init_tables();
    for (int sq = 0; sq < 64; ++sq) {
        g_threat_knight_attacks[sq] = hce_knight_attacks(sq);
    }
    for (int attacker = 0; attacker < 12; ++attacker) {
        int type = attacker % 6;
        int color = attacker < 6 ? PIECE_WHITE : PIECE_BLACK;
        uint16_t offset = 0;
        for (int from = 0; from < 64; ++from) {
            g_threat_from_offset[attacker][from] = offset;
            uint64_t attacks = threat_pseudo_attacks(type, color, from);
            uint8_t rank = 0;
            for (int to = 0; to < 64; ++to) {
                if ((attacks & (UINT64_C(1) << to)) != 0ULL) {
                    g_threat_target_rank[attacker][from][to] = rank++;
                }
            }
            offset = (uint16_t)(offset + rank);
        }
        g_threat_geometry_size[attacker] = offset;
    }
    for (int attacker = 0; attacker < 12; ++attacker) {
        for (int from = 0; from < 64; ++from) {
            for (int to = 0; to < 64; ++to) {
                g_threat_geometry_index[attacker][from][to] =
                    (uint16_t)(g_threat_from_offset[attacker][from] +
                               g_threat_target_rank[attacker][from][to]);
            }
        }
    }
    for (int perspective = PIECE_WHITE; perspective <= PIECE_BLACK; ++perspective) {
        for (int attacker_color = PIECE_WHITE;
             attacker_color <= PIECE_BLACK;
             ++attacker_color) {
            for (int attacker_type = 0; attacker_type < 5; ++attacker_type) {
                int attacker = attacker_type + (attacker_color == perspective ? 0 : 6);
                for (int attacked_color = PIECE_WHITE;
                     attacked_color <= PIECE_BLACK;
                     ++attacked_color) {
                    int color_slot = attacked_color == perspective ? 0 : 1;
                    bool enemy = attacker_color != attacked_color;
                    for (int attacked_type = 0; attacked_type < 5; ++attacked_type) {
                        int target_map = k_threat_target_map[attacker_type][attacked_type];
                        for (int from_before_to = 0; from_before_to <= 1; ++from_before_to) {
                            bool excluded = target_map < 0 ||
                                            (from_before_to != 0 &&
                                             attacker_type == attacked_type &&
                                             (enemy || attacker_type != 0));
                            uint16_t base = UINT16_MAX;
                            if (!excluded) {
                                uint32_t value = k_threat_piece_offset[attacker]
                                               + (uint32_t)(
                                                     color_slot *
                                                         (k_threat_valid_targets[attacker] / 2) +
                                                     target_map
                                                 ) *
                                                     g_threat_geometry_size[attacker];
                                if (value < NN_FULL_THREATS_DIM) {
                                    base = (uint16_t)value;
                                }
                            }
                            g_threat_index_base[perspective][attacker_color][attacker_type]
                                               [attacked_color][attacked_type][from_before_to] =
                                base;
                        }
                    }
                }
            }
        }
    }
    g_threat_tables_ready = true;
}

static int full_threat_index(int perspective,
                             int king_sq,
                             int attacker_color,
                             int attacker_type,
                             int from,
                             int to,
                             int attacked_color,
                             int attacked_type) {
    if (attacker_type < 0 || attacked_type < 0) return -1;
    int orientation = ((king_sq & 7) < 4 ? 0 : 7) ^ (perspective == PIECE_WHITE ? 0 : 56);
    int oriented_from = from ^ orientation;
    int oriented_to = to ^ orientation;
    uint16_t base = g_threat_index_base[perspective][attacker_color][attacker_type]
                                       [attacked_color][attacked_type]
                                       [oriented_from < oriented_to];
    if (base == UINT16_MAX) {
        return -1;
    }
    int attacker = attacker_type + (attacker_color == perspective ? 0 : 6);
    uint32_t index = (uint32_t)base +
                     g_threat_geometry_index[attacker][oriented_from][oriented_to];
    return index < NN_FULL_THREATS_DIM ? (int)index : -1;
}


static uint64_t threat_target_mask(const GameState *state, int attacker_type) {
    uint64_t mask = 0ULL;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        if (attacker_type == 0) {
            mask |= state->bb[color][PIECE_KNIGHT] | state->bb[color][PIECE_ROOK];
        } else {
            mask |= state->bb[color][PIECE_PAWN] | state->bb[color][PIECE_KNIGHT]
                  | state->bb[color][PIECE_BISHOP] | state->bb[color][PIECE_ROOK];
            if (attacker_type == 1 || attacker_type == 4) {
                mask |= state->bb[color][PIECE_QUEEN];
            }
        }
    }
    return mask;
}

static inline int threat_pop_lsb(uint64_t *bb) {
    int sq = __builtin_ctzll(*bb);
    *bb &= *bb - 1;
    return sq;
}

static uint16_t collect_full_threats(const GameState *state,
                                     int perspective,
                                     uint16_t out[NN_MAX_ACTIVE_THREATS]) {
    init_threat_tables();
    int king_sq = chess_find_king_square(state, perspective);
    if (king_sq < 0) return 0;
    uint16_t count = 0;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        for (int piece = PIECE_QUEEN; piece <= PIECE_PAWN; ++piece) {
            int type = threat_piece_type(piece);
            uint64_t attackers = state->bb[color][piece];
            uint64_t targets = threat_target_mask(state, type);
            while (attackers != 0ULL) {
                int from = chess_pop_lsb(&attackers);
                uint64_t attacks;
                if (type == 0) attacks = hce_pawn_attacks(color, from);
                else if (type == 1) attacks = hce_knight_attacks(from);
                else if (type == 2) attacks = hce_bishop_attacks(from, state->occ_all);
                else if (type == 3) attacks = hce_rook_attacks(from, state->occ_all);
                else attacks = hce_bishop_attacks(from, state->occ_all)
                             | hce_rook_attacks(from, state->occ_all);
                attacks &= targets;
                while (attacks != 0ULL) {
                    int to = chess_pop_lsb(&attacks);
                    int threat = full_threat_index(
                        perspective, king_sq, color, type, from, to,
                        state->sq_color[to], threat_piece_type(state->sq_piece[to])
                    );
                    if (threat < 0) continue;
                    if (count < NN_MAX_ACTIVE_THREATS) out[count++] = (uint16_t)threat;
                }
            }
        }
    }
    return count;
}


static void add_feature_row_i16(int16_t *acc,
                                const NnEvalModel *model,
                                uint32_t index,
                                int sign) {
    if (header_uses_all_i8_acc_weights(&model->header)) {
        const int8_t *row = model->acc_weight_i8 +
                            (size_t)index * model->header.accumulator_dim;
        add_row_i8_to_i16_fast(acc, row, model->header.accumulator_dim, sign);
    } else {
        const int16_t *row = model->acc_weight +
                             (size_t)index * model->header.accumulator_dim;
        add_row_i16_fast(acc, row, model->header.accumulator_dim, sign);
    }
}

static void add_threat_row_i16(int16_t *acc,
                               const NnEvalModel *model,
                               uint16_t threat,
                               int sign) {
    if (header_uses_all_i8_acc_weights(&model->header)) {
        size_t index = NN_EXPECTED_HALFKA_HM_DIM + (uint32_t)threat;
        add_feature_row_i16(acc, model, (uint32_t)index, sign);
    } else if (header_uses_i8_threat_weights(&model->header)) {
        const int8_t *row = model->threat_weight +
                            (size_t)threat * model->header.accumulator_dim;
        add_row_i8_to_i16_fast(acc, row, model->header.accumulator_dim, sign);
    } else {
        size_t index = NN_EXPECTED_HALFKA_HM_DIM + (uint32_t)threat;
        const int16_t *row = model->acc_weight + index * model->header.accumulator_dim;
        add_row_i16_fast(acc, row, model->header.accumulator_dim, sign);
    }
}


// Per-thread bitmap for threat-set diffs (see row_batch_push_threat_diff).
static _Thread_local uint64_t g_threat_marks[(NN_FULL_THREATS_DIM + 63u) / 64u];


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
        int first_piece = header_uses_halfka(&model->header)
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
                                       uint16_t *feature_count_out);

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
        int first_piece = header_uses_halfka(&model->header)
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
    add_feature_row_i16(acc, model, (uint32_t)idx, sign);
    return true;
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
    // Threat features are updated incrementally from the move (see
    // threat_incremental_delta), so frames no longer keep threat lists.
    frame->white_threat_count = 0;
    frame->black_threat_count = 0;
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
    // Fully written below before any read; zero-filling 2.8 KB per eval
    // showed up in profiles.
    int16_t transformed[NN_MAX_TRANSFORM_DIM];
    float hidden_f[NN_MAX_HIDDEN_DIM];
    int16_t hidden_q[NN_MAX_HIDDEN_DIM];

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
    if (NN_VERSION_IS_STOCKFISH_HEAD(model->header.version)) {
        const uint32_t narrow_dim = model->header.bottleneck_dim;
        float narrow_f[NN_MAX_HIDDEN_DIM] = {0};
        int16_t narrow_q[NN_MAX_HIDDEN_DIM] = {0};
        const int8_t *fc1_weight =
            model->fc1_weight + (size_t)bucket * (size_t)narrow_dim * (size_t)acc_dim * 2u;
        const float *fc1_bias = model->fc1_bias + (size_t)bucket * (size_t)narrow_dim;
        const float fc1_factor = model->header.act0_scale * model->header.act0_scale *
                                 model->header.fc1_scale;
        for (uint32_t out = 0; out < narrow_dim; ++out) {
            const int8_t *w = fc1_weight + ((size_t)out * (size_t)acc_dim * 2u);
            int64_t sum = dot_i8_i16(w, transformed, acc_dim * 2u);
            narrow_f[out] = (float)sum * fc1_factor + fc1_bias[out];
        }
        quantize_activation_fixed_relu(
            narrow_f, narrow_dim, model->header.act1_scale, narrow_q
        );
        square_quantized_activation(narrow_q, narrow_dim);

        const int8_t *fc2_weight =
            model->fc2_weight + (size_t)bucket * (size_t)hidden_dim * (size_t)narrow_dim;
        const float *fc2_bias = model->fc2_bias + (size_t)bucket * (size_t)hidden_dim;
        const float fc2_factor = model->header.act1_scale * model->header.act1_scale *
                                 model->header.fc2_scale;
        for (uint32_t out = 0; out < hidden_dim; ++out) {
            const int8_t *w = fc2_weight + (size_t)out * (size_t)narrow_dim;
            int64_t sum = dot_i8_i16(w, narrow_q, narrow_dim);
            hidden_f[out] = (float)sum * fc2_factor + fc2_bias[out];
        }
        quantize_activation_fixed_relu(
            hidden_f, hidden_dim, model->header.act2_scale, hidden_q
        );
        square_quantized_activation(hidden_q, hidden_dim);

        const int8_t *out_weight =
            model->out_weight + (size_t)bucket * (size_t)hidden_dim;
        int64_t out_sum = dot_i8_i16(out_weight, hidden_q, hidden_dim);
        const float out_factor = model->header.act2_scale * model->header.act2_scale *
                                 model->header.out_scale;
        float out_value = (float)out_sum * out_factor + model->out_bias_buckets[bucket];
        const int32_t *front_psqt = state->side_to_move == PIECE_WHITE
                                        ? frame->white_psqt
                                        : frame->black_psqt;
        out_value += (float)front_psqt[bucket] * model->header.psqt_scale;
        return nn_logit_to_cp(out_value, model->header.cp_scale);
    }
    const int8_t *fc1_weight =
        model->fc1_weight + (size_t)bucket * (size_t)hidden_dim * (size_t)acc_dim * 2u;
    const float *fc1_bias = model->fc1_bias + (size_t)bucket * (size_t)hidden_dim;
    const int8_t *out_weight = model->out_weight + (size_t)bucket * (size_t)hidden_dim;
    const float out_bias =
        model->out_bias_buckets != NULL ? model->out_bias_buckets[bucket] : model->out_bias;

    const float fc1_activation_factor =
        model->header.act0_scale * model->header.act0_scale;
    const uint32_t row_len = acc_dim * 2u;
    const int16_t *fc1_w16 = model->fc1_weight16 != NULL
                                 ? model->fc1_weight16 + (size_t)bucket * hidden_dim * row_len
                                 : NULL;
    int64_t sums[NN_MAX_HIDDEN_DIM];
    uint32_t done = 0;
    if (fc1_w16 != NULL) {
        for (; done + 4u <= hidden_dim; done += 4u) {
            const int16_t *r = fc1_w16 + (size_t)done * row_len;
            dot4_i16(r, r + row_len, r + 2u * row_len, r + 3u * row_len, transformed, row_len, sums + done);
        }
    }
    for (; done < hidden_dim; ++done) {
        sums[done] = dot_i8_i16(fc1_weight + (size_t)done * row_len, transformed, row_len);
    }
    for (uint32_t out = 0; out < hidden_dim; ++out) {
        int64_t sum = sums[out];
        float weight_scale =
            model->fc1_row_scales != NULL
                ? model->fc1_row_scales[(size_t)bucket * hidden_dim + out]
                : model->header.fc1_scale;
        hidden_f[out] =
            (float)sum * fc1_activation_factor * weight_scale + fc1_bias[out];
    }
    quantize_activation_fixed_relu(hidden_f, hidden_dim, model->header.act1_scale, hidden_q);
    square_quantized_activation(hidden_q, hidden_dim);

    int64_t out_sum = dot_i8_i16(out_weight, hidden_q, hidden_dim);
    const float out_weight_scale =
        model->out_row_scales != NULL
            ? model->out_row_scales[bucket]
            : model->header.out_scale;
    const float out_factor =
        model->header.act1_scale * model->header.act1_scale * out_weight_scale;
    float out_value = (float)out_sum * out_factor + out_bias;
    if (NN_VERSION_IS_LINEAR_BUCKETED_PSQT(model->header.version)) {
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
         !NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header.version) &&
         !NN_VERSION_IS_STOCKFISH_HEAD(header.version)) ||
        (header.halfkp_dim != NN_EXPECTED_HALFKP_DIM &&
         header.halfkp_dim != NN_EXPECTED_HALFKP_HM_DIM &&
         header.halfkp_dim != NN_EXPECTED_HALFKA_HM_DIM &&
         header.halfkp_dim != NN_EXPECTED_HALFKA_THREATS_HM_DIM) ||
        header.accumulator_dim == 0 || header.accumulator_dim > NN_MAX_ACC_DIM ||
        header.hidden_dim == 0 || header.hidden_dim > NN_MAX_HIDDEN_DIM ||
        header.dummy_index != header.halfkp_dim) {
        fclose(fp);
        return false;
    }
    if ((header.version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
         header.version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
         header.version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
         NN_VERSION_IS_STOCKFISH_HEAD(header.version)) &&
        (header.bottleneck_dim == 0 || header.bottleneck_dim > NN_MAX_HIDDEN_DIM)) {
        fclose(fp);
        return false;
    }
    if ((header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
         NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header.version) ||
         NN_VERSION_IS_STOCKFISH_HEAD(header.version)) &&
        (header.num_buckets == 0 || header.num_buckets > NN_MAX_OUTPUT_BUCKETS)) {
        fclose(fp);
        return false;
    }
    if ((NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header.version) ||
         NN_VERSION_IS_STOCKFISH_HEAD(header.version)) &&
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
    size_t fc2_buckets = NN_VERSION_IS_STOCKFISH_HEAD(header.version)
                             ? buckets : 1u;
    size_t fc2_count = fc2_buckets * (size_t)header.hidden_dim * fc1_out;

    if (header.version == NN_VERSION_QUANT ||
        header.version == NN_VERSION_LINEAR_HEAD_QUANT ||
        header.version == NN_VERSION_BOTTLENECK_HEAD_QUANT ||
        header.version == NN_VERSION_LINEAR_HEAD_FIXED_ACT ||
        header.version == NN_VERSION_BOTTLENECK_HEAD_FIXED_ACT ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU ||
        header.version == NN_VERSION_BOTTLENECK_HEAD_SCRELU ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC ||
        header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
        NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header.version) ||
        NN_VERSION_IS_STOCKFISH_HEAD(header.version)) {
        if (!allocate_quantized_model(&model, &header)) {
            fclose(fp);
            nn_eval_free_model(&model);
            return false;
        }
        model.header = header;
        model.kind = NN_MODEL_KIND_QUANT;
        if (header_uses_all_i8_acc_weights(&header)) {
            ok = read_exact(fp, model.acc_weight_i8, acc_count * sizeof(int8_t));
        } else if (header_uses_i8_threat_weights(&header)) {
            const size_t base_count = (size_t)NN_EXPECTED_HALFKA_HM_DIM *
                                      (size_t)header.accumulator_dim;
            const size_t dummy_offset = (size_t)header.dummy_index *
                                        (size_t)header.accumulator_dim;
            ok = read_exact(fp, model.acc_weight, base_count * sizeof(int16_t)) &&
                 read_exact(fp, model.threat_weight,
                            (size_t)NN_FULL_THREATS_DIM *
                                (size_t)header.accumulator_dim * sizeof(int8_t)) &&
                 read_exact(fp, model.acc_weight + dummy_offset,
                            (size_t)header.accumulator_dim * sizeof(int16_t));
        } else {
            ok = read_exact(fp, model.acc_weight, acc_count * sizeof(int16_t));
        }
        ok = ok &&
             read_exact(fp, model.fc1_weight, fc1_count * sizeof(int8_t)) &&
             read_exact(fp, model.fc1_bias, buckets * fc1_out * sizeof(float));
        if (ok && header_uses_fc2(&header)) {
            ok = read_exact(fp, model.fc2_weight, fc2_count * sizeof(int8_t)) &&
                 read_exact(fp, model.fc2_bias,
                            fc2_buckets * (size_t)header.hidden_dim * sizeof(float));
        }
        ok = ok &&
             read_exact(fp, model.out_weight, buckets * (size_t)header.hidden_dim * sizeof(int8_t));
        if (ok) {
            if (header.version == NN_VERSION_LINEAR_HEAD_SCRELU_I16_ACC_BUCKETS ||
                NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header.version) ||
                NN_VERSION_IS_STOCKFISH_HEAD(header.version)) {
                ok = read_exact(fp, model.out_bias_buckets, buckets * sizeof(float));
            } else {
                ok = read_exact(fp, &model.out_bias, sizeof(float));
            }
        }
        if (ok && (NN_VERSION_IS_LINEAR_BUCKETED_PSQT(header.version) ||
                   NN_VERSION_IS_STOCKFISH_HEAD(header.version))) {
            ok = read_exact(fp, model.psqt_weight,
                            acc_rows * buckets * sizeof(int16_t));
        }
        if (ok && header_uses_per_row_head_scales(&header)) {
            ok = read_exact(fp,
                            model.fc1_row_scales,
                            buckets * fc1_out * sizeof(float)) &&
                 read_exact(fp,
                            model.out_row_scales,
                            buckets * sizeof(float));
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
    if (header_uses_i16_accumulator(&model.header) &&
        !NN_VERSION_IS_STOCKFISH_HEAD(model.header.version) && model.fc1_weight != NULL) {
        size_t n = (size_t)header_output_buckets(&model.header) * model.header.hidden_dim *
                   (size_t)model.header.accumulator_dim * 2u;
        model.fc1_weight16 = malloc(n * sizeof(int16_t));
        if (model.fc1_weight16 != NULL) {
            for (size_t i = 0; i < n; ++i) {
                model.fc1_weight16[i] = model.fc1_weight[i];
            }
        }
    }
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

bool nn_eval_uses_full_threats(void) {
    return g_nn_model.loaded && header_uses_full_threats(&g_nn_model.header);
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

// Batched accumulator update: collect every changed feature row first, then
// write parent + added - removed into the child in one pass instead of one
// load/store sweep over the accumulator per row. int16 adds wrap exactly, so
// the order of rows does not matter and results are bit-identical.
#define NN_ROW_BATCH_CAP 128u
typedef struct NnRowBatch {
    const int16_t *r16[NN_ROW_BATCH_CAP];
    int8_t s16[NN_ROW_BATCH_CAP];
    uint32_t n16;
    const int8_t *r8[NN_ROW_BATCH_CAP];
    int8_t s8[NN_ROW_BATCH_CAP];
    uint32_t n8;
} NnRowBatch;

static void row_batch_apply_scalar(int16_t *dst, const int16_t *src, uint32_t dim,
                                   const NnRowBatch *b, uint32_t start) {
    for (uint32_t i = start; i < dim; ++i) {
        int16_t v = src[i];
        for (uint32_t k = 0; k < b->n16; ++k) {
            v = (int16_t)(b->s16[k] > 0 ? v + b->r16[k][i] : v - b->r16[k][i]);
        }
        for (uint32_t k = 0; k < b->n8; ++k) {
            v = (int16_t)(b->s8[k] > 0 ? v + b->r8[k][i] : v - b->r8[k][i]);
        }
        dst[i] = v;
    }
}

#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static uint32_t row_batch_apply_avx2(int16_t *dst, const int16_t *src, uint32_t dim,
                                     const NnRowBatch *b) {
    uint32_t i = 0;
    for (; i + 32u <= dim; i += 32u) {
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(const void *)(src + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(const void *)(src + i + 16u));
        for (uint32_t k = 0; k < b->n16; ++k) {
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(const void *)(b->r16[k] + i));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(const void *)(b->r16[k] + i + 16u));
            if (b->s16[k] > 0) { a0 = _mm256_add_epi16(a0, r0); a1 = _mm256_add_epi16(a1, r1); }
            else { a0 = _mm256_sub_epi16(a0, r0); a1 = _mm256_sub_epi16(a1, r1); }
        }
        for (uint32_t k = 0; k < b->n8; ++k) {
            __m256i packed = _mm256_loadu_si256((const __m256i *)(const void *)(b->r8[k] + i));
            __m256i r0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(packed));
            __m256i r1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(packed, 1));
            if (b->s8[k] > 0) { a0 = _mm256_add_epi16(a0, r0); a1 = _mm256_add_epi16(a1, r1); }
            else { a0 = _mm256_sub_epi16(a0, r0); a1 = _mm256_sub_epi16(a1, r1); }
        }
        _mm256_storeu_si256((__m256i *)(void *)(dst + i), a0);
        _mm256_storeu_si256((__m256i *)(void *)(dst + i + 16u), a1);
    }
    return i;
}
#endif

static void row_batch_apply(int16_t *dst, const int16_t *src, uint32_t dim, const NnRowBatch *b) {
    uint32_t i = 0;
#if defined(NN_HAS_X86_64) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx2()) {
        i = row_batch_apply_avx2(dst, src, dim, b);
    }
#endif
#if defined(NN_HAS_NEON)
    for (; i + 32u <= dim; i += 32u) {
        int16x8_t a0 = vld1q_s16(src + i), a1 = vld1q_s16(src + i + 8u);
        int16x8_t a2 = vld1q_s16(src + i + 16u), a3 = vld1q_s16(src + i + 24u);
        for (uint32_t k = 0; k < b->n16; ++k) {
            const int16_t *r = b->r16[k] + i;
            if (b->s16[k] > 0) {
                a0 = vaddq_s16(a0, vld1q_s16(r)); a1 = vaddq_s16(a1, vld1q_s16(r + 8u));
                a2 = vaddq_s16(a2, vld1q_s16(r + 16u)); a3 = vaddq_s16(a3, vld1q_s16(r + 24u));
            } else {
                a0 = vsubq_s16(a0, vld1q_s16(r)); a1 = vsubq_s16(a1, vld1q_s16(r + 8u));
                a2 = vsubq_s16(a2, vld1q_s16(r + 16u)); a3 = vsubq_s16(a3, vld1q_s16(r + 24u));
            }
        }
        for (uint32_t k = 0; k < b->n8; ++k) {
            int8x16_t p0 = vld1q_s8(b->r8[k] + i), p1 = vld1q_s8(b->r8[k] + i + 16u);
            int16x8_t r0 = vmovl_s8(vget_low_s8(p0)), r1 = vmovl_s8(vget_high_s8(p0));
            int16x8_t r2 = vmovl_s8(vget_low_s8(p1)), r3 = vmovl_s8(vget_high_s8(p1));
            if (b->s8[k] > 0) {
                a0 = vaddq_s16(a0, r0); a1 = vaddq_s16(a1, r1); a2 = vaddq_s16(a2, r2); a3 = vaddq_s16(a3, r3);
            } else {
                a0 = vsubq_s16(a0, r0); a1 = vsubq_s16(a1, r1); a2 = vsubq_s16(a2, r2); a3 = vsubq_s16(a3, r3);
            }
        }
        vst1q_s16(dst + i, a0); vst1q_s16(dst + i + 8u, a1);
        vst1q_s16(dst + i + 16u, a2); vst1q_s16(dst + i + 24u, a3);
    }
#endif
    row_batch_apply_scalar(dst, src, dim, b, i);
}

// Push a row; when the batch is full, flush it into dst (later flushes read
// dst itself as the source).
static void row_batch_flush(NnRowBatch *b, int16_t *dst, const int16_t **src, uint32_t dim) {
    row_batch_apply(dst, *src, dim, b);
    *src = dst;
    b->n16 = 0;
    b->n8 = 0;
}

static void row_batch_push_feature(NnRowBatch *b, const NnEvalModel *model, uint32_t index, int sign,
                                   int16_t *dst, const int16_t **src) {
    const uint32_t dim = model->header.accumulator_dim;
    if (b->n16 >= NN_ROW_BATCH_CAP || b->n8 >= NN_ROW_BATCH_CAP) {
        row_batch_flush(b, dst, src, dim);
    }
    if (header_uses_all_i8_acc_weights(&model->header)) {
        b->r8[b->n8] = model->acc_weight_i8 + (size_t)index * dim;
        b->s8[b->n8++] = (int8_t)(sign > 0 ? 1 : -1);
    } else {
        b->r16[b->n16] = model->acc_weight + (size_t)index * dim;
        b->s16[b->n16++] = (int8_t)(sign > 0 ? 1 : -1);
    }
}

static void row_batch_push_threat(NnRowBatch *b, const NnEvalModel *model, uint16_t threat, int sign,
                                  int16_t *dst, const int16_t **src) {
    const uint32_t dim = model->header.accumulator_dim;
    if (header_uses_all_i8_acc_weights(&model->header) ||
        !header_uses_i8_threat_weights(&model->header)) {
        row_batch_push_feature(b, model, NN_EXPECTED_HALFKA_HM_DIM + (uint32_t)threat, sign, dst, src);
        return;
    }
    if (b->n8 >= NN_ROW_BATCH_CAP) {
        row_batch_flush(b, dst, src, dim);
    }
    b->r8[b->n8] = model->threat_weight + (size_t)threat * dim;
    b->s8[b->n8++] = (int8_t)(sign > 0 ? 1 : -1);
}

// Same set logic as apply_full_threat_diff_i16, collecting rows instead.
static void row_batch_push_threat_diff(NnRowBatch *b, const NnEvalModel *model,
                                       const uint16_t *old_active, uint16_t old_count,
                                       const uint16_t *new_active, uint16_t new_count,
                                       int16_t *dst, const int16_t **src) {
    // One TLS lookup per call (Mach-O resolves each access via _tlv_get_addr).
    uint64_t *marks = g_threat_marks;
    for (uint16_t i = 0; i < old_count; ++i) {
        uint16_t t = old_active[i];
        marks[t >> 6] |= UINT64_C(1) << (t & 63u);
    }
    for (uint16_t i = 0; i < new_count; ++i) {
        uint16_t t = new_active[i];
        uint64_t bit = UINT64_C(1) << (t & 63u);
        if (marks[t >> 6] & bit) {
            marks[t >> 6] &= ~bit;
        } else {
            row_batch_push_threat(b, model, t, 1, dst, src);
        }
    }
    for (uint16_t i = 0; i < old_count; ++i) {
        uint16_t t = old_active[i];
        uint64_t bit = UINT64_C(1) << (t & 63u);
        if (marks[t >> 6] & bit) {
            marks[t >> 6] &= ~bit;
            row_batch_push_threat(b, model, t, -1, dst, src);
        }
    }
}

// Feature index for a piece, or -1 when the plane is absent; -2 when the
// index is out of range (the caller must rebuild, as update_piece_feature_i16).
static int piece_feature_index(const NnEvalModel *model, int perspective, int king_sq,
                               int color, int piece, int sq) {
    int plane = piece_plane(piece, color, perspective);
    if (plane < 0) {
        return -1;
    }
    int idx = halfkp_index(king_sq, plane, sq, perspective, model->header.halfkp_dim);
    if (idx < 0 || (uint32_t)idx > model->header.dummy_index) {
        return -2;
    }
    return idx;
}

static void accumulate_perspective_i16(const GameState *state,
                                       const NnEvalModel *model,
                                       int perspective,
                                       int16_t *out_acc,
                                       uint16_t *feature_count_out) {
    // Full refresh through a row batch: every row is read once and the
    // accumulator written once, instead of one read-modify-write per row.
    const uint32_t acc_dim = model->header.accumulator_dim;
    static const int16_t k_zero_acc[NN_MAX_ACC_DIM];
    static _Thread_local NnRowBatch batch;
    batch.n16 = batch.n8 = 0;
    const int16_t *src = k_zero_acc;

    int king_sq = chess_find_king_square(state, perspective);
    if (king_sq < 0 || king_sq >= 64) {
        memset(out_acc, 0, (size_t)acc_dim * sizeof(out_acc[0]));
        return;
    }

    bool had_feature = false;
    uint16_t feature_count = 0;
    for (int color = PIECE_WHITE; color <= PIECE_BLACK; ++color) {
        int first_piece = header_uses_halfka(&model->header)
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
                row_batch_push_feature(&batch, model, (uint32_t)idx, 1, out_acc, &src);
                had_feature = true;
                feature_count += piece != PIECE_KING;
            }
        }
    }

    if (header_uses_full_threats(&model->header)) {
        uint16_t active[NN_MAX_ACTIVE_THREATS];
        uint16_t count = collect_full_threats(state, perspective, active);
        for (uint16_t i = 0; i < count; ++i) {
            row_batch_push_threat(&batch, model, active[i], 1, out_acc, &src);
        }
    }

    if (!had_feature) {
        row_batch_push_feature(&batch, model, model->header.dummy_index, 1, out_acc, &src);
    }
    row_batch_apply(out_acc, src, acc_dim, &batch);
    if (feature_count_out != NULL) {
        *feature_count_out = feature_count;
    }
}

// ---- Incremental Full Threats ------------------------------------------
// After a move only threats whose attacker is the moved piece, the captured
// piece, or a piece attacking the from/to/captured square (before or after
// the move, which covers every slider whose ray opens or closes) can change.
// Recompute just those attackers' threats in the old and new positions and
// diff them. This replaces collecting and diffing every threat on the board.
typedef struct ThreatPosition {
    uint64_t bb[PIECE_COLOR_COUNT][PIECE_TYPE_COUNT];
    uint64_t occ;
    const int8_t *sq_piece;
    const int8_t *sq_color;
    int override_sq[3];
    int8_t override_piece[3];
    int8_t override_color[3];
    int override_count;
} ThreatPosition;

static inline void threat_position_piece(const ThreatPosition *pos, int sq, int *piece, int *color) {
    for (int i = 0; i < pos->override_count; ++i) {
        if (pos->override_sq[i] == sq) {
            *piece = pos->override_piece[i];
            *color = pos->override_color[i];
            return;
        }
    }
    *piece = pos->sq_piece[sq];
    *color = pos->sq_color[sq];
}

static uint64_t threat_attackers_to(const ThreatPosition *pos, int sq) {
    uint64_t a = 0;
    a |= pos->bb[PIECE_WHITE][PIECE_PAWN] & hce_pawn_attacks(PIECE_BLACK, sq);
    a |= pos->bb[PIECE_BLACK][PIECE_PAWN] & hce_pawn_attacks(PIECE_WHITE, sq);
    a |= (pos->bb[PIECE_WHITE][PIECE_KNIGHT] | pos->bb[PIECE_BLACK][PIECE_KNIGHT]) &
         g_threat_knight_attacks[sq];
    uint64_t diag = pos->bb[PIECE_WHITE][PIECE_BISHOP] | pos->bb[PIECE_BLACK][PIECE_BISHOP] |
                    pos->bb[PIECE_WHITE][PIECE_QUEEN] | pos->bb[PIECE_BLACK][PIECE_QUEEN];
    uint64_t line = pos->bb[PIECE_WHITE][PIECE_ROOK] | pos->bb[PIECE_BLACK][PIECE_ROOK] |
                    pos->bb[PIECE_WHITE][PIECE_QUEEN] | pos->bb[PIECE_BLACK][PIECE_QUEEN];
    a |= diag & hce_bishop_attacks(sq, pos->occ);
    a |= line & hce_rook_attacks(sq, pos->occ);
    return a;
}

// Emit threat indices of the attackers on `attackers` in `pos` for the
// perspectives in `persp_mask` (bit 0 white, bit 1 black).
static void threat_emit(const ThreatPosition *pos, uint64_t attackers, int persp_mask,
                        int white_orientation, int black_orientation,
                        uint16_t *white, uint16_t *white_count,
                        uint16_t *black, uint16_t *black_count) {
    uint64_t pawns = pos->bb[PIECE_WHITE][PIECE_PAWN] | pos->bb[PIECE_BLACK][PIECE_PAWN];
    uint64_t knights = pos->bb[PIECE_WHITE][PIECE_KNIGHT] | pos->bb[PIECE_BLACK][PIECE_KNIGHT];
    uint64_t bishops = pos->bb[PIECE_WHITE][PIECE_BISHOP] | pos->bb[PIECE_BLACK][PIECE_BISHOP];
    uint64_t rooks = pos->bb[PIECE_WHITE][PIECE_ROOK] | pos->bb[PIECE_BLACK][PIECE_ROOK];
    uint64_t queens = pos->bb[PIECE_WHITE][PIECE_QUEEN] | pos->bb[PIECE_BLACK][PIECE_QUEEN];
    uint64_t minor_slider_targets = pawns | knights | bishops | rooks;
    const uint64_t target_masks[5] = {
        knights | rooks,
        minor_slider_targets | queens,
        minor_slider_targets,
        minor_slider_targets,
        minor_slider_targets | queens,
    };
    while (attackers != 0ULL) {
        int from = threat_pop_lsb(&attackers);
        int piece, color;
        threat_position_piece(pos, from, &piece, &color);
        int type = threat_piece_type(piece);
        if (type < 0 || type > 4) {
            continue;
        }
        uint64_t attacks;
        if (type == 0) attacks = hce_pawn_attacks(color, from);
        else if (type == 1) attacks = g_threat_knight_attacks[from];
        else if (type == 2) attacks = hce_bishop_attacks(from, pos->occ);
        else if (type == 3) attacks = hce_rook_attacks(from, pos->occ);
        else attacks = hce_bishop_attacks(from, pos->occ) | hce_rook_attacks(from, pos->occ);
        attacks &= target_masks[type];
        while (attacks != 0ULL) {
            int to = threat_pop_lsb(&attacks);
            int tpiece, tcolor;
            threat_position_piece(pos, to, &tpiece, &tcolor);
            int attacked_type = threat_piece_type(tpiece);
            if (attacked_type < 0) {
                continue;
            }
            if (persp_mask & 1) {
                int wf = from ^ white_orientation, wt = to ^ white_orientation;
                uint16_t base = g_threat_index_base[PIECE_WHITE][color][type][tcolor][attacked_type][wf < wt];
                if (base != UINT16_MAX && *white_count < NN_MAX_ACTIVE_THREATS) {
                    int attacker = type + (color == PIECE_WHITE ? 0 : 6);
                    white[(*white_count)++] = (uint16_t)(base + g_threat_geometry_index[attacker][wf][wt]);
                }
            }
            if (persp_mask & 2) {
                int bf = from ^ black_orientation, bt = to ^ black_orientation;
                uint16_t base = g_threat_index_base[PIECE_BLACK][color][type][tcolor][attacked_type][bf < bt];
                if (base != UINT16_MAX && *black_count < NN_MAX_ACTIVE_THREATS) {
                    int attacker = type + (color == PIECE_BLACK ? 0 : 6);
                    black[(*black_count)++] = (uint16_t)(base + g_threat_geometry_index[attacker][bf][bt]);
                }
            }
        }
    }
}

// Push the threat-feature delta caused by the move in `undo` (state is the
// position after it) into the row batches of the perspectives in persp_mask.
static void threat_incremental_delta(const GameState *state, const UndoRecord *undo,
                                     int persp_mask, int white_king_sq, int black_king_sq,
                                     NnRowBatch *wb, int16_t *wdst, const int16_t **wsrc,
                                     NnRowBatch *bb, int16_t *bdst, const int16_t **bsrc) {
    init_threat_tables();
    const NnEvalModel *model = &g_nn_model;
    int mover = state->side_to_move ^ 1;
    int from = move_from(undo->move);
    int to = move_to(undo->move);
    int piece = move_piece(undo->move);
    int placed = move_has_flag(undo->move, MOVE_FLAG_PROMOTION) ? move_promo(undo->move) : piece;
    bool captured = undo->captured_piece != PIECE_NONE;
    int csq = captured ? undo->captured_square : -1;
    int victim = state->side_to_move;

    ThreatPosition now;
    memcpy(now.bb, state->bb, sizeof(now.bb));
    now.occ = state->occ_all;
    now.sq_piece = state->sq_piece;
    now.sq_color = state->sq_color;
    now.override_count = 0;

    ThreatPosition before = now;
    before.bb[mover][placed] &= ~(1ULL << to);
    before.bb[mover][piece] |= 1ULL << from;
    before.override_count = 0;
    before.override_sq[before.override_count] = from;
    before.override_piece[before.override_count] = (int8_t)piece;
    before.override_color[before.override_count++] = (int8_t)mover;
    if (captured) {
        before.bb[victim][undo->captured_piece] |= 1ULL << csq;
        before.override_sq[before.override_count] = csq;
        before.override_piece[before.override_count] = (int8_t)undo->captured_piece;
        before.override_color[before.override_count++] = (int8_t)victim;
    }
    if (!captured || csq != to) {
        before.override_sq[before.override_count] = to;
        before.override_piece[before.override_count] = (int8_t)PIECE_NONE;
        before.override_color[before.override_count++] = (int8_t)PIECE_COLOR_COUNT;
    }
    before.occ = (now.occ & ~(1ULL << to)) | (1ULL << from);
    if (captured) {
        before.occ |= 1ULL << csq;
    }

    uint64_t key = (1ULL << from) | (1ULL << to) | (captured ? (1ULL << csq) : 0ULL);
    uint64_t a_old = 0, a_new = 0;
    uint64_t k = key;
    while (k != 0ULL) {
        int sq = threat_pop_lsb(&k);
        a_old |= threat_attackers_to(&before, sq);
        a_new |= threat_attackers_to(&now, sq);
    }
    uint64_t unaffected = (a_old | a_new) & ~key;
    uint64_t old_set = unaffected | (1ULL << from) | (captured ? (1ULL << csq) : 0ULL);
    uint64_t new_set = unaffected | (1ULL << to);

    int white_orientation = (white_king_sq & 7) < 4 ? 0 : 7;
    int black_orientation = ((black_king_sq & 7) < 4 ? 0 : 7) ^ 56;
    uint16_t wold[NN_MAX_ACTIVE_THREATS], wnew[NN_MAX_ACTIVE_THREATS];
    uint16_t bold[NN_MAX_ACTIVE_THREATS], bnew[NN_MAX_ACTIVE_THREATS];
    uint16_t nwo = 0, nwn = 0, nbo = 0, nbn = 0;
    threat_emit(&before, old_set, persp_mask, white_orientation, black_orientation,
                wold, &nwo, bold, &nbo);
    threat_emit(&now, new_set, persp_mask, white_orientation, black_orientation,
                wnew, &nwn, bnew, &nbn);
    if (persp_mask & 1) {
        row_batch_push_threat_diff(wb, model, wold, nwo, wnew, nwn, wdst, wsrc);
    }
    if (persp_mask & 2) {
        row_batch_push_threat_diff(bb, model, bold, nbo, bnew, nbn, bdst, bsrc);
    }
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
    memcpy(frame->white_psqt, parent->white_psqt, sizeof(frame->white_psqt));
    memcpy(frame->black_psqt, parent->black_psqt, sizeof(frame->black_psqt));
    frame->white_threat_count = parent->white_threat_count;
    frame->black_threat_count = parent->black_threat_count;

    int white_king_sq = chess_find_king_square(state, PIECE_WHITE);
    int black_king_sq = chess_find_king_square(state, PIECE_BLACK);
    if (white_king_sq < 0 || black_king_sq < 0) {
        return rebuild_frame(state, frame);
    }
    if (piece == PIECE_KING) {
        if (move_has_flag(undo->move, MOVE_FLAG_CASTLE) ||
            undo->captured_piece != PIECE_NONE) {
            return rebuild_frame(state, frame);
        }
        memcpy(frame->white_acc16, parent->white_acc16, acc_bytes);
        memcpy(frame->black_acc16, parent->black_acc16, acc_bytes);
        if (mover == PIECE_WHITE) {
            accumulate_perspective_i16(state, model, PIECE_WHITE, frame->white_acc16, NULL);
            accumulate_psqt_perspective(state, model, PIECE_WHITE, frame->white_psqt);
            if (header_uses_full_threats(&model->header)) {
                static _Thread_local NnRowBatch kb;
                kb.n16 = kb.n8 = 0;
                const int16_t *ksrc = frame->black_acc16;
                threat_incremental_delta(state, undo, 2, white_king_sq, black_king_sq,
                                         NULL, NULL, NULL, &kb, frame->black_acc16, &ksrc);
                row_batch_apply(frame->black_acc16, ksrc, model->header.accumulator_dim, &kb);
            }
            if (header_uses_halfka(&model->header) &&
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
            if (header_uses_full_threats(&model->header)) {
                static _Thread_local NnRowBatch kb;
                kb.n16 = kb.n8 = 0;
                const int16_t *ksrc = frame->white_acc16;
                threat_incremental_delta(state, undo, 1, white_king_sq, black_king_sq,
                                         &kb, frame->white_acc16, &ksrc, NULL, NULL, NULL);
                row_batch_apply(frame->white_acc16, ksrc, model->header.accumulator_dim, &kb);
            }
            if (header_uses_halfka(&model->header) &&
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
        memcpy(frame->white_acc16, parent->white_acc16, acc_bytes);
        memcpy(frame->black_acc16, parent->black_acc16, acc_bytes);
        frame->key = state->zobrist_hash;
        return true;
    }
    // Collect feature rows per perspective, then apply once (see NnRowBatch).
    const uint32_t dim = model->header.accumulator_dim;
    static _Thread_local NnRowBatch wb, bb;
    wb.n16 = wb.n8 = bb.n16 = bb.n8 = 0;
    const int16_t *wsrc = parent->white_acc16;
    const int16_t *bsrc = parent->black_acc16;
    int16_t *wdst = frame->white_acc16;
    int16_t *bdst = frame->black_acc16;
    int victim = state->side_to_move;
    bool captured = undo->captured_piece != PIECE_NONE;
    int w_from = piece_feature_index(model, PIECE_WHITE, white_king_sq, mover, piece, from);
    int b_from = piece_feature_index(model, PIECE_BLACK, black_king_sq, mover, piece, from);
    int w_to = piece_feature_index(model, PIECE_WHITE, white_king_sq, mover, placed_piece, to);
    int b_to = piece_feature_index(model, PIECE_BLACK, black_king_sq, mover, placed_piece, to);
    int w_cap = captured ? piece_feature_index(model, PIECE_WHITE, white_king_sq, victim,
                                               undo->captured_piece, undo->captured_square) : -1;
    int b_cap = captured ? piece_feature_index(model, PIECE_BLACK, black_king_sq, victim,
                                               undo->captured_piece, undo->captured_square) : -1;
    if (w_from == -2 || b_from == -2 || w_to == -2 || b_to == -2 || w_cap == -2 || b_cap == -2) {
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
    if (captured) {
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
    bool dummy_before = parent->non_king_piece_count == 0;
    bool dummy_after = frame->non_king_piece_count == 0;
    if (dummy_before != dummy_after) {
        int sign = dummy_after ? 1 : -1;
        row_batch_push_feature(&wb, model, model->header.dummy_index, sign, wdst, &wsrc);
        row_batch_push_feature(&bb, model, model->header.dummy_index, sign, bdst, &bsrc);
    }
    if (w_from >= 0) row_batch_push_feature(&wb, model, (uint32_t)w_from, -1, wdst, &wsrc);
    if (b_from >= 0) row_batch_push_feature(&bb, model, (uint32_t)b_from, -1, bdst, &bsrc);
    if (w_to >= 0) row_batch_push_feature(&wb, model, (uint32_t)w_to, 1, wdst, &wsrc);
    if (b_to >= 0) row_batch_push_feature(&bb, model, (uint32_t)b_to, 1, bdst, &bsrc);
    if (w_cap >= 0) row_batch_push_feature(&wb, model, (uint32_t)w_cap, -1, wdst, &wsrc);
    if (b_cap >= 0) row_batch_push_feature(&bb, model, (uint32_t)b_cap, -1, bdst, &bsrc);
    if (header_uses_full_threats(&model->header)) {
        threat_incremental_delta(state, undo, 3, white_king_sq, black_king_sq,
                                 &wb, wdst, &wsrc, &bb, bdst, &bsrc);
    }
    row_batch_apply(wdst, wsrc, dim, &wb);
    row_batch_apply(bdst, bsrc, dim, &bb);
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
        if (move_has_flag(undo->move, MOVE_FLAG_CASTLE) ||
            undo->captured_piece != PIECE_NONE) {
            return rebuild_frame(state, frame);
        }
        if (mover == PIECE_WHITE) {
            accumulate_perspective(state, model, PIECE_WHITE, frame->white_acc, NULL);
            accumulate_psqt_perspective(state, model, PIECE_WHITE, frame->white_psqt);
            if (header_uses_halfka(&model->header) &&
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
            if (header_uses_halfka(&model->header) &&
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
    frame->white_threat_count = source->white_threat_count;
    frame->black_threat_count = source->black_threat_count;
    memcpy(frame->white_threats, source->white_threats,
           (size_t)source->white_threat_count * sizeof(frame->white_threats[0]));
    memcpy(frame->black_threats, source->black_threats,
           (size_t)source->black_threat_count * sizeof(frame->black_threats[0]));
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
