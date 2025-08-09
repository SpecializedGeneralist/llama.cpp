#include "matmul_cpu.h"
#include <math.h>
#include <stdlib.h>

static inline float fp16_to_fp32(ggml_half h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline ggml_half fp32_to_fp16(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) {
        return sign;
    } else if (exp >= 31) {
        return sign | 0x7C00;
    } else {
        return sign | ((exp & 0x1F) << 10) | (mant >> 13);
    }
}

static inline float bf16_to_fp32(ggml_bf16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; memcpy(&f, &u, sizeof(f));
    return f;
}

static inline ggml_bf16_t fp32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, sizeof(u));
    return (ggml_bf16_t)(u >> 16);
}

void ggml_cpu_fp32_to_fp32(const float * x, void * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, void * y, int64_t n) {
    ggml_half * yy = (ggml_half *)y;
    for (int64_t i = 0; i < n; ++i) {
        yy[i] = fp32_to_fp16(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, void * y, int64_t n) {
    ggml_bf16_t * yy = (ggml_bf16_t *)y;
    for (int64_t i = 0; i < n; ++i) {
        yy[i] = fp32_to_bf16(x[i]);
    }
}

void quantize_row_q4_0(const float * x, void * vy, int64_t k) {
    block_q4_0 * y = (block_q4_0 *)vy;
    int nb = k / QK4_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        float maxv = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            float v = x[i*QK4_0 + j];
            if (fabsf(v) > amax) { amax = fabsf(v); maxv = v; }
        }
        float d = maxv / -8.0f;
        float id = d ? 1.0f/d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK4_0/2; ++j) {
            float x0 = x[i*QK4_0 + j] * id;
            float x1 = x[i*QK4_0 + QK4_0/2 + j] * id;
            uint8_t xi0 = (uint8_t) (fminf(15.f, floorf(x0 + 8.5f)));
            uint8_t xi1 = (uint8_t) (fminf(15.f, floorf(x1 + 8.5f)));
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    *s = sum;
}

void ggml_vec_dot_f16(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const ggml_half * x = (const ggml_half *)vx;
    const float * y = (const float *)vy;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += fp16_to_fp32(x[i]) * y[i];
    }
    *s = sum;
}

void ggml_vec_dot_bf16(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const ggml_bf16_t * x = (const ggml_bf16_t *)vx;
    const float * y = (const float *)vy;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += bf16_to_fp32(x[i]) * y[i];
    }
    *s = sum;
}

void ggml_vec_dot_q4_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const block_q4_0 * x = (const block_q4_0 *)vx;
    const float * y = (const float *)vy;
    int nb = n / QK4_0;
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK4_0/2; ++j) {
            uint8_t q = x[i].qs[j];
            float v0 = ((q & 0x0F) - 8) * d;
            float v1 = ((q >> 4) - 8) * d;
            sum += v0 * y[i*QK4_0 + j];
            sum += v1 * y[i*QK4_0 + QK4_0/2 + j];
        }
    }
    *s = sum;
}

const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = { ggml_cpu_fp32_to_fp32, (ggml_vec_dot_t)ggml_vec_dot_f32,  GGML_TYPE_F32, 1 },
    [GGML_TYPE_F16]  = { ggml_cpu_fp32_to_fp16, ggml_vec_dot_f16,                GGML_TYPE_F32, 1 },
    [GGML_TYPE_BF16] = { ggml_cpu_fp32_to_bf16, ggml_vec_dot_bf16,               GGML_TYPE_F32, 1 },
    [GGML_TYPE_Q4_0] = { quantize_row_q4_0,     ggml_vec_dot_q4_0,               GGML_TYPE_F32, 1 },
};

static void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        struct ggml_tensor * dst,
        int64_t ir0_start, int64_t ir0_end,
        int64_t ir1_start, int64_t ir1_end) {
    const int64_t k = a->ne[0];
    for (int64_t i = ir0_start; i < ir0_end; ++i) {
        float * out = (float *)dst->data + i*b->ne[1];
        const char * ap = (const char *)a->data + i * a->nb[1];
        for (int64_t j = ir1_start; j < ir1_end; ++j) {
            const char * bp = (const char *)b->data + j * b->nb[1];
            float sum = 0.0f;
            type_traits_cpu[a->type].vec_dot(k, &sum, 0, ap, a->nb[0], bp, b->nb[0], 1);
            out[j] = sum;
        }
    }
}

void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params,
                                  struct ggml_tensor * dst) {
    const struct ggml_tensor * a = dst->src[0];
    const struct ggml_tensor * b = dst->src[1];

    const int64_t m = a->ne[1];
    const int64_t n = b->ne[1];
    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t dr = (m + nth - 1) / nth;
    const int64_t ir0_start = MIN(ith * dr, m);
    const int64_t ir0_end   = MIN(ir0_start + dr, m);

    ggml_compute_forward_mul_mat_one_chunk(a, b, dst, ir0_start, ir0_end, 0, n);
}
