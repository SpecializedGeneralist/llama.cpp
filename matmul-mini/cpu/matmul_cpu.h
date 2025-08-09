#pragma once
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include "../common/ggml_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ggml_from_float_t)(const float *, void *, int64_t);
typedef void (*ggml_vec_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct ggml_threadpool {
    _Atomic int current_chunk;
};

struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    char * wdata;
    struct ggml_threadpool * threadpool;
};

struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;
    ggml_vec_dot_t vec_dot;
    enum ggml_type vec_dot_type;
    int64_t nrows;
};

extern const struct ggml_type_traits_cpu type_traits_cpu[];

void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
void ggml_cpu_fp32_to_fp32(const float * x, void * y, int64_t n);
void ggml_cpu_fp32_to_fp16(const float * x, void * y, int64_t n);
void ggml_cpu_fp32_to_bf16(const float * x, void * y, int64_t n);
void quantize_row_q4_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_f16(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
void ggml_vec_dot_q4_0(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);

void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params,
                                  struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
