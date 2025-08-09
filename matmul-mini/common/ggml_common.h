#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// basic 16-bit floating point types
typedef uint16_t ggml_half;
typedef uint16_t ggml_bf16_t;

#define QK4_0 32

typedef struct {
    ggml_half d;
    uint8_t   qs[QK4_0/2];
} block_q4_0;

enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_COUNT,
};

struct ggml_tensor {
    enum ggml_type type;
    int64_t ne[4];
    size_t  nb[4];
    void *  data;
    struct ggml_tensor * src[2];
};

static inline size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return sizeof(float);
        case GGML_TYPE_F16:  return sizeof(ggml_half);
        case GGML_TYPE_BF16: return sizeof(ggml_bf16_t);
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        default: return 0;
    }
}

static inline size_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return QK4_0;
        default: return 1;
    }
}

static inline size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    const size_t ts = ggml_type_size(type);
    const size_t bs = ggml_blck_size(type);
    return ts * (ne / bs);
}

static inline bool ggml_is_contiguous(const struct ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type);
}

#ifdef __cplusplus
}
#endif
