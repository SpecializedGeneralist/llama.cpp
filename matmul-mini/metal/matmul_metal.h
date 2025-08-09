#pragma once
#include "../common/ggml_common.h"
#include "metal_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_context;

struct ggml_metal_context * ggml_metal_init(void);
void ggml_metal_free(struct ggml_metal_context * ctx);

void ggml_metal_mul_mm_q4_0_f32(struct ggml_metal_context * ctx,
                                const struct ggml_tensor * src0,
                                const struct ggml_tensor * src1,
                                struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
