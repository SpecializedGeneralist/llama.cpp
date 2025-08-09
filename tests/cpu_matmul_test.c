#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../matmul-mini/common/ggml_common.h"
#include "../matmul-mini/cpu/matmul_cpu.h"

int main(void) {
    const int k = 32;
    float a[k];
    float b[k];
    for (int i = 0; i < k; ++i) {
        a[i] = sinf((float)i);
        b[i] = cosf((float)i);
    }

    struct ggml_tensor ta = {
        .type = GGML_TYPE_F32,
        .ne = {k,1,1,1},
        .nb = {sizeof(float), k*sizeof(float),0,0},
        .data = a,
        .src = {NULL, NULL}
    };
    struct ggml_tensor tb = {
        .type = GGML_TYPE_F32,
        .ne = {k,1,1,1},
        .nb = {sizeof(float), k*sizeof(float),0,0},
        .data = b,
        .src = {NULL, NULL}
    };
    float c_f32 = 0.0f;
    struct ggml_tensor tc = {
        .type = GGML_TYPE_F32,
        .ne = {1,1,1,1},
        .nb = {sizeof(float), sizeof(float),0,0},
        .data = &c_f32,
        .src = {&ta, &tb}
    };
    struct ggml_compute_params params = {0,1,0,NULL,NULL};
    ggml_compute_forward_mul_mat(&params, &tc);

    size_t row = ggml_row_size(GGML_TYPE_Q4_0, k);
    block_q4_0 *aq = (block_q4_0 *)malloc(row);
    quantize_row_q4_0(a, aq, k);
    struct ggml_tensor taq = {
        .type = GGML_TYPE_Q4_0,
        .ne = {k,1,1,1},
        .nb = {sizeof(block_q4_0), row,0,0},
        .data = aq,
        .src = {NULL, NULL}
    };
    float c_q = 0.0f;
    struct ggml_tensor tcq = {
        .type = GGML_TYPE_F32,
        .ne = {1,1,1,1},
        .nb = {sizeof(float), sizeof(float),0,0},
        .data = &c_q,
        .src = {&taq, &tb}
    };
    ggml_compute_forward_mul_mat(&params, &tcq);

    printf("%f\n%f\n", c_f32, c_q);
    free(aq);
    return 0;
}
