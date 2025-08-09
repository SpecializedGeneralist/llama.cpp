#include <metal_stdlib>
using namespace metal;

#define QK4_0 32

typedef matrix<float,4,4> float4x4;
typedef matrix<half,4,4>  half4x4;
typedef simdgroup_matrix<half,8,8> simdgroup_half8x8;

typedef struct {
    half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

struct ggml_metal_kargs_mul_mm {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

// NOTE: this is not dequantizing - we are simply fitting the template
// from ggml's metal backend

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4) reg_f;
}

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4
#define THREAD_MAT_N 2
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2
#define THREAD_PER_COL 4
#define SG_MAT_SIZE 64
#define SG_MAT_ROW 8

template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)> 
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;
    const int im = tgpig.z;

    const short n_rows = (args.ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (args.ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (args.ne1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (args.ne1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    const short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    const short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){ mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); }

    short il = (tiitg % THREAD_PER_ROW);

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0*BLOCK_SIZE_M + thread_row) + offset0) + offset1;
    device const float   * y = (device const float   *)(src1 + args.nb13*i13 + args.nb12*i12 + args.nb11*(r1*BLOCK_SIZE_N + thread_col) + args.nb10*(BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < args.ne00; loop_k += BLOCK_SIZE_K) {
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8)
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8)
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup const float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= args.ne0 && (r1 + 1) * BLOCK_SIZE_N <= args.ne1) {
        device float * C = (device float *) dst + (BLOCK_SIZE_M * r0 + 32*(sgitg &  1)) + (BLOCK_SIZE_N * r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * args.ne0 * (i/4), args.ne0);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mul_mm_t;

template [[host_name("kernel_mul_mm_q4_0_f32")]]
kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, block_q4_0, 2, dequantize_q4_0>;
