#import "matmul_metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct ggml_metal_context {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> mul_mm_q4_0_f32;
};

struct ggml_metal_context * ggml_metal_init(void) {
    struct ggml_metal_context * ctx = calloc(1, sizeof(*ctx));
    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->queue  = [ctx->device newCommandQueue];

    NSError * err = nil;
    NSString * src = [NSString stringWithContentsOfFile:@"matmul-mini/metal/matmul_metal.metal" encoding:NSUTF8StringEncoding error:&err];
    ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&err];
    id<MTLFunction> fn = [ctx->library newFunctionWithName:@"kernel_mul_mm_q4_0_f32"];
    ctx->mul_mm_q4_0_f32 = [ctx->device newComputePipelineStateWithFunction:fn error:&err];
    return ctx;
}

void ggml_metal_free(struct ggml_metal_context * ctx) {
    if (!ctx) return;
    [ctx->queue release];
    [ctx->library release];
    [ctx->device release];
    free(ctx);
}

void ggml_metal_mul_mm_q4_0_f32(struct ggml_metal_context * ctx,
                                const struct ggml_tensor * src0,
                                const struct ggml_tensor * src1,
                                struct ggml_tensor * dst) {
    id<MTLBuffer> id_src0 = (__bridge id<MTLBuffer>) src0->data;
    id<MTLBuffer> id_src1 = (__bridge id<MTLBuffer>) src1->data;
    id<MTLBuffer> id_dst  = (__bridge id<MTLBuffer>) dst->data;

    ggml_metal_kargs_mul_mm args = {
        (int32_t) src0->ne[0],
        (int32_t) src0->ne[2],
        src0->nb[1],
        src0->nb[2],
        src0->nb[3],
        (int32_t) src1->ne[2],
        src1->nb[0],
        src1->nb[1],
        src1->nb[2],
        src1->nb[3],
        (int32_t) dst->ne[0],
        (int32_t) dst->ne[1],
        1,
        1,
    };

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:ctx->mul_mm_q4_0_f32];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:id_src0 offset:0 atIndex:1];
    [encoder setBuffer:id_src1 offset:0 atIndex:2];
    [encoder setBuffer:id_dst  offset:0 atIndex:3];
    [encoder setThreadgroupMemoryLength:8192 atIndex:0];

    MTLSize tg = MTLSizeMake((src1->ne[1] + 31)/32, (src0->ne[1] + 63)/64, src1->ne[2]*src1->ne[3]);
    MTLSize tp = MTLSizeMake(128, 1, 1);
    [encoder dispatchThreadgroups:tg threadsPerThreadgroup:tp];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}
