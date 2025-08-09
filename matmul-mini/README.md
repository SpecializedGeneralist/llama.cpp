# Matmul Mini

Minimal extraction of GGML matrix multiplication routines.

Currently includes:

* CPU backend covering F32, F16, BF16 and Q4_0 weights with simple threaded chunking
* Metal backend with Q4_0 quantized matrix-matrix kernel

More backends (additional quantization, CUDA) to be added.
