import numpy as np
from pathlib import Path
from q4_0_utils import QK4_0, quantize_row_q4_0_ref, dequantize_row_q4_0

def test_q4_0_matmul_matches_fp32():
    rng = np.random.default_rng(0)
    m, k, n = 2, QK4_0, 3
    a_full = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    blocks = [quantize_row_q4_0_ref(row) for row in a_full]
    a_deq = np.vstack([dequantize_row_q4_0(row_blocks) for row_blocks in blocks])

    c_ref = a_full @ b
    c_q = a_deq @ b
    assert np.max(np.abs(c_ref - c_q)) < 1.0

def test_metal_kernel_symbol_exists():
    metal_shader = Path('matmul-mini/metal/matmul_metal.metal').read_text()
    assert 'kernel_mul_mm_q4_0_f32' in metal_shader
