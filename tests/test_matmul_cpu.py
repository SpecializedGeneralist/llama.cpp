import subprocess
import numpy as np
from pathlib import Path
from q4_0_utils import quantize_row_q4_0_ref, dequantize_row_q4_0, QK4_0

SRC = Path('tests/cpu_matmul_test.c')

def test_cpu_matmul_matches_reference(tmp_path):
    exe = tmp_path / 'cpu_matmul_test'
    subprocess.run(['gcc', str(SRC), 'matmul-mini/cpu/matmul_cpu.c', '-I.', '-std=c11', '-lm', '-o', str(exe)], check=True)
    out = subprocess.check_output([str(exe)]).decode().strip().splitlines()
    c_f32 = float(out[0])
    c_q = float(out[1])

    k = QK4_0
    a = np.sin(np.arange(k, dtype=np.float32))
    b = np.cos(np.arange(k, dtype=np.float32))
    ref = float(np.dot(a, b))
    blocks = quantize_row_q4_0_ref(a)
    a_deq = dequantize_row_q4_0(blocks)
    ref_q = float(np.dot(a_deq, b))
    assert abs(c_f32 - ref) < 1e-5
    assert abs(c_q - ref_q) < 1.0
