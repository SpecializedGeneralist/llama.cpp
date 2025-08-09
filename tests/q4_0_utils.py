import numpy as np

QK4_0 = 32

def quantize_row_q4_0_ref(x):
    assert x.shape[0] % QK4_0 == 0
    nb = x.shape[0] // QK4_0
    blocks = []
    for i in range(nb):
        block = x[i*QK4_0:(i+1)*QK4_0]
        amax_index = np.argmax(np.abs(block))
        maxv = block[amax_index]
        d = maxv / -8.0
        id = 1.0 / d if d != 0 else 0.0
        qs = np.empty(QK4_0 // 2, dtype=np.uint8)
        for j in range(QK4_0 // 2):
            x0 = block[j] * id
            x1 = block[j + QK4_0 // 2] * id
            xi0 = min(15, int(x0 + 8.5))
            xi1 = min(15, int(x1 + 8.5))
            qs[j] = np.uint8(xi0 | (xi1 << 4))
        blocks.append((d, qs))
    return blocks

def dequantize_row_q4_0(blocks):
    y = np.empty(len(blocks) * QK4_0, dtype=np.float32)
    for i, (d, qs) in enumerate(blocks):
        for j, byte in enumerate(qs):
            x0 = (byte & 0x0F) - 8
            x1 = (byte >> 4) - 8
            y[i*QK4_0 + j] = x0 * d
            y[i*QK4_0 + j + QK4_0//2] = x1 * d
    return y
