"""
Cross-validate the C reference against harness/qembed.py's algorithm.

Two independent implementations of the same format, same input data. This is
the gate the whole thing rests on: if a checkpoint packed by one cannot be
read by the other, the format is not a format.
"""
import numpy as np, ctypes, subprocess, os

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=np.float32)
BOUND = (NF4[:-1] + NF4[1:]) / 2

QK, SUB = 1024, 64
NSUB = QK // SUB

def qembed_pack(x, block=SUB):
    """Exactly harness/qembed.py: absmax per block, fp16 scale, bucketize."""
    blocks = x.reshape(-1, block).astype(np.float32)
    absmax = np.abs(blocks).max(axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1e-12, absmax)
    idx = np.searchsorted(BOUND, (blocks / absmax), side='left').astype(np.uint8)
    return idx, absmax.astype(np.float16)

def qembed_unpack(idx, scales, block=SUB):
    return (NF4[idx] * scales.astype(np.float32)).reshape(-1)

# ---- C side, via ctypes
subprocess.run(["gcc","-O2","-std=c11","-fPIC","-shared",
                "-o","libnf4dq.so","ggml-nf4dq.c","-lm"], check=True)
lib = ctypes.CDLL("./libnf4dq.so")

class Block(ctypes.Structure):
    _fields_ = [("qs", ctypes.c_uint8*(QK//2)), ("sc", ctypes.c_uint8*NSUB),
                ("d", ctypes.c_uint16)]
assert ctypes.sizeof(Block) == QK//2 + NSUB + 2, ctypes.sizeof(Block)

lib.quantize_row_nf4dq_ref.argtypes = [ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(Block), ctypes.c_int64]
lib.dequantize_row_nf4dq.argtypes  = [ctypes.POINTER(Block),
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

def c_roundtrip(x):
    k = x.size; nb = k // QK
    blocks = (Block * nb)()
    xin = (ctypes.c_float * k)(*x.tolist())
    lib.quantize_row_nf4dq_ref(xin, blocks, k)
    out = (ctypes.c_float * k)()
    lib.dequantize_row_nf4dq(blocks, out, k)
    return np.frombuffer(out, dtype=np.float32).copy(), blocks

def relerr(a, b):
    return float(np.sqrt(((a-b)**2).sum() / (a**2).sum()))

rng = np.random.default_rng(0)
print(f"{'case':30s} {"qembed(4.25bpw)":>17s} {"nf4dq(4.14bpw)":>16s} {'nibbles':>9s}")
print("-"*78)

for name, x in [
    ("gaussian",              rng.standard_normal(QK*512).astype(np.float32)),
    ("gaussian + 16-sigma",   None),
    ("heavy-tailed (t, df=3)", (rng.standard_t(3, QK*512)/1.732).astype(np.float32)),
    ("real-ish: kurtosis 1.4", None),
]:
    if name == "gaussian + 16-sigma":
        x = rng.standard_normal(QK*512).astype(np.float32)
        x[::997] = np.where(np.arange(len(x[::997])) % 2 == 0, 16.0, -16.0)
    if name.startswith("real-ish"):
        # match Experiment 2's measured profile: kurtosis ~1.4, max/std ~16
        g = rng.standard_normal(QK*512).astype(np.float32)
        x = (g * (1 + 0.35*np.abs(rng.standard_normal(g.size)))).astype(np.float32)
        x = (x / x.std() ).astype(np.float32)

    # qembed path
    idx, sc = qembed_pack(x)
    rq = qembed_unpack(idx, sc)
    e_qembed = relerr(x, rq)

    # nf4dq path
    rc, blocks = c_roundtrip(x)
    e_nf4dq = relerr(x, rc)

    # do the two implementations choose the SAME codebook indices?
    nb = x.size // QK
    c_idx = np.zeros(x.size, dtype=np.uint8)
    for b in range(nb):
        qs = np.frombuffer(bytes(blocks[b].qs), dtype=np.uint8)
        lo, hi = qs & 0x0F, qs >> 4
        inter = np.empty(QK, dtype=np.uint8)
        inter[0::2], inter[1::2] = lo, hi
        c_idx[b*QK:(b+1)*QK] = inter
    agree = (c_idx == idx.reshape(-1)).mean()

    print(f"{name:30s} {e_qembed:17.6f} {e_nf4dq:16.6f} {agree*100:8.2f}%")

print()
print("qembed reference from real weights: 0.091933 / 0.091831 / 0.092064")
