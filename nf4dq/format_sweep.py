"""
NF4DQ format design: two untested decisions.

A. SUB (sub-block size) was set to 64 to match bitsandbytes. Never tested.
   Smaller SUB means more scales, so lower error but higher bpw. Where is the
   knee, and is 64 on the Pareto front at all?

B. Sub-block absmax values are quantised to uint8 UNIFORMLY. But absmax is
   strictly positive and roughly log-normal, so a uniform grid wastes codes
   at the top. If a log grid, or fewer bits, gives the same scale accuracy,
   that is bpw recovered for free.
"""
import numpy as np

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=np.float64)
BOUND = (NF4[:-1] + NF4[1:]) / 2

QK = 1024   # superblock fixed by the 5120 / 17408 divisibility constraint


def roundtrip(x, sub, scale_bits=8, scale_mode="uniform"):
    nsub = QK // sub
    sb = x.reshape(-1, QK).reshape(-1, nsub, sub)
    absmax = np.abs(sb).max(axis=2)                       # (nb, nsub), >= 0

    n_lv = 2 ** scale_bits - 1
    if scale_mode == "uniform":
        d = (absmax.max(axis=1, keepdims=True) / n_lv).astype(np.float16)
        de = d.astype(np.float64)
        sc = np.clip(np.rint(np.divide(absmax, de, out=np.zeros_like(absmax),
                                       where=de > 0)), 0, n_lv)
        scale = (de * sc)[:, :, None]
    else:  # log grid: quantise log(absmax) uniformly between min and max
        pos = np.maximum(absmax, 1e-30)
        lo = np.log(pos).min(axis=1, keepdims=True)
        hi = np.log(pos).max(axis=1, keepdims=True)
        step = np.maximum((hi - lo) / n_lv, 1e-30)
        q = np.clip(np.rint((np.log(pos) - lo) / step), 0, n_lv)
        scale = np.exp(lo + q * step)[:, :, None]

    q = np.divide(sb, scale, out=np.zeros_like(sb), where=scale > 0)
    return (NF4[np.searchsorted(BOUND, q, side='left')] * scale).reshape(-1)


def bpw(sub, scale_bits=8, extra=2):
    return (QK / 2 + (QK / sub) * scale_bits / 8 + extra) * 8 / QK


def relerr(x, r):
    x = x.reshape(-1)
    return float(np.sqrt(((x - r) ** 2).sum() / (x ** 2).sum()))


rng = np.random.default_rng(0)
N = QK * 4000
g = rng.standard_normal(N)
X = (g * (1 + 0.35 * np.abs(rng.standard_normal(N))))    # kurtosis ~1.4
X = (X / X.std())

print("=== A: sub-block size, at 8-bit scales ===\n")
print(f"{'SUB':>5} {'bpw':>7} {'rel err':>10} {'vs SUB=64':>10}")
print("-" * 36)
base = None
rows = []
for sub in (16, 32, 64, 128, 256):
    e = relerr(X, roundtrip(X, sub))
    b = bpw(sub)
    if sub == 64:
        base = e
    rows.append((sub, b, e))
for sub, b, e in rows:
    print(f"{sub:>5} {b:>7.4f} {e:>10.6f} {(base-e)/base*100:>9.2f}%")

print("\n=== B: scale precision and grid, at SUB=64 ===\n")
print(f"{'bits':>5} {'grid':>9} {'bpw':>7} {'rel err':>10} {'vs 8-bit uni':>13}")
print("-" * 50)
ref = relerr(X, roundtrip(X, 64, 8, "uniform"))
for bits in (4, 5, 6, 8):
    for mode in ("uniform", "log"):
        e = relerr(X, roundtrip(X, 64, bits, mode))
        print(f"{bits:>5} {mode:>9} {bpw(64, bits):>7.4f} {e:>10.6f} "
              f"{(ref-e)/ref*100:>12.2f}%")

print("\n=== C: matched-bpw comparison ===")
print("Can a smaller SUB with cheaper scales beat SUB=64 at 8 bits?\n")
print(f"{'config':>22} {'bpw':>7} {'rel err':>10}")
print("-" * 42)
cands = [(64, 8, "uniform"), (32, 4, "uniform"), (32, 4, "log"),
         (32, 5, "log"), (128, 8, "uniform"), (128, 8, "log")]
for sub, bits, mode in cands:
    e = relerr(X, roundtrip(X, sub, bits, mode))
    print(f"{f'SUB={sub} {bits}b {mode}':>22} {bpw(sub, bits):>7.4f} {e:>10.6f}")
