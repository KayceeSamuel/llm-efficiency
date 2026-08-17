"""
A log grid for sub-block scales beats a uniform one, but exp() in a decode
kernel is expensive. A 16-entry codebook is a lookup instead: same trick,
kernel-friendly, and it can be fitted rather than assumed.

Scales are absmax_subblock / absmax_superblock, so they live in (0, 1] with
exactly one equal to 1. Fit a codebook to that.
"""
import numpy as np
from format_sweep import NF4, BOUND, QK, relerr, bpw

def fit_positive_codebook(samples, k=16, iters=300, pin_one=True):
    """Lloyd-Max on (0,1] data. pin_one keeps an exact 1.0, which the
    definition guarantees occurs once per superblock."""
    s = np.sort(samples)
    lv = np.quantile(s, np.linspace(0, 1, k))
    for _ in range(iters):
        b = (lv[:-1] + lv[1:]) / 2
        idx = np.searchsorted(b, s, side='left')
        new = lv.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new[j] = s[m].mean()
        if pin_one:
            new[-1] = 1.0
        new = np.sort(new)
        if np.allclose(new, lv, atol=1e-14):
            lv = new; break
        lv = new
    return lv

def roundtrip_cb(x, sub, scale_lv):
    nsub = QK // sub
    sb = x.reshape(-1, QK).reshape(-1, nsub, sub)
    absmax = np.abs(sb).max(axis=2)
    top = absmax.max(axis=1, keepdims=True)
    ratio = np.divide(absmax, top, out=np.zeros_like(absmax), where=top > 0)
    sb_bounds = (scale_lv[:-1] + scale_lv[1:]) / 2
    sc = scale_lv[np.searchsorted(sb_bounds, ratio, side='left')]
    scale = (top * sc)[:, :, None]
    q = np.divide(sb, scale, out=np.zeros_like(sb), where=scale > 0)
    return (NF4[np.searchsorted(BOUND, q, side='left')] * scale).reshape(-1)

def ratios(x, sub):
    nsub = QK // sub
    sb = x.reshape(-1, QK).reshape(-1, nsub, sub)
    am = np.abs(sb).max(axis=2)
    top = am.max(axis=1, keepdims=True)
    return np.divide(am, top, out=np.zeros_like(am), where=top > 0).reshape(-1)

rng = np.random.default_rng(0)
N = QK * 4000
g = rng.standard_normal(N)
X = (g * (1 + 0.35*np.abs(rng.standard_normal(N))))
X = X / X.std()

r32 = ratios(X, 32)
print(f"sub-block scale ratios at SUB=32: min {r32.min():.4f} "
      f"mean {r32.mean():.4f} p1 {np.quantile(r32,0.01):.4f}")
cb32 = fit_positive_codebook(r32[rng.choice(r32.size, min(r32.size, 400_000), replace=False)])
print("fitted 16-entry scale codebook:")
print("  " + "  ".join(f"{v:.4f}" for v in cb32[:8]))
print("  " + "  ".join(f"{v:.4f}" for v in cb32[8:]))

print(f"\n{'variant':>26} {'bytes':>6} {'bpw':>7} {'rel err':>10} {'vs mine':>9}")
print("-" * 62)
from format_sweep import roundtrip
mine = relerr(X, roundtrip(X, 64, 8, "uniform"))
tests = [
    ("SUB=64 8b uniform (mine)", 530, bpw(64, 8), mine),
    ("SUB=32 4b uniform",        530, bpw(32, 4),
     relerr(X, roundtrip(X, 32, 4, "uniform"))),
    ("SUB=32 4b log",            530, bpw(32, 4),
     relerr(X, roundtrip(X, 32, 4, "log"))),
    ("SUB=32 4b codebook",       530, bpw(32, 4),
     relerr(X, roundtrip_cb(X, 32, cb32))),
]
for name, by, b, e in tests:
    print(f"{name:>26} {by:>6} {b:>7.4f} {e:>10.6f} {(mine-e)/mine*100:>8.2f}%")

# is the same true one step further?
r16 = ratios(X, 16)
cb16 = fit_positive_codebook(r16[rng.choice(r16.size, min(r16.size, 400_000), replace=False)])
e16 = relerr(X, roundtrip_cb(X, 16, cb16))
b16 = (QK/2 + (QK/16)*4/8 + 2)*8/QK
print(f"{'SUB=16 4b codebook':>26} {int(QK/2+(QK/16)*0.5+2):>6} {b16:>7.4f} "
      f"{e16:>10.6f} {(mine-e16)/mine*100:>8.2f}%")
print("\n(SUB=16 costs 0.125 bpw more; listed for the frontier, not as a pick)")
