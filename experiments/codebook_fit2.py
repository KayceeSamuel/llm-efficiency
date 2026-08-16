"""
Two follow-ups from the fitted-codebook result.

3b. Does the optimal codebook depend on BLOCK SIZE? It should: the absmax of
    64 draws is larger than of 32, so the post-absmax values are more
    concentrated at larger blocks. NF4 ships one fixed codebook. bitsandbytes
    uses block 64; ggml Q4_K uses 32. If the optimum moves, one of them is
    using a codebook fitted for the other.

3c. Are the +/-1 endpoints worth a level each? Exactly one value per block
    hits the absmax, so roughly 1.5% of values sit at an endpoint, yet they
    consume 2 of 16 levels. Letting Lloyd-Max place levels freely (and
    clipping the rare extremes) may trade a little tail error for a lot of
    bulk precision.
"""
import numpy as np
from codebook_fit import NF4, lloyd_max, quantise, normalised_values

def rel_err(x, levels, block):
    b = x.reshape(-1, block)
    am = np.abs(b).max(axis=1, keepdims=True)
    am = np.where(am == 0, 1e-12, am)
    r = quantise(b / am, levels) * am
    return float(np.sqrt(((b - r) ** 2).sum() / (b ** 2).sum()))

def norm_vals(x, block):
    b = x.reshape(-1, block)
    am = np.abs(b).max(axis=1, keepdims=True)
    am = np.where(am == 0, 1e-12, am)
    return (b / am).reshape(-1)

rng = np.random.default_rng(1)
N = 64 * 300_000
g = rng.standard_normal(N)
X = (g * (1 + 0.35 * np.abs(rng.standard_normal(N))))   # kurtosis ~1.4

print("=== 3b: does the optimal codebook move with block size? ===\n")
print(f"{'block':>6} {'post-absmax std':>16} {'nf4 err':>10} {'fitted err':>11} {'gain':>8}")
print("-" * 56)
fitted = {}
for blk in (16, 32, 64, 128, 256):
    v = norm_vals(X, blk)
    samp = v[rng.choice(v.size, 400_000, replace=False)]
    lv = lloyd_max(samp)
    fitted[blk] = lv
    e_nf4, e_fit = rel_err(X, NF4, blk), rel_err(X, lv, blk)
    print(f"{blk:>6} {v.std():>16.4f} {e_nf4:>10.6f} {e_fit:>11.6f} "
          f"{(e_nf4-e_fit)/e_nf4*100:>7.2f}%")

print("\ncross-application: a codebook fitted at one block size, used at another")
print(f"{'fitted@':>8}" + "".join(f"{f'used@{b}':>12}" for b in (32, 64, 128)))
print("-" * 46)
for fb in (32, 64, 128):
    row = "".join(f"{rel_err(X, fitted[fb], ub):>12.6f}" for ub in (32, 64, 128))
    print(f"{fb:>8}" + row)

print("\n\n=== 3c: are the +/-1 endpoints worth a level each? ===\n")
v = norm_vals(X, 64)
samp = v[rng.choice(v.size, 400_000, replace=False)]
variants = {
    "nf4 (shipped)":            NF4,
    "fitted, ends+zero pinned": lloyd_max(samp, pin_zero=True,  pin_ends=True),
    "fitted, zero pinned only": lloyd_max(samp, pin_zero=True,  pin_ends=False),
    "fitted, free":             lloyd_max(samp, pin_zero=False, pin_ends=False),
}
print(f"{'variant':28s} {'err':>10} {'gain':>8}  max level")
print("-" * 60)
base = rel_err(X, NF4, 64)
for name, lv in variants.items():
    e = rel_err(X, lv, 64)
    print(f"{name:28s} {e:>10.6f} {(base-e)/base*100:>7.2f}%  {lv[-1]:+.4f}")

print(f"\nfraction of post-absmax values beyond |0.9|: "
      f"{(np.abs(v) > 0.9).mean()*100:.2f}%")
