"""
Theory 3, sharpened.

NF4's 16 levels are the quantiles of a standard normal, mapped to [-1, 1].
But that is not what the quantiser actually sees. Every block of 64 weights is
divided by its own absmax first, so the values being quantised are

    v = x / max(|x| over the block)

The max of 64 draws from a normal sits around 2.5 sigma, so v is roughly a
normal squeezed into [-0.4, 0.4], with the endpoints rare by construction.
That distribution is NOT normal, and it depends on the block size.

So there are two separate reasons NF4's codebook may be leaving error on the
table:

  A. It is fitted to the wrong distribution (raw normal, not post-absmax).
  B. Equiprobable quantiles are not MSE-optimal anyway. Lloyd-Max is.

Three codebooks compared, all 16 levels, all in [-1, 1]:
  nf4        the shipped codebook
  lloyd-norm Lloyd-Max fitted to a normal
  lloyd-emp  Lloyd-Max fitted to the ACTUAL post-absmax values
"""
import numpy as np

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=np.float64)


def quantise(v, levels):
    b = (levels[:-1] + levels[1:]) / 2
    return levels[np.searchsorted(b, v, side='left')]


def lloyd_max(samples, k=16, iters=200, pin_zero=True, pin_ends=True, seed=0):
    """1-D Lloyd-Max. Optionally pin 0.0 and the +/-1 endpoints, since the
    absmax normalisation guarantees +/-1 occur and a zero level is worth
    keeping for exactly-zero weights."""
    s = np.sort(samples)
    lv = np.quantile(s, np.linspace(0, 1, k))          # init
    for _ in range(iters):
        b = (lv[:-1] + lv[1:]) / 2
        idx = np.searchsorted(b, s, side='left')
        new = lv.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new[j] = s[m].mean()
        if pin_ends:
            new[0], new[-1] = -1.0, 1.0
        if pin_zero:
            new[np.argmin(np.abs(new))] = 0.0
        new = np.sort(new)
        if np.allclose(new, lv, atol=1e-12):
            lv = new; break
        lv = new
    return lv


def normalised_values(x, block=64):
    b = x.reshape(-1, block)
    am = np.abs(b).max(axis=1, keepdims=True)
    am = np.where(am == 0, 1e-12, am)
    return (b / am).reshape(-1)


def rel_err(x, levels, block=64):
    b = x.reshape(-1, block)
    am = np.abs(b).max(axis=1, keepdims=True)
    am = np.where(am == 0, 1e-12, am)
    r = quantise(b / am, levels) * am
    return float(np.sqrt(((b - r) ** 2).sum() / (b ** 2).sum()))


rng = np.random.default_rng(0)
N = 64 * 200_000

cases = {}
cases["gaussian"] = rng.standard_normal(N)
g = rng.standard_normal(N)
cases["kurtosis ~1.4"] = (g * (1 + 0.35 * np.abs(rng.standard_normal(N))))
t = rng.standard_t(5, N)
cases["heavy tail (t5)"] = t / t.std()

print("Distribution of post-absmax values, block=64:")
v = normalised_values(cases["gaussian"])
print(f"  std {v.std():.4f}   |v|>0.5: {(np.abs(v)>0.5).mean()*100:5.2f}%"
      f"   |v|>0.9: {(np.abs(v)>0.9).mean()*100:.3f}%")
print("  (a standard normal would have std 1.0 and 61.7% beyond 0.5)\n")

print(f"{'case':18s} {'nf4':>10} {'lloyd-norm':>11} {'lloyd-emp':>11} {'gain':>8}")
print("-" * 62)
books = {}
for name, x in cases.items():
    v = normalised_values(x)
    lv_norm = lloyd_max(rng.standard_normal(400_000))
    lv_emp = lloyd_max(v[rng.choice(v.size, 400_000, replace=False)])
    books[name] = lv_emp

    e_nf4 = rel_err(x, NF4)
    e_ln = rel_err(x, lv_norm)
    e_le = rel_err(x, lv_emp)
    print(f"{name:18s} {e_nf4:10.6f} {e_ln:11.6f} {e_le:11.6f} "
          f"{(e_nf4-e_le)/e_nf4*100:7.2f}%")

print("\nfitted codebook (kurtosis ~1.4 case), 16 levels:")
print("  " + "  ".join(f"{x:+.4f}" for x in books["kurtosis ~1.4"][:8]))
print("  " + "  ".join(f"{x:+.4f}" for x in books["kurtosis ~1.4"][8:]))
print("\nnf4 for comparison:")
print("  " + "  ".join(f"{x:+.4f}" for x in NF4[:8]))
print("  " + "  ".join(f"{x:+.4f}" for x in NF4[8:]))
