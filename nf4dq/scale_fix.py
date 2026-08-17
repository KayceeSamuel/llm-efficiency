"""The scale codebook was fitted on outlier-free data and clamps badly when
a superblock contains a 16-sigma weight. Refit on realistic data and compare
against the per-superblock log grid, which adapts by construction."""
import numpy as np
from format_sweep import NF4, BOUND, QK, relerr, roundtrip
from scale_codebook import fit_positive_codebook, roundtrip_cb, ratios

rng = np.random.default_rng(0)
N = QK * 4000

def make(kind):
    g = rng.standard_normal(N)
    x = g * (1 + 0.35*np.abs(rng.standard_normal(N)))
    x = x / x.std()
    if kind == "outliers":
        x = x.copy(); x[::997] = np.where(np.arange(x[::997].size) % 2, -16.0, 16.0)
    return x

clean, outl = make("clean"), make("outliers")

print("ratio range at SUB=32:")
for name, x in (("clean", clean), ("with 16-sigma", outl)):
    r = ratios(x, 32)
    print(f"  {name:14s} min {r.min():.4f}  p1 {np.quantile(r,0.01):.4f}  "
          f"p50 {np.quantile(r,0.5):.4f}")

# refit on a mix so the codebook covers both regimes
mix = np.concatenate([ratios(clean, 32), ratios(outl, 32)])
cb_mix = fit_positive_codebook(mix[rng.choice(mix.size, min(mix.size, 500_000),
                                              replace=False)])
cb_out = fit_positive_codebook(ratios(outl, 32))

print(f"\nrefit codebook (mixed): {cb_mix[0]:.4f} .. {cb_mix[-1]:.4f}")
print("  " + "  ".join(f"{v:.4f}" for v in cb_mix))

print(f"\n{'variant':>28} {'clean':>10} {'outliers':>10}")
print("-"*50)
rows = [
  ("SUB=64 8b uniform (v1)", lambda x: roundtrip(x, 64, 8, "uniform")),
  ("SUB=32 4b uniform",      lambda x: roundtrip(x, 32, 4, "uniform")),
  ("SUB=32 4b log",          lambda x: roundtrip(x, 32, 4, "log")),
  ("SUB=32 cb (clean-fit)",  lambda x: roundtrip_cb(x, 32,
        fit_positive_codebook(ratios(clean,32)))),
  ("SUB=32 cb (mixed-fit)",  lambda x: roundtrip_cb(x, 32, cb_mix)),
]
for name, fn in rows:
    print(f"{name:>28} {relerr(clean, fn(clean)):>10.6f} "
          f"{relerr(outl, fn(outl)):>10.6f}")
