# =====================================================================
# Codebook fitting on REAL weights, with a pre-registered quality gate.
#
# PRE-REGISTERED, written before any number is seen:
#
#   Hypothesis: a 16-level codebook fitted to the actual post-absmax value
#   distribution beats NF4, and unpinning the +/-1 endpoints beats it further.
#
#   Mechanism prediction: rotation spread error evenly and LOST on quality
#   despite better reconstruction. Endpoint-unpinning does the opposite: it
#   concentrates MORE error on outliers and less on the bulk. If the
#   "concentrated outlier error is more survivable" mechanism is right, the
#   quality gain here should be at least as large as the reconstruction gain.
#
#   PROMOTE  if perplexity improves, or is flat within 0.2%.
#   REJECT   if perplexity degrades by more than 0.2%.
#   Either outcome is reported. The mechanism claim is on trial too.
#
# Runtime: about 40 minutes on an L4 for the 9B. Stage 1 needs no GPU.
# =====================================================================

import numpy as np, torch, json, time, os, gc

# ---------------------------------------------------------------- stage 1
# Fit codebooks on real weight blocks. CPU only, a few minutes.

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=np.float64)


def quantise(v, levels):
    b = (levels[:-1] + levels[1:]) / 2
    return levels[np.searchsorted(b, v, side='left')]


def lloyd_max(samples, k=16, iters=300, pin_zero=True, pin_ends=True):
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
        if pin_ends:
            new[0], new[-1] = -1.0, 1.0
        if pin_zero:
            new[np.argmin(np.abs(new))] = 0.0
        new = np.sort(new)
        if np.allclose(new, lv, atol=1e-12):
            lv = new
            break
        lv = new
    return lv


def post_absmax(x, block=64):
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


from harness.loader import load, free_gpu
from harness.config import RunConfig

MODEL = "qwen-9b"          # the 9B: fp16 fits an L4 and the effect is largest
model, tok, _ = load(RunConfig(experiment_id="cbfit", label="codebook fit",
                               base=MODEL, backend="fp16", attn_impl="sdpa"))

# Sample real decoder weights. Not the embedding or head: those are the
# tensors under test later, and fitting on them would leak.
samples, names = [], []
for n, p in model.named_parameters():
    if p.ndim == 2 and "embed" not in n and "lm_head" not in n:
        w = p.detach().float().cpu().numpy().reshape(-1)
        take = min(w.size - w.size % 64, 64 * 4000)
        samples.append(w[:take])
        names.append(n)
    if len(samples) >= 40:
        break

W = np.concatenate(samples)
print(f"sampled {len(samples)} tensors, {W.size/1e6:.1f}M weights")
print(f"kurtosis {float(((W-W.mean())**4).mean()/W.var()**2 - 3):.3f}, "
      f"max/std {float(np.abs(W).max()/W.std()):.1f}")

v = post_absmax(W)
print(f"post-absmax: std {v.std():.4f}, beyond |0.9|: {(np.abs(v)>0.9).mean()*100:.2f}%\n")

samp = v[np.random.default_rng(0).choice(v.size, min(v.size, 600_000),
                                         replace=False)]
BOOKS = {
    "nf4":          NF4,
    "fit_pinned":   lloyd_max(samp, pin_zero=True,  pin_ends=True),
    "fit_zeroonly": lloyd_max(samp, pin_zero=True,  pin_ends=False),
    "fit_free":     lloyd_max(samp, pin_zero=False, pin_ends=False),
}

base = rel_err(W, NF4)
print(f"{'codebook':16s} {'recon err':>11} {'gain':>8}  max level")
print("-" * 50)
for name, lv in BOOKS.items():
    e = rel_err(W, lv)
    print(f"{name:16s} {e:>11.6f} {(base-e)/base*100:>7.2f}%  {lv[-1]:+.4f}")

os.makedirs("/content/drive/MyDrive/llm-eff", exist_ok=True)
json.dump({k: list(map(float, v)) for k, v in BOOKS.items()},
          open("/content/drive/MyDrive/llm-eff/codebooks.json", "w"), indent=2)

# ---------------------------------------------------------------- stage 2
# Quality gate. Quantise-and-dequantise in place, fp16 storage, so this
# isolates the quality consequence from any size benefit. A clean model is
# reloaded between schemes because the modification is destructive.

from harness.quality import run_lm_eval

def apply_codebook_(model, levels, block=64):
    """In-place quantise-dequantise of every 2D weight with this codebook."""
    lv = torch.tensor(levels, dtype=torch.float32)
    bd = ((lv[:-1] + lv[1:]) / 2)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim != 2 or p.shape[-1] % block:
                continue
            lvd, bdd = lv.to(p.device), bd.to(p.device)
            flat = p.data.float().reshape(-1, block)
            am = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            idx = torch.bucketize(flat / am, bdd)
            p.data.copy_((lvd[idx] * am).reshape(p.shape).to(p.dtype))

results = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": MODEL, "recon": {}}
for name, lv in BOOKS.items():
    results["recon"][name] = rel_err(W, lv)

del model, tok; gc.collect(); free_gpu()

for name, lv in BOOKS.items():
    m, t, _ = load(RunConfig(experiment_id=f"cb_{name}", label=name,
                             base=MODEL, backend="fp16", attn_impl="sdpa"))
    apply_codebook_(m, np.asarray(lv))
    try:
        r = run_lm_eval(m, t, tasks=["wikitext"], limit=None,
                        batch_size=1, max_length=1024)
        ppl = r["scores"]["wikitext"]["word_perplexity,none"]
    except Exception as e:
        ppl = f"ERR {type(e).__name__}: {e}"
    results[name] = {"perplexity": ppl, "recon": results["recon"][name]}
    print(f"{name:16s} ppl {ppl}")
    del m, t; gc.collect(); free_gpu()

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 62)
b_ppl = results["nf4"]["perplexity"]
b_rec = results["recon"]["nf4"]
print(f"{'codebook':16s} {'recon gain':>11} {'ppl':>10} {'ppl gain':>10}  verdict")
print("-" * 62)
for name in BOOKS:
    p = results[name]["perplexity"]
    if isinstance(p, str):
        print(f"{name:16s} {'':>11} {p}")
        continue
    rg = (b_rec - results["recon"][name]) / b_rec * 100
    pg = (b_ppl - p) / b_ppl * 100
    verdict = "PROMOTE" if pg > -0.2 else "REJECT"
    print(f"{name:16s} {rg:>10.2f}% {p:>10.4f} {pg:>9.2f}%  {verdict}")

print("\nMechanism check: does ppl gain match or exceed recon gain?")
print("If yes, 'concentrated outlier error is more survivable' holds.")
print("If no, reconstruction error is unreliable in this direction too.")

json.dump(results, open(f"/content/drive/MyDrive/llm-eff/"
                        f"codebook_{results['utc'].replace(':','')}.json", "w"),
          indent=2)
