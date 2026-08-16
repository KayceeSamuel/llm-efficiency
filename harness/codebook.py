from harness.codebook import (NF4_LEVELS, fit_codebook, sample_post_absmax,
                              apply_codebook_, codebook_rel_error)
from harness.loader import load, free_gpu
from harness.config import RunConfig
from harness.quality import run_lm_eval
import numpy as np, torch, json, time, gc

RES = "/content/drive/MyDrive/llm-efficiency-results"

m, t, _ = load(RunConfig(experiment_id="cbfit", label="fit", base="qwen-9b",
                         backend="fp16", attn_impl="sdpa"))
v = sample_post_absmax(m, block=64, max_weights=8_000_000)
print(f"sampled {v.size/1e6:.1f}M post-absmax values, std {v.std():.4f}")

BOOKS = {
    "nf4":          NF4_LEVELS,
    "fit_pinned":   fit_codebook(v, pin_zero=True,  pin_ends=True),
    "fit_zeroonly": fit_codebook(v, pin_zero=True,  pin_ends=False),
    "fit_free":     fit_codebook(v, pin_zero=False, pin_ends=False),
}
json.dump({k: list(map(float, x)) for k, x in BOOKS.items()},
          open(f"{RES}/codebooks.json", "w"), indent=2)
del m, t; gc.collect(); free_gpu()

results = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
for name, lv in BOOKS.items():
    m, t, _ = load(RunConfig(experiment_id=f"cb_{name}", label=name,
                             base="qwen-9b", backend="fp16", attn_impl="sdpa"))
    st = apply_codebook_(m, lv)          # chunked, embeddings included
    r = run_lm_eval(m, t, tasks=["wikitext"], limit=None,
                    batch_size=1, max_length=1024)
    results[name] = {"ppl": r["scores"]["wikitext"]["word_perplexity,none"],
                     "recon": st["mean_rel_error"],
                     "max_tensor_err": st["max_rel_error"]}
    print(f"{name:14s} ppl {results[name]['ppl']:.4f}  recon {st['mean_rel_error']:.6f}")
    del m, t; gc.collect(); free_gpu()

b = results["nf4"]
print(f"\n{'codebook':14s} {'recon gain':>11} {'ppl gain':>10}  verdict")
for k in BOOKS:
    rg = (b["recon"] - results[k]["recon"]) / b["recon"] * 100
    pg = (b["ppl"] - results[k]["ppl"]) / b["ppl"] * 100
    print(f"{k:14s} {rg:>10.2f}% {pg:>9.2f}%  {'PROMOTE' if pg > -0.2 else 'REJECT'}")
json.dump(results, open(f"{RES}/codebook_{results['utc'].replace(':','')}.json","w"), indent=2)
