"""
production.py — the validated configuration. Do not experiment in this file.

WHAT THIS IS:

A single entry point that reproduces the measured, quality-validated model:

    4.326 GB VRAM, 12.77 tok/s decode, on an NVIDIA L4 (22.03 GB)

against a bitsandbytes NF4 baseline of 7.122 GB and 11.84 tok/s. That is
39.3% less memory and 7.9% faster, simultaneously.

This file exists so that the working configuration survives independently of
any experimental work. If lattice quantisation, trellis coding, or anything
else breaks the harness, `load_production_model()` still returns the model
that was measured and validated on 2026-08-09.

WHAT MAKES IT WORK -- two changes to the default bitsandbytes path:

  1. llm_int8_skip_modules=[]
     bitsandbytes skips lm_head by default. On this model lm_head is NOT tied
     to the embedding table, so it is a separate 248,320 x 4,096 matrix left
     at bf16. Emptying the skip list quantises it.
     Measured: 7.122 -> 5.722 GB, and 11.84 -> 12.67 tok/s. The speedup is
     real: decode is memory-bandwidth-bound and that matmul runs every token,
     so moving 4x fewer bytes through it helps.

  2. QuantizedEmbedding for the input table
     bitsandbytes handles nn.Linear, not nn.Embedding, so the table stays at
     bf16 regardless of configuration. qembed.py packs it to 4-bit NF4 and
     dequantises only the rows a prompt looks up.
     Measured: 5.722 -> 4.326 GB, 12.67 -> 12.77 tok/s.

MEASURED PROVENANCE:

  memory and speed   run T3-QEMBED, 2026-08-09, real bitsandbytes load
  quality            run T2-QUALITY (nf4+emb) at n=1000 x 3 tasks plus
                     wikitext perplexity:
                        accuracy    0.74533  vs 0.74667 for plain nf4
                        perplexity  11.7412  vs 11.6053 for plain nf4
                        fp16 reference 0.75300 / 11.0996
                     Accuracy delta is 1/10 of the 0.0136 noise floor.
                     Perplexity is the sensitive test here, because the LM
                     head IS the output distribution over all 248,320 tokens
                     and a multiple-choice task only checks rankings.

KNOWN LIMITS -- state these before anyone relies on the model:

  - All quality measurement is loglikelihood-based. Open-ended generation is
    untested.
  - Measured on Qwen3.5-9B only. The 27B projection (~14.2 GB) is arithmetic.
  - Latency figures are L4-specific. Decode is bandwidth-bound, so an A100 at
    ~5x the bandwidth will not show the same ratios.
  - Rotation was tested and REJECTED: it lowered reconstruction error while
    lowering both accuracy and perplexity. It is not in this configuration
    and should not be added without re-running the quality gate.
"""

import gc
import time
from typing import Dict, Any, Optional, Tuple

import torch


# The exact configuration measured on 2026-08-09. Changing any of these
# invalidates the numbers in this file's docstring.
PRODUCTION_CONFIG = {
    "model_id": "Qwen/Qwen3.5-9B",
    "quant_type": "nf4",
    "double_quant": True,
    "block_size": 64,
    "attn_impl": "sdpa",
    "skip_modules": [],          # empty: quantise lm_head too
    "quantise_embeddings": True,  # via qembed.QuantizedEmbedding
    "measured": {
        "vram_gb": 4.326,
        "decode_tokens_per_sec": 12.77,
        "gpu": "NVIDIA L4 22.03 GB",
        "date": "2026-08-09",
    },
    "baseline": {
        "vram_gb": 7.122,
        "decode_tokens_per_sec": 11.84,
        "description": "bitsandbytes NF4 defaults",
    },
    "quality": {
        "accuracy": 0.74533,
        "accuracy_baseline_nf4": 0.74667,
        "accuracy_fp16": 0.75300,
        "perplexity": 11.7412,
        "perplexity_baseline_nf4": 11.6053,
        "perplexity_fp16": 11.0996,
        "eval": "arc_easy + hellaswag + piqa @ n=1000 each, wikitext ppl",
        "noise_floor_2se": 0.0136,
    },
}


def load_production_model(
    model_id: Optional[str] = None,
    quantise_embeddings: bool = True,
    attn_impl: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load the validated configuration.

    Returns (model, tokenizer, info). `info` records what actually happened,
    including resting VRAM, so a silent regression is visible rather than
    assumed away.

    quantise_embeddings=False falls back to the intermediate 5.722 GB
    configuration (LM head only). Useful if the custom embedding path ever
    conflicts with something, since it needs no code beyond bitsandbytes.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    cfg = PRODUCTION_CONFIG
    model_id = model_id or cfg["model_id"]
    attn_impl = attn_impl or cfg["attn_impl"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    compute_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["double_quant"],
        # The one-line change worth 1.4 GB and 7% throughput.
        llm_int8_skip_modules=cfg["skip_modules"],
    )

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        # Explicit single-GPU placement. accelerate's "auto" has been observed
        # offloading to CPU despite ample free VRAM, which destroys both
        # memory and latency measurements.
        device_map={"": 0},
        trust_remote_code=True,
        # Without this, loading transiently holds both the original bf16
        # tensors and their quantised replacements, roughly doubling peak
        # load memory.
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    info: Dict[str, Any] = {
        "model_id": model_id,
        "load_seconds": round(time.perf_counter() - t0, 2),
        "attn_impl": attn_impl,
    }
    if torch.cuda.is_available():
        info["vram_after_bnb_gb"] = round(
            torch.cuda.memory_allocated() / 1024**3, 3)
    if verbose:
        print(f"bitsandbytes load: {info.get('vram_after_bnb_gb')} GB")

    if quantise_embeddings:
        from .qembed import quantize_model_embeddings
        q = quantize_model_embeddings(model, block_size=cfg["block_size"])
        info["embedding_quantisation"] = q
        if q.get("status") != "ok":
            # Do not fail silently: a skipped embedding means the model is
            # 1.4 GB larger than this function claims.
            info["warning"] = (
                f"embedding quantisation did not apply ({q.get('status')}): "
                f"{q.get('reason', 'no reason given')}. VRAM will be ~1.4 GB "
                f"above the documented figure.")
            if verbose:
                print("WARNING:", info["warning"])
        elif verbose:
            print(f"embedding table: {q['before_gb']} -> {q['total_gb']} GB "
                  f"({q['compression_ratio']}x, verify_err "
                  f"{q['verify_rel_error']})")

    if torch.cuda.is_available():
        info["vram_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        expected = cfg["measured"]["vram_gb"]
        drift = info["vram_gb"] - expected
        info["vram_vs_documented"] = round(drift, 3)
        # Flag divergence from the measured figure rather than trusting that
        # the same code produces the same result on different hardware or
        # library versions.
        if abs(drift) > 0.3:
            info["warning_vram"] = (
                f"resting VRAM {info['vram_gb']} GB differs from the "
                f"documented {expected} GB by {drift:+.3f} GB; the "
                f"configuration may not have applied as expected")
            if verbose:
                print("WARNING:", info["warning_vram"])
        elif verbose:
            print(f"total: {info['vram_gb']} GB "
                  f"(documented {expected} GB, delta {drift:+.3f})")

    return model, tok, info


def verify_production_model(model=None, tok=None,
                            prompt: str = "The mitochondrial respiratory chain",
                            max_new_tokens: int = 64) -> Dict[str, Any]:
    """
    Confirm a loaded model is healthy: VRAM in range, generates coherent text,
    decode speed in the expected band.

    Run this after any change to the harness. It is the difference between
    "the code still imports" and "the model still works".
    """
    close_after = False
    if model is None:
        model, tok, _ = load_production_model(verbose=False)
        close_after = True

    cfg = PRODUCTION_CONFIG
    out: Dict[str, Any] = {}

    if torch.cuda.is_available():
        out["vram_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)

    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]

    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, repetition_penalty=1.1,
                             pad_token_id=tok.pad_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = gen.shape[1] - n_in
    text = tok.decode(gen[0][n_in:], skip_special_tokens=True)

    out["decode_tokens_per_sec"] = round(new_tokens / elapsed, 2)
    out["sample_output"] = text[:300]

    checks = {}
    if "vram_gb" in out:
        checks["vram_in_range"] = abs(
            out["vram_gb"] - cfg["measured"]["vram_gb"]) < 0.3
    checks["speed_in_range"] = (
        out["decode_tokens_per_sec"] > cfg["measured"]["decode_tokens_per_sec"] * 0.8)
    # Coherence: a broken quantisation often still loads and runs fast while
    # emitting repeated tokens or punctuation.
    checks["output_nonempty"] = len(text.strip()) > 20
    checks["output_not_degenerate"] = len(set(text.split())) > 5

    out["checks"] = checks
    out["healthy"] = all(checks.values())

    if close_after:
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def summary() -> str:
    """One-screen description of the validated configuration."""
    c = PRODUCTION_CONFIG
    m, b, q = c["measured"], c["baseline"], c["quality"]
    return f"""
VALIDATED CONFIGURATION -- {c['model_id']}
Measured {m['date']} on {m['gpu']}

                        baseline      this config    change
  VRAM (resting)        {b['vram_gb']} GB       {m['vram_gb']} GB       -{(1-m['vram_gb']/b['vram_gb'])*100:.1f}%
  Decode                {b['decode_tokens_per_sec']} tok/s     {m['decode_tokens_per_sec']} tok/s     +{(m['decode_tokens_per_sec']/b['decode_tokens_per_sec']-1)*100:.1f}%
  Accuracy              {q['accuracy_baseline_nf4']}       {q['accuracy']}       {q['accuracy']-q['accuracy_baseline_nf4']:+.5f}
  Perplexity            {q['perplexity_baseline_nf4']}       {q['perplexity']}       {(q['perplexity']/q['perplexity_baseline_nf4']-1)*100:+.2f}%

  fp16 reference: {q['accuracy_fp16']} accuracy, {q['perplexity_fp16']} perplexity
  Noise floor (2 SE): {q['noise_floor_2se']} -- the accuracy change is 1/10 of it

WHAT PRODUCES IT
  1. llm_int8_skip_modules=[]   quantises the untied lm_head   (-1.400 GB)
  2. QuantizedEmbedding          packs the nn.Embedding table   (-1.391 GB)

  Neither is available from bitsandbytes defaults: the first is off by
  default, the second is outside what bitsandbytes handles at all.

NOT IN THIS CONFIGURATION
  Hadamard rotation -- tested and rejected. Lower reconstruction error,
  worse accuracy AND worse perplexity. Do not add without a quality gate.

UNTESTED
  Open-ended generation. All quality measurement is loglikelihood-based.
""".strip()
