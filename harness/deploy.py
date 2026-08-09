"""
deploy.py — does the 4.29 GB actually materialise, and how fast is it?

WHY THIS EXISTS:

Every result so far used SIMULATED quantisation: weights held in fp16 storage
with quantisation error written into them. That was the right way to measure
QUALITY -- it isolates the error from every other variable -- but it saves no
memory and changes no speed. The 4.29 GB figure is arithmetic (3.22 GB decoder
+ 2 x 0.474 GB), not a measurement.

This module closes that gap. It loads a real bitsandbytes 4-bit model with the
embedding table and LM head actually packed, then measures:

  1. real resting VRAM        -- does 4.29 GB materialise?
  2. decode tokens/sec        -- against the 11.4 tok/s NF4 baseline
  3. peak memory vs context   -- what fits on this card
  4. a generation sample      -- does it still produce coherent text?

THE CONFIG QUESTION THIS ANSWERS:

bitsandbytes exposes llm_int8_skip_modules, which lists modules to EXCLUDE
from quantisation. lm_head is conventionally in the default skip list, which
is the likely reason 3.79 GB (53% of the quantised model) sits at bf16. If
passing an empty list is sufficient, this is a configuration fix rather than
an engineering project.

Three variants are compared so the answer is unambiguous:
  default   -- whatever bitsandbytes does unprompted (the 7.13 GB baseline)
  no_skip   -- skip list emptied
  explicit  -- skip list emptied AND tie_word_embeddings left alone

If none of them shrinks the model, bitsandbytes cannot do this and a custom
packing path is required. Knowing that is the point.
"""

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch


# ---------------------------------------------------------------------------
# Load variants
# ---------------------------------------------------------------------------

def _bnb_config(variant: str, compute_dtype):
    from transformers import BitsAndBytesConfig

    base = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    if variant == "default":
        # Production baseline. Whatever bitsandbytes skips by default is
        # skipped. Measured at 7.13 GB, of which 3.79 GB is unquantised
        # embedding table plus untied LM head.
        return BitsAndBytesConfig(**base)

    if variant == "no_skip":
        # Empty skip list. If lm_head is skipped by default, this should
        # bring it into the quantised set.
        return BitsAndBytesConfig(**base, llm_int8_skip_modules=[])

    if variant == "explicit":
        # Same, stated explicitly rather than relying on the empty-list
        # default being interpreted as "skip nothing" rather than
        # "use the built-in list".
        return BitsAndBytesConfig(
            **base,
            llm_int8_skip_modules=[],
            bnb_4bit_quant_storage=torch.uint8,
        )

    raise ValueError(f"unknown variant: {variant}")


def load_real_quantised(model_id: str, variant: str = "default",
                        attn_impl: str = "sdpa"):
    """Load with real bitsandbytes packing and report what actually happened."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    bf16_ok = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config(variant, compute_dtype),
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = round(time.perf_counter() - t0, 2)

    info = {
        "variant": variant,
        "load_seconds": load_seconds,
        "resting_vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "peak_during_load_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    info.update(inspect_quantisation(model))
    return model, tok, info


def inspect_quantisation(model) -> Dict[str, Any]:
    """
    Report which modules were actually quantised, by parameter dtype.

    bitsandbytes replaces Linear layers with Linear4bit whose weight is a
    uint8 Params4bit tensor. Anything still sitting in bf16 or fp16 was NOT
    quantised, whatever the config asked for -- so this reads the result
    rather than trusting the request.
    """
    out: Dict[str, Any] = {}

    def _describe(t):
        if t is None or not torch.is_tensor(t):
            return None
        return {
            "dtype": str(t.dtype),
            "shape": tuple(t.shape),
            "gb": round(t.numel() * t.element_size() / 1024**3, 4),
            "quantised": t.dtype == torch.uint8,
        }

    emb = model.get_input_embeddings()
    out["embeddings"] = _describe(getattr(emb, "weight", None)) if emb else None

    head = getattr(model, "lm_head", None)
    head_w = getattr(head, "weight", None) if head is not None else None
    out["lm_head"] = _describe(head_w)
    out["lm_head_class"] = type(head).__name__ if head is not None else None

    if head_w is not None and emb is not None:
        out["tied"] = head_w.data_ptr() == emb.weight.data_ptr()

    # Bucket every parameter by dtype so nothing unquantised hides.
    by_dtype: Dict[str, Dict[str, float]] = {}
    for name, p in model.named_parameters():
        k = str(p.dtype)
        d = by_dtype.setdefault(k, {"count": 0, "gb": 0.0})
        d["count"] += 1
        d["gb"] += p.numel() * p.element_size() / 1024**3
    for k in by_dtype:
        by_dtype[k]["gb"] = round(by_dtype[k]["gb"], 4)
    out["params_by_dtype"] = by_dtype

    # The headline diagnostic: how much weight memory is still unquantised.
    unquantised = sum(v["gb"] for k, v in by_dtype.items()
                      if "uint8" not in k and "int8" not in k)
    out["unquantised_gb"] = round(unquantised, 3)

    return out


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def measure_throughput(model, tok, prompt_tokens: int = 512,
                       max_new_tokens: int = 128,
                       repeats: int = 3) -> Dict[str, Any]:
    """
    Decode tokens/sec on a real quantised model.

    This is the number that matters for a deployment: users wait on tokens
    appearing, not on the resting weight figure.

    Prefill and decode are separated because they are different regimes.
    Decode is memory-bandwidth-bound -- every generated token streams the
    whole weight set through the bus -- which is why quantising the
    248,320 x 4,096 LM head could plausibly SPEED UP generation as well as
    shrink it. Whether the dequantisation overhead eats that gain is exactly
    what this measures.
    """
    from harness.profiler import make_synthetic_prompt

    prompt = make_synthetic_prompt(tok, prompt_tokens)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]

    # Warm-up: first call triggers kernel compilation, which would otherwise
    # be attributed to the first measured run.
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()

    # Prefill alone
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**enc, use_cache=True)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    runs = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        new_tokens = out.shape[1] - n_in
        runs.append({
            "seconds": elapsed,
            "new_tokens": int(new_tokens),
            "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        })

    tps = sorted(r["tokens_per_sec"] for r in runs)
    sample = tok.decode(out[0][n_in:], skip_special_tokens=True)

    return {
        "prompt_tokens": n_in,
        "prefill_seconds": round(prefill_s, 4),
        "prefill_tokens_per_sec": round(n_in / prefill_s, 1) if prefill_s > 0 else None,
        "decode_tokens_per_sec_median": round(tps[len(tps)//2], 2),
        "decode_tokens_per_sec_min": round(tps[0], 2),
        "decode_tokens_per_sec_max": round(tps[-1], 2),
        "peak_gb": round(max(r["peak_gb"] for r in runs), 3),
        # Coherence check: a model that loads and runs fast but produces
        # gibberish has not been successfully quantised.
        "sample_output": sample[:300],
    }


def measure_context_scaling(model, tok,
                            lengths: List[int] = (512, 4096, 16384, 32768),
                            max_new_tokens: int = 32) -> List[Dict[str, Any]]:
    """Peak memory at several context lengths. OOM is recorded, not raised."""
    from harness.profiler import make_synthetic_prompt

    rows = []
    for n in lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row: Dict[str, Any] = {"target_tokens": n}
        try:
            prompt = make_synthetic_prompt(tok, n)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            row.update({
                "actual_tokens": enc["input_ids"].shape[1],
                "peak_gb": round(torch.cuda.max_memory_allocated()/1024**3, 3),
                "seconds": round(el, 3),
                "status": "ok",
            })
            del out, enc
        except torch.cuda.OutOfMemoryError:
            row["status"] = "oom"
            torch.cuda.empty_cache()
            rows.append(row)
            break
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            torch.cuda.empty_cache()
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_deployment_check(
    model_id: str = "Qwen/Qwen3.5-9B",
    variants: tuple = ("default", "no_skip"),
    context_lengths: List[int] = (512, 4096, 16384, 32768),
    results_dir=None,
) -> Dict[str, Any]:
    """
    The verification. Loads each variant for real and measures what happens.

    Baselines to compare against, both measured earlier in this project:
      resting VRAM   7.13 GB  (default bitsandbytes NF4)
      decode speed   11.4 tok/s
    Target from the simulated work: 4.29 GB at equivalent quality.
    """
    from harness.config import capture_environment, run_stamp

    record: Dict[str, Any] = {
        "run_id": f"T3-DEPLOY-{run_stamp()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "measured",
        "environment": capture_environment(),
        "model_id": model_id,
        "baselines": {
            "resting_vram_gb": 7.13,
            "decode_tokens_per_sec": 11.4,
            "target_vram_gb": 4.29,
            "note": "target is arithmetic from the simulated work, not yet measured",
        },
        "variants": {},
    }

    for variant in variants:
        print(f"\n=== {variant} ===")
        entry: Dict[str, Any] = {"variant": variant}
        model = None
        try:
            model, tok, info = load_real_quantised(model_id, variant)
            entry["load"] = info
            print(f"  resting VRAM: {info['resting_vram_gb']} GB   "
                  f"unquantised: {info['unquantised_gb']} GB")
            print(f"  lm_head: {info.get('lm_head')}")

            print("  measuring throughput ...")
            entry["throughput"] = measure_throughput(model, tok)
            print(f"  decode: {entry['throughput']['decode_tokens_per_sec_median']} tok/s")

            print("  context scaling ...")
            entry["context_scaling"] = measure_context_scaling(
                model, tok, lengths=list(context_lengths))

            entry["status"] = "ok"
        except Exception as e:
            import traceback
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            print(f"  ERROR: {e}")
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        record["variants"][variant] = entry

    record["verdict"] = _verdict(record)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if results_dir is not None:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{record['run_id']}.json").open("w") as f:
            json.dump(record, f, indent=2)

    return record


def _verdict(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Does bitsandbytes actually deliver the saving, and does speed change?

    Pre-registered: a variant counts as working if resting VRAM drops below
    5.5 GB, comfortably between the 7.13 GB baseline and the 4.29 GB target,
    allowing for quantisation constants and allocator overhead.
    """
    base = record["variants"].get("default", {})
    base_vram = base.get("load", {}).get("resting_vram_gb")
    base_tps = base.get("throughput", {}).get("decode_tokens_per_sec_median")

    results = []
    for name, v in record["variants"].items():
        if v.get("status") != "ok":
            results.append({"variant": name, "status": v.get("status")})
            continue
        vram = v["load"]["resting_vram_gb"]
        tps = v["throughput"]["decode_tokens_per_sec_median"]
        results.append({
            "variant": name,
            "resting_vram_gb": vram,
            "unquantised_gb": v["load"]["unquantised_gb"],
            "lm_head_quantised": (v["load"].get("lm_head") or {}).get("quantised"),
            "decode_tokens_per_sec": tps,
            "vram_vs_default": (round(1 - vram/base_vram, 4)
                                if base_vram else None),
            "speed_vs_default": (round(tps/base_tps, 3)
                                 if base_tps else None),
        })

    working = [r for r in results
               if r.get("resting_vram_gb") and r["resting_vram_gb"] < 5.5]

    if working:
        best = min(working, key=lambda r: r["resting_vram_gb"])
        conclusion = (
            f"bitsandbytes CAN do this. Variant '{best['variant']}' loads at "
            f"{best['resting_vram_gb']} GB, a {best['vram_vs_default']:.1%} "
            f"reduction, at {best['speed_vs_default']:.2f}x the baseline decode "
            f"speed. This is a configuration change, not an engineering project."
        )
        next_step = "Proceed to entropy coding on top of this configuration."
    else:
        conclusion = (
            "bitsandbytes did NOT shrink the model. The embedding table and "
            "LM head remain unquantised regardless of llm_int8_skip_modules. "
            "The 2.84 GB saving is real (quality validated) but not reachable "
            "through this backend."
        )
        next_step = (
            "Check GGUF/llama.cpp, which quantises token embeddings and the "
            "output layer by default, before considering a custom packing "
            "path. Building a backend to save 2.84 GB is a poor trade if an "
            "existing one already does it."
        )

    return {
        "results": results,
        "conclusion": conclusion,
        "next_step": next_step,
    }
