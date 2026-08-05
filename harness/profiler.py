"""
profiler.py — memory and latency instrumentation.

This closes the single most important gap identified in the design doc:
the GPRA pipeline prints memory once, after load, and never during
generation. 17.67 GB is a floor, not a peak, and the difference is
exactly where KV-cache and the quadratic attention term live.

Prefill and decode are timed separately because they are different
regimes: prefill is compute-bound and processes the whole prompt at once,
decode is memory-bandwidth-bound and produces one token at a time. A
technique can help one and hurt the other; a single averaged
tokens/sec number hides that.
"""

import time
import statistics
from typing import Dict, Any, List, Optional

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024**3, 3)


def _current_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / 1024**3, 3)


def _reserved_gb() -> float:
    """
    Reserved is what the caching allocator holds from the driver, which is
    what actually determines whether you OOM. Allocated can look fine while
    reserved is at the ceiling due to fragmentation.
    """
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_reserved() / 1024**3, 3)


def make_synthetic_prompt(tok, target_tokens: int) -> str:
    """
    Build a prompt of approximately target_tokens. Used for memory scaling
    curves where content is irrelevant and only length matters.

    Deliberately repetitive: this measures memory/latency as a function of
    sequence length, NOT quality. Never score model outputs on these.
    """
    unit = ("The mitochondrial respiratory chain comprises five complexes "
            "embedded in the inner membrane, each contributing to oxidative "
            "phosphorylation through electron transfer and proton pumping. ")
    unit_len = len(tok(unit)["input_ids"])
    reps = max(1, target_tokens // max(1, unit_len))
    text = unit * reps

    # Trim to length rather than overshooting.
    ids = tok(text)["input_ids"][:target_tokens]
    return tok.decode(ids)


def measure_prefill(model, tok, prompt: str) -> Dict[str, Any]:
    """
    Single forward pass over the prompt, no generation. Isolates the
    prompt-processing cost -- where the quadratic attention term bites.
    """
    _sync()
    torch.cuda.reset_peak_memory_stats()
    baseline = _current_gb()

    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_tokens = enc["input_ids"].shape[1]

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    _sync()
    elapsed = time.perf_counter() - t0

    peak = _peak_gb()

    # KV-cache size measured directly from the returned cache, rather than
    # inferred from architecture specs. Handles GQA/MQA and hybrid layouts
    # (where only some layers cache) without assuming a formula.
    kv_bytes = 0
    n_cached_layers = 0
    try:
        cache = out.past_key_values
        layers = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        for layer in layers:
            if layer is None:
                continue
            counted = False
            for t in layer:
                if torch.is_tensor(t) and t.numel() > 0:
                    kv_bytes += t.numel() * t.element_size()
                    counted = True
            if counted:
                n_cached_layers += 1
    except Exception:
        pass

    del out
    torch.cuda.empty_cache()

    return {
        "prompt_tokens": n_tokens,
        "prefill_seconds": round(elapsed, 4),
        "prefill_tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else None,
        "peak_gb": peak,
        "peak_above_weights_gb": round(peak - baseline, 3),
        "kv_cache_gb": round(kv_bytes / 1024**3, 4),
        "kv_bytes_per_token": int(kv_bytes / n_tokens) if n_tokens else None,
        "layers_with_cache": n_cached_layers,
    }


def measure_generation(
    model,
    tok,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
    repetition_penalty: float = 1.1,
) -> Dict[str, Any]:
    """
    Full generate() call with peak memory captured across the whole run.
    This is the number the GPRA notebook never recorded.
    """
    _sync()
    torch.cuda.reset_peak_memory_stats()
    baseline = _current_gb()

    enc = tok(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    _sync()
    elapsed = time.perf_counter() - t0

    new_tokens = out.shape[1] - input_len
    text = tok.decode(out[0][input_len:], skip_special_tokens=True)

    return {
        "prompt_tokens": input_len,
        "generated_tokens": int(new_tokens),
        "total_seconds": round(elapsed, 4),
        "tokens_per_sec": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
        "peak_gb": _peak_gb(),
        "peak_above_weights_gb": round(_peak_gb() - baseline, 3),
        "reserved_gb": _reserved_gb(),
        "sample_output": text[:600],
    }


def profile_context_scaling(
    model,
    tok,
    context_lengths: List[int],
    max_new_tokens: int = 64,
    warmup: bool = True,
) -> List[Dict[str, Any]]:
    """
    Memory and latency as a function of context length.

    This is the curve that matters for the research question. The design
    doc's computed figures predict KV-cache grows linearly while the
    attention score matrix grows quadratically -- this measures whether
    that shows up in practice, and at what length it starts to dominate.

    OOM at a given length is recorded, not raised: knowing where a config
    stops fitting is a result.
    """
    results = []

    if warmup:
        # First call triggers kernel compilation; excluding it prevents
        # a large one-off cost being attributed to the first context length.
        try:
            p = make_synthetic_prompt(tok, 128)
            measure_generation(model, tok, p, max_new_tokens=8)
        except Exception:
            pass

    for n in context_lengths:
        torch.cuda.empty_cache()
        row: Dict[str, Any] = {"target_context_tokens": n}
        try:
            prompt = make_synthetic_prompt(tok, n)
            row.update(measure_prefill(model, tok, prompt))
            gen = measure_generation(
                model, tok, prompt, max_new_tokens=max_new_tokens
            )
            row["generation"] = {
                k: v for k, v in gen.items() if k != "sample_output"
            }
            row["status"] = "ok"
        except torch.cuda.OutOfMemoryError:
            row["status"] = "oom"
            torch.cuda.empty_cache()
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            torch.cuda.empty_cache()

        results.append(row)

        if row["status"] == "oom":
            # Longer contexts will also OOM; stop rather than burn time.
            break

    return results


def latency_repeats(
    model, tok, prompt: str, max_new_tokens: int, repeats: int = 3
) -> Dict[str, Any]:
    """
    Repeat a generation to get variance. A single latency number is not
    a measurement -- thermal state, other processes, and allocator behaviour
    all move it. Report median and spread.
    """
    runs = []
    for _ in range(repeats):
        runs.append(measure_generation(model, tok, prompt, max_new_tokens))

    tps = [r["tokens_per_sec"] for r in runs if r["tokens_per_sec"]]
    secs = [r["total_seconds"] for r in runs]

    return {
        "repeats": repeats,
        "tokens_per_sec_median": round(statistics.median(tps), 2) if tps else None,
        "tokens_per_sec_min": round(min(tps), 2) if tps else None,
        "tokens_per_sec_max": round(max(tps), 2) if tps else None,
        "seconds_median": round(statistics.median(secs), 4),
        "peak_gb_max": max(r["peak_gb"] for r in runs),
    }
