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


def measure_cache(cache, n_tokens: int) -> Dict[str, Any]:
    """
    Measure cache memory directly off the cache object.

    Written against the transformers v5 Cache API, where `cache.layers` is a
    list of per-layer objects and `to_legacy_cache()` no longer exists. On a
    hybrid model the list is heterogeneous:

      DynamicLayer         -> full attention. Has .keys / .values, both
                              growing with sequence length.
      LinearAttentionLayer -> Gated DeltaNet. Has .recurrent_states and
                              .conv_states, both FIXED SIZE regardless of
                              context length.

    Those two are reported separately, because conflating them is precisely
    the mistake the hybrid architecture invites: the KV total scales with
    context and the GDN state does not, so a single combined number would
    hide the structural difference the research question depends on.

    Raises rather than returning zeros on an unrecognised cache type. The
    previous version swallowed the exception and reported 0.0 GB, which is a
    plausible-looking wrong answer -- the worst possible failure mode in a
    measurement harness.
    """
    if cache is None:
        raise ValueError("No cache returned; was use_cache=True set?")

    layers = getattr(cache, "layers", None)
    if layers is None:
        # Pre-v5 tuple-of-tuples fallback.
        if isinstance(cache, (list, tuple)):
            layers = cache
        else:
            raise TypeError(
                f"Unrecognised cache type {type(cache).__name__}: no .layers "
                f"attribute and not a sequence. Cache extraction must be "
                f"updated for this transformers version."
            )

    kv_bytes = 0            # attention layers: scales with context
    recurrent_bytes = 0     # GDN layers: fixed size
    conv_bytes = 0          # GDN layers: fixed size
    n_attn_layers = 0
    n_linear_layers = 0
    layer_types: Dict[str, int] = {}

    def _nbytes(t) -> int:
        return t.numel() * t.element_size() if torch.is_tensor(t) else 0

    for layer in layers:
        if layer is None:
            continue

        tname = type(layer).__name__
        layer_types[tname] = layer_types.get(tname, 0) + 1

        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        rec = getattr(layer, "recurrent_states", None)
        conv = getattr(layer, "conv_states", None)

        if keys is not None or values is not None:
            b = _nbytes(keys) + _nbytes(values)
            if b > 0:
                kv_bytes += b
                n_attn_layers += 1
        elif rec is not None or conv is not None:
            rb, cb = _nbytes(rec), _nbytes(conv)
            if rb + cb > 0:
                recurrent_bytes += rb
                conv_bytes += cb
                n_linear_layers += 1
        elif isinstance(layer, (list, tuple)):
            b = sum(_nbytes(t) for t in layer)
            if b > 0:
                kv_bytes += b
                n_attn_layers += 1

    state_bytes = recurrent_bytes + conv_bytes

    return {
        # Attention layers -- grows linearly with context.
        "kv_cache_gb": round(kv_bytes / 1024**3, 5),
        "kv_bytes_per_token": int(kv_bytes / n_tokens) if n_tokens else None,
        "kv_layers": n_attn_layers,

        # GDN layers -- constant in context length. Answers the design doc's
        # open item on GDN state size, which is not publicly specified.
        "gdn_state_gb": round(state_bytes / 1024**3, 5),
        "gdn_recurrent_gb": round(recurrent_bytes / 1024**3, 5),
        "gdn_conv_gb": round(conv_bytes / 1024**3, 5),
        "gdn_layers": n_linear_layers,

        "total_cache_gb": round((kv_bytes + state_bytes) / 1024**3, 5),
        "layer_types": layer_types,
    }


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

    cache_stats = measure_cache(out.past_key_values, n_tokens)

    del out
    torch.cuda.empty_cache()

    result = {
        "prompt_tokens": n_tokens,
        "prefill_seconds": round(elapsed, 4),
        "prefill_tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else None,
        "peak_gb": peak,
        "peak_above_weights_gb": round(peak - baseline, 3),
    }
    result.update(cache_stats)
    return result


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
