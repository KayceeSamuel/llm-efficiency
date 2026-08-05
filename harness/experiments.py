"""
experiments.py — predefined condition sets matching the design doc register.

Each function returns a list of RunConfig. Import and pass to run_matrix.

Hardware note: these are sized for a 22GB L4. On that card a 9B model at
fp16 is roughly 18GB, which fits but leaves little room for KV-cache at
long context. fp16 long-context conditions are therefore excluded here and
should be run on a larger card.

Wider caution: L4 and A100 differ ~5x in memory bandwidth (~300 GB/s vs
~1555 GB/s). Decode is bandwidth-bound, so MEMORY results transfer between
cards but LATENCY results do not. Quantization in particular tends to look
better on bandwidth-starved hardware. Confirm surviving methods on the
target card before drawing throughput conclusions.
"""

from typing import List
from .config import RunConfig


# ---------------------------------------------------------------------------
# Tier 0 — smoke test. Run this first, always.
# ---------------------------------------------------------------------------

def smoke(base: str = "qwen-9b") -> List[RunConfig]:
    """
    Minimal end-to-end validation: does the pipeline run at all?
    Small eval limit, short contexts. Expect a few minutes, not hours.
    """
    return [
        RunConfig(
            experiment_id="T0-smoke",
            label="smoke test, nf4, sdpa",
            base=base,
            backend="bnb-nf4",
            attn_impl="sdpa",
            max_new_tokens=64,
            profile_context_lengths=[256, 1024],
            standard_tasks=["arc_easy"],
            eval_limit=20,
            run_longcontext_eval=False,
            notes="Validates loading, profiling, and lm-eval wiring only. "
                  "Scores from this run are not meaningful.",
        )
    ]


# ---------------------------------------------------------------------------
# Tier 1 — cheap measurement. Design doc experiments 1-4.
# ---------------------------------------------------------------------------

def tier1_memory_baseline(base: str = "qwen-9b") -> List[RunConfig]:
    """
    Design doc experiment 4: real peak-memory instrumentation.

    Establishes the memory-vs-context curve the whole project reasons about.
    Predictions to test: KV-cache linear in context length, attention score
    memory quadratic, GDN state constant. Confirm or refute against the
    measured curve rather than assuming the computed figures hold.
    """
    return [
        RunConfig(
            experiment_id="T1-04",
            label=f"memory baseline, {impl}",
            base=base,
            backend="bnb-nf4",
            attn_impl=impl,
            max_new_tokens=128,
            profile_context_lengths=[512, 2048, 8192, 16384, 32768],
            run_standard_eval=False,   # this run is about memory only
            notes="Attention impl varied deliberately: eager materialises the "
                  "full n x n score matrix, sdpa/FA2 do not. The gap between "
                  "these curves IS the quadratic term, measured directly.",
        )
        for impl in ["eager", "sdpa"]
    ]


# ---------------------------------------------------------------------------
# Tier 2 — established baselines. Design doc experiments 5-8.
# ---------------------------------------------------------------------------

def tier2_backend_sweep(base: str = "qwen-9b") -> List[RunConfig]:
    """
    Design doc experiment 5: the competitive floor.

    Any novel method must beat these, not merely beat fp16. fp16 is
    included for the quality reference point but restricted to short
    context on a 22GB card.
    """
    configs = [
        RunConfig(
            experiment_id="T2-05a",
            label="fp16 reference (short context only)",
            base=base,
            backend="fp16",
            attn_impl="sdpa",
            profile_context_lengths=[512, 2048],
            notes="~18GB on a 9B model. Fits a 22GB L4 but with little "
                  "headroom. Long context excluded here by design.",
        ),
        RunConfig(
            experiment_id="T2-05b",
            label="bnb int8",
            base=base,
            backend="bnb-int8",
            attn_impl="sdpa",
        ),
        RunConfig(
            experiment_id="T2-05c",
            label="bnb nf4 (current GPRA config)",
            base=base,
            backend="bnb-nf4",
            attn_impl="sdpa",
            notes="Matches the production GPRA loading configuration.",
        ),
    ]
    return configs


def tier2_attention_impl(base: str = "qwen-9b") -> List[RunConfig]:
    """
    Design doc experiment 8: attention kernel selection.

    Closes the gap where the production pipeline never chose an
    implementation. Run AFTER installing flash-linear-attention and
    causal-conv1d, since those change the baseline for the GDN layers and
    any timing gathered without them is not representative.
    """
    return [
        RunConfig(
            experiment_id="T2-08",
            label=f"attention impl: {impl}",
            base=base,
            backend="bnb-nf4",
            attn_impl=impl,
            profile_context_lengths=[512, 2048, 8192, 16384],
            notes="Requires flash_attn installed for flash_attention_2. "
                  "FA2 needs Ampere+; FA3 is Hopper-only and untestable on L4.",
        )
        for impl in ["eager", "sdpa", "flash_attention_2"]
    ]


# ---------------------------------------------------------------------------
# Cross-base generalisation
# ---------------------------------------------------------------------------

def generalisation_check(backend: str = "bnb-nf4") -> List[RunConfig]:
    """
    Same condition across a hybrid and a conventional dense base.

    This is the split that separates a technique from an artifact. Qwen3.5
    is hybrid (majority linear-attention layers); Llama-3.1 is conventional
    dense. A result present on one and absent on the other is
    architecture-specific and must be reported as such.
    """
    return [
        RunConfig(
            experiment_id=f"GEN-{base}",
            label=f"generalisation: {base} @ {backend}",
            base=base,
            backend=backend,
            attn_impl="sdpa",
            profile_context_lengths=[512, 2048, 8192, 16384],
        )
        for base in ["qwen-9b", "llama-8b"]
    ]
