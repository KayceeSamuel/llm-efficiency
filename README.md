# LLM Inference Efficiency Harness

Measures memory, latency, and quality for inference-time efficiency
experiments on dense and hybrid decoder models. Companion to the
experiment design reference.

## What it does

Three jobs, deliberately separated so one can be replaced without
touching the others:

| Module | Job |
|---|---|
| `profiler.py` | Memory and latency. Peak memory during generation, KV-cache measured directly from the cache object, prefill and decode timed separately, memory-vs-context curves. |
| `quality.py` | Quality via EleutherAI `lm-eval`. Standard battery, long-context, domain — reported separately, never averaged. |
| `runner.py` | Orchestration. Loads, profiles, evaluates, writes a result record with full config and environment provenance. |
| `analyse.py` | Reads records, produces comparisons and gate verdicts. Never needs a GPU. |
| `config.py` | Experiment definitions. A run is fully described by a `RunConfig`. |
| `experiments.py` | Predefined condition sets matching the design doc register. |

## Install

```bash
pip install -q transformers accelerate bitsandbytes
pip install -q lm-eval

# Required for Qwen3.5/3.6 hybrid models. Without these the Gated DeltaNet
# layers fall back to a slow torch path and all timing is unrepresentative.
pip install -q flash-linear-attention causal-conv1d

# Optional, for attn_impl="flash_attention_2". Needs Ampere or newer.
pip install -q flash-attn --no-build-isolation
```

## Use

```python
from harness.experiments import smoke, tier1_memory_baseline, tier2_backend_sweep
from harness.runner import run_matrix
from harness.analyse import print_summary, compare

# Always start here. Validates wiring in a few minutes.
run_matrix(smoke())
print_summary()

# Then the real work.
run_matrix(tier1_memory_baseline())
run_matrix(tier2_backend_sweep())

# Head-to-head with the pre-registered quality gate.
compare("T2-05a-<fingerprint>", "T2-05c-<fingerprint>")
```

Results land in `results/` — one JSON per run plus an append-only
`ledger.jsonl`.

## Design decisions worth knowing

**Nothing is implicit.** Attention implementation is always set
explicitly, never left to the library default. The current GPRA pipeline
never chose one, which means nobody knows which kernel it has been
running.

**Profile before evaluating.** `lm-eval` allocates substantially on
long-context tasks. Running it first would pollute every memory number.

**The live model object is passed to `lm-eval`, not a model path.**
Reloading inside `lm-eval` would use its own loading path and could apply
a different quantization or attention config than the one profiled — so
the thing evaluated would not be the thing measured.

**KV-cache is measured, not calculated.** Read directly off the returned
cache object. This handles GQA, MQA, and hybrid layouts where only some
layers cache, without assuming a formula that may not hold.

**OOM is a result, not a crash.** `profile_context_scaling` records the
context length where a config stops fitting and moves on.

**Config fingerprinting.** Two runs with the same fingerprint are
directly comparable. Labels and notes are excluded from the hash;
anything that affects results is included.

**Environment is captured per run.** A `bitsandbytes` or `transformers`
version bump moves numbers with no code change.

## Cautions

**Latency does not transfer between GPUs.** Decode is
memory-bandwidth-bound. An L4 runs ~300 GB/s against an A100's ~1555
GB/s. Memory measurements transfer; throughput conclusions do not.
Quantization specifically tends to look *better* on bandwidth-starved
cards, because it reduces bytes moved per token. Confirm surviving
methods on the target card.

**fp16 on a 22GB L4 is tight.** A 9B model at fp16 is roughly 18GB. It
loads, but leaves little room for KV-cache at long context. The predefined
fp16 condition is short-context only by design.

**FlashAttention-3 is Hopper-only** and cannot be tested on L4. Qwen's
own FlashQLA kernels were benchmarked on Hopper. Record this as a scope
limit rather than discovering it mid-experiment.

**Two bases, not one.** Qwen3.5/3.6 are hybrid, not conventional dense —
most layers are linear-attention. A result measured only on Qwen may be a
property of that architecture rather than of dense serving generally.
`generalisation_check()` runs the same condition on a conventional dense
base for exactly this reason.

**The domain eval is a stub.** Left unimplemented rather than filled with
invented cases — a domain gate built from made-up clinical items produces
a number that looks rigorous and means nothing. Populate from real cases
with known-correct reasoning before wiring it in.

## Provenance labels

Raw records are `measured`. Anything derived in `analyse.py` is
`computed`. Figures reasoned from architecture specs without measurement
are `estimated` and do not appear in records at all — they belong in the
design doc, clearly marked.
