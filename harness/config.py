"""
config.py — experiment definitions and run identity.

Everything that defines *what* an experiment is lives here, so a run is
fully described by a config object and nothing is implicit. This is what
makes runs comparable: if two runs differ, the difference is visible in
the config, not buried in a notebook cell someone edited.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import hashlib
import json
import platform
import subprocess


# ---------------------------------------------------------------------------
# Model bases. Two bases by design -- see design doc section 1.1.
# Qwen is hybrid (GDN + full attention); Llama/Mistral are conventional dense.
# A result that only holds on Qwen is architecture-specific, not a technique.
# ---------------------------------------------------------------------------

BASES = {
    "qwen-9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "family": "qwen",
        "architecture": "hybrid-gdn-attention",
        "notes": "Primary development base. Hybrid: linear-attention majority.",
    },
    "llama-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "architecture": "dense-full-attention",
        "notes": "Generalisation control. Conventional dense transformer.",
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "architecture": "dense-full-attention",
        "notes": "Alternative generalisation control.",
    },
}


# ---------------------------------------------------------------------------
# Quantization backends. 'none' is the fp16/bf16 reference.
# ---------------------------------------------------------------------------

BACKENDS = {
    "fp16": {
        "kind": "none",
        "bits": 16,
        "desc": "Unquantized reference. On a 22GB L4 this is tight for a 9B "
                "model (~18GB) -- short-context only. Run long-context fp16 "
                "on a larger card.",
    },
    "bnb-nf4": {
        "kind": "bitsandbytes",
        "bits": 4,
        "desc": "Current GPRA production config. NF4 + double quant.",
    },
    "bnb-int8": {
        "kind": "bitsandbytes",
        "bits": 8,
        "desc": "8-bit reference point between fp16 and 4-bit.",
    },
    "gptq": {
        "kind": "prequantized",
        "bits": 4,
        "desc": "Requires a pre-quantized checkpoint. Set model_id override.",
    },
    "awq": {
        "kind": "prequantized",
        "bits": 4,
        "desc": "Requires a pre-quantized checkpoint. Set model_id override.",
    },
    "nvfp4": {
        "kind": "prequantized",
        "bits": 4,
        "desc": "Published NVFP4 checkpoint. Cheap comparison, no implementation.",
    },
}


# ---------------------------------------------------------------------------
# Attention implementations. Never leave this implicit -- the whole point of
# design doc gap #3 is that the current pipeline never chose one.
# ---------------------------------------------------------------------------

ATTN_IMPLS = ["eager", "sdpa", "flash_attention_2"]


@dataclass
class RunConfig:
    """A single experimental condition. Fully determines a run."""

    # identity
    experiment_id: str                    # e.g. "T2-05" from the register
    label: str                            # human-readable, e.g. "nf4 baseline"

    # what we're running
    base: str = "qwen-9b"                 # key into BASES
    backend: str = "bnb-nf4"              # key into BACKENDS
    attn_impl: str = "sdpa"               # explicit, never default
    model_id_override: Optional[str] = None   # for pre-quantized checkpoints

    # generation controls -- held fixed across compared runs or the
    # comparison is invalid (design doc 5.1)
    max_new_tokens: int = 512
    do_sample: bool = False               # greedy: deterministic
    repetition_penalty: float = 1.1
    batch_size: int = 1

    # KV cache
    kv_cache_dtype: Optional[str] = None  # None = default; "int8"/"int4" if supported

    # what to measure
    run_perf: bool = True                 # memory + latency wrapper
    run_standard_eval: bool = True        # lm-eval standard battery
    run_longcontext_eval: bool = False    # expensive; opt in
    run_domain_eval: bool = False         # GPRA clinical set

    # eval task selection
    standard_tasks: List[str] = field(default_factory=lambda: [
        # The battery the quantization literature has converged on.
        "arc_easy", "arc_challenge", "winogrande", "hellaswag", "piqa",
        # Reasoning, cheap enough for the iteration loop.
        "gsm8k",
    ])
    longcontext_tasks: List[str] = field(default_factory=lambda: [
        # Representative of the ACTUAL workload (15k-19k token generations),
        # unlike the short-prompt standard battery.
        "niah_single_1", "niah_multikey_1",
    ])
    perplexity_tasks: List[str] = field(default_factory=lambda: [
        "wikitext",
    ])

    num_fewshot: int = 0
    eval_limit: Optional[int] = None      # cap items for fast smoke runs

    # context lengths at which to profile memory/latency
    profile_context_lengths: List[int] = field(
        default_factory=lambda: [512, 2048, 8192, 16384]
    )

    # free-form
    notes: str = ""

    def resolved_model_id(self) -> str:
        if self.model_id_override:
            return self.model_id_override
        return BASES[self.base]["model_id"]

    def fingerprint(self) -> str:
        """
        Stable hash of the experimental condition. Two runs with the same
        fingerprint should be directly comparable; differing fingerprints
        mean something changed, deliberately or otherwise.
        """
        payload = {
            k: v for k, v in asdict(self).items()
            if k not in ("label", "notes", "experiment_id")
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolved_model_id"] = self.resolved_model_id()
        d["fingerprint"] = self.fingerprint()
        d["architecture"] = BASES[self.base]["architecture"]
        d["family"] = BASES[self.base]["family"]
        return d


def run_stamp() -> str:
    """
    Short UTC timestamp appended to every result filename.

    Without this, re-running an experiment with an unchanged config produces
    the same filename and SILENTLY OVERWRITES the previous record. That has
    already cost one run in this project. Results are expensive to produce and
    cheap to store, so every execution now gets its own file and nothing is
    ever destroyed by a rerun.

    The config fingerprint still identifies which runs are comparable; this
    only distinguishes separate executions of the same condition.
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def capture_environment() -> Dict[str, Any]:
    """
    Environment capture. Library versions silently change results --
    a bitsandbytes or transformers bump can move numbers with no code change,
    so this is recorded per run, not assumed constant.
    """
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    for mod in ["torch", "transformers", "bitsandbytes", "accelerate",
                "lm_eval", "flash_attn", "vllm"]:
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = None

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
            env["cuda"] = torch.version.cuda
            env["gpu_capability"] = f"{props.major}.{props.minor}"
            # FA2 needs Ampere+; FA3 is Hopper-only. Recording this prevents
            # silently comparing runs that used different kernels.
            env["supports_flash_attn_2"] = props.major >= 8
            env["is_hopper"] = props.major == 9
        else:
            env["gpu_name"] = None
    except Exception as e:
        env["gpu_error"] = str(e)

    try:
        env["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        env["nvidia_driver"] = None

    return env
