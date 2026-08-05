"""
loader.py — model loading with every choice made explicit.

The design doc identifies three gaps in the current GPRA pipeline:
attention implementation never chosen, peak memory never measured, and
the linear-attention fast path silently unavailable. This module closes
the first and third; profiler.py closes the second.
"""

import gc
import time
import warnings
from typing import Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import RunConfig, BACKENDS


def free_gpu():
    """Release any previously loaded model. Call between runs in a session."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def check_fast_path_available() -> Dict[str, bool]:
    """
    Qwen3.5/3.6 hybrid models need flash-linear-attention and causal-conv1d
    for the Gated DeltaNet fast path. Without them transformers silently
    falls back to a slow torch implementation -- it warns at load time and
    is easy to miss. Detecting it up front means a slow run is explained
    rather than mysterious.
    """
    status = {}
    for mod in ["fla", "causal_conv1d", "flash_attn"]:
        try:
            __import__(mod)
            status[mod] = True
        except Exception:
            status[mod] = False
    return status


def _build_quant_config(backend: str, compute_dtype):
    spec = BACKENDS[backend]

    if spec["kind"] == "none":
        return None

    if spec["kind"] == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        if spec["bits"] == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        if spec["bits"] == 8:
            return BitsAndBytesConfig(load_in_8bit=True)

    # prequantized: config travels with the checkpoint
    return None


def load(cfg: RunConfig) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load model + tokenizer for a run config.

    Returns (model, tokenizer, load_info) where load_info records what
    actually happened -- resting footprint, load time, fast-path status.
    """
    free_gpu()

    info: Dict[str, Any] = {}
    info["fast_path"] = check_fast_path_available()

    if not info["fast_path"].get("fla", False):
        warnings.warn(
            "flash-linear-attention not installed. On Qwen3.5/3.6 hybrid "
            "models the Gated DeltaNet layers will run on a slow torch "
            "fallback. Timing results will not reflect a properly "
            "configured deployment. Install: pip install flash-linear-attention causal-conv1d"
        )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    info["compute_dtype"] = str(compute_dtype)

    model_id = cfg.resolved_model_id()
    quant_config = _build_quant_config(cfg.backend, compute_dtype)

    # Guard: flash_attention_2 needs Ampere or newer. Requesting it on older
    # hardware fails at load with an opaque error; better to say so here.
    attn_impl = cfg.attn_impl
    if attn_impl == "flash_attention_2":
        if not info["fast_path"].get("flash_attn", False):
            raise RuntimeError(
                "attn_impl='flash_attention_2' requested but flash_attn is "
                "not installed. Install it or use 'sdpa'."
            )
        props = torch.cuda.get_device_properties(0)
        if props.major < 8:
            raise RuntimeError(
                f"flash_attention_2 requires compute capability 8.0+ "
                f"(Ampere). This GPU is {props.major}.{props.minor}."
            )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: Dict[str, Any] = dict(
        # Explicit single-GPU placement. accelerate's "auto" heuristic has
        # been observed offloading to CPU/disk despite ample free VRAM,
        # which silently destroys both memory and latency measurements.
        device_map={"": 0},
        trust_remote_code=True,
        # Without this, loading can transiently hold both the original
        # BF16 tensors and their quantized replacements simultaneously,
        # roughly doubling peak load memory and causing OOM at ~96% loaded.
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )

    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = compute_dtype

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    info["load_seconds"] = round(time.perf_counter() - t0, 2)

    # Resting footprint: weights only, before any generation.
    # This is the analogue of the 17.67 GB figure from the GPRA logs --
    # a floor, explicitly NOT a peak.
    if torch.cuda.is_available():
        info["weights_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)

    # Confirm what actually got used, rather than what was requested.
    info["attn_impl_requested"] = attn_impl
    try:
        info["attn_impl_actual"] = model.config._attn_implementation
    except Exception:
        info["attn_impl_actual"] = "unknown"

    info["num_parameters"] = sum(p.numel() for p in model.parameters())

    try:
        cfgm = model.config
        text_cfg = getattr(cfgm, "text_config", cfgm)
        info["num_layers"] = getattr(text_cfg, "num_hidden_layers", None)
        info["hidden_size"] = getattr(text_cfg, "hidden_size", None)
    except Exception:
        pass

    return model, tok, info
