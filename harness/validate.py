"""
validate.py — the quality gate. Does reconstruction error predict model quality?

THE PROBLEM THIS EXISTS TO SOLVE:

Every result in this project so far is reconstruction error measured on
sampled weight matrices. Not one benchmark score. The compression frontier
says int4+entropy+rot is 11.8% smaller and 6.8% more accurate than plain int4
-- but "more accurate" there means lower Frobenius error on 16 matrices, not
better answers on any task.

That distinction is not pedantic. The quantisation literature contains cases
where small reconstruction error still breaks a model (because the error lands
on weights that matter disproportionately) and cases where large error is
harmless. If benchmark quality does not track reconstruction error on THIS
architecture, the frontier is measuring the wrong quantity and every
conclusion built on it is unsafe.

HOW IT WORKS:

Load unquantised, then apply quantise-dequantise to every weight IN PLACE.
The tensors stay in fp16 storage, so no memory is saved -- the point is to
introduce exactly the error a real quantiser would, then measure what it does
to benchmark scores. Size and quality are separable questions and this module
answers only the second.

WHAT IT ALSO SETTLES:

The baseline throughout this project has been uniform int4. Production runs
bitsandbytes NF4, whose codebook is fitted to a normal distribution and
therefore already compensates for part of the outlier problem rotation
targets. If rotation's benefit is largely redundant with NF4, experiment 7's
promotion was overstated. The nf4 scheme below simulates that codebook so the
comparison is like for like.
"""

import copy
import gc
import math
from typing import Dict, Any, List, Optional, Tuple

import torch

from .rotation import (
    quantize_dequantize, relative_error,
    kronecker_rotation_factors, apply_kronecker_rotation,
)


# ---------------------------------------------------------------------------
# NF4 codebook
# ---------------------------------------------------------------------------

# The 16 NF4 levels, as used by bitsandbytes. Spaced by the quantiles of a
# normal distribution rather than uniformly, so more levels sit near zero
# where weights are dense and fewer at the extremes where they are sparse.
# This is why NF4 outperforms uniform int4 at the same bit width -- and why
# it may already capture part of what rotation provides.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


def quantize_dequantize_nf4(W: torch.Tensor,
                            block_size: int = 64) -> torch.Tensor:
    """
    Simulate bitsandbytes NF4: blockwise absmax scaling onto the NF4 codebook.

    Each block is normalised to [-1, 1] by its absolute maximum, then each
    value snaps to the nearest codebook level. Double quantisation of the
    scales is not simulated -- it affects stored size, not reconstruction
    error, which is what this module measures.
    """
    orig_shape = W.shape
    flat = W.detach().flatten().float()
    device = flat.device

    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=device)])

    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    absmax = torch.where(absmax == 0, torch.full_like(absmax, 1e-12), absmax)
    normed = blocks / absmax

    levels = NF4_LEVELS.to(device)
    # Nearest-level assignment via bucket boundaries (midpoints between levels).
    boundaries = (levels[:-1] + levels[1:]) / 2
    idx = torch.bucketize(normed.contiguous(), boundaries)
    deq = (levels[idx] * absmax).flatten()

    if pad:
        deq = deq[:-pad]
    return deq.view(orig_shape)


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

SCHEMES = {
    "fp16": {"desc": "unmodified reference", "bits": 16},
    "int4": {"desc": "uniform int4, blockwise absmax", "bits": 4},
    "int3": {"desc": "uniform int3", "bits": 3},
    "int2": {"desc": "uniform int2", "bits": 2},
    "nf4": {"desc": "NF4 codebook (simulates bitsandbytes production path)",
            "bits": 4},
    "int4+rot": {"desc": "uniform int4 with Kronecker rotation", "bits": 4},
    "int3+rot": {"desc": "uniform int3 with Kronecker rotation", "bits": 3},
    "nf4+rot": {"desc": "NF4 with rotation -- tests whether rotation adds "
                        "anything on top of the normal-fitted codebook",
                "bits": 4},
}


def _apply_scheme_to_matrix(W: torch.Tensor, scheme: str,
                            block_size: int = 64) -> torch.Tensor:
    """
    Return the reconstructed weight for one matrix under a scheme.

    Rotation is applied and inverted inside this function, so the returned
    tensor is directly substitutable for the original -- the model sees a
    weight with quantisation error in it, and nothing about the forward pass
    changes. That is what makes the in-place approach valid without
    implementing rotation absorption.
    """
    if scheme == "fp16":
        return W

    rotate = scheme.endswith("+rot")
    core = scheme[:-4] if rotate else scheme

    Wf = W.detach().float()
    rows, cols = Wf.shape

    if rotate:
        try:
            Hr, Qr, kr, mr = kronecker_rotation_factors(rows, device=Wf.device)
            Hc, Qc, kc, mc = kronecker_rotation_factors(cols, device=Wf.device)
        except ValueError:
            # No usable power-of-2 factor; fall back to unrotated rather than
            # silently skipping the matrix, which would leave it at fp16 and
            # flatter the scheme.
            rotate = False

    X = Wf
    if rotate:
        X = apply_kronecker_rotation(X, Hr, Qr, kr, mr, dim=0)
        X = apply_kronecker_rotation(X, Hc, Qc, kc, mc, dim=1)

    if core == "nf4":
        Xq = quantize_dequantize_nf4(X, block_size)
    else:
        bits = int(core.replace("int", ""))
        Xq = quantize_dequantize(X, bits, block_size)

    if rotate:
        Xq = apply_kronecker_rotation(Xq, Hc.T, Qc.T, kc, mc, dim=1)
        Xq = apply_kronecker_rotation(Xq, Hr.T, Qr.T, kr, mr, dim=0)

    return Xq.to(dtype=W.dtype)


# ---------------------------------------------------------------------------
# In-place model modification
# ---------------------------------------------------------------------------

def apply_scheme_in_place(model, scheme: str, block_size: int = 64,
                          skip_embeddings: bool = True,
                          min_numel: int = 4096) -> Dict[str, Any]:
    """
    Overwrite every eligible 2-D weight with its quantise-dequantise
    reconstruction.

    IRREVERSIBLE on the loaded model. The caller must reload between schemes;
    run_quality_validation does this.

    skip_embeddings defaults True to match production behaviour, where
    bitsandbytes leaves the embedding table and LM head unquantised. Setting
    it False directly tests the ~1.7 GB embedding-quantisation hypothesis --
    the single largest untested memory target identified so far.
    """
    from .weights import _get_layer_list

    stats = {"matrices_modified": 0, "params_modified": 0,
             "mean_rel_error": 0.0, "skipped": 0}
    errors = []

    layers = _get_layer_list(model)
    with torch.no_grad():
        for layer in layers:
            for name, module in layer.named_modules():
                W = getattr(module, "weight", None)
                if W is None or not torch.is_tensor(W) or W.dim() != 2:
                    continue
                if W.numel() < min_numel:
                    stats["skipped"] += 1
                    continue

                orig = W.detach().float().clone()
                new = _apply_scheme_to_matrix(W, scheme, block_size)
                errors.append(relative_error(orig, new.float()))
                W.copy_(new)

                stats["matrices_modified"] += 1
                stats["params_modified"] += W.numel()
                del orig

        if not skip_embeddings:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                W = emb.weight
                orig = W.detach().float().clone()
                new = _apply_scheme_to_matrix(W, scheme, block_size)
                errors.append(relative_error(orig, new.float()))
                W.copy_(new)
                stats["matrices_modified"] += 1
                stats["params_modified"] += W.numel()
                stats["embeddings_quantised"] = True
                del orig

    if errors:
        stats["mean_rel_error"] = round(sum(errors) / len(errors), 6)
        stats["max_rel_error"] = round(max(errors), 6)
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def measure_module_memory(model) -> Dict[str, Any]:
    """
    Break down weight memory by component.

    Directly tests the embedding hypothesis: a 9B model at 4-bit should be
    ~4.19 GB but measures 7.13 GB, and the arithmetic points at the embedding
    table held at bf16. This reports the actual figure instead of inferring it.
    """
    out = {"total_bytes": 0, "components": {}}

    def _bytes(t):
        return t.numel() * t.element_size() if torch.is_tensor(t) else 0

    emb = model.get_input_embeddings()
    emb_bytes = _bytes(emb.weight) if emb is not None and hasattr(emb, "weight") else 0

    lm_head = getattr(model, "lm_head", None)
    head_w = getattr(lm_head, "weight", None) if lm_head is not None else None
    head_bytes = _bytes(head_w)

    tied = False
    if emb is not None and head_w is not None:
        tied = head_w.data_ptr() == emb.weight.data_ptr()

    total = sum(_bytes(p) for p in model.parameters())
    layer_bytes = total - emb_bytes - (0 if tied else head_bytes)

    out["total_bytes"] = total
    out["total_gb"] = round(total / 1024**3, 3)
    out["components"] = {
        "embeddings": {
            "gb": round(emb_bytes / 1024**3, 3),
            "shape": tuple(emb.weight.shape) if emb is not None else None,
            "dtype": str(emb.weight.dtype) if emb is not None else None,
            "share_of_total": round(emb_bytes / total, 4) if total else None,
        },
        "lm_head": {
            "gb": round(head_bytes / 1024**3, 3),
            "tied_to_embeddings": tied,
            "note": ("shares storage with embeddings; counted once"
                     if tied else "separate matrix"),
        },
        "decoder_layers": {
            "gb": round(layer_bytes / 1024**3, 3),
            "share_of_total": round(layer_bytes / total, 4) if total else None,
        },
    }

    # What quantising the embedding table would save.
    if emb_bytes:
        for bits in (4, 8):
            key = f"embedding_at_{bits}bit"
            elem = emb.weight.element_size()
            new_bytes = emb_bytes * (bits / (elem * 8))
            out["components"][key] = {
                "gb": round(new_bytes / 1024**3, 3),
                "saving_gb": round((emb_bytes - new_bytes) / 1024**3, 3),
                "saving_share_of_total": round(
                    (emb_bytes - new_bytes) / total, 4) if total else None,
            }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_quality_validation(
    base: str = "qwen-9b",
    schemes: Tuple[str, ...] = ("fp16", "nf4", "int4", "int4+rot",
                                "nf4+rot", "int3", "int3+rot"),
    tasks: Optional[List[str]] = None,
    eval_limit: Optional[int] = 200,
    block_size: int = 64,
    quantise_embeddings: bool = False,
    results_dir=None,
) -> Dict[str, Any]:
    """
    The gate. For each scheme: reload clean, modify weights in place, evaluate.

    Reloading between schemes is mandatory -- the modification is destructive,
    and reusing a modified model would compound errors across schemes and
    silently invalidate everything after the first.

    Default eval_limit=200 keeps a full sweep to roughly an hour. Raise it for
    a final confirmation run; 200 items gives a standard error around 3.5%, so
    differences smaller than that should not be treated as real.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from .config import RunConfig, capture_environment
    from .loader import load, free_gpu
    from .quality import run_lm_eval, headline_score

    if tasks is None:
        tasks = ["arc_easy", "hellaswag", "piqa"]

    record: Dict[str, Any] = {
        "run_id": f"T2-QUALITY-{base}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "measured",
        "environment": capture_environment(),
        "eval_config": {"tasks": tasks, "limit": eval_limit,
                        "block_size": block_size,
                        "embeddings_quantised": quantise_embeddings},
        "scope_note": (
            "Weights are modified in place in fp16 storage. NO memory is "
            "saved; this measures the QUALITY consequence of quantisation "
            "error, not its size benefit."
        ),
        "runs": {},
    }

    for scheme in schemes:
        print(f"\n=== {scheme} ===")
        cfg = RunConfig(
            experiment_id=f"T2-QUAL-{scheme}",
            label=f"quality validation: {scheme}",
            base=base, backend="fp16", attn_impl="sdpa",
            run_perf=False, run_standard_eval=False,
        )

        model = None
        entry: Dict[str, Any] = {"scheme": scheme,
                                 "desc": SCHEMES.get(scheme, {}).get("desc")}
        try:
            model, tok, load_info = load(cfg)

            if scheme == schemes[0]:
                record["memory_breakdown"] = measure_module_memory(model)

            print("  applying scheme to weights ...")
            entry["modification"] = apply_scheme_in_place(
                model, scheme, block_size,
                skip_embeddings=not quantise_embeddings)
            print(f"  mean reconstruction error: "
                  f"{entry['modification']['mean_rel_error']}")

            print("  evaluating ...")
            ev = run_lm_eval(model, tok, tasks=tasks, num_fewshot=0,
                             limit=eval_limit, batch_size=1)
            entry["eval"] = ev
            entry["headline"] = headline_score(ev)
            entry["status"] = "ok"
            print(f"  headline: {entry['headline']}")

        except Exception as e:
            import traceback
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            print(f"  ERROR: {e}")
        finally:
            del model
            gc.collect()
            free_gpu()

        record["runs"][scheme] = entry

    record["analysis"] = _analyse(record["runs"])
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if results_dir is not None:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{record['run_id']}.json").open("w") as f:
            json.dump(record, f, indent=2)

    return record


def _analyse(runs: Dict[str, Any]) -> Dict[str, Any]:
    """
    The central question: does reconstruction error predict benchmark quality?

    Reports the correlation between the two across schemes. A strong negative
    correlation validates every reconstruction-error result in this project.
    A weak one means the frontier was measuring the wrong quantity.
    """
    ref = runs.get("fp16", {})
    ref_score = ref.get("headline")

    rows = []
    for scheme, r in runs.items():
        if r.get("status") != "ok" or r.get("headline") is None:
            continue
        err = r.get("modification", {}).get("mean_rel_error", 0.0)
        rows.append({
            "scheme": scheme,
            "recon_error": err,
            "headline": r["headline"],
            "quality_drop": (round(1 - r["headline"] / ref_score, 5)
                             if ref_score else None),
        })

    rows.sort(key=lambda x: x["recon_error"])

    # Correlation between reconstruction error and quality drop.
    pts = [(r["recon_error"], r["quality_drop"]) for r in rows
           if r["quality_drop"] is not None and r["recon_error"] > 0]
    corr = None
    if len(pts) >= 3:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        num = sum((x-mx)*(y-my) for x, y in pts)
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        corr = round(num/(dx*dy), 4) if dx > 0 and dy > 0 else None

    # Does rotation add anything on top of NF4's normal-fitted codebook?
    nf4 = next((r for r in rows if r["scheme"] == "nf4"), None)
    nf4r = next((r for r in rows if r["scheme"] == "nf4+rot"), None)
    int4 = next((r for r in rows if r["scheme"] == "int4"), None)

    nf4_note = None
    if nf4 and int4:
        nf4_note = (f"NF4 headline {nf4['headline']} vs uniform int4 "
                    f"{int4['headline']}: NF4's codebook is worth "
                    f"{round(nf4['headline'] - int4['headline'], 5)} on the battery")
    rot_note = None
    if nf4 and nf4r:
        delta = nf4r["headline"] - nf4["headline"]
        rot_note = (
            f"rotation on top of NF4 changes headline by {delta:+.5f}. "
            + ("Rotation adds little beyond NF4's codebook -- experiment 7's "
               "promotion was measured against uniform int4 and overstates "
               "the gain versus the production path."
               if abs(delta) < 0.01 else
               "Rotation adds real benefit beyond NF4."))

    return {
        "reference_score": ref_score,
        "rows": rows,
        "error_vs_quality_correlation": corr,
        "correlation_interpretation": (
            None if corr is None else
            "strong: reconstruction error is a valid proxy for quality, "
            "validating the frontier results" if corr > 0.8 else
            "moderate: reconstruction error is a rough guide only" if corr > 0.5
            else "weak: reconstruction error does NOT predict quality on this "
                 "architecture; frontier conclusions require re-examination"),
        "nf4_vs_int4": nf4_note,
        "rotation_on_top_of_nf4": rot_note,
    }
