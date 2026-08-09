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

# Tasks that use rolling loglikelihood and therefore materialise logits for
# every position. These need a capped window on a large-vocabulary model and
# are run in an isolated lm_eval call.
PERPLEXITY_TASKS = {"wikitext", "pile", "lambada_openai", "lambada_standard"}


def parse_scheme(scheme: str) -> Tuple[str, bool]:
    """
    Split a scheme name into its core and whether embeddings are included.

    The '+emb' suffix quantises the embedding table AND the untied LM head
    alongside the decoder layers. Without it, both are left at bf16, which is
    what bitsandbytes does by default and therefore what the production
    baseline looks like.

    Making this a per-scheme suffix rather than a global flag matters: it lets
    'nf4' and 'nf4+emb' be compared inside a single run, against a shared
    fp16 baseline, rather than across two runs whose baselines could drift.
    """
    if scheme.endswith("+emb"):
        return scheme[:-4], True
    return scheme, False


SCHEMES = {
    "fp16": {"desc": "unmodified reference", "bits": 16},
    "int4": {"desc": "uniform int4, blockwise absmax", "bits": 4},
    "int3": {"desc": "uniform int3", "bits": 3},
    "int2": {"desc": "uniform int2", "bits": 2},
    "nf4": {"desc": "NF4 codebook (simulates bitsandbytes production path)",
            "bits": 4},
    "nf4+emb": {"desc": "NF4 including embeddings and untied LM head "
                        "(3.79 GB of the 9B model, left at bf16 by default)",
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

                # Streaming error: accumulate squared norms rather than
                # keeping a full fp32 copy of the original. Cheap here since
                # decoder matrices are small, but it keeps peak memory flat
                # regardless of matrix size.
                new = _apply_scheme_to_matrix(W, scheme, block_size)
                with torch.no_grad():
                    diff = (W.detach().float() - new.float())
                    num = float(diff.pow(2).sum())
                    den = float(W.detach().float().pow(2).sum())
                    del diff
                errors.append((num ** 0.5) / (den ** 0.5) if den > 0 else 0.0)
                W.copy_(new)
                del new

                stats["matrices_modified"] += 1
                stats["params_modified"] += W.numel()

        if not skip_embeddings:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                stats.update(_quantise_big_matrix(
                    emb.weight, scheme, block_size, errors, label="embeddings"))
                stats["matrices_modified"] += 1
                stats["params_modified"] += emb.weight.numel()

            # The LM head is a SEPARATE untied matrix on this model
            # (measure_module_memory reports tied_to_embeddings=False), so it
            # is not reached by the embedding path above and must be handled
            # explicitly. Together the two are 3.79 GB, 53% of the NF4 model.
            head = getattr(model, "lm_head", None)
            head_w = getattr(head, "weight", None) if head is not None else None
            if head_w is not None and torch.is_tensor(head_w):
                emb_w = emb.weight if emb is not None else None
                tied = (emb_w is not None
                        and head_w.data_ptr() == emb_w.data_ptr())
                if not tied:
                    stats.update(_quantise_big_matrix(
                        head_w, scheme, block_size, errors, label="lm_head"))
                    stats["matrices_modified"] += 1
                    stats["params_modified"] += head_w.numel()
                else:
                    stats["lm_head_tied"] = True

    if errors:
        stats["mean_rel_error"] = round(sum(errors) / len(errors), 6)
        stats["max_rel_error"] = round(max(errors), 6)
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def _quantise_big_matrix(W: torch.Tensor, scheme: str, block_size: int,
                         errors: list, label: str,
                         chunk_rows: int = 8192) -> Dict[str, Any]:
    """
    Quantise a very large matrix in row chunks, on CPU, without ever holding
    a full fp32 copy.

    Necessary because the embedding table and LM head are 248,320 x 4096 each
    -- 1.895 GB in bf16, but 3.79 GB once converted to fp32. A naive
    `W.detach().float().clone()` allocates that on top of an already-resident
    model and OOMs on a 22 GB card.

    Three things make this fit:
      - chunks: only `chunk_rows` rows are in fp32 at any moment
      - CPU: the temporary never touches VRAM
      - streaming error: squared norms accumulate per chunk, so the relative
        error is exact without materialising the whole difference

    Rotation is deliberately NOT applied here. Rotating a 248k-row matrix is
    expensive, and the quality validation already showed rotation does not
    help on the benchmark, so there is nothing to gain from it.
    """
    if scheme == "fp16":
        return {f"{label}_quantised": False, f"{label}_note": "reference, unmodified"}

    core = scheme[:-4] if scheme.endswith("+rot") else scheme

    n_rows = W.shape[0]
    sq_err = 0.0
    sq_ref = 0.0

    with torch.no_grad():
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk = W[start:end].detach().to("cpu", torch.float32)

            if core == "nf4":
                q = quantize_dequantize_nf4(chunk, block_size)
            else:
                bits = int(core.replace("int", ""))
                q = quantize_dequantize(chunk, bits, block_size)

            sq_err += float(((chunk - q) ** 2).sum())
            sq_ref += float((chunk ** 2).sum())

            W[start:end].copy_(q.to(W.device, W.dtype))
            del chunk, q

    err = (sq_err ** 0.5) / (sq_ref ** 0.5) if sq_ref > 0 else 0.0
    errors.append(err)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        f"{label}_quantised": True,
        f"{label}_rel_error": round(err, 6),
        f"{label}_shape": tuple(W.shape),
        f"{label}_gb_at_bf16": round(W.numel() * 2 / 1024**3, 3),
    }


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
    perplexity_max_length: int = 1024,
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

    from .config import RunConfig, capture_environment, run_stamp
    from .loader import load, free_gpu
    from .quality import run_lm_eval, headline_score

    if tasks is None:
        tasks = ["arc_easy", "hellaswag", "piqa"]

    record: Dict[str, Any] = {
        "run_id": f"T2-QUALITY-{base}-{run_stamp()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "measured",
        "environment": capture_environment(),
        "eval_config": {"tasks": tasks, "limit": eval_limit,
                        "block_size": block_size,
                        "embeddings_quantised": quantise_embeddings,
                        "perplexity_max_length": perplexity_max_length},
        "scope_note": (
            "Weights are modified in place in fp16 storage. NO memory is "
            "saved; this measures the QUALITY consequence of quantisation "
            "error, not its size benefit."
        ),
        "runs": {},
    }

    for scheme in schemes:
        core, wants_emb = parse_scheme(scheme)
        # Global flag still forces embeddings on for every scheme; the
        # per-scheme suffix is the finer control.
        include_emb = wants_emb or quantise_embeddings

        print(f"\n=== {scheme} ===")
        cfg = RunConfig(
            experiment_id=f"T2-QUAL-{scheme}",
            label=f"quality validation: {scheme}",
            base=base, backend="fp16", attn_impl="sdpa",
            run_perf=False, run_standard_eval=False,
        )

        model = None
        entry: Dict[str, Any] = {"scheme": scheme,
                                 "desc": SCHEMES.get(scheme, {}).get("desc"),
                                 "embeddings_included": include_emb}
        try:
            model, tok, load_info = load(cfg)

            if scheme == schemes[0]:
                record["memory_breakdown"] = measure_module_memory(model)

            print("  applying scheme to weights ...")
            entry["modification"] = apply_scheme_in_place(
                model, core, block_size,
                skip_embeddings=not include_emb)
            print(f"  mean reconstruction error: "
                  f"{entry['modification']['mean_rel_error']}")

            print("  evaluating ...")

            # Accuracy and perplexity are run in SEPARATE lm_eval calls.
            #
            # simple_evaluate raises before returning anything, so a failure
            # in one task destroys the results of every task in the same call.
            # That already cost an hour of inference here: wikitext OOM'd on
            # logit allocation and took four schemes' worth of completed
            # ARC/HellaSwag/PIQA results down with it.
            acc_tasks = [t for t in tasks if t not in PERPLEXITY_TASKS]
            ppl_tasks = [t for t in tasks if t in PERPLEXITY_TASKS]

            ev = {"status": "ok", "scores": {}}
            if acc_tasks:
                ev_acc = run_lm_eval(model, tok, tasks=acc_tasks,
                                     num_fewshot=0, limit=eval_limit,
                                     batch_size=1)
                entry["eval_accuracy"] = ev_acc
                if ev_acc.get("status") == "ok":
                    ev["scores"].update(ev_acc["scores"])
                else:
                    ev["status"] = "error"
                    ev["error"] = ev_acc.get("error")

            if ppl_tasks:
                # Capped window: see run_lm_eval docstring. Failure here is
                # recorded and does not affect the accuracy results above.
                ev_ppl = run_lm_eval(model, tok, tasks=ppl_tasks,
                                     num_fewshot=0, limit=eval_limit,
                                     batch_size=1,
                                     max_length=perplexity_max_length)
                entry["eval_perplexity"] = ev_ppl
                if ev_ppl.get("status") == "ok":
                    ev["scores"].update(ev_ppl["scores"])
                else:
                    entry["perplexity_error"] = ev_ppl.get("error")

            entry["eval"] = ev
            entry["headline"] = headline_score(ev)
            entry["perplexity"] = _extract_perplexity(ev)
            entry["status"] = "ok" if ev["status"] == "ok" else "error"
            print(f"  headline: {entry['headline']}"
                  + (f"   ppl: {entry['perplexity']}"
                     if entry["perplexity"] else "   ppl: n/a"))

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

    n_items = (len(tasks) * eval_limit) if eval_limit else None
    record["analysis"] = _analyse(record["runs"], n_items=n_items)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if results_dir is not None:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{record['run_id']}.json").open("w") as f:
            json.dump(record, f, indent=2)

    return record


def _extract_perplexity(eval_result: Dict[str, Any]) -> Optional[float]:
    """
    Pull word-level perplexity out of an lm-eval result, if a perplexity task
    was run.

    Perplexity is the sensitive instrument for damage to the LM head. A
    multiple-choice task only asks which of four options scores highest, so
    it tolerates substantial distortion of the output distribution as long as
    the ranking survives. Perplexity measures the distribution over all
    248,320 vocabulary entries directly, so it registers degradation that
    accuracy hides.

    Lower is better, which is the reverse of every other metric here, so it is
    kept out of the headline average and reported separately.
    """
    if eval_result.get("status") != "ok":
        return None
    for task, metrics in eval_result.get("scores", {}).items():
        for key in ("word_perplexity,none", "perplexity,none",
                    "byte_perplexity,none"):
            if key in metrics:
                return round(metrics[key], 4)
    return None


def _analyse(runs: Dict[str, Any], n_items: Optional[int] = None) -> Dict[str, Any]:
    """
    The central question: does reconstruction error predict benchmark quality?

    Answered per family as well as pooled, because a high pooled correlation
    can mask two clean relationships that are offset from one another -- which
    is precisely what rotation turned out to do.

    Standard error is reported so that differences within noise are not read
    as findings.
    """
    ref = runs.get("fp16", {})
    ref_score = ref.get("headline")

    stderr = None
    if n_items and ref_score:
        stderr = math.sqrt(ref_score * (1 - ref_score) / n_items)

    rows = []
    for scheme, r in runs.items():
        if r.get("status") != "ok" or r.get("headline") is None:
            continue
        err = r.get("modification", {}).get("mean_rel_error", 0.0)
        rows.append({
            "scheme": scheme,
            "recon_error": err,
            "headline": r["headline"],
            "perplexity": r.get("perplexity"),
            "embeddings_included": r.get("embeddings_included", False),
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
    int4 = next((r for r in rows if r["scheme"] == "int4"), None)

    nf4_note = None
    if nf4 and int4:
        nf4_note = (f"NF4 headline {nf4['headline']} vs uniform int4 "
                    f"{int4['headline']}: NF4's codebook is worth "
                    f"{round(nf4['headline'] - int4['headline'], 5)} on the battery")

    # Paired comparison: does rotation help at MATCHED bit width?
    #
    # This is the question experiment 7 could not answer, because it measured
    # reconstruction error only. Sign matters here -- an earlier version of
    # this function tested abs(delta) and reported a QUALITY REGRESSION as
    # "rotation adds real benefit", which is the opposite of the truth.
    rot_pairs = []
    for bare in ("nf4", "int4", "int3", "int2"):
        a = next((r for r in rows if r["scheme"] == bare), None)
        b = next((r for r in rows if r["scheme"] == bare + "+rot"), None)
        if a and b:
            rot_pairs.append({
                "bit_scheme": bare,
                "headline_plain": a["headline"],
                "headline_rotated": b["headline"],
                "quality_delta": round(b["headline"] - a["headline"], 5),
                "recon_plain": a["recon_error"],
                "recon_rotated": b["recon_error"],
                "recon_delta": round(b["recon_error"] - a["recon_error"], 6),
                # The reversal to watch for: rotation lowering reconstruction
                # error while also lowering benchmark quality.
                "directions_disagree": (
                    b["recon_error"] < a["recon_error"]
                    and b["headline"] < a["headline"]),
            })

    rot_note = None
    if rot_pairs:
        n_worse = sum(1 for p in rot_pairs if p["quality_delta"] < 0)
        n_disagree = sum(1 for p in rot_pairs if p["directions_disagree"])
        mean_delta = sum(p["quality_delta"] for p in rot_pairs) / len(rot_pairs)

        if n_worse == len(rot_pairs) and n_disagree == len(rot_pairs):
            rot_note = (
                f"REVERSAL: in {n_disagree}/{len(rot_pairs)} matched pairs, "
                f"rotation LOWERED reconstruction error but ALSO lowered "
                f"benchmark quality (mean delta {mean_delta:+.5f}). "
                f"Reconstruction error is not a valid proxy across the "
                f"rotation boundary. Experiment 7's promotion, which rested "
                f"on reconstruction error alone, does not hold on quality."
            )
        elif mean_delta > 0.01:
            rot_note = (f"rotation improves headline by {mean_delta:+.5f} on "
                        f"average across matched bit widths")
        else:
            rot_note = (f"rotation changes headline by {mean_delta:+.5f} on "
                        f"average: no clear benefit at matched bit width")

    # Correlation computed separately within each family. A high pooled
    # correlation can hide two clean but OFFSET relationships, which is
    # exactly what rotation produces here.
    def _corr(subset):
        pts = [(r["recon_error"], r["quality_drop"]) for r in subset
               if r["quality_drop"] is not None]
        if len(pts) < 3:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        num = sum((x-mx)*(y-my) for x, y in pts)
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        return round(num/(dx*dy), 4) if dx > 0 and dy > 0 else None

    corr_plain = _corr([r for r in rows if not r["scheme"].endswith("+rot")])
    corr_rot = _corr([r for r in rows if r["scheme"].endswith("+rot")])

    # Embedding + LM head quantisation: the largest memory target found.
    # Accuracy and perplexity are BOTH checked, because a multiple-choice
    # battery can miss output-distribution damage that perplexity catches.
    emb_note = None
    emb_pairs = []
    for bare in ("nf4", "int4"):
        a = next((r for r in rows if r["scheme"] == bare), None)
        b = next((r for r in rows if r["scheme"] == bare + "+emb"), None)
        if a and b:
            ppl_delta = None
            if a["perplexity"] and b["perplexity"]:
                ppl_delta = round(b["perplexity"] / a["perplexity"] - 1, 5)
            emb_pairs.append({
                "bit_scheme": bare,
                "headline_without_emb": a["headline"],
                "headline_with_emb": b["headline"],
                "accuracy_delta": round(b["headline"] - a["headline"], 5),
                "perplexity_without_emb": a["perplexity"],
                "perplexity_with_emb": b["perplexity"],
                "perplexity_relative_increase": ppl_delta,
                "within_noise": (abs(b["headline"] - a["headline"]) < 2 * stderr
                                 if stderr else None),
            })

    if emb_pairs:
        p = emb_pairs[0]
        parts = [f"Quantising embeddings + LM head changes accuracy by "
                 f"{p['accuracy_delta']:+.5f}"]
        if p["within_noise"] is not None:
            parts.append("within noise" if p["within_noise"] else "OUTSIDE noise")
        if p["perplexity_relative_increase"] is not None:
            pr = p["perplexity_relative_increase"]
            parts.append(
                f"perplexity {pr:+.2%}"
                + (" -- output distribution intact" if abs(pr) < 0.05 else
                   " -- OUTPUT DISTRIBUTION DEGRADED; accuracy alone would "
                   "have missed this"))
        else:
            parts.append("no perplexity task run, so output-distribution "
                         "damage is untested")
        emb_note = ". ".join(parts)

    return {
        "reference_score": ref_score,
        "rows": rows,
        "n_eval_items": n_items,
        "single_score_stderr": round(stderr, 5) if stderr else None,
        "significance_note": (
            f"Differences smaller than ~{round(2*stderr, 4)} (2 SE) should be "
            f"treated as noise at this eval size."
            if stderr else "eval size unknown; treat small differences as noise"),
        "error_vs_quality_correlation": corr,
        "correlation_within_plain": corr_plain,
        "correlation_within_rotated": corr_rot,
        "correlation_interpretation": (
            None if corr is None else
            "strong: reconstruction error is a valid proxy for quality, "
            "validating the frontier results" if corr > 0.8 else
            "moderate: reconstruction error is a rough guide only" if corr > 0.5
            else "weak: reconstruction error does NOT predict quality on this "
                 "architecture; frontier conclusions require re-examination"),
        "correlation_caveat": (
            "A high pooled correlation is driven mainly by bit width. Check "
            "the within-family correlations and the rotation pairs below: if "
            "both families correlate cleanly but are offset, the proxy is "
            "valid WITHIN a family and invalid ACROSS it."),
        "nf4_vs_int4": nf4_note,
        "rotation_pairs": rot_pairs,
        "rotation_on_top_of_nf4": rot_note,
        "embedding_pairs": emb_pairs,
        "embedding_verdict": emb_note,
    }
