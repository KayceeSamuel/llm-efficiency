"""
compression.py — the compression frontier. Register experiments 6b, 12b, 13-prep.

MOTIVATION, from measured results in this project:

Experiments 1, 5, 5b and 18 all searched for exploitable redundancy in these
weights -- across repeating blocks, within matrix rank, along the depth axis.
All four found none. Cross-block cosine similarity was 0.005; weight matrices
sit at the random-matrix rank baseline; depth-frequency energy is uniform.

That is a redirection, not a dead end. If no structure can be exploited, the
only remaining lever is encoding each value more efficiently. This module
tests that lever.

Experiment 7 then measured how far from optimal the current encoding is.
After rotation removed the outlier penalty, every matrix role converged to a
common error floor of ~0.1076 at 4-bit. The Shannon rate-distortion bound for
a Gaussian source at 4 bits is 0.0625. The gap is 2.97x in MSE, which is
0.78 bits per weight of theoretical headroom -- roughly 1.4 GB on this model,
recoverable at IDENTICAL error by a better encoder alone.

WHAT THIS MODULE MEASURES:
For each method and combination, the honest (effective_bits, relative_error)
pair, and which points lie on the Pareto frontier. Effective bits include ALL
overhead -- scale factors, position indices, residual codebooks -- because
omitting overhead is the standard way these comparisons get inflated.

PRE-VALIDATED ON SYNTHETIC DATA:
  - entropy coding: 4.250 -> 3.740 bits at IDENTICAL error (indices are
    non-uniformly distributed, so 4 bits of storage carries ~3.5 bits of
    information)
  - rotation + entropy coding jointly: 3.751 bits at error 0.1076, versus
    plain int4 at 4.250 bits and error 0.1286. Strictly better on both axes.
  - 2:4 structured sparsity: error 0.360 versus 0.129 for plain int4.
    Catastrophic. Included as a control to confirm this on real weights
    rather than assume it.
"""

import gc
import math
from typing import Dict, Any, List, Optional, Tuple

import torch

from .rotation import (
    quantize_dequantize, relative_error, kronecker_rotation_factors,
    apply_kronecker_rotation,
)


# ---------------------------------------------------------------------------
# Accounting. Every method must declare its full cost.
# ---------------------------------------------------------------------------

SCALE_BITS = 16          # fp16 scale factor per block
DEFAULT_BLOCK = 64


def scale_overhead(block_size: int = DEFAULT_BLOCK) -> float:
    """Bits per weight spent on block scale factors."""
    return SCALE_BITS / block_size


# ---------------------------------------------------------------------------
# Entropy: the free saving
# ---------------------------------------------------------------------------

def quantize_indices(W: torch.Tensor, bits: int,
                     block_size: int = DEFAULT_BLOCK) -> torch.Tensor:
    """Return the integer indices a quantiser would store, before dequantising."""
    flat = W.detach().flatten().float()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    blocks = flat.view(-1, block_size)

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    scale = blocks.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.full_like(scale, 1e-12), scale)
    return torch.clamp(torch.round(blocks / scale), qmin, qmax).flatten()


def shannon_entropy(idx: torch.Tensor) -> float:
    """
    Entropy of the index distribution, in bits per symbol.

    This is the lower bound on lossless coded size. Because weights cluster
    near zero, the indices are far from uniform: a 4-bit grid typically
    carries only ~3.5 bits of actual information. The difference is storage
    spent on nothing, recoverable by Huffman or arithmetic coding with ZERO
    additional error -- no quantised value changes, only how it is written
    down.

    Cost is decode overhead at load time, paid once, not per token.
    """
    vals, counts = torch.unique(idx, return_counts=True)
    p = counts.float() / counts.sum()
    return float(-(p * torch.log2(p)).sum().item())


# ---------------------------------------------------------------------------
# Residual (multi-stage) quantisation -- fractional bit rates
# ---------------------------------------------------------------------------

def residual_quantize(W: torch.Tensor, base_bits: int, resid_bits: int,
                      block_size: int = DEFAULT_BLOCK):
    """
    Quantise, take the residual, quantise that too.

    Breaks the constraint that bit rates must be integers: a 4+2 scheme sits
    between 4 and 8 bits with its own error characteristic. Also the natural
    home for a low-rank-plus-sparse decomposition, where the second stage
    captures what the first missed.

    Both stages pay scale overhead, which is counted.
    """
    base = quantize_dequantize(W, base_bits, block_size)
    resid = W - base
    fine = quantize_dequantize(resid, resid_bits, block_size)
    return base + fine


# ---------------------------------------------------------------------------
# Sparsity -- an orthogonal axis (control arm)
# ---------------------------------------------------------------------------

def sparsify_n_m(W: torch.Tensor, n: int = 2, m: int = 4):
    """
    N:M structured sparsity -- keep the n largest of every m consecutive
    weights, zero the rest. 2:4 has hardware support on Ampere and later.

    Reduces the NUMBER of values rather than bits per value, so it multiplies
    with quantisation rather than competing.

    Included as a CONTROL. Synthetic testing showed catastrophic error (0.360
    versus 0.129 for plain int4) because zeroing half the weights destroys far
    more than the storage saves. Retained to confirm that on real weights
    rather than assume it -- and because a negative result here is worth
    recording, since 2:4 is widely assumed to be near-free.
    """
    flat = W.detach().flatten().float().clone()
    pad = (-flat.numel()) % m
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])

    groups = flat.view(-1, m)
    keep = torch.argsort(groups.abs(), dim=1, descending=True)[:, :n]
    mask = torch.zeros_like(groups)
    mask.scatter_(1, keep, 1.0)
    out = (groups * mask).flatten()
    if pad:
        out = out[:-pad]
    return out.view(W.shape)


def sparse_bits_per_weight(n: int, m: int, value_bits: int,
                           block_size: int = DEFAULT_BLOCK) -> float:
    """
    Honest cost of N:M sparsity: the kept values PLUS the position indices
    needed to know which slots they occupied.

    Position cost is ceil(log2(m)) bits per kept value. Omitting this is the
    usual way sparsity is made to look cheaper than it is.
    """
    idx_bits = math.ceil(math.log2(m))
    return (n * value_bits + n * idx_bits) / m + scale_overhead(block_size)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def evaluate_methods(W: torch.Tensor, rotate: bool = False,
                     block_size: int = DEFAULT_BLOCK,
                     bits_list: Tuple[int, ...] = (2, 3, 4, 5, 6),
                     include_sparsity: bool = True,
                     include_residual: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate every method on one matrix, optionally with rotation applied
    first. Error is always measured against the ORIGINAL W, with any inverse
    rotation inside the measured path, so no method gets to hide its own cost.
    """
    with torch.no_grad():
        Wf = W.detach().float()
        if Wf.is_cuda:
            Wf = Wf.cpu()

        rows, cols = Wf.shape
        rot_tag = "+rot" if rotate else ""

        if rotate:
            Hr, Qr, kr, mr = kronecker_rotation_factors(rows)
            Hc, Qc, kc, mc = kronecker_rotation_factors(cols)
            X = apply_kronecker_rotation(Wf, Hr, Qr, kr, mr, dim=0)
            X = apply_kronecker_rotation(X, Hc, Qc, kc, mc, dim=1)

            def unrot(Y):
                Y = apply_kronecker_rotation(Y, Hc.T, Qc.T, kc, mc, dim=1)
                return apply_kronecker_rotation(Y, Hr.T, Qr.T, kr, mr, dim=0)
        else:
            X = Wf

            def unrot(Y):
                return Y

        results = []
        ov = scale_overhead(block_size)

        for bits in bits_list:
            deq = quantize_dequantize(X, bits, block_size)
            err = relative_error(Wf, unrot(deq))
            idx = quantize_indices(X, bits, block_size)
            H = shannon_entropy(idx)

            results.append({
                "method": f"int{bits}{rot_tag}",
                "family": "uniform",
                "bits_nominal": bits,
                "effective_bits": round(bits + ov, 4),
                "rel_error": round(err, 6),
                "rotated": rotate,
            })
            # Identical error, fewer bits -- the only lossless saving available.
            results.append({
                "method": f"int{bits}+entropy{rot_tag}",
                "family": "entropy",
                "bits_nominal": bits,
                "effective_bits": round(H + ov, 4),
                "rel_error": round(err, 6),
                "index_entropy": round(H, 4),
                "entropy_saving": round(1 - (H + ov) / (bits + ov), 4),
                "rotated": rotate,
            })

        if include_residual:
            for base_b, res_b in [(3, 2), (4, 2), (2, 2), (3, 3)]:
                deq = residual_quantize(X, base_b, res_b, block_size)
                err = relative_error(Wf, unrot(deq))
                results.append({
                    "method": f"int{base_b}+res{res_b}{rot_tag}",
                    "family": "residual",
                    "bits_nominal": base_b + res_b,
                    "effective_bits": round(base_b + res_b + 2 * ov, 4),
                    "rel_error": round(err, 6),
                    "rotated": rotate,
                })

        if include_sparsity:
            for bits in (4, 8):
                Ws = sparsify_n_m(X, 2, 4)
                deq = quantize_dequantize(Ws, bits, block_size)
                err = relative_error(Wf, unrot(deq))
                results.append({
                    "method": f"2:4sparse+int{bits}{rot_tag}",
                    "family": "sparsity",
                    "bits_nominal": bits,
                    "effective_bits": round(
                        sparse_bits_per_weight(2, 4, bits, block_size), 4),
                    "rel_error": round(err, 6),
                    "rotated": rotate,
                })

        return results


# ---------------------------------------------------------------------------
# Frontier analysis
# ---------------------------------------------------------------------------

def pareto_frontier(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Points not dominated by any other: nothing else is both smaller AND more
    accurate. Everything off the frontier is strictly worse than something
    else and should never be chosen.
    """
    ordered = sorted(points, key=lambda p: (p["effective_bits"], p["rel_error"]))
    frontier, best_err = [], float("inf")
    for p in ordered:
        if p["rel_error"] < best_err:
            best_err = p["rel_error"]
            frontier.append(p)
    return frontier


def shannon_bound(bits: float) -> float:
    """
    Rate-distortion bound for a Gaussian source: D(R) = sigma^2 * 2^(-2R),
    expressed as relative error. No method can beat this at a given rate.

    Measured floor at 4 bits after rotation was 0.1076 against a bound of
    0.0625 -- a 2.97x MSE gap, or 0.78 bits per weight of headroom.
    """
    return 2.0 ** (-bits)


def gap_to_bound(effective_bits: float, rel_error: float) -> Dict[str, float]:
    bound = shannon_bound(effective_bits)
    mse_ratio = (rel_error / bound) ** 2 if bound > 0 else float("nan")
    return {
        "bound_rel_error": round(bound, 6),
        "mse_gap": round(mse_ratio, 3),
        "bits_recoverable": round(0.5 * math.log2(mse_ratio), 3) if mse_ratio > 0 else None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_compression_frontier(
    base: str = "qwen-9b",
    sample_matrices: int = 16,
    block_size: int = DEFAULT_BLOCK,
    results_dir=None,
) -> Dict[str, Any]:
    """
    Map the compression frontier on real weights.

    Requires an UNQUANTISED model. bitsandbytes stores NF4 weights packed in
    an opaque layout, so reading them back would give the quantisation grid
    rather than the underlying values.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from .config import RunConfig, capture_environment
    from .loader import load, free_gpu
    from .weights import classify_layers, _get_layer_list, _matrix_role

    cfg = RunConfig(
        experiment_id="T2-FRONTIER",
        label="compression frontier: entropy, residual, sparsity x rotation",
        base=base, backend="fp16", attn_impl="sdpa",
        run_perf=False, run_standard_eval=False,
        notes="Weight reconstruction error vs honest effective bits.",
    )

    record: Dict[str, Any] = {
        "run_id": f"T2-FRONTIER-{cfg.fingerprint()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "environment": capture_environment(),
        "provenance": "measured",
        "scope_note": (
            "Reconstruction error only. Effective bits include scale, index, "
            "and residual-codebook overhead. Entropy figures are the coded "
            "lower bound; a real codec adds a small constant."
        ),
    }

    model = None
    try:
        model, tok, load_info = load(cfg)
        record["load"] = load_info
        del tok

        classification = classify_layers(model)
        record["classification"] = classification
        layers = _get_layer_list(model)

        candidates = []
        for i, layer in enumerate(layers):
            kind = ("attention" if i in classification["attention_layers"]
                    else "linear")
            for name, module in layer.named_modules():
                W = getattr(module, "weight", None)
                if W is None or not torch.is_tensor(W) or W.dim() != 2:
                    continue
                if W.numel() < 4096:
                    continue
                candidates.append((i, kind, name, W))

        if len(candidates) > sample_matrices:
            step = len(candidates) // sample_matrices
            candidates = candidates[::step][:sample_matrices]

        all_points: List[Dict[str, Any]] = []
        for n, (i, kind, name, W) in enumerate(candidates, 1):
            print(f"  [{n}/{len(candidates)}] layer {i} {name}")
            for rotate in (False, True):
                for p in evaluate_methods(W, rotate=rotate,
                                          block_size=block_size):
                    p.update({"layer": i, "layer_kind": kind,
                              "matrix": name, "role": _matrix_role(name)})
                    all_points.append(p)

        record["points"] = all_points
        record["aggregate"] = _aggregate_methods(all_points)
        record["frontier"] = pareto_frontier(list(record["aggregate"].values()))
        record["verdict"] = _verdict(record["aggregate"], record["frontier"])
        record["status"] = "ok"

    except Exception as e:
        import traceback
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
    finally:
        del model
        gc.collect()
        free_gpu()

    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if results_dir is not None:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{record['run_id']}.json").open("w") as f:
            json.dump(record, f, indent=2)

    return record


def _aggregate_methods(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean across matrices, per method."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        groups.setdefault(p["method"], []).append(p)

    out = {}
    for method, items in groups.items():
        bits = sum(i["effective_bits"] for i in items) / len(items)
        err = sum(i["rel_error"] for i in items) / len(items)
        entry = {
            "method": method,
            "family": items[0]["family"],
            "rotated": items[0]["rotated"],
            "n_matrices": len(items),
            "effective_bits": round(bits, 4),
            "rel_error": round(err, 6),
        }
        entry.update(gap_to_bound(bits, err))
        if items[0]["family"] == "entropy":
            entry["mean_index_entropy"] = round(
                sum(i["index_entropy"] for i in items) / len(items), 4)
            entry["mean_entropy_saving"] = round(
                sum(i["entropy_saving"] for i in items) / len(items), 4)
        out[method] = entry
    return out


def _verdict(aggregate: Dict[str, Any],
             frontier: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pre-registered decision. The reference point is plain int4 -- the
    configuration this model currently ships with.
    """
    ref = aggregate.get("int4")
    if not ref:
        return {"status": "indeterminate", "reason": "no int4 reference point"}

    # Strictly dominant: fewer bits AND lower error than the current setup.
    dominant = [m for m in aggregate.values()
                if m["effective_bits"] < ref["effective_bits"]
                and m["rel_error"] < ref["rel_error"]]
    dominant.sort(key=lambda m: m["effective_bits"])

    ent4 = aggregate.get("int4+entropy")
    entropy_saving = (round(1 - ent4["effective_bits"] / ref["effective_bits"], 4)
                      if ent4 else None)

    sparse = [m for m in aggregate.values() if m["family"] == "sparsity"]
    sparse_verdict = None
    if sparse:
        worst = max(sparse, key=lambda m: m["rel_error"])
        sparse_verdict = (
            f"2:4 sparsity error {worst['rel_error']:.4f} versus int4 "
            f"{ref['rel_error']:.4f} -- "
            + ("killed: destroys more than it saves"
               if worst["rel_error"] > ref["rel_error"] * 1.5
               else "competitive, investigate further"))

    best = dominant[0] if dominant else None

    return {
        "reference": {"method": "int4",
                      "effective_bits": ref["effective_bits"],
                      "rel_error": ref["rel_error"]},
        "best_dominant_method": best["method"] if best else None,
        "size_reduction_vs_int4": (
            round(1 - best["effective_bits"] / ref["effective_bits"], 4)
            if best else None),
        "error_change_vs_int4": (
            round(best["rel_error"] / ref["rel_error"] - 1, 4) if best else None),
        "n_dominant_methods": len(dominant),
        "entropy_coding_saving": entropy_saving,
        "entropy_note": ("lossless: no quantised value changes, only how it "
                         "is written down"),
        "sparsity_verdict": sparse_verdict,
        "frontier_methods": [m["method"] for m in frontier],
        "int4_gap_to_shannon_bound": {
            "mse_gap": ref["mse_gap"],
            "bits_recoverable": ref["bits_recoverable"],
            "note": ("headroom a fundamentally better encoder (lattice, "
                     "trellis) could recover at identical error"),
        },
    }
