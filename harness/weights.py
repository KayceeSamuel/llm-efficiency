"""
weights.py — Tier 1 weight inspection. Experiments 1, 2, 5, and 18.

No generation, no evaluation. Loads a model and analyses its weight matrices
as numbers. Cheap relative to anything involving inference, and the results
gate roughly half the register:

  Exp 2  distribution profiling  -> gates 10 (mixed precision by layer type)
  Exp 1  cross-block similarity  -> gates 11 (delta encoding)
  Exp 5  rank profiling          -> gates 12 (low-rank + sparse)
  Exp 18 depth-axis DCT          -> free byproduct of exp 1's tensor stacking

IMPORTANT: quantised weights are useless for this analysis. bitsandbytes
stores NF4 weights packed two-per-byte in an opaque layout; reading them back
gives you the quantisation grid, not the underlying distribution. Every
function here therefore requires an UNQUANTISED model.

On a 22GB L4 a 9B model at bf16 is ~18GB, which fits but leaves little room.
These functions never run a forward pass, so activation memory is not a
concern -- but the model must be loaded with backend="fp16".
"""

import gc
import re
from typing import Dict, Any, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

def classify_layers(model) -> Dict[str, Any]:
    """
    Determine which layer indices are full-attention and which are linear
    (Gated DeltaNet), by inspecting module names rather than assuming the
    3:1 pattern holds.

    Returns indices plus the detected repeating period, which experiment 1
    depends on: 'corresponding layers across blocks' is only meaningful if we
    know the block size.
    """
    layers = _get_layer_list(model)
    attn_idx, linear_idx = [], []

    for i, layer in enumerate(layers):
        names = [n for n, _ in layer.named_modules()]
        joined = " ".join(names).lower()

        # GDN layers expose conv/gate/delta machinery that attention lacks.
        is_linear = any(k in joined for k in
                        ("conv1d", "linear_attn", "gated_delta", "deltanet"))
        # Attention layers expose k_proj/v_proj feeding a growing cache.
        has_kv = ("k_proj" in joined and "v_proj" in joined)

        if is_linear and not has_kv:
            linear_idx.append(i)
        elif has_kv and not is_linear:
            attn_idx.append(i)
        elif is_linear:
            linear_idx.append(i)
        else:
            attn_idx.append(i)

    period = _detect_period(attn_idx, len(layers))

    return {
        "n_layers": len(layers),
        "attention_layers": attn_idx,
        "linear_layers": linear_idx,
        "n_attention": len(attn_idx),
        "n_linear": len(linear_idx),
        "block_period": period,
        "n_blocks": len(layers) // period if period else None,
        "ratio": (f"{len(linear_idx)}:{len(attn_idx)}"
                  if attn_idx else "all-linear"),
    }


def _get_layer_list(model):
    for path in ("model.layers", "model.model.layers", "transformer.h",
                 "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate the decoder layer list on this model.")


def _detect_period(attn_idx: List[int], n_layers: int) -> Optional[int]:
    """Infer the repeating block size from attention layer spacing."""
    if len(attn_idx) < 2:
        return None
    gaps = [b - a for a, b in zip(attn_idx, attn_idx[1:])]
    return gaps[0] if len(set(gaps)) == 1 else None


# ---------------------------------------------------------------------------
# Experiment 2 — weight distribution and outlier profiling
# ---------------------------------------------------------------------------

def profile_distributions(model, max_elements: int = 2_000_000) -> Dict[str, Any]:
    """
    Experiment 2: does the normal-distribution prior behind NF4 hold, and does
    it hold equally for GDN and attention layers?

    NF4 spaces its 16 levels according to a normal distribution. If the real
    weights are heavier-tailed than normal, those levels are misplaced and a
    codebook fitted to the actual distribution would do better at the same
    bit width. If the two layer types differ, that is direct evidence for
    treating them differently (experiment 10).

    Reported per matrix:
      kurtosis      excess kurtosis. 0 = normal. >0 = heavier tails, which is
                    what makes quantisation hard, because a few large values
                    force a wide scale and waste precision on the rest.
      outlier_ratio fraction of weights beyond 4 standard deviations.
      max_over_std  largest magnitude in units of sigma. The single number
                    that most directly predicts quantisation difficulty.
    """
    classification = classify_layers(model)
    layers = _get_layer_list(model)
    per_matrix: List[Dict[str, Any]] = []

    for i, layer in enumerate(layers):
        kind = "attention" if i in classification["attention_layers"] else "linear"

        for name, module in layer.named_modules():
            W = getattr(module, "weight", None)
            if W is None or not torch.is_tensor(W) or W.dim() != 2:
                continue
            if W.numel() < 1024:
                continue

            stats = _tensor_stats(W, max_elements)
            stats.update({
                "layer": i,
                "layer_kind": kind,
                "matrix": name,
                "role": _matrix_role(name),
                "shape": tuple(W.shape),
            })
            per_matrix.append(stats)

    return {
        "classification": classification,
        "per_matrix": per_matrix,
        "by_layer_kind": _aggregate(per_matrix, "layer_kind"),
        "by_role": _aggregate(per_matrix, "role"),
    }


def _tensor_stats(W: torch.Tensor, max_elements: int) -> Dict[str, Any]:
    """Distribution statistics, computed in fp32 on a subsample if large."""
    with torch.no_grad():
        flat = W.detach().flatten().float()
        n_total = flat.numel()

        if n_total > max_elements:
            # Strided subsample: deterministic, and avoids sampling a single
            # contiguous region whose statistics may not be representative.
            step = n_total // max_elements
            flat = flat[::step][:max_elements]

        mean = flat.mean()
        std = flat.std()
        centred = flat - mean

        if std > 0:
            z = centred / std
            kurt = (z.pow(4).mean() - 3.0).item()
            skew = z.pow(3).mean().item()
            outlier_ratio = (z.abs() > 4).float().mean().item()
            max_over_std = z.abs().max().item()
        else:
            kurt = skew = outlier_ratio = max_over_std = 0.0

        return {
            "n_elements": n_total,
            "mean": round(mean.item(), 8),
            "std": round(std.item(), 6),
            "kurtosis": round(kurt, 4),
            "skew": round(skew, 4),
            "outlier_ratio": round(outlier_ratio, 8),
            "max_over_std": round(max_over_std, 3),
        }


def _matrix_role(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attn_proj"
    if any(k in n for k in ("gate_proj", "up_proj", "down_proj")):
        return "ffn"
    if "conv" in n:
        return "conv"
    if any(k in n for k in ("in_proj", "out_proj", "a_proj", "b_proj", "dt")):
        return "gdn_proj"
    return "other"


def _aggregate(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)

    out = {}
    for g, items in groups.items():
        def mean_of(f):
            vals = [it[f] for it in items if it.get(f) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None
        out[g] = {
            "n_matrices": len(items),
            "mean_kurtosis": mean_of("kurtosis"),
            "mean_outlier_ratio": mean_of("outlier_ratio"),
            "mean_max_over_std": mean_of("max_over_std"),
            "mean_std": mean_of("std"),
        }
    return out


# ---------------------------------------------------------------------------
# Experiment 1 — cross-block weight similarity
# ---------------------------------------------------------------------------

def cross_block_similarity(model, max_elements: int = 2_000_000) -> Dict[str, Any]:
    """
    Experiment 1: are corresponding layers across the repeating blocks similar
    enough that deltas could be stored instead of full weights?

    The model repeats a fixed pattern (here [GDN, GDN, GDN, ATTN] x 8). This
    compares each matrix against its counterpart in block 0 -- same position
    in the pattern, different block.

    KILL CRITERION for experiment 11: if mean cosine similarity is near zero,
    delta encoding saves nothing and the hypothesis dies here. That is a
    legitimate result obtained for the cost of one model load.

    delta_std_ratio is the operative number for compression: the standard
    deviation of (W_n - W_0) relative to that of W_0. Below 1.0 means the
    delta has a narrower spread than the weight itself and therefore
    quantises better at the same bit width. Near or above 1.0 means no gain.
    """
    classification = classify_layers(model)
    period = classification["block_period"]
    if not period:
        return {"status": "no_periodic_structure",
                "classification": classification}

    layers = _get_layer_list(model)
    n_blocks = len(layers) // period
    if n_blocks < 2:
        return {"status": "too_few_blocks", "classification": classification}

    results: List[Dict[str, Any]] = []

    for pos in range(period):
        ref_layer = layers[pos]
        ref_kind = ("attention" if pos in
                    [i % period for i in classification["attention_layers"]]
                    else "linear")

        for name, ref_mod in ref_layer.named_modules():
            W0 = getattr(ref_mod, "weight", None)
            if W0 is None or not torch.is_tensor(W0) or W0.dim() != 2:
                continue
            if W0.numel() < 1024:
                continue

            for b in range(1, n_blocks):
                idx = b * period + pos
                if idx >= len(layers):
                    break
                other = _get_module_by_name(layers[idx], name)
                Wn = getattr(other, "weight", None) if other is not None else None
                if Wn is None or Wn.shape != W0.shape:
                    continue

                results.append({
                    "position_in_block": pos,
                    "layer_kind": ref_kind,
                    "matrix": name,
                    "role": _matrix_role(name),
                    "block": b,
                    "ref_layer": pos,
                    "cmp_layer": idx,
                    **_compare_pair(W0, Wn, max_elements),
                })

    return {
        "status": "ok",
        "classification": classification,
        "period": period,
        "n_blocks": n_blocks,
        "pairs": results,
        "by_layer_kind": _aggregate_sim(results, "layer_kind"),
        "by_role": _aggregate_sim(results, "role"),
        "verdict": _similarity_verdict(results),
    }


def _get_module_by_name(root, name: str):
    obj = root
    for part in name.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _compare_pair(W0: torch.Tensor, Wn: torch.Tensor,
                  max_elements: int) -> Dict[str, Any]:
    with torch.no_grad():
        a = W0.detach().flatten().float()
        b = Wn.detach().flatten().float()

        if a.numel() > max_elements:
            step = a.numel() // max_elements
            a = a[::step][:max_elements]
            b = b[::step][:max_elements]

        cos = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
        delta = b - a
        std0 = a.std().item()
        dstd = delta.std().item()

        return {
            "cosine": round(cos, 5),
            "delta_std": round(dstd, 6),
            "ref_std": round(std0, 6),
            # < 1.0 means the delta quantises better than the weight itself
            "delta_std_ratio": round(dstd / std0, 4) if std0 > 0 else None,
            "relative_l2": round(
                (delta.norm() / a.norm()).item(), 5) if a.norm() > 0 else None,
        }


def _aggregate_sim(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)

    out = {}
    for g, items in groups.items():
        def mean_of(f):
            vals = [it[f] for it in items if it.get(f) is not None]
            return round(sum(vals) / len(vals), 5) if vals else None
        cosines = [it["cosine"] for it in items]
        out[g] = {
            "n_pairs": len(items),
            "mean_cosine": mean_of("cosine"),
            "max_cosine": round(max(cosines), 5) if cosines else None,
            "mean_delta_std_ratio": mean_of("delta_std_ratio"),
            "mean_relative_l2": mean_of("relative_l2"),
        }
    return out


def _similarity_verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Explicit pass/fail against the pre-registered kill criterion, so the
    outcome is not decided after seeing the numbers.
    """
    if not rows:
        return {"experiment_11": "indeterminate", "reason": "no pairs compared"}

    cosines = [r["cosine"] for r in rows]
    ratios = [r["delta_std_ratio"] for r in rows if r["delta_std_ratio"]]
    mean_cos = sum(cosines) / len(cosines)
    mean_ratio = sum(ratios) / len(ratios) if ratios else None

    if mean_cos < 0.1:
        status = "killed"
        reason = (f"mean cosine {mean_cos:.4f} is near zero: corresponding "
                  f"layers are effectively unrelated, so storing deltas saves "
                  f"nothing")
    elif mean_ratio and mean_ratio < 0.8:
        status = "promoted"
        reason = (f"mean cosine {mean_cos:.4f}, delta spread is "
                  f"{mean_ratio:.3f} of the weight's own: deltas should "
                  f"quantise better at equal bit width")
    else:
        status = "weak"
        reason = (f"mean cosine {mean_cos:.4f} but delta spread ratio "
                  f"{mean_ratio}: similarity exists without a compression "
                  f"advantage")

    return {"experiment_11": status, "mean_cosine": round(mean_cos, 5),
            "mean_delta_std_ratio": round(mean_ratio, 4) if mean_ratio else None,
            "reason": reason}


# ---------------------------------------------------------------------------
# Experiment 5 — rank profiling
# ---------------------------------------------------------------------------

def rank_profile(model, sample_matrices: int = 24,
                 max_dim: int = 4096) -> Dict[str, Any]:
    """
    Experiment 5: how much genuinely independent information does each weight
    matrix hold, and does it differ by layer type?

    The established result on dense models is that LLM weight matrices sit
    close to full rank, so plain SVD compression underperforms. Two things
    have not been checked on a hybrid model: whether GDN and attention layers
    differ, and (see rank_of_deltas below) whether the DIFFERENCES between
    blocks are low-rank even where the weights are not.

    energy_rank_90 is the count of singular values needed to capture 90% of
    the matrix's total energy, expressed as a fraction of full rank.

    CALIBRATION, verified on synthetic data: a random full-rank matrix scores
    ~0.51, NOT 1.0, because random matrices have a spread of singular values
    and 90% of energy already sits in about half the dimensions. A true rank-8
    matrix of size 256 scores ~0.027. So ~0.5 is the "no exploitable
    structure" baseline; values well below 0.5 indicate real compressibility.

    SVD is O(n^3); sampling rather than exhausting keeps this to minutes.
    """
    classification = classify_layers(model)
    layers = _get_layer_list(model)

    candidates = []
    for i, layer in enumerate(layers):
        kind = "attention" if i in classification["attention_layers"] else "linear"
        for name, module in layer.named_modules():
            W = getattr(module, "weight", None)
            if W is None or not torch.is_tensor(W) or W.dim() != 2:
                continue
            if min(W.shape) < 64 or max(W.shape) > max_dim:
                continue
            candidates.append((i, kind, name, W))

    # Even sample across the candidate list rather than the first N, which
    # would all come from early layers.
    if len(candidates) > sample_matrices:
        step = len(candidates) // sample_matrices
        candidates = candidates[::step][:sample_matrices]

    rows = []
    for i, kind, name, W in candidates:
        rows.append({
            "layer": i,
            "layer_kind": kind,
            "matrix": name,
            "role": _matrix_role(name),
            "shape": tuple(W.shape),
            **_svd_stats(W),
        })

    return {
        "classification": classification,
        "matrices": rows,
        "by_layer_kind": _aggregate_rank(rows, "layer_kind"),
        "by_role": _aggregate_rank(rows, "role"),
    }


def _svd_stats(W: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        M = W.detach().float()
        if M.is_cuda:
            M = M.cpu()
        try:
            s = torch.linalg.svdvals(M)
        except Exception as e:
            return {"svd_error": f"{type(e).__name__}: {e}"}

        total = (s ** 2).sum()
        if total <= 0:
            return {"svd_error": "zero-energy matrix"}

        cum = torch.cumsum(s ** 2, dim=0) / total
        full = len(s)

        def frac_for(threshold):
            k = int(torch.searchsorted(cum, torch.tensor(threshold)).item()) + 1
            return round(min(k, full) / full, 4)

        return {
            "full_rank": full,
            "energy_rank_50": frac_for(0.50),
            "energy_rank_90": frac_for(0.90),
            "energy_rank_99": frac_for(0.99),
            "condition_number": round((s[0] / s[-1]).item(), 2) if s[-1] > 0 else None,
            "top1_energy_share": round(((s[0] ** 2) / total).item(), 5),
        }


def _aggregate_rank(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if "svd_error" in r:
            continue
        groups.setdefault(r[key], []).append(r)

    out = {}
    for g, items in groups.items():
        def mean_of(f):
            vals = [it[f] for it in items if it.get(f) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        out[g] = {
            "n_matrices": len(items),
            "mean_energy_rank_90": mean_of("energy_rank_90"),
            "mean_energy_rank_99": mean_of("energy_rank_99"),
            "mean_top1_share": mean_of("top1_energy_share"),
        }
    return out


def rank_of_deltas(model, sample_positions: int = 4) -> Dict[str, Any]:
    """
    Experiment 5, second half. The question plain low-rank work does not ask.

    Weight matrices being near full rank does not imply their DIFFERENCES are.
    If (W_block_n - W_block_0) is low-rank even where W itself is not, then
    the model can be stored as one full block plus a set of skinny factors --
    a compression route that neither experiment 11 nor experiment 12 covers
    alone, and which only exists because of the repeating block structure.

    This composes experiments 1 and 5. It is the most novel item in Tier 1.
    """
    classification = classify_layers(model)
    period = classification["block_period"]
    if not period:
        return {"status": "no_periodic_structure"}

    layers = _get_layer_list(model)
    n_blocks = len(layers) // period
    if n_blocks < 2:
        return {"status": "too_few_blocks"}

    rows = []
    positions = list(range(min(period, sample_positions)))

    for pos in positions:
        for name, ref_mod in layers[pos].named_modules():
            W0 = getattr(ref_mod, "weight", None)
            if W0 is None or not torch.is_tensor(W0) or W0.dim() != 2:
                continue
            if min(W0.shape) < 64 or max(W0.shape) > 4096:
                continue

            other = _get_module_by_name(layers[pos + period], name)
            Wn = getattr(other, "weight", None) if other is not None else None
            if Wn is None or Wn.shape != W0.shape:
                continue

            with torch.no_grad():
                delta = (Wn.detach().float() - W0.detach().float())
                if delta.is_cuda:
                    delta = delta.cpu()

            w_stats = _svd_stats(W0)
            d_stats = _svd_stats(delta)

            if "svd_error" in w_stats or "svd_error" in d_stats:
                continue

            rows.append({
                "position_in_block": pos,
                "matrix": name,
                "role": _matrix_role(name),
                "shape": tuple(W0.shape),
                "weight_energy_rank_90": w_stats["energy_rank_90"],
                "delta_energy_rank_90": d_stats["energy_rank_90"],
                # < 1.0 means the delta is lower-rank than the weight, which
                # is the condition under which this route beats plain SVD.
                "rank_advantage": round(
                    d_stats["energy_rank_90"] / w_stats["energy_rank_90"], 4)
                    if w_stats["energy_rank_90"] > 0 else None,
            })

    advantages = [r["rank_advantage"] for r in rows if r["rank_advantage"]]
    mean_adv = sum(advantages) / len(advantages) if advantages else None

    return {
        "status": "ok",
        "comparisons": rows,
        "mean_rank_advantage": round(mean_adv, 4) if mean_adv else None,
        "verdict": (
            "promoted: deltas are markedly lower-rank than weights"
            if mean_adv and mean_adv < 0.7 else
            "killed: deltas are no lower-rank than the weights themselves"
            if mean_adv and mean_adv > 0.95 else
            "weak: modest rank advantage" if mean_adv else "indeterminate"
        ),
    }


# ---------------------------------------------------------------------------
# Experiment 18 — depth-axis frequency analysis
# ---------------------------------------------------------------------------

def depth_frequency_profile(model, sample_positions: int = 2) -> Dict[str, Any]:
    """
    Experiment 18, obtained almost free from the same tensor stacking as
    experiment 1.

    Stack the N instances of a given matrix position across blocks into a
    tensor with depth as the third axis, transform along depth, and measure
    how concentrated the energy is in low frequencies.

    The premise: if weights vary SMOOTHLY across the repeating blocks, most
    energy sits in low frequencies and a few coefficients could replace all N
    matrices. Predicted to fail -- depth-wise variation is more likely noisy
    than smooth, in which case energy spreads evenly across frequencies.

    low_freq_energy_share is the fraction of total energy in the lowest
    quarter of frequencies. Even spreading gives ~0.25 and means no structure
    to exploit. Values well above that indicate smooth depth-wise variation.

    Uses a real orthonormal DCT-II matrix rather than an FFT magnitude proxy,
    so the reported energy shares are exact.
    """
    classification = classify_layers(model)
    period = classification["block_period"]
    if not period:
        return {"status": "no_periodic_structure"}

    layers = _get_layer_list(model)
    n_blocks = len(layers) // period
    if n_blocks < 4:
        return {"status": "too_few_blocks_for_frequency_analysis"}

    rows = []
    for pos in range(min(period, sample_positions)):
        for name, ref_mod in layers[pos].named_modules():
            W0 = getattr(ref_mod, "weight", None)
            if W0 is None or not torch.is_tensor(W0) or W0.dim() != 2:
                continue
            if W0.numel() < 4096 or max(W0.shape) > 4096:
                continue

            stack = []
            ok = True
            for b in range(n_blocks):
                mod = _get_module_by_name(layers[b * period + pos], name)
                W = getattr(mod, "weight", None) if mod is not None else None
                if W is None or W.shape != W0.shape:
                    ok = False
                    break
                stack.append(W.detach().float().cpu().flatten())
            if not ok or len(stack) < 4:
                continue

            with torch.no_grad():
                # (n_blocks, n_weights): depth is the axis we transform along
                X = torch.stack(stack, dim=0)
                D = _dct_matrix(X.shape[0])
                coeffs = D @ X

                energy = coeffs.pow(2).sum(dim=1)
                total = energy.sum()
                if total <= 0:
                    continue

                k = max(1, X.shape[0] // 4)
                rows.append({
                    "position_in_block": pos,
                    "matrix": name,
                    "role": _matrix_role(name),
                    "n_blocks": X.shape[0],
                    "dc_energy_share": round((energy[0] / total).item(), 5),
                    "low_freq_energy_share": round(
                        (energy[:k].sum() / total).item(), 5),
                    "uniform_baseline": round(k / X.shape[0], 5),
                })

    shares = [r["low_freq_energy_share"] for r in rows]
    baselines = [r["uniform_baseline"] for r in rows]
    mean_share = sum(shares) / len(shares) if shares else None
    mean_base = sum(baselines) / len(baselines) if baselines else None

    return {
        "status": "ok",
        "matrices": rows,
        "mean_low_freq_share": round(mean_share, 5) if mean_share else None,
        "uniform_baseline": round(mean_base, 5) if mean_base else None,
        "verdict": (
            "promoted: energy concentrates in low frequencies"
            if mean_share and mean_base and mean_share > mean_base * 2 else
            "killed: energy spread roughly uniformly across depth frequencies"
            if mean_share and mean_base and mean_share < mean_base * 1.3 else
            "weak: mild low-frequency concentration"
            if mean_share else "indeterminate"
        ),
    }


def _dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix."""
    k = torch.arange(n).unsqueeze(1).float()
    i = torch.arange(n).unsqueeze(0).float()
    D = torch.cos(torch.pi * k * (2 * i + 1) / (2 * n))
    D[0] *= (1.0 / n) ** 0.5
    D[1:] *= (2.0 / n) ** 0.5
    return D


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_weight_analysis(
    base: str = "qwen-9b",
    experiments: Tuple[str, ...] = ("classify", "distributions",
                                    "similarity", "rank", "delta_rank",
                                    "depth_freq"),
    results_dir=None,
) -> Dict[str, Any]:
    """
    Load an UNQUANTISED model once and run the selected analyses.

    Quantised weights would give the quantisation grid rather than the true
    distribution, so backend is forced to fp16 regardless of what is
    configured elsewhere.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from .config import RunConfig, capture_environment
    from .loader import load, free_gpu

    cfg = RunConfig(
        experiment_id="T1-WEIGHTS",
        label="weight inspection (unquantised)",
        base=base,
        backend="fp16",
        attn_impl="sdpa",
        run_perf=False,
        run_standard_eval=False,
        notes="Experiments 1, 2, 5, 18. No forward pass; weights only.",
    )

    record: Dict[str, Any] = {
        "run_id": f"T1-WEIGHTS-{cfg.fingerprint()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "environment": capture_environment(),
        "provenance": "measured",
    }

    model = None
    try:
        model, tok, load_info = load(cfg)
        record["load"] = load_info
        del tok

        if "classify" in experiments:
            record["classification"] = classify_layers(model)
            print("classification:", record["classification"]["ratio"],
                  "linear:attention, period",
                  record["classification"]["block_period"])

        if "distributions" in experiments:
            print("exp 2: distributions ...")
            record["distributions"] = profile_distributions(model)

        if "similarity" in experiments:
            print("exp 1: cross-block similarity ...")
            record["similarity"] = cross_block_similarity(model)

        if "rank" in experiments:
            print("exp 5: rank profile ...")
            record["rank"] = rank_profile(model)

        if "delta_rank" in experiments:
            print("exp 5b: rank of deltas ...")
            record["delta_rank"] = rank_of_deltas(model)

        if "depth_freq" in experiments:
            print("exp 18: depth-axis frequency ...")
            record["depth_frequency"] = depth_frequency_profile(model)

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
