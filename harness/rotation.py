"""
rotation.py — Experiment 7: Hadamard rotation before quantisation.

MOTIVATION, from measured results in this project:
Experiment 2 found the most extreme weight in a typical matrix sits ~16
standard deviations from the mean, with roughly 0.1% of weights beyond 4
sigma. That is the classic quantisation problem: a 4-bit grid has 16 levels,
and if the scale must stretch to cover a 16-sigma outlier, the bulk of the
weights clustered near zero are squeezed into a handful of levels.

Experiment 2 also found GDN and attention layers have near-identical
distributions (kurtosis 1.50 vs 1.38), which killed the motivation for
splitting precision by layer type. Rotation attacks the outliers directly
instead, which is why it became the best-supported next step.

MECHANISM:
Multiply W by an orthogonal matrix Q before quantising and by Q^T after.
Q Q^T = I, so this is mathematically an identity -- but the rotated weights
have outlier magnitude spread across many entries, so the blockwise scale
narrows and every weight gets more of the grid. Hadamard matrices are the
practical choice: entries are +/-1 (exactly representable, no precision lost
in forming the rotation) and the transform is fast.

SCOPE -- read this before interpreting results:
This measures whether rotation reduces WEIGHT RECONSTRUCTION ERROR on this
model's actual weights. It does NOT implement QuaRot/SpinQuant end to end.
A deployable implementation must absorb the rotations into adjacent layers
(exploiting RMSNorm's computational invariance) so no rotation is performed
at inference time. That is real engineering, and it is only worth doing if
the premise tested here holds. This is the cheap falsification step first.

Validated on synthetic data before use: on a matrix seeded with 16-sigma
outliers, rotation reduced max/std from 33.7 to 4.4 and cut int4 relative
error by 16.9%. On a matched matrix WITHOUT outliers, the gain was 0.2%,
confirming the effect is specifically outlier-driven rather than an artifact
of the measurement.
"""

import gc
import math
from typing import Dict, Any, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Hadamard construction
# ---------------------------------------------------------------------------

def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Sylvester-construction Hadamard matrix, normalised to be orthogonal.

    Requires n to be a power of 2. For other sizes use
    block_diagonal_hadamard, which is what production implementations do.
    """
    if n & (n - 1) != 0:
        raise ValueError(f"Sylvester construction needs a power of 2, got {n}")

    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)

    return H / math.sqrt(n)


def largest_pow2_divisor(n: int) -> int:
    """Largest power of 2 that divides n."""
    return n & (-n)


def block_diagonal_hadamard(n: int, device=None,
                            dtype=torch.float32) -> Tuple[torch.Tensor, int]:
    """
    For dimensions that are not powers of 2, apply a Hadamard of size k
    block-wise, where k is the largest power-of-2 divisor of n.

    Still exactly orthogonal, so the transform remains lossless. Mixing is
    confined within blocks of size k rather than spanning the full dimension,
    so outlier spreading is weaker -- the block size is reported alongside
    every result so this limitation is visible rather than hidden.
    """
    k = largest_pow2_divisor(n)
    if k < 2:
        raise ValueError(f"Dimension {n} has no usable power-of-2 factor")
    return hadamard_matrix(k, device=device, dtype=dtype), k


def apply_blockwise_hadamard(X: torch.Tensor, H: torch.Tensor,
                             k: int, dim: int) -> torch.Tensor:
    """
    Apply a size-k Hadamard block-wise along `dim` of X.

    Reshapes so the last axis has length k, applies H, reshapes back. Avoids
    materialising a full n x n block-diagonal matrix.
    """
    X = X.transpose(dim, -1)
    shape = X.shape
    X = X.reshape(-1, k)
    X = X @ H.T
    X = X.reshape(shape).transpose(dim, -1)
    return X


# ---------------------------------------------------------------------------
# Quantisation simulation
# ---------------------------------------------------------------------------

def quantize_dequantize(W: torch.Tensor, bits: int = 4,
                        block_size: int = 64) -> torch.Tensor:
    """
    Blockwise symmetric absmax quantisation, simulated in floating point.

    This is the standard baseline used for rotation ablations. It is NOT
    bitsandbytes NF4: NF4 uses a normal-distributed codebook rather than a
    uniform grid. Uniform absmax is used here deliberately, because it makes
    the outlier effect visible in isolation -- NF4's codebook already
    partially compensates for the same problem, which would confound the
    comparison.

    Nothing is actually packed into 4-bit storage; the point is to measure
    the reconstruction error a real 4-bit format would incur.
    """
    orig_shape = W.shape
    flat = W.detach().flatten().float()

    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])

    blocks = flat.view(-1, block_size)

    qmax = 2 ** (bits - 1) - 1        # e.g. 7 for 4-bit
    qmin = -(2 ** (bits - 1))         # e.g. -8 for 4-bit

    scale = blocks.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.full_like(scale, 1e-12), scale)

    q = torch.clamp(torch.round(blocks / scale), qmin, qmax)
    deq = (q * scale).flatten()

    if pad:
        deq = deq[:-pad]

    return deq.view(orig_shape)


def relative_error(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    """Relative Frobenius error. Scale-free, so comparable across matrices."""
    num = torch.linalg.norm((W - W_hat).float())
    den = torch.linalg.norm(W.float())
    return (num / den).item() if den > 0 else float("nan")


def outlier_stats(W: torch.Tensor) -> Dict[str, float]:
    flat = W.detach().flatten().float()
    std = flat.std()
    if std == 0:
        return {"max_over_std": 0.0, "kurtosis": 0.0, "outlier_ratio": 0.0}
    z = (flat - flat.mean()) / std
    return {
        "max_over_std": round(z.abs().max().item(), 3),
        "kurtosis": round((z.pow(4).mean() - 3.0).item(), 4),
        "outlier_ratio": round((z.abs() > 4).float().mean().item(), 8),
    }


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_rotation(W: torch.Tensor, bits: int = 4,
                     block_size: int = 64) -> Dict[str, Any]:
    """
    Quantise one matrix with and without Hadamard rotation, and compare.

    Baseline:  W -> quantise -> W_hat
    Rotated:   W -> H W H^T -> quantise -> H^T (.) H -> W_hat

    Both are measured against the same original W, so the comparison is like
    for like. The rotated path includes the inverse rotation, so any error it
    introduces is counted against it rather than hidden.
    """
    with torch.no_grad():
        Wf = W.detach().float()
        if Wf.is_cuda:
            Wf = Wf.cpu()

        out_rows, out_cols = Wf.shape

        # Baseline
        W_base = quantize_dequantize(Wf, bits, block_size)
        err_base = relative_error(Wf, W_base)

        # Rotation, applied on both axes
        try:
            H_r, k_r = block_diagonal_hadamard(out_rows, dtype=torch.float32)
            H_c, k_c = block_diagonal_hadamard(out_cols, dtype=torch.float32)
        except ValueError as e:
            return {"status": "skipped", "reason": str(e)}

        Wr = apply_blockwise_hadamard(Wf, H_r, k_r, dim=0)
        Wr = apply_blockwise_hadamard(Wr, H_c, k_c, dim=1)

        stats_before = outlier_stats(Wf)
        stats_after = outlier_stats(Wr)

        Wr_q = quantize_dequantize(Wr, bits, block_size)

        # Invert: Hadamard is symmetric and orthogonal, so applying it again
        # undoes it (H H = I after normalisation).
        W_rec = apply_blockwise_hadamard(Wr_q, H_c.T, k_c, dim=1)
        W_rec = apply_blockwise_hadamard(W_rec, H_r.T, k_r, dim=0)

        err_rot = relative_error(Wf, W_rec)

        # Sanity: rotating and inverting WITHOUT quantising must be lossless.
        # If this is not ~0, the transform is wrong and the comparison is
        # meaningless, so it is checked rather than assumed.
        W_id = apply_blockwise_hadamard(Wr, H_c.T, k_c, dim=1)
        W_id = apply_blockwise_hadamard(W_id, H_r.T, k_r, dim=0)
        roundtrip_err = relative_error(Wf, W_id)

        improvement = (1 - err_rot / err_base) if err_base > 0 else None

        return {
            "status": "ok",
            "shape": tuple(Wf.shape),
            "bits": bits,
            "block_size": block_size,
            "hadamard_block_rows": k_r,
            "hadamard_block_cols": k_c,
            "full_rotation": (k_r == out_rows and k_c == out_cols),
            "err_baseline": round(err_base, 6),
            "err_rotated": round(err_rot, 6),
            "improvement": round(improvement, 5) if improvement is not None else None,
            "roundtrip_err": round(roundtrip_err, 9),
            "max_over_std_before": stats_before["max_over_std"],
            "max_over_std_after": stats_after["max_over_std"],
            "kurtosis_before": stats_before["kurtosis"],
            "kurtosis_after": stats_after["kurtosis"],
            "outlier_ratio_before": stats_before["outlier_ratio"],
            "outlier_ratio_after": stats_after["outlier_ratio"],
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_rotation_experiment(
    base: str = "qwen-9b",
    bits_list: Tuple[int, ...] = (4, 3),
    block_size: int = 64,
    sample_matrices: int = 32,
    results_dir=None,
) -> Dict[str, Any]:
    """
    Experiment 7 on real model weights.

    Requires an UNQUANTISED model: bitsandbytes stores NF4 weights in a packed
    opaque layout, so reading them back would give the quantisation grid
    rather than the underlying values this experiment needs.

    Samples matrices evenly across depth rather than taking the first N, which
    would draw entirely from early layers.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from .config import RunConfig, capture_environment
    from .loader import load, free_gpu
    from .weights import classify_layers, _get_layer_list, _matrix_role

    cfg = RunConfig(
        experiment_id="T2-07",
        label="Hadamard rotation before quantisation",
        base=base,
        backend="fp16",
        attn_impl="sdpa",
        run_perf=False,
        run_standard_eval=False,
        notes="Weight reconstruction error only. Does not implement "
              "inference-time rotation absorption.",
    )

    record: Dict[str, Any] = {
        "run_id": f"T2-07-{cfg.fingerprint()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "environment": capture_environment(),
        "provenance": "measured",
        "scope_note": (
            "Measures weight reconstruction error under simulated blockwise "
            "symmetric int-N quantisation. Not an end-to-end QuaRot "
            "implementation; no inference-time rotation absorption."
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

        all_rows: List[Dict[str, Any]] = []
        for bits in bits_list:
            print(f"  {bits}-bit: {len(candidates)} matrices ...")
            for i, kind, name, W in candidates:
                res = compare_rotation(W, bits=bits, block_size=block_size)
                if res.get("status") != "ok":
                    continue
                res.update({
                    "layer": i,
                    "layer_kind": kind,
                    "matrix": name,
                    "role": _matrix_role(name),
                })
                all_rows.append(res)

        record["comparisons"] = all_rows
        record["summary"] = _summarise(all_rows)
        record["verdict"] = _verdict(all_rows)
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


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group by bit width, then by layer kind and matrix role."""
    out: Dict[str, Any] = {}

    for bits in sorted({r["bits"] for r in rows}):
        subset = [r for r in rows if r["bits"] == bits]
        key = f"{bits}bit"
        out[key] = {
            "n_matrices": len(subset),
            "mean_err_baseline": _mean(subset, "err_baseline"),
            "mean_err_rotated": _mean(subset, "err_rotated"),
            "mean_improvement": _mean(subset, "improvement"),
            "n_improved": sum(1 for r in subset if (r["improvement"] or 0) > 0),
            "mean_max_over_std_before": _mean(subset, "max_over_std_before"),
            "mean_max_over_std_after": _mean(subset, "max_over_std_after"),
            "max_roundtrip_err": max(r["roundtrip_err"] for r in subset),
            "by_layer_kind": _group(subset, "layer_kind"),
            "by_role": _group(subset, "role"),
            "full_rotation_only": _group(
                [r for r in subset if r["full_rotation"]], "layer_kind"),
        }
    return out


def _mean(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _group(rows, key):
    groups: Dict[str, List] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {
        g: {
            "n": len(items),
            "mean_improvement": _mean(items, "improvement"),
            "mean_err_baseline": _mean(items, "err_baseline"),
            "mean_err_rotated": _mean(items, "err_rotated"),
        }
        for g, items in groups.items()
    }


def _verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pre-registered decision. Thresholds are set here, in code, rather than
    chosen after seeing the numbers.

    A first check on roundtrip error guards the whole result: if rotating and
    un-rotating without quantisation is not lossless, the transform is
    implemented incorrectly and every improvement figure is meaningless.
    """
    if not rows:
        return {"experiment_7": "indeterminate", "reason": "no comparisons"}

    worst_roundtrip = max(r["roundtrip_err"] for r in rows)
    if worst_roundtrip > 1e-4:
        return {
            "experiment_7": "invalid",
            "reason": (f"rotation round-trip error {worst_roundtrip:.2e} is "
                       f"not negligible; the transform is not orthogonal as "
                       f"implemented, so improvement figures cannot be trusted"),
        }

    four_bit = [r for r in rows if r["bits"] == 4]
    target = four_bit if four_bit else rows
    imps = [r["improvement"] for r in target if r["improvement"] is not None]
    if not imps:
        return {"experiment_7": "indeterminate", "reason": "no improvements computed"}

    mean_imp = sum(imps) / len(imps)
    frac_improved = sum(1 for i in imps if i > 0) / len(imps)

    if mean_imp > 0.05 and frac_improved > 0.8:
        status = "promoted"
        reason = (f"mean {mean_imp*100:.1f}% error reduction at 4-bit, "
                  f"improving on {frac_improved*100:.0f}% of matrices: worth "
                  f"building the inference-time absorption")
    elif mean_imp > 0.01:
        status = "weak"
        reason = (f"mean {mean_imp*100:.1f}% error reduction: real but small; "
                  f"weigh against the engineering cost of absorption")
    else:
        status = "killed"
        reason = (f"mean {mean_imp*100:.1f}% change: rotation does not "
                  f"meaningfully reduce quantisation error on these weights")

    return {
        "experiment_7": status,
        "mean_improvement_4bit": round(mean_imp, 5),
        "fraction_improved": round(frac_improved, 4),
        "max_roundtrip_err": worst_roundtrip,
        "reason": reason,
    }
