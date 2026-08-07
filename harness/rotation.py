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


def random_orthogonal(m: int, seed: int = 0, device=None,
                      dtype=torch.float32) -> torch.Tensor:
    """
    Deterministic random orthogonal matrix via QR.

    Sign-corrected against the diagonal of R so the result is reproducible
    across runs rather than depending on LAPACK's arbitrary sign convention.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, m, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(dtype=dtype, device=device)


def kronecker_rotation_factors(n: int, seed: int = 0, device=None,
                               dtype=torch.float32):
    """
    Build factors for an orthogonal rotation of ANY dimension n.

    Factorise n = k * m where k is the largest power-of-2 divisor. The
    rotation is the Kronecker product H_k (x) Q_m, which is orthogonal
    whenever both factors are, and mixes across the WHOLE dimension rather
    than only within blocks of size k.

    This resolves the confound in the first version of this experiment.
    Previously, dimensions that were not powers of 2 received only
    block-diagonal mixing, which is strictly weaker. Since FFN matrices
    happened to have non-power-of-2 dimensions while GDN projections did not,
    the apparent "rotation helps GDN 3x more than FFN" result conflated a
    genuine role effect with an artifact of unequal rotation strength.

    Verified on synthetic outlier-seeded matrices: at n=4864 (256 x 19),
    blockwise mixing reduced max/std from 55.7 to 6.4, Kronecker to 5.1.

    The n x n matrix is never materialised -- see apply_kronecker_rotation.
    """
    k = largest_pow2_divisor(n)
    m = n // k
    if k < 2:
        raise ValueError(f"Dimension {n} has no usable power-of-2 factor")
    H_k = hadamard_matrix(k, device=device, dtype=dtype)
    Q_m = (random_orthogonal(m, seed=seed, device=device, dtype=dtype)
           if m > 1 else torch.ones(1, 1, dtype=dtype, device=device))
    return H_k, Q_m, k, m


def apply_kronecker_rotation(X: torch.Tensor, H_k: torch.Tensor,
                             Q_m: torch.Tensor, k: int, m: int,
                             dim: int) -> torch.Tensor:
    """
    Apply (H_k (x) Q_m) along `dim` without forming the n x n product.

    Reshape the target axis into (k, m), mix the m-axis with Q_m and the
    k-axis with H_k. Forming the full Kronecker product would need n^2
    entries -- at n=11008 that is ~485 MB per rotation in fp32, which is how
    the first attempt at this ran out of memory.
    """
    X = X.transpose(dim, -1)
    shape = X.shape
    X = X.reshape(-1, k, m)
    if m > 1:
        X = X @ Q_m.T
    X = torch.einsum("ij,bjm->bim", H_k, X)
    return X.reshape(shape).transpose(dim, -1)


def block_diagonal_hadamard(n: int, device=None,
                            dtype=torch.float32) -> Tuple[torch.Tensor, int]:
    """
    For dimensions that are not powers of 2, apply a Hadamard of size k
    block-wise, where k is the largest power-of-2 divisor of n.

    Still exactly orthogonal, so the transform remains lossless. Mixing is
    confined within blocks of size k rather than spanning the full dimension,
    so outlier spreading is weaker.

    RETAINED AS A COMPARISON ARM ONLY. kronecker_rotation_factors is the
    correct construction; this one is kept so the strength difference between
    the two can be measured rather than assumed.
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

ROTATION_SCHEMES = ("none", "blockwise_hadamard", "kronecker", "random_orthogonal")


def _rotate(Wf: torch.Tensor, scheme: str, seed: int = 0):
    """
    Return (rotated, inverse_fn) for a given scheme.

    Every scheme is exactly orthogonal, so the inverse is always available and
    the round-trip is lossless up to floating-point precision. That is
    asserted per matrix rather than assumed.
    """
    rows, cols = Wf.shape

    if scheme == "none":
        return Wf, lambda X: X

    if scheme == "blockwise_hadamard":
        H_r, k_r = block_diagonal_hadamard(rows)
        H_c, k_c = block_diagonal_hadamard(cols)
        Wr = apply_blockwise_hadamard(Wf, H_r, k_r, dim=0)
        Wr = apply_blockwise_hadamard(Wr, H_c, k_c, dim=1)

        def inv(X):
            X = apply_blockwise_hadamard(X, H_c.T, k_c, dim=1)
            return apply_blockwise_hadamard(X, H_r.T, k_r, dim=0)

        return Wr, inv

    if scheme == "kronecker":
        Hr, Qr, kr, mr = kronecker_rotation_factors(rows, seed=seed)
        Hc, Qc, kc, mc = kronecker_rotation_factors(cols, seed=seed + 1)
        Wr = apply_kronecker_rotation(Wf, Hr, Qr, kr, mr, dim=0)
        Wr = apply_kronecker_rotation(Wr, Hc, Qc, kc, mc, dim=1)

        def inv(X):
            X = apply_kronecker_rotation(X, Hc.T, Qc.T, kc, mc, dim=1)
            return apply_kronecker_rotation(X, Hr.T, Qr.T, kr, mr, dim=0)

        return Wr, inv

    if scheme == "random_orthogonal":
        # CONTROL. Tests whether Hadamard structure matters at all, or
        # whether any orthogonal rotation flattens outliers equally well.
        # If this matches kronecker, Hadamard's advantage is purely
        # computational speed, not quantisation quality -- which changes what
        # the finding is.
        # Skipped on large dimensions: forming an n x n dense orthogonal
        # matrix is O(n^2) memory and O(n^3) to construct.
        if max(rows, cols) > 2048:
            return None, None
        Qr = random_orthogonal(rows, seed=seed + 2)
        Qc = random_orthogonal(cols, seed=seed + 3)
        Wr = Qr @ Wf @ Qc.T
        return Wr, (lambda X: Qr.T @ X @ Qc)

    raise ValueError(f"Unknown rotation scheme: {scheme}")


def compare_rotation(W: torch.Tensor, bits: int = 4, block_size: int = 64,
                     schemes: Tuple[str, ...] = ROTATION_SCHEMES,
                     seed: int = 0) -> Dict[str, Any]:
    """
    Quantise one matrix under several rotation schemes and compare.

    For each scheme:  W -> rotate -> quantise -> un-rotate -> W_hat
    All measured against the same original W, so comparisons are like for
    like. The inverse rotation is inside the measured path, so any error it
    introduces counts against the scheme rather than being hidden.

    The "none" arm is the baseline. Improvement is reported relative to it.
    """
    with torch.no_grad():
        Wf = W.detach().float()
        if Wf.is_cuda:
            Wf = Wf.cpu()

        rows, cols = Wf.shape
        results: Dict[str, Any] = {}
        err_baseline = None

        for scheme in schemes:
            try:
                Wr, inv = _rotate(Wf, scheme, seed=seed)
            except ValueError as e:
                results[scheme] = {"status": "skipped", "reason": str(e)}
                continue

            if Wr is None:
                results[scheme] = {"status": "skipped",
                                   "reason": "dimension too large for dense control"}
                continue

            # Round-trip WITHOUT quantisation must be lossless. If it is not,
            # the transform is wrong and any improvement figure is spurious.
            roundtrip_err = relative_error(Wf, inv(Wr))

            Wq = quantize_dequantize(Wr, bits, block_size)
            W_hat = inv(Wq)
            err = relative_error(Wf, W_hat)

            if scheme == "none":
                err_baseline = err

            stats = outlier_stats(Wr)
            results[scheme] = {
                "status": "ok",
                "err": round(err, 6),
                "roundtrip_err": round(roundtrip_err, 9),
                "max_over_std": stats["max_over_std"],
                "kurtosis": stats["kurtosis"],
                "outlier_ratio": stats["outlier_ratio"],
            }

        # Improvements relative to the unrotated baseline.
        if err_baseline:
            for scheme, r in results.items():
                if r.get("status") == "ok" and scheme != "none":
                    r["improvement"] = round(1 - r["err"] / err_baseline, 5)

        k_r = largest_pow2_divisor(rows)
        k_c = largest_pow2_divisor(cols)

        return {
            "status": "ok",
            "shape": (rows, cols),
            "bits": bits,
            "block_size": block_size,
            # Recorded so the old confound stays visible: these flag which
            # matrices the blockwise arm could only partially rotate.
            "pow2_rows": k_r == rows,
            "pow2_cols": k_c == cols,
            "blockwise_full": (k_r == rows and k_c == cols),
            "kron_factors": f"{k_r}x{rows//k_r} , {k_c}x{cols//k_c}",
            "schemes": results,
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
            print(f"  {bits}-bit: {len(candidates)} matrices x "
                  f"{len(ROTATION_SCHEMES)} schemes ...")
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
    """Group by bit width, then by rotation scheme, layer kind, and role."""
    out: Dict[str, Any] = {}

    for bits in sorted({r["bits"] for r in rows}):
        subset = [r for r in rows if r["bits"] == bits]
        per_scheme = {}

        for scheme in ROTATION_SCHEMES:
            entries = [(r, r["schemes"].get(scheme, {})) for r in subset]
            ok = [(r, e) for r, e in entries if e.get("status") == "ok"]
            if not ok:
                per_scheme[scheme] = {"n": 0, "note": "no successful runs"}
                continue

            imps = [e["improvement"] for _, e in ok if e.get("improvement") is not None]
            per_scheme[scheme] = {
                "n": len(ok),
                "mean_err": round(sum(e["err"] for _, e in ok) / len(ok), 6),
                "mean_max_over_std": round(
                    sum(e["max_over_std"] for _, e in ok) / len(ok), 3),
                "max_roundtrip_err": max(e["roundtrip_err"] for _, e in ok),
                "mean_improvement": round(sum(imps)/len(imps), 5) if imps else None,
                "n_improved": sum(1 for i in imps if i > 0) if imps else None,
                "by_role": _group_scheme(ok, "role"),
                "by_layer_kind": _group_scheme(ok, "layer_kind"),
            }

        # THE CONFOUND CHECK. Splits the blockwise arm by whether the matrix
        # had power-of-2 dimensions. If blockwise only helps the pow2 group
        # while kronecker helps both, the original "GDN beats FFN" result was
        # an artifact of unequal rotation strength, not a role effect.
        conf = {}
        for label, pred in [("pow2_dims", lambda r: r["blockwise_full"]),
                            ("non_pow2_dims", lambda r: not r["blockwise_full"])]:
            grp = [r for r in subset if pred(r)]
            if not grp:
                continue
            conf[label] = {"n_matrices": len(grp)}
            for scheme in ("blockwise_hadamard", "kronecker"):
                vals = [r["schemes"][scheme]["improvement"]
                        for r in grp
                        if r["schemes"].get(scheme, {}).get("improvement") is not None]
                conf[label][scheme] = round(sum(vals)/len(vals), 5) if vals else None
            conf[label]["roles"] = sorted({r["role"] for r in grp})

        out[f"{bits}bit"] = {
            "n_matrices": len(subset),
            "schemes": per_scheme,
            "confound_check": conf,
        }
    return out


def _group_scheme(ok_pairs, key):
    groups = {}
    for r, e in ok_pairs:
        groups.setdefault(r[key], []).append(e)
    res = {}
    for g, entries in groups.items():
        imps = [e["improvement"] for e in entries if e.get("improvement") is not None]
        res[g] = {
            "n": len(entries),
            "mean_err": round(sum(e["err"] for e in entries)/len(entries), 6),
            "mean_improvement": round(sum(imps)/len(imps), 5) if imps else None,
            "mean_max_over_std": round(
                sum(e["max_over_std"] for e in entries)/len(entries), 3),
        }
    return res


def _verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pre-registered decision, thresholds fixed in code before seeing results.

    Guarded first on round-trip error: if rotating and un-rotating without
    quantisation is not lossless, the transform is implemented incorrectly and
    every improvement figure is meaningless.
    """
    if not rows:
        return {"experiment_7": "indeterminate", "reason": "no comparisons"}

    four = [r for r in rows if r["bits"] == 4] or rows

    worst_rt = 0.0
    for r in four:
        for e in r["schemes"].values():
            if e.get("status") == "ok":
                worst_rt = max(worst_rt, e["roundtrip_err"])
    if worst_rt > 1e-4:
        return {"experiment_7": "invalid",
                "reason": (f"round-trip error {worst_rt:.2e} is not "
                           f"negligible; transform not orthogonal as "
                           f"implemented")}

    def scheme_imps(scheme):
        return [r["schemes"][scheme]["improvement"] for r in four
                if r["schemes"].get(scheme, {}).get("improvement") is not None]

    kron = scheme_imps("kronecker")
    block = scheme_imps("blockwise_hadamard")
    rand = scheme_imps("random_orthogonal")

    if not kron:
        return {"experiment_7": "indeterminate", "reason": "kronecker arm empty"}

    mean_kron = sum(kron)/len(kron)
    frac = sum(1 for i in kron if i > 0)/len(kron)
    mean_block = sum(block)/len(block) if block else None
    mean_rand = sum(rand)/len(rand) if rand else None

    if mean_kron > 0.05 and frac > 0.8:
        status = "promoted"
        reason = (f"kronecker rotation gives {mean_kron*100:.1f}% mean error "
                  f"reduction at 4-bit on {frac*100:.0f}% of matrices")
    elif mean_kron > 0.01:
        status = "weak"
        reason = (f"{mean_kron*100:.1f}% mean reduction: real but small "
                  f"against the cost of inference-time absorption")
    else:
        status = "killed"
        reason = (f"{mean_kron*100:.1f}% change: rotation does not "
                  f"meaningfully reduce quantisation error here")

    # Does Hadamard structure matter, or would any rotation do? If the random
    # orthogonal control matches, Hadamard's advantage is speed, not quality,
    # and the finding should be stated that way.
    if mean_rand is not None:
        gap = mean_kron - mean_rand
        if abs(gap) < 0.01:
            structure = ("Hadamard structure gives no quality advantage over a "
                         "generic orthogonal rotation; its benefit is "
                         "computational speed only")
        elif gap > 0:
            structure = (f"Hadamard-based rotation beats a random orthogonal "
                         f"control by {gap*100:.1f} points")
        else:
            structure = (f"random orthogonal BEATS Hadamard by "
                         f"{-gap*100:.1f} points -- unexpected, investigate")
    else:
        structure = "random orthogonal control not run (dimensions too large)"

    return {
        "experiment_7": status,
        "mean_improvement_kronecker_4bit": round(mean_kron, 5),
        "mean_improvement_blockwise_4bit": round(mean_block, 5) if mean_block else None,
        "mean_improvement_random_orth_4bit": round(mean_rand, 5) if mean_rand else None,
        "fraction_improved": round(frac, 4),
        "max_roundtrip_err": worst_rt,
        "reason": reason,
        "structure_question": structure,
    }
