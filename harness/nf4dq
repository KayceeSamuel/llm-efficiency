"""
harness/nf4dq.py

PyTorch simulation of the NF4DQ block format, so it can be quality-gated on
real weights before any C or CUDA work is committed to.

Tensors stay in their original dtype and no memory is saved. That is
deliberate: it isolates the quality consequence of the format from its size
benefit, which are separate questions. Same approach as
validate.apply_scheme_in_place.

WHY THIS EXISTS

Two format settings were originally copied from bitsandbytes without being
tested: a 64-weight sub-block, and 8-bit uniform sub-block scales. A sweep at
byte-identical size (530 bytes per 1024 weights, 4.1406 bpw) found both were
on the wrong side of the trade:

    SUB=64, 8-bit uniform scales   0.093626   (the copied layout)
    SUB=32, 4-bit fitted codebook  0.088713   5.2% better

Halving the sub-block is worth 6.1%; dropping scales from 8 bits to 4 costs
0.5%. Both layouts spend the same 16 bytes on scales.

That 5.2% is reconstruction error, and reconstruction error has twice
predicted the wrong direction in this project: Hadamard rotation (6.74%
better, worse on accuracy AND perplexity) and endpoint-unpinned codebooks
(3.12% better, 1.22% worse perplexity). So this needs a perplexity gate.

PRE-REGISTERED, fixed before the numbers were seen:

    Promote v2 (SUB=32) over v1 (SUB=64) only if perplexity improves or is
    flat within 0.2%. If it degrades beyond that, revert to SUB=64 and record
    that finer scale granularity does not translate either.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import torch

from .codebook import NF4_LEVELS, estimate_chunk_rows


QK_NF4DQ = 1024        # superblock: largest that divides 5120 and 17408
NF4DQ_SUB = 32         # sub-block, measured rather than copied

# Sub-block absmax as a fraction of the superblock's largest sub-block absmax.
# Fitted by Lloyd-Max on a MIX of outlier-free and 16-sigma-outlier data.
#
# The mix matters. A codebook fitted on outlier-free data alone spanned only
# 0.3355 to 1.0 and failed badly when a superblock held an extreme weight: the
# remaining sub-blocks have ratios near 0.06, clamp to the floor, and their
# weights collapse into the lowest NF4 levels. Measured 0.139 against 0.101
# for this codebook. Experiment 2 found max/std of 15.4 and 16.8 on real
# weights, so that regime is the normal case here, not an edge case.
NF4DQ_SCALE_LEVELS = np.array([
    0.1126, 0.1387, 0.1647, 0.1973, 0.2485, 0.3740, 0.4436, 0.4997,
    0.5505, 0.5998, 0.6500, 0.7036, 0.7624, 0.8286, 0.9051, 1.0000],
    dtype=np.float64)


def _fp16(t: torch.Tensor) -> torch.Tensor:
    """Force a value through fp16, as the stored super-scale is.

    The decoder only sees the rounded number, so the encoder must quantise
    against the same one or the two disagree by the rounding error.
    """
    return t.to(torch.float16).to(t.dtype)


def nf4dq_roundtrip(x: torch.Tensor, sub: int = NF4DQ_SUB, qk: int = QK_NF4DQ,
                    scale_levels: Optional[np.ndarray] = None,
                    uniform_scale_bits: Optional[int] = None) -> torch.Tensor:
    """
    Quantise-dequantise a flat tensor through NF4DQ.

    `uniform_scale_bits` selects the v1 layout instead (uniform scales at that
    bit width), so both arms of the comparison run through one code path and
    cannot differ by anything incidental.
    """
    n = x.numel()
    if n % qk:
        raise ValueError(f"length {n} is not a multiple of superblock {qk}")

    nsub = qk // sub
    lv = torch.tensor(NF4_LEVELS, dtype=torch.float32, device=x.device)
    bd = (lv[:-1] + lv[1:]) / 2

    sb = x.reshape(-1, nsub, sub).float()
    absmax = sb.abs().amax(dim=2)
    top = absmax.amax(dim=1, keepdim=True)

    if uniform_scale_bits is None:
        sl = torch.tensor(
            NF4DQ_SCALE_LEVELS if scale_levels is None else scale_levels,
            dtype=torch.float32, device=x.device)
        sbd = (sl[:-1] + sl[1:]) / 2
        d = _fp16(top)
        ratio = torch.where(d > 0, absmax / d.clamp_min(1e-30),
                            torch.zeros_like(absmax))
        scale = (d * sl[torch.bucketize(ratio, sbd)]).unsqueeze(-1)
    else:
        n_lv = 2 ** uniform_scale_bits - 1
        d = _fp16(top / n_lv)
        sc = torch.where(d > 0, (absmax / d.clamp_min(1e-30)).round(),
                         torch.zeros_like(absmax)).clamp(0, n_lv)
        scale = (d * sc).unsqueeze(-1)

    q = torch.where(scale > 0, sb / scale.clamp_min(1e-30),
                    torch.zeros_like(sb))
    return (lv[torch.bucketize(q, bd)] * scale).reshape(-1).to(x.dtype)


def apply_nf4dq_(model, sub: int = NF4DQ_SUB, qk: int = QK_NF4DQ,
                 uniform_scale_bits: Optional[int] = None,
                 scale_levels: Optional[np.ndarray] = None,
                 chunk_rows: Optional[int] = None,
                 min_numel: int = 4096) -> Dict[str, Any]:
    """
    Overwrite every eligible 2-D weight with its NF4DQ reconstruction.

    IRREVERSIBLE. Reload between arms.

    Processed in row chunks so peak memory does not depend on tensor size: a
    full fp32 promotion of a 248,320 x 4,096 embedding is 3.79 GB and will not
    fit alongside a loaded model. That bug has now appeared three times in
    this project, hence the shared helper in codebook.py.

    Tensors whose width is not a multiple of the superblock are skipped and
    counted, never silently mangled. On Qwen3.x-27B nothing is skipped: 5120
    and 17408 both divide by 1024.
    """
    stats: Dict[str, Any] = {
        "sub": sub, "qk": qk,
        "layout": "v1_uniform" if uniform_scale_bits else "v2_codebook",
        "scale_bits": uniform_scale_bits or 4,
        "matrices_modified": 0, "params_modified": 0,
        "skipped_shape": 0, "skipped_small": 0, "per_tensor": {},
    }
    nbytes = qk / 2 + (qk / sub) * (stats["scale_bits"] / 8) + 2
    stats["bytes_per_block"] = nbytes
    stats["bpw"] = round(nbytes * 8 / qk, 4)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim != 2:
                continue
            if p.numel() < min_numel:
                stats["skipped_small"] += 1
                continue
            if p.shape[-1] % qk:
                stats["skipped_shape"] += 1
                continue

            rows = chunk_rows or estimate_chunk_rows(p.shape[-1], p.device)
            num = den = 0.0
            for i in range(0, p.shape[0], rows):
                sl_ = p.data[i:i + rows]
                orig = sl_.float()
                recon = nf4dq_roundtrip(
                    sl_.reshape(-1), sub=sub, qk=qk,
                    scale_levels=scale_levels,
                    uniform_scale_bits=uniform_scale_bits).reshape(sl_.shape)
                num += float((orig - recon.float()).pow(2).sum())
                den += float(orig.pow(2).sum())
                sl_.copy_(recon.to(p.dtype))
                del orig, recon

            err = (num ** 0.5) / (den ** 0.5) if den > 0 else 0.0
            stats["per_tensor"][name] = round(err, 6)
            stats["matrices_modified"] += 1
            stats["params_modified"] += int(p.numel())

    if stats["matrices_modified"] == 0:
        raise RuntimeError(
            "no matrices modified. A silent no-op would report the unmodified "
            "model's perplexity as the format's, which is the same failure "
            "class as the KV extractor returning 0.0 GB.")

    errs = list(stats["per_tensor"].values())
    stats["mean_rel_error"] = round(float(np.mean(errs)), 6)
    stats["max_rel_error"] = round(float(np.max(errs)), 6)
    return stats


def fit_scale_codebook(model, sub: int = NF4DQ_SUB, qk: int = QK_NF4DQ,
                       k: int = 16, max_samples: int = 2_000_000,
                       chunk_rows: Optional[int] = None) -> np.ndarray:
    """
    Refit the sub-block scale codebook on this model's real weights.

    NF4DQ_SCALE_LEVELS was fitted on synthetic data chosen to match measured
    kurtosis and outlier statistics. Fitting on the actual tensors should do
    at least as well, and tells us how good the synthetic proxy was.
    """
    from .codebook import fit_codebook

    out, taken, nsub = [], 0, qk // sub
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (p.ndim != 2 or p.shape[-1] % qk
                    or "embed" in name or "lm_head" in name):
                continue
            rows = chunk_rows or estimate_chunk_rows(p.shape[-1], p.device)
            for i in range(0, p.shape[0], rows):
                sb = p.data[i:i + rows].reshape(-1, nsub, sub).float()
                am = sb.abs().amax(dim=2)
                top = am.amax(dim=1, keepdim=True).clamp_min(1e-30)
                out.append((am / top).reshape(-1).cpu().numpy())
                taken += out[-1].size
                if taken >= max_samples:
                    break
            if taken >= max_samples:
                break

    if not out:
        raise RuntimeError("no eligible tensors; check the superblock size "
                           "divides this model's row widths")

    r = np.concatenate(out)[:max_samples].astype(np.float64)
    # ratios live in (0, 1] with exactly one equal to 1 per superblock, so pin
    # the top level and leave the floor free to reach however low it needs
    lv = fit_codebook(r, k=k, pin_zero=False, pin_ends=False)
    lv[-1] = 1.0
    return np.sort(lv)
