"""
harness/codebook.py

Arbitrary-codebook quantisation, applied in row chunks so that peak memory
does not depend on tensor size.

WHY THIS MODULE EXISTS

Two things motivated it.

1. A real capability gap. NF4's sixteen levels are the quantiles of a normal
   distribution, but the quantiser never sees a normal distribution: every
   block is divided by its own absmax first, so what is actually quantised is
   x / max(|x| over the block). For block 64 that has std ~0.38, not 1.0, and
   only ~2.7% of values sit beyond |0.9|. NF4 spends two of sixteen levels
   reaching +/-1 to cover them. A codebook fitted to the real post-absmax
   distribution measured 3.4% lower reconstruction error on Qwen3.5-9B decoder
   weights, at zero cost in size or speed: it is sixteen different constants.

2. A recurring bug class. Whole-tensor float promotion OOMs on
   vocabulary-sized matrices. 248,320 x 4,096 x 4 bytes is 3.79 GB in one
   allocation, on top of a loaded model. This has now happened twice in this
   project: once in the embedding quantiser (harness bug #4) and once in an
   ad-hoc codebook script. validate.apply_scheme_in_place has the same shape
   and only avoids it by defaulting skip_embeddings=True, which hides the
   problem rather than fixing it.

   Every function here processes in row chunks and accumulates error as
   running squared norms, so peak memory is bounded by chunk_rows * n_cols
   regardless of how large the tensor is. estimate_chunk_rows() sizes the
   chunk against actually free VRAM and raises if even one row will not fit,
   rather than letting the allocator fail somewhere less informative.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import torch


# NF4: quantiles of a standard normal, mapped to [-1, 1]. Kept here so the
# codebook comparison has a baseline that is byte-identical to qembed.py.
NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=np.float64)


# --------------------------------------------------------------- memory

def estimate_chunk_rows(n_cols: int, device, headroom_gb: float = 1.0,
                        max_rows: int = 8192) -> int:
    """
    Largest row chunk that fits in free VRAM with headroom to spare.

    A chunk costs roughly 4 bytes per element for the fp32 view, plus the same
    again for the quantised result and the bucketize indices. Budget 12 bytes
    per element and round down.

    Raises rather than returning a chunk that cannot work. A silent fallback
    to chunk_rows=1 would take hours and look like a hang.
    """
    if device is None or (hasattr(device, "type") and device.type != "cuda"):
        return max_rows

    free, _total = torch.cuda.mem_get_info(device)
    usable = free - int(headroom_gb * 1024 ** 3)
    per_row = n_cols * 12

    if usable <= per_row:
        raise RuntimeError(
            f"not enough free VRAM to process even one row of width {n_cols}: "
            f"{free / 1024**3:.2f} GB free, {headroom_gb:.2f} GB reserved as "
            f"headroom, {per_row / 1024**2:.1f} MB needed per row. Free a "
            f"model from a notebook variable, or lower headroom_gb.")

    return max(1, min(max_rows, usable // per_row))


# ------------------------------------------------------------- fitting

def fit_codebook(samples: np.ndarray, k: int = 16, iters: int = 300,
                 pin_zero: bool = True, pin_ends: bool = False,
                 tol: float = 1e-12) -> np.ndarray:
    """
    Lloyd-Max scalar quantiser fitted to `samples`, which should be
    post-absmax values (see sample_post_absmax).

    pin_zero keeps an exact 0.0 level, so weights that are exactly zero
    reconstruct exactly. Cheap insurance, measured cost near zero.

    pin_ends forces the outer levels to +/-1. NF4 does this, and it is the
    single largest source of its suboptimality: the absmax normalisation
    guarantees exactly one value per block reaches an endpoint, so ~1.5% of
    values consume two of sixteen levels. Defaults False here because
    unpinning measured 3.4% better on real weights against 1.2% pinned.

    Note the trade this makes: unpinning clips the largest weight in each
    block slightly, concentrating MORE error on outliers and less on the bulk.
    That is the opposite of what Hadamard rotation did, and rotation lost on
    quality despite better reconstruction. This must clear a quality gate.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64))
    if s.size < k * 10:
        raise ValueError(f"need at least {k*10} samples to fit {k} levels, "
                         f"got {s.size}")

    levels = np.quantile(s, np.linspace(0, 1, k))

    for _ in range(iters):
        bounds = (levels[:-1] + levels[1:]) / 2
        idx = np.searchsorted(bounds, s, side="left")
        new = levels.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new[j] = s[m].mean()
        if pin_ends:
            new[0], new[-1] = -1.0, 1.0
        if pin_zero:
            new[np.argmin(np.abs(new))] = 0.0
        new = np.sort(new)
        if np.allclose(new, levels, atol=tol):
            levels = new
            break
        levels = new

    return levels


def sample_post_absmax(model, block: int = 64, max_weights: int = 8_000_000,
                       exclude: Sequence[str] = ("embed", "lm_head"),
                       chunk_rows: Optional[int] = None) -> np.ndarray:
    """
    Sample x / absmax(block) across the model's 2-D weights.

    `exclude` defaults to the embedding and LM head. Fitting on the tensors
    that will later be evaluated leaks: the codebook would be tuned to the
    thing under test. Set exclude=() deliberately if that is what you want.
    """
    out: List[np.ndarray] = []
    taken = 0

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim != 2 or p.shape[-1] % block:
                continue
            if any(e in name for e in exclude):
                continue

            rows = chunk_rows or estimate_chunk_rows(p.shape[-1], p.device)
            for i in range(0, p.shape[0], rows):
                sl = p.data[i:i + rows]
                flat = sl.float().reshape(-1, block)
                am = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
                out.append((flat / am).reshape(-1).cpu().numpy())
                taken += out[-1].size
                del flat, am
                if taken >= max_weights:
                    break
            if taken >= max_weights:
                break

    if not out:
        raise RuntimeError("no eligible 2-D weights found; check `exclude` "
                           "and that the model is loaded")
    return np.concatenate(out)[:max_weights]


# ------------------------------------------------------------- applying

def _quantise_chunk(sl: torch.Tensor, levels_t: torch.Tensor,
                    bounds_t: torch.Tensor, block: int) -> torch.Tensor:
    flat = sl.float().reshape(-1, block)
    am = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = torch.bucketize(flat / am, bounds_t)
    return (levels_t[idx] * am).reshape(sl.shape)


def codebook_rel_error(W: torch.Tensor, levels: np.ndarray, block: int = 64,
                       chunk_rows: Optional[int] = None) -> float:
    """Relative Frobenius error of a quantise-dequantise round trip.

    Accumulates running squared norms rather than holding a full fp32 copy,
    so this is safe on vocabulary-sized tensors.
    """
    levels_t = torch.tensor(levels, dtype=torch.float32, device=W.device)
    bounds_t = (levels_t[:-1] + levels_t[1:]) / 2
    rows = chunk_rows or estimate_chunk_rows(W.shape[-1], W.device)

    num = den = 0.0
    with torch.no_grad():
        for i in range(0, W.shape[0], rows):
            sl = W.data[i:i + rows]
            orig = sl.float()
            recon = _quantise_chunk(sl, levels_t, bounds_t, block)
            num += float((orig - recon).pow(2).sum())
            den += float(orig.pow(2).sum())
            del orig, recon

    return (num ** 0.5) / (den ** 0.5) if den > 0 else 0.0


def apply_codebook_(model, levels: np.ndarray, block: int = 64,
                    skip_embeddings: bool = False,
                    chunk_rows: Optional[int] = None,
                    min_numel: int = 4096) -> Dict[str, Any]:
    """
    Overwrite every eligible 2-D weight with its codebook reconstruction.

    IRREVERSIBLE on the loaded model. Reload between schemes; comparing two
    codebooks on one model compounds the error of both.

    skip_embeddings defaults False here, unlike validate.apply_scheme_in_place.
    That function's default exists to dodge an OOM on vocabulary-sized
    tensors. This one chunks, so the embedding and LM head are included by
    default, which is what the production configuration actually quantises.

    Returns per-tensor errors so a caller can see whether one tensor is
    carrying the whole regression.
    """
    levels = np.asarray(levels, dtype=np.float64)
    if levels.size < 2 or not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be sorted and strictly increasing")

    stats: Dict[str, Any] = {
        "matrices_modified": 0, "params_modified": 0, "skipped": 0,
        "per_tensor": {}, "n_levels": int(levels.size), "block": block,
        "max_level": float(levels[-1]),
        "has_exact_zero": bool(np.any(levels == 0.0)),
    }

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim != 2 or p.shape[-1] % block or p.numel() < min_numel:
                stats["skipped"] += 1
                continue
            if skip_embeddings and ("embed" in name or "lm_head" in name):
                stats["skipped"] += 1
                continue

            levels_t = torch.tensor(levels, dtype=torch.float32,
                                    device=p.device)
            bounds_t = (levels_t[:-1] + levels_t[1:]) / 2
            rows = chunk_rows or estimate_chunk_rows(p.shape[-1], p.device)

            num = den = 0.0
            for i in range(0, p.shape[0], rows):
                sl = p.data[i:i + rows]
                orig = sl.float()
                recon = _quantise_chunk(sl, levels_t, bounds_t, block)
                num += float((orig - recon).pow(2).sum())
                den += float(orig.pow(2).sum())
                sl.copy_(recon.to(p.dtype))
                del orig, recon

            err = (num ** 0.5) / (den ** 0.5) if den > 0 else 0.0
            stats["per_tensor"][name] = round(err, 6)
            stats["matrices_modified"] += 1
            stats["params_modified"] += int(p.numel())

    if stats["matrices_modified"] == 0:
        raise RuntimeError(
            "no matrices were modified. Either every tensor was filtered out "
            "or the model structure is not what this expects. A silent no-op "
            "here would report the unmodified model's quality as the "
            "codebook's, which is worse than crashing.")

    errs = list(stats["per_tensor"].values())
    stats["mean_rel_error"] = round(float(np.mean(errs)), 6)
    stats["max_rel_error"] = round(float(np.max(errs)), 6)
    return stats
