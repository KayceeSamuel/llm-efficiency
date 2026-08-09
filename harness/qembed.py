"""
qembed.py — 4-bit NF4 embedding table with row-wise dequantisation.

WHY THIS EXISTS:

bitsandbytes quantises nn.Linear layers. It does not touch nn.Embedding,
because there is no Linear4bit equivalent for a lookup. On Qwen3.5-9B that
leaves the 248,320 x 4,096 embedding table at bf16 -- 1.895 GB, which after
the LM head was handled by llm_int8_skip_modules=[] is the entire remaining
gap between the measured 5.722 GB and the 4.29 GB target.

WHY THIS IS THE EASY CASE:

Everything hard about custom quantisation is the matmul kernel: dequantising
inside the multiply, in registers, or you lose the bandwidth benefit and end
up slower than the library you replaced.

An embedding is not a matmul. It is a gather. For a 512-token prompt you need
512 rows out of 248,320 -- 0.2% of the table. Dequantise only those, once per
forward pass. No kernel, no fused op, no CUDA. The compute cost is negligible
because you touch a fraction of a percent of the data.

QUALITY IS ALREADY VALIDATED:

The nf4+emb scheme in validate.py quantised this exact table (plus the LM
head) and measured accuracy 0.74533 against nf4's 0.74667, and perplexity
11.7412 against 11.6053, at n=1000. This module implements a configuration
that already passed the gate; it is not a new hypothesis.

Verified before implementation: gather-then-dequantise produces bit-identical
results to dequantising the whole table and then indexing.
"""

import gc
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


# The 16 NF4 levels: quantiles of a standard normal, normalised to [-1, 1].
# Denser near zero where weights are dense, sparser at the extremes. Kept
# identical to validate.NF4_LEVELS so the quality measurement transfers.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


class QuantizedEmbedding(nn.Module):
    """
    Drop-in replacement for nn.Embedding holding a 4-bit NF4 table.

    Storage per row: hidden_size nibbles (packed two per byte) plus one fp16
    scale per block of `block_size` values.

    At hidden 4096 with block 64 that is 2048 bytes of weight plus 128 bytes
    of scale per row, against 8192 bytes at bf16 -- a 3.76x reduction
    including scale overhead.

    Forward gathers the requested rows, unpacks the nibbles, applies the
    per-block scale, and returns in the original dtype. Only the rows named
    by input_ids are ever materialised.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 block_size: int = 64, dtype=torch.bfloat16,
                 device=None, padding_idx: Optional[int] = None):
        super().__init__()
        if embedding_dim % block_size != 0:
            raise ValueError(
                f"embedding_dim {embedding_dim} must be divisible by "
                f"block_size {block_size}")
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for nibble packing")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.blocks_per_row = embedding_dim // block_size
        self.bytes_per_row = embedding_dim // 2
        self.out_dtype = dtype
        self.padding_idx = padding_idx

        # Buffers, not parameters: this is inference-only and must not be
        # picked up by an optimiser or a gradient pass.
        self.register_buffer(
            "packed",
            torch.zeros(num_embeddings, self.bytes_per_row,
                        dtype=torch.uint8, device=device),
            persistent=True)
        self.register_buffer(
            "scales",
            torch.zeros(num_embeddings, self.blocks_per_row,
                        dtype=torch.float16, device=device),
            persistent=True)
        self.register_buffer(
            "levels", NF4_LEVELS.to(device=device), persistent=False)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, block_size: int = 64,
                       chunk_rows: int = 8192) -> "QuantizedEmbedding":
        """
        Quantise an existing nn.Embedding.

        Processed in row chunks so the fp32 working copy stays small. A naive
        full-table .float() on a 248,320 x 4,096 bf16 table allocates 3.79 GB,
        which OOMs on a card already holding the model -- that failure
        happened once in this project and is avoided here by construction.
        """
        W = emb.weight
        n, d = W.shape
        out = cls(n, d, block_size=block_size, dtype=W.dtype,
                  device=W.device, padding_idx=emb.padding_idx)

        levels = NF4_LEVELS.to(W.device)
        boundaries = (levels[:-1] + levels[1:]) / 2

        with torch.no_grad():
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                chunk = W[start:end].detach().float()
                blocks = chunk.view(-1, block_size)

                absmax = blocks.abs().amax(dim=1, keepdim=True)
                absmax = torch.where(absmax == 0,
                                     torch.full_like(absmax, 1e-12), absmax)
                idx = torch.bucketize(
                    (blocks / absmax).contiguous(), boundaries
                ).to(torch.uint8)

                # Two 4-bit indices per byte: even positions in the low
                # nibble, odd in the high nibble.
                flat = idx.view(end - start, d)
                packed = (flat[:, 0::2] | (flat[:, 1::2] << 4))

                out.packed[start:end] = packed
                out.scales[start:end] = absmax.view(
                    end - start, out.blocks_per_row).to(torch.float16)

                del chunk, blocks, absmax, idx, flat, packed

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    # -- forward ---------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)

        # Gather ONLY the requested rows. This is the whole point: a 512-token
        # prompt touches 512 of 248,320 rows, so the dequantisation cost is
        # negligible relative to the forward pass.
        rows = self.packed[flat_ids]                    # (N, d/2) uint8
        row_scales = self.scales[flat_ids].float()      # (N, blocks)

        n = rows.shape[0]
        nib = torch.empty(n, self.embedding_dim,
                          dtype=torch.uint8, device=rows.device)
        nib[:, 0::2] = rows & 0x0F
        nib[:, 1::2] = rows >> 4

        vals = self.levels[nib.long()]                  # (N, d) float32
        vals = vals.view(n, self.blocks_per_row, self.block_size)
        vals = vals * row_scales.unsqueeze(-1)
        vals = vals.view(n, self.embedding_dim).to(self.out_dtype)

        if self.padding_idx is not None:
            vals = torch.where(
                (flat_ids == self.padding_idx).unsqueeze(-1),
                torch.zeros_like(vals), vals)

        return vals.view(*input_ids.shape, self.embedding_dim)

    # -- reporting -------------------------------------------------------

    def memory_bytes(self) -> Dict[str, Any]:
        p = self.packed.numel() * self.packed.element_size()
        s = self.scales.numel() * self.scales.element_size()
        original = self.num_embeddings * self.embedding_dim * 2  # bf16
        return {
            "packed_gb": round(p / 1024**3, 4),
            "scales_gb": round(s / 1024**3, 4),
            "total_gb": round((p + s) / 1024**3, 4),
            "original_bf16_gb": round(original / 1024**3, 4),
            "compression_ratio": round(original / (p + s), 3),
            "effective_bits_per_weight": round(
                (p + s) * 8 / (self.num_embeddings * self.embedding_dim), 4),
        }

    def extra_repr(self) -> str:
        return (f"{self.num_embeddings}, {self.embedding_dim}, "
                f"nf4, block_size={self.block_size}")


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def quantize_model_embeddings(model, block_size: int = 64,
                              verify: bool = True) -> Dict[str, Any]:
    """
    Replace the model's input embedding with a QuantizedEmbedding, in place.

    Skips the operation entirely if the LM head shares storage with the
    embedding table (tie_word_embeddings). Replacing a tied embedding would
    silently break the output head, since it reads the same tensor. On
    Qwen3.5-9B they are NOT tied, which is why this is worth 1.9 GB here and
    would be worth nothing on a model that ties them.
    """
    emb = model.get_input_embeddings()
    if emb is None:
        return {"status": "no_embedding_found"}
    if isinstance(emb, QuantizedEmbedding):
        return {"status": "already_quantised"}

    head = getattr(model, "lm_head", None)
    head_w = getattr(head, "weight", None) if head is not None else None
    if head_w is not None and torch.is_tensor(head_w) and torch.is_tensor(emb.weight):
        if head_w.data_ptr() == emb.weight.data_ptr():
            return {
                "status": "skipped_tied",
                "reason": ("lm_head shares storage with the embedding table; "
                           "replacing it would break the output head"),
            }

    before_gb = emb.weight.numel() * emb.weight.element_size() / 1024**3

    # Verification sample: compare a handful of rows before and after, so a
    # silent packing error cannot pass unnoticed.
    probe_ids = None
    reference = None
    if verify:
        probe_ids = torch.randint(0, emb.num_embeddings, (16,),
                                  device=emb.weight.device)
        reference = emb(probe_ids).detach().float().clone()

    qemb = QuantizedEmbedding.from_embedding(emb, block_size=block_size)

    result: Dict[str, Any] = {
        "status": "ok",
        "before_gb": round(before_gb, 4),
        **qemb.memory_bytes(),
    }
    result["saved_gb"] = round(before_gb - result["total_gb"], 4)

    if verify:
        got = qemb(probe_ids).detach().float()
        num = torch.linalg.norm(reference - got)
        den = torch.linalg.norm(reference)
        err = float(num / den) if den > 0 else 0.0
        result["verify_rel_error"] = round(err, 6)
        # Expected ~0.09, matching the 0.0921 measured for this table during
        # quality validation. An order of magnitude off means broken packing.
        result["verify_ok"] = err < 0.25
        if not result["verify_ok"]:
            result["status"] = "verification_failed"
            result["reason"] = (
                f"reconstruction error {err:.4f} is far above the ~0.09 "
                f"expected for NF4 on this table; packing is likely wrong")
            return result

    model.set_input_embeddings(qemb)

    # Drop the original tensor and reclaim the memory.
    del emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.cuda.is_available():
        result["vram_after_gb"] = round(
            torch.cuda.memory_allocated() / 1024**3, 3)

    return result


# ---------------------------------------------------------------------------
# End-to-end check
# ---------------------------------------------------------------------------

def run_embedding_quantisation_check(
    model_id: str = "Qwen/Qwen3.5-9B",
    context_lengths=(512, 4096, 16384),
    results_dir=None,
) -> Dict[str, Any]:
    """
    Load with the LM head already quantised by bitsandbytes, then quantise the
    embedding table on top, and measure the result.

    Baselines measured earlier in this project:
      7.122 GB / 11.84 tok/s   bitsandbytes default
      5.722 GB / 12.67 tok/s   llm_int8_skip_modules=[] (LM head quantised)
    Target: ~4.30 GB, with throughput unchanged -- embedding lookup is a
    rounding error in the forward pass either way.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from .config import capture_environment, run_stamp
    from .deploy import load_real_quantised, measure_throughput, measure_context_scaling

    record: Dict[str, Any] = {
        "run_id": f"T3-QEMBED-{run_stamp()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "measured",
        "environment": capture_environment(),
        "model_id": model_id,
        "baselines": {
            "bnb_default": {"vram_gb": 7.122, "tok_s": 11.84},
            "bnb_no_skip": {"vram_gb": 5.722, "tok_s": 12.67},
            "target_vram_gb": 4.30,
        },
    }

    model = None
    try:
        print("loading with LM head quantised (no_skip) ...")
        model, tok, info = load_real_quantised(model_id, variant="no_skip")
        record["load"] = info
        print(f"  before embedding quantisation: {info['resting_vram_gb']} GB")

        print("quantising embedding table ...")
        q = quantize_model_embeddings(model)
        record["quantisation"] = q
        print(f"  status: {q['status']}")
        if q.get("status") == "ok":
            print(f"  {q['before_gb']} GB -> {q['total_gb']} GB "
                  f"({q['compression_ratio']}x), "
                  f"verify_err={q.get('verify_rel_error')}")
            print(f"  VRAM now: {q.get('vram_after_gb')} GB")

        print("measuring throughput ...")
        record["throughput"] = measure_throughput(model, tok)
        print(f"  decode: {record['throughput']['decode_tokens_per_sec_median']} tok/s")

        print("context scaling ...")
        record["context_scaling"] = measure_context_scaling(
            model, tok, lengths=list(context_lengths))

        record["status"] = "ok"

    except Exception as e:
        import traceback
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        print(f"ERROR: {e}")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record["verdict"] = _verdict(record)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    if results_dir is not None:
        p = Path(results_dir)
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"{record['run_id']}.json").open("w") as f:
            json.dump(record, f, indent=2)

    return record


def _verdict(record: Dict[str, Any]) -> Dict[str, Any]:
    q = record.get("quantisation", {})
    t = record.get("throughput", {})

    if record.get("status") != "ok" or q.get("status") != "ok":
        return {"status": "failed",
                "reason": q.get("reason") or record.get("error")}

    vram = q.get("vram_after_gb")
    tps = t.get("decode_tokens_per_sec_median")
    base = record["baselines"]

    return {
        "status": "ok",
        "vram_gb": vram,
        "decode_tokens_per_sec": tps,
        "vs_bnb_default": {
            "vram_reduction": (round(1 - vram / base["bnb_default"]["vram_gb"], 4)
                               if vram else None),
            "speed_ratio": (round(tps / base["bnb_default"]["tok_s"], 3)
                            if tps else None),
        },
        "vs_no_skip": {
            "vram_reduction": (round(1 - vram / base["bnb_no_skip"]["vram_gb"], 4)
                               if vram else None),
            "speed_ratio": (round(tps / base["bnb_no_skip"]["tok_s"], 3)
                            if tps else None),
        },
        "quality_note": (
            "Quality for this configuration was validated separately: the "
            "nf4+emb scheme quantised both the embedding table and the LM "
            "head, measuring accuracy 0.74533 against nf4's 0.74667 and "
            "perplexity 11.7412 against 11.6053 at n=1000. This run measures "
            "memory and speed only."),
    }
