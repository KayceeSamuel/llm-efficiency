# NF4DQ: adding NF4 with double quantisation to llama.cpp

Status: **reference implementation validated on CPU. No GPU path yet.**

## What is done

`ggml-nf4dq.h` / `ggml-nf4dq.c` define the block type and a reference
quantise/dequantise pair, validated three ways:

| Check | Result |
|---|---|
| `sizeof(block_nf4dq)` | 530 bytes, 4.1406 bpw (compile-time assert) |
| Gaussian round-trip error | 0.091956 |
| Cross-check vs `harness/qembed.py` (numpy) | agrees to 5 decimals, 99.5%+ identical nibbles |
| Degenerate cases (all-zero, on-codebook) | 0.000000 / 0.000015 |

The 0.0918 target comes from `qembed.py`'s self-reported reconstruction error
on real embedding rows, which has reproduced across three models and two
hidden sizes (0.091933, 0.091831, 0.092064).

**The result worth noting:** nf4dq matches `qembed`'s error at 4.1875 bpw
against its 4.25. Double quantisation costs nothing measurable in quality and
saves 0.0625 bits per weight.

## Why QK = 1024

The super-scale is a fixed 2 bytes per superblock, so a larger superblock
amortises it further. A sweep showed the reconstruction error is flat across
the whole range, so the larger superblock is free:

| QK | bytes | bpw | gaussian | kurtosis 1.4 | 16-sigma tails |
|---|---|---|---|---|---|
| 256 | 134 | 4.1875 | 0.091919 | 0.093468 | 0.119467 |
| 512 | 266 | 4.1562 | 0.091919 | 0.093468 | 0.119473 |
| **1024** | **530** | **4.1406** | **0.091918** | **0.093465** | **0.119480** |
| 2048 | 1058 | 4.1328 | 0.091918 | 0.093464 | 0.119480 |
| 4096 | 2114 | 4.1289 | 0.091913 | 0.093466 | 0.119480 |

**1024 is a hard ceiling for this architecture, not a preference.** ggml
requires the block size to divide the row length. Qwen3.x-27B has hidden 5120
and FFN 17408: both divide by 1024 (5 and 17), neither divides by 2048.
A model with different dimensions could go further.

For reference, bitsandbytes quantises scales in groups of 256, implying a
16,384-weight superblock at ~4.126 bpw. The remaining gap to that is 0.015
bpw, which is roughly 0.05 GB on a 27B and not worth breaking ggml's
conventions for.

## Projected size, Qwen3.8-27B

All measured except the last row.

| Build | Size | bpw | Status |
|---|---|---|---|
| Stock Q4_K_M | 15.64 GB | 4.92 | measured (dry run) |
| Q4_K_M, head forced to Q4_K | 15.34 GB | 4.82 | measured (dry run) |
| `--pure` Q4_K | 14.32 GB | 4.50 | measured (dry run) |
| **`--pure` NF4DQ** | **~13.20 GB** | **4.1406** | projected |
| bitsandbytes + qembed | 12.968 GB | ~4.13 | measured (live load) |

Projection calibrated against the measured pure-Q4_K build: 0.229 GiB of
non-quantised overhead (f32 norms) plus 26.9e9 parameters at the target bpw.

**The two are closer than they look.** The GGUF file carries `blk.64.nextn.*`,
the multi-token-prediction head, which `AutoModelForCausalLM` discards at load
time. That is roughly 0.20 GB. Tensor for tensor, NF4DQ at QK=1024 lands at
about 13.00 GB against bitsandbytes' 12.968.

So the trade is not 0.35 GB for speed. It is approximately parity on weights,
plus a 0.20 GB MTP head that buys a reported 1.4 to 2.2x on decode, inside a
runtime that is 5 to 9x faster to begin with. The MTP estimate is read off the
dry-run tensor listing and should be confirmed against a real build.

## What is left

### 1. Register the type in ggml (mechanical, half a day)

- `ggml.h`: add `GGML_TYPE_NF4DQ` to the `ggml_type` enum. **Append at the
  end.** The enum is serialised into GGUF files; inserting in the middle
  silently reinterprets every existing checkpoint.
- `ggml.c`: add the `type_traits` entry — `blck_size = 256`,
  `type_size = sizeof(block_nf4dq)`, `is_quantized = true`, and the
  `to_float` / `from_float` function pointers.
- `ggml-quants.h` / `.c`: move the two functions here, renamed to
  `quantize_row_nf4dq_ref` / `dequantize_row_nf4dq` following the existing
  naming, and add `quantize_nf4dq` (the row-loop wrapper the quantiser calls).
- `ggml-cpu/`: a `vec_dot_nf4dq_q8_K` for CPU inference. Not needed if only
  the CUDA path is targeted, but `llama-quantize` will want the CPU path for
  verification.

### 2. Converter support (mechanical, an hour)

- `llama-quantize`'s type table, so `--pure NF4DQ` and
  `--output-tensor-type nf4dq` resolve.
- `llama.cpp`'s `llama_model_quantize_impl` type dispatch.

### 3. CUDA kernels (the real work)

Two entry points, following `ggml-cuda/dequantize.cuh` and
`ggml-cuda/mmvq.cu`:

- `dequantize_block_nf4dq` — bulk dequant, used by prefill and by any path
  that materialises the tensor. Straightforward port of the reference.
- `vec_dot_nf4dq_q8_1` — the decode path, one row against a quantised
  activation vector. **This is where the performance lives.** It must
  dequantise inside the dot product, in registers, or the bandwidth benefit
  is lost and the result is slower than what it replaced.

The codebook is 16 floats and should live in `__constant__` memory or be
materialised into registers; a global-memory lookup per weight would
dominate.

### 4. Gates before promotion

In order, and none of them optional:

1. CPU round-trip on real weights returns ~0.0918. Already passing on
   synthetic data.
2. CUDA dequant is bit-identical to the CPU reference on the same block.
3. `llama-perplexity` on wikitext at the same window, against the
   `--pure` Q4_K build. The number to beat is not "good", it is "no worse
   than Q4_K at 0.31 fewer bpw".
4. `llama-bench` decode against the same build. If the fused path is slower
   than Q4_K, the kernel is wrong and the size win is irrelevant.

Gate 3 matters more than it looks. Every method that restructures how error is
distributed across weights has to clear a quality gate *before* promotion, not
after. That lesson cost this project a rotation experiment that looked
convincing on reconstruction error and lost on both accuracy and perplexity.

## Files

- `ggml-nf4dq.h` — type definition, block struct, compile-time size assert
- `ggml-nf4dq.c` — reference quantise / dequantise / round-trip error
- `test_nf4dq.c` — standalone self-test, no dependencies
- `xcheck.py` — cross-validation against the `qembed.py` algorithm in numpy

Build and run the gate:

```bash
gcc -O2 -std=c11 -o test_nf4dq test_nf4dq.c ggml-nf4dq.c -lm && ./test_nf4dq
python3 xcheck.py
```
