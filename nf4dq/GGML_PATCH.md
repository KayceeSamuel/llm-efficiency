# Registering NF4DQ in llama.cpp

Against `ggml-org/llama.cpp` as of the clone used here. Line numbers drift, so
each hunk is anchored on surrounding text rather than a line number.

Steps 1 and 2 below are mechanical and complete. Step 3, the CUDA kernels, is
where the performance lives and is the part that cannot be validated without a
card.

---

## 0. Files to add

Copy into the ggml tree:

```
ggml/src/ggml-nf4dq.c      <- from this repo, minus the standalone fp16 helpers
ggml/src/ggml-nf4dq.h      <- ditto
```

`nf4dq/ggml_ready/` holds versions with those edits already applied and
syntax-checked against the real ggml headers:

- local `fp32_to_fp16` / `fp16_to_fp32` replaced by ggml's
  `GGML_FP32_TO_FP16` / `GGML_FP16_TO_FP32`
- `nf4dq_half` replaced by `ggml_half`, local typedef dropped
- `#define GGML_COMMON_DECL_C` added before the `ggml-common.h` include.
  Without it `ggml_half` and every block typedef are compiled out, and the
  first symptom is the `static_assert` failing with "size of array is
  negative", which points at the struct rather than the include. ggml-quants.h
  does the same thing for the same reason.

Verify after copying:

```bash
gcc -fsyntax-only -std=c11 -Iggml/include -Iggml/src ggml/src/ggml-nf4dq.c
```

Add `ggml-nf4dq.c` to the source list in `ggml/src/CMakeLists.txt` alongside
`ggml-quants.c`.

---

## 1. Register the type

### 1a. `ggml/include/ggml.h`, the `ggml_type` enum

**Append. Do not insert.** The enum value is written into every GGUF file that
uses the type, so renumbering silently reinterprets existing checkpoints.

```c
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_Q2_0    = 42,
+       GGML_TYPE_NF4DQ   = 43, // NF4 with double quantisation, 4.1406 bpw
-       GGML_TYPE_COUNT   = 43,
+       GGML_TYPE_COUNT   = 44,
```

### 1b. `ggml/include/ggml.h`, the `ggml_ftype` enum

```c
        GGML_FTYPE_MOSTLY_Q2_0    = 28, // except 1d tensors
+       GGML_FTYPE_MOSTLY_NF4DQ   = 29, // except 1d tensors
```

### 1c. `ggml/src/ggml-common.h`, the block struct

Put it next to the other 4-bit blocks, and keep the `static_assert`: a
silently padded struct produces a file that is the right size on one compiler
and the wrong size on another, which is invisible until someone else tries to
load the checkpoint.

```c
#define QK_NF4DQ  1024   // superblock
#define NF4DQ_SUB   32   // sub-block
#define NF4DQ_NSUB (QK_NF4DQ / NF4DQ_SUB)   // 32

typedef struct {
    uint8_t    qs[QK_NF4DQ / 2];      // 512 B: 4-bit weight indices
    uint8_t    sc[NF4DQ_NSUB / 2];    //  16 B: 4-bit scale indices
    ggml_half  d;                     //   2 B: super-scale (carries /127)
    uint8_t    pad[2];                //   2 B: 4-byte alignment
} block_nf4dq;                        // 532 B
static_assert(sizeof(block_nf4dq) == 532, "wrong nf4dq block size/padding");
static_assert(sizeof(block_nf4dq) % 4 == 0, "nf4dq block must be 4-byte aligned");

// The padding is not cosmetic. Without it the struct is 530 bytes with
// alignment 2, so consecutive blocks alternate between aligned and 2 bytes
// off (verified with nf4dq/align_test.c). get_int_b4 in the CUDA vec_dot
// reads four bytes at a time and assumes alignment; on the odd blocks that
// faults or silently reads across the boundary. Every shipped ggml block type
// is a multiple of 4 (iq4_xs is 136, q4_K is 144), which is presumably why
// this has not come up upstream. Cost: 0.0156 bpw, about 0.05 GB on a 27B.
```

Note `QK_NF4DQ` is 1024, not ggml's usual `QK_K` of 256. That is forced, not
chosen: ggml requires the block size to divide the row length, and 1024 is the
largest superblock that divides both 5120 (hidden) and 17408 (FFN) on
Qwen3.x-27B. A larger superblock amortises the 2-byte super-scale further and
measured no worse on error, so 1024 is the ceiling rather than a preference.

### 1d. `ggml/src/ggml.c`, the `type_traits` array

Modelled on `GGML_TYPE_IQ4_NL`, which is the closest existing analogue: it is
the only shipped type whose levels come from a codebook rather than
arithmetic.

```c
    [GGML_TYPE_NF4DQ] = {
        .type_name                = "nf4dq",
        .blck_size                = QK_NF4DQ,
        .type_size                = sizeof(block_nf4dq),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nf4dq,
        .from_float_ref           = (ggml_from_float_t) quantize_row_nf4dq_ref,
    },
```

### 1e. `ggml/src/ggml.c`, `ggml_ftype_to_ggml_type`

```c
        case GGML_FTYPE_MOSTLY_NF4DQ:  wtype = GGML_TYPE_NF4DQ;  break;
```

### 1f. `ggml/src/ggml.c`, `ggml_quantize_chunk`

```c
        case GGML_TYPE_NF4DQ: result = quantize_nf4dq(src + start,
                                  (char *) dst + start_row * row_size,
                                  nrows, n_per_row, imatrix); break;
```

### 1g. `ggml/src/ggml-quants.h`, declarations

Match the existing naming exactly, or the traits cast will compile and then
misbehave at runtime.

```c
GGML_API void   quantize_row_nf4dq_ref(const float * GGML_RESTRICT x,
                                       block_nf4dq * GGML_RESTRICT y, int64_t k);
GGML_API void   dequantize_row_nf4dq  (const block_nf4dq * GGML_RESTRICT x,
                                       float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_nf4dq        (const float * GGML_RESTRICT src,
                                       void * GGML_RESTRICT dst,
                                       int64_t nrows, int64_t n_per_row,
                                       const float * imatrix);
```

`quantize_nf4dq` is the row-loop wrapper the quantiser calls. The reference
implementation does not have one yet; it is four lines:

```c
size_t quantize_nf4dq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                      int64_t nrow, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);   // no importance-matrix support yet, see note below
    const size_t row_size = ggml_row_size(GGML_TYPE_NF4DQ, n_per_row);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_nf4dq_ref(src, (block_nf4dq *) qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}
```

**On the imatrix.** Ignoring it is honest but leaves value on the table: an
importance matrix weights the quantisation error by how much each input
channel actually matters, and every K-quant uses it. Worth adding later, and
worth measuring rather than assuming, since this project has twice seen
reconstruction improvements fail to reach quality.

---

## 2. Converter support

### 2a. `src/llama.h`, the `llama_ftype` enum

Append, same reasoning as 1a.

```c
        LLAMA_FTYPE_MOSTLY_NF4DQ = <next free>,  // except 1d tensors
```

### 2b. `src/llama-quant.cpp`, the ftype-to-type mapping

```c
        case LLAMA_FTYPE_MOSTLY_NF4DQ: default_type = GGML_TYPE_NF4DQ; break;
```

### 2c. `tools/quantize/quantize.cpp`, the type table

```c
    { "NF4DQ", LLAMA_FTYPE_MOSTLY_NF4DQ, " 4.16 bpw NF4 with double quantisation", },
```

After this, `--pure NF4DQ` and `--output-tensor-type nf4dq` both resolve.

### 2d. Check the tensor-type override path

`llama_model_quantize_impl` picks per-tensor types through
`llama_tensor_get_type`. Confirm it does not silently substitute a K-quant for
tensors it considers special. Under `--pure` it should not, but the earlier
dry-run measurements showed `--pure` interacting with `--output-tensor-type`
in ways worth verifying rather than assuming.

---

## 3. CUDA kernels

Two entry points, both in `ggml/src/ggml-cuda/`.

### 3a. `dequantize.cuh` — bulk dequantisation

Used by prefill and anything that materialises the tensor. A direct port of
the reference loop. Not performance-critical.

### 3b. `mmvq.cu` — `vec_dot_nf4dq_q8_1`, the decode path

**This is where the performance lives.** It must dequantise inside the dot
product, in registers. If it materialises to bf16 first and then multiplies,
the bandwidth benefit is gone and the result is slower than what it replaces,
which is exactly the failure mode bitsandbytes has: measured 6.44 tok/s on an
A100, 4.1% of that card's roofline.

Two implementation notes:

- Both codebooks are 16 floats. Put `NF4DQ_LEVELS` and `NF4DQ_SCALE_LEVELS` in
  `__constant__` memory, or materialise them into registers once per thread
  block. A global-memory lookup per weight would dominate the kernel.
- A 1024-weight superblock is four times ggml's usual `QK_K`. Check the
  register pressure and the `mmvq` block-size assumptions rather than assuming
  the existing loop bounds carry over.

---

## 4. Gates, in order, none optional

1. **CPU round-trip on real weights returns ~0.0918.** Already passing on
   synthetic data (0.087971 gaussian, 0.100847 with 16-sigma tails).
2. **CUDA dequant bit-identical to the CPU reference** on the same block.
   Not "close": identical. A discrepancy here is a packing bug and every
   number downstream would be measuring the bug.
3. **`llama-perplexity` on wikitext**, same window, against the `--pure` Q4_K
   build. The bar is not "good", it is "no worse than Q4_K at 0.36 fewer bpw".
   Expected from the simulation: better, since NF4DQ v2 measured 11.65059
   against bitsandbytes' 11.7413 on the 9B.
4. **`llama-bench` decode** against the same build. If the fused path is
   slower than Q4_K, the kernel is wrong and the size win is irrelevant.

Gate 3 matters more than it looks. Two methods in this project produced better
reconstruction error and worse quality: Hadamard rotation (6.74% better
reconstruction, worse on accuracy and perplexity) and endpoint-unpinned
codebooks (3.12% better, 1.22% worse perplexity). The rule that emerged is
that reconstruction gains from changing the *shape* of the error distribution
do not translate, while gains within the same shape do. NF4DQ v2 is a
within-shape change and did translate (5.17% reconstruction, 0.66%
perplexity), but that was established by measurement, not by argument.

---

## 5. Expected result

Projected from measured components on Qwen3.8-27B:

| Build | Size | bpw | Status |
|---|---|---|---|
| Stock Q4_K_M | 15.64 GB | 4.92 | measured |
| `--pure` Q4_K | 14.32 GB | 4.50 | measured |
| **`--pure` NF4DQ** | **~13.25 GB** | **4.1562** | projected |
| bitsandbytes + qembed | 12.968 GB | ~4.13 | measured |

The GGUF file carries `blk.64.nextn.*`, the multi-token-prediction head, which
`AutoModelForCausalLM` discards at load. That is roughly 0.20 GB. Tensor for
tensor, NF4DQ lands near parity with the bitsandbytes configuration, while
gaining a draft head worth a reported 1.4 to 2.2x on decode inside a runtime
already several times faster.
