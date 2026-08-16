// ggml-nf4dq.h
//
// NF4 with double quantisation, as a ggml block type.
//
// WHY THIS TYPE EXISTS
//
// ggml's Q4_K spends 4.50 bits per weight: 128 bytes of 4-bit indices per
// 256 weights, plus 16 bytes of per-sub-block scales and mins. Its levels
// are uniformly spaced within each sub-block.
//
// NF4 instead uses 16 levels placed at the quantiles of a normal
// distribution, densely near zero where weights are dense and sparsely at
// the extremes. Measured on real weights earlier in this project, NF4 beat
// uniform int4 by 28.6% in reconstruction error on outlier-heavy matrices.
//
// Double quantisation then removes most of the scale overhead: the
// per-sub-block absmax values are themselves quantised to 8 bits against a
// single fp16 super-scale. That is what takes the format from 4.25 bpw
// (one fp16 scale per 64 weights, as in harness/qembed.py) down to 4.1875.
//
// LAYOUT, per 256 weights:
//
//   qs[128]  4-bit codebook indices, two per byte.
//            Even positions in the LOW nibble, odd in the HIGH nibble.
//            This matches harness/qembed.py exactly, so a checkpoint packed
//            by either implementation is readable by the other.
//   sc[4]    absmax of each 64-weight sub-block, quantised to uint8.
//   d        fp16 super-scale: sub-block absmax = d * sc[i].
//
//   530 bytes / 1024 weights = 4.1406 bpw.
//
// WHY QK = 1024, MEASURED NOT ASSUMED
//
// The super-scale is a fixed 2 bytes per superblock, so a larger superblock
// amortises it further: 4.1875 bpw at QK=256, 4.1406 at 1024, 4.1328 at 2048.
// A sweep over Gaussian, kurtosis-1.4 and 16-sigma-tail data showed the
// reconstruction error is FLAT across that range (0.091919 to 0.091918), so
// the larger superblock is free in quality terms.
//
// 1024 is the ceiling for this architecture, not a preference. ggml requires
// the block size to divide the row length, and Qwen3.x-27B has hidden 5120
// and FFN 17408. Both divide by 1024 (5 and 17) and neither divides by 2048.
//
// PROVENANCE NOTE
//
// The reference figure this is validated against is 0.0918, the relative
// reconstruction error that harness/qembed.py self-reports on sampled
// embedding rows. That number has now reproduced across three models and
// two hidden sizes (0.091933, 0.091831, 0.092064), so it is a stable
// target rather than a one-off.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QK_NF4DQ     1024   // weights per superblock (see note below)
#define NF4DQ_SUB      64   // weights per sub-block (matches bitsandbytes)
#define NF4DQ_NSUB    (QK_NF4DQ / NF4DQ_SUB)

typedef uint16_t nf4dq_half;   // stand-in for ggml_half in standalone builds

typedef struct {
    uint8_t     qs[QK_NF4DQ / 2];   // 128 bytes: packed 4-bit indices
    uint8_t     sc[NF4DQ_NSUB];     //   4 bytes: quantised sub-block absmax
    nf4dq_half  d;                  //   2 bytes: super-scale for sc[]
} block_nf4dq;                      // 134 bytes total

// Compile-time guarantee that no padding crept in. A silently padded struct
// would produce a file that is the right size on one compiler and the wrong
// size on another, which is exactly the class of bug that is invisible until
// someone else tries to load the checkpoint.
typedef char nf4dq_size_check[(sizeof(block_nf4dq) == 530) ? 1 : -1];

// The 16 NF4 levels: quantiles of a standard normal, normalised to [-1, 1].
// Byte-identical to NF4_LEVELS in harness/qembed.py and validate.py.
extern const float NF4DQ_LEVELS[16];

// Reference (non-SIMD) quantise and dequantise.
//   k must be a multiple of QK_NF4DQ.
void quantize_row_nf4dq_ref  (const float * restrict x, block_nf4dq * restrict y, int64_t k);
void dequantize_row_nf4dq    (const block_nf4dq * restrict x, float * restrict y, int64_t k);

// Convenience: quantise then dequantise, reporting relative Frobenius error.
// This is the gate. It should return ~0.0918 on real transformer weights.
float nf4dq_roundtrip_error(const float * x, int64_t k);

#ifdef __cplusplus
}
#endif
