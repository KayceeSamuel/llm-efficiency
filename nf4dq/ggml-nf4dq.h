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
//   qs[512]  4-bit codebook indices, two per byte.
//            Even positions in the LOW nibble, odd in the HIGH nibble.
//            This matches harness/qembed.py exactly, so a checkpoint packed
//            by either implementation is readable by the other.
//   sc[16]   4-bit indices into NF4DQ_SCALE_LEVELS, two per byte, one per
//            32-weight sub-block. Sub-block absmax = d * SCALE_LEVELS[sc[i]].
//   d        fp16 super-scale: the largest sub-block absmax in the superblock,
//            pre-divided by 127 so the int8 codebook needs no runtime division.
//   pad      2 bytes, so the struct is 4-byte aligned. See below.
//
//   532 bytes / 1024 weights = 4.1562 bpw.
//
// WHY THE PADDING, MEASURED
//
// Without it the struct is 530 bytes with alignment 2, so consecutive blocks
// alternate between 4-byte aligned and 2 bytes off:
//
//     block 0: qs offset % 4 = 0
//     block 1: qs offset % 4 = 2   <-- misaligned
//     block 2: qs offset % 4 = 0
//     ...
//
// ggml's CUDA vec_dot helpers read the packed nibbles four bytes at a time
// through get_int_b4, which assumes 4-byte alignment. On misaligned blocks
// that either faults or silently reads across the boundary, and a silent
// wrong answer is far worse than a crash in a quantisation format.
//
// Every shipped ggml block type is a multiple of 4: block_iq4_xs is 136,
// block_q4_K is 144. That is presumably why this has not come up upstream.
//
// Cost of the fix: 0.0156 bpw, which is about 0.05 GB on a 27B model.
//
// WHY SUB = 32 WITH 4-BIT SCALES, MEASURED
//
// The first version of this file used SUB = 64 with 8-bit scales, copied from
// bitsandbytes without testing. At byte-identical size, on weights matching
// the measured kurtosis-1.4 profile:
//
//   SUB=64, 8-bit uniform scales   0.093626   (the copied layout)
//   SUB=32, 4-bit uniform scales   0.088904   5.04% better
//   SUB=32, 4-bit log grid         0.088199   5.80% better
//   SUB=32, 4-bit codebook         0.088288   5.70% better
//
// Both layouts spend 16 bytes on scales. Halving the sub-block is worth 6.11%;
// dropping scales from 8 bits to 4 costs only 0.49%. The trade is one-sided
// because the sub-block absmax ratios span a narrow range (0.21 to 1.0, mean
// 0.64), so fifteen levels cover them comfortably.
//
// The codebook is chosen over the log grid despite being 0.1% worse: a log
// grid needs exp() per sub-block in the decode kernel, a codebook is a
// 16-entry lookup that fits in constant memory. It also has no zero-collapse
// failure mode, which a uniform 4-bit grid does: with d = max/15, a sub-block
// whose absmax falls below max/30 rounds to zero and the whole sub-block is
// zeroed.
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

// GGML_COMMON_DECL_C must be defined before ggml-common.h or ggml_half and
// the block typedefs are compiled out. ggml-quants.h does the same.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK_NF4DQ     1024   // weights per superblock (see note below)
#define NF4DQ_SUB      32   // weights per sub-block (measured, see above)
#define NF4DQ_NSUB    (QK_NF4DQ / NF4DQ_SUB)

typedef struct {
    uint8_t     qs[QK_NF4DQ / 2];     // 512 bytes: packed 4-bit weight indices
    uint8_t     sc[NF4DQ_NSUB / 2];   //  16 bytes: packed 4-bit scale indices
    ggml_half   d;                    //   2 bytes: super-scale
    uint8_t     pad[2];               //   2 bytes: 4-byte alignment, see above
} block_nf4dq;                        // 532 bytes total

// Compile-time guarantee that no padding crept in. A silently padded struct
// would produce a file that is the right size on one compiler and the wrong
// size on another, which is exactly the class of bug that is invisible until
// someone else tries to load the checkpoint.
typedef char nf4dq_size_check[(sizeof(block_nf4dq) == 532) ? 1 : -1];

// The alignment assert is the one that actually matters: a 532-byte struct
// that the compiler still aligns to 2 would pass the size check and fail in
// the CUDA kernel.
typedef char nf4dq_align_check[(sizeof(block_nf4dq) % 4 == 0) ? 1 : -1];

// The 16 NF4 levels: quantiles of a standard normal, normalised to [-1, 1].
// Byte-identical to NF4_LEVELS in harness/qembed.py and validate.py.
extern const float NF4DQ_LEVELS[16];

// The same levels on a 1/127 grid, which is what both the CPU reference and
// the CUDA dp4a path actually use. NF4DQ_LEVELS is kept for documentation and
// for fitting work; it is not on the encode or decode path.
extern const int8_t NF4DQ_I8[16];

// The 16 sub-block scale levels, as fractions of the superblock's largest
// sub-block absmax. Fitted by Lloyd-Max to measured ratios; the top level is
// pinned to 1.0 because one sub-block per superblock attains it by definition.
extern const float NF4DQ_SCALE_LEVELS[16];

// NOTE: block_nf4dq is declared here for the standalone build. When wiring
// into ggml proper, move the struct and the QK_ defines into ggml-common.h
// alongside the other block types, and reduce this header to declarations.

// Reference (non-SIMD) quantise and dequantise.
//   k must be a multiple of QK_NF4DQ.
void quantize_row_nf4dq_ref  (const float * restrict x, block_nf4dq * restrict y, int64_t k);
void dequantize_row_nf4dq    (const block_nf4dq * restrict x, float * restrict y, int64_t k);

// Convenience: quantise then dequantise, reporting relative Frobenius error.
// This is the gate. It should return ~0.0918 on real transformer weights.
float nf4dq_roundtrip_error(const float * x, int64_t k);

// Row-loop wrapper, matching ggml's quantize_<type> convention.
// Returns bytes written, or 0 if n_per_row is not a multiple of QK_NF4DQ.
size_t quantize_nf4dq(const float * restrict src, void * restrict dst,
                      int64_t nrow, int64_t n_per_row);

#ifdef __cplusplus
}
#endif
