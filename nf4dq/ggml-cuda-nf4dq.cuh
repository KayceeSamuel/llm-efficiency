// ggml-cuda/nf4dq.cuh
//
// CUDA path for GGML_TYPE_NF4DQ.
//
// STATUS: written against the IQ4_XS kernels as a template, NOT YET COMPILED
// OR PROFILED. Treat every performance claim here as a hypothesis. The
// correctness gate is bit-identical output against dequantize_row_nf4dq on
// the CPU, on the same block; anything less and downstream numbers are
// measuring the bug rather than the format.
//
// WHY IQ4_XS IS THE TEMPLATE
//
// It is the only shipped ggml type that combines a codebook lookup with
// per-sub-block scales inside a superblock, which is exactly NF4DQ's shape.
// IQ4_NL has the codebook but no sub-block scales; the K-quants have the
// scales but compute levels arithmetically.
//
// THE dp4a REQUIREMENT, AND WHAT IT COSTS
//
// IQ4_NL stores its codebook as int8 (kvalues_iq4nl) rather than float, so
// the inner loop can use dp4a: one instruction for a 4-way int8 dot product.
// A float codebook would need a multiply per weight and give up most of the
// benefit.
//
// NF4DQ therefore also needs an int8 codebook, which means rounding the NF4
// levels to a 1/127 grid. Measured cost of that rounding on weights matching
// the kurtosis-1.4 profile: 0.088714 to 0.088922, about 0.23%. On
// 16-sigma-outlier data: 0.101801 to 0.102338, about 0.53%. Both are far
// below what the fused path buys, so the trade is clearly worth taking.
//
// ENCODER CHANGE THIS IMPLIES
//
// The reference encoder stores d = max_absmax. With an int8 codebook the
// reconstruction is (tab[idx] / 127) * d * SCALE[si], and that division would
// run per weight. Store d = max_absmax / 127 instead, exactly as IQ4_NL folds
// the same constant into its scale, and the decode path becomes
// tab[idx] * d * SCALE[si] with no division at all.
//
// ggml-nf4dq.c must be updated to match:
//     yb->d = GGML_FP32_TO_FP16(max_absmax / 127.0f);
// and its dequantise path to:
//     yb[pos] = (float) NF4DQ_I8[idx] * d * NF4DQ_SCALE_LEVELS[si];
//
// A CPU/GPU mismatch here would be silent: both would produce plausible
// output differing by a factor of 127 somewhere, which perplexity would
// report as a quality regression rather than a bug.

#pragma once

#include "common.cuh"

// NF4 levels on a 1/127 grid. Symmetric, unlike kvalues_iq4nl, because NF4's
// codebook is near-symmetric where IQ4_NL's is not.
GGML_TABLE_BEGIN(int8_t, kvalues_nf4dq, 16)
    -127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127,
GGML_TABLE_END()

// Sub-block absmax as a fraction of the superblock's largest. Left in float:
// there is one of these per 32 weights, not per weight, so a float multiply
// here is negligible and int8-ing it would cost accuracy for nothing.
GGML_TABLE_BEGIN(float, kscales_nf4dq, 16)
    0.1126f, 0.1387f, 0.1647f, 0.1973f, 0.2485f, 0.3740f, 0.4436f, 0.4997f,
    0.5505f, 0.5998f, 0.6500f, 0.7036f, 0.7624f, 0.8286f, 0.9051f, 1.0000f,
GGML_TABLE_END()

// ---------------------------------------------------------------------------
// Bulk dequantisation. Used by prefill and anything that materialises the
// tensor. Not on the decode critical path, so clarity over cleverness.
// ---------------------------------------------------------------------------

static __global__ void dequantize_block_nf4dq(const void * __restrict__ vx,
                                              dst_t * __restrict__ yy,
                                              const int64_t nb) {
    const int64_t i = blockIdx.x;
    if (i >= nb) return;

    const block_nf4dq * x = (const block_nf4dq *) vx + i;
    dst_t * y = yy + i * QK_NF4DQ;

    const float d = __half2float(x->d);

    // One sub-block per thread: 32 sub-blocks of 32 weights each.
    const int s = threadIdx.x;
    if (s >= NF4DQ_NSUB) return;

    const uint8_t sbyte = x->sc[s >> 1];
    const uint8_t si    = (s & 1) ? (sbyte >> 4) : (sbyte & 0x0F);
    const float   scale = d * kscales_nf4dq[si];

#pragma unroll
    for (int j = 0; j < NF4DQ_SUB; ++j) {
        const int     pos  = s * NF4DQ_SUB + j;
        const uint8_t byte = x->qs[pos >> 1];
        const uint8_t idx  = (pos & 1) ? (byte >> 4) : (byte & 0x0F);
        y[pos] = (dst_t) ((float) kvalues_nf4dq[idx] * scale);
    }
}

// ---------------------------------------------------------------------------
// Decode path. THIS IS WHERE THE PERFORMANCE LIVES.
//
// It must unpack inside the dot product, in registers. If it materialises to
// half or float first and then multiplies, the bandwidth benefit is gone and
// the result is slower than what it replaces. That is exactly the failure
// mode bitsandbytes has: measured 6.44 tok/s on an A100-80GB, which is 4.1%
// of that card's roofline for a 12.968 GB model.
//
// LAYOUT LUCK WORTH NOTING
//
// A q8_1 activation block is 32 elements and an NF4DQ sub-block is also 32
// weights, so they align exactly one-to-one. IQ4_XS has the same alignment,
// which is why its loop structure transfers with only the scale extraction
// changed. Our superblock holds 32 sub-blocks against IQ4_XS's 8, so the
// caller iterates four times as far, but each step is identical.
// ---------------------------------------------------------------------------

#define VDR_NF4DQ_Q8_1_MMVQ 4
#define VDR_NF4DQ_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_nf4dq_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_nf4dq * bq = (const block_nf4dq *) vbq + kbx;

    // iqs advances 4 ints (32 nibbles, 32 weights) per sub-block, so iqs/4 is
    // the sub-block index and also the q8_1 block index.
    const int s = iqs / 4;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int  aux_q4 = get_int_b4(bq->qs, iqs + j);
        const int2 v      = get_int_from_table_16(aux_q4, kvalues_nf4dq);

        const int u0 = get_int_b4(bq8_1[s].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[s].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    // Two scale lookups, both cheap: one packed 4-bit index into a 16-entry
    // constant table, applied once per 32 weights rather than per weight.
    const uint8_t sbyte = bq->sc[s >> 1];
    const uint8_t si    = (s & 1) ? (sbyte >> 4) : (sbyte & 0x0F);

    // bq->d already carries the /127 from the int8 codebook, so there is no
    // division on this path. See the encoder note at the top of this file.
    const float d = __half2float(bq->d) * __low2float(bq8_1[s].ds);
    return d * kscales_nf4dq[si] * sumi;
}

// ---------------------------------------------------------------------------
// OPEN QUESTIONS, to settle on hardware rather than by reading
//
// 1. Superblock size. QK_NF4DQ is 1024, four times ggml's usual QK_K of 256.
//    The mmvq launcher derives its grid from blck_size, so the loop bounds
//    should follow, but the register budget and the shared-memory assumptions
//    in mmq.cu were written for 256 and have not been checked against 1024.
//    If occupancy drops, the superblock is not load-bearing for correctness:
//    512 also divides 5120 and 17408, at 4.1562 bpw instead of 4.1406.
//
// 2. get_int_b4 alignment. It assumes 4-byte alignment within qs. block_nf4dq
//    starts with qs, so the first element is aligned, but confirm the struct
//    itself is 4-byte aligned in the tensor buffer. 530 is not a multiple of
//    4, so consecutive blocks will NOT all start aligned. This is the single
//    most likely source of a wrong answer or a misaligned-access fault, and
//    it is worth checking before anything else.
//
//    If it bites, the fix is a 2-byte pad to 532 bytes (4.1562 bpw), which is
//    cheap insurance. IQ4_XS is 136 bytes and Q4_K is 144, both multiples of
//    4, which may be why this has not come up upstream.
//
// 3. MMQ path. Only MMVQ is written here. mmq.cu needs its own tile loader
//    for batched prefill. Decode is the priority; prefill already runs at
//    1684 tok/s on an A100 through the existing path.
// ---------------------------------------------------------------------------
