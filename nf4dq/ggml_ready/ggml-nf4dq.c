// ggml-nf4dq.c — reference implementation.
//
// Correctness first, speed later. Nothing here is SIMD or CUDA. The point of
// this file is to establish that the format round-trips to the expected error
// so that a fast path has something to be checked against.

#include "ggml-nf4dq.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

// The int8 codebook, NF4 levels on a 1/127 grid. This is what the CUDA path
// uses so it can dp4a; the CPU reference uses the same values so the two are
// bit-identical. Measured cost of the rounding: 0.088714 -> 0.088922 on
// kurtosis-1.4 data, 0.101801 -> 0.102338 with 16-sigma outliers.
const int8_t NF4DQ_I8[16] = {
    -127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127,
};

const float NF4DQ_LEVELS[16] = {
    -1.0f,               -0.6961928009986877f, -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
    -0.09105003625154495f, 0.0f,
     0.07958029955625534f, 0.16093020141124725f,  0.24611230194568634f,
     0.33791524171829224f, 0.44070982933044434f,  0.5626170039176941f,
     0.7229568362236023f,  1.0f,
};

// Sub-block scale levels: absmax_subblock / absmax_superblock. Fitted by
// Lloyd-Max, top level pinned to 1.0 (one sub-block attains it by definition).
//
// Fitted on a MIX of outlier-free and 16-sigma-outlier data. An earlier
// version fitted on outlier-free data alone spanned only 0.3355 to 1.0 and
// failed badly when a superblock contained an extreme weight: the other
// sub-blocks have ratios near 0.06, clamp to the 0.3355 floor, and their
// weights collapse into the lowest NF4 levels. Measured 0.139 against 0.101
// for this codebook on outlier data. Experiment 2 measured max/std of 15.4
// and 16.8 on real weights, so that regime is the normal case, not an edge.
const float NF4DQ_SCALE_LEVELS[16] = {
    0.1126f, 0.1387f, 0.1647f, 0.1973f, 0.2485f, 0.3740f, 0.4436f, 0.4997f,
    0.5505f, 0.5998f, 0.6500f, 0.7036f, 0.7624f, 0.8286f, 0.9051f, 1.0000f,
};

static float nf4dq_scale_bounds[15];

// Midpoints between adjacent levels. A value is assigned to the level whose
// cell it falls in, which for a monotonic codebook is nearest-neighbour.
static float nf4dq_boundaries[15];
static int   nf4dq_boundaries_ready = 0;

static void nf4dq_init_boundaries(void) {
    if (nf4dq_boundaries_ready) return;
    for (int i = 0; i < 15; i++) {
        nf4dq_boundaries[i] = ((float) NF4DQ_I8[i] + (float) NF4DQ_I8[i + 1])
                              * 0.5f / 127.0f;
        nf4dq_scale_bounds[i] =
            (NF4DQ_SCALE_LEVELS[i] + NF4DQ_SCALE_LEVELS[i + 1]) * 0.5f;
    }
    nf4dq_boundaries_ready = 1;
}

// Mirrors torch.bucketize(v, boundaries, right=False): the index of the first
// boundary >= v, which equals the count of boundaries strictly less than v.
// Ties land on the upper level in both implementations.
static inline uint8_t nf4dq_nearest(float v) {
    uint8_t i = 0;
    while (i < 15 && v > nf4dq_boundaries[i]) i++;
    return i;
}

static inline uint8_t nf4dq_nearest_scale(float r) {
    uint8_t i = 0;
    while (i < 15 && r > nf4dq_scale_bounds[i]) i++;
    return i;
}

// ---------------------------------------------------------------- fp16
// ggml's conversion, not a local copy. The standalone build carries its own
// so it can compile without ggml; inside the tree these are better tested and
// may be hardware-accelerated.

#include "ggml-impl.h"

#define fp32_to_fp16(x) GGML_FP32_TO_FP16(x)
#define fp16_to_fp32(x) GGML_FP16_TO_FP32(x)

// ---------------------------------------------------------------- quantise

void quantize_row_nf4dq_ref(const float * restrict x, block_nf4dq * restrict y,
                            int64_t k) {
    assert(k % QK_NF4DQ == 0);
    nf4dq_init_boundaries();

    const int64_t nb = k / QK_NF4DQ;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_NF4DQ;
        block_nf4dq * yb = &y[i];

        // Level 1: absmax per sub-block, and the superblock's largest.
        float absmax[NF4DQ_NSUB];
        float max_absmax = 0.0f;

        for (int s = 0; s < NF4DQ_NSUB; s++) {
            float m = 0.0f;
            for (int j = 0; j < NF4DQ_SUB; j++) {
                const float a = fabsf(xb[s * NF4DQ_SUB + j]);
                if (a > m) m = a;
            }
            absmax[s] = m;
            if (m > max_absmax) max_absmax = m;
        }

        // Fold the codebook's /127 into the stored scale, as IQ4_NL does,
        // so neither the CPU nor the CUDA decode path needs a division.
        yb->d = fp32_to_fp16(max_absmax / 127.0f);

        // Read the super-scale back through fp16 before using it. The decoder
        // only ever sees the rounded value, so the encoder must quantise
        // against the same number or the two disagree by the rounding error.
        const float d_eff = fp16_to_fp32(yb->d);

        // Level 2: each sub-block absmax as a fraction of the superblock's,
        // mapped to a 16-entry codebook and packed two indices per byte.
        yb->pad[0] = yb->pad[1] = 0;   // deterministic files
        memset(yb->sc, 0, sizeof(yb->sc));
        memset(yb->qs, 0, sizeof(yb->qs));

        for (int s = 0; s < NF4DQ_NSUB; s++) {
            const float ratio = (d_eff > 0.0f)
                                ? (absmax[s] / (d_eff * 127.0f)) : 0.0f;
            const uint8_t si = nf4dq_nearest_scale(ratio);
            if ((s & 1) == 0) yb->sc[s >> 1] |= si;
            else              yb->sc[s >> 1] |= (uint8_t)(si << 4);

            // Weights, against the reconstructed sub-block scale, for the
            // same reason: quantise against what the decoder will use.
            // d_eff now carries the /127, so the reconstruction is
            // I8[idx] * d_eff * SCALE[si] and the value being quantised
            // against is that same grid.
            const float scale = d_eff * 127.0f * NF4DQ_SCALE_LEVELS[si];
            const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int     pos = s * NF4DQ_SUB + j;
                const float   v   = (inv > 0.0f) ? xb[pos] * inv : 0.0f;
                const uint8_t idx = nf4dq_nearest(v);

                // ggml packing: within a 32-element sub-block, byte k holds
                // element k in the low nibble and element k+16 in the high
                // nibble. NOT interleaved. get_int_from_table_16 in the CUDA
                // vec_dot depends on this exact layout.
                const int byte = s * (NF4DQ_SUB / 2) + (j & 15);
                if (j < 16) yb->qs[byte] |= idx;
                else        yb->qs[byte] |= (uint8_t)(idx << 4);
            }
        }
    }
}

// -------------------------------------------------------------- dequantise

void dequantize_row_nf4dq(const block_nf4dq * restrict x, float * restrict y,
                          int64_t k) {
    assert(k % QK_NF4DQ == 0);
    const int64_t nb = k / QK_NF4DQ;

    for (int64_t i = 0; i < nb; i++) {
        const block_nf4dq * xb = &x[i];
        float * yb = y + i * QK_NF4DQ;
        const float d = fp16_to_fp32(xb->d);
        for (int s = 0; s < NF4DQ_NSUB; s++) {
            const uint8_t sbyte = xb->sc[s >> 1];
            const uint8_t si    = ((s & 1) == 0) ? (sbyte & 0x0F) : (sbyte >> 4);
            const float scale   = d * NF4DQ_SCALE_LEVELS[si];

            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int pos = s * NF4DQ_SUB + j;
                const uint8_t byte = xb->qs[s * (NF4DQ_SUB / 2) + (j & 15)];
                const uint8_t idx  = (j < 16) ? (byte & 0x0F) : (byte >> 4);
                // I8 grid, matching vec_dot_nf4dq_q8_1 exactly. d carries
                // the /127.
                yb[pos] = (float) NF4DQ_I8[idx] * scale;
            }
        }
    }
}

// ------------------------------------------------------------------- gate

float nf4dq_roundtrip_error(const float * x, int64_t k) {
    const int64_t nb = k / QK_NF4DQ;
    block_nf4dq * q = (block_nf4dq *)malloc((size_t)nb * sizeof(block_nf4dq));
    float * r = (float *)malloc((size_t)k * sizeof(float));
    if (!q || !r) { free(q); free(r); return -1.0f; }

    quantize_row_nf4dq_ref(x, q, k);
    dequantize_row_nf4dq(q, r, k);

    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < k; i++) {
        const double e = (double)x[i] - (double)r[i];
        num += e * e;
        den += (double)x[i] * (double)x[i];
    }

    free(q); free(r);
    return (den > 0.0) ? (float)sqrt(num / den) : 0.0f;
}

// ---------------------------------------------------------- row wrapper
// The entry point ggml_quantize_chunk calls. Kept here rather than in the
// integration patch so the standalone build exercises the same code path.

size_t quantize_nf4dq(const float * restrict src, void * restrict dst,
                      int64_t nrow, int64_t n_per_row) {
    if (n_per_row % QK_NF4DQ) return 0;   // caller must check; 0 means refused
    const size_t row_size = (size_t)(n_per_row / QK_NF4DQ) * sizeof(block_nf4dq);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_nf4dq_ref(src, (block_nf4dq *) qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return (size_t) nrow * row_size;
}


// ------------------------------------------------------- CPU vec_dot
// Dot product of one NF4DQ row against a Q8_0-quantised activation row.
//
// Q8_0 blocks are 32 elements and an NF4DQ sub-block is also 32 weights, so
// they align one to one: sub-block s pairs with q8 block s. Same alignment
// IQ4_NL relies on, which is why Q8_0 is the right vec_dot_type here.
//
// Scalar and unvectorised. This exists so CPU inference runs as a correctness
// gate; the performance path is vec_dot_nf4dq_q8_1 in CUDA.

#include "ggml-common.h"

void ggml_vec_dot_nf4dq_q8_0(int n, float * restrict s, size_t bs,
                             const void * restrict vx, size_t bx,
                             const void * restrict vy, size_t by, int nrc) {
    (void) bs; (void) bx; (void) by; (void) nrc;

    const block_nf4dq * restrict x = (const block_nf4dq *) vx;
    const block_q8_0  * restrict y = (const block_q8_0  *) vy;

    const int nb = n / QK_NF4DQ;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);   // already carries the /127

        for (int s_ = 0; s_ < NF4DQ_NSUB; s_++) {
            const uint8_t sbyte = x[i].sc[s_ >> 1];
            const uint8_t si    = (s_ & 1) ? (sbyte >> 4) : (sbyte & 0x0F);

            // one q8_0 block per sub-block
            const block_q8_0 * yb = &y[i * NF4DQ_NSUB + s_];

            int sumi = 0;
            for (int j = 0; j < NF4DQ_SUB; j++) {
                const uint8_t byte = x[i].qs[s_ * (NF4DQ_SUB / 2) + (j & 15)];
                const uint8_t idx  = (j < 16) ? (byte & 0x0F) : (byte >> 4);
                sumi += (int) NF4DQ_I8[idx] * (int) yb->qs[j];
            }

            sumf += d * NF4DQ_SCALE_LEVELS[si] * fp16_to_fp32(yb->d) * sumi;
        }
    }

    *s = sumf;
}
