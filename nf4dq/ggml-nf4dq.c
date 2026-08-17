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
        nf4dq_boundaries[i] = (NF4DQ_LEVELS[i] + NF4DQ_LEVELS[i + 1]) * 0.5f;
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
// Self-contained fp16 conversion so this file can be tested without pulling
// in ggml. Replace with ggml_fp32_to_fp16 / ggml_fp16_to_fp32 when wiring in.

static nf4dq_half fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;

    if (exp <= 0) return (nf4dq_half)sign;              // underflow to zero
    if (exp >= 31) return (nf4dq_half)(sign | 0x7C00u); // overflow to inf

    // round to nearest even
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man & 0x1FFFu) > 0x1000u ||
        (((man & 0x1FFFu) == 0x1000u) && (h & 1u))) h++;
    return (nf4dq_half)h;
}

static float fp16_to_fp32(nf4dq_half h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t x;

    if (exp == 0) {
        if (man == 0) { x = sign; }
        else {                                   // subnormal
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            x = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (man << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &x, sizeof(f));
    return f;
}

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

        yb->d = fp32_to_fp16(max_absmax);

        // Read the super-scale back through fp16 before using it. The decoder
        // only ever sees the rounded value, so the encoder must quantise
        // against the same number or the two disagree by the rounding error.
        const float d_eff = fp16_to_fp32(yb->d);

        // Level 2: each sub-block absmax as a fraction of the superblock's,
        // mapped to a 16-entry codebook and packed two indices per byte.
        memset(yb->sc, 0, sizeof(yb->sc));
        memset(yb->qs, 0, sizeof(yb->qs));

        for (int s = 0; s < NF4DQ_NSUB; s++) {
            const float ratio = (d_eff > 0.0f) ? (absmax[s] / d_eff) : 0.0f;
            const uint8_t si = nf4dq_nearest_scale(ratio);
            if ((s & 1) == 0) yb->sc[s >> 1] |= si;
            else              yb->sc[s >> 1] |= (uint8_t)(si << 4);

            // Weights, against the reconstructed sub-block scale, for the
            // same reason: quantise against what the decoder will use.
            const float scale = d_eff * NF4DQ_SCALE_LEVELS[si];
            const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int     pos = s * NF4DQ_SUB + j;
                const float   v   = (inv > 0.0f) ? xb[pos] * inv : 0.0f;
                const uint8_t idx = nf4dq_nearest(v);

                if ((pos & 1) == 0) yb->qs[pos >> 1] |= idx;
                else                yb->qs[pos >> 1] |= (uint8_t)(idx << 4);
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
                const uint8_t byte = xb->qs[pos >> 1];
                const uint8_t idx  = ((pos & 1) == 0) ? (byte & 0x0F)
                                                      : (byte >> 4);
                yb[pos] = NF4DQ_LEVELS[idx] * scale;
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
