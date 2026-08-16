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

// Midpoints between adjacent levels. A value is assigned to the level whose
// cell it falls in, which for a monotonic codebook is nearest-neighbour.
static float nf4dq_boundaries[15];
static int   nf4dq_boundaries_ready = 0;

static void nf4dq_init_boundaries(void) {
    if (nf4dq_boundaries_ready) return;
    for (int i = 0; i < 15; i++) {
        nf4dq_boundaries[i] = (NF4DQ_LEVELS[i] + NF4DQ_LEVELS[i + 1]) * 0.5f;
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

        // Level 1: absmax per 64-weight sub-block.
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

        // Level 2: quantise those absmax values to uint8 against one fp16
        // super-scale. This is the "double" in double quantisation, and it
        // is where the format beats a plain fp16-scale layout on size.
        //
        // absmax is non-negative by construction, so an unsigned 0..255 grid
        // with no zero point is the right shape. No offset term is used; the
        // 27B measurement it is validated against does not use one either.
        float d = (max_absmax > 0.0f) ? (max_absmax / 255.0f) : 0.0f;
        yb->d = fp32_to_fp16(d);

        // Read the super-scale back through fp16 before using it. The decoder
        // only ever sees the rounded value, so the encoder must quantise
        // against the same number or the two disagree by the rounding error.
        const float d_eff = fp16_to_fp32(yb->d);

        for (int s = 0; s < NF4DQ_NSUB; s++) {
            int q = (d_eff > 0.0f) ? (int)lrintf(absmax[s] / d_eff) : 0;
            if (q < 0)   q = 0;
            if (q > 255) q = 255;
            yb->sc[s] = (uint8_t)q;
        }

        // Weights, against the reconstructed sub-block scale, for the same
        // reason: quantise against what the decoder will actually use.
        memset(yb->qs, 0, sizeof(yb->qs));

        for (int s = 0; s < NF4DQ_NSUB; s++) {
            const float scale = d_eff * (float)yb->sc[s];
            const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int      pos = s * NF4DQ_SUB + j;
                const float    v   = (inv > 0.0f) ? xb[pos] * inv : 0.0f;
                const uint8_t  idx = nf4dq_nearest(v);

                // Low nibble for even positions, high for odd. Identical to
                // the packing in harness/qembed.py.
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
            const float scale = d * (float)xb->sc[s];

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
