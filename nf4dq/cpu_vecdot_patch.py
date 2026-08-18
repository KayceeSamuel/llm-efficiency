"""Adds the CPU-side pieces NF4DQ needs: a scalar vec_dot and the switch cases.

Run inside Colab. Edits /content/llama.cpp in place.
"""
P = "/content/llama.cpp"

# ---------------------------------------------------------------- vec_dot
# Appended to ggml-nf4dq.c. Scalar and unoptimised on purpose: this exists so
# CPU inference works as a correctness gate, not for speed. The fast path is
# the CUDA kernel.
VECDOT = r'''

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
                const int     pos  = s_ * NF4DQ_SUB + j;
                const uint8_t byte = x[i].qs[pos >> 1];
                const uint8_t idx  = (pos & 1) ? (byte >> 4) : (byte & 0x0F);
                sumi += (int) NF4DQ_I8[idx] * (int) yb->qs[j];
            }

            sumf += d * NF4DQ_SCALE_LEVELS[si] * fp16_to_fp32(yb->d) * sumi;
        }
    }

    *s = sumf;
}
'''

p = f"{P}/ggml/src/ggml-nf4dq.c"
src = open(p).read()
if "ggml_vec_dot_nf4dq_q8_0" not in src:
    open(p, "w").write(src + VECDOT)
    print("vec_dot appended")

# declaration
p = f"{P}/ggml/src/ggml-cpu/quants.h"
s = open(p).read()
if "ggml_vec_dot_nf4dq_q8_0" not in s:
    s = s.replace(
        "void ggml_vec_dot_iq4_nl_q8_0 (int n,",
        "void ggml_vec_dot_nf4dq_q8_0 (int n, float * GGML_RESTRICT s, size_t bs,"
        " const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,"
        " size_t by, int nrc);\nvoid ggml_vec_dot_iq4_nl_q8_0 (int n,", 1)
    open(p, "w").write(s)
    print("declaration added")

# CPU traits table
p = f"{P}/ggml/src/ggml-cpu/ggml-cpu.c"
s = open(p).read()
if "GGML_TYPE_NF4DQ" not in s:
    s = s.replace(
"""    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,""",
"""    [GGML_TYPE_NF4DQ] = {
        // No .from_float: quantisation goes through quantize_nf4dq in
        // ggml_quantize_chunk, and nothing converts activations to this type.
        .vec_dot                  = ggml_vec_dot_nf4dq_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,""", 1)
    open(p, "w").write(s)
    print("cpu traits added")

# fall-through switches in ops.cpp
p = f"{P}/ggml/src/ggml-cpu/ops.cpp"
s = open(p).read()
before = s.count("case GGML_TYPE_NF4DQ:")
if before:
    print(f"ops.cpp: already has {before} cases, skipping")
s = s if before else s.replace("        case GGML_TYPE_IQ4_NL:\n        case GGML_TYPE_IQ4_XS:",
              "        case GGML_TYPE_NF4DQ:\n        case GGML_TYPE_IQ4_NL:\n"
              "        case GGML_TYPE_IQ4_XS:")
open(p, "w").write(s)
print(f"ops.cpp cases: {before} -> {s.count('case GGML_TYPE_NF4DQ:')}")
