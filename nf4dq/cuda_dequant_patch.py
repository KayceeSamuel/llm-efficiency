"""
Stage 1 of the CUDA path: dequantisation only.

This makes NF4DQ models run on the GPU by unpacking to fp16 and handing the
result to cuBLAS. It is not the fast path, but it is the one with the least
that can go wrong, and it gives a correctness gate the fused kernel can be
checked against.

Stage 2 (vec_dot_nf4dq_q8_1 in mmvq) is where decode throughput comes from,
and it is much easier to debug once the model is known to load and generate
correctly through this route.

Run in Colab. Edits /content/llama.cpp in place.
"""

P = "/content/llama.cpp"

# ---------------------------------------------------------------- convert.cu
# The dequantise kernel, plus its launcher and the three dispatch sites.

KERNEL = r'''
// NF4DQ: 1024-weight superblock, 32 sub-blocks of 32, 4-bit weight indices
// into an int8 codebook and 4-bit scale indices into a float codebook.
//
// The codebooks live in __constant__ memory. A global-memory lookup per
// weight would dominate the kernel: there are two lookups per weight here,
// one for the level and one for the sub-block scale.
__constant__ static const int8_t knf4dq_i8[16] = {
    -127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127,
};
__constant__ static const float knf4dq_scale[16] = {
    0.1126f, 0.1387f, 0.1647f, 0.1973f, 0.2485f, 0.3740f, 0.4436f, 0.4997f,
    0.5505f, 0.5998f, 0.6500f, 0.7036f, 0.7624f, 0.8286f, 0.9051f, 1.0000f,
};

template<typename dst_t>
static __global__ void dequantize_block_nf4dq(const void * __restrict__ vx,
                                              dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;                    // superblock
    const block_nf4dq * x = (const block_nf4dq *) vx + i;
    dst_t * y = yy + i * QK_NF4DQ;

    // d already carries the /127 folded in by the encoder, so there is no
    // division on this path. See ggml-nf4dq.c.
    const float d = __half2float(x->d);

    const int s = threadIdx.x;                       // one sub-block per thread
    if (s >= NF4DQ_NSUB) return;

    const uint8_t sbyte = x->sc[s >> 1];
    const uint8_t si    = (s & 1) ? (sbyte >> 4) : (sbyte & 0x0F);
    const float   scale = d * knf4dq_scale[si];

#pragma unroll
    for (int j = 0; j < NF4DQ_SUB; ++j) {
        const int     pos  = s * NF4DQ_SUB + j;
        const uint8_t byte = x->qs[pos >> 1];
        const uint8_t idx  = (pos & 1) ? (byte >> 4) : (byte & 0x0F);
        y[pos] = (dst_t) ((float) knf4dq_i8[idx] * scale);
    }
}

'''

LAUNCHER = r'''
template<typename dst_t>
static void dequantize_row_nf4dq_cuda(const void * vx, dst_t * y,
                                      const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_NF4DQ - 1) / QK_NF4DQ;
    // 32 threads, one per sub-block. Not 32 because that is ggml's habit:
    // NF4DQ_NSUB happens to be 32 as well, so the two coincide.
    dequantize_block_nf4dq<<<nb, NF4DQ_NSUB, 0, stream>>>(vx, y);
}

'''

p = f"{P}/ggml/src/ggml-cuda/convert.cu"
s = open(p).read()

if "dequantize_block_nf4dq" not in s:
    # kernel goes before the iq4_nl kernel
    s = s.replace("static __global__ void dequantize_block_iq4_nl(",
                  KERNEL + "static __global__ void dequantize_block_iq4_nl(", 1)
    # launcher before the iq4_nl launcher
    s = s.replace("template<typename dst_t>\nstatic void dequantize_row_iq4_nl_cuda(",
                  LAUNCHER + "template<typename dst_t>\nstatic void dequantize_row_iq4_nl_cuda(", 1)
    # all three dispatch sites
    n = s.count("        case GGML_TYPE_IQ4_NL:\n            return dequantize_row_iq4_nl_cuda;")
    s = s.replace("        case GGML_TYPE_IQ4_NL:\n            return dequantize_row_iq4_nl_cuda;",
                  "        case GGML_TYPE_NF4DQ:\n            return dequantize_row_nf4dq_cuda;\n"
                  "        case GGML_TYPE_IQ4_NL:\n            return dequantize_row_iq4_nl_cuda;")
    if '#include "ggml-nf4dq.h"' not in s:
        s = s.replace('#include "convert.cuh"',
                      '#include "convert.cuh"\n#include "../ggml-nf4dq.h"', 1)
    open(p, "w").write(s)
    print(f"convert.cu: kernel + launcher + {n} dispatch sites")
else:
    print("convert.cu: already patched")

# ------------------------------------------------------------- ggml-cuda.cu
# supports_op, so the backend advertises the type instead of falling back to
# CPU. Two switch sites, both simple case additions.

p = f"{P}/ggml/src/ggml-cuda/ggml-cuda.cu"
s = open(p).read()
before = s.count("case GGML_TYPE_NF4DQ:")
s = s.replace("                    case GGML_TYPE_IQ4_NL:",
              "                    case GGML_TYPE_NF4DQ:\n"
              "                    case GGML_TYPE_IQ4_NL:")
open(p, "w").write(s)
print(f"ggml-cuda.cu: NF4DQ cases {before} -> {s.count('case GGML_TYPE_NF4DQ:')}")

print("\nNOT patched yet, deliberately: mmvq.cu (the fused decode path).")
print("Dequantise-then-cuBLAS works first; the fused kernel is stage 2.")
