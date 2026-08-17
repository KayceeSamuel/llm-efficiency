"""
Stage 2 of the CUDA path: the fused decode kernel.

Stage 1 (dequantise then cuBLAS) works and gives 2.6 tok/s on an L4. That path
unpacks the entire 4.35 GB weight set to fp16 on every forward pass, which is
most of what it costs. The fused kernel keeps the weights packed and unpacks
inside the dot product, in registers.

The activation type is Q8_0 rather than Q8_1. Q8_0 blocks are 32 elements and
an NF4DQ sub-block is 32 weights, so they pair one to one. That is the same
alignment IQ4_NL relies on.

WHAT TO CHECK IF THE OUTPUT IS WRONG

Stage 1 is the reference. It produced coherent text, which means the packing,
the alignment, the codebooks and the encoder are all correct. So anything
broken after this patch is in the vec_dot itself, not upstream. Set
ggml_cuda_should_use_mmvq back to false for NF4DQ to fall back and compare.

Run in Colab. Edits /content/llama.cpp in place.
"""

P = "/content/llama.cpp"

# --------------------------------------------------------------- vecdotq.cuh
# The kernel. Uses dp4a on the int8 codebook, which is why the codebook was
# rounded to a 1/127 grid in the first place: measured cost 0.23% on
# reconstruction, against a fused path that is the whole point of the exercise.

VECDOT = r'''
// NF4DQ: 1024-weight superblock, 32 sub-blocks of 32 weights.
//
// Two constant tables. The weight codebook is int8 so the inner loop can use
// dp4a, one instruction per four multiply-accumulates. The scale codebook
// stays float because it is consulted once per 32 weights, not per weight.
__constant__ static const int8_t kvalues_nf4dq[16] = {
    -127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127,
};
__constant__ static const float kscales_nf4dq[16] = {
    0.1126f, 0.1387f, 0.1647f, 0.1973f, 0.2485f, 0.3740f, 0.4436f, 0.4997f,
    0.5505f, 0.5998f, 0.6500f, 0.7036f, 0.7624f, 0.8286f, 0.9051f, 1.0000f,
};

#define VDR_NF4DQ_Q8_1_MMVQ 4
#define VDR_NF4DQ_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_nf4dq_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_nf4dq * bq = (const block_nf4dq *) vbq + kbx;

    // iqs advances 4 ints (32 nibbles, 32 weights) per sub-block, so iqs/4 is
    // both the sub-block index and the q8_1 block index.
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

    const uint8_t sbyte = bq->sc[s >> 1];
    const uint8_t si    = (s & 1) ? (sbyte >> 4) : (sbyte & 0x0F);

    // bq->d already carries the /127 from the int8 codebook, folded in by the
    // encoder, so there is no division here.
    const float d = __half2float(bq->d) * __low2float(bq8_1[s].ds);
    return d * kscales_nf4dq[si] * sumi;
}

'''

p = f"{P}/ggml/src/ggml-cuda/vecdotq.cuh"
s = open(p).read()
if "vec_dot_nf4dq_q8_1" not in s:
    anchor = "#define VDR_IQ4_NL_Q8_1_MMVQ"
    if anchor not in s:
        anchor = "static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1("
    s = s.replace(anchor, VECDOT + anchor, 1)
    if '#include "../ggml-nf4dq.h"' not in s:
        s = s.replace('#include "common.cuh"',
                      '#include "common.cuh"\n#include "../ggml-nf4dq.h"', 1)
    open(p, "w").write(s)
    print("vecdotq.cuh: kernel added")
else:
    print("vecdotq.cuh: already present")

# ------------------------------------------------------------------- mmvq.cu
# Nine dispatch sites. The numeric ones are per-architecture tuning constants
# (rows per block); copying IQ4_NL's values is a starting point, not a tuned
# choice. Worth revisiting once it works.

p = f"{P}/ggml/src/ggml-cuda/mmvq.cu"
s = open(p).read()

# 1. remove the stage-1 fallback
s = s.replace("""    // NF4DQ has no fused vec_dot yet, so route it through
    // dequantise-then-cuBLAS instead of aborting in the dispatch switch.
    // Remove this once vec_dot_nf4dq_q8_1 is wired into vecdotq.cuh.
    if (type == GGML_TYPE_NF4DQ) {
        return false;
    }
""", "")

pairs = [
    ("        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;",
     "        case GGML_TYPE_NF4DQ:   return vec_dot_nf4dq_q8_1;\n"),
    ("        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;",
     "        case GGML_TYPE_NF4DQ:   return VDR_NF4DQ_Q8_1_MMVQ;\n"),
]
for anchor, add in pairs:
    if anchor in s:
        s = s.replace(anchor, add + anchor, 1)

# rows-per-block tables: mirror IQ4_NL's value at each site
import re
def mirror(s, val_pattern):
    out, n = [], 0
    for line in s.split("\n"):
        m = re.match(r"^(\s*)case GGML_TYPE_IQ4_NL:(\s*)return (\d+);$", line)
        if m:
            out.append(f"{m.group(1)}case GGML_TYPE_NF4DQ:{m.group(2)}return {m.group(3)};")
            n += 1
        out.append(line)
    return "\n".join(out), n

if "case GGML_TYPE_NF4DQ:   return 6;" not in s and "case GGML_TYPE_NF4DQ:" not in s.split("mul_mat_vec_q_switch_ncols_dst")[0][-2000:]:
    s, n = mirror(s, None)
    print(f"mmvq.cu: mirrored {n} rows-per-block entries")

# fall-through case lists
s = s.replace("                case GGML_TYPE_IQ4_NL:\n                case GGML_TYPE_IQ4_XS:\n                    return 8;",
              "                case GGML_TYPE_NF4DQ:\n                case GGML_TYPE_IQ4_NL:\n                case GGML_TYPE_IQ4_XS:\n                    return 8;")

# the big template dispatch
s = s.replace("""        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>""",
"""        case GGML_TYPE_NF4DQ:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NF4DQ>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>""", 1)

open(p, "w").write(s)
print("mmvq.cu patched")

# ---------------------------------------------------------------- ggml-cuda.cu
# remove the second stage-1 fallback
p = f"{P}/ggml/src/ggml-cuda/ggml-cuda.cu"
s = open(p).read()
s = s.replace("""    // NF4DQ has no fused vec_dot yet. This gate is separate from
    // ggml_cuda_should_use_mmvq and does not consult it, so it needs its own
    // exclusion or decode aborts in the mmvq dispatch switch.
    if (src0->type == GGML_TYPE_NF4DQ) {
        return false;
    }

""", "")
open(p, "w").write(s)
print("ggml-cuda.cu: fallback removed")

print("\nBoth stage-1 fallbacks removed. If decode breaks, put this back in\n"
      "ggml_cuda_should_use_mmvq to compare against the working reference:\n"
      "    if (type == GGML_TYPE_NF4DQ) return false;")
