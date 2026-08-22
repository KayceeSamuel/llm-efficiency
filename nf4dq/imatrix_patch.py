"""
Add importance-matrix support to the NF4DQ encoder.

WHAT IS CURRENTLY WRONG

quantize_row_nf4dq_ref picks each sub-block's scale index by nearest ratio:

    ratio = absmax[s] / d
    si    = nearest(ratio, NF4DQ_SCALE_LEVELS)

That minimises the error in the SCALE, which is not the quantity anyone
cares about. What matters is the error in the reconstructed WEIGHTS after
rounding them to the codebook at that scale. A slightly "wrong" scale can
round the weights better, and nearest-ratio never considers this.

Every K-quant does consider it. ggml's make_qx_quants starts from absmax,
derives the least-squares optimal scale, then searches +/-9 perturbations,
scoring each by importance-weighted error.

WHAT THIS DOES INSTEAD

NF4DQ's sub-block scale is not a free parameter: it must be one of the 16
values in NF4DQ_SCALE_LEVELS. So rather than searching, try all 16 and keep
whichever minimises

    sum_j  w[j] * (x[j] - I8[idx_j] * d * SCALE[si])^2

That is exact optimisation over the available choices, not a heuristic. It
costs 16 passes over each sub-block at quantisation time and nothing at
inference.

Two consequences worth separating:

  1. This improves the encoder even with NO imatrix, because uniform weights
     still make error-based selection better than ratio-based selection.
     That arm is worth measuring on its own, since it isolates how much of
     the gain is the search and how much is the calibration data.

  2. With an imatrix, w[j] comes from the calibration statistics and the
     selection becomes sensitive to which input channels actually matter.

CALIBRATION DATA, AND WHAT WOULD BE CHEATING

Generating an imatrix from wikitext and then reporting wikitext perplexity
is tuning on the test set. The improvement would not transfer and the number
would be meaningless. Use a different corpus: wikitext TRAIN rather than
test, or a general corpus, or domain data matching the intended workload.

Run in Colab. Edits /content/llama.cpp in place.
"""

P = "/content/llama.cpp"

# ------------------------------------------------------------- ggml-nf4dq.c

NEW_SIG = r'''
// Weighted squared error of one sub-block quantised at a given scale index.
// w may be NULL, in which case every weight counts equally, which is still
// better than nearest-ratio selection because it scores the reconstruction
// rather than the scale.
static float nf4dq_subblock_err(const float * x, const float * w,
                                float d, uint8_t si) {
    const float scale = d * 127.0f * NF4DQ_SCALE_LEVELS[si];
    if (scale <= 0.0f) {
        float e = 0.0f;
        for (int j = 0; j < NF4DQ_SUB; j++) {
            const float wj = w ? w[j] : 1.0f;
            e += wj * x[j] * x[j];
        }
        return e;
    }
    const float inv = 1.0f / scale;
    float err = 0.0f;
    for (int j = 0; j < NF4DQ_SUB; j++) {
        const uint8_t idx = nf4dq_nearest(x[j] * inv);
        const float   rec = (float) NF4DQ_I8[idx] * d * NF4DQ_SCALE_LEVELS[si];
        const float   dj  = x[j] - rec;
        const float   wj  = w ? w[j] : 1.0f;
        err += wj * dj * dj;
    }
    return err;
}

// Best scale index for a sub-block: exhaustive over all 16, scored by
// weighted reconstruction error. NF4DQ's scale is not a free parameter, so
// this is exact rather than the perturbation search K-quants need.
static uint8_t nf4dq_best_scale(const float * x, const float * w, float d) {
    uint8_t best_si  = 0;
    float   best_err = -1.0f;
    for (uint8_t si = 0; si < 16; si++) {
        const float e = nf4dq_subblock_err(x, w, d, si);
        if (best_err < 0.0f || e < best_err) { best_err = e; best_si = si; }
    }
    return best_si;
}

'''

p = f"{P}/ggml/src/ggml-nf4dq.c"
s = open(p).read()

if "nf4dq_best_scale" not in s:
    s = s.replace("// ---------------------------------------------------------------- quantise",
                  NEW_SIG + "// ---------------------------------------------------------------- quantise", 1)

    # widen the signature to carry the imatrix
    s = s.replace(
        "void quantize_row_nf4dq_ref(const float * NF4DQ_RESTRICT x, block_nf4dq * NF4DQ_RESTRICT y,\n"
        "                            int64_t k) {",
        "void quantize_row_nf4dq_imatrix(const float * NF4DQ_RESTRICT x,\n"
        "                                block_nf4dq * NF4DQ_RESTRICT y,\n"
        "                                int64_t k, const float * qw) {")
    if "quantize_row_nf4dq_imatrix" not in s:   # spacing may differ
        import re
        s = re.sub(r"void quantize_row_nf4dq_ref\s*\([^)]*\)\s*\{",
                   "void quantize_row_nf4dq_imatrix(const float * NF4DQ_RESTRICT x,\n"
                   "                                block_nf4dq * NF4DQ_RESTRICT y,\n"
                   "                                int64_t k, const float * qw) {",
                   s, count=1)

    # replace nearest-ratio selection with error-based selection
    OLD = """        for (int s = 0; s < NF4DQ_NSUB; s++) {
            const float ratio = (d_eff > 0.0f)
                                ? (absmax[s] / (d_eff * 127.0f)) : 0.0f;
            const uint8_t si = nf4dq_nearest_scale(ratio);"""
    NEW = """        for (int s = 0; s < NF4DQ_NSUB; s++) {
            // Pick the scale index that minimises weighted reconstruction
            // error, not the one closest to the true ratio. See the note at
            // the top of this file: the two are different objectives and only
            // the first is the one that matters.
            const float * xs = xb + s * NF4DQ_SUB;
            const float * ws = qw ? (qw + i * QK_NF4DQ + s * NF4DQ_SUB) : NULL;
            const uint8_t si = nf4dq_best_scale(xs, ws, d_eff);"""
    if OLD in s:
        s = s.replace(OLD, NEW, 1); print("scale selection replaced")
    else:
        print("!! scale-selection pattern not found, check ggml-nf4dq.c")

    # keep the old entry point as a thin wrapper so nothing else breaks
    s += r'''

// Backwards-compatible entry point. ggml's type_traits.from_float_ref has a
// fixed signature with no imatrix, and is used for paths that do not have
// one, so it forwards with qw = NULL.
void quantize_row_nf4dq_ref(const float * NF4DQ_RESTRICT x,
                            block_nf4dq * NF4DQ_RESTRICT y, int64_t k) {
    quantize_row_nf4dq_imatrix(x, y, k, NULL);
}
'''
    open(p, "w").write(s)
    print("ggml-nf4dq.c patched")
else:
    print("ggml-nf4dq.c already patched")

# --- row wrapper takes and forwards the imatrix
s = open(p).read()
s = s.replace("""size_t quantize_nf4dq(const float * NF4DQ_RESTRICT src, void * NF4DQ_RESTRICT dst,
                      int64_t nrow, int64_t n_per_row) {""",
"""size_t quantize_nf4dq(const float * NF4DQ_RESTRICT src, void * NF4DQ_RESTRICT dst,
                      int64_t nrow, int64_t n_per_row, const float * imatrix) {""")
s = s.replace("        quantize_row_nf4dq_ref(src, (block_nf4dq *) qrow, n_per_row);",
              "        // the imatrix is per-column, so every row uses the same weights\n"
              "        quantize_row_nf4dq_imatrix(src, (block_nf4dq *) qrow, n_per_row, imatrix);")
open(p, "w").write(s)

# ------------------------------------------------------------- header
p = f"{P}/ggml/src/ggml-nf4dq.h"
s = open(p).read()
if "quantize_row_nf4dq_imatrix" not in s:
    s = s.replace("GGML_API void quantize_row_nf4dq_ref",
"""GGML_API void quantize_row_nf4dq_imatrix(const float * NF4DQ_RESTRICT x,
                                         block_nf4dq * NF4DQ_RESTRICT y,
                                         int64_t k, const float * qw);

GGML_API void quantize_row_nf4dq_ref""", 1)
s = s.replace("""GGML_API size_t quantize_nf4dq(const float * NF4DQ_RESTRICT src, void * NF4DQ_RESTRICT dst,
                      int64_t nrow, int64_t n_per_row);""",
"""GGML_API size_t quantize_nf4dq(const float * NF4DQ_RESTRICT src, void * NF4DQ_RESTRICT dst,
                      int64_t nrow, int64_t n_per_row, const float * imatrix);""")
open(p, "w").write(s)
print("header patched")

# ------------------------------------------------------------- ggml.c
p = f"{P}/ggml/src/ggml.c"
s = open(p).read()
s = s.replace("result = quantize_nf4dq(src + start, (char *) dst + start_row * row_size, nrows, n_per_row);",
              "result = quantize_nf4dq(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);")
open(p, "w").write(s)
print("ggml.c dispatch passes imatrix")

print("\nRebuild, then quantise with and without --imatrix to separate the two "
      "effects:\n"
      "  A: no imatrix  -> measures the error-based scale search alone\n"
      "  B: --imatrix   -> adds the calibration data on top")
