"""
Switch NF4DQ to ggml's nibble packing convention.

THE BUG THIS FIXES

NF4DQ inherited its packing from harness/qembed.py, which interleaves: byte k
holds element 2k in the low nibble and element 2k+1 in the high nibble.

ggml packs differently. From dequantize_iq4_nl in dequantize.cuh:

    y[j+ 0] = kvalues[q4[j] & 0xf];
    y[j+16] = kvalues[q4[j] >>  4];

So within a 32-element group, byte k holds element k in the low nibble and
element k+16 in the high nibble. Every ggml 4-bit type does this, and
get_int_from_table_16 assumes it: it splits a 32-bit word into even-indexed
and odd-indexed nibbles and the caller pairs them with activation ints 16
elements apart.

Symptom of the mismatch: the dequant path produced coherent text (our own
kernel used our own convention consistently) while the fused vec_dot produced
fluent garbage at full speed. Nothing upstream was wrong.

WHAT CHANGES

Four places, all the same one-line mapping:
  old:  byte = pos >> 1,          nibble = pos & 1
  new:  byte = s*16 + (e & 15),   nibble = e >= 16
where pos = s*NF4DQ_SUB + e.

This changes the file format, so any existing NF4DQ GGUF must be regenerated.
There are no NF4DQ files in the wild, so no compatibility concern; if that
ever changes, the ggml type number would need bumping rather than reusing 43.

Run in Colab. Edits /content/llama.cpp in place.
"""

P = "/content/llama.cpp"

# ------------------------------------------------------------- ggml-nf4dq.c
p = f"{P}/ggml/src/ggml-nf4dq.c"
s = open(p).read()

# --- encoder
OLD_ENC = """            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int     pos = s * NF4DQ_SUB + j;
                const float   v   = (inv > 0.0f) ? xb[pos] * inv : 0.0f;
                const uint8_t idx = nf4dq_nearest(v);

                if ((pos & 1) == 0) yb->qs[pos >> 1] |= idx;
                else                yb->qs[pos >> 1] |= (uint8_t)(idx << 4);
            }"""
NEW_ENC = """            for (int j = 0; j < NF4DQ_SUB; j++) {
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
            }"""
if OLD_ENC in s:
    s = s.replace(OLD_ENC, NEW_ENC, 1); print("encoder repacked")
else:
    print("!! encoder pattern not found")

# --- dequantiser
OLD_DEQ = """            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int pos = s * NF4DQ_SUB + j;
                const uint8_t byte = xb->qs[pos >> 1];
                const uint8_t idx  = ((pos & 1) == 0) ? (byte & 0x0F)
                                                      : (byte >> 4);"""
NEW_DEQ = """            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int pos = s * NF4DQ_SUB + j;
                const uint8_t byte = xb->qs[s * (NF4DQ_SUB / 2) + (j & 15)];
                const uint8_t idx  = (j < 16) ? (byte & 0x0F) : (byte >> 4);"""
if OLD_DEQ in s:
    s = s.replace(OLD_DEQ, NEW_DEQ, 1); print("dequantiser repacked")
else:
    print("!! dequantiser pattern not found")

# --- CPU vec_dot
OLD_VD = """            for (int j = 0; j < NF4DQ_SUB; j++) {
                const int     pos  = s_ * NF4DQ_SUB + j;
                const uint8_t byte = x[i].qs[pos >> 1];
                const uint8_t idx  = (pos & 1) ? (byte >> 4) : (byte & 0x0F);
                sumi += (int) NF4DQ_I8[idx] * (int) yb->qs[j];
            }"""
NEW_VD = """            for (int j = 0; j < NF4DQ_SUB; j++) {
                const uint8_t byte = x[i].qs[s_ * (NF4DQ_SUB / 2) + (j & 15)];
                const uint8_t idx  = (j < 16) ? (byte & 0x0F) : (byte >> 4);
                sumi += (int) NF4DQ_I8[idx] * (int) yb->qs[j];
            }"""
if OLD_VD in s:
    s = s.replace(OLD_VD, NEW_VD, 1); print("cpu vec_dot repacked")
else:
    print("!! cpu vec_dot pattern not found")

open(p, "w").write(s)

# ------------------------------------------------------------- convert.cu
p = f"{P}/ggml/src/ggml-cuda/convert.cu"
s = open(p).read()
OLD_CU = """    for (int j = 0; j < NF4DQ_SUB; ++j) {
        const int     pos  = s * NF4DQ_SUB + j;
        const uint8_t byte = x->qs[pos >> 1];
        const uint8_t idx  = (pos & 1) ? (byte >> 4) : (byte & 0x0F);
        y[pos] = (dst_t) ((float) knf4dq_i8[idx] * scale);
    }"""
NEW_CU = """    for (int j = 0; j < NF4DQ_SUB; ++j) {
        // ggml packing: byte k holds element k (low) and k+16 (high)
        const uint8_t byte = x->qs[s * (NF4DQ_SUB / 2) + (j & 15)];
        const uint8_t idx  = (j < 16) ? (byte & 0x0F) : (byte >> 4);
        y[s * NF4DQ_SUB + j] = (dst_t) ((float) knf4dq_i8[idx] * scale);
    }"""
if OLD_CU in s:
    s = s.replace(OLD_CU, NEW_CU, 1)
    open(p, "w").write(s); print("cuda dequant repacked")
else:
    print("!! cuda dequant pattern not found")

print("\nFile format changed. The existing GGUF must be regenerated:")
print("  llama-quantize --pure <BF16.gguf> <out.gguf> NF4DQ")
