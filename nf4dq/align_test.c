// Confirms block_nf4dq is 4-byte aligned throughout an array, which
// ggml's get_int_b4 requires in the CUDA vec_dot path.
#include <stdio.h>
#include <stdint.h>
#include "ggml-nf4dq.h"

int main(void) {
    printf("sizeof(block_nf4dq) = %zu   (%.4f bpw)\n",
           sizeof(block_nf4dq), sizeof(block_nf4dq)*8.0/QK_NF4DQ);
    printf("alignof             = %zu\n\n", _Alignof(block_nf4dq));

    static block_nf4dq arr[8];
    int bad = 0;
    for (int i = 0; i < 8; i++) {
        int off = (int)(((uintptr_t) arr[i].qs) % 4);
        if (off) { printf("block %d: qs offset %% 4 = %d  MISALIGNED\n", i, off); bad = 1; }
    }
    printf("%s\n", bad
        ? "FAIL: get_int_b4 would fault or read garbage."
        : "PASS: 4-byte aligned throughout.");
    return bad;
}
