// Self-test. Runs without ggml, without a GPU, without PyTorch.
#include "ggml-nf4dq.h"
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static unsigned long long s = 88172645463325252ULL;
static double rnd(void){ s^=s<<13; s^=s>>7; s^=s<<17; return (double)(s>>11)/9007199254740992.0; }
static float gauss(void){
    double u1 = rnd(), u2 = rnd();
    if (u1 < 1e-12) u1 = 1e-12;
    return (float)(sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2));
}

int main(void){
    printf("sizeof(block_nf4dq) = %zu bytes  (%.4f bpw)\n",
           sizeof(block_nf4dq), sizeof(block_nf4dq)*8.0/QK_NF4DQ);

    const int64_t K = 256*4096;
    float *x = malloc(K*sizeof(float));

    // 1. Gaussian, the distribution NF4's codebook is fitted to.
    for (int64_t i=0;i<K;i++) x[i]=gauss();
    printf("gaussian                 : %.6f\n", nf4dq_roundtrip_error(x,K));

    // 2. Gaussian with 16-sigma outliers, matching Experiment 2's measured
    //    max/std of 15.4 to 16.8 on real transformer weights.
    for (int64_t i=0;i<K;i++) x[i]=gauss();
    for (int64_t i=0;i<K;i+=997) x[i] = (i&1?16.0f:-16.0f);
    printf("gaussian + 16-sigma tails: %.6f\n", nf4dq_roundtrip_error(x,K));

    // 3. All zeros: the degenerate case that divides by a zero scale.
    for (int64_t i=0;i<K;i++) x[i]=0.0f;
    printf("all zeros                : %.6f  (expect 0.000000)\n",
           nf4dq_roundtrip_error(x,K));

    // 4. Exact-level reconstruction: values already on the codebook should
    //    survive a round trip essentially untouched.
    for (int64_t i=0;i<K;i++) x[i]=NF4DQ_LEVELS[i%16];
    printf("on-codebook values       : %.6f  (expect ~0)\n",
           nf4dq_roundtrip_error(x,K));

    free(x);
    return 0;
}
