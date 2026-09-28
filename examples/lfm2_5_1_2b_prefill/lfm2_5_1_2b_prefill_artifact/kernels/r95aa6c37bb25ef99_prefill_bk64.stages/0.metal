const uint lid = thread_position_in_threadgroup.x;
const int K = in0_shape[2];
const int M = in0_shape[0] * in0_shape[1];
const int N = in1_shape[0];
const int KW = K / 8;
const int KG = K / 64;
const int n0 = (int)threadgroup_position_in_grid.x * 64;
const int m0 = (int)threadgroup_position_in_grid.y * 64;
threadgroup T As[64 * 72];
threadgroup T Bs[64 * 72];
if (M < 32) {
    const uint qsg = lid / 32u;
    const uint qlane = lid % 32u;
    const int nb = (int)threadgroup_position_in_grid.x * 16 + (int)qsg * 4;
    const int mend = min(m0 + 64, M);
    for (int m = m0; m < mend; ++m) {
        const device T* xp = in0 + (size_t)m * K;
        float res[4];
        UNR for (int c = 0; c < 4; ++c) res[c] = 0.0f;
        for (int kb = 0; kb < K; kb += 512) {
            const int k = kb + (int)qlane * 16;
            float xt[16];
            float sum = 0.0f;
            UNR for (int i = 0; i < 16; i += 4) {
                const float x0 = (float)xp[k + i];
                const float x1 = (float)xp[k + i + 1];
                const float x2 = (float)xp[k + i + 2];
                const float x3 = (float)xp[k + i + 3];
                float xs = (float)((T)(x0 + x1));
                xs = (float)((T)(xs + x2));
                xs = (float)((T)(xs + x3));
                sum = sum + xs;
                xt[i] = x0;
                xt[i + 1] = x1 / 16.0f;
                xt[i + 2] = x2 / 256.0f;
                xt[i + 3] = x3 / 4096.0f;
            }
            const int g = k / 64;
            UNR for (int c = 0; c < 4; ++c) {
                const int n = min(nb + c, N - 1);
                const device uint* wrow = in1 + (size_t)n * KW;
                const float s = (float)in2[n * KG + g];
                const float b = (float)in3[n * KG + g];
                const device ushort* ws = (const device ushort*)(wrow + k / 8);
                float accum = 0.0f;
                UNR for (int i = 0; i < 4; ++i) {
                    const ushort wv = ws[i];
                    float t = xt[4 * i + 1] * (float)(wv & (ushort)0x00f0);
                    t = fma(xt[4 * i], (float)(wv & (ushort)0x000f), t);
                    t = fma(xt[4 * i + 2], (float)(wv & (ushort)0x0f00), t);
                    t = fma(xt[4 * i + 3], (float)(wv & (ushort)0xf000), t);
                    accum = accum + t;
                }
                res[c] = res[c] + fma(s, accum, sum * b);
            }
        }
        UNR for (int c = 0; c < 4; ++c) {
            const float r = simd_sum(res[c]);
            const int n = nb + c;
            if (qlane == 0u && n < N) {
                const size_t idx = (size_t)m * N + n;
                const float q = (float)((T)r);
                out0[idx] = (T)((float)in4[idx] + q);
            }
        }
    }
    return;
}
const uint sg = lid / 32u;
const uint lane = lid % 32u;
const int qid = (int)lane / 4;
const int fm = (qid & 4) + (((int)lane / 2) % 4);
const int fn = (qid & 2) * 2 + ((int)lane % 2) * 2;
const int sm = ((int)sg / 2) * 32;
const int sn = ((int)sg % 2) * 32;
simdgroup_float8x8 c[4][4];
UNR for (int i = 0; i < 4; ++i) {
    UNR for (int j = 0; j < 4; ++j) {
        c[i][j] = simdgroup_float8x8(0.0f);
    }
}
const int ar = (int)lid / 2;
const int ak = ((int)lid % 2) * 32;
const int am = m0 + ar;
const bool aok = am < M;
const device T* ap = in0 + (size_t)(aok ? am : 0) * K + ak;
const int bn = (int)lid / 2;
const int bkw = ((int)lid % 2) * 4;
const device uint* wp = in1 + (size_t)(n0 + bn) * KW;
threadgroup uint4* adst = (threadgroup uint4*)(As + ar * 72 + ak);
for (int k0 = 0; k0 < K; k0 += 64) {
    if (aok) {
        const device uint4* src = (const device uint4*)(ap + k0);
        adst[0] = src[0];
        adst[1] = src[1];
        adst[2] = src[2];
        adst[3] = src[3];
    } else {
        adst[0] = uint4(0u);
        adst[1] = uint4(0u);
        adst[2] = uint4(0u);
        adst[3] = uint4(0u);
    }
    const float s = (float)in2[(n0 + bn) * KG + k0 / 64];
    const float b = (float)in3[(n0 + bn) * KG + k0 / 64];
    UNR for (int w = 0; w < 4; ++w) {
        const uint p = wp[k0 / 8 + bkw + w];
        threadgroup T* bd = Bs + bn * 72 + (bkw + w) * 8;
        UNR for (int j = 0; j < 8; ++j) {
            bd[j] = (T)(s * (float)((p >> (4u * (uint)j)) & 15u) + b);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNR for (int kk = 0; kk < 64; kk += 8) {
        simdgroup_float8x8 a[4];
        simdgroup_float8x8 bm[4];
        UNR for (int i = 0; i < 4; ++i) {
            const threadgroup T* pa = As + (sm + i * 8 + fm) * 72 + kk + fn;
            a[i].thread_elements()[0] = (float)pa[0];
            a[i].thread_elements()[1] = (float)pa[1];
        }
        UNR for (int j = 0; j < 4; ++j) {
            const threadgroup T* pb = Bs + (sn + j * 8 + fn) * 72 + kk + fm;
            bm[j].thread_elements()[0] = (float)pb[0];
            bm[j].thread_elements()[1] = (float)pb[72];
        }
        UNR for (int i = 0; i < 4; ++i) {
            UNR for (int j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(c[i][j], a[i], bm[j], c[i][j]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
UNR for (int i = 0; i < 4; ++i) {
    const int row = m0 + sm + i * 8 + fm;
    UNR for (int j = 0; j < 4; ++j) {
        const int col = n0 + sn + j * 8 + fn;
        const float v0 = c[i][j].thread_elements()[0];
        const float v1 = c[i][j].thread_elements()[1];
        if (row < M) {
            const size_t idx = (size_t)row * N + col;
            out0[idx] = (T)((float)in4[idx] + (float)((T)v0));
            out0[idx + 1] = (T)((float)in4[idx + 1] + (float)((T)v1));
        }
    }
}
