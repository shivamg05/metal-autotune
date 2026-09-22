const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u;
const uint tgx = threadgroup_position_in_grid.x;
const uint tgy = threadgroup_position_in_grid.y;
const uint M = (uint)in0_shape[0];
const uint K = (uint)in0_shape[1];
const uint N = (uint)in1_shape[1];
threadgroup float As[32 * 32];
threadgroup float Bs[32 * 32];
simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);
const uint sm = (sg / 2u) * 16u;
const uint sn = (sg % 2u) * 16u;
const uint lr = tid / 4u;
const uint lc = (tid % 4u) * 8u;
const uint m0 = tgy * 32u;
const uint n0 = tgx * 32u;
for (uint k0 = 0; k0 < K; k0 += 32u) {
    device const float4* ap = (device const float4*)(in0 + (m0 + lr) * K + k0 + lc);
    device const float4* bp = (device const float4*)(in1 + (k0 + lr) * N + n0 + lc);
    float4 a0 = ap[0];
    float4 a1 = ap[1];
    float4 b0 = bp[0];
    float4 b1 = bp[1];
    threadgroup float4* asd = (threadgroup float4*)(As + lr * 32u + lc);
    threadgroup float4* bsd = (threadgroup float4*)(Bs + lr * 32u + lc);
    asd[0] = a0; asd[1] = a1;
    bsd[0] = b0; bsd[1] = b1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32u; kk += 8u) {
        simdgroup_float8x8 a0m, a1m, b0m, b1m;
        simdgroup_load(a0m, As + sm * 32u + kk, 32u);
        simdgroup_load(a1m, As + (sm + 8u) * 32u + kk, 32u);
        simdgroup_load(b0m, Bs + kk * 32u + sn, 32u);
        simdgroup_load(b1m, Bs + kk * 32u + sn + 8u, 32u);
        simdgroup_multiply_accumulate(acc00, a0m, b0m, acc00);
        simdgroup_multiply_accumulate(acc01, a0m, b1m, acc01);
        simdgroup_multiply_accumulate(acc10, a1m, b0m, acc10);
        simdgroup_multiply_accumulate(acc11, a1m, b1m, acc11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(acc00, out0 + (m0 + sm) * N + n0 + sn, N);
simdgroup_store(acc01, out0 + (m0 + sm) * N + n0 + sn + 8u, N);
simdgroup_store(acc10, out0 + (m0 + sm + 8u) * N + n0 + sn, N);
simdgroup_store(acc11, out0 + (m0 + sm + 8u) * N + n0 + sn + 8u, N);
