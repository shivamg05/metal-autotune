const uint lid = thread_position_in_threadgroup.x;
const uint sg = lid / 32u;
const uint sr = sg / 2u;
const uint sc = sg % 2u;
const uint tn = threadgroup_position_in_grid.x;
const uint tm = threadgroup_position_in_grid.y;
const uint N = 256u;
const uint K = 256u;
threadgroup float As[2][32 * 32];
threadgroup float Bs[2][32 * 32];
threadgroup float Cs[32 * 32];
metal::simdgroup_float8x8 acc[2][2];
for (uint i = 0; i < 2u; ++i)
    for (uint j = 0; j < 2u; ++j)
        acc[i][j] = metal::simdgroup_float8x8(0.0f);
float4 ra[2];
float4 rb[2];
uint rr[2];
uint rc[2];
for (uint i = 0; i < 2u; ++i) {
    const uint idx = lid + i * 128u;
    rr[i] = idx / 8u;
    rc[i] = (idx % 8u) * 4u;
}
for (uint i = 0; i < 2u; ++i) {
    ra[i] = *((const device float4*)(in0 + (tm * 32u + rr[i]) * K + rc[i]));
    rb[i] = *((const device float4*)(in1 + rr[i] * N + tn * 32u + rc[i]));
}
for (uint i = 0; i < 2u; ++i) {
    *((threadgroup float4*)(As[0] + rr[i] * 32u + rc[i])) = ra[i];
    *((threadgroup float4*)(Bs[0] + rr[i] * 32u + rc[i])) = rb[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint cur = 0u;
for (uint k0 = 0; k0 < K; k0 += 32u) {
    const bool has_next = (k0 + 32u) < K;
    if (has_next) {
        const uint kn = k0 + 32u;
        for (uint i = 0; i < 2u; ++i) {
            ra[i] = *((const device float4*)(in0 + (tm * 32u + rr[i]) * K + kn + rc[i]));
            rb[i] = *((const device float4*)(in1 + (kn + rr[i]) * N + tn * 32u + rc[i]));
        }
    }
    const threadgroup float* Ac = As[cur];
    const threadgroup float* Bc = Bs[cur];
    for (uint kk = 0; kk < 32u; kk += 8u) {
        metal::simdgroup_float8x8 a[2];
        metal::simdgroup_float8x8 b[2];
        for (uint i = 0; i < 2u; ++i)
            metal::simdgroup_load(a[i], Ac + (sr * 16u + i * 8u) * 32u + kk, 32u);
        for (uint j = 0; j < 2u; ++j)
            metal::simdgroup_load(b[j], Bc + kk * 32u + sc * 16u + j * 8u, 32u);
        for (uint i = 0; i < 2u; ++i)
            for (uint j = 0; j < 2u; ++j)
                metal::simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    if (has_next) {
        const uint nxt = cur ^ 1u;
        for (uint i = 0; i < 2u; ++i) {
            *((threadgroup float4*)(As[nxt] + rr[i] * 32u + rc[i])) = ra[i];
            *((threadgroup float4*)(Bs[nxt] + rr[i] * 32u + rc[i])) = rb[i];
        }
        cur = nxt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = 0; i < 2u; ++i)
    for (uint j = 0; j < 2u; ++j)
        metal::simdgroup_store(acc[i][j], Cs + (sr * 16u + i * 8u) * 32u + sc * 16u + j * 8u, 32u);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint e = lid; e < 1024u; e += 128u) {
    const uint r = e / 32u;
    const uint c = e % 32u;
    out0[(tm * 32u + r) * N + tn * 32u + c] = (T)(Cs[e] + (float)in2[tn * 32u + c]);
}
