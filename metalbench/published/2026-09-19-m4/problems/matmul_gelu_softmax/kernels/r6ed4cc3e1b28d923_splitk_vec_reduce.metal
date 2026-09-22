threadgroup float red[4u * 16u * 64u];
const uint lid = thread_index_in_threadgroup;
const uint sg = lid / 32u;
const uint K = (uint)in0_shape[1];
const uint N = (uint)in1_shape[1];
const uint row0 = threadgroup_position_in_grid.y * 16u;
const uint col0 = threadgroup_position_in_grid.x * 64u;
const uint kq = K / 4u;
const uint kbeg = sg * kq;
const uint kend = kbeg + kq;
metal::simdgroup_float8x8 acc[2][8];
for (uint i = 0; i < 2u; ++i) {
    for (uint j = 0; j < 8u; ++j) acc[i][j] = metal::simdgroup_float8x8(0.0f);
}
for (uint k = kbeg; k < kend; k += 8u) {
    metal::simdgroup_float8x8 a0, a1;
    metal::simdgroup_load(a0, in0 + (row0 + 0u) * K + k, (ulong)K);
    metal::simdgroup_load(a1, in0 + (row0 + 8u) * K + k, (ulong)K);
    for (uint j = 0; j < 8u; ++j) {
        metal::simdgroup_float8x8 b;
        metal::simdgroup_load(b, in1 + k * N + col0 + j * 8u, (ulong)N);
        metal::simdgroup_multiply_accumulate(acc[0][j], a0, b, acc[0][j]);
        metal::simdgroup_multiply_accumulate(acc[1][j], a1, b, acc[1][j]);
    }
}
threadgroup float* mine = red + sg * 1024u;
for (uint i = 0; i < 2u; ++i) {
    for (uint j = 0; j < 8u; ++j) {
        metal::simdgroup_store(acc[i][j], mine + (i * 8u) * 64u + j * 8u, 64u);
    }
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (uint q = 0; q < 2u; ++q) {
    const uint e = (q * 128u + lid) * 4u;
    float4 s = *((threadgroup float4*)(red + e));
    s += *((threadgroup float4*)(red + 1024u + e));
    s += *((threadgroup float4*)(red + 2048u + e));
    s += *((threadgroup float4*)(red + 3072u + e));
    const uint r = e / 64u;
    const uint c = e % 64u;
    *((device float4*)(out0 + (row0 + r) * N + col0 + c)) = s;
}
