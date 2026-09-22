uint lid = thread_position_in_threadgroup.x;
uint rows = (uint)in0_shape[0];
uint n = (uint)in0_shape[1];
uint sub = lid / 256u;
uint t = lid % 256u;
uint row = threadgroup_position_in_grid.x * 4u + sub;
float acc = 0.0f;
if (row < rows) {
    device const float* a = in0 + row * n;
    device const float* b = in1 + row * n;
    uint n4 = (n / 4u) * 4u;
    for (uint j = t * 4u; j < n4; j += 1024u) {
        float4 x = *((device const float4*)(a + j));
        float4 y = *((device const float4*)(b + j));
        acc += x.x * y.x;
        acc += x.y * y.y;
        acc += x.z * y.z;
        acc += x.w * y.w;
    }
    for (uint j = n4 + t; j < n; j += 256u) {
        acc += a[j] * b[j];
    }
}
acc = simd_sum(acc);
threadgroup float part[32];
uint sg = lid / 32u;
if ((lid % 32u) == 0u) part[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0u && row < rows) {
    float s = 0.0f;
    for (uint i = 0; i < 8u; ++i) s += part[sub * 8u + i];
    out0[row] = -s;
}
