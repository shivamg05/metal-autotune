const uint lid = thread_position_in_threadgroup.x;
const uint sgid = lid / 32u;
const uint slane = lid % 32u;
const int N = in0_shape[2];
const int row = (int)threadgroup_position_in_grid.y;
threadgroup float inv[32];
const int base = (int)lid * 4;
const device T* xp = in0 + (size_t)row * N + base;
float acc = 0.0f;
if (base + 4 <= N) {
    const float x0 = (float)xp[0];
    const float x1 = (float)xp[1];
    const float x2 = (float)xp[2];
    const float x3 = (float)xp[3];
    acc = fma(x0, x0, acc);
    acc = fma(x1, x1, acc);
    acc = fma(x2, x2, acc);
    acc = fma(x3, x3, acc);
} else {
    for (int i = 0; i < 4; ++i) {
        if (base + i < N) {
            const float xi = (float)xp[i];
            acc = fma(xi, xi, acc);
        }
    }
}
acc = simd_sum(acc);
if (sgid == 0u) inv[slane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (slane == 0u) inv[sgid] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sgid == 0u) {
    const float t = simd_sum(inv[slane]);
    if (slane == 0u) inv[0] = metal::precise::rsqrt(metal::precise::divide(t, (float)N) + 1e-05f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float sc = inv[0];
device T* op = out0 + (size_t)row * N + base;
for (int i = 0; i < 4; ++i) {
    if (base + i < N) {
        const float xn = (float)((T)(((float)xp[i]) * sc));
        op[i] = (T)(((float)in1[base + i]) * xn);
    }
}
