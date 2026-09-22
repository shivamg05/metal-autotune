const uint lid = thread_position_in_threadgroup.x;
const uint lane = lid & 31u;
const uint row = threadgroup_position_in_grid.x * 8u + (lid >> 5u);
const uint rows = (uint)in0_shape[0];
if (row >= rows) return;
const device float4* x4 = (const device float4*)(in0 + row * 1024u);
const device float4* t4 = (const device float4*)(in1 + row * 1024u);
float4 xv[8];
float m = -INFINITY;
for (uint k = 0u; k < 8u; ++k) {
    xv[k] = x4[lane + 32u * k];
    for (uint c = 0u; c < 4u; ++c) {
        float v = xv[k][c];
        m = ((m != m) || (v != v)) ? (m + v) : metal::max(m, v);
    }
}
for (uint off = 16u; off > 0u; off >>= 1u) {
    float om = simd_shuffle_xor(m, off);
    m = ((m != m) || (om != om)) ? (m + om) : metal::max(m, om);
}
float s = 0.0f;
if (!metal::isinf(m)) {
    for (uint k = 0u; k < 8u; ++k) {
        float4 e = xv[k] - m;
        s += metal::precise::exp(e.x) + metal::precise::exp(e.y) + metal::precise::exp(e.z) + metal::precise::exp(e.w);
    }
}
s = simd_sum(s);
const float lse = metal::isinf(m) ? m : (metal::precise::log(s) + m);
float acc = 0.0f;
for (uint k = 0u; k < 8u; ++k) {
    float4 tv = t4[lane + 32u * k];
    float4 p = tv * (xv[k] - lse);
    acc += (p.x + p.y) + (p.z + p.w);
}
acc = simd_sum(acc);
if (lane == 0u) out0[row] = -acc;
