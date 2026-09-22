const uint lane = thread_index_in_simdgroup;
const uint pair = thread_position_in_grid.x / 32u;
const uint row = pair * 2u;
const device float4* qa = (const device float4*)(in1 + row * 64u);
const device float4* qb = (const device float4*)(in1 + (row + 1u) * 64u);
const device float4* k0 = (const device float4*)(in0 + lane * 64u);
const device float4* k1 = (const device float4*)(in0 + (lane + 32u) * 64u);
const device float4* k2 = (const device float4*)(in0 + (lane + 64u) * 64u);
const device float4* k3 = (const device float4*)(in0 + (lane + 96u) * 64u);
float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
for (uint t = 0; t < 16u; ++t) {
    float4 va = qa[t];
    float4 vb = qb[t];
    float4 v0 = k0[t];
    float4 v1 = k1[t];
    float4 v2 = k2[t];
    float4 v3 = k3[t];
    a0 += metal::dot(va, v0); b0 += metal::dot(vb, v0);
    a1 += metal::dot(va, v1); b1 += metal::dot(vb, v1);
    a2 += metal::dot(va, v2); b2 += metal::dot(vb, v2);
    a3 += metal::dot(va, v3); b3 += metal::dot(vb, v3);
}
{
    float s0 = a0 * 0.125f, s1 = a1 * 0.125f, s2 = a2 * 0.125f, s3 = a3 * 0.125f;
    float m = metal::simd_max(metal::max(metal::max(s0, s1), metal::max(s2, s3)));
    float e0 = metal::precise::exp(s0 - m);
    float e1 = metal::precise::exp(s1 - m);
    float e2 = metal::precise::exp(s2 - m);
    float e3 = metal::precise::exp(s3 - m);
    float inv = 1.0f / metal::simd_sum((e0 + e1) + (e2 + e3));
    device float* o = out0 + row * 128u + lane;
    o[0] = e0 * inv; o[32] = e1 * inv; o[64] = e2 * inv; o[96] = e3 * inv;
}
{
    float s0 = b0 * 0.125f, s1 = b1 * 0.125f, s2 = b2 * 0.125f, s3 = b3 * 0.125f;
    float m = metal::simd_max(metal::max(metal::max(s0, s1), metal::max(s2, s3)));
    float e0 = metal::precise::exp(s0 - m);
    float e1 = metal::precise::exp(s1 - m);
    float e2 = metal::precise::exp(s2 - m);
    float e3 = metal::precise::exp(s3 - m);
    float inv = 1.0f / metal::simd_sum((e0 + e1) + (e2 + e3));
    device float* o = out0 + (row + 1u) * 128u + lane;
    o[0] = e0 * inv; o[32] = e1 * inv; o[64] = e2 * inv; o[96] = e3 * inv;
}
