const uint gid = thread_position_in_grid.x;
const uint row = gid / 32u;
const uint lane = gid % 32u;
const device float4* x4 = (const device float4*)(in0 + row * 1024u);
const device float4* w4 = (const device float4*)(in1 + row * 1024u);
float4 xs[8];
float m = -INFINITY;
float nanflag = 0.0f;
for (uint k = 0; k < 8u; ++k) {
    xs[k] = x4[lane + k * 32u];
    m = metal::max(m, metal::max(metal::max(xs[k].x, xs[k].y), metal::max(xs[k].z, xs[k].w)));
    if (metal::any(metal::isnan(xs[k]))) nanflag = 1.0f;
}
float M = metal::simd_max(m);
nanflag = metal::simd_max(nanflag);
float s = 0.0f;
for (uint k = 0; k < 8u; ++k) {
    s += metal::precise::exp(xs[k].x - M) + metal::precise::exp(xs[k].y - M) + metal::precise::exp(xs[k].z - M) + metal::precise::exp(xs[k].w - M);
}
float S = metal::simd_sum(s);
float lse;
if (nanflag > 0.0f) lse = NAN;
else if (metal::isinf(M)) lse = M;
else lse = metal::precise::log(S) + M;
float acc = 0.0f;
for (uint k = 0; k < 8u; ++k) {
    float4 wv = w4[lane + k * 32u];
    float4 d = xs[k] - lse;
    acc += wv.x * d.x + wv.y * d.y + wv.z * d.z + wv.w * d.w;
}
float A = metal::simd_sum(acc);
if (lane == 0u) out0[row] = -A;
