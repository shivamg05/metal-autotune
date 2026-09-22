const uint tg = threadgroup_position_in_grid.x;
const uint t = thread_position_in_threadgroup.x;
const uint lane = t & 31u;
const uint sg = t >> 5;
const uint row = tg * 2u + sg;
threadgroup float p[2][128];

const device float4* src = (const device float4*)(in0 + row * 128u);
float4 s = src[lane] * 0.125f;
float m = metal::max(metal::max(s.x, s.y), metal::max(s.z, s.w));
m = simd_max(m);
float4 e = float4(metal::precise::exp(s.x - m), metal::precise::exp(s.y - m), metal::precise::exp(s.z - m), metal::precise::exp(s.w - m));
float l = simd_sum((e.x + e.y) + (e.z + e.w));
float inv = 1.0f / l;
((threadgroup float4*)p[sg])[lane] = e * inv;
threadgroup_barrier(mem_flags::mem_threadgroup);

const device float2* vb = (const device float2*)in1;
float2 acc = float2(0.0f);
const threadgroup float* pr = p[sg];
for (uint k = 0; k < 128u; ++k) {
    acc += pr[k] * vb[k * 32u + lane];
}
((device float2*)(out0 + row * 64u))[lane] = acc;
