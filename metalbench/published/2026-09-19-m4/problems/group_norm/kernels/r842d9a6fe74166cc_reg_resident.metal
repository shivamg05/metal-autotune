const uint g = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = lid & 31u;
const uint sid = lid >> 5u;
threadgroup float red0[32];
threadgroup float red1[32];
threadgroup float bcast[2];
const device float4* src = (const device float4*)(in0 + g * 8192u);
device float4* dst = (device float4*)(out0 + g * 8192u);
float4 v0 = src[lid];
float4 v1 = src[lid + 1024u];
float s = (v0.x + v0.y) + (v0.z + v0.w) + (v1.x + v1.y) + (v1.z + v1.w);
s = simd_sum(s);
if (lane == 0u) red0[sid] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sid == 0u) {
    float t = simd_sum(red0[lane]);
    if (lane == 0u) bcast[0] = t / 8192.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float mean = bcast[0];
float4 d0 = v0 - mean;
float4 d1 = v1 - mean;
float q = (d0.x * d0.x + d0.y * d0.y) + (d0.z * d0.z + d0.w * d0.w) + (d1.x * d1.x + d1.y * d1.y) + (d1.z * d1.z + d1.w * d1.w);
q = simd_sum(q);
if (lane == 0u) red1[sid] = q;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sid == 0u) {
    float t = simd_sum(red1[lane]);
    if (lane == 0u) bcast[1] = metal::precise::rsqrt(t / 8192.0f + 1e-05f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float rstd = bcast[1];
dst[lid] = d0 * rstd;
dst[lid + 1024u] = d1 * rstd;
