const uint row = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint sid = tid / 32u;
const uint lane = tid % 32u;
const uint n = 1024u;
device const float4* rp = (device const float4*)(in0 + row * n);
float4 v = rp[tid];
bool nanf = (v.x != v.x) || (v.y != v.y) || (v.z != v.z) || (v.w != v.w);
float m = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
m = metal::simd_max(m);
nanf = metal::simd_any(nanf);
threadgroup float shm[8];
threadgroup int shn[8];
threadgroup float shs[8];
if (lane == 0u) { shm[sid] = m; shn[sid] = nanf ? 1 : 0; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float M = shm[0];
int N = shn[0];
for (uint i = 1u; i < 8u; ++i) { M = metal::max(M, shm[i]); N |= shn[i]; }
float s = metal::precise::exp(v.x - M) + metal::precise::exp(v.y - M) + metal::precise::exp(v.z - M) + metal::precise::exp(v.w - M);
s = metal::simd_sum(s);
if (lane == 0u) { shs[sid] = s; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float S = 0.0f;
for (uint i = 0u; i < 8u; ++i) S += shs[i];
float lse;
if (N) lse = NAN;
else lse = metal::isinf(M) ? M : (metal::precise::log(S) + M);
device float4* op = (device float4*)(out0 + row * n);
op[tid] = float4(v.x - lse, v.y - lse, v.z - lse, v.w - lse);
