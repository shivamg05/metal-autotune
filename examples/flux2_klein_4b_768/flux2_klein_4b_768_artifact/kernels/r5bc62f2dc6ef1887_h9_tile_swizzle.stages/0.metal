uint row = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
uint lane = tid & 31u;
uint sg = tid >> 5;
threadgroup float red[8];
const device bfloat16_t* x = in0 + row * 3072u;
float v[12];
float s = 0.0f;
for (uint i = 0; i < 12u; ++i) { v[i] = float(x[tid + i * 256u]); s += v[i]; }
s = simd_sum(s);
if (lane == 0u) red[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = 0.0f;
for (uint j = 0; j < 8u; ++j) tot += red[j];
float mean = tot / 3072.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
float q = 0.0f;
for (uint i = 0; i < 12u; ++i) { float d = v[i] - mean; q += d * d; }
q = simd_sum(q);
if (lane == 0u) red[sg] = q;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tq = 0.0f;
for (uint j = 0; j < 8u; ++j) tq += red[j];
float inv = metal::precise::rsqrt(tq / 3072.0f + 1e-6f);
for (uint i = 0; i < 12u; ++i) {
  uint c = tid + i * 256u;
  bfloat16_t ln = bfloat16_t((v[i] - mean) * inv);
  bfloat16_t a = bfloat16_t(1.0f + float(in1[c]));
  bfloat16_t p = bfloat16_t(float(a) * float(ln));
  out0[row * 3072u + c] = bfloat16_t(float(p) + float(in2[c]));
}
