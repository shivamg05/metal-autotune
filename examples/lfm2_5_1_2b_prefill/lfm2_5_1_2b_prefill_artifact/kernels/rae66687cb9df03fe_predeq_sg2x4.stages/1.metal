uint M = (uint)(in0_shape[0] * in0_shape[1]);
uint K = (uint)in0_shape[2];
uint third = (uint)in1_shape[0] / 3u;
uint tid = thread_position_in_threadgroup.x;
uint sg = tid >> 5;
uint lane = tid & 31u;
uint tgx = threadgroup_position_in_grid.x;
uint tgy = threadgroup_position_in_grid.y;
uint tx = tgx >> 3;
uint ty = (tgy << 3) + (tgx & 7u);
uint j0 = tx * 32u;
threadgroup T As[64 * 72];
threadgroup T Bs[96 * 72];
if (M < 6u) {
  if (ty != 0u) return;
  const uint in_vec_size_w = K * 4u / 8u;
  const uint in_vec_size_g = K / 64u;
  const device uint8_t* wbase = (const device uint8_t*)in2;
  for (uint m = 0; m < M; ++m) {
    float result[12];
    for (uint q = 0; q < 12u; ++q) result[q] = 0.0f;
    const device T* x = in0 + m * K + lane * 16u;
    for (uint k = 0; k < K; k += 512u) {
      float x_thread[16];
      float sum = 0;
      for (int i = 0; i < 16; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      for (uint c = 0; c < 3u; ++c) {
        for (uint r = 0; r < 4u; ++r) {
          uint n = c * third + j0 + sg * 4u + r;
          const device uint16_t* ws = (const device uint16_t*)(wbase + n * in_vec_size_w + (k / 512u) * 256u + lane * 8u);
          uint gidx = n * in_vec_size_g + (k / 64u) + lane / 4u;
          float s = in3[gidx];
          float b = in4[gidx];
          float accum = 0;
          for (int i = 0; i < 4; i++) {
            accum += (x_thread[4 * i] * (ws[i] & 0x000f) + x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                      x_thread[4 * i + 2] * (ws[i] & 0x0f00) + x_thread[4 * i + 3] * (ws[i] & 0xf000));
          }
          result[c * 4u + r] += s * accum + sum * b;
        }
      }
      x += 512;
    }
    for (uint q = 0; q < 12u; ++q) result[q] = simd_sum(result[q]);
    if (lane == 0u) {
      for (uint r = 0; r < 4u; ++r) {
        uint o = m * third + j0 + sg * 4u + r;
        out0[o] = static_cast<T>(result[4u + r]);
        T gb = static_cast<T>(result[r]);
        T vb = static_cast<T>(result[8u + r]);
        out1[o] = T(float(gb) * float(vb));
      }
    }
  }
  return;
}
uint m0 = ty * 64u;
if (m0 >= M) return;
uint sgm = sg >> 2;
uint sgn = sg & 3u;
simdgroup_float8x8 acc[3][4];
for (uint c = 0; c < 3u; ++c) for (uint i = 0; i < 4u; ++i) acc[c][i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
uint qid = lane >> 2;
uint fm = (qid & 4u) + ((lane >> 1) & 3u);
uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
uint ar = tid >> 2;
uint av = tid & 3u;
uint agm = m0 + ar;
bool arow_ok = agm < M;
threadgroup uint4* adst = (threadgroup uint4*)(As + ar * 72u);
const device uint4* asrc = (const device uint4*)(in0 + (ulong)(arow_ok ? agm : 0u) * K);
uint brr = tid >> 3;
uint bv = tid & 7u;
const device uint4* bp0 = (const device uint4*)(in1 + (ulong)(j0 + brr) * K) + bv;
const device uint4* bp1 = (const device uint4*)(in1 + (ulong)(third + j0 + brr) * K) + bv;
const device uint4* bp2 = (const device uint4*)(in1 + (ulong)(2u * third + j0 + brr) * K) + bv;
threadgroup uint4* bd0 = (threadgroup uint4*)(Bs + brr * 72u) + bv;
threadgroup uint4* bd1 = (threadgroup uint4*)(Bs + (32u + brr) * 72u) + bv;
threadgroup uint4* bd2 = (threadgroup uint4*)(Bs + (64u + brr) * 72u) + bv;
uint4 ra0 = uint4(0u);
uint4 ra1 = uint4(0u);
if (arow_ok) {
  ra0 = asrc[av];
  ra1 = asrc[av + 4u];
}
uint4 rb0 = bp0[0];
uint4 rb1 = bp1[0];
uint4 rb2 = bp2[0];
uint bcol = sgn * 8u + fn;
uint arow = sgm * 32u + fm;
for (uint k0 = 0; k0 < K; k0 += 64u) {
  adst[av] = ra0;
  adst[av + 4u] = ra1;
  bd0[0] = rb0;
  bd1[0] = rb1;
  bd2[0] = rb2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint kn = k0 + 64u;
  if (kn < K) {
    uint ko = kn >> 3;
    if (arow_ok) {
      ra0 = asrc[ko + av];
      ra1 = asrc[ko + av + 4u];
    }
    rb0 = bp0[ko];
    rb1 = bp1[ko];
    rb2 = bp2[ko];
  }
  for (uint kk = 0; kk < 8u; ++kk) {
    uint kb = kk * 8u + fm;
    simdgroup_float8x8 b0;
    simdgroup_float8x8 b1;
    simdgroup_float8x8 b2;
    {
      thread float2& e0 = *(thread float2*)&(b0.thread_elements());
      uint nb = bcol;
      e0[0] = float(Bs[nb * 72u + kb]);
      e0[1] = float(Bs[(nb + 1u) * 72u + kb]);
      thread float2& e1 = *(thread float2*)&(b1.thread_elements());
      nb = 32u + bcol;
      e1[0] = float(Bs[nb * 72u + kb]);
      e1[1] = float(Bs[(nb + 1u) * 72u + kb]);
      thread float2& e2 = *(thread float2*)&(b2.thread_elements());
      nb = 64u + bcol;
      e2[0] = float(Bs[nb * 72u + kb]);
      e2[1] = float(Bs[(nb + 1u) * 72u + kb]);
    }
    for (uint i = 0; i < 4u; ++i) {
      simdgroup_float8x8 a;
      thread float2& ae = *(thread float2*)&(a.thread_elements());
      uint base = (arow + i * 8u) * 72u + kk * 8u + fn;
      ae[0] = float(As[base]);
      ae[1] = float(As[base + 1u]);
      simdgroup_multiply_accumulate(acc[0][i], a, b0, acc[0][i]);
      simdgroup_multiply_accumulate(acc[1][i], a, b1, acc[1][i]);
      simdgroup_multiply_accumulate(acc[2][i], a, b2, acc[2][i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint col = j0 + bcol;
for (uint i = 0; i < 4u; ++i) {
  uint row = m0 + arow + i * 8u;
  if (row >= M) continue;
  auto g = acc[0][i].thread_elements();
  auto h = acc[1][i].thread_elements();
  auto v = acc[2][i].thread_elements();
  uint o = row * third + col;
  for (uint e = 0; e < 2u; ++e) {
    out0[o + e] = T(h[e]);
    T gb = T(g[e]);
    T vb = T(v[e]);
    out1[o + e] = T(float(gb) * float(vb));
  }
}
