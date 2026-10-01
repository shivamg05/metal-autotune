constexpr uint BK = 32;
constexpr uint LD = BK + 8;
threadgroup float S[4096];
threadgroup bfloat16_t* As = (threadgroup bfloat16_t*)S;
threadgroup bfloat16_t* Bs = As + 64u * LD;
const uint K1 = (uint)in2_shape[2];
const uint K2 = (uint)in0_shape[2];
const uint K = K1 + K2;
const uint N = (uint)in3_shape[0];
const uint KW = (uint)in3_shape[1];
const uint G = (uint)in4_shape[1];
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint n0 = threadgroup_position_in_grid.x * 128u;
uint m0 = threadgroup_position_in_grid.y * 64u;
uint sm = sg >> 2;
uint sn = sg & 3u;
uint qid = lane >> 2;
uint fm = (qid & 4u) + ((lane >> 1) & 3u);
uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
simdgroup_matrix<float, 8, 8> acc[4][4];
#pragma clang loop unroll(full)
for (uint i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
  for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
uint ar = tid >> 2;
uint ac = (tid & 3u) * 8u;
uint br = tid >> 1;
uint bc = (tid & 1u) * 16u;
ulong arow = (ulong)(m0 + ar);
ulong brow = (ulong)(n0 + br);
threadgroup bfloat16_t* ad = As + ar * LD + ac;
threadgroup bfloat16_t* bd = Bs + br * LD + bc;
uint4 x0 = uint4(0u), y0 = uint4(0u);
bool inA = true;
uint w0 = 0u, w1 = 0u;
float s = 0.0f, b = 0.0f;
{
  uint k = ac;
  inA = k < K1;
  if (inA) {
    x0 = *((const device uint4*)(in2 + arow * (ulong)K1 + k));
  } else {
    ulong o = arow * (ulong)K2 + (ulong)(k - K1);
    x0 = *((const device uint4*)(in0 + o));
    y0 = *((const device uint4*)(in1 + o));
  }
  uint kb = bc;
  ulong wb = brow * (ulong)KW + (ulong)(kb >> 3);
  w0 = in3[wb]; w1 = in3[wb + 1ul];
  uint g = kb >> 6;
  s = float(in4[brow * (ulong)G + g]);
  b = float(in5[brow * (ulong)G + g]);
}
for (uint k0 = 0; k0 < K; k0 += BK) {
  if (inA) {
    *((threadgroup uint4*)ad) = x0;
  } else {
    uint xs[4] = {x0.x, x0.y, x0.z, x0.w};
    uint ys[4] = {y0.x, y0.y, y0.z, y0.w};
    uint pw[4];
#pragma clang loop unroll(full)
    for (uint e = 0; e < 4; ++e) {
      float p0 = as_type<float>(xs[e] << 16) * as_type<float>(ys[e] << 16);
      float p1 = as_type<float>(xs[e] & 0xffff0000u) * as_type<float>(ys[e] & 0xffff0000u);
      pw[e] = uint(as_type<ushort>(bfloat16_t(p0))) | (uint(as_type<ushort>(bfloat16_t(p1))) << 16);
    }
    *((threadgroup uint4*)ad) = uint4(pw[0], pw[1], pw[2], pw[3]);
  }
  {
    uint q0[4], q1[4];
#pragma clang loop unroll(full)
    for (uint e = 0; e < 4; ++e) {
      bfloat16_t a0 = bfloat16_t(s * float((w0 >> (8u * e)) & 0xfu) + b);
      bfloat16_t a1 = bfloat16_t(s * float((w0 >> (8u * e + 4u)) & 0xfu) + b);
      bfloat16_t c0 = bfloat16_t(s * float((w1 >> (8u * e)) & 0xfu) + b);
      bfloat16_t c1 = bfloat16_t(s * float((w1 >> (8u * e + 4u)) & 0xfu) + b);
      q0[e] = uint(as_type<ushort>(a0)) | (uint(as_type<ushort>(a1)) << 16);
      q1[e] = uint(as_type<ushort>(c0)) | (uint(as_type<ushort>(c1)) << 16);
    }
    *((threadgroup uint4*)bd) = uint4(q0[0], q0[1], q0[2], q0[3]);
    *((threadgroup uint4*)(bd + 8)) = uint4(q1[0], q1[1], q1[2], q1[3]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint kn = k0 + BK;
  if (kn < K) {
    uint k = kn + ac;
    inA = k < K1;
    if (inA) {
      x0 = *((const device uint4*)(in2 + arow * (ulong)K1 + k));
    } else {
      ulong o = arow * (ulong)K2 + (ulong)(k - K1);
      x0 = *((const device uint4*)(in0 + o));
      y0 = *((const device uint4*)(in1 + o));
    }
    uint kb = kn + bc;
    ulong wb = brow * (ulong)KW + (ulong)(kb >> 3);
    w0 = in3[wb]; w1 = in3[wb + 1ul];
    uint g = kb >> 6;
    s = float(in4[brow * (ulong)G + g]);
    b = float(in5[brow * (ulong)G + g]);
  }
#pragma clang loop unroll(full)
  for (uint kk = 0; kk < BK; kk += 8) {
    simdgroup_matrix<float, 8, 8> a[4];
    simdgroup_matrix<float, 8, 8> w[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
      const threadgroup bfloat16_t* ap = As + (sm * 32u + i * 8u + fm) * LD + kk + fn;
      a[i].thread_elements()[0] = float(ap[0]);
      a[i].thread_elements()[1] = float(ap[1]);
    }
#pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
      const threadgroup bfloat16_t* bp = Bs + (sn * 32u + j * 8u + fn) * LD + kk + fm;
      w[j].thread_elements()[0] = float(bp[0]);
      w[j].thread_elements()[1] = float(bp[LD]);
    }
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
      for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], w[j], acc[i][j]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint cc = (lane & 3u) * 8u;
uint col = n0 + sn * 32u + cc;
float sc[8];
#pragma clang loop unroll(full)
for (uint e = 0; e < 8; ++e) sc[e] = float(in6[col + e]);
threadgroup float* C = S + sn * 1024u;
for (uint ph = 0; ph < 2u; ++ph) {
  if (sm == ph) {
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
      for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], C + (i * 8u) * 32u + j * 8u, 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint it = 0; it < 4; ++it) {
      uint rr = it * 8u + (lane >> 2);
      ulong o = (ulong)(m0 + sm * 32u + rr) * (ulong)N + (ulong)col;
      uint4 rv = *((const device uint4*)(in7 + o));
      uint rw[4] = {rv.x, rv.y, rv.z, rv.w};
      uint ow[4];
      threadgroup float* cr = C + rr * 32u + cc;
#pragma clang loop unroll(full)
      for (uint q = 0; q < 4; ++q) {
        float r0 = as_type<float>(rw[q] << 16);
        float r1 = as_type<float>(rw[q] & 0xffff0000u);
        float v0 = float(bfloat16_t(cr[2u * q]));
        float v1 = float(bfloat16_t(cr[2u * q + 1u]));
        float t0 = float(bfloat16_t(sc[2u * q] * v0));
        float t1 = float(bfloat16_t(sc[2u * q + 1u] * v1));
        bfloat16_t o0 = bfloat16_t(r0 + t0);
        bfloat16_t o1 = bfloat16_t(r1 + t1);
        ow[q] = uint(as_type<ushort>(o0)) | (uint(as_type<ushort>(o1)) << 16);
      }
      *((device uint4*)(out0 + o)) = uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
