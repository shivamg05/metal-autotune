uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32u;
uint sg = tid / 32u;
uint K = (uint)in0_shape[2];
uint S1 = (uint)in0_shape[1];
uint M = (uint)in0_shape[0] * S1;
uint st0 = (uint)in0_strides[0];
uint st1 = (uint)in0_strides[1];
uint st2 = (uint)in0_strides[2];
uint N = (uint)in1_shape[0];
uint GXt = (N + 63u) / 64u;
uint GYt = (M + 63u) / 64u;
uint lin = threadgroup_position_in_grid.y * GXt + threadgroup_position_in_grid.x;
uint band = lin / (4u * GXt);
uint bh = metal::min(4u, GYt - band * 4u);
uint within = lin - band * 4u * GXt;
uint m0 = (band * 4u + within % bh) * 64u;
uint n0 = (within / bh) * 64u;
uint kw = K / 8u;
uint kg = K / 64u;
uint nsteps = K / 32u;
threadgroup uint4 Asv[512];
threadgroup bfloat16_t B1s[4096];
threadgroup bfloat16_t B3s[4096];
if (M < 6u) {
  uint cbase = n0 + sg * 8u;
  for (uint m = 0; m < M; ++m) {
    uint xb = (m / S1) * st0 + (m % S1) * st1;
    float r1[8];
    float r3[8];
    for (uint c = 0; c < 8u; ++c) { r1[c] = 0.0f; r3[c] = 0.0f; }
    for (uint k0 = lane * 16u; k0 < K; k0 += 512u) {
      float xt[16];
      float sum = 0.0f;
      for (uint i = 0; i < 16u; i += 4u) {
        bfloat16_t h0 = in0[xb + (k0 + i) * st2];
        bfloat16_t h1 = in0[xb + (k0 + i + 1u) * st2];
        bfloat16_t h2 = in0[xb + (k0 + i + 2u) * st2];
        bfloat16_t h3 = in0[xb + (k0 + i + 3u) * st2];
        bfloat16_t hs = h0 + h1 + h2 + h3;
        sum += float(hs);
        xt[i] = float(h0);
        xt[i + 1u] = float(h1) / 16.0f;
        xt[i + 2u] = float(h2) / 256.0f;
        xt[i + 3u] = float(h3) / 4096.0f;
      }
      uint wo = k0 / 8u;
      uint go = k0 / 64u;
      for (uint c = 0; c < 8u; ++c) {
        uint col = cbase + c;
        if (col < N) {
          r1[c] += qmv_qdot16(in1 + col * kw + wo, xt, float(in2[col * kg + go]), float(in3[col * kg + go]), sum);
          r3[c] += qmv_qdot16(in4 + col * kw + wo, xt, float(in5[col * kg + go]), float(in6[col * kg + go]), sum);
        }
      }
    }
    for (uint c = 0; c < 8u; ++c) {
      float a1 = metal::simd_sum(r1[c]);
      float a3 = metal::simd_sum(r3[c]);
      uint col = cbase + c;
      if (lane == 0u && col < N) {
        out0[m * N + col] = bfloat16_t(a1);
        out1[m * N + col] = bfloat16_t(a3);
      }
    }
  }
  return;
}
threadgroup bfloat16_t* As = (threadgroup bfloat16_t*)Asv;
const device ushort* in0u = (const device ushort*)in0;
uint sgm = sg / 2u;
uint sgn = sg % 2u;
uint qid = lane / 4u;
uint fm = (qid & 4u) + ((lane / 2u) % 4u);
uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
metal::simdgroup_float8x8 P[2][4];
metal::simdgroup_float8x8 Q[2][4];
#pragma unroll
for (uint i = 0; i < 2u; ++i) {
  #pragma unroll
  for (uint c = 0; c < 4u; ++c) {
    P[i][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    Q[i][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
}
uint bn = tid / 4u;
uint bw = tid % 4u;
uint gbn = n0 + bn;
uint ar = tid / 4u;
uint ac8 = (tid % 4u) * 8u;
uint agm = m0 + ar;
bool arow = agm < M;
uint arb = arow ? (agm / S1) * st0 + (agm % S1) * st1 : 0u;
bool vecA = (st2 == 1u) && (st1 % 8u == 0u) && ((uint)in0_shape[0] == 1u || st0 % 8u == 0u);
const threadgroup bfloat16_t* Ap = As + (sgm * 16u + fm) * 32u + fn;
const threadgroup bfloat16_t* B1p = B1s + fm * 64u + sgn * 32u + fn;
const threadgroup bfloat16_t* B3p = B3s + fm * 64u + sgn * 32u + fn;
uint dq = (bw * 8u) * 64u + bn;
uint4 va = uint4(0u);
uint q1 = 0u, q3 = 0u;
bfloat16_t s1 = bfloat16_t(0.0f), b1 = bfloat16_t(0.0f), s3 = bfloat16_t(0.0f), b3 = bfloat16_t(0.0f);
{
  va = load_a8(in0u, arb, ac8, st2, vecA, arow);
  uint kk = bw * 8u;
  if (gbn < N) {
    q1 = in1[gbn * kw + kk / 8u];
    s1 = in2[gbn * kg + kk / 64u];
    b1 = in3[gbn * kg + kk / 64u];
    q3 = in4[gbn * kw + kk / 8u];
    s3 = in5[gbn * kg + kk / 64u];
    b3 = in6[gbn * kg + kk / 64u];
  }
  Asv[tid] = va;
  mlx_deq8(q1, s1, b1, B1s + dq, 64u);
  mlx_deq8(q3, s3, b3, B3s + dq, 64u);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint t = 0; t < nsteps; ++t) {
  uint cur = t & 1u;
  bool has = (t + 1u) < nsteps;
  if (has) {
    uint k0 = (t + 1u) * 32u;
    va = load_a8(in0u, arb, k0 + ac8, st2, vecA, arow);
    uint kk = k0 + bw * 8u;
    if (gbn < N) {
      q1 = in1[gbn * kw + kk / 8u];
      s1 = in2[gbn * kg + kk / 64u];
      b1 = in3[gbn * kg + kk / 64u];
      q3 = in4[gbn * kw + kk / 8u];
      s3 = in5[gbn * kg + kk / 64u];
      b3 = in6[gbn * kg + kk / 64u];
    }
  }
  uint base = cur * 2048u;
  const threadgroup bfloat16_t* Ac = Ap + base;
  const threadgroup bfloat16_t* B1c = B1p + base;
  const threadgroup bfloat16_t* B3c = B3p + base;
  #pragma unroll
  for (uint kk = 0; kk < 32u; kk += 8u) {
    metal::simdgroup_float8x8 a[2];
    #pragma unroll
    for (uint i = 0; i < 2u; ++i) {
      uint off = i * 256u + kk;
      a[i].thread_elements()[0] = float(Ac[off]);
      a[i].thread_elements()[1] = float(Ac[off + 1u]);
    }
    #pragma unroll
    for (uint c = 0; c < 4u; ++c) {
      uint boff = kk * 64u + c * 8u;
      metal::simdgroup_float8x8 bm;
      bm.thread_elements()[0] = float(B1c[boff]);
      bm.thread_elements()[1] = float(B1c[boff + 1u]);
      metal::simdgroup_multiply_accumulate(P[0][c], a[0], bm, P[0][c]);
      metal::simdgroup_multiply_accumulate(P[1][c], a[1], bm, P[1][c]);
      bm.thread_elements()[0] = float(B3c[boff]);
      bm.thread_elements()[1] = float(B3c[boff + 1u]);
      metal::simdgroup_multiply_accumulate(Q[0][c], a[0], bm, Q[0][c]);
      metal::simdgroup_multiply_accumulate(Q[1][c], a[1], bm, Q[1][c]);
    }
  }
  if (has) {
    uint nb = cur ^ 1u;
    Asv[nb * 256u + tid] = va;
    mlx_deq8(q1, s1, b1, B1s + nb * 2048u + dq, 64u);
    mlx_deq8(q3, s3, b3, B3s + nb * 2048u + dq, 64u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma unroll
for (uint i = 0; i < 2u; ++i) {
  uint row = m0 + sgm * 16u + i * 8u + fm;
  if (row < M) {
    #pragma unroll
    for (uint c = 0; c < 4u; ++c) {
      uint col = n0 + sgn * 32u + c * 8u + fn;
      if (col < N) {
        out0[row * N + col] = bfloat16_t(P[i][c].thread_elements()[0]);
        out1[row * N + col] = bfloat16_t(Q[i][c].thread_elements()[0]);
      }
      if (col + 1u < N) {
        out0[row * N + col + 1u] = bfloat16_t(P[i][c].thread_elements()[1]);
        out1[row * N + col + 1u] = bfloat16_t(Q[i][c].thread_elements()[1]);
      }
    }
  }
}
