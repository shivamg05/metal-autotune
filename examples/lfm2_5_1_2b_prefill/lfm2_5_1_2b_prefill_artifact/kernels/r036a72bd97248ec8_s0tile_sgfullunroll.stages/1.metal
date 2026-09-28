const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u;
const uint lane = tid % 32u;
const uint M = (uint)in0_shape[0] * (uint)in0_shape[1];
const uint K = (uint)in0_shape[2];
const uint N = (uint)in1_shape[0];
const uint KW = K / 8u;
const uint KG = K / 64u;
const uint n0 = threadgroup_position_in_grid.x * 128u;
const uint m0 = threadgroup_position_in_grid.y * 64u;
threadgroup float smem[4096];
if (M < 64u) {
  for (uint c = 0; c < 16u; ++c) {
    const uint n = n0 + sg * 16u + c;
    for (uint m = 0; m < M; ++m) {
      const device bfloat16_t* xr = in0 + (size_t)m * K;
      float result = 0.0f;
      for (uint k = 0; k < K; k += 512u) {
        const device bfloat16_t* x = xr + k + lane * 16u;
        float xt[16];
        float sum = 0.0f;
        for (uint i = 0; i < 16u; i += 4u) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          xt[i] = x[i];
          xt[i + 1] = x[i + 1] / 16.0f;
          xt[i + 2] = x[i + 2] / 256.0f;
          xt[i + 3] = x[i + 3] / 4096.0f;
        }
        const device uint16_t* ws = (const device uint16_t*)(in1 + (size_t)n * KW + (k / 8u) + lane * 2u);
        const size_t g = (size_t)n * KG + (k / 64u) + lane / 4u;
        float s = in2[g];
        float b = in3[g];
        float accum = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
          accum += (xt[4 * i] * (ws[i] & 0x000f) + xt[4 * i + 1] * (ws[i] & 0x00f0) +
                    xt[4 * i + 2] * (ws[i] & 0x0f00) + xt[4 * i + 3] * (ws[i] & 0xf000));
        }
        result += s * accum + sum * b;
      }
      result = metal::simd_sum(result);
      if (lane == 0u) {
        size_t o = (size_t)m * N + n;
        float t = float(static_cast<bfloat16_t>(result));
        out0[o] = static_cast<bfloat16_t>(float(in4[o]) + t);
      }
    }
  }
  return;
}
threadgroup bfloat16_t* As = (threadgroup bfloat16_t*)smem;
threadgroup bfloat16_t* Bs = As + 64 * 40;
const uint sm = sg / 4u;
const uint sn = sg % 4u;
metal::simdgroup_float8x8 acc[4][4];
for (uint i = 0; i < 4u; ++i) for (uint j = 0; j < 4u; ++j) acc[i][j] = metal::simdgroup_float8x8(0.0f);
const uint arow = tid / 4u;
const uint acol = (tid % 4u) * 8u;
const uint gma = m0 + arow;
const bool aval = gma < M;
const uint gmc = aval ? gma : (M - 1u);
const device uint4* aptr = (const device uint4*)(in0 + (size_t)gmc * K + acol);
const uint brow = tid / 8u;
const uint bcol = (tid % 8u) * 16u;
const device uint4* bptr = (const device uint4*)(in5 + (size_t)brow * N + n0 + bcol);
const size_t bstep = (size_t)N * 4u;
threadgroup uint4* adst = (threadgroup uint4*)(As + arow * 40u + acol);
threadgroup uint4* bdst = (threadgroup uint4*)(Bs + brow * 136u + bcol);
const threadgroup bfloat16_t* abase = As + (sm * 32u) * 40u;
const threadgroup bfloat16_t* bbase = Bs + sn * 32u;
uint4 a0 = aval ? aptr[0] : uint4(0u);
uint4 b0 = bptr[0];
uint4 b1 = bptr[1];
size_t boff = 0;
metal::simdgroup_matrix<bfloat16_t, 8, 8> fa0[4];
metal::simdgroup_matrix<bfloat16_t, 8, 8> fb0[4];
metal::simdgroup_matrix<bfloat16_t, 8, 8> fa1[4];
metal::simdgroup_matrix<bfloat16_t, 8, 8> fb1[4];
for (uint k0 = 0; k0 < K; k0 += 32u) {
  adst[0] = a0;
  bdst[0] = b0;
  bdst[1] = b1;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint kn = k0 + 32u;
  if (kn < K) {
    const uint kv = kn >> 3;
    a0 = aval ? aptr[kv] : uint4(0u);
    boff += bstep;
    b0 = bptr[boff];
    b1 = bptr[boff + 1u];
  }
  for (uint i = 0; i < 4u; ++i) metal::simdgroup_load(fa0[i], abase + i * 320u + 0u, 40);
  for (uint j = 0; j < 4u; ++j) metal::simdgroup_load(fb0[j], bbase + 0u * 136u + j * 8u, 136);
  for (uint i = 0; i < 4u; ++i) metal::simdgroup_load(fa1[i], abase + i * 320u + 8u, 40);
  for (uint j = 0; j < 4u; ++j) metal::simdgroup_load(fb1[j], bbase + 8u * 136u + j * 8u, 136);
  for (uint i = 0; i < 4u; ++i)
    for (uint j = 0; j < 4u; ++j)
      metal::simdgroup_multiply_accumulate(acc[i][j], fa0[i], fb0[j], acc[i][j]);
  for (uint i = 0; i < 4u; ++i) metal::simdgroup_load(fa0[i], abase + i * 320u + 16u, 40);
  for (uint j = 0; j < 4u; ++j) metal::simdgroup_load(fb0[j], bbase + 16u * 136u + j * 8u, 136);
  for (uint i = 0; i < 4u; ++i)
    for (uint j = 0; j < 4u; ++j)
      metal::simdgroup_multiply_accumulate(acc[i][j], fa1[i], fb1[j], acc[i][j]);
  for (uint i = 0; i < 4u; ++i) metal::simdgroup_load(fa1[i], abase + i * 320u + 24u, 40);
  for (uint j = 0; j < 4u; ++j) metal::simdgroup_load(fb1[j], bbase + 24u * 136u + j * 8u, 136);
  for (uint i = 0; i < 4u; ++i)
    for (uint j = 0; j < 4u; ++j)
      metal::simdgroup_multiply_accumulate(acc[i][j], fa0[i], fb0[j], acc[i][j]);
  for (uint i = 0; i < 4u; ++i)
    for (uint j = 0; j < 4u; ++j)
      metal::simdgroup_multiply_accumulate(acc[i][j], fa1[i], fb1[j], acc[i][j]);
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
threadgroup float* sp = smem + sg * 512u;
const uint prow = lane / 16u;
const uint pcol = (lane % 16u) * 2u;
const uint ncol = n0 + sn * 32u + pcol;
uint res[16];
for (uint h = 0; h < 2u; ++h) {
  for (uint r = 0; r < 8u; ++r) {
    uint gm = m0 + sm * 32u + h * 16u + r * 2u + prow;
    res[h * 8u + r] = (gm < M) ? *((const device uint*)(in4 + (size_t)gm * N + ncol)) : 0u;
  }
}
for (uint h = 0; h < 2u; ++h) {
  for (uint ii = 0; ii < 2u; ++ii)
    for (uint j = 0; j < 4u; ++j)
      metal::simdgroup_store(acc[h * 2u + ii][j], sp + ii * 8u * 32u + j * 8u, 32);
  metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint r = 0; r < 8u; ++r) {
    uint row = r * 2u + prow;
    uint gm = m0 + sm * 32u + h * 16u + row;
    if (gm < M) {
      bfloat16_t rv[2];
      *((thread uint*)rv) = res[h * 8u + r];
      float t0 = float(static_cast<bfloat16_t>(sp[row * 32u + pcol]));
      float t1 = float(static_cast<bfloat16_t>(sp[row * 32u + pcol + 1u]));
      bfloat16_t ov[2];
      ov[0] = static_cast<bfloat16_t>(float(rv[0]) + t0);
      ov[1] = static_cast<bfloat16_t>(float(rv[1]) + t1);
      *((device uint*)(out0 + (size_t)gm * N + ncol)) = *((thread uint*)ov);
    }
  }
  metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
}
