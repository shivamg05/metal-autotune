uint tid = thread_position_in_threadgroup.x;
uint sg = tid >> 5;
const uint NT = 216u;
const uint GM = 4u;
uint lin = threadgroup_position_in_grid.y * NT + threadgroup_position_in_grid.x;
uint grp = lin / (GM * NT);
uint within = lin - grp * (GM * NT);
uint tm = grp * GM + (within % GM);
uint tn = within / GM;
const uint K = 3072u;
const uint KP = 384u;
const uint G = 48u;
uint m0 = tm * 64u;
uint n0 = tn * 128u;
threadgroup float smem[64 * 68];
threadgroup bfloat* As = (threadgroup bfloat*)smem;
threadgroup bfloat* Bs = As + 64 * 40;
uint sm = sg >> 2;
uint sn = sg & 3u;
simdgroup_matrix<float, 8, 8> acc[4][4];
for (uint i = 0; i < 4u; ++i) for (uint j = 0; j < 4u; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
uint ar = tid >> 2;
uint ac = (tid & 3u) * 8u;
const device bfloat16_t* arow = in0 + (m0 + ar) * K + ac;
threadgroup uint4* adst = (threadgroup uint4*)(As + ar * 40u + ac);
for (uint k0 = 0; k0 < K; k0 += 32u) {
  *adst = *((const device uint4*)(arow + k0));
  uint g = k0 >> 6;
  for (uint j = 0; j < 2u; ++j) {
    uint idx = tid + j * 256u;
    uint nr = idx >> 2;
    uint w = idx & 3u;
    uint n = n0 + nr;
    uint p = in1[n * KP + (k0 >> 3) + w];
    float sc = float(in2[n * G + g]);
    float bi = float(in3[n * G + g]);
    uint pk[4];
    for (uint t = 0; t < 4u; ++t) {
      bfloat lo = static_cast<bfloat>(sc * float((p >> (8u * t)) & 15u) + bi);
      bfloat hi = static_cast<bfloat>(sc * float((p >> (8u * t + 4u)) & 15u) + bi);
      pk[t] = uint(as_type<ushort>(lo)) | (uint(as_type<ushort>(hi)) << 16);
    }
    *((threadgroup uint4*)(Bs + nr * 40u + w * 8u)) = uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint kk = 0; kk < 32u; kk += 8u) {
    simdgroup_matrix<bfloat, 8, 8> a[4];
    simdgroup_matrix<bfloat, 8, 8> b[4];
    for (uint i = 0; i < 4u; ++i) simdgroup_load(a[i], As + (sm * 32u + i * 8u) * 40u + kk, 40);
    for (uint j = 0; j < 4u; ++j) simdgroup_load(b[j], Bs + (sn * 32u + j * 8u) * 40u + kk, 40, ulong2(0, 0), true);
    for (uint i = 0; i < 4u; ++i) for (uint j = 0; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint pass = 0; pass < 2u; ++pass) {
  if ((sn >> 1) == pass) {
    for (uint i = 0; i < 4u; ++i) for (uint j = 0; j < 4u; ++j) simdgroup_store(acc[i][j], smem + (sm * 32u + i * 8u) * 68u + (sn & 1u) * 32u + j * 8u, 68);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < 16u; ++i) {
    uint idx = tid + i * 256u;
    uint r = idx >> 6;
    uint c = idx & 63u;
    bfloat16_t val = bfloat16_t(smem[r * 68u + c]);
    uint n = n0 + pass * 64u + c;
    uint m = m0 + r;
    if (n0 >= 9216u) {
      out0[m * 18432u + (n - 9216u)] = val;
    } else if (n0 >= 6144u) {
      uint nn = n - 6144u;
      out1[((nn >> 7) * 2816u + m) * 128u + (nn & 127u)] = val;
    } else {
      out2[m * 6144u + n] = val;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
