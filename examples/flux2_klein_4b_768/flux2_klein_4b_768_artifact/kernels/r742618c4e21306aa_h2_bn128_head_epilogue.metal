const uint K = (uint)in0_shape[2];
const uint N = (uint)in1_shape[0];
const uint Mm = (uint)in0_shape[1];
const uint Ma = (uint)in8_shape[1];
const uint KW = K / 8u;
const uint KG = K / 64u;
const uint S = Mm + Ma;
const uint H = N / 128u;
const uint mtm = (Mm + 63u) / 64u;
const uint mta = (Ma + 63u) / 64u;
const uint tmain = mtm * H;
const uint tadd = mta * H;
const uint tq = tmain + 3u * tadd;
uint t = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint sg = tid / 32u;
uint lane = tid % 32u;
threadgroup float smem[3840];
if (t >= tq) {
  uint total = 2u * H * Mm;
  uint base = (t - tq) * 128u + sg * 16u;
  uint dbase = lane * 4u;
  for (uint r = 0; r < 16u; ++r) {
    uint idx = base + r;
    if (idx >= total) break;
    uint which = idx / (H * Mm);
    uint rem = idx - which * (H * Mm);
    uint h = rem / Mm;
    uint tok = rem - h * Mm;
    uint s = Ma + tok;
    const device bfloat16_t* src = ((which == 0u) ? in4 : in5) + (ulong)tok * N + h * 128u + dbase;
    const device bfloat16_t* wsrc = ((which == 0u) ? in6 : in7) + dbase;
    float4 x = float4(float(src[0]), float(src[1]), float(src[2]), float(src[3]));
    float4 w = float4(float(wsrc[0]), float(wsrc[1]), float(wsrc[2]), float(wsrc[3]));
    uint fi = s * 64u + lane * 2u;
    float2 cs = float2(in20[fi], in20[fi + 1u]);
    float2 sn = float2(in21[fi], in21[fi + 1u]);
    device bfloat16_t* dst = ((which == 0u) ? out1 : out2) + (ulong)(h * S + s) * 128u + dbase;
    nr_apply(x, w, cs, sn, dst);
  }
  return;
}
uint job; uint lt;
if (t < tmain) { job = 0u; lt = t; } else { uint u = t - tmain; job = 1u + u / tadd; lt = u % tadd; }
const uint M = (job == 0u) ? Mm : Ma;
const uint mt = (job == 0u) ? mtm : mta;
const uint GM = 4u;
uint gw = GM * H;
uint grp = lt / gw;
uint first = grp * GM;
uint gs = min(mt - first, GM);
uint inner = lt % gw;
const uint tm = first + inner % gs;
const uint tn = inner / gs;
const uint row0 = tm * 64u;
const uint n0 = tn * 128u;
const uint LDA = 40u;
const uint LDB = 40u;
const uint LDC = 132u;
threadgroup bfloat16_t* As = (threadgroup bfloat16_t*)smem;
threadgroup bfloat16_t* Bs = As + 64u * LDA;
const device bfloat16_t* ap = (job == 0u) ? in0 : in8;
const device uint* wp; const device bfloat16_t* sp; const device bfloat16_t* bp;
if (job == 0u) { wp = in1; sp = in2; bp = in3; }
else if (job == 1u) { wp = in9; sp = in10; bp = in11; }
else if (job == 2u) { wp = in12; sp = in13; bp = in14; }
else { wp = in15; sp = in16; bp = in17; }
uint sm = sg / 4u;
uint sn = sg % 4u;
metal::simdgroup_float8x8 acc[4][4];
for (uint i = 0; i < 4u; ++i) for (uint j = 0; j < 4u; ++j) acc[i][j] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
uint ar = tid / 4u;
uint ak = (tid % 4u) * 8u;
uint arow = row0 + ar;
bool aok = arow < M;
const device bfloat16_t* arp = ap + (ulong)(aok ? arow : 0u) * K + ak;
threadgroup uint4* dstA = (threadgroup uint4*)(As + ar * LDA + ak);
for (uint k0 = 0; k0 < K; k0 += 32u) {
  uint4 v = uint4(0u);
  if (aok) v = *((const device uint4*)(arp + k0));
  *dstA = v;
  for (uint ii = 0; ii < 2u; ++ii) {
    uint idx = tid * 2u + ii;
    uint nn = idx / 4u;
    uint w = idx % 4u;
    uint gn = n0 + nn;
    uint word = wp[(ulong)gn * KW + k0 / 8u + w];
    uint si = gn * KG + k0 / 64u;
    float sc = float(sp[si]);
    float bi = float(bp[si]);
    uint4 pk = uint4(0u);
    thread bfloat16_t* pb = (thread bfloat16_t*)&pk;
    for (uint j = 0; j < 8u; ++j) {
      float q = float((word >> (4u * j)) & 0xFu);
      pb[j] = static_cast<bfloat16_t>(sc * q + bi);
    }
    *((threadgroup uint4*)(Bs + nn * LDB + w * 8u)) = pk;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint kk = 0; kk < 32u; kk += 8u) {
    metal::simdgroup_matrix<bfloat16_t, 8, 8> a[4];
    metal::simdgroup_matrix<bfloat16_t, 8, 8> b[4];
    for (uint i = 0; i < 4u; ++i) metal::simdgroup_load(a[i], As + (sm * 32u + i * 8u) * LDA + kk, LDA);
    for (uint j = 0; j < 4u; ++j) metal::simdgroup_load(b[j], Bs + (sn * 32u + j * 8u) * LDB + kk, LDB, ulong2(0, 0), true);
    for (uint i = 0; i < 4u; ++i) for (uint j = 0; j < 4u; ++j) metal::simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = 0; i < 4u; ++i) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 4u; ++j) metal::simdgroup_store(acc[i][j], smem + (sm * 8u) * LDC + sn * 32u + j * 8u, LDC);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (job == 0u || job == 3u) {
    uint er = tid / 16u;
    uint ec = (tid % 16u) * 8u;
    uint grow = row0 + (er / 8u) * 32u + i * 8u + (er % 8u);
    if (grow < M) {
      uint4 pk = uint4(0u);
      thread bfloat16_t* pb = (thread bfloat16_t*)&pk;
      for (uint j = 0; j < 8u; ++j) pb[j] = static_cast<bfloat16_t>(smem[er * LDC + ec + j]);
      uint s = (job == 0u) ? (Ma + grow) : grow;
      *((device uint4*)(out0 + (ulong)(tn * S + s) * 128u + ec)) = pk;
    }
  } else {
    for (uint q = 0; q < 2u; ++q) {
      uint er = sg * 2u + q;
      uint grow = row0 + (er / 8u) * 32u + i * 8u + (er % 8u);
      if (grow < M) {
        threadgroup float* src = smem + er * LDC + lane * 4u;
        float4 x = float4(float(static_cast<bfloat16_t>(src[0])), float(static_cast<bfloat16_t>(src[1])), float(static_cast<bfloat16_t>(src[2])), float(static_cast<bfloat16_t>(src[3])));
        const device bfloat16_t* wsrc = ((job == 1u) ? in18 : in19) + lane * 4u;
        float4 w = float4(float(wsrc[0]), float(wsrc[1]), float(wsrc[2]), float(wsrc[3]));
        uint fi = grow * 64u + lane * 2u;
        float2 cs = float2(in20[fi], in20[fi + 1u]);
        float2 snv = float2(in21[fi], in21[fi + 1u]);
        device bfloat16_t* dst = ((job == 1u) ? out1 : out2) + (ulong)(tn * S + grow) * 128u + lane * 4u;
        nr_apply(x, w, cs, snv, dst);
      }
    }
  }
}
