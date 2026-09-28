const uint M = (uint)in0_shape[0] * (uint)in0_shape[1];
if (M < 64u) return;
const uint K = (uint)in0_shape[2];
const uint N = (uint)in1_shape[0];
const uint KW = K / 8u;
const uint KG = K / 64u;
const uint tid = thread_position_in_threadgroup.x;
const uint NT = N / 64u;
const uint tg = threadgroup_position_in_grid.x;
const uint nb = (tg % NT) * 64u;
const uint wb = (tg / NT) * 8u;
threadgroup uint Tu[64 * 33];
threadgroup bfloat16_t* T = (threadgroup bfloat16_t*)Tu;
for (uint i = 0; i < 2u; ++i) {
  const uint idx = tid + i * 256u;
  const uint nl = idx / 8u;
  const uint wl = idx % 8u;
  const uint n = nb + nl;
  const uint word = in1[(size_t)n * KW + wb + wl];
  const size_t g = (size_t)n * KG + (wb >> 3);
  const float s = float(in2[g]);
  const float b = float(in3[g]);
  for (uint j = 0; j < 8u; ++j) {
    float q = float((word >> (4u * j)) & 0xFu);
    T[(wl * 8u + j) * 66u + nl] = static_cast<bfloat16_t>(metal::fma(s, q, b));
  }
}
metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (uint i = 0; i < 8u; ++i) {
  const uint p = tid + i * 256u;
  const uint kl = p / 32u;
  const uint np2 = (p % 32u) * 2u;
  const uint v = *((const threadgroup uint*)(T + kl * 66u + np2));
  *((device uint*)(out0 + (size_t)(wb * 8u + kl) * N + nb + np2)) = v;
}
