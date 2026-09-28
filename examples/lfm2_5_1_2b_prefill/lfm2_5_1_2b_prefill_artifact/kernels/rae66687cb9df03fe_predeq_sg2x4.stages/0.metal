uint M = (uint)(in0_shape[0] * in0_shape[1]);
if (M < 6u) return;
uint w = thread_position_in_grid.x;
uint n = thread_position_in_grid.y;
uint W = (uint)in1_shape[1];
uint G = (uint)in2_shape[1];
uint K = W * 8u;
uint word = in1[n * W + w];
float s = float(in2[n * G + (w >> 3)]);
float b = float(in3[n * G + (w >> 3)]);
uint base = n * K + w * 8u;
for (uint i = 0; i < 8u; ++i) {
  float q = float((word >> (4u * i)) & 0xFu);
  out0[base + i] = T(metal::fma(s, q, b));
}
