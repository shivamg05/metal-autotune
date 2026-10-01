const uint t = thread_position_in_grid.x;
const uint rows = (uint)in0_shape[0];
const uint wpr = (uint)in0_shape[1];
if (t >= rows * wpr) return;
const uint n = t / wpr;
const uint j = t - n * wpr;
const uint Kg = (wpr * 8u) / 64u;
const uint g = (j * 8u) / 64u;
T scale = in1[n * Kg + g];
T bias = in2[n * Kg + g];
T loc[8];
dequantize<T, 8, 4>((const device uint8_t*)(in0 + t), scale, bias, loc);
device T* dst = out0 + (size_t)t * 8u;
for (int i = 0; i < 8; ++i) { dst[i] = loc[i]; }
