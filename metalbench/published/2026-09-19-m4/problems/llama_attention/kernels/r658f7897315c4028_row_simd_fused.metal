const uint row = threadgroup_position_in_grid.x;
const uint lane = thread_position_in_threadgroup.x;
const uint b = row / 64u;
const float s = metal::precise::sqrt(32.0f);
const device float* sc = in0 + row * 64u;
float x0 = ((float)sc[lane]) / s;
float x1 = ((float)sc[lane + 32u]) / s;
float m = metal::max(x0, x1);
m = simd_max(m);
float e0 = metal::precise::exp(x0 - m);
float e1 = metal::precise::exp(x1 - m);
float sum = simd_sum(e0 + e1);
threadgroup float p[64];
p[lane] = e0 / sum;
p[lane + 32u] = e1 / sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
const device float* v = in1 + b * 2048u;
float acc = 0.0f;
for (uint k = 0; k < 64u; ++k) {
    acc += p[k] * ((float)v[k * 32u + lane]);
}
out0[row * 32u + lane] = (T)acc;
