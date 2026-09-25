const uint gidx = thread_position_in_grid.x;
const uint row = gidx / 32u;
const uint lane = gidx % 32u;
const uint nrows = 24u * (uint)in0_shape[2];
if (row >= nrows) { return; }
const uint base = row * 128u + lane * 4u;
float xv[4];
float acc = 0.0f;
for (uint i = 0; i < 4u; ++i) {
    xv[i] = in0[base + i];
    acc = metal::fma(xv[i], xv[i], acc);
}
acc = metal::simd_sum(acc);
const float normalizer = metal::precise::rsqrt(acc / 128.0f + 1e-05f);
for (uint i = 0; i < 4u; ++i) {
    const float w = (float)in1[lane * 4u + i];
    const float y = w * (xv[i] * normalizer);
    out0[base + i] = (bfloat16_t)y;
}
