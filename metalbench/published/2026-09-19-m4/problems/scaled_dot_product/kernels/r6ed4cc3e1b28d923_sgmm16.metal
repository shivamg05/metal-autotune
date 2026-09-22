const uint tgx = threadgroup_position_in_grid.x;
const uint tgy = threadgroup_position_in_grid.y;
const uint sg = thread_position_in_threadgroup.x / 32u;
const uint sr = sg / 2u;
const uint sc = sg % 2u;
const uint row0 = tgy * 32u + sr * 16u;
const uint col0 = tgx * 32u + sc * 16u;
metal::simdgroup_float8x8 acc00 = metal::simdgroup_float8x8(0.0f);
metal::simdgroup_float8x8 acc01 = metal::simdgroup_float8x8(0.0f);
metal::simdgroup_float8x8 acc10 = metal::simdgroup_float8x8(0.0f);
metal::simdgroup_float8x8 acc11 = metal::simdgroup_float8x8(0.0f);
for (uint k = 0; k < 128u; k += 8u) {
    metal::simdgroup_float8x8 a0, a1, b0, b1;
    metal::simdgroup_load(a0, in0 + row0 * 128u + k, 128u);
    metal::simdgroup_load(a1, in0 + (row0 + 8u) * 128u + k, 128u);
    metal::simdgroup_load(b0, in1 + k * 128u + col0, 128u);
    metal::simdgroup_load(b1, in1 + k * 128u + col0 + 8u, 128u);
    metal::simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
    metal::simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
    metal::simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
    metal::simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
}
metal::simdgroup_store(acc00, out0 + row0 * 128u + col0, 128u);
metal::simdgroup_store(acc01, out0 + row0 * 128u + col0 + 8u, 128u);
metal::simdgroup_store(acc10, out0 + (row0 + 8u) * 128u + col0, 128u);
metal::simdgroup_store(acc11, out0 + (row0 + 8u) * 128u + col0 + 8u, 128u);
