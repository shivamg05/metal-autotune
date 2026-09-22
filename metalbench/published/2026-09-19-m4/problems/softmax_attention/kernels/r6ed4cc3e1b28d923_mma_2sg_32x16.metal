const uint lid = thread_position_in_threadgroup.x;
const uint sg = lid / 32u;
const uint cb = thread_position_in_grid.x / 64u;
const uint rb = thread_position_in_grid.y;
const uint row0 = rb * 32u + sg * 16u;
const uint col0 = cb * 16u;
metal::simdgroup_float8x8 acc00 = metal::simdgroup_float8x8(0.0f);
metal::simdgroup_float8x8 acc01 = metal::simdgroup_float8x8(0.0f);
metal::simdgroup_float8x8 acc10 = metal::simdgroup_float8x8(0.0f);
metal::simdgroup_float8x8 acc11 = metal::simdgroup_float8x8(0.0f);
const device float* A = (const device float*)in0;
const device float* B = (const device float*)in1;
for (uint k = 0; k < 64u; k += 8u) {
    metal::simdgroup_float8x8 a0, a1, b0, b1;
    metal::simdgroup_load(a0, A + row0 * 64u + k, 64u);
    metal::simdgroup_load(a1, A + (row0 + 8u) * 64u + k, 64u);
    metal::simdgroup_load(b0, B + k * 128u + col0, 128u);
    metal::simdgroup_load(b1, B + k * 128u + col0 + 8u, 128u);
    metal::simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
    metal::simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
    metal::simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
    metal::simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
}
device float* C = (device float*)out0;
metal::simdgroup_store(acc00, C + row0 * 128u + col0, 128u);
metal::simdgroup_store(acc01, C + row0 * 128u + col0 + 8u, 128u);
metal::simdgroup_store(acc10, C + (row0 + 8u) * 128u + col0, 128u);
metal::simdgroup_store(acc11, C + (row0 + 8u) * 128u + col0 + 8u, 128u);
