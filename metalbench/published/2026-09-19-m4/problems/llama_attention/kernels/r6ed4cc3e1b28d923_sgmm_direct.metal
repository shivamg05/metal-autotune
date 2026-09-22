const uint K = 128u;
const uint N = (uint)in1_shape[1];
const uint sg = thread_position_in_threadgroup.x / 32u;
const uint tile_m = thread_position_in_grid.y * 32u;
const uint tile_n = (thread_position_in_grid.x / 128u) * 32u;
const uint rbase = tile_m + (sg / 2u) * 16u;
const uint cbase = tile_n + (sg % 2u) * 16u;
metal::simdgroup_float8x8 acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);
for (uint k = 0; k < K; k += 8u) {
    metal::simdgroup_float8x8 a0, a1, b0, b1;
    metal::simdgroup_load(a0, in0 + (rbase + 0u) * K + k, K);
    metal::simdgroup_load(a1, in0 + (rbase + 8u) * K + k, K);
    metal::simdgroup_load(b0, in1 + k * N + cbase + 0u, N);
    metal::simdgroup_load(b1, in1 + k * N + cbase + 8u, N);
    metal::simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
    metal::simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
    metal::simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
    metal::simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
}
metal::simdgroup_store(acc00, out0 + (rbase + 0u) * N + cbase + 0u, N);
metal::simdgroup_store(acc01, out0 + (rbase + 0u) * N + cbase + 8u, N);
metal::simdgroup_store(acc10, out0 + (rbase + 8u) * N + cbase + 0u, N);
metal::simdgroup_store(acc11, out0 + (rbase + 8u) * N + cbase + 8u, N);
