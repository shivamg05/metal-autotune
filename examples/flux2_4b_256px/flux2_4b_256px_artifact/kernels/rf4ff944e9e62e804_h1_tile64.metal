threadgroup T Xs[64 * (32 + 16 / sizeof(T))];
threadgroup T Ws[64 * (32 + 16 / sizeof(T))];
qmm_t_impl<T, 64, 4, true, 64, 32, 64>(
    in1, in2, in3, in0, out0, Xs, Ws,
    in0_shape[in0_ndim - 1], in1_shape[0], in0_shape[in0_ndim - 2], in0_shape[in0_ndim - 1],
    threadgroup_position_in_grid, thread_index_in_threadgroup,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
