constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int WM = 2;
constexpr int WN = 2;
constexpr int BKp = BK + 16 / sizeof(T);
struct RV { uint4 a; uint4 b; };
threadgroup uint4 Xraw[(BM * BKp * sizeof(T)) / 16];
threadgroup uint4 Wraw[(BN * BKp * sizeof(T)) / 16];
threadgroup T* Xs = (threadgroup T*)Xraw;
threadgroup T* Ws = (threadgroup T*)Wraw;
using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BKp, BKp>;
const int K = in0_shape[in0_ndim - 1];
const int N = in1_shape[0];
const uint3 tid = threadgroup_position_in_grid;
const int y_row = tid.y * BM;
const int y_col = tid.x * BN;
const int lt = (int)simdgroup_index_in_threadgroup * 32 + (int)thread_index_in_simdgroup;
const int bi = lt >> 1;
const int bj = (lt & 1) * 16;
const device T* xs = (const device T*)in0 + (int64_t)(y_row + bi) * K + bj;
const device T* ws = (const device T*)in1 + (int64_t)(y_col + bi) * K + bj;
threadgroup RV* dx = (threadgroup RV*)(Xs + bi * BKp + bj);
threadgroup RV* dw = (threadgroup RV*)(Ws + bi * BKp + bj);
device T* y = out0 + (int64_t)y_row * N + y_col;
mma_t mma_op(simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
RV rx = *((const device RV*)xs);
RV rw = *((const device RV*)ws);
for (int k = 0; k < K; k += BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  *dx = rx;
  *dw = rw;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k + BK < K) {
    xs += BK;
    ws += BK;
    rx = *((const device RV*)xs);
    rw = *((const device RV*)ws);
  }
  mma_op.mma(Xs, Ws);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
mma_op.store_result(y, N);
