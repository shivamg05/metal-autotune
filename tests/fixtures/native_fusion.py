"""Native recurrence surrounded by ordinary operations, with live state."""
import mlx.core as mx
import mlx.nn as nn


class Step(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = mx.fast.metal_kernel(
            name='fusion_recurrence', input_names=['x', 'state', 'count'],
            output_names=['value', 'next_state'],
            source='''uint i = thread_position_in_grid.x;
if (i < count) { T v = x[i] + state[i]; value[i] = v * 2; next_state[i] = v; }''',
            compile_options={'math_mode': 'safe'})

    def __call__(self, x, state):
        x = mx.multiply(x, 2)
        value, next_state = self.kernel(
            inputs=[x, state, x.size], template=[('T', x.dtype)],
            grid=(x.size, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[x.shape, state.shape], output_dtypes=[x.dtype, state.dtype])
        return mx.add(value, 1), next_state


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = Step()

    def __call__(self, x, state):
        return self.step(x, state)


def build():
    return Model()
