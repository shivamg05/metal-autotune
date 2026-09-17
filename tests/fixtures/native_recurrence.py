"""Small native two-output recurrence for the search/export integration tests."""
import mlx.core as mx
import mlx.nn as nn

kernel = mx.fast.metal_kernel(
    name='native_recurrence', input_names=['x', 'state', 'count'], output_names=['y', 'next_state'],
    source='''uint i = thread_position_in_grid.x;
if (i < count) { T value = x[i] + state[i]; y[i] = value * 2; next_state[i] = value; }''',
    compile_options={'math_mode': 'safe'})


class Step(nn.Module):
    def __call__(self, x, state):
        return kernel(inputs=[x, state, x.size], template=[('T', x.dtype), ('Width', x.size)],
                      grid=(x.size, 1, 1), threadgroup=(32, 1, 1),
                      output_shapes=[x.shape, state.shape], output_dtypes=[x.dtype, state.dtype])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = Step()

    def __call__(self, x, state):
        return self.step(x, state)

def build():
    return Model()
