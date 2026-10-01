"""Prepare a harness-owned original op sequence, without interpreting Metal.

This is a correctness-preserving search starter, not a fused implementation.
All arguments and native definitions come from the trace. Calls and constants
are resolved once; the generated function only wires tensors between calls.
No optimizer imports are needed to reconstruct a serialized starter.
"""
import mlx.core as mx

from .captured_kernels import captured, definition_key
from .swap import flatten_arrays


def _resolve(op):
    if op.startswith('array.'):
        name = op.removeprefix('array.')
        if name in {'__setitem__', '__iadd__', '__isub__', '__imul__', '__itruediv__',
                    '__ifloordiv__', '__imod__', '__ipow__', '__imatmul__', '__iand__',
                    '__ior__', '__ixor__', '__ilshift__', '__irshift__'}:
            raise ValueError(f'mutating operation {op} is not a fusion starter')
        fn = getattr(mx.array, name)
        return fn if callable(fn) else lambda x: fn.__get__(x, mx.array)
    if not op.startswith('mx.'):
        raise ValueError(f'operation {op} cannot be replayed as a portable fusion starter')
    fn = mx
    for part in op.split('.')[1:]:
        fn = getattr(fn, part)
    if not callable(fn):
        raise ValueError(f'operation {op} is not callable')
    return fn


def prepare_sequence(sequence):
    """Compile the recorded wiring once, retaining the original dispatches."""
    if 'variants' in sequence:
        def key(signature):
            return tuple((tuple(shape), dtype) for shape, dtype in signature)
        variants = {key(v['signature']): prepare_sequence(v['sequence'])
                    for v in sequence['variants']}

        def run_variant(inputs):
            signature = tuple((tuple(a.shape), str(a.dtype).removeprefix('mlx.core.')) for a in inputs)
            return variants[signature](inputs)

        return run_variant

    namespace = {'_outputs': flatten_arrays, '_slice': slice}
    names = {aid: f'v{i}' for i, aid in enumerate(sequence['input_ids'])}
    lines = ['def run(inputs):']
    lines.extend(f'    {name} = inputs[{i}]' for i, name in enumerate(names.values()))

    def constant(value):
        name = f'c{len(namespace)}'
        namespace[name] = value
        return name

    def expression(value, bound):
        if not isinstance(value, dict) or '$' not in value:
            return constant(value)
        tag = value['$']
        if tag == 'ref':
            return bound[value['i']]
        if tag in ('tuple', 'list'):
            contents = ','.join(expression(v, bound) for v in value['items'])
            return '(' + contents + ',)' if tag == 'tuple' and contents else (
                '()' if tag == 'tuple' else '[' + contents + ']')
        if tag == 'dict':
            return '{' + ','.join(constant(k) + ':' + expression(v, bound)
                                  for k, v in value['items'].items()) + '}'
        if tag == 'slice':
            return '_slice(' + ','.join(expression(v, bound) for v in value['parts']) + ')'
        if tag == 'dtype':
            return constant(getattr(mx, 'bool_' if value['name'] == 'bool' else value['name']))
        if tag == 'ellipsis':
            return constant(Ellipsis)
        raise ValueError(f'unsupported recorded argument {tag!r}')

    for index, node in enumerate(sequence['nodes']):
        definition = node.get('kernel_definition')
        fn = captured(definition_key(definition)) if definition is not None else _resolve(node['op'])
        bound = [names[aid] for aid in node['in_arrays']]
        args = expression(node['args'], bound)
        kwargs = expression({'$': 'dict', 'items': node['kwargs']}, bound)
        lines.append(f'    r{index} = {constant(fn)}(*{args}, **{kwargs})')
        outs = [f'o{index}_{i}' for i in range(len(node['out_arrays']))]
        if outs:
            lines.append(f'    {",".join(outs)}, = _outputs(r{index})')
        else:
            lines.append(f'    if _outputs(r{index}): raise ValueError("unexpected original outputs")')
        names.update(zip(node['out_arrays'], outs))
    lines.append('    return [' + ','.join(names[aid] for aid in sequence['output_ids']) + ']')
    exec(compile('\n'.join(lines), '<original-fusion-sequence>', 'exec'), namespace)
    return namespace['run']
