"""Editable native seeds keep the original ABI; no source rewriting."""
import copy
import mlx.core as mx
from autotuner.trace.recorder import ArrayRef, dtype_name
from autotuner_runtime.kernels import KernelSpec


def native_seed(trace, span):
    node = trace.nodes[span.start_seq]
    if span.start_seq != span.end_seq or node.kernel_definition is None:
        raise ValueError('native seeds require one captured kernel')
    definition = copy.deepcopy(node.kernel_definition['kwargs'])
    if set(definition) - {"name", "input_names", "output_names", "source", "header",
                          "ensure_row_contiguous", "atomic_outputs", "compile_options"}:
        raise ValueError("unsupported native factory settings")
    if not span.input_ids:
        raise ValueError("native search requires at least one tensor input")
    launch = node.scalar_args['kwargs']
    if node.scalar_args['args'] or set(launch) - {
        'inputs', 'template', 'grid', 'threadgroup', 'output_shapes', 'output_dtypes',
        'init_value', 'verbose', 'stream'}:
        raise ValueError('unsupported native launch arguments')
    if launch.get('stream') is not None:
        raise ValueError('native kernels with explicit streams are not search targets')
    bindings = []
    for arg in launch['inputs']:
        if isinstance(arg, ArrayRef):
            bindings.append({'array': span.input_ids.index(node.in_arrays[arg.index])})
        elif type(arg) in (int, float, bool):
            bindings.append({'scalar': arg})
        else:
            raise ValueError('unsupported native scalar input')
    # Keep every native output, including currently unused outputs. Removing
    # one can change the source contract or hide a bad store during validation.
    if tuple(span.output_ids) != node.out_arrays:
        raise ValueError('native target must expose all outputs')
    template = []
    for name, value in launch.get('template', []):
        if isinstance(value, mx.Dtype):
            value = {'dtype': dtype_name(value)}
        elif type(value) not in (int, bool):
            raise ValueError('unsupported native template value')
        template.append([name, value])
    spec_of = trace.span_specs(span.start_seq, span.end_seq)
    return KernelSpec(
        kernel_id='native_seed', name='at_native_seed',
        input_names=tuple(f'in{i}' for i in range(len(span.input_ids))),
        output_names=tuple(f'out{i}' for i in range(len(span.output_ids))),
        source=definition.pop('source'), header=definition.pop('header', ''),
        grid=tuple(str(x) for x in launch['grid']),
        threadgroup=tuple(str(x) for x in launch['threadgroup']),
        output_shapes=tuple(tuple(str(x) for x in s) for s in launch['output_shapes']),
        output_dtypes=tuple(dtype_name(d) for d in launch['output_dtypes']),
        ensure_row_contiguous=definition.get('ensure_row_contiguous', True),
        atomic_outputs=definition.get('atomic_outputs', False),
        native_call={'factory': definition, 'bindings': bindings, 'template': template,
                     'signature': [[list(spec_of[a][0]), spec_of[a][1]] for a in span.input_ids],
                     'init_value': launch.get('init_value')})


def reference_sequence_seed(trace, span, instances=()):
    """Keep original calls as a valid starter when no single kernel exists.

    The judge receives all captured source and the complete wiring. Its next
    proposal replaces this sequence with custom code using the region ABI,
    in one dispatch or explicit ordered stages;
    no generated Python or replay metadata is editable by the judge.
    """
    from dataclasses import replace
    from autotuner.trace.serialize import node_to_dict
    from autotuner.trace.optable import MUTATING_METHODS
    from autotuner_runtime.original_sequence import _resolve
    from .symshape import NoScaffold

    if instances:
        seeds = [reference_sequence_seed(t, s) for t, s in [(trace, span), *instances]]
        variants = []
        for i, seed in enumerate(seeds):
            previous = next((s for s in seeds[:i]
                             if s.input_signature == seed.input_signature), None)
            if previous is not None and previous.output_shapes != seed.output_shapes:
                raise NoScaffold('ambiguous-reference', 'identical input shapes require different output shapes')
            if seed.input_signature not in [v['signature'] for v in variants]:
                variants.append({'signature': seed.input_signature, 'sequence': seed.reference_sequence})
        if len(variants) > 1:
            return replace(seeds[0], input_signature=None,
                           input_signatures=[v['signature'] for v in variants],
                           reference_sequence={'variants': variants})
        return seeds[0]

    nodes = trace.nodes[span.start_seq:span.end_seq + 1]
    for node in nodes:
        try:
            if node.kernel_definition is not None:
                # Validate factory and launch support with the same rules as
                # individual native targets; retain every intermediate output.
                native_seed(trace, replace(span, start_seq=node.seq, end_seq=node.seq,
                                           input_ids=tuple(dict.fromkeys(node.in_arrays)),
                                           output_ids=node.out_arrays))
            elif node.op.removeprefix('array.') in MUTATING_METHODS:
                raise ValueError('mutating operations cannot be fusion targets')
            else:
                _resolve(node.op)
        except (ValueError, KeyError, AttributeError) as exc:
            raise NoScaffold(str(exc)) from exc
    spec_of = trace.span_specs(span.start_seq, span.end_seq)
    return KernelSpec(
        kernel_id='sequence_seed', name='at_sequence_seed',
        input_names=tuple(f'in{i}' for i in range(len(span.input_ids))),
        output_names=tuple(f'out{i}' for i in range(len(span.output_ids))),
        source='// Original sequence starter: see reference_sequence for source and wiring.\n'
               '// Propose a replacement Metal body using the region input/output names.',
        grid=('1', '1', '1'), threadgroup=('1', '1', '1'),
        output_shapes=tuple(tuple(str(x) for x in spec_of[a][0]) for a in span.output_ids),
        output_dtypes=tuple(spec_of[a][1] for a in span.output_ids),
        input_signature=[[list(spec_of[a][0]), spec_of[a][1]] for a in span.input_ids],
        reference_sequence={'nodes': [node_to_dict(n) for n in nodes],
                            'input_ids': list(span.input_ids), 'output_ids': list(span.output_ids)})
