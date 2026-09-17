"""Ordered candidates use the same correctness, timing, bind and export gates."""
import copy
import json
from dataclasses import replace

import mlx.core as mx
import pytest

from autotuner.judge.schema import validate_response, MalformedResponse
from autotuner.ladder.static_checks import RegionContract, check, launch_resource_failure
from autotuner.loop import kernel_from_proposal, _kernel_view
from autotuner_runtime.kernels import KernelSpec, LoadedKernel, load_spec


def stage(inputs, outputs, body, *, shapes=None, dtypes=None):
    return dict(inputs=inputs, outputs=outputs, source=body,
                grid=['in0.shape[0]', '1', '1'], threadgroup=['32', '1', '1'],
                output_shapes=shapes or [['in0.shape[0]'] for _ in outputs],
                output_dtypes=dtypes or ['float32' for _ in outputs])


def proposal():
    return {'parent_kernel_id': 'head', 'output_shapes': [['in0.shape[0]']], 'stages': [
        stage(['in0'], ['tmp0'], 'uint i=thread_position_in_grid.x; out0[i]=in0[i]*2.0f;'),
        stage(['tmp0'], ['out0'], 'uint i=thread_position_in_grid.x; out0[i]=in0[i]+1.0f;'),
    ]}


def contract():
    return RegionContract(('in0',), (1,), ('float32',), ('out0',), (1,),
                          ('float32',), ('out0',), input_shapes=((32,),), output_shapes=((32,),))


def parse(obj=None, *, parent=None, boundary=None):
    parsed = validate_response({'mutations': [], 'kernel': obj or proposal()}).kernel
    return kernel_from_proposal(boundary or contract(), parent, parsed, 'two_stages')


def test_stages_execute_exactly_and_roundtrip_with_dynamic_shapes(tmp_path):
    from autotuner.artifact.emit import write_kernel
    spec = parse()
    assert check(spec, contract()) == []
    assert KernelSpec.from_json(spec.to_json()) == spec
    write_kernel(tmp_path, spec)
    assert (tmp_path/'two_stages.stages/1.metal').read_text() == spec.stages[1].kernel.source
    restored = load_spec(tmp_path/'two_stages.metal')
    assert restored == spec
    for n in (1, 7, 32, 83):
        x = mx.random.normal((n,), key=mx.random.key(n))
        y = LoadedKernel(restored)([x], init_value=float('nan'))[0]
        assert mx.array_equal(y, x*2+1).item()
    assert _kernel_view(spec)['stages'][1]['inputs'] == ['tmp0']


def test_stage_intermediates_are_poisoned_too():
    obj = proposal()
    obj['stages'][0]['source'] = 'uint i=thread_position_in_grid.x; if(i>0) out0[i]=in0[i]*2.0f;'
    out = LoadedKernel(parse(obj))([mx.ones((32,))], init_value=float('nan'))[0]
    assert mx.isnan(out[0]).item()
    assert mx.array_equal(out[1:], mx.full((31,), 3.)).item()


def test_early_output_survives_and_stage_grammar_uses_local_input_slots():
    obj = proposal()
    obj['stages'][0]['outputs'] = ['out0']
    obj['stages'][1]['inputs'] = ['out0']
    obj['stages'][1]['outputs'] = ['out1']
    obj['output_shapes'] *= 2
    c = replace(contract(), output_names=('out0','out1'), output_ranks=(1,1),
                output_dtypes=('float32','float32'), live_outputs=('out0','out1'), output_shapes=((32,),(32,)))
    spec = parse(obj,boundary=c)
    assert check(spec,c) == []
    a,b = LoadedKernel(spec)([mx.ones((32,))])
    assert mx.array_equal(a,mx.full((32,),2.)).item()
    assert mx.array_equal(b,mx.full((32,),3.)).item()


@pytest.mark.parametrize('change', ['forward', 'overwrite', 'duplicate', 'dead', 'missing', 'dtype', 'shape', 'local_input', 'resource'])
def test_bad_stage_plans_are_rejected_before_gpu(change):
    obj=proposal()
    if change=='forward': obj['stages'][0]['inputs']=['tmp0']
    if change=='overwrite': obj['stages'][1]['outputs']=['in0']
    if change=='duplicate': obj['stages'][1]['outputs']=['tmp0']
    if change=='dead': obj['stages'][1]['inputs']=['in0']
    if change=='missing': obj['stages'][1]['outputs']=['tmp1']
    if change=='dtype': obj['stages'][1]['output_dtypes']=['float16']
    if change=='shape': obj['stages'][1]['output_shapes']=[['in0.shape[0]+1']]
    if change=='local_input': obj['stages'][1]['grid'][0]='in1.shape[0]'
    if change=='resource': obj['stages'][1]['threadgroup']=['1025','1','1']
    try:
        spec=parse(obj)
    except MalformedResponse:
        return
    assert check(spec,contract())


def test_resource_limits_apply_on_each_sweep_shape():
    obj=proposal()
    obj['stages'][1]['threadgroup'][0]='in0.shape[0]'
    spec=parse(obj)
    assert check(spec,contract()) == []
    failure=launch_resource_failure(spec,[(2048,)])
    assert failure and failure.check=='launch_extent' and 'stage 1' in failure.detail


def test_global_fallback_and_full_parent_header_inheritance():
    obj=proposal();obj['fallback_predicate']='in0.shape[0] != 32'
    parent=KernelSpec('parent','parent',('in0',),('out0',),'unused',header='// shared helpers')
    spec=parse(obj,parent=parent)
    assert all(s.kernel.header==parent.header for s in spec.stages)
    loaded=LoadedKernel(spec)
    assert not loaded.fallback_fires([mx.ones((32,))])
    assert loaded.fallback_fires([mx.ones((7,))])


@pytest.mark.parametrize('field,value',[('stages',[]),('source','x'),('scratch',[]),('template',[])])
def test_program_cannot_silently_ignore_single_kernel_fields(field,value):
    obj=proposal();obj[field]=value
    with pytest.raises(MalformedResponse): parse(obj)


@pytest.mark.parametrize("association", ["preserving", "changing"])
def test_native_candidate_uses_stages_and_preserves_all_state_outputs(tmp_path, association):
    from tests.test_native_search import runner_for
    runner,region=runner_for(tmp_path)
    try:
        seed=runner._build_scaffold(region)
        obj={'parent_kernel_id':'head','output_shapes':[['in0.shape[0]'],['in0.shape[0]']],
             'stages':[
                 stage(['in0','in1'],['out1'],'uint i=thread_position_in_grid.x; out0[i]=in0[i]+in1[i];'),
                 stage(['out1'],['out0'],'uint i=thread_position_in_grid.x; out0[i]=in0[i]*2.0f;')]}
        spec=parse(obj,parent=seed,boundary=runner._contract(region))
        assert check(spec,runner._contract(region)) == []
        result=runner._evaluate_kernel(region,spec,association,run_clock=False)
        assert result.outcome=='correct_slower', result
        assert result.detail['fallback_engaged']
        # Wrong state is a real numeric failure, not an internal exception.
        broken=replace(spec, stages=(replace(spec.stages[0],kernel=replace(spec.stages[0].kernel,
            source=spec.stages[0].kernel.source.replace('in0[i]+in1[i]','in0[i]+in1[i]+1.0f'))),spec.stages[1]))
        assert runner._evaluate_kernel(region,broken,association,run_clock=False).outcome=='failed'
        bad_compile=replace(spec,stages=(spec.stages[0],replace(spec.stages[1],kernel=replace(spec.stages[1].kernel,
            source='this_will_not_compile;'))))
        result=runner._evaluate_kernel(region,bad_compile,association,run_clock=False)
        assert result.failed_gate=='compile', result
    finally:
        runner.tracer.uninstall()


@pytest.mark.parametrize('baseline', ['plain', 'compiled'])
def test_full_model_install_retrace_checkpoint_and_fresh_export(tmp_path, monkeypatch, baseline):
    from pathlib import Path
    from autotuner.loop import JobRunner, RegionRun
    from autotuner.measure.session import Session
    from autotuner.ladder.gates import LadderResult
    from autotuner.bind.verify import verify_retrace
    from autotuner_runtime.stats import PairedComparison
    model=Path(__file__).parent/'fixtures/two_stage_model.py'
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(f'''model: {model}
baseline: {baseline}
workloads:
  - name: main
    inputs: [{{shape: [L], dtype: float32}}]
primary: {{L: 32}}
sweep: {{L: [7, 32]}}
budget: {{per_region: 1, total: 1}}
''')
    runner=JobRunner(manifest,tmp_path/'work',clock_pairs=4,
                     session=Session(sleep=lambda s:None),judge_factory=lambda r:None)
    try:
        runner.load_model();runner.trace_workloads()
        region=next(r for r in runner.build_regions() if len(r.ops)==2)
        runner._capture([region]);runner.tracer.uninstall()
        region.t_orig_ms['main']=region.t_rep_ms['main']=1.
        region.p['main']=1.
        seed=runner._build_scaffold(region)
        spec=parse(parent=seed,boundary=runner._contract(region))
        assert check(spec,runner._contract(region))==[]
        result=runner._evaluate_kernel(region,spec,'preserving',run_clock=True)
        assert result.outcome in ('correct_slower','tentative_ship'),result
        assert result.region_ms is not None and result.region_ms > 0
        # Control speed decisions only. This two-dispatch identity is not a
        # speedup claim; actual GPU correctness, timing and packaging still run.
        monkeypatch.setattr(runner,'_model_win',lambda r:all(c.passed for c in r.checks))
        monkeypatch.setattr(PairedComparison,'wins_by',lambda self,margin_ms:True)
        monkeypatch.setattr(PairedComparison,'loses_by',lambda self,margin_ms:False)
        assert runner._bind_and_promote(RegionRun(region),spec,
            LadderResult('tentative_ship',None,{},.01,.02,.01,0.,[]))
        runner.tracer.install()
        retrace,_=runner.tracer.trace(runner.model,runner.tensors['main'])
        spans=sorted(runner.cuts['main'])
        report=verify_retrace(runner.traces['main'],retrace,spans,[runner.cuts['main'][s] for s in spans])
        assert report.ok,report.reasons
        runner.tracer.uninstall()
        runner._final_check()
        assert runner.final_ok
        artifact=runner.emit_artifact(tmp_path/'artifact')
        assert (artifact/'kernels/two_stages.stages/1.metal').is_file()
        assert load_spec(artifact/'kernels/two_stages.metal')==spec
        assert list((tmp_path/'work/checkpoints').glob('accepted-*'))
    finally:
        runner.tracer.uninstall()


def test_split_k_reduction_uses_stage_local_shapes_and_declared_intermediate():
    obj={'parent_kernel_id':'head','output_shapes':[['in0.shape[0]','in1.shape[1]']], 'stages':[
        stage(['in0','in1'],['tmp0'], '''uint i=thread_position_in_grid.x;
uint M=in0_shape[0], K=in0_shape[1], N=in1_shape[1];
uint part=i/(M*N), row=(i/N)%M, col=i%N;
float acc=0; for(uint k=part*K/2; k<(part+1)*K/2; ++k) acc+=in0[row*K+k]*in1[k*N+col];
out0[i]=acc;''',shapes=[['2','in0.shape[0]','in1.shape[1]']]),
        stage(['tmp0'],['out0'], '''uint i=thread_position_in_grid.x;
uint size=in0_shape[1]*in0_shape[2]; out0[i]=in0[i]+in0[size+i];''',
              shapes=[['in0.shape[1]','in0.shape[2]']]) ]}
    obj['stages'][0]['grid'][0]='2*in0.shape[0]*in1.shape[1]'
    obj['stages'][1]['grid'][0]='in0.shape[1]*in0.shape[2]'
    c=RegionContract(('in0','in1'),(2,2),('float32','float32'),('out0',),(2,),('float32',),('out0',),
                     input_shapes=((32,64),(64,16)),output_shapes=((32,16),))
    spec=parse(obj,boundary=c)
    assert check(spec,c)==[]
    loaded=LoadedKernel(spec)
    for m,k,n in ((32,64,16),(7,18,13),(1,256,32)):
        a=mx.random.normal((m,k),key=mx.random.key(m))
        b=mx.random.normal((k,n),key=mx.random.key(n))
        got=loaded([a,b],init_value=float('nan'))[0]
        assert got.shape==(m,n)
        assert mx.allclose(got,a@b,rtol=1e-4,atol=1e-4).item()


def test_two_stage_reduction_can_return_a_scalar():
    obj={'parent_kernel_id':'head','output_shapes':[[]],'stages':[
        stage(['in0'],['tmp0'],'''uint part=thread_position_in_grid.x; float s=0;
for(uint i=part*16; i<(part+1)*16; ++i) s+=in0[i]; out0[part]=s;''',shapes=[['2']]),
        stage(['tmp0'],['out0'],'out0[0]=in0[0]+in0[1];',shapes=[[]])]}
    obj['stages'][0]['grid']=['2','1','1'];obj['stages'][0]['threadgroup']=['1','1','1']
    obj['stages'][1]['grid']=['1','1','1'];obj['stages'][1]['threadgroup']=['1','1','1']
    c=replace(contract(),output_ranks=(0,),output_shapes=((),))
    spec=parse(obj,boundary=c)
    assert check(spec,c)==[]
    x=mx.arange(32,dtype=mx.float32)
    y=LoadedKernel(spec)([x])[0]
    assert y.shape==() and mx.array_equal(y,mx.sum(x)).item()


def test_stages_handle_noncontiguous_inputs_and_local_dtype_templates():
    obj = proposal()
    obj['stages'][0]['source'] = 'uint i=thread_position_in_grid.x; out0[i]=T(in0[i]*2);'
    obj['stages'][0]['template'] = [['T', 'in0']]
    spec = parse(obj)
    loaded = LoadedKernel(spec)
    base = mx.arange(96, dtype=mx.float32).reshape(3, 32)
    for x in (base.T[:, 1], base[0, ::-1], mx.broadcast_to(mx.array(3.), (32,))):
        y = loaded([x])[0]
        assert mx.array_equal(y, x*2+1).item()


def test_buffer_limit_is_per_stage_not_whole_program():
    names = tuple(f'in{i}' for i in range(32))
    obj = {'parent_kernel_id': 'head', 'output_shapes': [['in0.shape[0]']], 'stages': []}
    for i in range(32):
        inputs = ['in0'] if i == 0 else [f'tmp{i-1}', f'in{i}']
        outputs = ['out0' if i == 31 else f'tmp{i}']
        body = 'uint i=thread_position_in_grid.x; out0[i]=in0[i]'
        body += ';' if i == 0 else '+in1[i];'
        obj['stages'].append(stage(inputs, outputs, body))
    c = replace(contract(), input_names=names, input_ranks=(1,)*32,
                input_dtypes=('float32',)*32, input_shapes=((32,),)*32)
    spec = parse(obj, boundary=c)
    assert check(spec, c) == []
    y = LoadedKernel(spec)([mx.full((32,), float(i)) for i in range(32)])[0]
    assert mx.array_equal(y, mx.full((32,), float(sum(range(32))))).item()
    # A single stage cannot bypass the same Metal argument-table limit.
    obj['stages'][0]['inputs'] = list(names)
    assert any(f.check == 'buffer_limit' for f in check(parse(obj, boundary=c), c))


def test_returning_to_single_native_kernel_restores_original_factory_policy():
    c = replace(contract(), native_call={'factory': {'atomic_outputs': True}})
    parent = KernelSpec('native', 'native', c.input_names, c.output_names, 'original',
                        native_call=c.native_call, atomic_outputs=True)
    staged = parse(parent=parent, boundary=c)
    assert not staged.atomic_outputs
    single = proposal()['stages'][0]
    single = {k: v for k, v in single.items() if k not in {'inputs', 'outputs', 'output_dtypes'}}
    single['parent_kernel_id'] = 'head'
    restored = parse(single, parent=staged, boundary=c)
    assert not restored.stages and restored.atomic_outputs


def test_staged_sources_keep_bounded_lossless_judge_navigation():
    from autotuner.judge.source_context import SourceContext, PREVIEW_CHARS
    obj = proposal()
    obj['header'] = '// shared helper definitions\n' * 1000
    obj['stages'][1]['source'] += '\n// stage detail' * 1000
    spec = parse(obj)
    context = SourceContext()
    focused = context.focus({'head': 'head', 'kernels': {'head': _kernel_view(spec)}})
    stages = focused['kernels']['head']['stages']
    assert stages[0]['header'] == stages[1]['header']
    sid = stages[1]['source']['source_id']
    chunks, start = [], 0
    while start is not None:
        part = context.read({'read_source': [{'id': sid, 'start': start}]})[0]
        chunks.append(part['text']); start = part['next_start']
    assert ''.join(chunks) == spec.stages[1].kernel.source
    assert sum(len(e['text']) for v in focused['source_catalog'].values()
               for e in v['excerpts']) <= PREVIEW_CHARS
