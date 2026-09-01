"""Gate 1 on its own, no GPU: what a proposal turns into and what the static
checker refuses before any child is spawned."""

from autotuner.judge.schema import validate_response
from autotuner.ladder.static_checks import RegionContract, check
from autotuner.loop import RegionRun, kernel_from_proposal, resolve_parent
from autotuner.regions.types import Region
from autotuner_runtime.kernels import KernelSpec

CONTRACT = RegionContract(
    input_names=("in0",), input_ranks=(2,), input_dtypes=("float32",),
    output_names=("out0",), output_ranks=(2,), output_dtypes=("float32",),
    live_outputs=("out0",), input_shapes=((8, 16),), output_shapes=((8, 16),),
)
PARENT = KernelSpec(
    kernel_id="rabc_scaffold", name="at_rabc_scaffold", input_names=("in0",),
    output_names=("out0", "tmp0"), source="out0[0] = in0[0];",
    header="inline float twice(float x) { return 2.0f * x; }",
    grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("32", "1", "1"),
    output_shapes=(("in0.shape[0]", "in0.shape[1]"), ("in0.shape[0]", "in0.shape[1]")),
    output_dtypes=("float32", "float32"), template=(("T", "in0"),),
)


def proposal(**over):
    d = {"source": "out0[0] = in0[0];", "parent_kernel_id": "rabc_scaffold",
         "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"], "threadgroup": ["32", "1", "1"],
         "output_shapes": [["in0.shape[0]", "in0.shape[1]"]]}
    d.update(over)
    return validate_response({"mutations": [], "kernel": d}).kernel


def test_scratch_buffers_become_extra_outputs_and_pass_gate_one():
    spec = kernel_from_proposal(
        CONTRACT, PARENT,
        proposal(scratch=[["tmp0", "float32", ["in0.shape[0]", "in0.shape[1]"]]]), "rabc_h1")
    assert spec.output_names == ("out0", "tmp0")
    assert spec.output_dtypes == ("float32", "float32")
    assert spec.output_shapes[1] == ("in0.shape[0]", "in0.shape[1]")
    assert check(spec, CONTRACT) == []


def test_missing_header_and_template_inherit_the_parents():
    spec = kernel_from_proposal(CONTRACT, PARENT, proposal(), "rabc_h2")
    assert spec.header == PARENT.header and spec.template == PARENT.template
    own = kernel_from_proposal(CONTRACT, PARENT,
                               proposal(header="// mine", template=[["T", "float32"]]), "rabc_h3")
    assert own.header == "// mine" and own.template == (("T", "float32"),)
    orphan = kernel_from_proposal(CONTRACT, None, proposal(parent_kernel_id="nope"), "rabc_h4")
    assert orphan.header == "" and orphan.template == ()


def test_parent_may_be_named_by_role_hypothesis_or_kernel_id():
    """The spec lets the judge edit head, the scaffold, or any earlier kernel;
    all three spellings must find the same kernel."""
    run = RegionRun(region=Region(fingerprint="abcdef0123456789", ops=("mx.exp",)))
    h1 = KernelSpec(**{**{k: getattr(PARENT, k) for k in PARENT.__dataclass_fields__},
                       "kernel_id": "rabcdef_h1", "name": "at_rabcdef_h1"})
    run.scaffold = run.head = PARENT
    run.kernels = {PARENT.kernel_id: PARENT, h1.kernel_id: h1}
    assert resolve_parent(run, "scaffold") is PARENT
    assert resolve_parent(run, "head") is PARENT
    assert resolve_parent(run, "shipped") is None
    assert resolve_parent(run, "h1") is h1
    assert resolve_parent(run, "rabcdef_h1") is h1
    assert resolve_parent(run, "h9") is None


def test_wrong_output_shape_expression_fails_gate_one_not_the_child():
    spec = kernel_from_proposal(
        CONTRACT, None, proposal(output_shapes=[["in0.shape[1]", "in0.shape[0]"]]), "rabc_h5")
    checks = {f.check: f.detail for f in check(spec, CONTRACT)}
    assert "output_shape" in checks
    assert "(16, 8)" in checks["output_shape"] and "(8, 16)" in checks["output_shape"]
