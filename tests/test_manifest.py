"""M2: manifest parsing, validation, defaults, and the build() probe."""

import textwrap

import pytest

from autotuner.manifest import (
    DEFAULT_BUDGET_PER_REGION,
    DEFAULT_BUDGET_TOTAL,
    DEFAULT_SWEEP_SIZES,
    Manifest,
    ManifestError,
    check_build,
    load,
)


def write_manifest(tmp_path, body, model_body="def build():\n    return lambda x: x\n"):
    (tmp_path / "model.py").write_text(model_body)
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(body))
    return p


GOOD = """
model: ./model.py
workloads:
  - inputs: [{shape: [8, L], dtype: int32}]
    name: midbatch
  - inputs: [{shape: [1, 1], dtype: int32}]
    name: decode
sweep: {L: [1, 7, 512, 4096]}
"""


def test_good_manifest_loads_with_defaults(tmp_path):
    m = load(write_manifest(tmp_path, GOOD))
    assert m.model_path == (tmp_path / "model.py").resolve()
    assert [w.name for w in m.workloads] == ["midbatch", "decode"]
    assert m.workloads[0].inputs[0].shape == (8, "L")
    assert m.sweep["L"] == (1, 7, 512, 4096)
    assert m.primary["L"] == 4096
    assert m.budget_per_region == DEFAULT_BUDGET_PER_REGION
    assert m.budget_total == DEFAULT_BUDGET_TOTAL
    assert m.tolerances is None
    assert set(m.defaulted) == {"primary.L", "tolerances", "budget.per_region", "budget.total", "seed"}


def test_primary_override_and_explicit_budget(tmp_path):
    m = load(write_manifest(tmp_path, GOOD + "primary: {L: 512}\nbudget: {per_region: 5, total: 50}\n"))
    assert m.primary["L"] == 512
    assert m.budget_per_region == 5
    assert m.budget_total == 50
    assert "primary.L" not in m.defaulted


def test_sweep_defaults_when_omitted(tmp_path):
    body = GOOD.replace("sweep: {L: [1, 7, 512, 4096]}\n", "")
    m = load(write_manifest(tmp_path, body))
    assert m.sweep["L"] == DEFAULT_SWEEP_SIZES
    assert "sweep.L" in m.defaulted


def test_tolerances_explicit_and_per_dtype_defaults(tmp_path):
    m = load(write_manifest(tmp_path, GOOD))
    assert m.tolerance_for("float16") == (1e-2, 2e-2)
    m2 = load(write_manifest(tmp_path, GOOD + "tolerances: {rtol: 1.0e-3, atol: 1.0e-4}\n"))
    assert m2.tolerance_for("float16") == (1e-3, 1e-4)
    with pytest.raises(ManifestError, match="no tolerance default"):
        m.tolerance_for("int32")


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"model": "./missing.py"}, "does not exist"),
        ({"workloads": []}, "non-empty workloads"),
        ({"sweep": {"K": [2]}}, "appears in no workload shape"),
        ({"primary": {"K": 2}}, "appears in no workload shape"),
        ({"budget": {"per_region": 0}}, "positive integer"),
        ({"budget": {"per_region": -3}}, "positive integer"),
        ({"tolerances": {"rtol": 1e-3}}, "rtol, atol"),
        ({"dtype_policy": "bf16"}, "unknown manifest keys"),
    ],
)
def test_bad_manifests_fail_with_pointed_messages(tmp_path, mutation, message):
    import yaml

    base = yaml.safe_load(textwrap.dedent(GOOD))
    base.update(mutation)
    with pytest.raises(ManifestError, match=message):
        load(write_manifest(tmp_path, yaml.safe_dump(base)))


@pytest.mark.parametrize(
    "inputs, message",
    [
        ("[{shape: [], dtype: int32}]", "non-empty list"),
        ("[{shape: [0], dtype: int32}]", ">= 1"),
        ("[{shape: [2], dtype: f16}]", "unknown dtype"),
        ("[{shape: [2], dtype: float16, low: 0, high: 5}]", "integer dtypes only"),
        ("[{shape: [2], dtype: int32, low: 5, high: 5}]", "low < high"),
        ("[{shape: [2], dtype: int32, extra: 1}]", "unknown input keys"),
        ("[{shape: ['not an ident!'], dtype: int32}]", "identifier"),
    ],
)
def test_bad_inputs_fail(tmp_path, inputs, message):
    body = f"""
    model: ./model.py
    workloads:
      - inputs: {inputs}
    """
    with pytest.raises(ManifestError, match=message):
        load(write_manifest(tmp_path, body))


def test_duplicate_workload_names_fail(tmp_path):
    body = """
    model: ./model.py
    workloads:
      - {inputs: [{shape: [2], dtype: int32}], name: a}
      - {inputs: [{shape: [3], dtype: int32}], name: a}
    """
    with pytest.raises(ManifestError, match="unique"):
        load(write_manifest(tmp_path, body))


def test_check_build_passes_on_good_model(tmp_path):
    m = load(write_manifest(tmp_path, GOOD))
    check_build(m)


def test_check_build_missing_build(tmp_path):
    m = load(write_manifest(tmp_path, GOOD, model_body="x = 1\n"))
    with pytest.raises(ManifestError, match="define build\\(\\)"):
        check_build(m)


def test_check_build_required_args(tmp_path):
    m = load(write_manifest(tmp_path, GOOD, model_body="def build(cfg):\n    return cfg\n"))
    with pytest.raises(ManifestError, match="no arguments"):
        check_build(m)


def test_check_build_cwd_relative_read(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"\0")
    model_body = 'data = open("weights.bin", "rb").read()\ndef build():\n    return lambda x: x\n'
    m = load(write_manifest(tmp_path, GOOD, model_body=model_body))
    with pytest.raises(ManifestError, match="relative to __file__"):
        check_build(m)


def test_manifest_is_frozen(tmp_path):
    m = load(write_manifest(tmp_path, GOOD))
    with pytest.raises(AttributeError):
        m.budget_total = 1
    with pytest.raises(TypeError):
        m.sweep["L"] = (2,)
