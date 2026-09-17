"""report.json: the fields a reader needs to trust a hypothesis row, and a
stranded list that does not repeat one reason a hundred times."""

import json

from autotuner.report import LEGEND, Report


def test_hypothesis_rows_carry_text_and_the_ship_clock_numbers():
    r = Report()
    r.add_hypothesis(hypothesis_id="h1", region="abc", kind="retile",
                     hypothesis_text="tile K by 8", assoc_tag="preserving", parent="scaffold",
                     kernel="rabc_h1", verdict="correct_slower", failed_gate=None,
                     region_ms=1.2, library_ms=1.0, win_ms=-0.2, sigma_ms=0.01)
    row = r.hypotheses[0]
    assert row["hypothesis"] == "tile K by 8"
    assert (row["kernel"], row["library_ms"], row["win_ms"], row["sigma_ms"]) == ("rabc_h1", 1.0, -0.2, 0.01)
    for key in ("region_ms", "library_ms", "win_ms", "sigma_ms", "p", "s", "s_max", "bound"):
        assert key in LEGEND


def test_region_row_tallies_its_hypotheses():
    r = Report()
    for i, verdict in enumerate(["failed", "correct_slower", "correct_slower"]):
        r.add_hypothesis(hypothesis_id=f"h{i}", region="abc", kind="fix", parent=None,
                         verdict=verdict, failed_gate="static" if verdict == "failed" else None,
                         region_ms=None)
    r.add_hypothesis(hypothesis_id="other", region="xyz", kind="fix", parent=None,
                     verdict="shipped", failed_gate=None, region_ms=None)
    r.add_region(fingerprint="abc", ops=["mx.add"], copies=2, workloads=["w"], p={"w": 0.1},
                 t_orig_ms={"w": 2.0}, bound="memory", s_max=1.5, hypotheses=3, head_ms=0.9)
    row = r.regions[0]
    assert row["outcomes"] == {"failed": 1, "correct_slower": 2}
    assert (row["hypotheses"], row["head_ms"]) == (3, 0.9)


def test_stranded_regions_group_by_reason():
    r = Report()
    for i in range(3):
        r.stranded.append({"fingerprint": f"f{i}", "ops": ["mx.exp"], "reason": "same reason"})
    r.stranded.append({"fingerprint": "g", "ops": ["mx.sin"], "reason": "another"})
    grouped = r.to_dict()["stranded"]
    assert [(g["reason"], g["count"]) for g in grouped] == [("same reason", 3), ("another", 1)]
    assert grouped[0]["regions"][1] == {"fingerprint": "f1", "ops": ["mx.exp"]}


def test_failed_numeric_checks_still_produce_valid_json(tmp_path):
    r = Report()
    r.final = {"passed": False, "checks": [{"max_abs": float("inf"),
                                          "cosine": float("nan")}]}
    r.accepted = [{"timings": {"main": {"sigma_ms": float("inf")}}}]
    path = tmp_path / "report.json"
    r.write(path)

    def reject_constant(value):
        raise AssertionError(f"invalid JSON constant: {value}")

    saved = json.loads(path.read_text(), parse_constant=reject_constant)
    assert saved["final"]["checks"] == [{"max_abs": "inf", "cosine": "nan"}]
    assert saved["accepted"][0]["timings"]["main"]["sigma_ms"] == "inf"
