"""Evidence boundaries at the Arena to Bench artifact bridge, without model calls."""

import copy
import json

import pytest
from sharpearena import bench_bridge as bridge
from test_local_agents import FixedModel, _plan
from sharpearena.local_agents import EvidenceJournal, LocalFieldRunner


@pytest.fixture
def field_records(tmp_path):
    path = tmp_path / "source.jsonl"
    result = LocalFieldRunner(FixedModel()).run(
        _plan(repetitions=1), EvidenceJournal(path)
    )
    assert result["failed"] == 0
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _compile(tmp_path, records):
    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return bridge.compile_benchmark_evidence([path], tmp_path / "compiled")


@pytest.mark.parametrize(
    "cost", [None, True, "0", -1, float("nan"), float("inf"), 10**400]
)
def test_bridge_refuses_invalid_recorded_cost(tmp_path, field_records, cost):
    for record in field_records:
        record["cost"] = cost
    with pytest.raises(bridge.BenchBridgeError, match="run cost"):
        _compile(tmp_path, field_records)
    assert not (tmp_path / "compiled").exists()


def test_bridge_does_not_invent_zero_for_missing_cost(tmp_path, field_records):
    for record in field_records:
        record.pop("cost")
    with pytest.raises(bridge.BenchBridgeError, match="run cost"):
        _compile(tmp_path, field_records)
    assert not (tmp_path / "compiled").exists()


@pytest.mark.parametrize(
    "periods", [None, True, "252", 0, -1, float("nan"), float("inf"), 10**400]
)
def test_bridge_refuses_invalid_annualization(tmp_path, field_records, periods):
    for record in field_records:
        record["dataset"]["periods_per_year"] = periods
    with pytest.raises(bridge.BenchBridgeError, match="periods_per_year"):
        _compile(tmp_path, field_records)
    assert not (tmp_path / "compiled").exists()


def test_bridge_retains_explicit_zero_and_declared_annualization(
    tmp_path, field_records
):
    originals = copy.deepcopy(field_records)
    manifest = _compile(tmp_path, field_records)
    output = manifest["outputs"][0]
    assert output["periods_per_year"] == 252.0
    submissions = json.loads(open(output["submissions_path"], encoding="utf-8").read())
    assert all(run["cost"] == 0.0 for entry in submissions for run in entry["runs"])
    assert [run["returns"] for run in submissions[0]["runs"]] == [
        r["returns"] for r in originals
    ]
    assert field_records == originals
