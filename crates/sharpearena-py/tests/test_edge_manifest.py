"""Edge manifests: closed schema, typed thresholds, and no vacuous health."""

from __future__ import annotations

import hashlib
import json

import pytest
from sharpearena.edge_manifest import (
    CandidateValidation,
    EdgeManifestError,
    EdgeManifestLedger,
    bind_idea_source,
    monitor_edge,
    parse_edge_manifest,
    parse_threshold,
    read_manifest_ledger,
)


def _manifest_payload(**overrides):
    payload = {
        "hypothesis": "Funding-rate extremes trap leveraged longs into forced unwinds.",
        "mechanism": "Perp longs pay to hold; at extreme funding the marginal long is "
        "financing-constrained and liquidates into thin books.",
        "regimes": ["high_funding", "trending"],
        "instruments": ["BTCUSDT"],
        "invariants": [
            {
                "condition_id": "net-edge-positive",
                "metric": "net_edge",
                "comparator": "gt",
                "threshold": {"value": 2.0, "unit": "basis_points"},
                "description": "Edge must clear costs by two basis points.",
            }
        ],
        "kill_conditions": [
            {
                "condition_id": "funding-normalized",
                "metric": "funding_percentile",
                "comparator": "lt",
                "threshold": {"value": 0.5, "unit": "fraction"},
                "description": "Funding back to the middle of its distribution.",
            }
        ],
        "verification_plan": {
            "selection_metric": "deflated_sharpe",
            "selection_split": "validation",
            "confirmation_split": "test",
            "minimum_observations": 60,
        },
    }
    payload.update(overrides)
    return payload


def _candidate(**overrides):
    candidate = {
        "id": "funding-trap",
        "gross_target": 0.4,
        "edge_manifest": _manifest_payload(),
    }
    candidate.update(overrides)
    return candidate


def test_valid_manifest_parses_and_hashes_independently_of_key_order():
    payload = _manifest_payload()
    first = parse_edge_manifest(payload)
    shuffled = json.loads(json.dumps(dict(reversed(list(payload.items())))))
    second = parse_edge_manifest(shuffled)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.invariants[0].threshold.unit == "basis_points"
    assert first.invariants[0].threshold.value == 2.0


def test_a_misspelt_section_invalidates_rather_than_defaulting():
    payload = _manifest_payload()
    payload["kill_condition"] = payload.pop("kill_conditions")
    with pytest.raises(EdgeManifestError) as error:
        parse_edge_manifest(payload)
    message = str(error.value)
    assert "unknown fields" in message and "kill_condition" in message
    assert "kill_conditions" in message


def test_an_empty_invariant_list_is_refused_because_it_can_never_fail():
    with pytest.raises(EdgeManifestError, match="at least 1 condition"):
        parse_edge_manifest(_manifest_payload(invariants=[]))


def test_a_percent_string_threshold_is_a_parse_error_not_a_silent_nan():
    with pytest.raises(EdgeManifestError, match="carries no unit"):
        parse_threshold("2%", "threshold")
    with pytest.raises(EdgeManifestError, match="put the unit in the 'unit' field"):
        parse_threshold({"value": "2%", "unit": "percent"}, "threshold")


def test_a_bare_number_threshold_is_refused():
    with pytest.raises(EdgeManifestError, match="carries no unit"):
        parse_threshold(2.0, "threshold")


def test_a_unit_outside_the_closed_set_is_refused():
    with pytest.raises(EdgeManifestError, match="outside the closed unit set"):
        parse_threshold({"value": 2.0, "unit": "bps"}, "threshold")


def test_monitoring_the_selection_sample_is_refused():
    manifest = parse_edge_manifest(_manifest_payload())
    with pytest.raises(EdgeManifestError, match="not an out-of-selection sample"):
        monitor_edge(manifest, {"net_edge": 4.0}, sample="selection")


def test_absent_metrics_produce_indeterminate_not_healthy():
    manifest = parse_edge_manifest(_manifest_payload())
    report = monitor_edge(manifest, {}, sample="test")
    assert report.health == "indeterminate"
    assert report.resolved_invariants == 0
    assert sorted(report.unresolved_metrics) == ["funding_percentile", "net_edge"]
    assert "could not be resolved" in report.reason


def test_a_mis_united_observation_does_not_compare_against_the_threshold():
    manifest = parse_edge_manifest(_manifest_payload())
    report = monitor_edge(
        manifest,
        {
            "net_edge": {"value": 0.0004, "unit": "fraction"},
            "funding_percentile": {"value": 0.9, "unit": "fraction"},
        },
        sample="test",
    )
    assert report.health == "indeterminate"
    assert report.unresolved_metrics == ("net_edge",)


def test_healthy_requires_every_condition_to_resolve():
    manifest = parse_edge_manifest(_manifest_payload())
    report = monitor_edge(
        manifest,
        {
            "net_edge": {"value": 4.0, "unit": "basis_points"},
            "funding_percentile": {"value": 0.9, "unit": "fraction"},
        },
        sample="forward",
    )
    assert report.health == "healthy"
    assert report.resolved_invariants == 1
    assert "1 invariant(s) hold" in report.reason


def test_a_fired_kill_condition_outranks_a_violated_invariant():
    manifest = parse_edge_manifest(_manifest_payload())
    report = monitor_edge(
        manifest,
        {"net_edge": -1.0, "funding_percentile": 0.1},
        sample="test",
    )
    assert report.health == "retired"
    assert "funding-normalized" in report.reason


def test_a_violated_invariant_without_a_kill_reports_violated():
    manifest = parse_edge_manifest(_manifest_payload())
    report = monitor_edge(
        manifest,
        {"net_edge": -1.0, "funding_percentile": 0.9},
        sample="test",
    )
    assert report.health == "violated"
    assert "net-edge-positive" in report.reason


def test_membership_comparators_require_categorical_units():
    payload = _manifest_payload()
    payload["invariants"] = [
        {
            "condition_id": "regime-holds",
            "metric": "regime",
            "comparator": "in",
            "threshold": {"value": ["trending", "high_funding"], "unit": "categorical"},
            "description": "Only claimed in these regimes.",
        }
    ]
    manifest = parse_edge_manifest(payload)
    holds = monitor_edge(
        manifest, {"regime": "trending", "funding_percentile": 0.9}, sample="test"
    )
    breaks = monitor_edge(
        manifest, {"regime": "chop", "funding_percentile": 0.9}, sample="test"
    )
    assert holds.health == "healthy"
    assert breaks.health == "violated"

    payload["invariants"][0]["threshold"]["unit"] = "count"
    with pytest.raises(EdgeManifestError, match="unit 'categorical'"):
        parse_edge_manifest(payload)


def test_ledger_counts_every_proposal_before_validation_and_deduplication(tmp_path):
    path = tmp_path / "manifests.jsonl"
    ledger = EdgeManifestLedger(
        path, model_digest="sha256:model", split_plan_sha256="sha256:split"
    )
    broken = _candidate(id="broken")
    broken["edge_manifest"].pop("mechanism")
    duplicate = _candidate(id="duplicate-of-first")
    ledger.record_pool(
        [
            _candidate(),
            broken,
            duplicate,
            {"id": "no-manifest", "gross_target": 0.2},
        ]
    )
    summary = ledger.summary()
    assert summary["observed_trials"] == 4
    assert summary["invalid"] == 2
    assert summary["duplicates"] == 1
    assert summary["selectable"] == 1

    rows = read_manifest_ledger(path)
    assert [row["trial_ordinal"] for row in rows] == [0, 1, 2, 3]
    assert rows[1]["manifest"] is None
    assert "mechanism" in rows[1]["invalid_reason"]
    assert rows[2]["duplicate_of_ordinal"] == 0
    assert rows[3]["invalid_reason"].endswith("'edge_manifest' field")


def test_a_duplicate_is_recorded_rather_than_dropped(tmp_path):
    ledger = EdgeManifestLedger(
        tmp_path / "m.jsonl",
        model_digest="sha256:model",
        split_plan_sha256="sha256:split",
    )
    first = ledger.record(_candidate(id="a"))
    second = ledger.record(_candidate(id="b"))
    assert first.is_selectable is True
    assert second.is_selectable is False
    assert second.duplicate_of_ordinal == 0
    assert ledger.observed_trials == 2
    assert len(ledger.selectable()) == 1


def test_strategy_validation_runs_after_manifest_parse_and_before_deduplication():
    ledger = EdgeManifestLedger(
        None,
        model_digest="sha256:model",
        split_plan_sha256="sha256:split",
    )
    calls = []

    def validator(candidate):
        calls.append(candidate["id"])
        if candidate["id"] == "broken-strategy":
            raise ValueError("strategy DSL is invalid")
        return "same-semantic-strategy"

    malformed_manifest = _candidate(id="broken-manifest")
    malformed_manifest["edge_manifest"].pop("mechanism")
    records = ledger.record_pool(
        [
            malformed_manifest,
            _candidate(id="broken-strategy"),
            _candidate(id="first-valid"),
            _candidate(id="duplicate-valid"),
        ],
        candidate_validator=validator,
    )
    assert calls == ["broken-strategy", "first-valid", "duplicate-valid"]
    assert records[0].invalid_reason and "mechanism" in records[0].invalid_reason
    assert records[1].invalid_reason == "strategy DSL is invalid"
    assert records[2].is_selectable is True
    assert records[3].duplicate_of_ordinal == 2
    assert ledger.observed_trials == 4


def test_binding_hash_ties_the_manifest_to_the_model_and_split_plan(tmp_path):
    left = EdgeManifestLedger(
        None, model_digest="sha256:model-a", split_plan_sha256="sha256:split"
    ).record(_candidate())
    right = EdgeManifestLedger(
        None, model_digest="sha256:model-b", split_plan_sha256="sha256:split"
    ).record(_candidate())
    other_split = EdgeManifestLedger(
        None, model_digest="sha256:model-a", split_plan_sha256="sha256:other"
    ).record(_candidate())
    assert left.manifest_sha256 == right.manifest_sha256
    assert (
        len({left.binding_sha256, right.binding_sha256, other_split.binding_sha256})
        == 3
    )


def test_lineage_resolves_only_prior_candidates_and_plan_bound_sources():
    source = bind_idea_source(
        "exact source bytes",
        source_type="paper",
        url_or_doi="doi:10.0000/example",
        authors=("A. Researcher",),
        license="CC-BY-4.0",
    )
    identity = {
        "generator": "fixture",
        "digest": "sha256:model",
        "runtime": "test",
        "runtime_version": "1",
    }
    ledger = EdgeManifestLedger(
        None,
        model_digest="sha256:model",
        split_plan_sha256="sha256:split",
        generator_identity=identity,
        idea_provenance=(source,),
    )

    def validate(candidate):
        return CandidateValidation(
            semantic_fingerprint=candidate["id"],
            family_preimage={"signal": "momentum", "scope": "BTC"},
            candidate_id=candidate["id"],
        )

    first = ledger.record(
        _candidate(
            id="parent",
            lineage={
                "parent_candidate_ids": [],
                "idea_source_digests": [source.source_digest],
            },
        ),
        candidate_validator=validate,
    )
    child = ledger.record(
        _candidate(
            id="child",
            gross_target=0.3,
            lineage={
                "parent_candidate_ids": ["parent"],
                "idea_source_digests": [source.source_digest],
            },
        ),
        candidate_validator=validate,
    )

    assert child.parent_candidate_digests == (first.raw_candidate_sha256,)
    assert child.idea_provenance == (source,)
    assert child.generator_identity == identity
    assert child.family_digest == first.family_digest
    assert child.lineage_status == "declared"
    expected_lineage_binding = _canonical_sha256_for_test(
        {
            "family_digest": child.family_digest,
            "generator_identity_sha256": child.generator_identity_sha256,
            "idea_source_digests": [source.source_digest],
            "parent_candidate_digests": [first.raw_candidate_sha256],
            "raw_candidate_sha256": child.raw_candidate_sha256,
        }
    )
    assert child.lineage_binding_sha256 == expected_lineage_binding
    assert ledger.summary()["observed_trials"] == 2
    assert ledger.summary()["family_count"] == 1
    assert ledger.summary()["family_grouping_role"].startswith("diagnostic-only")


def _canonical_sha256_for_test(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def test_lineage_rejects_forward_parents_and_unbound_source_digests_but_counts_trials():
    unknown_source = "sha256:" + "a" * 64
    ledger = EdgeManifestLedger(
        None, model_digest="sha256:model", split_plan_sha256="sha256:split"
    )

    def validate(candidate):
        return CandidateValidation(candidate["id"], {"family": "x"}, candidate["id"])

    future_parent = ledger.record(
        _candidate(
            id="first",
            lineage={
                "parent_candidate_ids": ["future"],
                "idea_source_digests": [],
            },
        ),
        candidate_validator=validate,
    )
    unbound_source = ledger.record(
        _candidate(
            id="second",
            lineage={
                "parent_candidate_ids": [],
                "idea_source_digests": [unknown_source],
            },
        ),
        candidate_validator=validate,
    )

    assert "earlier valid proposals" in (future_parent.invalid_reason or "")
    assert "plan-bound sources" in (unbound_source.invalid_reason or "")
    assert ledger.observed_trials == 2
    assert not ledger.selectable()


def test_undeclared_lineage_is_backward_compatible_and_explicitly_labelled():
    record = EdgeManifestLedger(
        None, model_digest="sha256:model", split_plan_sha256="sha256:split"
    ).record(_candidate())
    assert record.is_selectable is True
    assert record.declared_lineage is None
    assert record.lineage_status == "host-derived-unreferenced"
    assert record.idea_provenance == ()


def test_generator_identity_digest_must_match_the_ledger_model():
    with pytest.raises(EdgeManifestError, match="must equal"):
        EdgeManifestLedger(
            None,
            model_digest="sha256:model",
            split_plan_sha256="sha256:split",
            generator_identity={"digest": "sha256:different"},
        )


def test_a_ledger_without_a_model_digest_cannot_be_opened():
    with pytest.raises(EdgeManifestError, match="model digest"):
        EdgeManifestLedger(None, model_digest="", split_plan_sha256="sha256:split")


def test_strict_ledger_read_rejects_a_corrupt_row(tmp_path):
    path = tmp_path / "m.jsonl"
    ledger = EdgeManifestLedger(
        path, model_digest="sha256:model", split_plan_sha256="sha256:split"
    )
    ledger.record(_candidate())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(EdgeManifestError, match="is not JSON"):
        read_manifest_ledger(path)


def test_strict_ledger_read_recomputes_v2_lineage_bindings(tmp_path):
    path = tmp_path / "m.jsonl"
    ledger = EdgeManifestLedger(
        path,
        model_digest="sha256:model",
        split_plan_sha256="sha256:split",
    )
    ledger.record(_candidate())
    [row] = read_manifest_ledger(path)
    row["family_preimage"]["tampered"] = True
    path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EdgeManifestError, match="stale family_digest"):
        read_manifest_ledger(path)


def test_verification_plan_must_separate_selection_from_confirmation():
    payload = _manifest_payload()
    payload["verification_plan"]["confirmation_split"] = "validation"
    with pytest.raises(EdgeManifestError, match="must be different"):
        parse_edge_manifest(payload)


def test_a_condition_id_may_not_be_both_invariant_and_kill():
    payload = _manifest_payload()
    payload["kill_conditions"][0]["condition_id"] = "net-edge-positive"
    with pytest.raises(EdgeManifestError, match="shared between invariants"):
        parse_edge_manifest(payload)
