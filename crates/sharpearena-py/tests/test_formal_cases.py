"""The shared conformance cases in formal/cases/forecast.json, checked against the production ledger.

The Lean model in formal/SharpeArenaFormal evaluates the same file against its classify,
effectiveClaim, rebind and brierNumerator while it builds (SharpeArenaFormal/Cases.lean). These
tests run each case through ForecastLedger.submit, ForecastLedger.effective_claims and score_claim.
One table, two implementations, so a change to either that the other does not share fails a build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sharpearena.deferred import PROBABILITY, Claim, score_claim
from sharpearena.forecast_contract import BINARY_BRIER, ForecastContract
from sharpearena.forecast_evidence import ForecastEvidenceError, ForecastLedger, ForecastRunIdentity

CASES = json.loads(
    (Path(__file__).resolve().parents[3] / "formal" / "cases" / "forecast.json").read_text(
        encoding="utf-8"
    )
)


def identity() -> ForecastRunIdentity:
    return ForecastRunIdentity(
        agent_id="agent-a",
        model_id="model-a",
        model_sha256="a" * 64,
        scaffold_id="scaffold-a",
        scaffold_sha256="b" * 64,
        prompt_sha256="c" * 64,
        operator_id="operator-a",
        config_sha256="d" * 64,
    )


def contract(contract_id: str = "es-up", *, opens_at: int = 10, deadline: int = 20) -> ForecastContract:
    return ForecastContract(
        contract_id=contract_id,
        question=f"Will ES finish above the frozen threshold? ({contract_id})",
        instrument="ES",
        target="ES.close_above_threshold",
        kind=PROBABILITY,
        opens_at=opens_at,
        deadline=deadline,
        resolves_at=deadline + 10,
        observation_source="fixture:es-daily-v1",
        open_definition="official close at the opening bar",
        close_definition="official close at the resolution bar",
        unit="binary",
        scoring_rule=BINARY_BRIER,
        neutral_threshold=0.001,
        boundary_ownership="threshold is false",
        missing_data_policy="cancel",
        fallback_policy="retry once, then cancel",
    )


def submit(ledger: ForecastLedger, claim_id: str, bound: ForecastContract, submitted_at: int, ordinal: int):
    return ledger.submit(
        claim_id=claim_id,
        contract=bound,
        # The prediction carries the ordinal, so the effective claim says which revision it is.
        prediction=(ordinal + 1) / 10,
        confidence=0.5,
        rationale="formal case",
        submitted_at=submitted_at,
        idempotency_key=f"{claim_id}-{ordinal}",
        revision_reason=None if ordinal == 0 else "formal case revision",
    )


def test_the_case_table_is_declared_and_non_empty():
    assert CASES["schema_version"] == "sharpe.formal-cases.v1"
    for key in ("classify", "effective", "rebind", "brier"):
        assert CASES[key], key


@pytest.mark.parametrize("case", CASES["classify"])
def test_submission_status_matches_the_model_classify(case):
    ledger = ForecastLedger(identity())
    bound = contract(opens_at=case["opens_at"], deadline=case["deadline"])
    revision = submit(ledger, "claim", bound, case["submitted_at"], 0)
    assert revision.status == case["status"]


@pytest.mark.parametrize("case", CASES["effective"], ids=lambda case: case["name"])
def test_effective_claims_match_the_model_effective_claim(case):
    ledger = ForecastLedger(identity())
    bound = contract(opens_at=case["opens_at"], deadline=case["deadline"])
    seen: dict[str, int] = {}
    for entry in case["history"]:
        ordinal = seen.get(entry["claim"], 0)
        seen[entry["claim"]] = ordinal + 1
        revision = submit(ledger, entry["claim"], bound, entry["submitted_at"], ordinal)
        assert revision.ordinal == ordinal
    effective = {
        claim.claim_id: round(claim.prediction[0] * 10) - 1 for claim in ledger.effective_claims()
    }
    for claim_id, expected in case["effective"].items():
        assert effective.get(claim_id) == expected, claim_id


@pytest.mark.parametrize("case", CASES["rebind"])
def test_claim_binding_matches_the_model_rebind(case):
    ledger = ForecastLedger(identity())
    submit(ledger, "claim", contract(case["prior"]), 12, 0)
    if case["accepted"]:
        submit(ledger, "claim", contract(case["candidate"]), 13, 1)
    else:
        with pytest.raises(ForecastEvidenceError, match="already bound"):
            submit(ledger, "claim", contract(case["candidate"]), 13, 1)


@pytest.mark.parametrize("case", CASES["brier"])
def test_scored_brier_matches_the_model_numerator(case):
    probability, truth, scale = case["probability"], case["truth"], case["scale"]
    checked = Claim(
        claim_id="brier-case",
        contract=contract(),
        prediction=(probability / scale,),
        committed_at=12,
        confidence=0.5,
        rationale="formal case",
    )
    loss_zero = score_claim(checked, 0.0)["brier"]
    loss_one = score_claim(checked, 1.0)["brier"]
    expected = (truth / scale) * loss_one + (1.0 - truth / scale) * loss_zero
    assert expected * scale**3 == pytest.approx(case["numerator"], abs=1e-6 * scale**3)
