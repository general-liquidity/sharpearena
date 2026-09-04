"""Deferred resolution: predictions that resolve after the episode ends.

The tests that matter here are the integrity ones. A deferred prediction is only worth
anything if the agent could not have seen the answer when it committed, so most of this
file is about proving that no path from a desk to resolving data exists, rather than about
scoring arithmetic.
"""

from __future__ import annotations

import json
import math

import pytest

from sharpearena.deferred import (
    CATEGORICAL,
    DIRECTION,
    INTERVAL,
    NORMAL,
    POINT,
    PROBABILITY,
    Claim,
    ClaimRejected,
    DeferredDesk,
    DeferredError,
    LeakedResolution,
    Outcome,
    UnresolvedClaim,
    claims_from_json,
    outcomes_from_series,
    resolve_claims,
    score_claim,
    summarize,
)
from sharpearena.forecast_contract import (
    BINARY_LOG,
    CATEGORICAL_BRIER,
    CATEGORICAL_LOG,
    ForecastContract,
    ForecastContractError,
)


def desk_at(bar: int, **kwargs) -> DeferredDesk:
    desk = DeferredDesk(**kwargs)
    desk.tick(bar)
    return desk


def contract(
    kind: str,
    *,
    scoring_rule: str | None = None,
    categories: tuple[str, ...] = (),
    interval_alpha: float | None = None,
    neutral_threshold: float = 0.0,
) -> ForecastContract:
    default = {
        POINT: "point_errors",
        PROBABILITY: "binary_brier",
        CATEGORICAL: "categorical_brier",
        NORMAL: "normal_crps",
        DIRECTION: "direction_accuracy",
        INTERVAL: "interval_score",
    }
    return ForecastContract(
        contract_id=f"contract-{kind}",
        question=f"settle {kind}",
        instrument="ES",
        target="ES.close",
        kind=kind,
        opens_at=2,
        deadline=5,
        resolves_at=10,
        observation_source="frozen-fixture.csv",
        open_definition="close at bar 2",
        close_definition="close at bar 10",
        unit="USD",
        scoring_rule=scoring_rule or default[kind],
        neutral_threshold=neutral_threshold,
        categories=categories,
        interval_alpha=interval_alpha,
    )


# -- the structural guarantee -----------------------------------------------


def test_a_desk_holds_no_data_only_claims_and_a_clock():
    # The leak-free argument is about the object's shape: there is nowhere on a desk for a
    # future value to live, so no agent action can reach one.
    desk = desk_at(10)
    desk.commit("will it rise", DIRECTION, 1.0, horizon=5)
    for name, value in vars(desk).items():
        assert isinstance(
            value, (int, type(None), list)
        ), f"{name} is a {type(value).__name__}, which could hold resolving data"
    assert all(isinstance(c, Claim) for c in vars(desk)["_claims"])


def test_a_claim_carries_nothing_that_was_not_knowable_at_commit_time():
    desk = desk_at(7)
    claim = desk.commit("level in 4 bars", POINT, 101.5, horizon=4, rationale="trend")
    payload = claim.to_dict()
    assert set(payload) == {
        "claim_id",
        "contract",
        "contract_sha256",
        "prediction",
        "committed_at",
        "resolves_at",
        "confidence",
        "rationale",
    }
    assert "outcome" not in payload and "score" not in payload


def test_the_agent_cannot_choose_its_own_commit_time():
    # The desk stamps the bar from the clock the harness advances, so a claim can neither
    # be backdated onto information it did not have nor forward-dated past it.
    desk = desk_at(12)
    with pytest.raises(TypeError):
        desk.commit("x", POINT, 1.0, horizon=2, committed_at=0)
    assert desk.commit("x", POINT, 1.0, horizon=2).committed_at == 12


def test_the_desk_clock_cannot_be_rewound():
    desk = desk_at(5)
    with pytest.raises(ValueError):
        desk.tick(4)


def test_a_claim_cannot_be_committed_before_the_run_starts():
    with pytest.raises(ClaimRejected):
        DeferredDesk().commit("x", POINT, 1.0, horizon=1)


def test_a_desk_exposes_no_score_and_no_resolution_entry_point():
    desk = desk_at(3)
    for attribute in ("resolve", "score", "outcome", "outcomes", "settle"):
        assert not hasattr(desk, attribute), f"a desk must not expose {attribute}"


def test_an_open_claim_cannot_be_queried_for_its_answer():
    desk = desk_at(2)
    desk.commit("x", POINT, 1.0, horizon=10)
    (open_claim,) = desk.open_claims()
    assert not hasattr(open_claim, "outcome")
    assert open_claim.resolves_at == 12


# -- the resolution boundary ------------------------------------------------


def test_a_claim_cannot_be_scored_against_data_available_at_commit_time():
    # The central rule. An outcome observable at or before the commit bar is not a
    # resolution, it is the answer key.
    desk = desk_at(20)
    claim = desk.commit("level in 10", POINT, 100.0, horizon=10)
    with pytest.raises(LeakedResolution):
        resolve_claims([claim], [Outcome(claim.claim_id, 100.0, available_at=20)])
    with pytest.raises(LeakedResolution):
        resolve_claims([claim], [Outcome(claim.claim_id, 100.0, available_at=19)])


def test_a_leaked_resolution_is_never_scored_even_when_it_is_not_raised():
    desk = desk_at(20)
    claim = desk.commit("level in 10", POINT, 100.0, horizon=10)
    report = resolve_claims(
        [claim], [Outcome(claim.claim_id, 100.0, available_at=20)], strict=False
    )
    assert report["resolved"] == []
    assert len(report["rejected"]) == 1
    assert [c["claim_id"] for c in report["pending"]] == [claim.claim_id]
    assert report["summary"]["n_resolved"] == 0


def test_a_claim_cannot_be_settled_before_its_horizon():
    desk = desk_at(5)
    claim = desk.commit("level in 10", POINT, 100.0, horizon=10)
    with pytest.raises(UnresolvedClaim):
        resolve_claims([claim], [Outcome(claim.claim_id, 100.0, available_at=9)])
    settled = resolve_claims([claim], [Outcome(claim.claim_id, 100.0, available_at=15)])
    assert len(settled["resolved"]) == 1


def test_a_claim_with_no_outcome_is_pending_not_wrong():
    desk = desk_at(1)
    claim = desk.commit("earnings beat", PROBABILITY, 0.7, horizon=30)
    report = resolve_claims([claim], [])
    assert report["resolved"] == []
    assert [c["claim_id"] for c in report["pending"]] == [claim.claim_id]
    assert report["summary"]["resolution_rate"] == 0.0


def test_an_outcome_for_an_unknown_claim_is_refused():
    with pytest.raises(DeferredError):
        resolve_claims([], [Outcome("ghost", 1.0, available_at=5)])


def test_duplicate_claims_and_duplicate_outcomes_are_refused():
    claim = desk_at(1).commit("x", POINT, 1.0, horizon=2)
    with pytest.raises(ClaimRejected, match="duplicate claim_id"):
        resolve_claims([claim, claim], [])
    outcome = Outcome(claim.claim_id, 1.0, available_at=3)
    with pytest.raises(DeferredError, match="duplicate outcome"):
        resolve_claims([claim], [outcome, outcome])


def test_non_strict_duplicate_outcome_is_visible_but_never_double_scored():
    claim = desk_at(1).commit("x", POINT, 1.0, horizon=2)
    outcome = Outcome(claim.claim_id, 1.0, available_at=3)
    report = resolve_claims([claim], [outcome, outcome], strict=False)
    assert len(report["resolved"]) == 1
    assert len(report["rejected"]) == 1
    assert report["summary"]["n_resolved"] == 1


def test_series_resolution_stamps_availability_at_the_horizon_bar():
    desk = desk_at(3)
    claim = desk.commit("level in 4", POINT, 0.0, horizon=4)
    series = [float(i) for i in range(20)]
    (outcome,) = outcomes_from_series([claim], series)
    assert outcome.available_at == 7 == claim.resolves_at
    assert outcome.value == 7.0
    # And it survives the boundary check, which is the whole point of stamping it there.
    resolve_claims([claim], [outcome])


def test_series_resolution_skips_claims_that_run_past_the_data():
    desk = desk_at(8)
    claim = desk.commit("level in 100", POINT, 0.0, horizon=100)
    assert outcomes_from_series([claim], [1.0] * 20) == []


def test_series_resolution_supports_change_and_return_references():
    desk = desk_at(2)
    claim = desk.commit("direction in 3", DIRECTION, 1.0, horizon=3)
    series = [10.0, 10.0, 100.0, 110.0, 120.0, 130.0]
    (change,) = outcomes_from_series([claim], series, reference="change")
    (ret,) = outcomes_from_series([claim], series, reference="return")
    assert change.value == pytest.approx(30.0)
    assert ret.value == pytest.approx(0.3)


# -- claims outlive the episode ---------------------------------------------


def test_claims_round_trip_through_json():
    desk = desk_at(4)
    desk.commit("a", POINT, 1.25, horizon=3, confidence=0.8, rationale="because")
    desk.commit("b", INTERVAL, (0.0, 2.0), horizon=6)
    restored = claims_from_json(desk.to_json())
    assert [c.to_dict() for c in restored] == [c.to_dict() for c in desk.claims()]


def test_a_prospective_contract_is_frozen_before_the_prediction():
    desk = desk_at(3)
    frozen = contract(PROBABILITY)
    claim = desk.commit_contract(frozen, 0.7, claim_id="prob-1")
    assert claim.contract.sha256 == claim.to_dict()["contract_sha256"]
    assert claim.resolves_at == 10
    assert claim.horizon == 7


def test_contract_clock_and_scoring_rule_are_closed():
    with pytest.raises(ForecastContractError, match="clock"):
        ForecastContract(
            contract_id="bad",
            question="bad",
            instrument="ES",
            target="close",
            kind=POINT,
            opens_at=5,
            deadline=4,
            resolves_at=6,
            observation_source="fixture",
            open_definition="open",
            close_definition="close",
            unit="USD",
            scoring_rule="point_errors",
        )
    raw = contract(POINT).to_dict()
    raw["surprise"] = True
    with pytest.raises(ForecastContractError, match="unknown=.*surprise"):
        ForecastContract.from_dict(raw)


def test_claim_document_refuses_unknown_fields_bad_hashes_and_duplicates():
    desk = desk_at(4)
    desk.commit("a", POINT, 1.25, horizon=3, claim_id="one")
    original = json.loads(desk.to_json())

    unknown = json.loads(desk.to_json())
    unknown["claims"][0]["surprise"] = True
    with pytest.raises(ClaimRejected, match="unknown=.*surprise"):
        claims_from_json(json.dumps(unknown))

    wrong_hash = json.loads(desk.to_json())
    wrong_hash["claims"][0]["contract_sha256"] = "0" * 64
    with pytest.raises(ClaimRejected, match="contract_sha256"):
        claims_from_json(json.dumps(wrong_hash))

    duplicate = original
    duplicate["claims"].append(dict(duplicate["claims"][0]))
    with pytest.raises(ClaimRejected, match="duplicate claim_id"):
        claims_from_json(json.dumps(duplicate))


def test_claim_document_refuses_inconsistent_resolution_and_invalid_confidence():
    desk = desk_at(4)
    desk.commit("a", POINT, 1.25, horizon=3)
    inconsistent = json.loads(desk.to_json())
    inconsistent["claims"][0]["resolves_at"] += 1
    with pytest.raises(ClaimRejected, match="resolves_at"):
        claims_from_json(json.dumps(inconsistent))

    confidence = json.loads(desk.to_json())
    confidence["claims"][0]["confidence"] = 1.5
    with pytest.raises(ClaimRejected, match="confidence"):
        claims_from_json(json.dumps(confidence))


def test_a_serialized_claim_carries_no_answer():
    desk = desk_at(4)
    desk.commit("a", POINT, 1.25, horizon=3)
    document = json.loads(desk.to_json())
    assert document["schema_version"] == "sharpearena.deferred-claims.v1"
    payload = document["claims"][0]
    assert "outcome" not in payload
    assert "score" not in payload


# -- scoring ----------------------------------------------------------------


def test_point_claims_score_on_error():
    claim = desk_at(0).commit("q", POINT, 10.0, horizon=1)
    score = score_claim(claim, 12.0)
    assert score["error"] == pytest.approx(2.0)
    assert score["squared_error"] == pytest.approx(4.0)


def test_probability_claims_score_on_brier():
    claim = desk_at(0).commit("q", PROBABILITY, 0.75, horizon=1)
    assert score_claim(claim, 1.0)["brier"] == pytest.approx(0.0625)
    assert score_claim(claim, 0.0)["brier"] == pytest.approx(0.5625)
    with pytest.raises(ClaimRejected):
        score_claim(claim, 0.5)


def test_binary_log_score_is_supported_without_clipping_endpoints():
    desk = desk_at(3)
    claim = desk.commit_contract(contract(PROBABILITY, scoring_rule=BINARY_LOG), 0.8)
    assert score_claim(claim, 1.0)["log_loss"] == pytest.approx(-math.log(0.8))
    with pytest.raises(ClaimRejected, match="strictly inside"):
        desk.commit_contract(contract(PROBABILITY, scoring_rule=BINARY_LOG), 1.0)


def test_categorical_brier_and_log_scores_use_the_full_probability_vector():
    categories = ("bearish", "neutral", "bullish")
    desk = desk_at(3)
    brier = desk.commit_contract(
        contract(CATEGORICAL, scoring_rule=CATEGORICAL_BRIER, categories=categories),
        (0.2, 0.3, 0.5),
        claim_id="cat-brier",
    )
    assert score_claim(brier, "bullish")["brier"] == pytest.approx(
        0.2**2 + 0.3**2 + (0.5 - 1.0) ** 2
    )
    log = desk.commit_contract(
        contract(CATEGORICAL, scoring_rule=CATEGORICAL_LOG, categories=categories),
        (0.2, 0.3, 0.5),
        claim_id="cat-log",
    )
    assert score_claim(log, "neutral")["log_loss"] == pytest.approx(-math.log(0.3))
    with pytest.raises(ClaimRejected, match="sum to 1"):
        desk.commit_contract(
            contract(CATEGORICAL, categories=categories),
            (0.2, 0.3, 0.4),
        )


def test_normal_forecast_uses_closed_form_crps():
    desk = desk_at(3)
    claim = desk.commit_contract(contract(NORMAL), (10.0, 2.0))
    score = score_claim(claim, 10.0)
    expected = 2.0 * (math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi))
    assert score["crps"] == pytest.approx(expected)
    assert score["z"] == 0.0


def test_interval_score_penalizes_misses_by_the_precommitted_alpha():
    desk = desk_at(3)
    claim = desk.commit_contract(contract(INTERVAL, interval_alpha=0.1), (8.0, 12.0))
    assert score_claim(claim, 10.0)["interval_score"] == pytest.approx(4.0)
    assert score_claim(claim, 14.0)["interval_score"] == pytest.approx(44.0)


def test_direction_claims_score_on_the_sign_of_the_move():
    claim = desk_at(0).commit("q", DIRECTION, 1.0, horizon=1)
    assert score_claim(claim, 0.4)["correct"] is True
    assert score_claim(claim, -0.4)["correct"] is False
    assert score_claim(claim, 0.0)["correct"] is False


def test_direction_claim_uses_its_frozen_neutral_threshold():
    desk = desk_at(3)
    claim = desk.commit_contract(
        contract(DIRECTION, neutral_threshold=0.01),
        1.0,
    )
    assert score_claim(claim, 0.009)["realized_direction"] == 0.0
    assert score_claim(claim, 0.011)["realized_direction"] == 1.0


def test_interval_claims_score_on_coverage():
    claim = desk_at(0).commit("q", INTERVAL, (1.0, 3.0), horizon=1)
    assert score_claim(claim, 2.0)["covered"] is True
    assert score_claim(claim, 3.5)["covered"] is False
    assert score_claim(claim, 2.0)["width"] == pytest.approx(2.0)


def test_the_summary_reports_the_resolution_rate_alongside_the_scores():
    desk = desk_at(0)
    a = desk.commit("a", DIRECTION, 1.0, horizon=1)
    b = desk.commit("b", DIRECTION, -1.0, horizon=1)
    desk.commit("c", POINT, 5.0, horizon=99)
    report = resolve_claims(
        desk.claims(),
        [Outcome(a.claim_id, 1.0, 1), Outcome(b.claim_id, 1.0, 1)],
    )
    summary = report["summary"]
    assert summary["n_committed"] == 3
    assert summary["n_resolved"] == 2
    assert summary["resolution_rate"] == pytest.approx(2 / 3)
    assert summary["by_kind"][DIRECTION]["accuracy"] == pytest.approx(0.5)


def test_the_summary_keeps_the_kinds_apart():
    resolved = [
        {"kind": POINT, "score": {"abs_error": 2.0, "squared_error": 4.0}},
        {
            "kind": INTERVAL,
            "score": {"covered": True, "width": 1.0, "interval_score": 1.0},
        },
    ]
    summary = summarize(resolved, n_committed=2)
    assert summary["by_kind"][POINT]["mae"] == pytest.approx(2.0)
    assert summary["by_kind"][INTERVAL]["coverage"] == pytest.approx(1.0)
    assert "coverage" not in summary["by_kind"][POINT]


# -- claim validation -------------------------------------------------------


def test_a_zero_horizon_is_not_a_prediction():
    with pytest.raises(ClaimRejected):
        desk_at(3).commit("q", POINT, 1.0, horizon=0)


def test_malformed_claims_are_rejected_at_commit():
    desk = desk_at(3)
    with pytest.raises(ClaimRejected):
        desk.commit("q", "vibes", 1.0, horizon=2)
    with pytest.raises(ClaimRejected):
        desk.commit("q", PROBABILITY, 1.4, horizon=2)
    with pytest.raises(ClaimRejected):
        desk.commit("q", DIRECTION, 0.0, horizon=2)
    with pytest.raises(ClaimRejected):
        desk.commit("q", INTERVAL, (3.0, 1.0), horizon=2)
    with pytest.raises(ClaimRejected):
        desk.commit("q", INTERVAL, 1.0, horizon=2)
    with pytest.raises(ClaimRejected):
        desk.commit("q", POINT, 1.0, horizon=2, confidence=1.5)


def test_duplicate_claim_ids_are_rejected():
    desk = desk_at(3)
    desk.commit("q", POINT, 1.0, horizon=2, claim_id="dup")
    with pytest.raises(ClaimRejected):
        desk.commit("q", POINT, 2.0, horizon=2, claim_id="dup")


def test_the_open_claim_budget_is_enforced():
    desk = desk_at(0, max_open=1)
    desk.commit("a", POINT, 1.0, horizon=50)
    with pytest.raises(ClaimRejected):
        desk.commit("b", POINT, 1.0, horizon=50)


# -- end to end over a real episode -----------------------------------------


def test_a_claim_committed_mid_episode_resolves_on_bars_the_agent_never_saw():
    from sharpearena.gym import SharpeArenaEnv

    env = SharpeArenaEnv(n_symbols=1, n_days=200, seed=4)
    obs, _ = env.reset()

    desk = DeferredDesk()
    cutoff = 20
    seen: list[float] = [float(obs["closes"][0])]
    flat = env.action_space.sample() * 0.0

    # The agent's episode: it observes bars up to `cutoff` and commits there. The desk
    # clock is advanced by the harness after each observation, so it always equals the
    # index of the last bar the agent has actually seen.
    for bar in range(1, cutoff + 1):
        obs, _reward, terminated, truncated, _info = env.step(flat)
        seen.append(float(obs["closes"][0]))
        desk.tick(bar)
        if terminated or truncated:
            break
    claim = desk.commit("level 15 bars out", POINT, seen[-1], horizon=15)
    committed_at = claim.committed_at
    assert committed_at == len(seen) - 1 == cutoff

    # The episode ends for the agent here. The harness keeps walking the same path to
    # collect the resolving bars, which are strictly outside everything the agent saw.
    full = list(seen)
    while True:
        obs, _reward, terminated, truncated, _info = env.step(flat)
        full.append(float(obs["closes"][0]))
        if terminated or truncated:
            break
    env.close()

    assert len(full) > claim.resolves_at
    assert full[: len(seen)] == seen

    (outcome,) = outcomes_from_series([claim], full)
    assert outcome.available_at > committed_at
    assert outcome.available_at >= len(seen), "the resolving bar is past the agent's episode"

    report = resolve_claims([claim], [outcome])
    assert len(report["resolved"]) == 1
    assert report["summary"]["resolution_rate"] == 1.0

    # And the same claim cannot be settled against anything the agent had.
    with pytest.raises(LeakedResolution):
        resolve_claims([claim], [Outcome(claim.claim_id, seen[-1], committed_at)])
