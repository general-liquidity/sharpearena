"""Ecological population dynamics: selection, innovation, and perturbation.

The mechanisms are tested against injected payoff functions rather than a live market, so
each one is isolated: a selection test fails because selection is wrong, not because the
market moved. Two smoke tests at the end run the whole loop over the real shared-book and
competition envs to prove the wiring holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from sharpearena.ecology import (
    COLLAPSED,
    DOMINANT,
    Shock,
    Species,
    allocate_seats,
    baseline_species,
    classify_outcomes,
    competition_payoffs,
    detect_coalitions,
    liquidity_shocks,
    market_payoffs,
    mutating_innovator,
    population_table,
    regime_shocks,
    replicator_step,
    run_ecology,
    steady_shocks,
)


class ParamPolicy:
    """A do-nothing policy that carries its species parameters, so an injected payoff
    function can attribute a seat's fitness without inspecting the market."""

    def __init__(self, **params: float) -> None:
        self.params = params

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.zeros((n,), dtype=np.float32)


def edge_payoffs(policies, *, generation, seed, shock):
    """Each seat earns its species' ``edge``. The simplest possible selection pressure."""
    return [float(p.params["edge"]) for p in policies]


# -- seat allocation --------------------------------------------------------


def test_seats_sum_to_the_field_and_track_the_shares():
    seats = allocate_seats([0.5, 0.3, 0.2], 10)
    assert seats.sum() == 10
    assert list(seats) == [5, 3, 2]


def test_seat_allocation_is_deterministic_under_ties():
    a = allocate_seats([1 / 3, 1 / 3, 1 / 3], 10)
    b = allocate_seats([1 / 3, 1 / 3, 1 / 3], 10)
    assert list(a) == list(b)
    assert a.sum() == 10
    # Largest remainder with an index tie-break gives the leftover seat to the lower index.
    assert list(a) == [4, 3, 3]


def test_every_live_species_gets_a_seat_when_the_field_is_wide_enough():
    # A species on a sliver of the population still has to play, or selection never sees it.
    seats = allocate_seats([0.97, 0.01, 0.01, 0.01], 8)
    assert seats.sum() == 8
    assert all(s >= 1 for s in seats)


def test_extinct_species_are_never_seated():
    seats = allocate_seats([0.6, 0.0, 0.4], 5)
    assert seats[1] == 0


# -- selection --------------------------------------------------------------


def test_selection_moves_share_toward_above_average_fitness():
    shares, extinct = replicator_step([0.5, 0.5], [1.0, 0.0], selection_rate=0.5)
    assert shares[0] > 0.5 > shares[1]
    assert pytest.approx(shares.sum()) == 1.0
    assert extinct == []


def test_selection_is_scale_free_in_the_units_of_fitness():
    # The replicator step normalizes by the spread of payoffs, so the same selection_rate
    # moves the population identically whether fitness is a return or a Sharpe.
    small = replicator_step([0.5, 0.5], [0.001, 0.002], selection_rate=0.4)[0]
    large = replicator_step([0.5, 0.5], [100.0, 200.0], selection_rate=0.4)[0]
    assert np.allclose(small, large)


def test_selection_culls_below_the_extinction_threshold():
    shares, extinct = replicator_step(
        [0.9, 0.1], [1.0, -1.0], selection_rate=1.0, extinction_threshold=0.2
    )
    assert extinct == [1]
    assert shares[1] == 0.0
    assert pytest.approx(shares[0]) == 1.0


def test_an_unseated_species_is_carried_not_punished():
    # nan fitness means "did not play". It must neither gain nor lose relative standing.
    shares, extinct = replicator_step([0.4, 0.4, 0.2], [1.0, 0.0, np.nan], selection_rate=0.5)
    assert extinct == []
    assert shares[2] > 0.0
    assert shares[0] > shares[1]


def test_a_strictly_better_strategy_takes_over_the_population():
    species = [
        Species("strong", ParamPolicy, {"edge": 1.0}),
        Species("weak", ParamPolicy, {"edge": 0.0}),
    ]
    report = run_ecology(species, edge_payoffs, generations=12, field_size=6)
    outcomes = report["outcomes"]
    assert outcomes["strong"]["outcome"] == DOMINANT
    assert outcomes["strong"]["final_share"] > outcomes["weak"]["final_share"]


# -- innovation -------------------------------------------------------------


def test_innovation_adds_variants_bred_from_the_current_leader():
    species = [
        Species("alpha", ParamPolicy, {"edge": 1.0}),
        Species("beta", ParamPolicy, {"edge": 0.5}),
    ]
    report = run_ecology(
        species,
        edge_payoffs,
        generations=9,
        field_size=8,
        innovate_every=3,
        innovator=mutating_innovator(scale=0.25),
        innovation_share=0.15,
    )
    births = [e for e in report["events"] if e["kind"] == "innovation"]
    assert len(births) == 3
    assert all(b["parent"] in {s["name"] for s in report["species"]} for b in births)
    # The variant is a perturbation of its parent's parameters, not a fresh strategy.
    first = births[0]
    parent = next(s for s in report["species"] if s["name"] == first["parent"])
    assert first["params"].keys() == parent["params"].keys()
    assert first["params"]["edge"] != parent["params"]["edge"]


def test_innovation_can_unseat_an_incumbent():
    # A lone incumbent with nothing to beat looks unbeatable. Entry is what tests it.
    species = [Species("incumbent", ParamPolicy, {"edge": 1.0})]
    report = run_ecology(
        species,
        edge_payoffs,
        generations=10,
        field_size=8,
        innovate_every=2,
        innovator=mutating_innovator(scale=0.5),
        innovation_share=0.25,
        selection_rate=1.0,
    )
    assert len(report["species"]) > 1
    final = report["final_shares"]
    winner = max(final, key=lambda k: final[k])
    assert winner != "incumbent", f"entry never displaced the incumbent: {final}"


def test_innovation_stops_at_max_species():
    species = [Species("alpha", ParamPolicy, {"edge": 1.0})]
    report = run_ecology(
        species,
        edge_payoffs,
        generations=10,
        field_size=6,
        innovate_every=1,
        innovator=mutating_innovator(),
        max_species=3,
    )
    assert len(report["species"]) == 3
    assert any(e["kind"] == "innovation_skipped" for e in report["events"])


def test_no_innovator_means_no_new_species():
    species = [Species("alpha", ParamPolicy, {"edge": 1.0})]
    report = run_ecology(species, edge_payoffs, generations=5, field_size=4, innovate_every=2)
    assert len(report["species"]) == 1
    assert not [e for e in report["events"] if e["kind"] == "innovation"]


# -- environmental perturbation ---------------------------------------------


def regime_dependent_payoffs(policies, *, generation, seed, shock):
    """A species' edge is worth its face value in calm and inverted in extreme: the
    strategy is not good or bad, it is good or bad *given the environment*."""
    sign = 1.0 if shock.distribution_mode == "calm" else -1.0
    return [sign * float(p.params["edge"]) for p in policies]


def test_a_perturbation_reverses_which_strategy_wins():
    species = [
        Species("trend", ParamPolicy, {"edge": 1.0}),
        Species("fade", ParamPolicy, {"edge": -1.0}),
    ]
    calm = run_ecology(
        species, regime_dependent_payoffs, generations=8, field_size=6,
        shocks=steady_shocks(8, distribution_mode="calm"),
    )
    stressed = run_ecology(
        species, regime_dependent_payoffs, generations=8, field_size=6,
        shocks=steady_shocks(8, distribution_mode="extreme"),
    )
    assert calm["final_shares"]["trend"] > calm["final_shares"]["fade"]
    assert stressed["final_shares"]["fade"] > stressed["final_shares"]["trend"]


def test_a_regime_rotation_shows_up_as_a_shock_event_per_generation():
    species = [Species("a", ParamPolicy, {"edge": 1.0}), Species("b", ParamPolicy, {"edge": 0.5})]
    report = run_ecology(
        species,
        regime_dependent_payoffs,
        generations=6,
        field_size=4,
        shocks=regime_shocks(6, period=2),
        extinction_threshold=0.0,
    )
    assert report["shocks"] == [
        "regime:calm",
        "regime:calm",
        "regime:hard",
        "regime:hard",
        "regime:extreme",
        "regime:extreme",
    ]
    # Transitions only: the calm opening, then the switch into hard, then into extreme.
    shocks = [e for e in report["events"] if e["kind"] == "shock"]
    assert [(e["generation"], e["distribution_mode"]) for e in shocks] == [
        (0, "calm"),
        (2, "hard"),
        (4, "extreme"),
    ]


def test_a_liquidity_shock_schedule_switches_on_and_off():
    schedule = liquidity_shocks(6, onset=2, impact_scale=3.0, duration=2)
    assert [s.impact_scale for s in schedule] == [1.0, 1.0, 3.0, 3.0, 1.0, 1.0]


def test_a_perturbation_can_collapse_a_dominant_strategy():
    # The shape a static backtest cannot produce: the same strategy dominates, then dies,
    # with nothing about the strategy having changed.
    species = [
        Species("turnover", ParamPolicy, {"edge": 1.0}),
        Species("patient", ParamPolicy, {"edge": 0.2}),
    ]

    def cost_sensitive(policies, *, generation, seed, shock):
        # Impact scale is a cost on the strategy that trades: above 1 it eats the edge.
        return [
            float(p.params["edge"]) - (shock.impact_scale - 1.0) * float(p.params["edge"])
            * 3.0
            for p in policies
        ]

    report = run_ecology(
        species,
        cost_sensitive,
        generations=15,
        field_size=8,
        shocks=liquidity_shocks(15, onset=3, impact_scale=2.0),
        selection_rate=0.3,
        extinction_threshold=0.0,
    )
    turnover = report["outcomes"]["turnover"]
    assert turnover["peak_share"] > 0.5
    assert turnover["peak_generation"] == 3, report["shares"]
    assert turnover["final_share"] < turnover["peak_share"]
    assert turnover["outcome"] == COLLAPSED, report["outcomes"]


# -- the trajectory is the output -------------------------------------------


def test_the_report_carries_the_whole_population_trajectory():
    species = [
        Species("a", ParamPolicy, {"edge": 1.0}),
        Species("b", ParamPolicy, {"edge": 0.4}),
    ]
    report = run_ecology(
        species,
        edge_payoffs,
        generations=5,
        field_size=4,
        innovate_every=5,
        innovator=mutating_innovator(),
    )
    width = len(report["species"])
    assert len(report["shares"]) == report["generations"] + 1
    assert all(len(row) == width for row in report["shares"])
    assert len(report["fitness"]) == report["generations"]
    assert len(report["seats"]) == report["generations"]
    assert all(abs(sum(row) - 1.0) < 1e-12 for row in report["shares"])
    # A species born mid-run has zero share in the rows before its birth.
    born = [s for s in report["species"] if s["born"] > 0]
    assert born
    idx = [s["name"] for s in report["species"]].index(born[0]["name"])
    assert report["shares"][0][idx] == 0.0


def test_fitness_is_none_for_a_species_that_held_no_seat():
    species = [
        Species("big", ParamPolicy, {"edge": 1.0}),
        Species("small", ParamPolicy, {"edge": 0.0}),
        Species("tiny", ParamPolicy, {"edge": 0.0}),
    ]
    # A field of one seat cannot hold three species, so two go unplayed each generation.
    report = run_ecology(species, edge_payoffs, generations=3, field_size=1)
    assert any(v is None for row in report["fitness"] for v in row)


def test_the_run_is_deterministic():
    species = [
        Species("a", ParamPolicy, {"edge": 1.0}),
        Species("b", ParamPolicy, {"edge": 0.3}),
    ]
    kwargs = dict(
        generations=8,
        field_size=6,
        innovate_every=3,
        innovator=mutating_innovator(scale=0.3),
        shocks=regime_shocks(8),
    )
    first = run_ecology(species, edge_payoffs, **kwargs)
    second = run_ecology(species, edge_payoffs, **kwargs)
    assert first == second


def test_outcomes_label_the_shape_not_the_endpoint():
    names = ["riser", "faller"]
    shares = [[0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.95, 0.05]]
    labels = classify_outcomes(names, shares)
    assert labels["riser"]["outcome"] == DOMINANT
    assert labels["faller"]["outcome"] == COLLAPSED
    assert labels["faller"]["peak_generation"] == 0


def test_coalitions_are_read_from_fitness_not_from_share():
    # Shares are compositional and so mechanically anticorrelated; measuring on fitness is
    # what makes "these two need the same conditions" a real reading.
    names = ["x", "y", "z"]
    fitness = [
        [1.0, 0.9, -1.0],
        [-1.0, -0.8, 1.0],
        [0.5, 0.6, -0.4],
        [-0.5, -0.4, 0.5],
    ]
    found = detect_coalitions(names, fitness, threshold=0.8)
    assert [c["members"] for c in found] == [["x", "y"]]
    assert found[0]["correlation"] > 0.8


def test_coalitions_need_enough_shared_generations():
    names = ["x", "y"]
    fitness = [[1.0, 1.0], [None, None], [-1.0, -1.0]]
    assert detect_coalitions(names, fitness, min_overlap=3) == []


def test_a_species_payoff_can_depend_on_what_it_competes_against():
    # The claim the whole module exists to support: fitness is a function of the field, so
    # the same strategy can dominate one population and die in another.
    def field_dependent(policies, *, generation, seed, shock):
        tags = [p.params["edge"] for p in policies]
        n_prey = sum(1 for t in tags if t == 1.0)
        # The predator only earns when there is prey to feed on; prey pays for the predator.
        return [
            (0.1 * n_prey if t == 0.0 else -0.05 * (len(tags) - n_prey)) for t in tags
        ]

    species = [
        Species("predator", ParamPolicy, {"edge": 0.0}),
        Species("prey", ParamPolicy, {"edge": 1.0}),
    ]
    report = run_ecology(
        species,
        field_dependent,
        generations=10,
        field_size=8,
        selection_rate=0.8,
        extinction_threshold=0.05,
    )
    trajectory = [row[[s["name"] for s in report["species"]].index("predator")] for row in report["shares"]]
    prey_trajectory = [row[[s["name"] for s in report["species"]].index("prey")] for row in report["shares"]]
    assert trajectory[-1] > trajectory[0], "the predator should gain while prey is around"
    assert prey_trajectory[-1] < prey_trajectory[0], "the prey should lose share"


def test_population_table_renders_every_generation():
    species = [Species("a", ParamPolicy, {"edge": 1.0}), Species("b", ParamPolicy, {"edge": 0.0})]
    report = run_ecology(species, edge_payoffs, generations=3, field_size=4)
    table = population_table(report)
    assert "| species |" in table
    assert "g0" in table and "g3" in table
    assert table.count("\n") == len(report["species"]) + 1


# -- validation -------------------------------------------------------------


def test_duplicate_species_names_are_rejected():
    species = [Species("a", ParamPolicy, {"edge": 1.0}), Species("a", ParamPolicy, {"edge": 0.0})]
    with pytest.raises(ValueError):
        run_ecology(species, edge_payoffs, generations=2, field_size=2)


def test_a_payoff_function_must_return_one_value_per_seat():
    species = [Species("a", ParamPolicy, {"edge": 1.0})]
    with pytest.raises(ValueError):
        run_ecology(species, lambda policies, **kw: [0.0], generations=1, field_size=4)


def test_a_short_shock_schedule_is_rejected():
    species = [Species("a", ParamPolicy, {"edge": 1.0})]
    with pytest.raises(ValueError):
        run_ecology(
            species, edge_payoffs, generations=5, field_size=2, shocks=steady_shocks(2)
        )


# -- wiring over the real envs ----------------------------------------------


def test_ecology_runs_over_the_shared_book_market():
    pytest.importorskip("pettingzoo")
    species = baseline_species()[:3]
    report = run_ecology(
        species,
        market_payoffs(n_symbols=2, n_days=40, max_steps=40),
        generations=3,
        field_size=3,
        seed=11,
    )
    assert len(report["shares"]) == 4
    assert set(report["final_shares"]) == {s.name for s in species}
    # Every species is seated in the opening field, so the first generation is fully
    # observed: the market really did produce a payoff for each of them.
    assert all(v is not None for v in report["fitness"][0])


def test_ecology_over_the_market_is_reproducible():
    pytest.importorskip("pettingzoo")
    species = baseline_species()[:3]

    def run():
        return run_ecology(
            species,
            market_payoffs(n_symbols=2, n_days=40, max_steps=40),
            generations=2,
            field_size=3,
            seed=5,
        )

    assert run() == run()


def test_ecology_runs_over_the_competition_env():
    pytest.importorskip("pettingzoo")
    species = baseline_species()[:2]
    report = run_ecology(
        species,
        competition_payoffs(n_symbols=2, n_days=40, max_steps=40),
        generations=2,
        field_size=2,
        shocks=[Shock("calm"), Shock("stress", "extreme")],
        seed=3,
    )
    assert len(report["fitness"]) == 2
