"""Ecological population dynamics over the multi-agent market (FinEvo).

A backtest scores a strategy against a tape. A multi-agent episode scores it against a
*fixed* field. Neither answers the question that decides whether a strategy is worth
running: what happens to it when the field itself changes because of how well it did.
FinEvo ("From Isolated Backtests to Ecological Market Games for Multi-Agent Financial
Strategy Evolution") frames strategies as an ecology rather than a ranking, and this
module is that layer over SharpeArena's existing multi-agent machinery
(:mod:`sharpearena.market_env`, :mod:`sharpearena.pettingzoo_env`,
:mod:`sharpearena.curriculum`, and the policies in :mod:`sharpearena.baselines`).

Three mechanisms, run once per generation, in this order:

* **Selection.** Each generation seats a field from the current population shares, plays
  one episode, and moves share toward whatever earned above the field's average payoff.
  The update is the discrete replicator map, normalized by the cross-species spread of
  payoffs so ``selection_rate`` means the same thing whatever units fitness is in.
  Species whose share falls under ``extinction_threshold`` are removed outright: a
  strategy nobody is running is not a strategy.
* **Innovation.** Every ``innovate_every`` generations a new variant is bred from the
  current leader and seeded with a small share taken pro rata from the incumbents. Two
  reasons it matters: a population with no entry converges and stops being informative,
  and a strategy that only dominates because nothing new arrived has not been tested.
* **Environmental perturbation.** A :class:`Shock` schedule drives the exogenous side of
  the episode, rotating the synthetic distribution mode (the ``calm`` / ``hard`` /
  ``extreme`` vocabulary :mod:`sharpearena.curriculum` already uses) and optionally
  scaling the impact coefficients, which is a liquidity shock: the same order costs more.

**The output is the trajectory, not a score.** ``run_ecology`` returns the full
generation-by-generation share matrix, the fitness matrix behind it, and the events that
moved them, because the finding a static backtest cannot produce is a shape over time: a
strategy that dominates one field and collapses in another, or a pair that only survives
in each other's company. A single final number would throw exactly that away.
:func:`classify_outcomes` and :func:`detect_coalitions` read the trajectory; they are
descriptive summaries of what happened in this run, not claims about an equilibrium.

**Determinism.** No RNG anywhere. Field seats are allocated by largest remainder, ties
broken by species index; the leader is the highest share with the lowest index; shocks
come from a fixed schedule; variants are bred by a deterministic rule. The same species
set, config, and ``seed`` reproduce the run exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

# The observation-in, weights-out protocol every SharpeArena policy already satisfies (the
# same alias :mod:`sharpearena.baselines` exports). Restated here rather than imported so
# building an ecology never drags in the baseline module, its native scorer, or its env.
Policy = Callable[[dict], np.ndarray]

#: Called once per generation with the seated field, in seat order, and asked for one
#: fitness per seat: ``payoffs(policies, generation=..., seed=..., shock=...)``.
PayoffFn = Callable[..., Sequence[float]]

#: Breeds a variant from a parent: ``innovator(parent, generation) -> Species``.
InnovatorFn = Callable[["Species", int], "Species"]

CALM = "calm"
HARD = "hard"
EXTREME = "extreme"
REGIME_MODES = (CALM, HARD, EXTREME)

DOMINANT = "dominant"
PERSISTENT = "persistent"
COLLAPSED = "collapsed"
EXTINCT = "extinct"
MARGINAL = "marginal"


# -- species and shocks -----------------------------------------------------


@dataclass(frozen=True)
class Species:
    """One strategy in the population: a name, its parameters, and how to build it.

    ``build`` is called with ``params`` to produce a **fresh** policy instance at the start
    of every episode, so a stateful strategy never carries state across generations. The
    parameters are held separately from the built policy because innovation breeds by
    perturbing them, and because they are what the report needs to explain a variant.
    """

    name: str
    build: Callable[..., Policy]
    params: dict = field(default_factory=dict)
    parent: Optional[str] = None
    born: int = 0

    def instantiate(self) -> Policy:
        """A fresh policy instance for one episode."""
        return self.build(**self.params)


@dataclass(frozen=True)
class Shock:
    """One generation's exogenous environment: the perturbation mechanism.

    ``distribution_mode`` selects the synthetic path family (``calm`` / ``hard`` /
    ``extreme``). ``seed_offset`` moves the episode to a different path within it.
    ``impact_scale`` multiplies the market's impact coefficients, so a value above ``1``
    is a liquidity shock: the same order moves the price further and costs more to fill.
    """

    name: str
    distribution_mode: str = CALM
    seed_offset: int = 0
    impact_scale: float = 1.0


def steady_shocks(generations: int, *, distribution_mode: str = CALM) -> list[Shock]:
    """An unperturbed schedule: the same environment every generation.

    The control condition. Whatever the population does under this schedule is caused by
    selection and innovation alone, which is what any claim about a shock has to beat.
    """
    return [Shock("steady", distribution_mode) for _ in range(int(generations))]


def regime_shocks(
    generations: int,
    *,
    modes: Sequence[str] = REGIME_MODES,
    period: int = 1,
    seed_stride: int = 997,
) -> list[Shock]:
    """Rotate the distribution mode every ``period`` generations, on chained seeds.

    The ecological reading of :func:`sharpearena.curriculum.regime_curriculum`: instead of
    walking one agent through calm then shock then recovery, the whole population is walked
    through it, so the question becomes which strategies survive the transition rather than
    which one scores best inside a regime. ``seed_stride`` is the same coprime-ish chaining
    step the curriculum uses, so consecutive generations land far apart in seed space.
    """
    if int(generations) < 1:
        raise ValueError("generations must be >= 1")
    if not modes:
        raise ValueError("modes must be non-empty")
    if int(period) < 1:
        raise ValueError("period must be >= 1")
    out: list[Shock] = []
    for g in range(int(generations)):
        mode = modes[(g // int(period)) % len(modes)]
        out.append(Shock(f"regime:{mode}", mode, g * int(seed_stride)))
    return out


def liquidity_shocks(
    generations: int,
    *,
    onset: int,
    impact_scale: float = 3.0,
    duration: Optional[int] = None,
    distribution_mode: str = CALM,
) -> list[Shock]:
    """A calm run interrupted at ``onset`` by a spell of ``impact_scale`` impact.

    The cleanest ecological experiment in the module: hold the price path family fixed and
    change only what trading costs. Strategies that were winning on turnover lose their
    edge without the tape doing anything, which is a result a static backtest on the same
    path cannot show. ``duration`` of ``None`` means the shock never lifts.
    """
    if int(generations) < 1:
        raise ValueError("generations must be >= 1")
    if int(onset) < 0:
        raise ValueError("onset must be >= 0")
    if impact_scale <= 0.0:
        raise ValueError("impact_scale must be > 0")
    end = int(generations) if duration is None else int(onset) + int(duration)
    out: list[Shock] = []
    for g in range(int(generations)):
        if int(onset) <= g < end:
            out.append(Shock("liquidity", distribution_mode, 0, float(impact_scale)))
        else:
            out.append(Shock("steady", distribution_mode, 0, 1.0))
    return out


# -- the three mechanisms ---------------------------------------------------


def allocate_seats(shares: Sequence[float], field_size: int) -> np.ndarray:
    """Seat a field of ``field_size`` agents in proportion to ``shares``.

    Largest remainder (Hamilton): floor the proportional entitlement, then hand the
    leftover seats to the largest fractional remainders, ties broken by the lower species
    index. Every species with a positive share is guaranteed at least one seat whenever the
    field is wide enough to hold them all, taken from the largest holder, because an
    unseated species plays no episode and so contributes no fitness for selection to read.
    Pure and deterministic: no RNG, no rounding drift, seats always sum to ``field_size``.
    """
    weights = np.asarray(shares, dtype=np.float64)
    size = int(field_size)
    if size < 1:
        raise ValueError("field_size must be >= 1")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("shares must sum to a positive number")
    entitlement = weights / total * size
    seats = np.floor(entitlement).astype(np.int64)
    leftover = size - int(seats.sum())
    if leftover > 0:
        remainder = entitlement - seats
        order = np.lexsort((np.arange(weights.shape[0]), -remainder))
        seats[order[:leftover]] += 1
    live = weights > 0.0
    if int(live.sum()) <= size:
        for i in np.flatnonzero(live & (seats == 0)):
            donor = int(np.argmax(seats))
            if seats[donor] <= 1:
                break
            seats[donor] -= 1
            seats[i] += 1
    return seats


def replicator_step(
    shares: Sequence[float],
    fitness: Sequence[float],
    *,
    selection_rate: float = 1.0,
    extinction_threshold: float = 0.02,
) -> tuple[np.ndarray, list[int]]:
    """One selection step: move share toward above-average fitness, then cull.

    The discrete replicator map ``share_i * (1 + selection_rate * z_i)`` where ``z_i`` is
    species ``i``'s fitness minus the share-weighted field average, divided by the spread
    of the observed fitnesses. The normalization is what makes ``selection_rate`` portable:
    without it the same rate means something different for a payoff measured in compound
    return than in Sharpe, and the population's speed would silently depend on units.

    A species with ``nan`` fitness was not seated this generation. It is carried at its
    current share rather than punished, since a strategy that was not run did not lose.

    Growth is floored at zero (share cannot go negative), the result is renormalized, and
    anything left under ``extinction_threshold``, or driven to zero outright by a payoff
    far enough below the field, is removed and its mass redistributed. The last survivors
    are never all culled at once: a population has to contain something.

    Returns the new shares and the indices that went extinct on this step.
    """
    weights = np.asarray(shares, dtype=np.float64).copy()
    scores = np.asarray(fitness, dtype=np.float64)
    if weights.shape != scores.shape:
        raise ValueError("shares and fitness must have the same shape")
    observed = np.isfinite(scores) & (weights > 0.0)
    if not observed.any():
        return _renormalize(weights), []

    mass = float(weights[observed].sum())
    mean = float((weights[observed] * scores[observed]).sum() / mass) if mass > 0.0 else 0.0
    centered = scores[observed] - mean
    spread = float(np.sqrt((weights[observed] * centered * centered).sum() / mass))
    if not math.isfinite(spread) or spread <= 0.0:
        spread = 1.0

    growth = np.ones_like(weights)
    growth[observed] = np.clip(1.0 + float(selection_rate) * centered / spread, 0.0, None)
    updated = _renormalize(weights * growth)

    alive = weights > 0.0
    culled = alive & ((updated < float(extinction_threshold)) | (updated <= 0.0))
    doomed = [int(i) for i in np.flatnonzero(culled)]
    if doomed and len(doomed) < int(alive.sum()):
        updated[doomed] = 0.0
        updated = _renormalize(updated)
    else:
        doomed = []
    return updated, doomed


def mutating_innovator(
    *, scale: float = 0.25, floor: float = 1e-6
) -> InnovatorFn:
    """An innovator that breeds a variant by perturbing one of the parent's parameters.

    The parameter is chosen by rotating through the parent's numeric parameters in sorted
    key order using the generation index, and it is scaled up on even generations and down
    on odd ones, so a run explores both directions without drawing a single random number.
    Integer parameters stay integers (and stay at least ``1``), because a lookback of 37.5
    bars is not a strategy. A parent with no numeric parameters yields an exact clone,
    which is still a real ecological event: a second copy of the leader now competes with
    the first for the same flow.
    """

    def _innovate(parent: Species, generation: int) -> Species:
        numeric = sorted(
            k for k, v in parent.params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        params = dict(parent.params)
        suffix = f"v{generation}"
        if numeric:
            key = numeric[generation % len(numeric)]
            factor = 1.0 + float(scale) if generation % 2 == 0 else 1.0 - float(scale)
            value = parent.params[key]
            if isinstance(value, int):
                params[key] = max(1, int(round(value * factor)))
            else:
                params[key] = max(float(floor), float(value) * factor)
            suffix = f"{key}{generation}"
        return Species(
            name=f"{parent.name}~{suffix}",
            build=parent.build,
            params=params,
            parent=parent.name,
            born=int(generation),
        )

    return _innovate


# -- the loop ---------------------------------------------------------------


def run_ecology(
    species: Sequence[Species],
    payoffs: PayoffFn,
    *,
    generations: int = 20,
    field_size: int = 8,
    initial_shares: Optional[Sequence[float]] = None,
    selection_rate: float = 1.0,
    extinction_threshold: float = 0.02,
    innovate_every: int = 0,
    innovator: Optional[InnovatorFn] = None,
    innovation_share: float = 0.1,
    max_species: int = 24,
    shocks: Optional[Sequence[Shock]] = None,
    seed: int = 0,
) -> dict:
    """Run the ecology and return its population trajectory.

    ``payoffs`` is called once per generation with the seated field in seat order and must
    return one fitness per seat; :func:`market_payoffs` and :func:`competition_payoffs`
    build one over the existing envs, and any callable with that shape works, which is what
    keeps the population dynamics testable without spinning up a market.

    The report is JSON-friendly plain types:

    ``species``
        every species that ever existed, in birth order, with its parent and birth
        generation. This is the column axis of every matrix below.
    ``shares``
        ``generations + 1`` rows by ``len(species)`` columns. Row ``g`` is the population
        *entering* generation ``g``; the last row is the final population. Columns for
        species not yet born are ``0.0``.
    ``fitness``
        ``generations`` rows: what each species earned, or ``None`` where it held no seat.
    ``seats``
        ``generations`` rows of the integer field composition actually played.
    ``events``
        every innovation, extinction, and shock, stamped with its generation, in order.
    ``outcomes`` / ``coalitions``
        the read of the trajectory: see :func:`classify_outcomes` and
        :func:`detect_coalitions`.
    """
    roster = list(species)
    if not roster:
        raise ValueError("an ecology needs at least one species")
    if len({s.name for s in roster}) != len(roster):
        raise ValueError("species names must be unique")
    if int(generations) < 1:
        raise ValueError("generations must be >= 1")
    if int(field_size) < 1:
        raise ValueError("field_size must be >= 1")
    if not 0.0 <= float(innovation_share) < 1.0:
        raise ValueError("innovation_share must lie in [0, 1)")

    schedule = list(shocks) if shocks is not None else steady_shocks(int(generations))
    if len(schedule) < int(generations):
        raise ValueError("the shock schedule is shorter than the run")

    if initial_shares is None:
        shares = np.full(len(roster), 1.0 / len(roster), dtype=np.float64)
    else:
        shares = _renormalize(np.asarray(initial_shares, dtype=np.float64))
        if shares.shape[0] != len(roster):
            raise ValueError("initial_shares must have one entry per species")

    share_rows: list[list[float]] = [_row(shares, len(roster))]
    fitness_rows: list[list[Optional[float]]] = []
    seat_rows: list[list[int]] = []
    events: list[dict] = []

    previous: Optional[Shock] = None
    for g in range(int(generations)):
        shock = schedule[g]
        # Log transitions, not every generation: a shock that has not changed is the
        # environment, and an event log that repeats it drowns the ones that matter.
        if previous is None or (
            shock.distribution_mode,
            shock.impact_scale,
        ) != (previous.distribution_mode, previous.impact_scale):
            events.append(
                {
                    "generation": g,
                    "kind": "shock",
                    "name": shock.name,
                    "distribution_mode": shock.distribution_mode,
                    "impact_scale": float(shock.impact_scale),
                }
            )
        previous = shock

        seats = allocate_seats(shares, int(field_size))
        seat_rows.append([int(x) for x in seats])

        # Seat the field in canonical (species, then repeat) order and remember which seat
        # belongs to which species, so payoffs come back attributable.
        field_policies: list[Policy] = []
        seat_owner: list[int] = []
        for i, count in enumerate(seats):
            for _ in range(int(count)):
                field_policies.append(roster[i].instantiate())
                seat_owner.append(i)

        raw = list(payoffs(field_policies, generation=g, seed=int(seed), shock=shock))
        if len(raw) != len(field_policies):
            raise ValueError(
                f"payoffs returned {len(raw)} values for {len(field_policies)} seats"
            )

        totals = np.zeros(len(roster), dtype=np.float64)
        counts = np.zeros(len(roster), dtype=np.int64)
        for value, owner in zip(raw, seat_owner):
            totals[owner] += float(value)
            counts[owner] += 1
        fitness = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        fitness_rows.append(
            [None if not math.isfinite(x) else float(x) for x in fitness]
        )

        shares, doomed = replicator_step(
            shares,
            fitness,
            selection_rate=float(selection_rate),
            extinction_threshold=float(extinction_threshold),
        )
        for i in doomed:
            events.append(
                {"generation": g, "kind": "extinction", "name": roster[i].name}
            )

        if (
            innovator is not None
            and int(innovate_every) > 0
            and (g + 1) % int(innovate_every) == 0
        ):
            if len(roster) >= int(max_species):
                events.append(
                    {"generation": g, "kind": "innovation_skipped", "reason": "max_species"}
                )
            else:
                parent = roster[int(np.argmax(shares))]
                variant = innovator(parent, g + 1)
                if variant.name in {s.name for s in roster}:
                    events.append(
                        {
                            "generation": g,
                            "kind": "innovation_skipped",
                            "reason": "duplicate_name",
                            "name": variant.name,
                        }
                    )
                else:
                    roster.append(variant)
                    shares = np.append(
                        shares * (1.0 - float(innovation_share)), float(innovation_share)
                    )
                    shares = _renormalize(shares)
                    events.append(
                        {
                            "generation": g,
                            "kind": "innovation",
                            "name": variant.name,
                            "parent": parent.name,
                            "params": dict(variant.params),
                        }
                    )

        share_rows.append(_row(shares, len(roster)))

    width = len(roster)
    share_matrix = [_row(np.asarray(r, dtype=np.float64), width) for r in share_rows]
    fitness_matrix = [list(r) + [None] * (width - len(r)) for r in fitness_rows]
    seat_matrix = [list(r) + [0] * (width - len(r)) for r in seat_rows]
    names = [s.name for s in roster]
    final = {name: share_matrix[-1][i] for i, name in enumerate(names)}

    return {
        "species": [
            {"name": s.name, "parent": s.parent, "born": s.born, "params": dict(s.params)}
            for s in roster
        ],
        "generations": int(generations),
        "field_size": int(field_size),
        "shares": share_matrix,
        "fitness": fitness_matrix,
        "seats": seat_matrix,
        "shocks": [schedule[g].name for g in range(int(generations))],
        "events": events,
        "final_shares": final,
        "outcomes": classify_outcomes(names, share_matrix),
        "coalitions": detect_coalitions(names, fitness_matrix),
    }


# -- reading the trajectory -------------------------------------------------


def classify_outcomes(
    names: Sequence[str],
    shares: Sequence[Sequence[float]],
    *,
    dominance: float = 0.5,
    collapse_peak: float = 0.25,
    collapse_final: float = 0.05,
) -> dict:
    """Label each species by the *shape* of its share trajectory, not its final number.

    ``dominant`` ended holding ``dominance`` or more of the population. ``collapsed`` once
    held ``collapse_peak`` or more and ended at or under ``collapse_final``: it worked, and
    then it stopped working, which is the single most useful thing an ecology reports and
    the one a static backtest is structurally unable to say. ``extinct`` was culled without
    ever getting big. ``persistent`` survived above ``collapse_final`` without dominating.
    ``marginal`` is everything else: alive, small, never large.
    """
    matrix = np.asarray(shares, dtype=np.float64)
    out: dict = {}
    for i, name in enumerate(names):
        column = matrix[:, i]
        final = float(column[-1])
        peak = float(column.max())
        if peak >= float(collapse_peak) and final <= float(collapse_final):
            label = COLLAPSED
        elif final >= float(dominance):
            label = DOMINANT
        elif final <= 0.0:
            label = EXTINCT
        elif final > float(collapse_final):
            label = PERSISTENT
        else:
            label = MARGINAL
        out[name] = {
            "outcome": label,
            "final_share": final,
            "peak_share": peak,
            "peak_generation": int(np.argmax(column)),
        }
    return out


def detect_coalitions(
    names: Sequence[str],
    fitness: Sequence[Sequence[Optional[float]]],
    *,
    threshold: float = 0.6,
    min_overlap: int = 3,
) -> list[dict]:
    """Pairs whose fitness moved together across the generations they both played.

    Coalition here is a **descriptive** statement about this run: two species did well in
    the same generations and badly in the same generations, so whatever conditions one of
    them needs, the other needs too, and a field containing one is a field the other can
    live in. It is deliberately measured on fitness rather than population share, because
    shares are compositional (they sum to one) and so are mechanically anticorrelated,
    which would manufacture the opposite finding for free.

    Pairs need ``min_overlap`` generations where both were seated. This is not a claim of
    cooperation, collusion, or equilibrium: nothing here observes intent.
    """
    matrix = np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in fitness],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[0] < int(min_overlap):
        return []
    found: list[dict] = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            both = np.isfinite(matrix[:, a]) & np.isfinite(matrix[:, b])
            if int(both.sum()) < int(min_overlap):
                continue
            x = matrix[both, a]
            y = matrix[both, b]
            sx = float(x.std())
            sy = float(y.std())
            if sx <= 0.0 or sy <= 0.0:
                continue
            r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
            if r >= float(threshold):
                found.append(
                    {
                        "members": [names[a], names[b]],
                        "correlation": r,
                        "generations": int(both.sum()),
                    }
                )
    found.sort(key=lambda d: (-d["correlation"], d["members"]))
    return found


def population_table(report: dict) -> str:
    """Render the run as a markdown table: one row per species, the trajectory in columns.

    Deliberately shows the whole share path rather than the endpoint, since the endpoint is
    the part a backtest could already have told you.
    """
    names = [s["name"] for s in report["species"]]
    shares = report["shares"]
    header = "| species | parent | born | " + " | ".join(
        f"g{g}" for g in range(len(shares))
    ) + " | outcome |"
    rule = "|" + "---|" * (len(shares) + 4)
    lines = [header, rule]
    for i, meta in enumerate(report["species"]):
        path = " | ".join(f"{row[i]:.3f}" for row in shares)
        lines.append(
            f"| {names[i]} | {meta['parent'] or ''} | {meta['born']} | {path} | "
            f"{report['outcomes'][names[i]]['outcome']} |"
        )
    return "\n".join(lines)


# -- payoff sources over the existing envs ----------------------------------


def _episode_fitness(returns: Sequence[float], mode: str, n_trials: int) -> float:
    """Collapse one agent's realized bar returns into a single fitness number."""
    series = [float(r) for r in returns]
    if mode == "compound_return":
        total = 1.0
        for r in series:
            total *= 1.0 + r
        return total - 1.0
    if mode == "mean_return":
        return float(np.mean(series)) if series else 0.0
    if mode == "deflated_sharpe":
        if len(series) < 2:
            return 0.0
        from .sharpearena_py import score_run

        return float(json.loads(score_run(series, int(n_trials))).get("deflated_sharpe", 0.0))
    raise ValueError(
        "fitness must be 'compound_return', 'mean_return', or 'deflated_sharpe'"
    )


def market_payoffs(
    *,
    n_symbols: int = 4,
    n_days: int = 120,
    capital: float = 1.0,
    kyle_lambda: float = 0.1,
    eta: float = 0.05,
    volume_scale: float = 1.0,
    max_steps: int = 512,
    fitness: str = "compound_return",
    n_trials: int = 0,
    **env_kwargs: Any,
) -> PayoffFn:
    """Payoffs from the **shared-book** market: the ecologically meaningful one.

    Every seated policy trades one book per symbol through
    :class:`sharpearena.market_env.EndogenousMarketEnv`, so a species that gains population
    share is, by the next generation, a larger fraction of the flow moving the price
    everyone else fills against. That feedback is the reason population share is worth
    tracking at all: a strategy's payoff genuinely depends on how many copies of it and of
    its rivals are running, which is exactly what a fixed-field episode holds constant.

    The :class:`~sharpearena.market_env.EndogenousMarketEnv` construction is deferred to
    call time, so importing this module never requires ``pettingzoo``.
    """

    def _payoffs(
        policies: Sequence[Policy], *, generation: int, seed: int, shock: Shock
    ) -> list[float]:
        from .market_env import EndogenousMarketEnv

        env = EndogenousMarketEnv(
            n_agents=len(policies),
            n_symbols=int(n_symbols),
            n_days=int(n_days),
            seed=int(seed) + int(shock.seed_offset) + generation,
            capital=float(capital),
            kyle_lambda=float(kyle_lambda) * float(shock.impact_scale),
            eta=float(eta) * float(shock.impact_scale),
            volume_scale=float(volume_scale),
            distribution_mode=shock.distribution_mode,
            **env_kwargs,
        )
        roster = list(env.possible_agents)
        assigned = dict(zip(roster, policies))
        series: dict[str, list[float]] = {a: [] for a in roster}
        obs, _ = env.reset()
        for _ in range(int(max_steps)):
            live = list(env.agents)
            if not live:
                break
            actions = {
                a: np.asarray(assigned[a](obs[a]), dtype=np.float32) for a in live
            }
            obs, rewards, _terminated, _truncated, _infos = env.step(actions)
            for a, r in rewards.items():
                series[a].append(float(r))
        env.close()
        return [_episode_fitness(series[a], fitness, int(n_trials)) for a in roster]

    return _payoffs


def competition_payoffs(
    *,
    n_symbols: int = 4,
    n_days: int = 120,
    max_steps: int = 512,
    fitness: str = "compound_return",
    n_trials: int = 0,
    **env_kwargs: Any,
) -> PayoffFn:
    """Payoffs from the **competition** env: the same tape, no cross-agent impact.

    :class:`sharpearena.pettingzoo_env.MultiAgentSharpeArenaEnv` gives every seat its own
    copy of one frozen path, so a species' payoff does not depend on the field at all. That
    makes it the control for :func:`market_payoffs`: selection and innovation still run,
    perturbation still runs, but the impact channel is switched off. Any population effect
    that survives here was not caused by agents moving the price.
    """

    def _payoffs(
        policies: Sequence[Policy], *, generation: int, seed: int, shock: Shock
    ) -> list[float]:
        from .pettingzoo_env import MultiAgentSharpeArenaEnv

        env = MultiAgentSharpeArenaEnv(
            n_agents=len(policies),
            n_symbols=int(n_symbols),
            n_days=int(n_days),
            seed=int(seed) + int(shock.seed_offset) + generation,
            distribution_mode=shock.distribution_mode,
            **env_kwargs,
        )
        roster = list(env.possible_agents)
        assigned = dict(zip(roster, policies))
        series: dict[str, list[float]] = {a: [] for a in roster}
        obs, _ = env.reset()
        for _ in range(int(max_steps)):
            live = list(env.agents)
            if not live:
                break
            actions = {
                a: np.asarray(assigned[a](obs[a]), dtype=np.float32) for a in live
            }
            obs, rewards, _terminated, _truncated, _infos = env.step(actions)
            for a, r in rewards.items():
                series[a].append(float(r))
        env.close()
        return [_episode_fitness(series[a], fitness, int(n_trials)) for a in roster]

    return _payoffs


def baseline_species(*, include_behavioral: bool = False) -> list[Species]:
    """The reference policies from :mod:`sharpearena.baselines` as a starting population.

    Parameter-free policies get empty ``params``; the parameterized ones are given their
    documented defaults explicitly so :func:`mutating_innovator` has something to breed
    from. ``include_behavioral`` adds the biased counterparties, which is worth doing when
    the question is whether an edge survives contact with irrational flow rather than only
    with rational flow.

    The import is deferred to call time: an ecology built from custom species never loads
    the baseline module, its native scorer, or its env.
    """
    from .baselines import (
        EqualWeightLongPolicy,
        FlatPolicy,
        KellyVolTargetPolicy,
        MaxSharpePolicy,
        MinVariancePolicy,
        MomentumPolicy,
    )

    out = [
        Species(FlatPolicy.name, FlatPolicy),
        Species(EqualWeightLongPolicy.name, EqualWeightLongPolicy),
        Species(MomentumPolicy.name, MomentumPolicy),
        Species(MinVariancePolicy.name, MinVariancePolicy, {"lookback": 60}),
        Species(MaxSharpePolicy.name, MaxSharpePolicy, {"lookback": 60}),
        Species(
            KellyVolTargetPolicy.name,
            KellyVolTargetPolicy,
            {"lookback": 60, "kelly_fraction": 0.25, "target_vol": 0.01},
        ),
    ]
    if include_behavioral:
        from .baselines import DispositionEffectPolicy, OverconfidentPolicy

        out.append(Species(DispositionEffectPolicy.name, DispositionEffectPolicy))
        out.append(Species(OverconfidentPolicy.name, OverconfidentPolicy))
    return out


# -- helpers ----------------------------------------------------------------


def _renormalize(weights: np.ndarray) -> np.ndarray:
    """Scale non-negative weights to sum to one, falling back to uniform if all are zero."""
    clipped = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape[0], 1.0 / clipped.shape[0], dtype=np.float64)
    return clipped / total


def _row(shares: np.ndarray, width: int) -> list[float]:
    """One share row padded out to the final species count with zeros (unborn species)."""
    row = [float(x) for x in shares]
    return row + [0.0] * (width - len(row))


__all__ = [
    "Species",
    "Shock",
    "Policy",
    "PayoffFn",
    "InnovatorFn",
    "REGIME_MODES",
    "DOMINANT",
    "PERSISTENT",
    "COLLAPSED",
    "EXTINCT",
    "MARGINAL",
    "steady_shocks",
    "regime_shocks",
    "liquidity_shocks",
    "allocate_seats",
    "replicator_step",
    "mutating_innovator",
    "run_ecology",
    "classify_outcomes",
    "detect_coalitions",
    "population_table",
    "market_payoffs",
    "competition_payoffs",
    "baseline_species",
]
