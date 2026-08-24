"""Red-team probe: can flow that moves the cleared price then trade against its own move?

Following "Can Reinforcement Learning Efficiently Discover Price Manipulation?".

**Read this before reading anything else in the module.** This is a diagnostic aimed at
the simulator. It is not a strategy, not a recommended policy, and not trading advice, and
it will be misread as all three unless that is said first. No output of this module should
be shipped, fitted, or handed to an execution surface. The schedules below are written to
be crude and obvious on purpose: they exist to be caught, not to work.

It has exactly two jobs.

1. **A realism check on the impact model.** :class:`~sharpearena.market_env.EndogenousMarketEnv`
   moves its cleared price with aggregate flow (Kyle permanent impact, Almgren-Chriss
   temporary impact), which is what makes it a market rather than a replay. Any such
   specification can in principle be gamed by pushing the price and unwinding into the
   push. If that turns out to be trivially and unboundedly profitable, the conclusion is
   that the impact specification is wrong, not that the agent is clever. A simulator whose
   dominant strategy is a pump is not a market, and every score measured on it is a score
   of the bug.
2. **A trajectory generator.** The runs produced here are, by construction, exactly the
   behaviour a scoring layer's process gates are supposed to catch. Feeding them through
   those gates is how you find out whether the gates work, and a gate that has never been
   shown a positive example has not been tested.

**The useful output is the boundary, not the profit.** ``impact_pnl > 0`` at one parameter
setting says very little. :func:`impact_boundary_sweep` says where along the impact axis
the pump stops paying, and :func:`size_response` says whether the profit peaks and turns
over or grows forever with size. Those two are statements about the simulator's realism
and they are the reason this module exists. A finite boundary with a bounded size response
is the healthy result.

**Attribution.** Raw end-of-episode P&L is not evidence of manipulation: a manipulator that
happens to be long through an exogenous rally books a profit it did not cause. So every
run is scored twice, once in the live market and once in a reference market with
``kyle_lambda = eta = 0`` where flow cannot move price, on the same seed and the same
schedule. The difference, ``impact_pnl``, is the part attributable to having moved the
price. That is the number the boundary is drawn on.

Determinism: the market is seeded, every policy here is a pure function of the observation
and the bar index, and there is no clock and no I/O, so a sweep reproduces byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np

from .market_env import EndogenousMarketEnv

# The manipulator is always the first agent in the canonical roster; the rest are followers.
_MANIPULATOR = "agent_0"

DISCLAIMER = (
    "sharpearena.manipulation is a red-team diagnostic on the simulator's impact model. "
    "It is not a strategy, not a recommended policy, and not trading advice."
)


@dataclass(frozen=True)
class ManipulationParams:
    """Market configuration plus the crude push-and-unwind schedule under test.

    The schedule is a round trip and nothing more: the manipulator ramps its target weight
    from flat to ``push_weight`` over ``push_bars``, holds for ``hold_bars``, then unwinds
    to flat over ``dump_bars``. Because
    :class:`~sharpearena.market_env.EndogenousMarketEnv` derives order size from the
    *change* in target weight, a rising weight is sustained buying and a falling weight is
    sustained selling, which is what makes the ramp a push.

    ``follower_gain`` sets how hard the other agents chase the move. At ``0.0`` they are
    inert and the probe measures the impact specification alone. Above zero they supply the
    thing a pump actually needs: someone to unwind into.

    ``impact_exponent`` (default ``1.0`` = linear) selects the permanent-impact shape the
    probe runs against. Under linear permanent impact, round-trip unprofitability is a
    theorem (Huberman-Stanzl 2004), so the linear probe can only confirm theory; an
    exponent below one makes permanent impact concave in flow, the regime in which theory
    predicts manipulation can pay, which is what makes the probe falsifiable.
    """

    n_symbols: int = 1
    n_days: int = 80
    n_followers: int = 3
    capital: float = 1.0
    kyle_lambda: float = 0.1
    eta: float = 0.05
    volume_scale: float = 1.0
    distribution_mode: str = "calm"
    max_weight: float = 1.0
    start_bar: int = 5
    push_bars: int = 5
    hold_bars: int = 1
    dump_bars: int = 5
    push_weight: float = 0.8
    follower_gain: float = 30.0
    impact_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.n_followers < 0:
            raise ValueError("n_followers must be >= 0")
        if min(self.push_bars, self.dump_bars) < 1:
            raise ValueError("push_bars and dump_bars must be >= 1")
        if self.hold_bars < 0:
            raise ValueError("hold_bars must be >= 0")
        if self.start_bar < 0:
            raise ValueError("start_bar must be >= 0")
        if not self.impact_exponent > 0.0:
            raise ValueError("impact_exponent must be positive (1.0 = linear)")
        span = self.start_bar + self.push_bars + self.hold_bars + self.dump_bars
        if span >= self.n_days:
            raise ValueError("the round trip must finish inside the episode")


@dataclass(frozen=True)
class ManipulationResult:
    """One seeded probe run, scored against its own zero-impact reference.

    ``impact_pnl`` is the manipulation-attributable P&L: live-market P&L minus the P&L the
    identical schedule earns on the identical seed in a market where flow cannot move
    price. ``profitable`` is the sign of that number and nothing more, which is why a
    single :class:`ManipulationResult` is a data point rather than a finding.
    """

    seed: int
    profitable: bool
    impact_pnl: float
    live_pnl: float
    reference_pnl: float
    peak_price_move: float
    kyle_lambda: float
    eta: float
    push_weight: float
    follower_gain: float
    impact_exponent: float = 1.0

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "profitable": self.profitable,
            "impact_pnl": self.impact_pnl,
            "live_pnl": self.live_pnl,
            "reference_pnl": self.reference_pnl,
            "peak_price_move": self.peak_price_move,
            "kyle_lambda": self.kyle_lambda,
            "eta": self.eta,
            "push_weight": self.push_weight,
            "follower_gain": self.follower_gain,
            "impact_exponent": self.impact_exponent,
            "disclaimer": DISCLAIMER,
        }


@dataclass(frozen=True)
class BoundaryReport:
    """Where along one impact axis the pump stops paying.

    ``boundary`` is the crossing point, linearly interpolated between the last profitable
    grid value and the first unprofitable one, or ``None`` when the sweep never crosses.
    A ``None`` boundary is the finding that matters: paired with
    ``profitable_everywhere`` it says the specification is exploitable across the whole
    range tested, and the impact model needs a second look rather than the agent needing a
    better gate.
    """

    axis: str
    values: tuple[float, ...]
    impact_pnl: tuple[float, ...]
    profitable: tuple[bool, ...]
    boundary: Optional[float]
    profitable_everywhere: bool
    unprofitable_everywhere: bool

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "values": list(self.values),
            "impact_pnl": list(self.impact_pnl),
            "profitable": list(self.profitable),
            "boundary": self.boundary,
            "profitable_everywhere": self.profitable_everywhere,
            "unprofitable_everywhere": self.unprofitable_everywhere,
            "disclaimer": DISCLAIMER,
        }


@dataclass(frozen=True)
class SizeResponse:
    """How manipulation P&L scales with the size of the push.

    ``bounded`` is the realism verdict: an impact model that pays more for every extra unit
    of size, forever, is broken, because real books get more expensive to push than the
    push is worth. ``peak_push_weight`` is where the profit turns over, which is the
    simulator's implicit statement about how big a position it thinks a market can absorb.
    """

    push_weights: tuple[float, ...]
    impact_pnl: tuple[float, ...]
    peak_push_weight: float
    peak_impact_pnl: float
    bounded: bool
    monotone_increasing: bool

    def to_dict(self) -> dict:
        return {
            "push_weights": list(self.push_weights),
            "impact_pnl": list(self.impact_pnl),
            "peak_push_weight": self.peak_push_weight,
            "peak_impact_pnl": self.peak_impact_pnl,
            "bounded": self.bounded,
            "monotone_increasing": self.monotone_increasing,
            "disclaimer": DISCLAIMER,
        }


# -- policies ---------------------------------------------------------------

BarPolicy = Callable[[dict, int], np.ndarray]


def pump_and_dump_schedule(p: ManipulationParams) -> Callable[[int], float]:
    """The round trip as a bar-indexed target weight: ramp up, hold, ramp back to flat.

    Deliberately crude. A schedule this legible is the point of a red-team probe: if even
    this pays, the impact model is the problem, and if a scoring layer's process gates
    cannot flag this, the gates are the problem.
    """
    a = p.start_bar
    b = a + p.push_bars
    c = b + p.hold_bars
    d = c + p.dump_bars

    def weight(bar: int) -> float:
        if bar < a or bar >= d:
            return 0.0
        if bar < b:
            return p.push_weight * (bar - a + 1) / p.push_bars
        if bar < c:
            return p.push_weight
        return p.push_weight * (1.0 - (bar - c + 1) / p.dump_bars)

    return weight


def momentum_follower_policy(gain: float, max_weight: float) -> BarPolicy:
    """A follower that chases the last cleared move: the crowd a pump needs to exist.

    Target weight is ``clip(gain * last_bar_return, -max_weight, max_weight)`` on the first
    symbol. Stateful only in remembering the previous close, which it reads from its own
    observation, so it never sees another agent's pending order.
    """
    prev: dict[str, float] = {}

    def policy(obs: dict, bar: int) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        out = np.zeros(closes.shape, dtype=np.float32)
        last = prev.get("close")
        prev["close"] = float(closes[0])
        if last is None or last <= 0.0:
            return out
        ret = float(closes[0]) / last - 1.0
        out[0] = float(np.clip(gain * ret, -max_weight, max_weight))
        return out

    return policy


# -- the probe --------------------------------------------------------------


def _rollout(p: ManipulationParams, seed: int) -> tuple[float, float]:
    """Run one episode and return ``(manipulator_pnl, peak_fractional_price_move)``.

    P&L is the manipulator's end-of-episode NAV less its starting capital. The peak price
    move is the largest fractional excursion of the first symbol's cleared mid away from
    its opening level, which is the diagnostic's read on how far the push actually got.
    """
    env = EndogenousMarketEnv(
        n_agents=1 + p.n_followers,
        n_symbols=p.n_symbols,
        n_days=p.n_days,
        seed=int(seed),
        capital=p.capital,
        kyle_lambda=p.kyle_lambda,
        eta=p.eta,
        volume_scale=p.volume_scale,
        distribution_mode=p.distribution_mode,
        max_weight=p.max_weight,
        impact_exponent=p.impact_exponent,
    )
    obs, _ = env.reset(seed=int(seed))
    n_symbols = len(env.symbols)
    weight_at = pump_and_dump_schedule(p)
    followers = {
        a: momentum_follower_policy(p.follower_gain, p.max_weight)
        for a in env.possible_agents
        if a != _MANIPULATOR
    }

    open_mid: Optional[float] = None
    peak_move = 0.0
    nav = p.capital
    bar = 0
    while env.agents:
        actions: dict[str, np.ndarray] = {}
        for agent in env.agents:
            if agent == _MANIPULATOR:
                vec = np.zeros(n_symbols, dtype=np.float32)
                vec[0] = weight_at(bar)
                actions[agent] = vec
            else:
                actions[agent] = followers[agent](obs[agent], bar)
        obs, _rewards, _terms, _truncs, infos = env.step(actions)
        info = infos.get(_MANIPULATOR)
        if info is not None:
            nav = float(info["nav"])
            mid = float(np.asarray(info["cleared_mids"], dtype=np.float64).reshape(-1)[0])
            if open_mid is None:
                open_mid = mid
            elif open_mid > 0.0:
                peak_move = max(peak_move, abs(mid / open_mid - 1.0))
        bar += 1

    env.close()
    return nav - p.capital, peak_move


def run_manipulation_probe(
    *, params: Optional[ManipulationParams] = None, seed: int = 0
) -> ManipulationResult:
    """Score one push-and-unwind round trip against its own zero-impact reference.

    The live run and the reference run share the seed, the schedule and the follower
    population; the reference differs only in that ``kyle_lambda`` and ``eta`` are zero, so
    flow cannot move price there. The difference in the manipulator's P&L is what moving
    the price was worth. See the module docstring: this is a diagnostic, not a strategy.
    """
    p = params or ManipulationParams()
    live_pnl, peak_move = _rollout(p, seed)
    reference_pnl, _ = _rollout(replace(p, kyle_lambda=0.0, eta=0.0), seed)
    impact_pnl = live_pnl - reference_pnl
    return ManipulationResult(
        seed=int(seed),
        profitable=impact_pnl > 0.0,
        impact_pnl=impact_pnl,
        live_pnl=live_pnl,
        reference_pnl=reference_pnl,
        peak_price_move=peak_move,
        kyle_lambda=p.kyle_lambda,
        eta=p.eta,
        push_weight=p.push_weight,
        follower_gain=p.follower_gain,
        impact_exponent=p.impact_exponent,
    )


def _mean_impact_pnl(p: ManipulationParams, seeds: Sequence[int]) -> float:
    """Mean impact-attributable P&L over ``seeds``. A single seed is a coin flip about the
    exogenous path; the boundary should not be drawn on one."""
    if not seeds:
        raise ValueError("at least one seed is required")
    return sum(run_manipulation_probe(params=p, seed=s).impact_pnl for s in seeds) / len(
        seeds
    )


def impact_boundary_sweep(
    *,
    params: Optional[ManipulationParams] = None,
    axis: str = "kyle_lambda",
    values: Optional[Sequence[float]] = None,
    seeds: Sequence[int] = (0, 1, 2),
) -> BoundaryReport:
    """Sweep one impact parameter and report where the pump stops paying.

    ``axis`` names a field of :class:`ManipulationParams` to vary, normally
    ``kyle_lambda`` (permanent impact), ``eta`` (temporary impact) or ``follower_gain``
    (how hard the crowd chases). The crossing between the last profitable value and the
    first unprofitable one is interpolated linearly, which is a reading of a coarse grid
    and should be quoted as such.

    This is the module's most useful output. "Manipulation paid" is a fact about one
    parameter setting. "Manipulation stopped paying above this much permanent impact" is a
    statement about how realistic the simulator is.
    """
    p = params or ManipulationParams()
    if not hasattr(p, axis):
        raise ValueError(f"unknown axis {axis!r}")
    grid = tuple(
        float(v)
        for v in (values if values is not None else (0.0, 0.05, 0.1, 0.2, 0.4, 0.8))
    )
    pnls = tuple(_mean_impact_pnl(replace(p, **{axis: v}), seeds) for v in grid)
    profitable = tuple(v > 0.0 for v in pnls)

    boundary: Optional[float] = None
    for i in range(len(grid) - 1):
        if profitable[i] and not profitable[i + 1]:
            lo, hi = pnls[i], pnls[i + 1]
            span = lo - hi
            frac = lo / span if span != 0.0 else 0.0
            boundary = grid[i] + frac * (grid[i + 1] - grid[i])
            break

    return BoundaryReport(
        axis=axis,
        values=grid,
        impact_pnl=pnls,
        profitable=profitable,
        boundary=boundary,
        profitable_everywhere=all(profitable),
        unprofitable_everywhere=not any(profitable),
    )


def size_response(
    *,
    params: Optional[ManipulationParams] = None,
    push_weights: Optional[Sequence[float]] = None,
    seeds: Sequence[int] = (0, 1, 2),
) -> SizeResponse:
    """Sweep the size of the push and report whether the payoff is bounded.

    An impact specification is realistic only if pushing harder eventually costs more than
    it returns. ``monotone_increasing`` is the failure signal: profit that still rises at
    the largest size tested has no visible ceiling in this range, which means either the
    grid is too narrow or the model lets a large enough agent print money.
    """
    p = params or ManipulationParams()
    grid = tuple(
        float(w)
        for w in (
            push_weights
            if push_weights is not None
            else (0.1, 0.25, 0.5, 0.75, 1.0)
        )
    )
    pnls = tuple(_mean_impact_pnl(replace(p, push_weight=w), seeds) for w in grid)
    peak = max(range(len(pnls)), key=lambda i: pnls[i])
    monotone = all(pnls[i] < pnls[i + 1] for i in range(len(pnls) - 1))
    return SizeResponse(
        push_weights=grid,
        impact_pnl=pnls,
        peak_push_weight=grid[peak],
        peak_impact_pnl=pnls[peak],
        bounded=not monotone,
        monotone_increasing=monotone,
    )


__all__ = [
    "DISCLAIMER",
    "BoundaryReport",
    "ManipulationParams",
    "ManipulationResult",
    "SizeResponse",
    "impact_boundary_sweep",
    "momentum_follower_policy",
    "pump_and_dump_schedule",
    "run_manipulation_probe",
    "size_response",
]
