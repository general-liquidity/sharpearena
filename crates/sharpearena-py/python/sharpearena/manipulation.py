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
import math
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
        if self.n_symbols < 1 or self.n_days < 1:
            raise ValueError("n_symbols and n_days must be >= 1")
        if min(self.push_bars, self.dump_bars) < 1:
            raise ValueError("push_bars and dump_bars must be >= 1")
        if self.hold_bars < 0:
            raise ValueError("hold_bars must be >= 0")
        if self.start_bar < 0:
            raise ValueError("start_bar must be >= 0")
        if not (math.isfinite(self.impact_exponent) and self.impact_exponent > 0.0):
            raise ValueError("impact_exponent must be positive (1.0 = linear)")
        for name, value, lower, inclusive in (
            ("capital", self.capital, 0.0, False),
            ("volume_scale", self.volume_scale, 0.0, False),
            ("max_weight", self.max_weight, 0.0, False),
            ("kyle_lambda", self.kyle_lambda, 0.0, True),
            ("eta", self.eta, 0.0, True),
            ("push_weight", self.push_weight, 0.0, True),
            ("follower_gain", self.follower_gain, 0.0, True),
        ):
            if not math.isfinite(value) or (value < lower if inclusive else value <= lower):
                relation = "non-negative" if inclusive else "positive"
                raise ValueError(f"{name} must be a finite {relation} number")
        if self.push_weight > self.max_weight:
            raise ValueError("push_weight must not exceed max_weight")
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


@dataclass(frozen=True)
class AsymmetricSchedule:
    """An asymmetric round trip: accumulate over ``up_bars``, unwind over ``down_bars``.

    The positive-control family for the concave-impact ablation. Under the
    nonlinear permanent-impact assumptions studied in the cited manipulation
    literature, asymmetric round trips are a relevant failure mode. For a
    concave power law, the *sum of impact increments* from splitting normalized
    flow into ``n`` equal pieces is ``n**(1 - exponent)`` times the one-block
    increment; price compounding and temporary impact still determine whether a
    particular simulated trip is profitable. The symmetric
    :func:`pump_and_dump_schedule` never searches this shape; this family does.

    ``up_bars`` / ``down_bars`` set the duration ratio of the two legs. ``size_split`` is
    the legacy shared block fraction: the share of each leg's notional executed in a single
    bar at the turn of the trip (the last pump bar and the first dump bar), with the
    remainder spread uniformly over the leg's other bars. ``up_size_split`` and
    ``down_size_split`` override it per leg. They matter for unequal durations: a uniform
    9:1 trip has fractions 1/9 and 1, not one shared 1/9 fraction that silently leaves the
    one-bar liquidation incomplete. A one-bar leg is a block regardless of its configured
    split. Everything else about the trip (start bar, hold, peak weight) comes from the
    :class:`ManipulationParams` it is paired with, and the trip is still flat-to-flat, which
    is what keeps ``impact_pnl`` an attribution rather than a directional bet.
    """

    up_bars: int
    down_bars: int
    size_split: float
    up_size_split: float | None = None
    down_size_split: float | None = None

    def __post_init__(self) -> None:
        if min(self.up_bars, self.down_bars) < 1:
            raise ValueError("up_bars and down_bars must be >= 1")
        for name, value in (
            ("size_split", self.size_split),
            ("up_size_split", self.up_size_split),
            ("down_size_split", self.down_size_split),
        ):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")

    @property
    def up_split(self) -> float:
        return 1.0 if self.up_bars == 1 else (self.up_size_split or self.size_split)

    @property
    def down_split(self) -> float:
        return 1.0 if self.down_bars == 1 else (self.down_size_split or self.size_split)

    @classmethod
    def uniform(cls, up_bars: int, down_bars: int) -> "AsymmetricSchedule":
        """Linear ramps of asymmetric duration (no block concentration on either leg)."""
        return cls(
            up_bars=up_bars,
            down_bars=down_bars,
            # Keep the legacy field meaningful for serialisers while making the actual
            # uniform increments explicit and independently normalized per leg.
            size_split=1.0 / max(up_bars, down_bars),
            up_size_split=1.0 / up_bars,
            down_size_split=1.0 / down_bars,
        )

    @property
    def duration_ratio(self) -> float:
        return self.up_bars / self.down_bars

    def to_dict(self) -> dict:
        return {
            "up_bars": self.up_bars,
            "down_bars": self.down_bars,
            "size_split": self.size_split,
            "up_size_split": self.up_split,
            "down_size_split": self.down_split,
            "duration_ratio": self.duration_ratio,
        }


def asymmetric_round_trip_schedule(
    p: ManipulationParams, schedule: AsymmetricSchedule, side: int = 1
) -> Callable[[int], float]:
    """The asymmetric round trip as a bar-indexed target weight, flat before and after.

    ``p.push_bars`` and ``p.dump_bars`` are ignored; the leg lengths come from
    ``schedule``. ``p.start_bar``, ``p.hold_bars`` and ``p.push_weight`` are honoured.

    ``side`` is ``+1`` for the long round trip (accumulate long, liquidate) and ``-1`` for
    its mirror (accumulate short, buy back). The mirror is the same path with the sign
    flipped bar for bar, so it is still flat-to-flat and the attribution still measures
    what moving the price was worth. Whether the two sides pay the same is an empirical
    question about the engine's multiplicative price compounding, not a property of the
    schedule.
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 (long round trip) or -1 (short round trip)")
    span = p.start_bar + schedule.up_bars + p.hold_bars + schedule.down_bars
    if span >= p.n_days:
        raise ValueError("the asymmetric round trip must finish inside the episode")
    a = p.start_bar
    b = a + schedule.up_bars
    c = b + p.hold_bars
    d = c + schedule.down_bars
    n_up, n_down = schedule.up_bars, schedule.down_bars

    def _leg_increments(n: int, split: float, block_first: bool) -> tuple[float, ...]:
        if n == 1:
            return (1.0,)
        remainder = (1.0 - split) / (n - 1)
        out = (split,) + (remainder,) * (n - 1) if block_first else (remainder,) * (n - 1) + (split,)
        assert math.isclose(sum(out), 1.0, rel_tol=0.0, abs_tol=1e-15)
        return out

    # Each leg executes exactly one peak position in opposite directions. This invariant
    # prevents duration sweeps from accidentally changing total liquidation notional.
    up_increments = _leg_increments(n_up, schedule.up_split, block_first=False)
    down_increments = _leg_increments(n_down, schedule.down_split, block_first=True)
    up_targets = np.cumsum(up_increments)
    down_targets = 1.0 - np.cumsum(down_increments)
    # Make the contractual endpoints exact rather than exposing harmless cumulative binary
    # roundoff (for example 0.8000000000000004) to a target-weight protocol.
    up_targets[-1] = 1.0
    down_targets[-1] = 0.0
    assert math.isclose(float(up_targets[-1]), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(float(down_targets[-1]), 0.0, rel_tol=0.0, abs_tol=1e-15)

    def weight(bar: int) -> float:
        if bar < a or bar >= d:
            return 0.0
        if bar < b:
            return side * p.push_weight * float(up_targets[bar - a])
        if bar < c:
            return side * p.push_weight
        return side * p.push_weight * float(down_targets[bar - c])

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


def _rollout(
    p: ManipulationParams,
    seed: int,
    weight_at: Optional[Callable[[int], float]] = None,
) -> tuple[float, float]:
    """Run one episode and return ``(manipulator_pnl, peak_fractional_price_move)``.

    P&L is the manipulator's end-of-episode NAV less its starting capital. The peak price
    move is the largest fractional excursion of the first symbol's cleared mid away from
    its opening level, which is the diagnostic's read on how far the push actually got.
    ``weight_at`` defaults to the symmetric :func:`pump_and_dump_schedule`.
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
    if weight_at is None:
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


@dataclass(frozen=True)
class AsymmetricResult(ManipulationResult):
    """A :class:`ManipulationResult` for an asymmetric round trip, carrying its shape."""

    up_bars: int = 0
    down_bars: int = 0
    size_split: float = 0.0
    side: int = 1

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update(
            up_bars=self.up_bars,
            down_bars=self.down_bars,
            size_split=self.size_split,
            duration_ratio=self.up_bars / self.down_bars if self.down_bars else None,
            side=self.side,
        )
        return out


def run_asymmetric_probe(
    *,
    params: Optional[ManipulationParams] = None,
    schedule: AsymmetricSchedule,
    seed: int = 0,
    side: int = 1,
) -> AsymmetricResult:
    """Score one asymmetric round trip against its own zero-impact reference.

    Identical to :func:`run_manipulation_probe` except that the manipulator follows
    :func:`asymmetric_round_trip_schedule` for ``schedule`` on ``side`` (``+1`` long,
    ``-1`` the mirrored short round trip). Same attribution, same pairing, same
    followers, same seed on both legs. This is the positive-control probe: the shape
    under which theory permits a profit under concave permanent impact, so a profit
    found here under an exponent below one and not under the linear exponent is the
    instrument firing where it should and staying silent where it should.
    """
    p = params or ManipulationParams()
    weight_at = asymmetric_round_trip_schedule(p, schedule, side)
    live_pnl, peak_move = _rollout(p, seed, weight_at)
    reference_pnl, _ = _rollout(replace(p, kyle_lambda=0.0, eta=0.0), seed, weight_at)
    impact_pnl = live_pnl - reference_pnl
    return AsymmetricResult(
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
        up_bars=schedule.up_bars,
        down_bars=schedule.down_bars,
        size_split=schedule.size_split,
        side=side,
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
    "AsymmetricResult",
    "AsymmetricSchedule",
    "BoundaryReport",
    "ManipulationParams",
    "ManipulationResult",
    "SizeResponse",
    "asymmetric_round_trip_schedule",
    "impact_boundary_sweep",
    "momentum_follower_policy",
    "pump_and_dump_schedule",
    "run_asymmetric_probe",
    "run_manipulation_probe",
    "size_response",
]
