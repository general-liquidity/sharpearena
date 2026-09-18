"""Rank-neutral impact diagnostics for the endogenous shared-book market.

Two measurements over :class:`~sharpearena.sharpearena_py.PyMarketClearing`, the native
clearing engine behind :class:`~sharpearena.market_env.EndogenousMarketEnv`. Both drive the
engine directly, so neither needs ``pettingzoo``.

**Rank-neutral.** Nothing here feeds SharpeBench, the kernel score, a leaderboard, a
pass^k gate or any rank. Both results are diagnostics a report may quote next to a score.
Neither changes the clearing arithmetic, ``SPEC_HASH`` or any golden file: every run
goes through the published binding with its existing arguments.

1. :func:`impact_misspecification_gap`, after "Robust Reinforcement Learning in Finance:
   Modeling Market Impact with Elliptic Uncertainty Sets" (Ma and Huang, 2025). That
   paper judges robustness by how far a policy's value moves when market impact changes.
   SharpeArena can clear each bar at the worst case inside an elliptic uncertainty set over
   the impact coefficients (``EllipticUncertaintySet`` in ``sharpearena::market``), which
   is an adaptation of the paper's set, not its construction. Clearing that way does not
   say whether a policy's profit depended on impact being exactly the point estimate. This
   report runs the same policy twice on each seed, once at the point estimate and once
   against a declared set, and returns per-seed and aggregate return and Sharpe under each
   arm and their difference. The policy trades alone (one agent), so the worst case is
   resolved against its own flow. The linear kernel (``impact_exponent = 1.0``) is used
   throughout, because the pairing proof below replays its multiplier exactly.

2. :func:`meta_order_impact_shape`, prompted by "When AI Trading Agents Compete" (pp. 3-4),
   where impact grows as a square root of the executed quantity during a meta-order and
   decays after it. The probe measures what the SharpeArena kernel does instead (next
   section).

**Pairing is checked, not assumed.** Each arm runs on its own market object. Before the
policy runs, that object is replayed with flat orders: with zero flow the permanent
multiplier stays exactly ``1.0``, so the replayed cleared mids are the object's exogenous
mids bit for bit. The object is then reset (``reset`` never touches the exogenous path)
and the policy is run. :func:`check_arm_pairing` refuses with :class:`UnpairedArmsError`
unless all of the following hold:

* both arms were built with the same market settings;
* the two exogenous tapes, and the burn-in closes each arm's first observation shows, are
  bitwise identical;
* each arm's traded tape equals ``exogenous_mid * M`` bit for bit, where ``M`` is rebuilt
  from the arm's own reported net flow and applied ``lambda`` in the kernel's float order
  (``M = M * (1.0 + lambda * Q / V)``).

The replays are compared exactly. The last check ties the tape the policy actually traded
on to its replay: a traded tape could pass against a different path only if, on every
bar, the two exogenous mids differ by less than one rounding step of the product
``exogenous_mid * M``. On the first traded bar ``M`` is exactly ``1.0``, so that bar
must match exactly.

**Degenerate inputs are typed.** A zero-radius set resolves every bar to the point
estimate, and a policy that never trades moves nothing, so in both cases the two arms
are bitwise identical and ``return_gap`` is exactly ``0.0``. A reward track whose values
are all exactly equal has no Sharpe ratio and reports :class:`Unavailable` with
:data:`CONSTANT_TRACK`; a track that is dispersed but whose standard deviation underflows
to zero, and one carrying a non-finite bar, have none either and report
:data:`NON_FINITE_SHARPE`, which is what the kernel refuses under that name. A Sharpe gap
needs both Sharpe ratios, so it is :data:`SHARPE_UNAVAILABLE` whenever either arm lacks
one, identical arms included, and the
across-seed mean and interval then name the seed positions without one instead of
averaging structural zeros. Fewer than two seeds give no dispersion estimate, and the
interval reports :data:`INSUFFICIENT_SEEDS`. A cleared mid that is not positive (the
linear multiplier crossed zero) refuses the arm with :class:`NonPositiveMidError`.

**Which sign is guaranteed.** Write ``return_gap = point_return - robust_return``. Both
returns mark the final position at the exogenous mid, so a positive gap is value the worst
case cost the policy. The engine's NAV marks at the cleared mid, which carries the arm's
own permanent impact; each row reports that difference as ``*_own_impact_mark``.

* *Eta-only set* (``lambda_radius == 0``). The resolved ``lambda`` is bitwise the point
  estimate. If the policy sends the same weights in both arms, the permanent multiplier,
  the cleared mids, the order sizes (``capital * dw / mid``) and the volatility factor are
  identical in both arms, and only ``eta`` changes: on a bar that trades it resolves to
  about ``eta + eta_radius`` (never below ``eta``). At a positive mid each fill is then at
  least as expensive, so cash is never higher, the shares are equal, and the gap is
  ``>= 0`` on every seed (``> 0`` in exact arithmetic once anything trades). Rounding is
  monotone at every step, so the weak inequality also holds in floating point. Both
  premises are checked. Mids that are not positive are refused, and ``identical_actions``
  records whether the weights matched. A policy that reads ``cash`` or ``avg_price`` (the
  only observation fields that differ between arms), or draws from state kept outside its
  factory, can send different weights, and the report's ``sign_guaranteed`` is then false.
* *Any set with* ``lambda_radius > 0``. No sign is guaranteed: the worst-case ``lambda``
  moves the cleared mids, so weights, sizes and later fills can differ between arms. The
  exogenous mark removes one channel that favours the worst case. With ``vol_scale = 0``, a
  single buy of ``q`` shares held to the end has ``d NAV / d lambda = q**2 / V * (exo_T -
  exo_0)`` under the engine's mark, so the engine's NAV ends higher under the worst case
  on a rising path, while the exogenously marked NAV has slope ``-q**2 / V * exo_0``, the
  extra cost of the fill. After ``k`` equal weight steps bought at a flat exogenous mid
  and held until that mid ends at ``r`` times its entry value, the engine-marked slope at
  ``lambda = eta = 0`` is positive unless ``r < 2 / (k + 1)``, while the exogenously
  marked slope is negative for every ``r``. Over
  seeds 0 to 31 (one symbol, 60 days), for a single buy and for ten-bar long and short
  scale-ins held to the end, the engine-marked gap under ``lambda_radius = 0.05`` was
  negative on 19 of 32 seeds (the rising paths) and on 32 of 32 respectively; the
  exogenously marked gap was positive on every seed, as it was for a ten-bar round trip
  and an alternating policy. The report does not clamp a negative gap.

**What the meta-order probe measures.** The kernel's permanent impact is
``M_{t+1} = M_t * (1 + lambda * g(Q_t / V))`` with ``g(x) = sign(x) * |x|**beta``. Each
bar's increment is concave in that bar's flow when ``beta < 1``, but increments compound
and never decay. For ``k`` bars of equal size ``q``, ``M_k - 1 ~ k * lambda * g(q / V)``
while the executed quantity is ``X_k = k * q``, so impact grows about linearly in ``X``
(log-log exponent near ``1``) at every ``beta``. A fixed total ``X`` spread over ``D`` bars
gives ``D * lambda * (X / (D * V))**beta``, which grows as ``D**(1 - beta)``. Once
execution stops, ``M`` is unchanged bit for bit, so impact never relaxes. The empirical
square-root law instead has an exponent near ``0.5`` in ``X``, little dependence on ``D``
and decay after execution, so ``beta < 1`` is not that law. The probe reports all three
numbers so a caller can check them. A transient-impact kernel is future work.

Determinism: the engine is seeded and has no clock or RNG, and every aggregate here is
an exactly rounded sum (``math.fsum``). A report reproduces byte for byte from its
arguments on one platform. The logarithms and square roots in the fits and intervals are
diagnostics, not kernel arithmetic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from .market_env import _normalize_uncertainty
from .sharpearena_py import PyMarketClearing

CONSTANT_TRACK = "constant_track"
"""A reward track whose values are all exactly equal: no Sharpe ratio exists."""

TOO_FEW_BARS = "too_few_bars"
"""Fewer than two traded bars: no sample standard deviation exists."""

INSUFFICIENT_SEEDS = "insufficient_seeds"
"""Fewer than two seeds: no across-seed dispersion estimate exists."""

SHARPE_UNAVAILABLE = "sharpe_unavailable"
"""A Sharpe gap or mean needs a Sharpe ratio that at least one arm or seed lacks."""

NON_FINITE_SHARPE = "non_finite_sharpe"
"""A non-finite bar, or a standard deviation that underflows to zero on a track whose
values differ: the ratio is not finite, so there is no Sharpe ratio. Matches the kernel's
`Sharpe ratio is not finite` in `leaderboard_ci::check_sharpe_defined`."""

Policy = Callable[[Mapping[str, Any], int], Sequence[float]]
"""``policy(observation, bar) -> target weights``. ``observation`` is the engine's wire
``MarketObservation`` dict (``date``, ``cash``, ``symbols``, ``portfolio``); ``bar`` counts
traded bars from zero."""


class ImpactDiagnosticError(ValueError):
    """An impact diagnostic was given an input it cannot measure."""


class UnpairedArmsError(ImpactDiagnosticError):
    """Two arms of a paired report did not trade on the same exogenous path.

    ``seed`` is the seed label the arms carried and ``check`` names the failed
    comparison.
    """

    def __init__(self, seed: int, check: str, detail: str) -> None:
        super().__init__(f"seed {seed}: {check}: {detail}")
        self.seed = seed
        self.check = check


class NonPositiveMidError(ImpactDiagnosticError):
    """An arm cleared a symbol at a mid that is not positive.

    The linear permanent multiplier ``M * (1 + lambda * Q / V)`` crosses zero when one
    bar's flow is large enough against ``V``. Past that point a fill's impact cost changes
    sign and no return or gap means anything, so the arm is refused. ``bar`` is the traded
    bar, or ``None`` for the flat replay.
    """

    def __init__(self, seed: int, bar: Optional[int], symbol: str, mid: float) -> None:
        where = "flat replay" if bar is None else f"bar {bar}"
        super().__init__(f"seed {seed}: {where}: symbol {symbol} cleared at {mid!r}")
        self.seed = seed
        self.bar = bar
        self.symbol = symbol
        self.mid = mid


@dataclass(frozen=True)
class Unavailable:
    """A statistic that does not exist for this input, with the reason code."""

    reason: str
    detail: str


Statistic = Union[float, Unavailable]


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------


#: Knobs both entry points scale the market by. Zero or negative makes the cleared mid,
#: and so every number derived from it, meaningless rather than merely extreme.
_POSITIVE_KNOBS = ("capital", "volume_scale")


def _positive(name: str, value: float) -> float:
    """Refuse a knob that is not finite and positive, in one place, so the paired report
    and the meta-order probe cannot drift into two standards for the same knob."""

    if not (math.isfinite(value) and value > 0.0):
        raise ImpactDiagnosticError(f"{name} must be finite and positive, got {value!r}")
    return value


@dataclass(frozen=True)
class MarketSettings:
    """The native market settings both arms of a paired report share."""

    n_symbols: int = 4
    n_days: int = 120
    capital: float = 1.0
    kyle_lambda: float = 0.1
    eta: float = 0.05
    volume_scale: float = 1.0
    vol_scale: float = 0.0
    distribution_mode: str = "calm"
    richness: str = "standard"

    def validated(self) -> "MarketSettings":
        for name in _POSITIVE_KNOBS:
            _positive(name, getattr(self, name))
        for name in ("kyle_lambda", "eta", "vol_scale"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ImpactDiagnosticError(
                    f"{name} must be finite and non-negative, got {value!r}"
                )
        return self


@dataclass(frozen=True)
class ArmTrace:
    """Everything one arm observed, as recorded from the engine's wire output.

    Per-bar tuples run over traded bars; per-symbol rows are in the engine's sorted symbol
    order. ``exogenous_mids`` comes from the flat replay of the same market object and
    ``applied_lambda`` is the permanent coefficient each bar cleared at. ``weights`` are
    the target weights the policy sent on each bar, and ``final_cash`` and
    ``final_shares`` are the book after the last bar, as the engine's last observation
    reports them.
    """

    seed: int
    settings: MarketSettings
    uncertainty: Optional[tuple[float, float, float]]
    symbols: tuple[str, ...]
    n_bars: int
    start_bar: int
    initial_closes: tuple[tuple[float, ...], ...]
    exogenous_mids: tuple[tuple[float, ...], ...]
    cleared_mids: tuple[tuple[float, ...], ...]
    net_flow: tuple[tuple[float, ...], ...]
    applied_lambda: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    navs: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    final_cash: float
    final_shares: tuple[float, ...]


def _market(settings: MarketSettings, seed: int, uncertainty) -> PyMarketClearing:
    kwargs: dict[str, Any] = dict(
        n_symbols=settings.n_symbols,
        n_days=settings.n_days,
        seed=int(seed),
        n_agents=1,
        capital=settings.capital,
        kyle_lambda=settings.kyle_lambda,
        eta=settings.eta,
        volume_scale=settings.volume_scale,
        vol_scale=settings.vol_scale,
        distribution_mode=settings.distribution_mode,
        richness=settings.richness,
    )
    if uncertainty is not None:
        lam, eta_r, rho = uncertainty
        kwargs.update(lambda_radius=lam, eta_radius=eta_r, uncertainty_correlation=rho)
    return PyMarketClearing(**kwargs)


def _weights(action: Any, n_symbols: int, bar: int) -> list[float]:
    weights = [float(w) for w in np.asarray(action, dtype=np.float64).reshape(-1)]
    if len(weights) != n_symbols:
        raise ImpactDiagnosticError(
            f"policy returned {len(weights)} weights at bar {bar}, expected {n_symbols}"
        )
    if not all(math.isfinite(w) for w in weights):
        raise ImpactDiagnosticError(f"policy returned a non-finite weight at bar {bar}")
    return weights


def _closes(observation: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(c) for c in s["close_history"]) for s in observation["symbols"])


def _positive_mids(
    mids: tuple[float, ...], symbols: tuple[str, ...], seed: int, bar: Optional[int]
) -> tuple[float, ...]:
    for symbol, mid in zip(symbols, mids):
        if not mid > 0.0:
            raise NonPositiveMidError(seed, bar, symbol, mid)
    return mids


def run_impact_arm(
    make_policy: Callable[[], Policy],
    seed: int,
    uncertainty: Any = None,
    settings: Optional[MarketSettings] = None,
) -> ArmTrace:
    """Run one arm on one seed: a flat replay for the exogenous path, then the policy.

    ``make_policy`` is called once, after the replay, so a policy whose state lives in the
    factory's closure starts each arm from the same state. State shared outside the
    factory (a module-level random generator, for example) is not reset, and the two arms
    then send different weights; the report shows that as ``identical_actions``.
    ``uncertainty`` takes the forms :class:`~sharpearena.market_env.EndogenousMarketEnv`
    accepts; ``None`` is the point estimate. Raises :class:`NonPositiveMidError` at the
    first mid that is not positive, in the replay or in the traded run.
    """
    s = (settings or MarketSettings()).validated()
    norm = _normalize_uncertainty(uncertainty)
    market = _market(s, seed, norm)
    meta = json.loads(market.reset_market())
    symbols = tuple(meta["symbols"])
    flat = json.dumps([[0.0] * len(symbols)])
    exogenous: list[tuple[float, ...]] = []
    while True:
        step = json.loads(market.step_market(flat))
        if any(flow != 0.0 for flow in step["net_flow"]):
            raise ImpactDiagnosticError(
                f"flat replay of seed {seed} reported net flow {step['net_flow']!r}"
            )
        mids = tuple(float(m) for m in step["cleared_mids"])
        exogenous.append(_positive_mids(mids, symbols, seed, None))
        if step["done"]:
            break

    start = json.loads(market.reset_market())
    policy = make_policy()
    observation = start["observations"][0]
    cleared, flows, lambdas, rewards, navs, sent = [], [], [], [], [], []
    bar = 0
    while True:
        weights = _weights(policy(observation, bar), len(symbols), bar)
        sent.append(tuple(weights))
        step = json.loads(market.step_market(json.dumps([weights])))
        robust = step.get("robust_impact")
        if (robust is None) != (norm is None):
            raise ImpactDiagnosticError(
                f"seed {seed} bar {bar}: robust_impact presence does not match the "
                f"declared uncertainty {norm!r}"
            )
        mids = tuple(float(m) for m in step["cleared_mids"])
        cleared.append(_positive_mids(mids, symbols, seed, bar))
        flows.append(tuple(float(q) for q in step["net_flow"]))
        if robust is None:
            lambdas.append(tuple(s.kyle_lambda for _ in symbols))
        else:
            lambdas.append(tuple(float(c["lambda"]) for c in robust))
        rewards.append(float(step["rewards"][0]))
        navs.append(float(step["navs"][0]))
        observation = step["observations"][0]
        bar += 1
        if step["done"]:
            break

    held = {row["symbol"]: float(row["shares"]) for row in observation["portfolio"]}
    return ArmTrace(
        seed=int(seed),
        settings=s,
        uncertainty=norm,
        symbols=symbols,
        n_bars=int(start["n_bars"]),
        start_bar=int(start["start_bar"]),
        initial_closes=_closes(start["observations"][0]),
        exogenous_mids=tuple(exogenous),
        cleared_mids=tuple(cleared),
        net_flow=tuple(flows),
        applied_lambda=tuple(lambdas),
        rewards=tuple(rewards),
        navs=tuple(navs),
        weights=tuple(sent),
        final_cash=float(observation["cash"]),
        final_shares=tuple(held[symbol] for symbol in symbols),
    )


# ---------------------------------------------------------------------------
# The pairing proof
# ---------------------------------------------------------------------------


def _bits(rows: Sequence[Sequence[float]]) -> list[list[str]]:
    return [[float(x).hex() for x in row] for row in rows]


def _check_traded_tape(trace: ArmTrace, arm: str) -> None:
    if len(trace.cleared_mids) != len(trace.exogenous_mids):
        raise UnpairedArmsError(
            trace.seed,
            f"{arm} tape length",
            f"{len(trace.cleared_mids)} traded bars against "
            f"{len(trace.exogenous_mids)} replayed bars",
        )
    first = [closes[-1] for closes in trace.initial_closes]
    if _bits([first]) != _bits([trace.exogenous_mids[0]]):
        raise UnpairedArmsError(
            trace.seed,
            f"{arm} first reference mid",
            f"observation shows {first!r}, replay shows {list(trace.exogenous_mids[0])!r}",
        )
    v = trace.settings.volume_scale
    mult = [1.0] * len(trace.symbols)
    rows = zip(trace.exogenous_mids, trace.cleared_mids, trace.net_flow, trace.applied_lambda)
    for bar, (exo, cleared, flow, lam) in enumerate(rows):
        for s, (e, c) in enumerate(zip(exo, cleared)):
            expected = e * mult[s]
            if expected.hex() != c.hex():
                raise UnpairedArmsError(
                    trace.seed,
                    f"{arm} traded tape",
                    f"bar {bar} symbol {trace.symbols[s]} cleared at {c!r}, but the "
                    f"replayed path and reported impact give {expected!r}",
                )
        for s in range(len(mult)):
            mult[s] = mult[s] * (1.0 + lam[s] * flow[s] / v)


def check_arm_pairing(point: ArmTrace, robust: ArmTrace) -> None:
    """Refuse unless both arms traded on the same exogenous path (see the module docs).

    Raises :class:`UnpairedArmsError` naming the failed comparison.
    """
    seed = point.seed
    if point.uncertainty is not None or robust.uncertainty is None:
        raise UnpairedArmsError(
            seed,
            "arm roles",
            f"point arm set {point.uncertainty!r}, robust arm set {robust.uncertainty!r}",
        )
    same = (
        ("seed label", point.seed, robust.seed),
        ("market settings", point.settings, robust.settings),
        ("symbols", point.symbols, robust.symbols),
        ("bar axis", (point.n_bars, point.start_bar), (robust.n_bars, robust.start_bar)),
    )
    for check, left, right in same:
        if left != right:
            raise UnpairedArmsError(seed, check, f"point {left!r}, robust {right!r}")
    if _bits(point.initial_closes) != _bits(robust.initial_closes):
        raise UnpairedArmsError(seed, "burn-in closes", "the first observations differ")
    if _bits(point.exogenous_mids) != _bits(robust.exogenous_mids):
        raise UnpairedArmsError(seed, "exogenous path", "the flat replays differ")
    _check_traded_tape(point, "point")
    _check_traded_tape(robust, "robust")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def track_sharpe(rewards: Sequence[float]) -> Statistic:
    """Per-bar Sharpe ratio ``mean / sd`` (sample ``sd``, unannualized) of a reward track.

    A track whose values are all exactly equal (for example a flat policy's zeros) is
    :data:`CONSTANT_TRACK`. The predicate is exact equality, not a variance threshold.

    Two further shapes have no Sharpe ratio and report :data:`NON_FINITE_SHARPE`: a track
    carrying a non-finite bar, and a dispersed track whose sample standard deviation
    underflows to exactly zero, where the ratio is not finite. The order of the checks
    follows ``leaderboard_ci::check_sharpe_defined``, which is the definition: the
    non-finite test runs before value equality, because NaN never equals itself.
    """
    r = [float(x) for x in rewards]
    if len(r) < 2:
        return Unavailable(TOO_FEW_BARS, f"{len(r)} traded bars")
    if any(not math.isfinite(x) for x in r):
        return Unavailable(NON_FINITE_SHARPE, "a bar returned a non-finite value")
    if all(x == r[0] for x in r):
        return Unavailable(CONSTANT_TRACK, f"every bar returned {r[0]!r}")
    mean = _mean(r)
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in r) / (len(r) - 1))
    if sd == 0.0 or not math.isfinite(mean / sd):
        # A track can be dispersed and still have no Sharpe ratio: squaring deviations
        # this small underflows, so the sample standard deviation is exactly zero. The
        # kernel is the definition and already refuses this by name, in
        # `leaderboard_ci::check_sharpe_defined` and its pinned
        # `an_underflowing_track_is_withheld_as_a_non_finite_sharpe`.
        return Unavailable(NON_FINITE_SHARPE, f"mean {mean!r} over standard deviation {sd!r}")
    return mean / sd


def _betacf(a: float, b: float, x: float) -> float:
    # Lentz's continued fraction for the regularized incomplete beta function.
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-16:
            break
    return h


def _student_t_upper_tail(t: float, df: int) -> float:
    # P(T > t) for t >= 0, from the regularized incomplete beta I_x(df/2, 1/2).
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        ibeta = front * _betacf(a, b, x) / a
    else:
        ibeta = 1.0 - front * _betacf(b, a, 1.0 - x) / b
    return 0.5 * ibeta


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Two-sided Student-t critical value, found by bisection (no SciPy dependency)."""
    if df < 1:
        raise ImpactDiagnosticError(f"t_critical needs df >= 1, got {df}")
    if not 0.0 < confidence < 1.0:
        raise ImpactDiagnosticError(f"confidence must lie in (0, 1), got {confidence!r}")
    tail = (1.0 - confidence) / 2.0
    lo, hi = 0.0, 1.0
    while _student_t_upper_tail(hi, df) > tail:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_upper_tail(mid, df) > tail:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class GapInterval:
    """Across-seed mean of a paired gap with a two-sided Student-t interval."""

    n: int
    mean: float
    sd: float
    se: float
    t_critical: float
    lo: float
    hi: float
    confidence: float


def gap_interval(gaps: Sequence[Statistic], confidence: float = 0.95) -> Union[GapInterval, Unavailable]:
    """t-based interval over per-seed gaps, or :class:`Unavailable` when none exists."""
    missing = [i for i, g in enumerate(gaps) if isinstance(g, Unavailable)]
    if missing:
        return Unavailable(SHARPE_UNAVAILABLE, f"no gap at seed positions {missing}")
    values = [float(g) for g in gaps]
    if len(values) < 2:
        return Unavailable(INSUFFICIENT_SEEDS, f"{len(values)} seed, a dispersion estimate needs 2")
    mean = _mean(values)
    sd = math.sqrt(math.fsum((g - mean) ** 2 for g in values) / (len(values) - 1))
    se = sd / math.sqrt(len(values))
    t = t_critical(len(values) - 1, confidence)
    return GapInterval(
        n=len(values),
        mean=mean,
        sd=sd,
        se=se,
        t_critical=t,
        lo=mean - t * se,
        hi=mean + t * se,
        confidence=confidence,
    )


def _mean_statistic(values: Sequence[Statistic]) -> Statistic:
    missing = [i for i, v in enumerate(values) if isinstance(v, Unavailable)]
    if missing:
        return Unavailable(SHARPE_UNAVAILABLE, f"no value at seed positions {missing}")
    return _mean([float(v) for v in values])


# ---------------------------------------------------------------------------
# The paired report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedGap:
    """One seed's paired outcome. ``*_gap`` is point minus worst case.

    ``point_return`` and ``robust_return`` are final returns on ``capital`` with the held
    position marked at the exogenous mid, so neither includes the arm's own permanent
    impact on its mark. ``*_own_impact_mark`` is ``final_shares * (cleared_mid -
    exogenous_mid) / capital`` on the last bar, summed over symbols: what the engine's NAV,
    which marks at the cleared mid, adds on top. The engine's final return is the sum of
    the two, up to rounding. ``identical_arms`` compares the reward and NAV tracks bit for
    bit, and ``identical_actions`` the weights the policy sent. ``sharpe_gap`` is
    :class:`Unavailable` whenever either arm has no Sharpe ratio, identical arms included.
    """

    seed: int
    traded_bars: int
    identical_arms: bool
    identical_actions: bool
    point_return: float
    robust_return: float
    return_gap: float
    point_own_impact_mark: float
    robust_own_impact_mark: float
    point_sharpe: Statistic
    robust_sharpe: Statistic
    sharpe_gap: Statistic


@dataclass(frozen=True)
class ImpactGapReport:
    """Paired point-estimate versus worst-case report. Rank-neutral: see the module docs.

    ``return_gap`` is ``point_return - robust_return``: with both positions marked at the
    exogenous mid, a positive gap is value the worst case cost the policy. It compares the
    point estimate with the worst case of a set; the paper's relative portfolio gap
    compares a market with impact to one without, so the two are not the same number.
    ``sign_guaranteed`` is true when the set is eta-only and the policy sent identical
    weights in both arms on every seed; every ``return_gap`` is then ``>= 0`` (module docs).
    """

    n_seeds: int
    seeds: tuple[int, ...]
    uncertainty: tuple[float, float, float]
    settings: MarketSettings
    per_seed: tuple[SeedGap, ...]
    mean_point_return: float
    mean_robust_return: float
    mean_return_gap: float
    mean_point_sharpe: Statistic
    mean_robust_sharpe: Statistic
    mean_sharpe_gap: Statistic
    return_gap_interval: Union[GapInterval, Unavailable]
    sharpe_gap_interval: Union[GapInterval, Unavailable]
    sign_guaranteed: bool


def _exogenous_marked(trace: ArmTrace) -> tuple[float, float]:
    """(final return with the position at the exogenous mid, own-impact mark), on capital."""
    capital = trace.settings.capital
    exogenous = trace.exogenous_mids[-1]
    cleared = trace.cleared_mids[-1]
    nav = math.fsum([trace.final_cash, *(h * e for h, e in zip(trace.final_shares, exogenous))])
    own = math.fsum(
        h * (x - e) for h, x, e in zip(trace.final_shares, cleared, exogenous)
    )
    return nav / capital - 1.0, own / capital


def _seed_gap(point: ArmTrace, robust: ArmTrace) -> SeedGap:
    point_return, point_own = _exogenous_marked(point)
    robust_return, robust_own = _exogenous_marked(robust)
    identical = _bits([point.rewards, point.navs]) == _bits([robust.rewards, robust.navs])
    point_sharpe = track_sharpe(point.rewards)
    robust_sharpe = track_sharpe(robust.rewards)
    if isinstance(point_sharpe, Unavailable) or isinstance(robust_sharpe, Unavailable):
        sharpe_gap: Statistic = Unavailable(
            SHARPE_UNAVAILABLE,
            f"point {point_sharpe!r}, robust {robust_sharpe!r}",
        )
    else:
        sharpe_gap = point_sharpe - robust_sharpe
    return SeedGap(
        seed=point.seed,
        traded_bars=len(point.rewards),
        identical_arms=identical,
        identical_actions=_bits(point.weights) == _bits(robust.weights),
        point_return=point_return,
        robust_return=robust_return,
        return_gap=point_return - robust_return,
        point_own_impact_mark=point_own,
        robust_own_impact_mark=robust_own,
        point_sharpe=point_sharpe,
        robust_sharpe=robust_sharpe,
        sharpe_gap=sharpe_gap,
    )


def impact_misspecification_gap(
    make_policy: Callable[[], Policy],
    seeds: Sequence[int],
    uncertainty: Any,
    settings: Optional[MarketSettings] = None,
    confidence: float = 0.95,
) -> ImpactGapReport:
    """Run ``make_policy()`` at the point estimate and against ``uncertainty`` on each seed.

    ``uncertainty`` is required and takes the forms
    :class:`~sharpearena.market_env.EndogenousMarketEnv` accepts. Seeds must be distinct:
    a repeated seed would count one paired unit twice and understate the dispersion. Every
    seed's arms pass :func:`check_arm_pairing` before any number is computed.
    """
    norm = _normalize_uncertainty(uncertainty)
    if norm is None:
        raise ImpactDiagnosticError("uncertainty is required, got None")
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ImpactDiagnosticError("seeds is empty")
    if len(set(seed_list)) != len(seed_list):
        raise ImpactDiagnosticError(f"seeds must be distinct, got {seed_list!r}")
    s = (settings or MarketSettings()).validated()

    rows = []
    for seed in seed_list:
        point = run_impact_arm(make_policy, seed, None, s)
        robust = run_impact_arm(make_policy, seed, norm, s)
        check_arm_pairing(point, robust)
        rows.append(_seed_gap(point, robust))

    return ImpactGapReport(
        n_seeds=len(rows),
        seeds=tuple(seed_list),
        uncertainty=norm,
        settings=s,
        per_seed=tuple(rows),
        mean_point_return=_mean([r.point_return for r in rows]),
        mean_robust_return=_mean([r.robust_return for r in rows]),
        mean_return_gap=_mean([r.return_gap for r in rows]),
        mean_point_sharpe=_mean_statistic([r.point_sharpe for r in rows]),
        mean_robust_sharpe=_mean_statistic([r.robust_sharpe for r in rows]),
        mean_sharpe_gap=_mean_statistic([r.sharpe_gap for r in rows]),
        return_gap_interval=gap_interval([r.return_gap for r in rows], confidence),
        sharpe_gap_interval=gap_interval([r.sharpe_gap for r in rows], confidence),
        sign_guaranteed=norm[0] == 0.0 and all(r.identical_actions for r in rows),
    )


# ---------------------------------------------------------------------------
# The meta-order impact-shape probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaOrderImpactShape:
    """Measured permanent-impact shape of a buy meta-order (see the module docs).

    ``impact[k]`` is ``cleared_mid / exogenous_mid - 1`` at traded bar ``k + 1`` and
    ``executed_quantity[k]`` is the shares bought before that bar. ``relaxation_ratios[j]``
    is impact ``j`` bars after execution ends divided by impact when it ends.
    ``duration_impact[i]`` is the impact once ``total_weight`` has been bought over
    ``durations[i]`` bars.
    """

    impact_exponent: float
    seed: int
    executed_quantity: tuple[float, ...]
    impact: tuple[float, ...]
    execution_exponent: float
    relaxation_ratios: tuple[float, ...]
    max_relaxation_deviation: float
    durations: tuple[int, ...]
    duration_impact: tuple[float, ...]
    duration_exponent: float


def _impact_path(
    impact_exponent: float,
    seed: int,
    n_days: int,
    kyle_lambda: float,
    volume_scale: float,
    capital: float,
    distribution_mode: str,
    weight_at: Callable[[int], float],
) -> tuple[list[float], list[float]]:
    market = PyMarketClearing(
        n_symbols=1,
        n_days=n_days,
        seed=int(seed),
        n_agents=1,
        capital=capital,
        kyle_lambda=kyle_lambda,
        eta=0.0,
        volume_scale=volume_scale,
        distribution_mode=distribution_mode,
        impact_exponent=impact_exponent,
    )
    json.loads(market.reset_market())
    exogenous = []
    while True:
        step = json.loads(market.step_market("[[0.0]]"))
        exogenous.append(float(step["cleared_mids"][0]))
        if step["done"]:
            break
    json.loads(market.reset_market())
    executed, impact = [], []
    cumulative = 0.0
    bar = 0
    while True:
        step = json.loads(market.step_market(json.dumps([[float(weight_at(bar))]])))
        executed.append(cumulative)
        impact.append(float(step["cleared_mids"][0]) / exogenous[bar] - 1.0)
        cumulative += float(step["net_flow"][0])
        bar += 1
        if step["done"]:
            break
    return executed, impact


def _log_log_slope(xs: Sequence[float], ys: Sequence[float], what: str) -> float:
    if any(not (x > 0.0) for x in xs) or any(not (y > 0.0) for y in ys):
        raise ImpactDiagnosticError(
            f"{what} needs positive quantities and impacts, got impacts {list(ys)!r}"
        )
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = _mean(lx), _mean(ly)
    return math.fsum((a - mx) * (b - my) for a, b in zip(lx, ly)) / math.fsum(
        (a - mx) ** 2 for a in lx
    )


def meta_order_impact_shape(
    impact_exponent: float = 1.0,
    seed: int = 0,
    kyle_lambda: float = 0.1,
    volume_scale: float = 1.0,
    capital: float = 1.0,
    weight_step: float = 0.02,
    execution_bars: int = 40,
    hold_bars: int = 20,
    durations: Sequence[int] = (5, 10, 20, 40),
    total_weight: float = 0.8,
    distribution_mode: str = "calm",
) -> MetaOrderImpactShape:
    """Measure how the kernel's permanent impact grows with a buy meta-order.

    One symbol and one agent, with ``eta = 0`` (temporary impact never enters the
    cleared mid). Impact is read against a flat replay of the same market object, so the
    exogenous path cancels exactly. Three measurements:

    * the execution exponent: the log-log slope of impact on executed quantity while the
      target weight ramps by ``weight_step`` for ``execution_bars`` bars;
    * relaxation: impact over the next ``hold_bars`` bars at a held weight, as a ratio to
      impact when execution ends;
    * the duration exponent: the log-log slope of end-of-execution impact on execution
      length when ``total_weight`` is bought over each of ``durations`` bars.

    The square-root law would give about ``0.5``, about ``0`` and ratios below one. See
    the module docs for what this kernel gives and why.
    """
    if not (math.isfinite(kyle_lambda) and kyle_lambda > 0.0):
        raise ImpactDiagnosticError(f"kyle_lambda must be positive, got {kyle_lambda!r}")
    # The same rule `MarketSettings.validated` applies for the paired report. These reach
    # the engine directly here, so without this they arrived unchecked.
    for name, value in (
        ("capital", capital),
        ("volume_scale", volume_scale),
        ("impact_exponent", impact_exponent),
    ):
        _positive(name, value)
    if not (math.isfinite(weight_step) and weight_step > 0.0):
        raise ImpactDiagnosticError(f"weight_step must be positive, got {weight_step!r}")
    if not (math.isfinite(total_weight) and total_weight > 0.0):
        raise ImpactDiagnosticError(f"total_weight must be positive, got {total_weight!r}")
    if execution_bars < 2 or hold_bars < 1:
        raise ImpactDiagnosticError(
            f"needs execution_bars >= 2 and hold_bars >= 1, got {execution_bars} and {hold_bars}"
        )
    duration_list = [int(d) for d in durations]
    if len(set(duration_list)) < 2 or min(duration_list) < 1:
        raise ImpactDiagnosticError(
            f"durations needs two distinct positive lengths, got {duration_list!r}"
        )
    needed = max(execution_bars, max(duration_list)) + hold_bars + 1
    # The engine burns in up to 20 untraded bars before the first decision.
    n_days = needed + 20

    def path(weight_at: Callable[[int], float]) -> tuple[list[float], list[float]]:
        return _impact_path(
            impact_exponent, seed, n_days, kyle_lambda, volume_scale, capital,
            distribution_mode, weight_at,
        )

    executed, impact = path(lambda bar: weight_step * min(bar + 1, execution_bars))
    if len(impact) < needed:
        raise ImpactDiagnosticError(f"the path has {len(impact)} traded bars, needs {needed}")
    ramp_x = executed[1 : execution_bars + 1]
    ramp_i = impact[1 : execution_bars + 1]
    end = impact[execution_bars]
    if not end > 0.0:
        raise ImpactDiagnosticError(
            "relaxation is measured as a ratio to the impact when execution ends, and "
            f"that impact is {end!r}: the meta-order moved the cleared mid by less than "
            "its float resolution, so there is no shape to report. A larger weight_step "
            "or a smaller volume_scale gives a measurable one."
        )
    ratios = tuple(impact[execution_bars + j] / end for j in range(hold_bars + 1))

    duration_impact = []
    for d in duration_list:
        _, dur_i = path(lambda bar, d=d: total_weight * min(bar + 1, d) / d)
        duration_impact.append(dur_i[d])

    return MetaOrderImpactShape(
        impact_exponent=float(impact_exponent),
        seed=int(seed),
        executed_quantity=tuple(ramp_x),
        impact=tuple(ramp_i),
        execution_exponent=_log_log_slope(ramp_x, ramp_i, "the execution exponent"),
        relaxation_ratios=ratios,
        max_relaxation_deviation=max(abs(r - 1.0) for r in ratios),
        durations=tuple(duration_list),
        duration_impact=tuple(duration_impact),
        duration_exponent=_log_log_slope(
            [float(d) for d in duration_list], duration_impact, "the duration exponent"
        ),
    )


__all__ = [
    "CONSTANT_TRACK",
    "INSUFFICIENT_SEEDS",
    "NON_FINITE_SHARPE",
    "SHARPE_UNAVAILABLE",
    "TOO_FEW_BARS",
    "ArmTrace",
    "GapInterval",
    "ImpactDiagnosticError",
    "ImpactGapReport",
    "MarketSettings",
    "MetaOrderImpactShape",
    "NonPositiveMidError",
    "SeedGap",
    "Unavailable",
    "UnpairedArmsError",
    "check_arm_pairing",
    "gap_interval",
    "impact_misspecification_gap",
    "meta_order_impact_shape",
    "run_impact_arm",
    "t_critical",
    "track_sharpe",
]
