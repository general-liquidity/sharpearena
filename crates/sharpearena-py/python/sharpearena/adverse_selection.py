"""Adverse selection of meta-orders against market makers, and the markout that measures it.

Adverse selection is the market maker's core risk: the fills it gets are disproportionately
the ones it did not want, because the counterparty knew something. A single-agent market
making environment cannot express this. In :mod:`sharpearena.market_making` every market
order arrives from a memoryless Poisson stream whose direction is independent of the next
price move, so a maker's fills carry no information and the only losses are inventory
losses. The scenario here supplies the missing counterparty: an **informed trader** whose
flow precedes the price move, sliced into a meta-order rather than printed in one clip.

Following "When AI Trading Agents Compete: Adverse Selection of Meta-Orders by
Reinforcement Learning-Based Market Making".

**The informational edge is explicit, not hoped for.** Each meta-order carries a signed
efficient-price move ``alpha`` that is realized in equal increments over its children's
lifetime. In *informed* mode the meta-order's side is ``sign(alpha)``: it buys ahead of a
rise, sells ahead of a fall. In *uninformed* mode the identical schedule, the identical
child sizes and the identical price path are used, but the side is drawn from an
independent coin. That is a paired control: the two modes differ only in whether the
counterparty's direction is correlated with what happens next, which is the definition of
adverse selection. :func:`compare_informed_vs_uninformed` runs both and reports the gap. If
that gap is not there, the scenario is broken and nothing downstream of it means anything.

Several meta-orders run per episode, with independently drawn ``alpha`` signs. One parent
order gives an episode statistic that is mostly a coin flip about which way that single
move went; repeated parents are both what the paper studies and what makes the markout
estimate stable enough to compare across variants.

The model deliberately does not separate "the informed trader knew" from "the informed
trader's own flow moved the price". Both produce the same thing from the maker's seat: a
fill followed by a move against it. Splitting foreknowledge from permanent impact needs a
counterfactual price path the maker never observes, so the scenario folds them into one
drift term and says so rather than pretending to attribute.

**The measurement is markout**, the canonical adverse-selection metric: for each fill, how
the mid moved after it, signed from the maker's perspective. At horizon ``h``:

* ``spread_capture = signed_qty * (mid_t - fill_price) = qty * quote_depth``, always
  positive. This is the edge the maker booked at the moment of the fill.
* ``adverse_drift_h = signed_qty * (mid_{t+h} - mid_t)``, the price move after the fill.
* ``markout_h = spread_capture + adverse_drift_h``, exactly. The identity holds per fill,
  so the decomposition is a partition and not an estimate.

A maker that looks profitable on spread capture while bleeding on markout is the exact
failure this measures, and :class:`MakerMarkout` states it in a field
(``picked_off``) and in a sentence (``verdict``) rather than leaving it to be inferred from
two numbers of opposite sign.

Being picked off systematically and being picked off occasionally are different diseases
with different fixes, so both are reported: ``toxic_fill_rate`` is the share of fills whose
markout is negative, and ``worst_decile_share`` is the share of all markout losses
concentrated in the worst tenth of fills. A high rate with a low tail share means the
quotes are mispriced against the whole flow. A low rate with a high tail share means a
handful of meta-orders are doing the damage.

**The endogenous arm.** The scenario above fixes the efficient-price path: the meta-order's
information is realized as an exogenous drift, and nothing any taker does moves the mid.
That is enough to measure the informed-versus-uninformed *gap*, but it cannot speak to
maker profitability *levels*, because a maker whose fills never move the price against it
books the full quoted depth on every print. :class:`EndogenousImpact` supplies the missing
mechanism, using the permanent-impact law of the shared-book clearing engine
(``sharpearena::market``, Kyle 1985 as implemented for the M2 environment). Let ``S_t`` be
the efficient price (the alpha drift plus ordinary volatility, identical across the two legs
of the control) and ``Q_t`` the signed taker quantity actually filled against the maker
ladder in bar ``t`` (meta-order children plus noise flow). The mid the makers quote around is

    ``mid_t = S_t * M_t``,  ``M_{t+1} = M_t * (1 + lambda * Q_t / V)``,  ``M_0 = 1``,

the engine's running permanent-impact multiplier, updated after the bar clears so that the
reference price a maker quotes against is moved only by flow strictly before it. Temporary
impact is not a separate term here: the maker ladder itself is the temporary component (a
taker pays the quoted depth, which is what the engine's ``eta`` term models in the
maker-less M2 market), so the arm adds exactly one thing to the exogenous scenario, that a
filled unit moves the next reference price by ``mid_t * lambda / V`` in the taker's
direction. With ``lambda = 0`` the arm reproduces the exogenous scenario bit for bit.
:func:`compare_endogenous_arms` runs the paired control under both arms.

Determinism: every draw comes from seeded ``np.random.default_rng`` streams derived from
one master seed, and there is no clock and no I/O, so a report reproduces byte for byte
from ``(seed, params, policies)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .market_making import MMParams, Policy

_INFORMED = "informed"
_UNINFORMED = "uninformed"
_NOISE = "noise"

_COUNTERPARTIES = (_INFORMED, _UNINFORMED, _NOISE)


@dataclass(frozen=True)
class AdverseSelectionParams:
    """Scenario parameters: an Avellaneda-Stoikov quoting world plus one meta-order.

    ``mm`` supplies the maker-side physics (volatility, fill decay ``kappa``, noise arrival
    rate, horizon, inventory cap) so the policies shipped in
    :mod:`sharpearena.market_making` drop straight in. The remaining fields describe the
    informed counterparty: ``n_meta_orders`` parent orders of ``parent_size`` units each,
    every one sliced into ``n_children`` children, evenly spaced from ``meta_start_step``,
    each accompanied by an efficient-price move of magnitude ``alpha`` realized over its
    own window.

    ``informed`` is the switch the paired control turns: ``True`` sides each meta-order
    with its own ``alpha``, ``False`` sides it on an independent coin while leaving the
    price path and the slice schedule untouched.
    """

    mm: MMParams = field(default_factory=MMParams)
    n_makers: int = 2
    n_meta_orders: int = 4
    parent_size: int = 60
    n_children: int = 10
    meta_start_step: int = 20
    alpha: float = 2.0
    informed: bool = True
    markout_horizons: tuple[int, ...] = (1, 5, 20)

    def __post_init__(self) -> None:
        if self.n_makers < 1:
            raise ValueError("n_makers must be >= 1")
        if self.n_meta_orders < 1:
            raise ValueError("n_meta_orders must be >= 1")
        if self.n_children < 1:
            raise ValueError("n_children must be >= 1")
        if self.parent_size < 1:
            raise ValueError("parent_size must be >= 1")
        if self.meta_start_step < 0:
            raise ValueError("meta_start_step must be >= 0")
        if not self.markout_horizons:
            raise ValueError("at least one markout horizon is required")
        if any(h < 1 for h in self.markout_horizons):
            raise ValueError("markout horizons must be >= 1")
        if self.meta_start_step + self.n_meta_orders * self.n_children > self.mm.n_steps:
            raise ValueError("the meta-orders must finish inside the episode horizon")

    @property
    def spacing(self) -> int:
        """Bars between consecutive meta-order starts, spread over the remaining episode so
        the parents do not overlap and each one's move is attributable to it alone."""
        room = self.mm.n_steps - self.meta_start_step
        return max(self.n_children, room // self.n_meta_orders)


@dataclass(frozen=True)
class EndogenousImpact:
    """Permanent-impact coefficients for the endogenous arm, in the clearing engine's units.

    ``kyle_lambda`` is Kyle's coefficient per unit of dimensionless flow and
    ``volume_scale`` the ADV-like normalizer ``V`` that makes flow dimensionless, exactly as
    in ``MarketParams`` of the shared-book engine: a bar whose filled taker flow is ``Q``
    multiplies the next reference mid by ``1 + kyle_lambda * Q / V``. That factor must
    remain finite and strictly positive on every realized bar: a non-positive value would
    make the modeled mid non-positive and the log-markout statistic undefined, so the
    runtime rejects it with ``ValueError`` rather than silently crossing the domain. The default
    ``volume_scale`` is one episode's realized taker volume under the default scenario
    (mean 239 filled units over the 48 exogenous-arm episodes of the paired control), the
    scenario's analogue of a day's volume; at the default ``kyle_lambda`` a filled unit
    then moves the mid by ``100 * 0.1 / 240 = 0.042`` price units at the initial mid.
    """

    kyle_lambda: float = 0.1
    volume_scale: float = 240.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.kyle_lambda) and self.kyle_lambda >= 0.0):
            raise ValueError("kyle_lambda must be a finite non-negative number")
        if not (math.isfinite(self.volume_scale) and self.volume_scale > 0.0):
            raise ValueError("volume_scale must be a finite positive number")

    def impact_per_unit(self, mid: float) -> float:
        """Permanent mid move, in price units, per filled unit of taker flow at ``mid``."""
        return mid * self.kyle_lambda / self.volume_scale

    def multiplier_for_flow(self, flow: float) -> float:
        """Return the valid next-bar permanent-impact multiplier for signed filled flow.

        This is deliberately a domain check, not a clamp: clamping would change the clearing
        law at the very parameter values where the calibration has become incoherent.
        """
        factor = 1.0 + self.kyle_lambda * flow / self.volume_scale
        if not (math.isfinite(factor) and factor > 0.0):
            raise ValueError(
                "endogenous impact requires a finite positive multiplier "
                "1 + kyle_lambda * Q / volume_scale; "
                f"got {factor!r} for kyle_lambda={self.kyle_lambda!r}, "
                f"Q={flow!r}, volume_scale={self.volume_scale!r}"
            )
        return factor


@dataclass(frozen=True)
class MetaOrder:
    """One parent order and the price move that accompanies it.

    ``children`` is the realized slice schedule, so the report can be read back without
    re-deriving it. ``side`` is ``+1`` for a buying parent and ``-1`` for a selling one; in
    informed mode it equals ``sign(alpha)`` and in uninformed mode it does not, which is
    the entire difference between the two legs of the control.
    """

    start_step: int
    children: tuple[int, ...]
    alpha: float
    side: int

    @property
    def end_step(self) -> int:
        return self.start_step + len(self.children)


@dataclass(frozen=True)
class Fill:
    """One maker fill, carrying everything markout needs and nothing it does not.

    ``signed_qty`` is positive when the maker bought (its bid was hit) and negative when it
    sold (its ask was lifted), so every downstream quantity is a plain signed product and
    no side flag has to be re-read. ``counterparty`` records who took the quote, which is
    what lets the report separate meta-order damage from ordinary noise flow.
    """

    step: int
    maker: str
    signed_qty: int
    price: float
    mid_at_fill: float
    depth: float
    counterparty: str


@dataclass(frozen=True)
class MakerMarkout:
    """Fill-conditional markout for one maker, decomposed.

    ``markout``, ``adverse_drift`` and ``markout_per_unit`` are keyed by horizon in steps.
    ``spread_capture`` is horizon independent by construction: it is booked at the fill.
    ``by_counterparty`` splits ``adverse_drift`` across informed / uninformed / noise flow,
    so a maker can see whether its losses come from the meta-order or from carrying
    inventory through ordinary volatility.

    ``meta_markout_per_unit`` is broken out separately because the aggregate hides the
    diagnosis. A maker can be comfortably positive overall while every meta-order fill it
    takes loses money, subsidised by the noise flow around it. That is not a healthy book,
    it is a book whose losses scale with how much informed flow shows up, and the aggregate
    number will keep saying so right up until the informed share rises.
    """

    maker: str
    n_fills: int
    filled_qty: int
    spread_capture: float
    markout: dict[int, float]
    adverse_drift: dict[int, float]
    markout_per_unit: dict[int, float]
    toxic_fill_rate: dict[int, float]
    worst_decile_share: dict[int, float]
    by_counterparty: dict[str, dict[int, float]]
    meta_filled_qty: int
    meta_markout_per_unit: dict[int, float]
    picked_off: dict[int, bool]
    verdict: str

    def to_dict(self) -> dict:
        """Plain-JSON view: horizon keys become strings so the block survives a round trip."""

        def _h(d: dict[int, float]) -> dict:
            return {str(k): v for k, v in d.items()}

        return {
            "maker": self.maker,
            "n_fills": self.n_fills,
            "filled_qty": self.filled_qty,
            "spread_capture": self.spread_capture,
            "markout": _h(self.markout),
            "adverse_drift": _h(self.adverse_drift),
            "markout_per_unit": _h(self.markout_per_unit),
            "toxic_fill_rate": _h(self.toxic_fill_rate),
            "worst_decile_share": _h(self.worst_decile_share),
            "by_counterparty": {k: _h(v) for k, v in self.by_counterparty.items()},
            "meta_filled_qty": self.meta_filled_qty,
            "meta_markout_per_unit": _h(self.meta_markout_per_unit),
            "picked_off": {str(k): v for k, v in self.picked_off.items()},
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class AdverseSelectionReport:
    """The outcome of one seeded episode: per-maker markout plus the scenario's own facts."""

    seed: int
    informed: bool
    meta_orders: tuple[MetaOrder, ...]
    meta_offered_qty: int
    meta_filled_qty: int
    makers: tuple[MakerMarkout, ...]
    fills: tuple[Fill, ...]
    mid_path: tuple[float, ...]
    # The endogenous decomposition of ``mid_path``: ``mid_t = efficient_path[t] *
    # impact_multiplier_path[t]``. In the exogenous arm the multiplier is identically 1 and
    # the efficient path is the mid path. ``net_flow[t]`` is the signed taker quantity
    # filled in bar ``t``, the flow that moves ``M_{t+1}``.
    impact: Optional[EndogenousImpact] = None
    efficient_path: tuple[float, ...] = ()
    impact_multiplier_path: tuple[float, ...] = ()
    net_flow: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        out = {
            "seed": self.seed,
            "informed": self.informed,
            "meta_orders": [
                {
                    "start_step": m.start_step,
                    "children": list(m.children),
                    "alpha": m.alpha,
                    "side": m.side,
                }
                for m in self.meta_orders
            ],
            "meta_offered_qty": self.meta_offered_qty,
            "meta_filled_qty": self.meta_filled_qty,
            "makers": [m.to_dict() for m in self.makers],
        }
        if self.impact is not None:
            out["impact"] = {
                "kyle_lambda": self.impact.kyle_lambda,
                "volume_scale": self.impact.volume_scale,
                "final_multiplier": self.impact_multiplier_path[-1],
                "informed_displacement": informed_displacement(self),
            }
        return out


# -- internals --------------------------------------------------------------


def _maker_ids(n: int) -> list[str]:
    """The canonical sorted roster. Sorted order is also the tie-break in quote priority,
    so the whole simulation is order-stable."""
    return [f"maker_{i}" for i in range(int(n))]


def _slice_schedule(parent_size: int, n_children: int) -> list[int]:
    """Split a parent order into ``n_children`` children as evenly as integers allow.

    Deterministic and total: the children sum to ``parent_size`` exactly, with the
    remainder spread over the earliest children rather than dumped on the last one.
    """
    base, rem = divmod(int(parent_size), int(n_children))
    return [base + (1 if i < rem else 0) for i in range(int(n_children))]


def _default_policies(p: MMParams, n_makers: int) -> list[Policy]:
    """Makers used when the caller supplies none: the Avellaneda-Stoikov optimum first,
    then progressively tighter fixed-spread quoters. Tighter quotes win priority and so
    absorb more of the meta-order, which is the behaviour the report should make visible.
    """
    from .market_making import closed_form_reference_policy, fixed_spread_policy

    policies: list[Policy] = [closed_form_reference_policy(p)]
    for i in range(1, n_makers):
        policies.append(fixed_spread_policy(max(p.tick_size, 0.9 - 0.25 * (i - 1))))
    return policies[:n_makers]


class _MakerBook:
    """Per-maker inventory and cash, kept out of the simulation loop so the loop reads as
    the market it is rather than as bookkeeping."""

    def __init__(self, makers: Sequence[str], p: MMParams) -> None:
        self.inventory = {m: 0 for m in makers}
        self.cash = {m: p.initial_cash for m in makers}
        self._cap = p.inventory_cap

    def can_buy(self, maker: str) -> bool:
        return self.inventory[maker] < self._cap

    def can_sell(self, maker: str) -> bool:
        return self.inventory[maker] > -self._cap

    def apply(self, maker: str, signed_qty: int, price: float) -> None:
        self.inventory[maker] += signed_qty
        self.cash[maker] -= signed_qty * price


def _route(
    *,
    taker_is_buy: bool,
    qty: int,
    step: int,
    mid: float,
    depths: dict[str, tuple[float, float]],
    makers: Sequence[str],
    book: _MakerBook,
    counterparty: str,
    kappa: float,
    rng: np.random.Generator,
    out: list[Fill],
) -> int:
    """Route one taker order across the maker ladder, price-first, and record the fills.

    Priority is by quote depth ascending with the sorted roster breaking ties, so the
    tightest quote is hit first: this is what makes tight quoting both profitable on spread
    and dangerous on markout, which is the tension the whole scenario exists to expose.
    Each unit then fills at the Avellaneda-Stoikov probability ``exp(-kappa * depth)`` and
    walks to the next maker if it does not. Returns the quantity actually filled.
    """
    side = 1 if taker_is_buy else 0  # index into (bid_depth, ask_depth)
    ladder = sorted(makers, key=lambda m: (depths[m][side], m))
    filled = 0
    for _ in range(int(qty)):
        for maker in ladder:
            depth = depths[maker][side]
            if taker_is_buy and not book.can_sell(maker):
                continue
            if not taker_is_buy and not book.can_buy(maker):
                continue
            if float(rng.random()) >= math.exp(-kappa * depth):
                continue
            price = mid + depth if taker_is_buy else mid - depth
            signed_qty = -1 if taker_is_buy else 1
            book.apply(maker, signed_qty, price)
            out.append(
                Fill(
                    step=step,
                    maker=maker,
                    signed_qty=signed_qty,
                    price=price,
                    mid_at_fill=mid,
                    depth=depth,
                    counterparty=counterparty,
                )
            )
            filled += 1
            break
    return filled


def _obs(maker: str, book: _MakerBook, mid: float, t: int, p: MMParams) -> dict:
    """A :class:`~sharpearena.market_making.MarketMakingEnv`-shaped observation, so any
    policy written against that env quotes here unmodified."""
    return {
        "inventory": np.array([float(book.inventory[maker])], dtype=np.float64),
        "mid": np.array([mid], dtype=np.float64),
        "time_remaining": np.array([(p.n_steps - t) * p.dt], dtype=np.float64),
        "cash": np.array([book.cash[maker]], dtype=np.float64),
    }


# -- markout ----------------------------------------------------------------


def fill_markout(
    fills: Sequence[Fill],
    mid_path: Sequence[float],
    horizon: int,
) -> list[float]:
    """Per-fill markout at ``horizon`` steps, signed from the maker's perspective.

    ``markout = signed_qty * (mid_{t+h} - fill_price)``. Fills inside ``horizon`` of the
    episode end mark out against the last observed mid rather than being dropped: dropping
    them would silently exclude exactly the end-of-episode fills a maker takes when it is
    unwinding, which is where its markout is often worst.
    """
    last = len(mid_path) - 1
    out: list[float] = []
    for f in fills:
        future = float(mid_path[min(f.step + int(horizon), last)])
        out.append(f.signed_qty * (future - f.price))
    return out


def _worst_decile_share(markouts: Sequence[float]) -> float:
    """Share of total markout losses concentrated in the worst tenth of fills.

    Near ``1.0`` the damage is a few catastrophic fills (occasional pick-off); near the
    decile fraction itself the losses are spread evenly across the flow (systematic
    mispricing). ``0.0`` when nothing lost money.
    """
    losses = [m for m in markouts if m < 0.0]
    if not losses:
        return 0.0
    total = -sum(losses)
    if total <= 0.0:
        return 0.0
    k = max(1, math.ceil(0.1 * len(markouts)))
    worst = sorted(markouts)[:k]
    return -sum(m for m in worst if m < 0.0) / total


def _verdict(
    per_unit: dict[int, float],
    toxic: dict[int, float],
    tail: dict[int, float],
    meta_per_unit: dict[int, float],
    meta_share: float,
    spread_capture: float,
    horizons: Sequence[int],
) -> str:
    """One sentence naming the failure, at the longest horizon reported.

    The interesting case is a maker whose spread capture is positive and whose markout is
    not: it books an edge on every print and hands it back, which reads as a healthy P&L
    right up until it does not. The near miss is the maker whose aggregate is positive only
    because noise flow is paying for the meta-order fills, so that case gets said out loud
    rather than being left to a reader who compares two dictionaries.
    """
    h = max(horizons)
    if per_unit.get(h, 0.0) >= 0.0:
        if meta_per_unit.get(h, 0.0) < 0.0:
            return (
                f"subsidised: aggregate markout is positive at h={h} but meta-order fills "
                f"lose {abs(meta_per_unit[h]):.3f} per unit on {meta_share:.0%} of filled "
                "quantity, so noise flow is paying for the informed flow"
            )
        return f"markout positive at h={h}: the flow is not selecting against this maker"
    if spread_capture <= 0.0:
        return f"markout negative at h={h} with no spread capture to lose"
    if toxic.get(h, 0.0) >= 0.6:
        return (
            f"systematically picked off: {toxic[h]:.0%} of fills mark out negative at "
            f"h={h}, so the quotes are mispriced against the whole flow"
        )
    if tail.get(h, 0.0) >= 0.5:
        return (
            f"occasionally picked off: {tail[h]:.0%} of the markout loss at h={h} sits in "
            "the worst tenth of fills, so a few meta-orders are doing the damage"
        )
    return (
        f"profitable on spread capture, negative on markout at h={h}: the booked edge is "
        "handed back after the fill"
    )


def _summarize(
    maker: str,
    fills: Sequence[Fill],
    mid_path: Sequence[float],
    horizons: Sequence[int],
) -> MakerMarkout:
    mine = [f for f in fills if f.maker == maker]
    meta = [f for f in mine if f.counterparty != _NOISE]
    filled_qty = sum(abs(f.signed_qty) for f in mine)
    meta_qty = sum(abs(f.signed_qty) for f in meta)
    spread_capture = sum(f.signed_qty * (f.mid_at_fill - f.price) for f in mine)

    markout: dict[int, float] = {}
    adverse: dict[int, float] = {}
    per_unit: dict[int, float] = {}
    toxic: dict[int, float] = {}
    tail: dict[int, float] = {}
    picked_off: dict[int, bool] = {}
    meta_per_unit: dict[int, float] = {}
    by_cp: dict[str, dict[int, float]] = {c: {} for c in _COUNTERPARTIES}

    last = len(mid_path) - 1
    for h in horizons:
        per_fill = fill_markout(mine, mid_path, h)
        total = float(sum(per_fill))
        markout[h] = total
        adverse[h] = total - spread_capture
        per_unit[h] = total / filled_qty if filled_qty else 0.0
        toxic[h] = (
            sum(1 for m in per_fill if m < 0.0) / len(per_fill) if per_fill else 0.0
        )
        tail[h] = _worst_decile_share(per_fill)
        picked_off[h] = spread_capture > 0.0 and total < 0.0
        meta_per_unit[h] = (
            float(sum(fill_markout(meta, mid_path, h))) / meta_qty if meta_qty else 0.0
        )
        for cp in _COUNTERPARTIES:
            by_cp[cp][h] = float(
                sum(
                    f.signed_qty
                    * (float(mid_path[min(f.step + h, last)]) - f.mid_at_fill)
                    for f in mine
                    if f.counterparty == cp
                )
            )

    return MakerMarkout(
        maker=maker,
        n_fills=len(mine),
        filled_qty=filled_qty,
        spread_capture=float(spread_capture),
        markout=markout,
        adverse_drift=adverse,
        markout_per_unit=per_unit,
        toxic_fill_rate=toxic,
        worst_decile_share=tail,
        by_counterparty=by_cp,
        meta_filled_qty=meta_qty,
        meta_markout_per_unit=meta_per_unit,
        picked_off=picked_off,
        verdict=_verdict(
            per_unit,
            toxic,
            tail,
            meta_per_unit,
            meta_qty / filled_qty if filled_qty else 0.0,
            float(spread_capture),
            horizons,
        ),
    )


# -- the scenario -----------------------------------------------------------


def run_adverse_selection(
    *,
    params: Optional[AdverseSelectionParams] = None,
    policies: Optional[Sequence[Policy]] = None,
    seed: int = 0,
    impact: Optional[EndogenousImpact] = None,
) -> AdverseSelectionReport:
    """Run one seeded episode of informed meta-order flow against the maker roster.

    Each step: every maker quotes, ordinary Poisson noise flow arrives, the meta-order
    sends its child if the window is open, and only then does the mid move. The ordering is
    the point. Fills are struck at the pre-move mid, so an informed child that lifts an ask
    is followed by the rise it anticipated, and the maker's markout carries the loss.

    ``policies`` are ``(obs) -> [bid_depth, ask_depth]`` callables in roster order, shaped
    for :class:`~sharpearena.market_making.MarketMakingEnv`. When omitted, the roster is
    the Avellaneda-Stoikov optimum plus tighter fixed-spread quoters.

    ``impact`` switches on the endogenous arm: the bar's filled taker flow then also moves
    the next mid through the clearing engine's permanent-impact multiplier (see the module
    docstring). ``None`` is the exogenous scenario, unchanged.
    """
    p = params or AdverseSelectionParams()
    mm = p.mm
    makers = _maker_ids(p.n_makers)
    pols = list(policies) if policies is not None else _default_policies(mm, p.n_makers)
    if len(pols) != p.n_makers:
        raise ValueError(f"expected {p.n_makers} policies, got {len(pols)}")

    # Independent streams so the informed / uninformed control differs only where intended.
    master = int(seed)
    rng_price = np.random.default_rng([master, 1])
    rng_noise = np.random.default_rng([master, 2])
    rng_fill = np.random.default_rng([master, 3])
    rng_side = np.random.default_rng([master, 4])
    rng_meta = np.random.default_rng([master, 5])

    # The alpha signs come from rng_meta in BOTH legs, so the price path is identical and
    # only the counterparty's direction changes. That is what makes this a paired control.
    children = _slice_schedule(p.parent_size, p.n_children)
    meta_orders: list[MetaOrder] = []
    for k in range(p.n_meta_orders):
        alpha = abs(p.alpha) * (1 if float(rng_meta.random()) < 0.5 else -1)
        coin = 1 if float(rng_side.random()) < 0.5 else -1
        side = (1 if alpha >= 0.0 else -1) if p.informed else coin
        meta_orders.append(
            MetaOrder(
                start_step=p.meta_start_step + k * p.spacing,
                children=tuple(children),
                alpha=alpha,
                side=side,
            )
        )
    active: dict[int, MetaOrder] = {}
    for order in meta_orders:
        for t in range(order.start_step, order.end_step):
            active[t] = order

    book = _MakerBook(makers, mm)
    fills: list[Fill] = []
    efficient = mm.s0
    mult = 1.0
    mid = efficient
    mid_path = [mid]
    efficient_path = [efficient]
    mult_path = [mult]
    net_flow: list[int] = []
    meta_filled = 0
    meta_cp = _INFORMED if p.informed else _UNINFORMED
    lam = mm.arrival_rate * mm.dt

    for t in range(mm.n_steps):
        n_fills_before = len(fills)
        depths: dict[str, tuple[float, float]] = {}
        for maker, policy in zip(makers, pols):
            action = np.asarray(policy(_obs(maker, book, mid, t, mm)), dtype=np.float64)
            depths[maker] = (
                float(np.clip(action.reshape(-1)[0], 0.0, mm.max_depth)),
                float(np.clip(action.reshape(-1)[1], 0.0, mm.max_depth)),
            )

        # Ordinary uninformed flow: direction independent of the next move, as in A-S.
        n_buy = int(rng_noise.poisson(lam))
        n_sell = int(rng_noise.poisson(lam))
        _route(
            taker_is_buy=True, qty=n_buy, step=t, mid=mid, depths=depths, makers=makers,
            book=book, counterparty=_NOISE, kappa=mm.kappa, rng=rng_fill, out=fills,
        )
        _route(
            taker_is_buy=False, qty=n_sell, step=t, mid=mid, depths=depths, makers=makers,
            book=book, counterparty=_NOISE, kappa=mm.kappa, rng=rng_fill, out=fills,
        )

        # The live meta-order's child for this bar, struck before the price moves.
        order = active.get(t)
        if order is not None:
            meta_filled += _route(
                taker_is_buy=order.side > 0,
                qty=order.children[t - order.start_step],
                step=t,
                mid=mid,
                depths=depths,
                makers=makers,
                book=book,
                counterparty=meta_cp,
                kappa=mm.kappa,
                rng=rng_fill,
                out=fills,
            )

        # The bar's filled taker flow, signed from the taker's side (a maker fill with
        # signed_qty = -1 is a taker buy). This is what moves the endogenous mid.
        flow = -sum(f.signed_qty for f in fills[n_fills_before:])
        net_flow.append(flow)

        # The efficient price moves: the meta-order's information (and, in the exogenous
        # arm, its impact, which that arm does not try to separate from it) plus ordinary
        # volatility.
        if order is not None:
            efficient += order.alpha / len(order.children)
        efficient += mm.sigma * math.sqrt(mm.dt) * float(rng_price.standard_normal())
        efficient_path.append(efficient)
        if impact is None:
            mid = efficient
        else:
            # The engine's permanent-impact update, applied after the bar clears so that
            # the mid a maker quotes against is moved only by flow strictly before it.
            mult *= impact.multiplier_for_flow(flow)
            mid = efficient * mult
        mult_path.append(mult)
        mid_path.append(mid)

    return AdverseSelectionReport(
        seed=master,
        informed=p.informed,
        meta_orders=tuple(meta_orders),
        meta_offered_qty=p.n_meta_orders * p.parent_size,
        meta_filled_qty=meta_filled,
        makers=tuple(_summarize(m, fills, mid_path, p.markout_horizons) for m in makers),
        fills=tuple(fills),
        mid_path=tuple(mid_path),
        impact=impact,
        efficient_path=tuple(efficient_path),
        impact_multiplier_path=tuple(mult_path),
        net_flow=tuple(net_flow),
    )


def informed_displacement(report: AdverseSelectionReport) -> float:
    """Mean log-multiplier move over the meta-order windows, signed by the parent's side.

    ``side * (ln M_end - ln M_start)`` averaged over the episode's parents: positive when
    the parents' own flow pushed the reference mid the way they were trading, which is the
    endogenous arm's defining property. Identically zero in the exogenous arm.
    """
    path = report.impact_multiplier_path
    moves = [
        m.side * (math.log(path[m.end_step]) - math.log(path[m.start_step]))
        for m in report.meta_orders
    ]
    return sum(moves) / len(moves)


def compare_informed_vs_uninformed(
    *,
    params: Optional[AdverseSelectionParams] = None,
    policies: Optional[Sequence[Policy]] = None,
    n_episodes: int = 24,
    seed_base: int = 0,
) -> dict:
    """The non-vacuity check: does informed flow actually hurt more than the same flow would?

    Runs the scenario twice per seed, once with the meta-order sided on ``alpha`` and once
    with it sided on a coin, holding the price path, the slice schedule and the parent size
    fixed. If the informed leg does not mark out worse, the informed trader has no
    informational edge and every adverse-selection number produced by this module is noise
    wearing a metric's name. Reported per horizon as mean markout per filled unit across
    all makers, so the two legs are comparable even when they fill different quantities.
    """
    p = params or AdverseSelectionParams()
    horizons = p.markout_horizons

    def _leg(informed: bool) -> dict[int, float]:
        totals = {h: 0.0 for h in horizons}
        qty = 0
        cfg = replace(p, informed=informed)
        for i in range(int(n_episodes)):
            report = run_adverse_selection(
                params=cfg, policies=policies, seed=seed_base + i
            )
            qty += sum(m.filled_qty for m in report.makers)
            for h in horizons:
                totals[h] += sum(m.markout[h] for m in report.makers)
        return {h: (totals[h] / qty if qty else 0.0) for h in horizons}

    informed = _leg(True)
    uninformed = _leg(False)
    gap = {h: uninformed[h] - informed[h] for h in horizons}
    return {
        "n_episodes": int(n_episodes),
        "horizons": tuple(horizons),
        "informed_markout_per_unit": informed,
        "uninformed_markout_per_unit": uninformed,
        "gap_per_unit": gap,
        "informed_is_worse": {h: informed[h] < uninformed[h] for h in horizons},
    }


def _episode_row(report: AdverseSelectionReport, horizons: Sequence[int]) -> dict:
    """One episode's pooled maker statistics: markout per filled unit (episode markout over
    episode filled quantity, all makers) and the toxic-fill rate (share of fills, all
    makers, whose markout is negative), per horizon."""
    qty = sum(m.filled_qty for m in report.makers)
    per_unit: dict[int, float] = {}
    toxic: dict[int, float] = {}
    for h in horizons:
        per_unit[h] = sum(m.markout[h] for m in report.makers) / qty if qty else 0.0
        marks = fill_markout(report.fills, report.mid_path, h)
        toxic[h] = sum(1 for x in marks if x < 0.0) / len(marks) if marks else 0.0
    return {"markout_per_unit": per_unit, "toxic_fill_rate": toxic}


def compare_endogenous_arms(
    *,
    params: Optional[AdverseSelectionParams] = None,
    impact: Optional[EndogenousImpact] = None,
    policies: Optional[Sequence[Policy]] = None,
    n_episodes: int = 24,
    seed_base: int = 0,
) -> dict:
    """The paired informed / uninformed control under both arms, per episode.

    For each seed the scenario runs four times: informed and uninformed, each with the
    price path exogenous (``impact=None``) and endogenous (``impact``). The efficient
    price, the slice schedule, the parent sizes and the alpha signs are identical across
    all four; the endogenous arm differs only in that filled taker flow moves the next mid.
    Per-episode vectors are returned rather than pooled means so a caller can put an
    interval on every cell; ``means`` pools each vector for convenience.

    The economically meaningful question the endogenous arm answers is whether makers
    still earn a positive markout against informed flow once their own fills move the
    price, which is ``means["endogenous"]["informed"]["markout_per_unit"][h] > 0``.
    """
    p = params or AdverseSelectionParams()
    imp = impact or EndogenousImpact()
    horizons = tuple(p.markout_horizons)
    arms = {"exogenous": None, "endogenous": imp}
    legs = {"informed": True, "uninformed": False}

    def _vec() -> dict[int, list[float]]:
        return {h: [] for h in horizons}

    out: dict = {
        arm: {
            **{leg: {"markout_per_unit": _vec(), "toxic_fill_rate": _vec()} for leg in legs},
            "gap_per_unit": _vec(),
            "informed_displacement": [],
            "uninformed_displacement": [],
        }
        for arm in arms
    }
    for i in range(int(n_episodes)):
        seed = seed_base + i
        for arm, arm_impact in arms.items():
            rows: dict[str, dict] = {}
            for leg, informed in legs.items():
                report = run_adverse_selection(
                    params=replace(p, informed=informed),
                    policies=policies,
                    seed=seed,
                    impact=arm_impact,
                )
                rows[leg] = _episode_row(report, horizons)
                out[arm][f"{leg}_displacement"].append(informed_displacement(report))
                for key in ("markout_per_unit", "toxic_fill_rate"):
                    for h in horizons:
                        out[arm][leg][key][h].append(rows[leg][key][h])
            for h in horizons:
                out[arm]["gap_per_unit"][h].append(
                    rows["uninformed"]["markout_per_unit"][h]
                    - rows["informed"]["markout_per_unit"][h]
                )

    def _mean(v: Sequence[float]) -> float:
        return sum(v) / len(v) if v else 0.0

    means = {
        arm: {
            **{
                leg: {
                    key: {h: _mean(out[arm][leg][key][h]) for h in horizons}
                    for key in ("markout_per_unit", "toxic_fill_rate")
                }
                for leg in legs
            },
            "gap_per_unit": {h: _mean(out[arm]["gap_per_unit"][h]) for h in horizons},
            "informed_displacement": _mean(out[arm]["informed_displacement"]),
            "uninformed_displacement": _mean(out[arm]["uninformed_displacement"]),
        }
        for arm in arms
    }
    return {
        "n_episodes": int(n_episodes),
        "seed_base": int(seed_base),
        "horizons": horizons,
        "impact": {
            "kyle_lambda": imp.kyle_lambda,
            "volume_scale": imp.volume_scale,
            "impact_per_unit_at_s0": imp.impact_per_unit(p.mm.s0),
        },
        "per_episode": out,
        "means": means,
        "informed_is_worse": {
            arm: {h: means[arm]["gap_per_unit"][h] > 0.0 for h in horizons} for arm in arms
        },
        "makers_profit_against_informed_flow": {
            arm: {
                h: means[arm]["informed"]["markout_per_unit"][h] > 0.0 for h in horizons
            }
            for arm in arms
        },
    }


__all__ = [
    "AdverseSelectionParams",
    "AdverseSelectionReport",
    "EndogenousImpact",
    "Fill",
    "MakerMarkout",
    "MetaOrder",
    "compare_endogenous_arms",
    "compare_informed_vs_uninformed",
    "fill_markout",
    "informed_displacement",
    "run_adverse_selection",
]
