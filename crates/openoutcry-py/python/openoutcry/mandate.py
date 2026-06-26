"""Per-scenario trading mandates — MiniGrid's ``Mission`` pattern for trading.

In MiniGrid's *Fetch* task each episode ships a per-episode objective ("pick up the
red key") and the agent is graded on satisfying *that* objective, not a fixed one —
fetching the wrong object is penalized even when it is otherwise competent behavior.
The trading analogue: each scenario draws a :class:`Mandate` (a sampled trading
objective — a style constraint, an optional drawdown cap, an optional benchmark) that
the episode is graded against. Two rollouts on the same market but different mandates
are held to different standards.

The deterministic, leak-free logic — :func:`sample_mandate` (derives the whole mandate
from the scenario ``seed``, never from future bars) and :func:`mandate_breach` (a pure
penalty in ``[0, 1]``, ``0`` = clean, ``1`` = fully breached) — lives in the Rust core
(``openoutcry::mandate``) so it is **byte-identical across every surface** (Rust / WASM /
Python). This module is a thin wrapper: it reconstructs the :class:`Mandate` dataclass
from the binding's JSON and adapts the per-bar weight *events* into the weight-vector
shape the kernel scores. The mandate round-trips through plain-JSON
(:meth:`Mandate.to_dict` / :func:`mandate_from_dict`) so it survives a trace/replay; the
reward layer turns the breach into ``1 - breach`` — bounded, hence GRPO-safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Union

from .openoutcry_py import mandate_breach as _rs_mandate_breach
from .openoutcry_py import sample_mandate_json as _rs_sample_mandate_json

# The constraint families a scenario can draw. ``unconstrained`` is the permissive
# control (no structural breach); the others each carry a distinct structural rule.
# Kept in sync with the Rust ``MandateStyle`` wire labels (canonical draw order).
STYLES = ("long_only", "market_neutral", "momentum", "unconstrained", "pairs_convergence")


@dataclass(frozen=True)
class Mandate:
    """A per-scenario objective the episode is graded on satisfying.

    ``style`` is the structural constraint; ``max_drawdown`` an optional realized-DD cap
    (a fraction, e.g. ``0.10`` = 10%); ``max_inventory`` an optional per-bar gross-exposure
    cap on ``Σ|w_i|`` (e.g. ``1.0`` = at most 100% gross — exceeding it draws a *squared*
    breach, the Avellaneda-Stoikov inventory penalty); ``benchmark`` an optional symbol to
    beat (carried in the prompt text — informational, not breach-scored, since the breach
    checker has no per-symbol returns); ``text`` the human-readable rendering shown to the
    model.
    """

    style: str
    max_drawdown: Optional[float] = None
    max_inventory: Optional[float] = None
    benchmark: Optional[str] = None
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form for the dataset ``info`` dict / trace (scalars + null only)."""
        return {
            "style": self.style,
            "max_drawdown": self.max_drawdown,
            "max_inventory": self.max_inventory,
            "benchmark": self.benchmark,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Mandate":
        return cls(
            style=str(d["style"]),
            max_drawdown=(None if d.get("max_drawdown") is None else float(d["max_drawdown"])),
            max_inventory=(None if d.get("max_inventory") is None else float(d["max_inventory"])),
            benchmark=(None if d.get("benchmark") is None else str(d["benchmark"])),
            text=str(d.get("text", "")),
        )


def sample_mandate(
    seed: int,
    n_symbols: int = 4,
    *,
    allow_short: bool = True,
) -> Mandate:
    """Deterministically draw a :class:`Mandate` from a scenario ``seed`` (Rust kernel).

    Leak-free: the draw depends only on ``seed`` (and the static ``n_symbols`` /
    ``allow_short`` env shape), so it is reproducible at ``reset`` and never peeks at
    future bars. When ``allow_short`` is ``False`` the short-requiring styles are dropped
    so the mandate stays satisfiable on a long-only market.
    """
    payload = _rs_sample_mandate_json(int(seed), int(n_symbols), bool(allow_short))
    return Mandate.from_dict(json.loads(payload))


def mandate_text(m: Union[Mandate, dict]) -> str:
    """The human-readable objective string (for the scenario prompt)."""
    return _as_mandate(m).text


def mandate_from_dict(d: dict[str, Any]) -> Mandate:
    """Reconstruct a :class:`Mandate` from its plain-JSON form (trace/replay)."""
    return Mandate.from_dict(d)


def validate_mandate(obj: Union[Mandate, dict, None]) -> bool:
    """True iff ``obj`` is a structurally valid mandate (round-trip / replay guard)."""
    if obj is None:
        return False
    try:
        m = _as_mandate(obj)
    except Exception:  # noqa: BLE001 - malformed payload is simply invalid
        return False
    if m.style not in STYLES:
        return False
    if m.max_drawdown is not None and not (0.0 < float(m.max_drawdown) <= 1.0):
        return False
    # Gross-exposure cap is a positive ceiling on Σ|w_i| (may exceed 1, e.g. 2.0 = 200%).
    if m.max_inventory is not None and not (float(m.max_inventory) > 0.0):
        return False
    return True


def _as_mandate(m: Union[Mandate, dict]) -> Mandate:
    return m if isinstance(m, Mandate) else Mandate.from_dict(m)


def _weights_per_step(events: Any) -> list[list[float]]:
    """The per-step target-weight vectors the rollout recorded as events, if any.

    The multi-turn env appends a ``{"event": "target_weights", "weights": [...]}`` record
    each bar; the breach checker reads structural constraints off those. Events without a
    ``weights`` payload (real market events) are ignored.
    """
    out: list[list[float]] = []
    for e in events or []:
        if isinstance(e, dict) and "weights" in e:
            w = e.get("weights")
            if isinstance(w, (list, tuple)):
                out.append([float(x) for x in w])
    return out


def mandate_breach(
    m: Union[Mandate, dict],
    returns: list[float],
    events: list[dict],
) -> float:
    """A bounded breach penalty in ``[0, 1]`` (0 = clean, 1 = fully breached) — Rust kernel.

    Three independent breach sources, combined by worst-case (``max``) so a clean structure
    with a blown drawdown — or a blown inventory cap — still scores the violation:

    * **structural** — a short under ``long_only`` (fraction of bars holding a short), or
      net exposure away from zero under ``market_neutral`` *or* ``pairs_convergence`` (mean
      ``|net| / gross``). Read off the per-step weight events. ``momentum`` /
      ``unconstrained`` carry no structural rule. (``pairs_convergence`` uses a beta-free
      dollar-neutrality proxy — see the Rust ``mandate_breach`` docs.)
    * **inventory** — per-bar gross exposure ``Σ|w_i|`` over ``max_inventory``, normalized
      by the cap and *squared* (Avellaneda-Stoikov), saturated at 1 per bar then meaned.
    * **drawdown** — realized max drawdown over the cap, normalized by the cap and
      saturated at 1.

    Pure and numpy-light; safe on empty inputs (returns 0). This wrapper adapts the event
    dicts into the weight-vector shape the kernel scores, then delegates the math to Rust.
    """
    mandate = _as_mandate(m)
    weights = _weights_per_step(events)
    return float(
        _rs_mandate_breach(
            json.dumps(mandate.to_dict()),
            [float(r) for r in returns or []],
            [[float(x) for x in w] for w in weights],
        )
    )


__all__ = [
    "Mandate",
    "STYLES",
    "sample_mandate",
    "mandate_text",
    "mandate_breach",
    "mandate_from_dict",
    "validate_mandate",
]
