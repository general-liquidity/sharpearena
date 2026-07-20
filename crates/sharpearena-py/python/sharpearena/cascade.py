"""Forced-liquidation cascade scenario wrapper over :class:`~sharpearena.gym.SharpeArenaEnv`.

A drawdown that breaches the maintenance-margin floor mid-episode triggers a *cascade*: a
margin call forces a position reduction, the forced selling depresses the mark, the lower
mark deepens the equity deficit, and the chain repeats. :class:`LiquidationCascadeEnv`
models that loop as a **deterministic, bounded, in-step** projection — not a generic
event-driven engine — so it stays reproducible and leak-free while still emitting the
structured events (``margin_call`` -> ``forced_reduce`` -> ``cascade_impact``) that the
verifiers ``process_check_reward`` / ``mandate_reward`` machinery already consume from
``info["events"]``.

The cascade is a pure function of the realized post-step NAV, the running peak, and the
three parameters — no RNG, no future bars — so the same path yields byte-identical events.
It is bounded by ``cascade_steps``: each step compounds one ``impact_per_step`` mark drop,
and the chain stops early only when equity is wiped.

terminated vs truncated (mirroring :class:`~sharpearena.risk.DrawdownStopper`):

* a cascade that wipes equity (``final_nav <= 0``) is **terminated** — bankruptcy is an
  absorbing MDP state, there is no future to bootstrap past.
* a cascade that the position survives (``final_nav > 0``) is **truncated** — a margin
  stop-out is an episode *cut*, not an absorbing state; the value estimate should bootstrap
  past it.

While liquidating, the action handed to the underlying env is overridden to flat (zeros):
the operator's position is being force-reduced, so the agent cannot re-lever into the
breach until equity recovers above the margin floor.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym


def _run_cascade(
    nav: float,
    peak: float,
    *,
    maintenance_margin: float,
    cascade_steps: int,
    impact_per_step: float,
) -> tuple[list[dict[str, Any]], float]:
    """The deterministic cascade chain for a breached step. Pure: events + final NAV.

    Emits one ``margin_call`` then up to ``cascade_steps`` ``forced_reduce`` /
    ``cascade_impact`` pairs. ``forced_reduce.fraction`` is the cumulative reduction toward
    flat; ``cascade_impact.mark_drop`` is the equity lost to that step's forced selling —
    ``impact_per_step`` of the liquidated notional (the running ``peak`` is the high-water
    notional proxy), the notional shrinking each step as the position is reduced, so the
    drops compound down. The chain breaks early only if equity is wiped (``<= 0``).
    """
    threshold = maintenance_margin * peak
    events: list[dict[str, Any]] = [
        {"event": "margin_call", "nav": nav, "deficit": threshold - nav}
    ]
    cur = nav
    notional = peak
    for k in range(1, cascade_steps + 1):
        events.append({"event": "forced_reduce", "fraction": min(1.0, k / cascade_steps)})
        mark_drop = notional * impact_per_step
        cur -= mark_drop
        notional *= 1.0 - impact_per_step
        events.append({"event": "cascade_impact", "step": k, "mark_drop": mark_drop})
        if cur <= 0.0:
            break
    return events, cur


class LiquidationCascadeEnv(gym.Wrapper):
    """Wrap an env so a maintenance-margin breach fires a deterministic liquidation cascade.

    After each ``env.step`` the realized equity (``info["nav"]``) is compared to
    ``maintenance_margin * running_peak``. On breach the in-step cascade runs, its events are
    appended to ``info["events"]``, and ``info["cascade"] = {"triggered", "events",
    "final_nav"}`` is surfaced. ``terminated`` is set when the cascade wipes equity, else
    ``truncated`` (stop-out convention). While the breach persists the executed action is
    forced flat.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        maintenance_margin: float = 0.4,
        cascade_steps: int = 3,
        impact_per_step: float = 0.02,
    ) -> None:
        super().__init__(env)
        if not 0.0 < maintenance_margin <= 1.0:
            raise ValueError("maintenance_margin must be in (0, 1]")
        if cascade_steps < 1:
            raise ValueError("cascade_steps must be >= 1")
        if not 0.0 <= impact_per_step < 1.0:
            raise ValueError("impact_per_step must be in [0, 1)")
        self._maintenance_margin = float(maintenance_margin)
        self._cascade_steps = int(cascade_steps)
        self._impact_per_step = float(impact_per_step)
        self._peak: Optional[float] = None
        self._liquidating = False

    def reset(self, **kwargs: Any):
        self._peak = None
        self._liquidating = False
        return self.env.reset(**kwargs)

    def step(self, action):
        executed = np.zeros_like(np.asarray(action)) if self._liquidating else action
        obs, reward, terminated, truncated, info = self.env.step(executed)
        nav = info.get("nav")
        if nav is None:
            return obs, reward, bool(terminated), bool(truncated), info
        nav = float(nav)
        if self._peak is None or nav > self._peak:
            self._peak = nav

        info = dict(info)
        events = list(info.get("events") or [])
        cascade: dict[str, Any] = {"triggered": False, "events": [], "final_nav": nav}

        if self._peak > 0.0 and nav <= self._maintenance_margin * self._peak:
            self._liquidating = True
            chain, final_nav = _run_cascade(
                nav,
                self._peak,
                maintenance_margin=self._maintenance_margin,
                cascade_steps=self._cascade_steps,
                impact_per_step=self._impact_per_step,
            )
            events.extend(chain)
            cascade = {"triggered": True, "events": chain, "final_nav": final_nav}
            if final_nav <= 0.0:
                terminated = True
            else:
                truncated = True
        else:
            self._liquidating = False

        info["events"] = events
        info["cascade"] = cascade
        return obs, reward, bool(terminated), bool(truncated), info


def cascade_survived(info: dict[str, Any]) -> bool:
    """True if no cascade fired this step, or one fired but equity was not wiped.

    The basis for a "survive the cascade" mandate-style evaluation: a survived stop-out is
    acceptable, a wiped one is not.
    """
    cascade = info.get("cascade")
    if not cascade or not cascade.get("triggered"):
        return True
    return float(cascade.get("final_nav", 0.0)) > 0.0


def cascade_summary(info: dict[str, Any]) -> dict[str, Any]:
    """A process-check-friendly digest of a step's cascade, composable with the verifiers
    event consumers. Counts each chain event-type and reports survival."""
    cascade = info.get("cascade") or {}
    events = cascade.get("events", []) if cascade.get("triggered") else []
    counts = {"margin_call": 0, "forced_reduce": 0, "cascade_impact": 0}
    for e in events:
        name = str(e.get("event", ""))
        if name in counts:
            counts[name] += 1
    return {
        "triggered": bool(cascade.get("triggered", False)),
        "margin_calls": counts["margin_call"],
        "forced_reduces": counts["forced_reduce"],
        "cascade_impacts": counts["cascade_impact"],
        "final_nav": float(cascade.get("final_nav", info.get("nav", 0.0))),
        "survived": cascade_survived(info),
    }


__all__ = ["LiquidationCascadeEnv", "cascade_survived", "cascade_summary"]
