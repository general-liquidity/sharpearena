"""Stylized-facts realism diagnostic for generated market panels.

A market simulator is only useful if the tapes it emits *look like* real markets. This
module is the diagnostic that certifies exactly that: given a returns/price panel it
computes the canonical empirical stylized facts of financial returns (Cont 2001) and
grades each against a believability threshold.

The facts computed by :func:`stylized_facts`:

* **excess_kurtosis**: leptokurtosis. Real return distributions are fat-tailed (positive
  excess kurtosis over the Gaussian's 3); a Gaussian scores ~0.
* **abs_return_autocorr**: volatility clustering. Signed returns are ~uncorrelated but
  ``|return|`` is positively autocorrelated and decays slowly; the mean of that
  autocorrelation over the first few lags is a one-number clustering score.
* **zumbach_asymmetry**: time-reversal asymmetry (Zumbach 2009). Past coarse-grained
  volatility predicts future fine-grained volatility better than the reverse, so the
  coarse->fine minus fine->coarse correlation gap is positive in real markets and ~0 for a
  time-symmetric process. Computed at a lag equal to the coarse window so the two vol
  measures never share a return.
* **gain_loss_skew**: the gain/loss asymmetry of the return distribution (its skewness);
  equities are typically left-skewed (fatter loss tail).
* **aggregational_gaussianity**: as returns are summed over longer horizons the
  distribution drifts back toward Gaussian, so excess kurtosis decays. The fact is the
  excess kurtosis at horizon 1 minus that of the horizon-aggregated series; positive in
  real markets.
* **fano_factor**: intermittency. Large moves (``|return|`` exceedances) arrive in bursts,
  not as an even Poisson stream, so the Fano factor (variance/mean of exceedance counts per
  window) exceeds the Poisson value of 1.

:func:`certify_realism` grades a panel against :data:`DEFAULT_THRESHOLDS` (or caller
overrides) and returns a :class:`RealismReport` with a per-fact pass/fail and an overall
verdict.

This is a **test gate / diagnostic**, not part of the byte-identical generation hot path,
so it is free to use ``log``/``sqrt`` and other transcendentals. It is pure over an
injected panel: it never touches the native engine or any RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _as_returns(panel, kind: str) -> np.ndarray:
    """Coerce ``panel`` to a 2-D ``(T, n_series)`` returns array.

    ``kind="price"`` takes log returns of a strictly-positive price panel; ``kind="return"``
    treats the panel as returns already. A 1-D panel is read as a single series.
    """
    a = np.asarray(panel, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError("panel must be 1-D or 2-D (time on axis 0, series on axis 1)")
    if kind == "return":
        r = a
    elif kind == "price":
        if a.shape[0] < 2:
            raise ValueError("a price panel needs at least 2 rows to form a return")
        if not np.all(a > 0.0):
            raise ValueError("a price panel must be strictly positive to take log returns")
        r = np.diff(np.log(a), axis=0)
    else:
        raise ValueError("kind must be 'price' or 'return'")
    if r.shape[0] < 2:
        raise ValueError("need at least 2 returns to compute stylized facts")
    return r


def _columns(r: np.ndarray):
    return (r[:, c] for c in range(r.shape[1]))


def _excess_kurtosis_series(x: np.ndarray) -> float:
    x = x - x.mean()
    s = x.std()
    if s == 0.0:
        return np.nan
    return float(np.mean(x**4) / s**4 - 3.0)


def _excess_kurtosis(r: np.ndarray) -> float:
    return float(np.nanmean([_excess_kurtosis_series(x) for x in _columns(r)]))


def _abs_return_autocorr(r: np.ndarray, lags: int) -> float:
    """Mean autocorrelation of ``|return|`` over lags ``1..lags`` (volatility clustering)."""
    vals: list[float] = []
    for x in _columns(r):
        a = np.abs(x)
        a = a - a.mean()
        denom = float(np.sum(a * a))
        if denom == 0.0:
            continue
        for k in range(1, min(lags, len(a) - 1) + 1):
            vals.append(float(np.sum(a[k:] * a[:-k]) / denom))
    return float(np.nanmean(vals)) if vals else np.nan


def _zumbach_asymmetry(r: np.ndarray, window: int) -> float:
    """Coarse->fine minus fine->coarse volatility-correlation gap at lag == ``window``.

    ``fine`` is instantaneous ``|return|``; ``coarse`` is its trailing ``window``-bar mean.
    The lag equals the window so the coarse measure and the lagged fine measure never share
    a return (which would inject a spurious contemporaneous correlation).
    """
    lag = window
    vals: list[float] = []
    kernel = np.ones(window) / window
    for x in _columns(r):
        fine = np.abs(x)
        if len(fine) <= lag + 1:
            continue
        coarse = np.convolve(fine, kernel, mode="full")[: len(fine)]  # trailing MA
        cf = np.corrcoef(coarse[:-lag], fine[lag:])[0, 1]
        fc = np.corrcoef(fine[:-lag], coarse[lag:])[0, 1]
        vals.append(float(cf - fc))
    return float(np.nanmean(vals)) if vals else np.nan


def _gain_loss_skew(r: np.ndarray) -> float:
    vals: list[float] = []
    for x in _columns(r):
        x = x - x.mean()
        s = x.std()
        if s == 0.0:
            continue
        vals.append(float(np.mean(x**3) / s**3))
    return float(np.nanmean(vals)) if vals else np.nan


def _aggregational_gaussianity(r: np.ndarray, horizon: int) -> float:
    """Excess kurtosis at horizon 1 minus that of the ``horizon``-aggregated return series.

    Positive when tails thin toward Gaussian as returns are summed over longer horizons.
    """
    n_blocks = r.shape[0] // horizon
    if n_blocks < 2:
        return np.nan
    agg = r[: n_blocks * horizon].reshape(n_blocks, horizon, r.shape[1]).sum(axis=1)
    return float(_excess_kurtosis(r) - _excess_kurtosis(agg))


def _fano_factor(r: np.ndarray, window: int, z: float) -> float:
    """Fano factor (variance/mean) of large-move exceedance counts per ``window``.

    A large move is ``|return|`` above ``mean + z*std``; counting exceedances per window
    turns the tape into a point process whose Fano factor is 1 under a Poisson (memoryless)
    arrival and > 1 when large moves cluster (intermittency).
    """
    vals: list[float] = []
    for x in _columns(r):
        a = np.abs(x)
        thr = a.mean() + z * a.std()
        events = (a > thr).astype(np.float64)
        n_blocks = len(events) // window
        if n_blocks < 2:
            continue
        counts = events[: n_blocks * window].reshape(n_blocks, window).sum(axis=1)
        mean = counts.mean()
        if mean > 0.0:
            vals.append(float(counts.var() / mean))
    return float(np.nanmean(vals)) if vals else np.nan


def stylized_facts(
    panel,
    *,
    kind: str = "price",
    abs_acf_lags: int = 10,
    coarse_window: int = 5,
    agg_horizon: int = 4,
    fano_window: int = 10,
    fano_z: float = 2.0,
) -> dict[str, float]:
    """Compute the stylized-facts vector of a returns/price ``panel``.

    ``panel`` is a 1-D series or a 2-D ``(T, n_series)`` array; per-series facts are
    averaged. ``kind="price"`` (default) takes log returns of a positive price panel,
    ``kind="return"`` treats the panel as returns. Returns a dict of the six facts described
    in the module docstring. Any fact that is undefined for the given panel (too few bars,
    a constant series) is returned as ``nan`` rather than raising.
    """
    r = _as_returns(panel, kind)
    return {
        "excess_kurtosis": _excess_kurtosis(r),
        "abs_return_autocorr": _abs_return_autocorr(r, abs_acf_lags),
        "zumbach_asymmetry": _zumbach_asymmetry(r, coarse_window),
        "gain_loss_skew": _gain_loss_skew(r),
        "aggregational_gaussianity": _aggregational_gaussianity(r, agg_horizon),
        "fano_factor": _fano_factor(r, fano_window, fano_z),
    }


# Directional believability bounds for a real market, ``fact -> (low, high)`` inclusive
# (``None`` == unbounded). A fact with no entry here is reported but not gated. Defaults
# encode the sign of each empirical stylized fact; skew and Zumbach asymmetry are left
# informational because their sign is asset- and regime-dependent.
DEFAULT_THRESHOLDS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "excess_kurtosis": (0.0, None),            # fat-tailed / leptokurtic
    "abs_return_autocorr": (0.0, None),        # volatility clustering present
    "aggregational_gaussianity": (0.0, None),  # kurtosis decays on aggregation
    "fano_factor": (1.0, None),                # super-Poisson (bursty) large moves
}


@dataclass(frozen=True)
class RealismReport:
    """The graded output of :func:`certify_realism`.

    ``facts`` is the full stylized-facts vector; ``checks`` maps each *gated* fact to its
    pass/fail; ``passed`` is the conjunction over ``checks`` (a panel with no gated facts
    trivially passes). ``thresholds`` records the bounds actually applied.
    """

    facts: dict[str, float]
    checks: dict[str, bool]
    passed: bool
    thresholds: dict[str, tuple[Optional[float], Optional[float]]] = field(default_factory=dict)


def certify_realism(
    panel,
    thresholds: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
    **facts_kwargs,
) -> RealismReport:
    """Grade a ``panel`` against believability ``thresholds`` (defaults merged in).

    ``thresholds`` maps a fact name to an inclusive ``(low, high)`` band (either bound may
    be ``None``). Provided entries override :data:`DEFAULT_THRESHOLDS`; a fact whose value is
    ``nan`` fails its check. Extra keyword args flow through to :func:`stylized_facts`.
    """
    merged = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        merged.update(thresholds)
    facts = stylized_facts(panel, **facts_kwargs)
    checks: dict[str, bool] = {}
    for name, (low, high) in merged.items():
        value = facts.get(name, np.nan)
        ok = np.isfinite(value)
        if ok and low is not None:
            ok = value >= low
        if ok and high is not None:
            ok = value <= high
        checks[name] = bool(ok)
    return RealismReport(
        facts=facts,
        checks=checks,
        passed=all(checks.values()),
        thresholds=merged,
    )


__all__ = [
    "stylized_facts",
    "certify_realism",
    "RealismReport",
    "DEFAULT_THRESHOLDS",
]
