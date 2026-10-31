"""Shared measurement for the generator calibration study.

The replicate is the seed: one seed is one independently drawn market, so every
confidence interval below is taken across seeds, never across symbols or bars.
``realism.py`` is loaded by path because it needs only numpy, and the gate it applies is
the shipped one with its default thresholds.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "sharpearena_realism_by_path",
    REPO / "crates/sharpearena-py/python/sharpearena/realism.py",
)
realism = importlib.util.module_from_spec(_SPEC)
import sys  # noqa: E402

sys.modules[_SPEC.name] = realism  # dataclasses resolves annotations through sys.modules
_SPEC.loader.exec_module(realism)

BARS_PER_YEAR = 252.0
TREND_LOOKBACK = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panels(path: Path, n_days: int, n_symbols: int) -> np.ndarray:
    panels = np.fromfile(path, dtype="<f8")
    return panels.reshape(-1, n_days, n_symbols)


def simple_returns(closes: np.ndarray) -> np.ndarray:
    return closes[1:] / closes[:-1] - 1.0


def lag1_autocorr(r: np.ndarray) -> float:
    """Mean over symbols of the demeaned lag-1 autocorrelation of ``r`` (T, N)."""
    x = r - r.mean(axis=0, keepdims=True)
    denom = np.sum(x * x, axis=0)
    keep = denom > 0.0
    if not keep.any():
        return float("nan")
    return float(np.mean(np.sum(x[1:] * x[:-1], axis=0)[keep] / denom[keep]))


def annualized_sharpe(r: np.ndarray) -> np.ndarray:
    """Per-column annualized Sharpe; a constant column is undefined, not zero."""
    sd = r.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r.mean(axis=0) / sd * np.sqrt(BARS_PER_YEAR)
    return np.where(sd > 0.0, out, np.nan)


def strategy_returns(r: np.ndarray) -> dict[str, np.ndarray]:
    """Gross per-bar returns of four rules that see only past bars, one-bar lagged."""
    equal = r.mean(axis=1)
    one_bar = np.sign(r[:-1])
    momentum = (one_bar * r[1:]).mean(axis=1)
    cum = np.cumsum(r, axis=0)
    past = cum[TREND_LOOKBACK - 1 :] - np.vstack(
        [np.zeros((1, r.shape[1])), cum[: -TREND_LOOKBACK]]
    )
    trend = (np.sign(past[:-1]) * r[TREND_LOOKBACK:]).mean(axis=1)
    return {
        "buy_and_hold_equal_weight": equal,
        "momentum_1bar": momentum,
        "reversal_1bar": -momentum,
        "trend_20bar": trend,
    }


def vol_forecast_corr(r: np.ndarray, lookback: int = 10) -> float:
    """Mean over symbols of corr(trailing mean |r|, next |r|): what risk an agent can see."""
    vals = []
    a = np.abs(r)
    kernel = np.ones(lookback) / lookback
    for c in range(a.shape[1]):
        trailing = np.convolve(a[:, c], kernel, mode="valid")[:-1]
        following = a[lookback:, c]
        if trailing.std() > 0.0 and following.std() > 0.0:
            vals.append(float(np.corrcoef(trailing, following)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def cross_symbol_corr(r: np.ndarray) -> float:
    """Mean off-diagonal contemporaneous correlation."""
    c = np.corrcoef(r, rowvar=False)
    return float(c[~np.eye(c.shape[0], dtype=bool)].mean())


def seed_row(closes: np.ndarray) -> dict[str, float]:
    """Every statistic for one panel of closes shaped (T, N)."""
    r = simple_returns(closes)
    report = realism.certify_realism(np.log(closes[1:] / closes[:-1]), kind="return")
    row: dict[str, float] = {
        "lag1_autocorr": lag1_autocorr(r),
        "mean_daily_return": float(r.mean()),
        "daily_vol": float(r.std(axis=0, ddof=1).mean()),
        "excess_kurtosis": report.facts["excess_kurtosis"],
        "abs_return_autocorr": report.facts["abs_return_autocorr"],
        "aggregational_gaussianity": report.facts["aggregational_gaussianity"],
        "realism_pass": float(report.passed),
        "vol_forecast_corr": vol_forecast_corr(r),
        "cross_symbol_corr": cross_symbol_corr(r),
    }
    for name, passed in report.checks.items():
        row[f"check_{name}"] = float(passed)
    sharpe = annualized_sharpe(r)
    row["buy_and_hold_sharpe_mean_symbol"] = float(np.nanmean(sharpe))
    for i, value in enumerate(sharpe):
        row[f"buy_and_hold_sharpe_sym{i:02d}"] = float(value)
    for name, series in strategy_returns(r).items():
        row[f"sharpe_{name}"] = float(annualized_sharpe(series[:, None])[0])
    return row


def summarize(values: np.ndarray) -> dict[str, float | int]:
    """Mean with a t-based 95% interval across seeds, plus the spread of seeds."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return {"n": int(n)}
    sem = float(v.std(ddof=1) / np.sqrt(n))
    half = 1.96 * sem  # n >= 128 everywhere it is used
    q = np.quantile(v, [0.05, 0.5, 0.95])
    return {
        "n": int(n),
        "mean": float(v.mean()),
        "ci95_low": float(v.mean() - half),
        "ci95_high": float(v.mean() + half),
        "sd": float(v.std(ddof=1)),
        "q05": float(q[0]),
        "median": float(q[1]),
        "q95": float(q[2]),
    }


def measure(panels: np.ndarray) -> dict[str, dict[str, float | int]]:
    rows = [seed_row(p) for p in panels]
    keys = sorted({k for row in rows for k in row})
    return {k: summarize(np.array([row.get(k, np.nan) for row in rows])) for k in keys}


def realism_counts(panels: np.ndarray) -> dict[str, int]:
    """Pass counts under the shipped gate. A development measurement only."""
    rows = [seed_row(p) for p in panels]
    out = {"panels": len(rows), "passed": int(sum(r["realism_pass"] for r in rows))}
    for key in rows[0]:
        if key.startswith("check_"):
            out[key] = int(sum(r[key] for r in rows))
    return out
