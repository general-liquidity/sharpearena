"""Statistical confidence for the leaderboard ranking (bootstrap CI + paired A/B test).

The benchmark ranks on the **deflated Sharpe** (which discounts overfit-luck) plus the
**pass^k** rate (per-run reliability). Neither answers the question a leaderboard has to
defend when two entries are close: *is A's number better than B's beyond seed noise, or did
A just draw a kinder held-out band?* This module closes that leg (Advances in Financial
Machine Learning, Ch. 19, A/B testing under sampling uncertainty):

* :func:`deflated_sharpe_ci` puts a **seed-paired bootstrap CI** around an entry's deflated
  Sharpe by resampling its held-out seeds (the independent sampling units) with replacement.
  A wide interval means the headline number rests on a few lucky seeds.
* :func:`paired_dsr_diff` runs a **paired-difference significance test** across the *shared*
  held-out seed band: each bootstrap draw feeds the same resampled indices to both entries.
  Pairing retains shared-path covariance; it does not remove all luck or isolate skill.
  A difference CI containing zero does not establish a difference, or prove equivalence.
* :func:`pairwise_significance` applies the paired test down a ranked leaderboard, so each
  neighbouring pair has a directional or unresolved diagnostic. These unadjusted pairwise
  intervals do not establish a simultaneous ranking guarantee or official eligibility.

The heavy lifting is the self-contained Rust core (no ``sharpebench-stats`` dependency).
It uses empirical population standardized moments with an n-normalized second moment,
the convention the packaged ``score_run`` kernel has carried since SharpeBench 0.19.0.
``run_baselines`` keeps the interval under its own name as an Arena diagnostic and
attaches it to the scoring-kernel row only where the kernel reproduces the point
estimate bit for bit; a disagreeing or erring kernel withholds it rather than
presenting an interval for a value it did not bracket. Everything is deterministic in
``resample_seed``.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from typing import Sequence

from .sharpearena_py import bootstrap_dsr_ci as _bootstrap_dsr_ci
from .sharpearena_py import paired_dsr_diff as _paired_dsr_diff
from .kernel_score import is_kernel_score_unavailable, kernel_score_or_unavailable

# The scoring kernel's own bootstrap seed (``ScoreConfig::default().bootstrap_seed``), reused
# so the confidence layer's resampling shares the benchmark's canonical seed by default.
DEFAULT_RESAMPLE_SEED = 0x5BA7_2026
DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05

PerSeedReturns = Sequence[Sequence[float]]


def deflated_sharpe_ci(
    per_seed_returns: PerSeedReturns,
    n_trials: int = 0,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    resample_seed: int = DEFAULT_RESAMPLE_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Seed-paired percentile bootstrap CI on an entry's deflated Sharpe.

    ``per_seed_returns`` is one per-bar return series per held-out seed. ``n_trials`` is the
    entry's *declared* in-sample search budget, folded onto the configured baseline
    footprint Rust-side. This is the corrected Arena estimator, not the older pinned
    ``score_run`` estimator. Returns
    ``{point, lo, hi, width, confidence, n_boot}``.
    """
    rows = [list(map(float, r)) for r in per_seed_returns]
    return json.loads(
        _bootstrap_dsr_ci(rows, int(n_trials), int(n_boot), int(resample_seed), float(alpha))
    )


def paired_dsr_diff(
    a_per_seed_returns: PerSeedReturns,
    b_per_seed_returns: PerSeedReturns,
    n_trials: int = 0,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    resample_seed: int = DEFAULT_RESAMPLE_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Paired-difference significance test between two entries on the **same** seed band.

    ``a_per_seed_returns[i]`` and ``b_per_seed_returns[i]`` must be the two entries' return
    series on the *same* seed ``i``. Positional pairing assumes the caller has aligned the
    identities; it does not independently verify them or their statistical independence.
    Returns ``{point_diff, lo, hi, p_value, confidence, significant, verdict, n_boot}`` with
    ``verdict`` one of ``"a_better"`` / ``"b_better"`` / ``"tied"``. The legacy wire label
    ``tied`` means only that the interval includes zero, not that equivalence was shown.
    """
    a = [list(map(float, r)) for r in a_per_seed_returns]
    b = [list(map(float, r)) for r in b_per_seed_returns]
    return json.loads(
        _paired_dsr_diff(
            a, b, int(n_trials), int(n_boot), int(resample_seed), float(alpha)
        )
    )


def pairwise_significance(
    rows: Sequence[dict],
    n_trials: int = 0,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    resample_seed: int = DEFAULT_RESAMPLE_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    """Paired significance verdict for each adjacent pair down a ranked leaderboard.

    ``rows`` are leaderboard entries (each carrying ``"policy"`` and ``"per_seed_returns"``,
    as produced by :func:`~sharpearena.baselines.run_baselines`). They are ranked by deflated
    Sharpe (desc) and each neighbouring pair ``(A, B)`` is tested; ``A`` is the higher-ranked
    entry under that displayed estimator. The paired diagnostic can point in either
    direction; ``tied`` means the difference was not established, not equivalence.
    Rows without ``"per_seed_returns"`` or a usable kernel score are skipped.
    These adjacent comparisons are
    exploratory and not multiplicity-adjusted.
    """
    usable = [r for r in rows if r.get("per_seed_returns")
              and not is_kernel_score_unavailable(kernel_score_or_unavailable(r))]
    ordered = sorted(usable, key=lambda r: r.get("deflated_sharpe", 0.0), reverse=True)
    out: list[dict] = []
    for higher, lower in zip(ordered, ordered[1:]):
        diff = paired_dsr_diff(
            higher["per_seed_returns"],
            lower["per_seed_returns"],
            n_trials,
            n_boot=n_boot,
            resample_seed=resample_seed,
            alpha=alpha,
        )
        out.append(
            {
                "a": higher.get("policy", "?"),
                "b": lower.get("policy", "?"),
                **diff,
            }
        )
    return out


def _finite_field(record: dict, key: str) -> float:
    value = record.get(key)
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"confidence field {key} must be a finite real number")
    return float(value)


def _validated_interval(record: dict) -> tuple[float, float, float]:
    if not isinstance(record, dict):
        raise ValueError("confidence interval must be a dict")
    lo, hi = _finite_field(record, "lo"), _finite_field(record, "hi")
    confidence = _finite_field(record, "confidence")
    if lo > hi or not 0.0 < confidence < 1.0:
        raise ValueError("confidence interval requires lo <= hi and 0 < confidence < 1")
    return lo, hi, confidence


def _comparison_label(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"comparison {key} must name an entry")
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def significance_markdown(comparisons: Sequence[dict]) -> str:
    """Render each actual direction and confidence level, refusing inconsistent records.

    A CI containing zero says ``difference not established``. This table does not turn
    lack of separation into an equivalence claim or unadjusted comparisons into a ranking
    guarantee. The p-value is displayed as supplied, not used to relabel the CI decision.
    """
    lines = [
        "| A | B | Deflated Sharpe diff | Bootstrap CI | Confidence | p-value | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        lo, hi, confidence = _validated_interval(c)
        point = _finite_field(c, "point_diff")
        p = _finite_field(c, "p_value")
        if not 0.0 <= p <= 1.0 or type(c.get("significant")) is not bool:
            raise ValueError("comparison requires a probability and boolean significant")
        expected = "a_better" if lo > 0.0 else "b_better" if hi < 0.0 else "tied"
        if (c.get("verdict") != expected or c["significant"] != (expected != "tied")
                or (expected == "a_better" and point <= 0.0)
                or (expected == "b_better" and point >= 0.0)):
            raise ValueError("comparison verdict, direction and interval disagree")
        a, b = _comparison_label(c, "a"), _comparison_label(c, "b")
        if expected == "a_better":
            verdict = f"{a} > {b} (CI excludes zero)"
        elif expected == "b_better":
            verdict = f"{b} > {a} (CI excludes zero)"
        else:
            verdict = "difference not established"
        lines.append(
            f"| {a} | {b} | {point:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{100 * confidence:.6g}% | {p:.3g} | {verdict} |"
        )
    return "\n".join(lines)


__all__ = [
    "deflated_sharpe_ci",
    "paired_dsr_diff",
    "pairwise_significance",
    "significance_markdown",
    "DEFAULT_RESAMPLE_SEED",
    "DEFAULT_N_BOOT",
    "DEFAULT_ALPHA",
]
