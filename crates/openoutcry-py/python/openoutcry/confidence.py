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
  held-out seed band: each bootstrap draw feeds the same resampled seeds to both entries, so
  the common price-path luck cancels and the difference isolates skill. A difference CI that
  straddles zero means the two entries are statistically tied.
* :func:`pairwise_significance` applies the paired test down a ranked leaderboard, so each
  neighbouring pair is labelled ``a_better`` / ``tied`` and the ranking states which gaps are
  real and which are within seed noise.

The heavy lifting is the self-contained Rust core (no ``sharpebench-stats`` dependency); the
deflation footprint is folded on the Rust side so a CI here brackets the same point deflated
Sharpe :func:`~openoutcry.score_run` reports. Everything is deterministic in
``resample_seed``, so a confidence report replays bit-for-bit.
"""

from __future__ import annotations

import json
from typing import Sequence

from .openoutcry_py import bootstrap_dsr_ci as _bootstrap_dsr_ci
from .openoutcry_py import paired_dsr_diff as _paired_dsr_diff

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
    entry's *declared* in-sample search budget (folded onto the kernel's baseline footprint
    Rust-side, so the CI brackets the ``score_run`` point). Returns
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
    series on the *same* seed ``i`` (the pairing is what cancels the shared price-path luck).
    Returns ``{point_diff, lo, hi, p_value, confidence, significant, verdict, n_boot}`` with
    ``verdict`` one of ``"a_better"`` / ``"b_better"`` / ``"tied"``.
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
    as produced by :func:`~openoutcry.baselines.run_baselines`). They are ranked by deflated
    Sharpe (desc) and each neighbouring pair ``(A, B)`` is tested; ``A`` is the higher-ranked
    entry, so ``verdict == "a_better"`` means the rank gap is real and ``"tied"`` means the
    two are within seed noise. Rows without ``"per_seed_returns"`` are skipped.
    """
    usable = [r for r in rows if r.get("per_seed_returns")]
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


def significance_markdown(comparisons: Sequence[dict]) -> str:
    """Render :func:`pairwise_significance` output as a markdown table.

    One row per adjacent leaderboard pair: the ranked-above entry ``A``, the ranked-below
    entry ``B``, the deflated-Sharpe difference with its bootstrap CI, the two-sided p-value,
    and a plain-English verdict (``A > B beyond seed noise`` vs ``statistically tied``).
    """
    lines = [
        "| A (ranked above) | B (ranked below) | Deflated Sharpe diff | 95% CI | p-value | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        verdict = (
            f"{c['a']} > {c['b']} beyond seed noise"
            if c.get("significant")
            else "statistically tied"
        )
        lines.append(
            "| {a} | {b} | {diff:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {verdict} |".format(
                a=c.get("a", "?"),
                b=c.get("b", "?"),
                diff=float(c.get("point_diff", 0.0)),
                lo=float(c.get("lo", 0.0)),
                hi=float(c.get("hi", 0.0)),
                p=float(c.get("p_value", 1.0)),
                verdict=verdict,
            )
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
