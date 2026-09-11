"""The one definition of the held-out seed-band boundary.

``EVAL_SEED_BASE`` was restated as a bare literal in :mod:`sharpearena.dataset`,
:mod:`sharpearena.gym`, :mod:`sharpearena.vector` and :mod:`sharpearena.minari_export`
(the last as a fallback beside an optional import). Four copies of a boundary that has to
agree for the eval band to be disjoint from the train band is a contract kept by review,
and a leak is what disagreement buys. This module holds the value and imports nothing, so
it is reachable from the modules that must not depend on the native binding.

Two deliberate restatements remain, and neither is a copy of this one:
:mod:`sharpearena.effective_config` restates the split rule so the readback is derived
independently of the environment it checks, and the native
``sharpearena_py.EVAL_SEED_BASE`` is the Rust constant, cross-checked against this one at
import time in :mod:`sharpearena.eval_seeds`.
"""

from __future__ import annotations

EVAL_SEED_BASE = 1_000_000

__all__ = ["EVAL_SEED_BASE"]
