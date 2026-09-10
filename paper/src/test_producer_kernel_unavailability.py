"""Producers stop on a withheld kernel score instead of writing its floor.

``make-f3-generalization.py``, ``make-predictability.py`` and ``make-witness.py``
regenerate frozen evidence from ``score_run``. Since SharpeBench 0.19.0 a
composite the kernel could not score carries ``deflation_error`` /
``bootstrap_error`` beside a no-skill floor. Each producer's scoring helper is
exercised here with a monkeypatched ``score_run`` returning that shape and must
raise ``KernelScoreUnavailable`` naming the error, never return ``0.0``. No
evidence file is read or written.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
_PY_PKG = REPO / "crates" / "sharpearena-py" / "python"
if _PY_PKG.is_dir() and str(_PY_PKG) not in sys.path:
    sys.path.insert(0, str(_PY_PKG))

try:
    from sharpearena.kernel_score import KernelScoreUnavailable
except ImportError:  # pragma: no cover - reported as a skip reason
    KernelScoreUnavailable = None

REASON = "observation 1 must be finite"
FLOORED = json.dumps(
    {
        "deflated_sharpe": 0.0,
        "psr": None,
        "passed_k": False,
        "bootstrap_p": 1.0,
        "process_ok": True,
        "mandate_ok": True,
        "rank_eligible": False,
        "deflation_error": REASON,
        "bootstrap_error": REASON,
    }
)


def _load(stem: str):
    path = SRC / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _try_load(stem: str):
    try:
        module = _load(stem)
    except Exception as exc:  # noqa: BLE001 - reported as a skip reason
        return None, f"{stem} not importable: {exc}"
    if getattr(module, "score_run", None) is None:
        return None, f"{stem} loaded without the native binding"
    return module, None


F3, F3_SKIP = _try_load("make-f3-generalization")
PRED, PRED_SKIP = _try_load("make-predictability")
WITNESS, WITNESS_SKIP = _try_load("make-witness")


@unittest.skipIf(KernelScoreUnavailable is None, "sharpearena package not importable")
class ProducersRaiseOnWithheldScores(unittest.TestCase):
    def _floor(self, module):
        original = module.score_run
        module.score_run = lambda *args, **kwargs: FLOORED
        self.addCleanup(setattr, module, "score_run", original)

    @unittest.skipIf(F3 is None, F3_SKIP)
    def test_f3_pooled_dsr_raises(self):
        self._floor(F3)
        with self.assertRaises(KernelScoreUnavailable) as caught:
            F3._pooled_dsr([[0.01, -0.02], [0.03, 0.0]])
        self.assertIn("deflation_error", str(caught.exception))

    @unittest.skipIf(PRED is None, PRED_SKIP)
    def test_predictability_dsr_raises(self):
        self._floor(PRED)
        with self.assertRaises(KernelScoreUnavailable) as caught:
            PRED.dsr(np.array([0.01, -0.02, 0.03]))
        self.assertIn("deflation_error", str(caught.exception))

    @unittest.skipIf(WITNESS is None, WITNESS_SKIP)
    def test_witness_score_strength_raises_before_any_row(self):
        self._floor(WITNESS)
        original = WITNESS.rollout
        WITNESS.rollout = lambda *a, **k: {
            "returns": [0.01, -0.02, 0.03],
            "accuracy": 0.5,
            "tape_invariant": True,
        }
        self.addCleanup(setattr, WITNESS, "rollout", original)
        with self.assertRaises(KernelScoreUnavailable) as caught:
            WITNESS.score_strength([1], "calm", 0.5, "signal")
        self.assertIn("bootstrap_error", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
