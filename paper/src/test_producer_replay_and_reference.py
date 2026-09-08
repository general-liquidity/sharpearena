"""Regressions for two producer honesty defects in the 2026-09-07 audit.

AP5 (``make-sealed-seeds.py``): the commit-reveal check reported
``reveal_replay_verified`` while comparing only the opening bar of a two-day calm
environment, although the declared evaluation is 120 days on the hard tier. These
tests pin the reported quantity to full deployed-tier trajectory equality.

AP3 (``make-f6-adverse-selection.py``): the endogenous gate was named as a
comparison against committed vectors while both of its sides were computed by the
current run. These tests pin the fresh-path gate and the separate frozen-reference
comparison to honest names and honest behaviour.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
_PY_PKG = REPO / "crates" / "sharpearena-py" / "python"
if _PY_PKG.is_dir() and str(_PY_PKG) not in sys.path:
    sys.path.insert(0, str(_PY_PKG))


def _load(stem: str):
    """Import a hyphenated producer module by path."""
    path = SRC / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _try_load(stem: str):
    try:
        return _load(stem), None
    except Exception as exc:  # noqa: BLE001 - reported as a skip reason
        return None, f"{stem} not importable: {exc}"


SEALED, SEALED_SKIP = _try_load("make-sealed-seeds")
F6, F6_SKIP = _try_load("make-f6-adverse-selection")


@unittest.skipIf(SEALED is None, SEALED_SKIP)
class SealedSeedRevealReplay(unittest.TestCase):
    """AP5: the reveal check must replay whole scenarios, not one opening bar."""

    def test_reveal_replay_reports_full_deployed_tier_scope(self):
        seeds = {"held_out_00": 1_000_000, "held_out_01": 1_000_007}
        result = SEALED.reveal_replay(dict(seeds), dict(seeds))
        self.assertEqual(result["verified"], len(seeds))
        self.assertEqual(result["distribution_mode"], SEALED.TIER)
        self.assertEqual(result["n_days"], SEALED.N_DAYS)
        # A first-bar proxy compares one row; the declared evaluation is N_DAYS long.
        self.assertEqual(result["bars_compared"], SEALED.N_DAYS + 1)
        self.assertGreater(result["bars_compared"], 1)

    def test_matching_opening_bar_alone_does_not_count_as_replayed(self):
        import numpy as np

        tapes = {
            1: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            2: np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 99.0]]),
        }
        original_extract = SEALED.extract_closes
        original_first = SEALED.first_closes
        SEALED.extract_closes = lambda seed, mode: tapes[seed]
        SEALED.first_closes = lambda seed, *a, **k: tapes[seed][0]
        try:
            result = SEALED.reveal_replay({"slot": 1}, {"slot": 2})
        finally:
            SEALED.extract_closes = original_extract
            SEALED.first_closes = original_first
        self.assertEqual(result["verified"], 0)
        self.assertFalse(result["slots"][0]["replayed"])

    def test_replay_scope_is_serialized_next_to_the_count(self):
        source = inspect.getsource(SEALED.main)
        self.assertIn("reveal_replay_scope", source)
        self.assertIn("reveal_replay(revealed, sealed_seeds)", source)
        # The opening-bar helper must no longer be the reveal verification.
        self.assertNotIn("first_closes(revealed", source)


@unittest.skipIf(F6 is None, F6_SKIP)
class F6FrozenReferenceNaming(unittest.TestCase):
    """AP3: name the fresh-path gate honestly and add a real frozen comparison."""

    HORIZONS = [1, 5]

    def _vectors(self, tail):
        return {
            "informed": [{"1": 0.1, "5": 0.2}],
            "uninformed": [{"1": 0.3, "5": tail}],
        }

    def test_fresh_path_gate_is_not_named_a_committed_comparison(self):
        params = list(inspect.signature(F6.endogenous_block).parameters)
        self.assertEqual(params[1], "current_run_vectors")
        gate = inspect.getsource(F6.endogenous_block)
        # Neither the compared value nor the failure message may claim a committed
        # reference: both sides of this gate are computed by the current run.
        self.assertNotIn("committed[", gate)
        self.assertNotIn("drifted from committed vectors", gate)
        module = inspect.getsource(F6)
        self.assertNotIn("exogenous_arm_matches_committed_vectors", module)
        self.assertIn("exogenous_arm_matches_current_run_vectors", module)

    def test_frozen_reference_comparison_reads_the_committed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f6-adverse-selection.json"
            path.write_text(
                json.dumps({"per_episode_markout_per_unit": self._vectors(0.4)})
            )
            same = F6.frozen_reference_comparison(
                self._vectors(0.4), self.HORIZONS, path=path
            )
            self.assertEqual(same["status"], "compared")
            self.assertTrue(same["matches"])

            drifted = F6.frozen_reference_comparison(
                self._vectors(0.5), self.HORIZONS, path=path
            )
            self.assertEqual(drifted["status"], "compared")
            self.assertFalse(drifted["matches"])
            self.assertEqual(drifted["n_mismatches"], 1)

    def test_absent_or_unreadable_reference_is_reported_not_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            absent = F6.frozen_reference_comparison(
                self._vectors(0.4), self.HORIZONS, path=missing
            )
            self.assertEqual(absent["status"], "absent")
            self.assertIsNone(absent["matches"])

            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json")
            unreadable = F6.frozen_reference_comparison(
                self._vectors(0.4), self.HORIZONS, path=broken
            )
            self.assertEqual(unreadable["status"], "unreadable")
            self.assertIsNone(unreadable["matches"])

    def test_frozen_reference_is_read_before_the_artifact_is_overwritten(self):
        source = inspect.getsource(F6.main)
        read_at = source.index("frozen_reference_comparison(per_episode")
        write_at = source.index("FROZEN_REFERENCE.write_text")
        self.assertLess(read_at, write_at)


if __name__ == "__main__":
    unittest.main()
