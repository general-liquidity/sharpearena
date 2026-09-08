"""Regressions for three producer-scope defects in the 2026-09-07 audit.

AP1 (``make-predictability.py``): the oracle was scored on the whole tape while
the two causal adversaries were scored from ``WARMUP`` on, so the reported DSR
gap mixed predictive power with a longer scoring window. These tests pin every
adversary to one window and pin the window and the cost model to the record.

AP4 (``make-sealed-seeds.py``): the salt commitment was a local variable that
first reached disk in the same file as the reveal, after both attacks. These
tests pin the commitment to a separate artifact written before anything is
observed, carrying no reveal field and stating what it does not witness.

AP6 (``make-figures.py``): the all-figure renderer dispatched eight functions
and left seven committed PDFs unreachable. These tests pin the registry to the
committed figure set and rebuild every entry from frozen input with the native
bindings made unavailable.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
FIGURES = REPO / "paper" / "figures"
_PY_PKG = REPO / "crates" / "sharpearena-py" / "python"
if _PY_PKG.is_dir() and str(_PY_PKG) not in sys.path:
    sys.path.insert(0, str(_PY_PKG))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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


PRED, PRED_SKIP = _try_load("make-predictability")
SEALED, SEALED_SKIP = _try_load("make-sealed-seeds")
FIGS, FIGS_SKIP = _try_load("make-figures")


@unittest.skipIf(PRED is None, PRED_SKIP)
class PredictabilityCommonSupport(unittest.TestCase):
    """AP1: the three adversaries must be scored on the same bars, at the same costs."""

    def _returns(self) -> np.ndarray:
        rng = np.random.default_rng(11)
        return rng.normal(0.0, 0.01, size=(PRED.WARMUP + 40, 3))

    def test_oracle_is_masked_to_the_causal_window(self):
        rets = self._returns()
        preds = PRED.oracle_predictions(rets)
        self.assertTrue(np.all(np.isnan(preds[: PRED.WARMUP])))
        np.testing.assert_allclose(preds[PRED.WARMUP :], rets[PRED.WARMUP :])

    def test_every_adversary_is_scored_on_the_same_bars(self):
        rets = self._returns()
        evals = [
            PRED.evaluate(PRED.prefix_mean_predictions(rets), rets),
            PRED.evaluate(PRED.ridge_ar_predictions(rets), rets),
            PRED.evaluate(PRED.oracle_predictions(rets), rets),
        ]
        self.assertEqual(
            {ev["scored_bars"] for ev in evals}, {rets.shape[0] - PRED.WARMUP}
        )
        self.assertEqual({ev["first_scored_bar"] for ev in evals}, {PRED.WARMUP})
        # The oracle must not trade a prefix during which the others are flat.
        self.assertEqual(
            {len(ev["policy_returns"]) for ev in evals}, {evals[0]["scored_bars"]}
        )

    def test_scoring_the_oracle_on_the_full_tape_is_a_different_support(self):
        # Guards against the defect returning as an unmasked oracle: the old call
        # gave the oracle WARMUP extra scored bars.
        rets = self._returns()
        full = PRED.evaluate(rets.copy(), rets)
        masked = PRED.evaluate(PRED.oracle_predictions(rets), rets)
        self.assertEqual(full["scored_bars"] - masked["scored_bars"], PRED.WARMUP)

    def test_unequal_support_fails_the_run_rather_than_reporting_the_gap(self):
        source = inspect.getsource(PRED.main)
        self.assertIn("oracle_predictions(rets)", source)
        self.assertNotIn("evaluate(rets.copy(), rets)", source)
        self.assertIn("adversaries scored on different bars", source)

    def test_window_and_costs_are_serialized_next_to_the_numbers(self):
        source = inspect.getsource(PRED.main)
        for key in ("scoring_window", "warmup_bars", "scored_bars", "costs"):
            self.assertIn(key, source)


@unittest.skipIf(SEALED is None, SEALED_SKIP)
class SealedSeedPreRunCommitment(unittest.TestCase):
    """AP4: the commitment must be its own artifact, written before any outcome."""

    SALT = b"\x01" * 32

    def _write(self, tmp: str) -> tuple[Path, dict]:
        path = Path(tmp) / "sealed-seeds-commitment.json"
        digest = SEALED.hashlib.sha256(self.SALT).hexdigest()
        record = SEALED.write_commitment(digest, path=path)
        return path, record

    def test_commitment_is_persisted_before_either_attack_runs(self):
        source = inspect.getsource(SEALED.main)
        commit_at = source.index("write_commitment(commitment)")
        self.assertLess(commit_at, source.index("public = attack("))
        self.assertLess(commit_at, source.index("sealed = attack("))
        # ... and before the file that carries the reveal is written.
        self.assertLess(commit_at, source.index("out.write_text"))

    def test_commitment_artifact_is_separate_from_the_reveal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, record = self._write(tmp)
            self.assertTrue(path.exists())
            self.assertNotEqual(path.name, "sealed-seeds.json")
            on_disk = json.loads(path.read_text())
            self.assertEqual(
                [k for k in on_disk if "salt" in k], ["salt_commitment_sha256"]
            )
            self.assertNotIn(self.SALT.hex(), path.read_text())
            self.assertEqual(record["reveal_artifact"], "sealed-seeds.json")

    def test_record_states_what_it_does_not_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, record = self._write(tmp)
            self.assertIn("witnesses:", record["limits"])
            self.assertIn("does not witness:", record["limits"])

    def test_reveal_is_checked_against_the_file_not_the_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._write(tmp)
            good = SEALED.verify_commitment(self.SALT, path=path)
            self.assertEqual(good["status"], "compared")
            self.assertTrue(good["matches"])
            self.assertEqual(good["reveal_fields_in_commitment"], [])

            wrong = SEALED.verify_commitment(b"\x02" * 32, path=path)
            self.assertEqual(wrong["status"], "compared")
            self.assertFalse(wrong["matches"])

    def test_absent_or_leaky_commitment_is_reported_not_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = SEALED.verify_commitment(self.SALT, path=Path(tmp) / "nope.json")
            self.assertEqual(missing["status"], "absent")
            self.assertIsNone(missing["matches"])

            leaky = Path(tmp) / "leaky.json"
            leaky.write_text(
                json.dumps(
                    {
                        "salt_commitment_sha256": SEALED.hashlib.sha256(
                            self.SALT
                        ).hexdigest(),
                        "salt_revealed_hex": self.SALT.hex(),
                    }
                )
            )
            leaked = SEALED.verify_commitment(self.SALT, path=leaky)
            self.assertEqual(
                leaked["reveal_fields_in_commitment"], ["salt_revealed_hex"]
            )

    def test_the_run_refuses_a_commitment_it_cannot_verify(self):
        source = inspect.getsource(SEALED.main)
        self.assertIn('commitment_check["status"] == "compared"', source)
        self.assertIn(
            'assert not commitment_check["reveal_fields_in_commitment"]', source
        )


@unittest.skipIf(FIGS is None, FIGS_SKIP)
class FigureRendererRegistry(unittest.TestCase):
    """AP6: the registry must reach every committed figure, from frozen input."""

    EXTENSIONS = (
        "f4-calm-calibration.pdf",
        "f5-concave.pdf",
        "f5-positive-control.pdf",
        "f5-extended-sweeps.pdf",
        "f6-endogenous.pdf",
        "predictability.pdf",
        "witness.pdf",
    )

    def test_registry_claims_every_committed_figure(self):
        self.assertEqual(FIGS.unregistered_figures(), [])
        for name in self.EXTENSIONS:
            self.assertIn(name, FIGS.REGISTERED_FIGURES)

    def test_every_entry_reads_a_committed_evidence_file(self):
        for entry in FIGS.REGISTRY:
            self.assertTrue((FIGS.EVIDENCE / entry.evidence).exists(), entry.evidence)

    def test_registry_rebuilds_every_claimed_figure_without_bindings(self):
        """The real rebuild, not a dispatch-name proxy.

        ``sharpearena`` is replaced by a module that refuses to import, so this
        also establishes the frozen-input claim: the committed JSON is the only
        input. The full rebuild was measured at 4.9 s before it was chosen
        over an assertion over the registry table.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / "stub" / "sharpearena"
            stub.mkdir(parents=True)
            (stub / "__init__.py").write_text(
                'raise ImportError("native bindings withheld by the test")'
            )
            out = root / "figures"
            out.mkdir()
            driver = root / "driver.py"
            driver.write_text(
                textwrap.dedent(
                    f"""
                    import importlib.util, sys
                    from pathlib import Path
                    sys.path.insert(0, {str(stub.parent)!r})
                    src = Path({str(SRC)!r})
                    sys.path.insert(1, str(src))
                    spec = importlib.util.spec_from_file_location(
                        "make_figures", src / "make-figures.py"
                    )
                    mf = importlib.util.module_from_spec(spec)
                    sys.modules["make_figures"] = mf
                    spec.loader.exec_module(mf)
                    out = Path({str(out)!r})
                    mf.FIGURES = out
                    inner = mf.producer
                    def patched(stem):
                        module = inner(stem)
                        module.FIGURES = out
                        return module
                    mf.producer = patched
                    mf.main()
                    """
                )
            )
            started = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(driver)],
                capture_output=True,
                text=True,
                cwd=str(root),
            )
            elapsed = time.perf_counter() - started
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("omission:", proc.stdout)
            rebuilt = {p.name for p in out.glob("*.pdf")}
            self.assertEqual(rebuilt, set(FIGS.REGISTERED_FIGURES))
            self.assertLess(elapsed, 120.0)
            # The committed PDFs are frozen evidence; the rebuild went elsewhere.
            self.assertEqual(rebuilt - {p.name for p in FIGURES.glob("*.pdf")}, set())

    def test_an_unreachable_figure_is_named_not_silently_skipped(self):
        original = FIGS.REGISTERED_FIGURES
        FIGS.REGISTERED_FIGURES = frozenset(original - {"witness.pdf"})
        try:
            self.assertEqual(FIGS.unregistered_figures(), ["witness.pdf"])
        finally:
            FIGS.REGISTERED_FIGURES = original


if __name__ == "__main__":
    unittest.main()
