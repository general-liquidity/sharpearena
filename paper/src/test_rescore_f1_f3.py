"""The F1 and F3 rescore: recomputed from recorded series, typed where unavailable.

Two tests hold the producer's scoring to the package functions it stands in for, on
a small real rollout, so a change to either side fails here. The rest hold the
committed outputs to the rules the rescore exists to keep: a flat row reports the
kernel's reason and never a number, the frozen inputs are untouched, the manuscript's
worked instance reads 0.1114, and a written output is never replaced.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
EVIDENCE = SRC.parent / "evidence"
_PY_PKG = REPO / "crates" / "sharpearena-py" / "python"
if _PY_PKG.is_dir() and str(_PY_PKG) not in sys.path:
    sys.path.insert(0, str(_PY_PKG))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load():
    spec = importlib.util.spec_from_file_location("make_rescore", SRC / "make-rescore-f1-f3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    RESCORE = _load()
    import sharpearena
    from sharpearena import SharpeArenaEnv

    SKIP = None
except ImportError as error:  # pragma: no cover - the native extension is absent
    RESCORE, sharpearena, SharpeArenaEnv, SKIP = None, None, None, str(error)

KERNEL = {"sharpebench_pin": "0.28.0"}
F1_OUT = EVIDENCE / "f1-baselines.rescore-v1-sb0.28.0.json"
F3_OUT = EVIDENCE / "f3-generalization.rescore-v1-sb0.28.0.json"


def _without_series(row):
    return {key: value for key, value in row.items() if key != "per_seed_returns"}


@unittest.skipIf(SKIP, SKIP)
class ScoringParity(unittest.TestCase):
    def test_a_row_rebuilt_from_series_equals_the_row_run_baselines_wrote(self):
        rows = sharpearena.run_baselines(
            n_symbols=2,
            n_days=40,
            seeds=[0, 1, 2],
            distribution_mode="hard",
            confidence=True,
            n_boot=50,
            resample_seed=7,
        )
        trials = len(RESCORE.BASELINE_POLICIES)
        for row in rows:
            rebuilt = RESCORE.rescore_policy(
                row["policy"],
                row["per_seed_returns"],
                trials,
                n_boot=50,
                resample_seed=7,
                alpha=0.05,
            )
            self.assertEqual(rebuilt, _without_series(row), row["policy"])

    def test_a_band_rebuilt_from_series_equals_evaluate_seeds(self):
        def make_env(seed):
            return SharpeArenaEnv(n_symbols=2, n_days=40, seed=seed, distribution_mode="calm")

        def policy(obs):
            n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
            return np.full((n,), 1.0 / n, dtype=np.float32)

        series = []
        for seed in (0, 1, 2):
            env = make_env(seed)
            obs, _ = env.reset()
            returns = []
            while True:
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                returns.append(float(reward))
                if terminated or truncated or len(returns) >= 512:
                    break
            series.append(returns)
        expected = sharpearena.generalization.evaluate_seeds(
            make_env, [0, 1, 2], policy, n_trials=0
        )
        self.assertEqual(RESCORE.evaluate_band(series, 0), expected)


@unittest.skipIf(SKIP, SKIP)
class Refusals(unittest.TestCase):
    def test_a_constant_policy_reports_the_kernels_reason_and_no_number(self):
        row = RESCORE.rescore_policy(
            "flat", [[0.0] * 20, [0.0] * 20], 6, n_boot=20, resample_seed=1, alpha=0.05
        )
        reason = "unavailable_scoring_kernel_error: deflation_error:"
        self.assertTrue(row["deflated_sharpe"].startswith(reason), row)
        self.assertTrue(row["passed_k_rate"].startswith(reason), row)
        self.assertIsNone(row["deflated_sharpe_ci"])
        self.assertEqual(row["confidence_status"], row["deflated_sharpe"])

    def test_an_interval_the_kernel_did_not_reproduce_is_withheld(self):
        series = [[0.01, -0.02, 0.03, 0.005], [0.02, -0.01, 0.015, -0.004]]
        real = RESCORE.deflated_sharpe_ci
        RESCORE.deflated_sharpe_ci = lambda *a, **k: {**real(*a, **k), "point": 0.123}
        try:
            row = RESCORE.rescore_policy("p", series, 6, n_boot=20, resample_seed=1, alpha=0.05)
        finally:
            RESCORE.deflated_sharpe_ci = real
        self.assertIsNone(row["deflated_sharpe_ci"])
        self.assertEqual(row["confidence_status"], "unavailable_scoring_kernel_mismatch")

    def test_a_series_that_is_not_the_scored_one_stops_the_run(self):
        with self.assertRaises(SystemExit):
            RESCORE.require_series_match("cell", [[0.1, 0.2]], 0.1500000001)

    def test_a_frozen_input_with_other_bytes_stops_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "f.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                RESCORE.load_frozen("f.json", "0" * 64, root)

    def test_a_written_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            self.assertEqual(RESCORE.write_versioned(path, "one\n"), "wrote")
            self.assertEqual(RESCORE.write_versioned(path, "one\n"), "unchanged")
            with self.assertRaises(SystemExit):
                RESCORE.write_versioned(path, "two\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "one\n")

    def test_the_output_name_carries_the_kernel_it_was_scored_under(self):
        self.assertEqual(
            RESCORE.output_path("f1-baselines", KERNEL, Path(".")).name,
            "f1-baselines.rescore-v1-sb0.28.0.json",
        )

    def test_the_worked_instance_stops_the_run_when_it_does_not_read_0_1114(self):
        frozen_f1 = RESCORE.load_frozen(RESCORE.FROZEN_F1, RESCORE.FROZEN_F1_SHA256)
        frozen_f3 = RESCORE.load_frozen(RESCORE.FROZEN_F3, RESCORE.FROZEN_F3_SHA256)
        self.assertEqual(RESCORE.worked_instance(frozen_f1, frozen_f3)["printed"], "0.1114")
        original = RESCORE.WORKED_INSTANCE_TRIALS
        RESCORE.WORKED_INSTANCE_TRIALS = 0  # the same series reads 0.1231 at no trials
        try:
            with self.assertRaises(SystemExit):
                RESCORE.worked_instance(frozen_f1, frozen_f3)
        finally:
            RESCORE.WORKED_INSTANCE_TRIALS = original


class CommittedRescore(unittest.TestCase):
    def setUp(self):
        self.f1 = json.loads(F1_OUT.read_text(encoding="utf-8"))
        self.f3 = json.loads(F3_OUT.read_text(encoding="utf-8"))

    def test_every_flat_cell_is_typed_and_no_flat_cell_carries_a_number(self):
        for tier, body in self.f1["tiers"].items():
            flat = next(row for row in body["rows"] if row["policy"] == "flat")
            for key in ("deflated_sharpe", "passed_k_rate", "deflated_sharpe_delta"):
                self.assertIsInstance(flat[key], str, (tier, key))
                self.assertTrue(flat[key].startswith("unavailable_scoring_kernel_error"))
            self.assertIsNone(flat["deflated_sharpe_ci"])
            self.assertIsNone(flat["arena_deflated_sharpe_ci"])
            self.assertEqual(body["unavailable_policies"], ["flat"])

    def test_every_scored_row_carries_an_interval_the_kernel_reproduced(self):
        for body in self.f1["tiers"].values():
            for row in body["rows"]:
                if row["policy"] == "flat":
                    continue
                self.assertEqual(row["confidence_status"], "scoring_kernel_reproduced")
                self.assertEqual(row["deflated_sharpe_ci"]["point"], row["deflated_sharpe"])

    def test_the_claims_the_tables_make_survive_the_rescore(self):
        self.assertEqual(
            {tier: body["n_policies_all_seeds_pass"] for tier, body in self.f1["tiers"].items()},
            {"calm": 0, "hard": 0, "extreme": 0},
        )
        self.assertEqual(self.f1["tiers"]["calm"]["best_passed_k_rate"], 0.75)
        gaps = self.f3["generalization_gap"]
        printed = {"calm": "+0.0012", "hard": "-0.3388", "extreme": "-0.1012"}
        for tier, text in printed.items():
            self.assertEqual(f"{gaps[tier]['gap_deflated_sharpe']:+.4f}", text)
        transfer = self.f3["cross_regime_transfer"]["calm->extreme"]
        self.assertEqual(f"{transfer['transfer_gap_deflated_sharpe']:+.3f}", "+0.973")

    def test_both_outputs_name_the_frozen_input_and_the_worked_instance(self):
        for document, name in ((self.f1, "f1-baselines.json"), (self.f3, "f3-generalization.json")):
            self.assertEqual(document["artifact_kind"], "rescore")
            self.assertEqual(document["frozen_input"]["path"], f"paper/evidence/{name}")
            self.assertEqual(document["worked_instance"]["printed"], "0.1114")
            self.assertEqual(document["worked_instance"]["trials"], 6)

    @unittest.skipIf(SKIP, SKIP)
    def test_the_frozen_inputs_still_hash_to_what_the_rescore_is_bound_to(self):
        RESCORE.load_frozen(RESCORE.FROZEN_F1, RESCORE.FROZEN_F1_SHA256)
        RESCORE.load_frozen(RESCORE.FROZEN_F3, RESCORE.FROZEN_F3_SHA256)


if __name__ == "__main__":
    unittest.main()
