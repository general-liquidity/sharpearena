"""Exercise the Harbor grader, metric, collect hook and fixture scripts.

Nothing here needs Docker or the Harbor SDK. The shell cases run the real fixture
scripts and the real task.toml hook against a throwaway root, which proves their
logic, not a container boundary. They need POSIX symlinks and are skipped on Windows.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

HARBOR = Path(__file__).resolve().parents[1] / "integrations/harbor"
TASK = HARBOR / "tasks/sa-replay-feasibility"
GRADER_PATH = TASK / "tests/grade.py"
METRIC_PATH = HARBOR / "metric.py"
SCRIPTS_PATH = HARBOR / "fixtures/sa_harbor_fixtures/scripts.py"

posix_only = unittest.skipUnless(os.name == "posix", "needs POSIX shell and symlinks")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point_grader_at(grader, artifacts, private, verifier):
    grader.ARTIFACTS_DIR = artifacts
    grader.ARTIFACT = artifacts / "actions.jsonl"
    grader.TRANSFER_CHECK = artifacts / ".sa-transfer-check.json"
    grader.PRIVATE = private
    grader.VERIFIER_DIR = verifier
    grader.REWARD_PATH = verifier / "reward.txt"
    grader.REWARD_JSON_PATH = verifier / "reward.json"
    grader.REPORT_PATH = verifier / "report.json"


class HarborGraderTests(unittest.TestCase):
    def setUp(self):
        self.grader = load_module("harbor_grader", GRADER_PATH)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        point_grader_at(
            self.grader, root / "artifacts", root / "future_prices.json", root / "verifier"
        )
        self.grader.ARTIFACTS_DIR.mkdir()
        self.grader.TRANSFER_CHECK.write_text('{"version": 1, "removed": 0}\n', encoding="utf-8")
        self.grader.PRIVATE.write_text(
            json.dumps({"returns": [0.001] * 8}), encoding="utf-8"
        )

    def records(self):
        return [
            {"step": i, "action": "buy" if i < 7 else "close", "size": 1.0}
            for i in range(8)
        ]

    def write_records(self, records):
        self.grader.ARTIFACT.write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8"
        )

    def clear_outputs(self):
        self.grader.REWARD_PATH.unlink()
        self.grader.REWARD_JSON_PATH.unlink()

    def run_grader(self, refusal):
        self.assertEqual(self.grader.main(), 0)
        report = json.loads(self.grader.REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["refusal"], refusal)
        self.assertEqual(report["accepted"], refusal is None)
        reward = float(self.grader.REWARD_PATH.read_text(encoding="utf-8"))
        self.assertEqual(reward, report["reward"])
        if refusal is not None:
            self.assertEqual(reward, 0.0)
        return report

    def test_honest_submission_is_accepted(self):
        self.write_records(self.records())
        self.assertAlmostEqual(self.run_grader(None)["reward"], 0.85)

    def test_invalid_utf8_at_size_boundary_writes_refusal_and_reward(self):
        self.grader.ARTIFACT.write_bytes(b"\xff" * self.grader.MAX_ARTIFACT_BYTES)
        self.run_grader("artifact_unparseable")

    def test_oversized_artifact_is_not_decoded(self):
        self.grader.ARTIFACT.write_bytes(b"\xff" * (self.grader.MAX_ARTIFACT_BYTES + 1))
        self.run_grader("artifact_oversized")

    def test_unhashable_action_is_a_schema_refusal(self):
        records = self.records()
        records[0]["action"] = ["buy"]
        self.write_records(records)
        self.run_grader("illegal_action")

    def test_boolean_step_is_not_an_integer_index(self):
        records = self.records()
        records[0]["step"] = False
        self.write_records(records)
        self.run_grader("step_index_mismatch")

    def test_oversized_numeric_value_is_a_refusal(self):
        records = self.records()
        records[0]["size"] = 10**400
        self.write_records(records)
        self.run_grader("size_out_of_range")

    def test_artifact_read_error_is_reason_coded(self):
        self.write_records(self.records())
        original_open = Path.open

        def denied(path, *args, **kwargs):
            if path == self.grader.ARTIFACT:
                raise PermissionError("fixture: unreadable artifact")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", denied):
            self.run_grader("artifact_unreadable")

    def test_reward_json_carries_acceptance_for_the_job_metric(self):
        self.write_records(self.records())
        self.run_grader(None)
        reward = json.loads(self.grader.REWARD_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(reward["accepted"], 1)
        self.clear_outputs()
        self.grader.ARTIFACT.unlink()
        self.run_grader("artifact_missing")
        reward = json.loads(self.grader.REWARD_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(reward, {"accepted": 0, "reward": 0.0})

    def test_absent_transfer_check_is_refused(self):
        self.write_records(self.records())
        self.grader.TRANSFER_CHECK.unlink()
        self.run_grader("transfer_check_unavailable")

    def test_malformed_transfer_check_is_refused(self):
        self.write_records(self.records())
        for payload in (
            b"not json",
            b"\xff",
            b"[]",
            b'{"version": 1, "removed": true}',
            b'{"version": 1, "removed": -1}',
            b'{"removed": 0}',
        ):
            with self.subTest(payload=payload):
                self.grader.TRANSFER_CHECK.write_bytes(payload)
                self.run_grader("transfer_check_unavailable")
                self.clear_outputs()

    def test_entries_removed_before_transfer_are_refused_by_reason_code(self):
        self.write_records(self.records())
        self.grader.TRANSFER_CHECK.write_text('{"version": 1, "removed": 2}', encoding="utf-8")
        self.run_grader("artifact_tree_unsafe")

    @posix_only
    def test_link_beside_the_graded_file_is_refused(self):
        self.write_records(self.records())
        (self.grader.ARTIFACTS_DIR / "shadow.jsonl").symlink_to("/etc/shadow")
        self.run_grader("artifact_tree_unsafe")

    @posix_only
    def test_special_file_beside_the_graded_file_is_refused(self):
        self.write_records(self.records())
        os.mkfifo(self.grader.ARTIFACTS_DIR / "pipe")
        self.run_grader("artifact_tree_unsafe")

    @posix_only
    def test_symlinked_graded_file_keeps_its_own_reason_code(self):
        self.grader.ARTIFACT.symlink_to(self.grader.PRIVATE)
        self.run_grader("artifact_not_regular_file")


class HarborMetricTests(unittest.TestCase):
    def setUp(self):
        self.metric = load_module("harbor_metric", METRIC_PATH)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def trial(self, name, report=None, exception=None):
        trial = self.root / name
        (trial / "verifier").mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"exception_info": exception}), encoding="utf-8"
        )
        if report is not None:
            payload = {"grader": self.metric.GRADER, **report}
            (trial / "verifier/report.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_crash_refusal_and_scored_zero_are_three_different_results(self):
        self.trial("a", {"accepted": True, "refusal": None, "reward": 0.0})
        self.trial("b", {"accepted": True, "refusal": None, "reward": 0.8})
        self.trial("c", {"accepted": False, "refusal": "artifact_missing", "reward": 0.0})
        self.trial("d", exception={"exception_type": "RuntimeError"})
        self.assertEqual(
            self.metric.summarize(self.metric.job_outcomes(self.root)),
            {
                "trials": 4,
                "accepted": 2,
                "refused": 1,
                "crashed": 1,
                "mean_accepted_reward": 0.4,
                "refusal_codes": {"artifact_missing": 1},
            },
        )

    def test_report_from_a_trial_that_raised_is_not_trusted(self):
        self.trial(
            "a",
            {"accepted": True, "refusal": None, "reward": 1.0},
            exception={"exception_type": "RuntimeError"},
        )
        self.assertEqual(self.metric.classify_trial(self.root / "a"), self.metric.CRASH)

    def test_trial_with_no_report_is_a_crash(self):
        self.trial("a")
        self.assertEqual(self.metric.classify_trial(self.root / "a"), self.metric.CRASH)

    def test_unclassifiable_reports_are_crashes_not_scores(self):
        grader = self.metric.GRADER
        for report in (
            None,
            [],
            {"grader": "PLANTED-BY-AGENT", "accepted": True, "refusal": None, "reward": 1.0},
            {"grader": grader, "accepted": "yes", "refusal": None, "reward": 1.0},
            {"grader": grader, "accepted": True, "refusal": None, "reward": None},
            {"grader": grader, "accepted": True, "refusal": None, "reward": True},
            {"grader": grader, "accepted": True, "refusal": None, "reward": float("nan")},
            {"grader": grader, "accepted": True, "refusal": "x", "reward": 1.0},
            {"grader": grader, "accepted": False, "refusal": "x", "reward": 1.0},
            {"grader": grader, "accepted": False, "refusal": None, "reward": 0.0},
        ):
            with self.subTest(report=report):
                self.assertEqual(self.metric.classify_report(report), self.metric.CRASH)

    def test_no_accepted_trial_leaves_the_mean_unavailable(self):
        summary = self.metric.summarize([self.metric.CRASH])
        self.assertIsNone(summary["mean_accepted_reward"])
        self.assertEqual(summary["crashed"], 1)

    def test_harbor_contract_splits_null_refused_and_scored_zero(self):
        rewards = self.root / "rewards.jsonl"
        rewards.write_text(
            "null\n"
            '{"accepted": 0, "reward": 0.0}\n'
            '{"accepted": 1, "reward": 0.0}\n'
            '{"reward": 1.0}\n',
            encoding="utf-8",
        )
        output = self.root / "metric.json"
        self.assertEqual(self.metric.main(["-i", str(rewards), "-o", str(output)]), 0)
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")),
            {
                "trials": 4,
                "accepted": 1,
                "refused": 1,
                "crashed": 2,
                "mean_accepted_reward": 0.0,
            },
        )

    def test_a_job_with_no_trials_is_an_error_not_an_empty_success(self):
        with self.assertRaises(SystemExit) as caught:
            self.metric.main(["--job-dir", str(self.root)])
        self.assertIn("no trial directories", str(caught.exception))


class FixtureScriptTests(unittest.TestCase):
    """Run agent script, collect hook, then grader, with separate agent and verifier roots."""

    OUTCOMES = {
        "honest": None,
        "forged_reward": "artifact_missing",
        "writable_grader_path": None,
        "missing_final_action": "missing_final_action",
        "path_traversal": "artifact_tree_unsafe",
        "stale_outputs": None,
        "cross_task_state": None,
        "private_evaluator_read": None,
    }

    def setUp(self):
        self.cases = load_module("harbor_scripts", SCRIPTS_PATH).CASE_SCRIPTS
        task = tomllib.loads((TASK / "task.toml").read_text(encoding="utf-8"))
        self.hooks = task["verifier"]["collect"]

    @staticmethod
    def sh(script, root):
        rooted = re.sub(r"(?<![\w/.])/(tests|logs|app)\b", lambda m: f"{root}/{m[1]}", script)
        subprocess.run(["bash", "-c", rooted], check=True)

    def run_case(self, case):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        agent = Path(temp.name, "agent")
        (agent / "app").mkdir(parents=True)
        self.sh(self.cases[case], agent)
        for hook in self.hooks:
            self.sh(hook["command"], agent)
        grader = load_module(f"harbor_grader_{case}", GRADER_PATH)
        point_grader_at(
            grader,
            agent / "logs/artifacts",
            TASK / "tests/private/future_prices.json",
            Path(temp.name, "verifier"),
        )
        self.assertEqual(grader.main(), 0)
        return json.loads(grader.REPORT_PATH.read_text(encoding="utf-8")), agent

    def test_the_outcome_table_names_every_fixture(self):
        self.assertEqual(set(self.OUTCOMES), set(self.cases))

    @posix_only
    def test_every_fixture_reaches_its_named_grader_outcome(self):
        honest, _ = self.run_case("honest")
        for case, refusal in self.OUTCOMES.items():
            with self.subTest(case=case):
                report, _ = self.run_case(case)
                self.assertEqual(report["refusal"], refusal)
                if refusal is None:
                    self.assertEqual(report["reward"], honest["reward"])

    @posix_only
    def test_collect_hook_leaves_nothing_the_reupload_cannot_transfer(self):
        _, agent = self.run_case("path_traversal")
        artifacts = agent / "logs/artifacts"
        self.assertFalse(os.path.lexists(artifacts / "actions.jsonl"))
        self.assertFalse(os.path.lexists(artifacts / "shadow.jsonl"))
        marker = json.loads((artifacts / ".sa-transfer-check.json").read_text(encoding="utf-8"))
        self.assertEqual(marker, {"version": 1, "removed": 2})

    @posix_only
    def test_collect_hook_replaces_a_planted_marker(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        artifacts = Path(temp.name, "logs/artifacts")
        artifacts.mkdir(parents=True)
        (artifacts / ".sa-transfer-check.json").mkdir()
        (artifacts / "evil").symlink_to("/etc/shadow")
        for hook in self.hooks:
            self.sh(hook["command"], temp.name)
        marker = json.loads((artifacts / ".sa-transfer-check.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["removed"], 1)

    @posix_only
    def test_writable_grader_probe_measures_a_stock_container(self):
        _, agent = self.run_case("writable_grader_path")
        probe = json.loads((agent / "logs/artifacts/probe.json").read_text(encoding="utf-8"))
        self.assertIs(probe["tests_dir_exists"], False)
        self.assertIs(probe["private_readable"], False)
        self.assertEqual(probe["private_bytes"], "")
        # The attack must still run, after the measurement.
        self.assertTrue((agent / "tests/grade.py").is_file())


if __name__ == "__main__":
    unittest.main()
