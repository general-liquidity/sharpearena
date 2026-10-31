"""Exercise the Harbor grader, metric, collect hook, test.sh, task images and fixture scripts.

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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from unittest.mock import patch

HARBOR = Path(__file__).resolve().parents[1] / "integrations/harbor"
TASK = HARBOR / "tasks/sa-replay-feasibility"
GRADER_PATH = TASK / "tests/grade.py"
METRIC_PATH = HARBOR / "metric.py"
SCRIPTS_PATH = HARBOR / "fixtures/sa_harbor_fixtures/scripts.py"

posix_only = unittest.skipUnless(os.name == "posix", "needs POSIX shell and symlinks")

MARKER_CLEAN = '{"version": 2, "removed": 0, "oversized": 0}\n'
ROOTED = re.compile(r"(?<![\w/.])/(tests|logs|app)\b")


def rooted(text, root):
    """Move the container's /tests, /logs and /app under ``root``."""
    return ROOTED.sub(lambda m: f"{root}/{m[1]}", text)


def run_bounded(grader, seconds=20):
    """Run grader.main() and fail, rather than hang the suite, if it blocks."""
    outcome = {}

    def target():
        outcome["rc"] = grader.main()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise AssertionError(f"grader still running after {seconds} s: it blocked")
    return outcome["rc"]


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
        self.grader.TRANSFER_CHECK.write_text(MARKER_CLEAN, encoding="utf-8")
        self.grader.PRIVATE.write_text(
            json.dumps({"returns": [0.001] * 8}), encoding="utf-8"
        )

    def records(self):
        return [
            {"step": i, "action": "buy" if i < 7 else "close", "size": 1.0 if i < 7 else 0}
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
        self.assertEqual(run_bounded(self.grader), 0)
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
        original_open = os.open

        def denied(path, *args, **kwargs):
            if Path(path) == self.grader.ARTIFACT:
                raise PermissionError("fixture: unreadable artifact")
            return original_open(path, *args, **kwargs)

        with patch.object(self.grader.os, "open", denied):
            self.run_grader("artifact_unreadable")

    def test_reward_json_carries_acceptance_for_the_job_metric(self):
        self.write_records(self.records())
        self.run_grader(None)
        reward = json.loads(self.grader.REWARD_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual((reward["accepted"], reward["agent_fault"], reward["evaluator_fault"]), (1, 0, 0))
        self.clear_outputs()
        self.grader.ARTIFACT.unlink()
        self.assertEqual(self.run_grader("artifact_missing")["fault"], "agent")
        reward = json.loads(self.grader.REWARD_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            reward, {"accepted": 0, "agent_fault": 1, "evaluator_fault": 0, "reward": 0.0}
        )

    def test_final_close_with_a_nonzero_size_is_refused(self):
        """instruction.md requires the last line to be a close with size 0."""
        records = self.records()
        records[-1]["size"] = 0.5
        self.write_records(records)
        self.run_grader("final_close_size_nonzero")
        self.clear_outputs()
        records[-1]["size"] = 0.0
        self.write_records(records)
        self.run_grader(None)

    def test_unreadable_private_input_is_an_evaluator_fault(self):
        self.write_records(self.records())
        self.grader.PRIVATE.write_text("{not json", encoding="utf-8")
        report = self.run_grader("private_input_unreadable")
        self.assertEqual(report["fault"], "evaluator")
        reward = json.loads(self.grader.REWARD_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual((reward["accepted"], reward["agent_fault"], reward["evaluator_fault"]), (0, 0, 1))

    def test_files_removed_as_oversized_before_transfer_are_refused_by_their_own_code(self):
        self.write_records(self.records())
        self.grader.TRANSFER_CHECK.write_text(
            '{"version": 2, "removed": 0, "oversized": 1}', encoding="utf-8"
        )
        self.run_grader("artifact_transfer_oversized")

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
            b'{"version": 2, "removed": true, "oversized": 0}',
            b'{"version": 2, "removed": -1, "oversized": 0}',
            b'{"version": 2, "removed": 0}',
            b'{"version": 1, "removed": 0, "oversized": 0}',
            b'{"removed": 0, "oversized": 0}',
        ):
            with self.subTest(payload=payload):
                self.grader.TRANSFER_CHECK.write_bytes(payload)
                self.run_grader("transfer_check_unavailable")
                self.clear_outputs()

    def test_entries_removed_before_transfer_are_refused_by_reason_code(self):
        self.write_records(self.records())
        self.grader.TRANSFER_CHECK.write_text(
            '{"version": 2, "removed": 2, "oversized": 0}', encoding="utf-8"
        )
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

    @posix_only
    def test_fifo_graded_file_is_refused_without_blocking(self):
        """Karotte class (b), at the grader with no hook in front of it."""
        os.mkfifo(self.grader.ARTIFACT)
        self.run_grader("artifact_not_regular_file")

    @posix_only
    def test_graded_link_to_an_endless_device_is_refused(self):
        """Karotte class (c): a link to /dev/zero is never opened through."""
        self.grader.ARTIFACT.symlink_to("/dev/zero")
        self.run_grader("artifact_not_regular_file")


class HarborMetricTests(unittest.TestCase):
    def setUp(self):
        self.metric = load_module("harbor_metric", METRIC_PATH)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def trial(self, name, report=None, exception=None, agent="sa-honest", config=None):
        trial = self.root / name
        (trial / "verifier").mkdir(parents=True)
        result = {
            "exception_info": exception,
            "task_checksum": "c0",
            "agent_info": {"name": agent, "version": "1.0.0", "model_info": None},
            "config": {"trial_name": name, "job_id": "j", "agent": {"import_path": agent}, **(config or {})},
        }
        (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
        if report is not None:
            payload = {"grader": self.metric.GRADER, **report}
            (trial / "verifier/report.json").write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def accepted(reward):
        return {"accepted": True, "refusal": None, "fault": None, "reward": reward}

    @staticmethod
    def refused(code, fault):
        return {"accepted": False, "refusal": code, "fault": fault, "reward": 0.0}

    def test_agent_faults_score_zero_and_evaluator_faults_leave_the_denominator(self):
        """Regression (2026-10-31 bench2 review): the headline averaged accepted
        trials only, so a refused submission did not lower it."""
        self.trial("a", self.accepted(0.8))
        self.trial("b", self.accepted(0.4))
        self.trial("c", self.refused("artifact_missing", "agent"))
        self.trial("d", self.refused("private_input_missing", "evaluator"))
        out = self.metric.summarize_job(self.root)
        group = out["pooled"]
        self.assertEqual(
            {k: group[k] for k in ("trials", "accepted", "agent_fault", "evaluator_fault", "crashed", "scored_trials")},
            {"trials": 4, "accepted": 2, "agent_fault": 1, "evaluator_fault": 1, "crashed": 0, "scored_trials": 3},
        )
        self.assertAlmostEqual(group["mean_reward_all_trials"], 1.2 / 3)
        self.assertAlmostEqual(group["mean_reward_accepted_only"], 0.6)
        self.assertEqual(group["agent_fault_codes"], {"artifact_missing": 1})
        self.assertEqual(group["evaluator_fault_codes"], {"private_input_missing": 1})

    def test_a_crash_makes_the_headline_unavailable_not_a_score(self):
        self.trial("a", self.accepted(0.8))
        self.trial("b", exception={"exception_type": "RuntimeError"})
        group = self.metric.summarize_job(self.root)["pooled"]
        self.assertIsNone(group["mean_reward_all_trials"])
        self.assertIn("1 crashed", group["mean_reward_all_trials_unavailable"])
        self.assertAlmostEqual(group["mean_reward_crashes_as_zero"], 0.4)

    def test_trials_from_different_agents_or_configs_are_not_pooled(self):
        self.trial("a", self.accepted(0.8), agent="sa-honest")
        self.trial("b", self.accepted(0.2), agent="sa-forged-reward")
        self.trial("c", self.accepted(0.2), agent="sa-honest", config={"timeout_multiplier": 2.0})
        out = self.metric.summarize_job(self.root)
        self.assertIsNone(out["pooled"])
        self.assertIn("3 agent or job-configuration groups", out["pooled_refused"])
        self.assertEqual(len(out["groups"]), 3)
        # Trials that differ only in per-trial fields stay one group.
        same = self.root / "same"
        same.mkdir()
        self.root, saved = same, self.root
        self.trial("x", self.accepted(0.8))
        self.trial("y", self.accepted(0.4))
        self.root = saved
        pooled = self.metric.summarize_job(same)["pooled"]
        self.assertEqual(pooled["trials"], 2)

    def test_a_trial_without_a_readable_result_blocks_pooling(self):
        self.trial("a", self.accepted(0.8))
        (self.root / "b").mkdir()
        out = self.metric.summarize_job(self.root)
        self.assertIsNone(out["pooled"])
        self.assertEqual(out["unattributed_trials"], ["b"])

    def test_report_from_a_trial_that_raised_is_not_trusted(self):
        self.trial("a", self.accepted(1.0), exception={"exception_type": "RuntimeError"})
        self.assertEqual(self.metric.classify_trial(self.root / "a"), self.metric.CRASH)

    def test_trial_with_no_report_is_a_crash(self):
        self.trial("a")
        self.assertEqual(self.metric.classify_trial(self.root / "a"), self.metric.CRASH)

    def test_unclassifiable_reports_are_crashes_not_scores(self):
        grader = self.metric.GRADER
        for report in (
            None,
            [],
            {"grader": "PLANTED-BY-AGENT", "accepted": True, "refusal": None, "fault": None, "reward": 1.0},
            {"grader": grader, "accepted": "yes", "refusal": None, "fault": None, "reward": 1.0},
            {"grader": grader, "accepted": True, "refusal": None, "fault": None, "reward": None},
            {"grader": grader, "accepted": True, "refusal": None, "fault": None, "reward": True},
            {"grader": grader, "accepted": True, "refusal": None, "fault": None, "reward": float("nan")},
            {"grader": grader, "accepted": True, "refusal": "x", "fault": None, "reward": 1.0},
            {"grader": grader, "accepted": True, "refusal": None, "fault": "agent", "reward": 1.0},
            {"grader": grader, "accepted": False, "refusal": "x", "fault": "agent", "reward": 1.0},
            {"grader": grader, "accepted": False, "refusal": None, "fault": "agent", "reward": 0.0},
            {"grader": grader, "accepted": False, "refusal": "x", "fault": None, "reward": 0.0},
            {"grader": grader, "accepted": False, "refusal": "x", "fault": "nobody", "reward": 0.0},
        ):
            with self.subTest(report=report):
                self.assertEqual(self.metric.classify_report(report), self.metric.CRASH)

    def test_no_scored_trial_leaves_the_means_unavailable(self):
        summary = self.metric.summarize([("evaluator_fault", None, "private_input_missing")])
        self.assertIsNone(summary["mean_reward_all_trials"])
        self.assertIsNone(summary["mean_reward_accepted_only"])
        self.assertEqual(summary["evaluator_fault"], 1)

    def test_harbor_contract_splits_null_faults_and_scored_zero(self):
        rewards = self.root / "rewards.jsonl"
        rewards.write_text(
            "null\n"
            '{"accepted": 0, "agent_fault": 1, "evaluator_fault": 0, "reward": 0.0}\n'
            '{"accepted": 0, "agent_fault": 0, "evaluator_fault": 1, "reward": 0.0}\n'
            '{"accepted": 1, "agent_fault": 0, "evaluator_fault": 0, "reward": 0.5}\n'
            '{"accepted": 0, "reward": 0.0}\n'
            '{"accepted": 1, "agent_fault": 1, "evaluator_fault": 0, "reward": 0.5}\n',
            encoding="utf-8",
        )
        output = self.root / "metric.json"
        self.assertEqual(self.metric.main(["-i", str(rewards), "-o", str(output)]), 0)
        out = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            {k: out[k] for k in ("trials", "accepted", "agent_fault", "evaluator_fault", "crashed")},
            {"trials": 6, "accepted": 1, "agent_fault": 1, "evaluator_fault": 1, "crashed": 3},
        )
        self.assertIsNone(out["mean_reward_all_trials"])
        self.assertAlmostEqual(out["mean_reward_crashes_as_zero"], 0.1)
        self.assertIn("not visible", out["grouping"])

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
        "fifo_artifact": "artifact_tree_unsafe",
        "sparse_artifact": "artifact_transfer_oversized",
        "lingering_swap": "artifact_not_regular_file",
        "planted_interpreter": None,
    }

    def setUp(self):
        self.cases = load_module("harbor_scripts", SCRIPTS_PATH).CASE_SCRIPTS
        task = tomllib.loads((TASK / "task.toml").read_text(encoding="utf-8"))
        self.hooks = task["verifier"]["collect"]

    @staticmethod
    def sh(script, root):
        subprocess.run(["bash", "-c", rooted(script, root)], check=True)

    @staticmethod
    def wait_for_lingering(agent):
        """A fixture that leaves a process running marks /app/.sa-pending until it acts."""
        pending = agent / "app/.sa-pending"
        deadline = time.monotonic() + 20
        while pending.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if pending.exists():
            raise AssertionError("the lingering fixture process never acted")

    def run_case(self, case):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        agent = Path(temp.name, "agent")
        (agent / "app").mkdir(parents=True)
        self.sh(self.cases[case], agent)
        for hook in self.hooks:
            self.sh(hook["command"], agent)
        self.wait_for_lingering(agent)
        grader = load_module(f"harbor_grader_{case}", GRADER_PATH)
        point_grader_at(
            grader,
            agent / "logs/artifacts",
            TASK / "tests/private/future_prices.json",
            Path(temp.name, "verifier"),
        )
        self.assertEqual(run_bounded(grader), 0)
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
        self.assertEqual(marker, {"version": 2, "removed": 2, "oversized": 0})

    @posix_only
    def test_collect_hook_removes_a_sparse_file_before_transfer(self):
        """Karotte classes (c) and (g): the 8 GiB file never reaches the transfer."""
        _, agent = self.run_case("sparse_artifact")
        artifacts = agent / "logs/artifacts"
        self.assertFalse(os.path.lexists(artifacts / "actions.jsonl"))
        marker = json.loads((artifacts / ".sa-transfer-check.json").read_text(encoding="utf-8"))
        self.assertEqual(marker, {"version": 2, "removed": 0, "oversized": 1})

    @posix_only
    def test_planted_interpreter_and_sitecustomize_never_run_under_test_sh(self):
        """Karotte class (e). The verifier's own test.sh runs with the artifact
        directory first on PATH and on PYTHONPATH and as the working directory,
        which is more than a real verifier hands it. The planted python3 and
        sitecustomize.py must not run, and the real grader must score the honest
        submission."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "app").mkdir()
        self.sh(self.cases["planted_interpreter"], root)
        for hook in self.hooks:
            self.sh(hook["command"], root)
        tests = root / "tests"
        (tests / "private").mkdir(parents=True)
        shutil.copy(TASK / "tests/private/future_prices.json", tests / "private")
        (tests / "grade.py").write_text(
            rooted(GRADER_PATH.read_text(encoding="utf-8"), root), encoding="utf-8"
        )
        script = rooted((TASK / "tests/test.sh").read_text(encoding="utf-8"), root)
        script = script.replace("/usr/local/bin/python3", sys.executable)
        (tests / "test.sh").write_text(script, encoding="utf-8")
        artifacts = root / "logs/artifacts"
        hostile = {
            **os.environ,
            "PATH": f"{artifacts}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "PYTHONPATH": str(artifacts),
        }
        subprocess.run(["bash", str(tests / "test.sh")], cwd=artifacts, env=hostile, check=True)
        self.assertFalse((root / "logs/PWNED").exists(), (root / "logs/PWNED").read_text() if (root / "logs/PWNED").exists() else "")
        report = json.loads((root / "logs/verifier/report.json").read_text(encoding="utf-8"))
        self.assertEqual((report["accepted"], report["refusal"]), (True, None))
        reward = float((root / "logs/verifier/reward.txt").read_text(encoding="utf-8"))
        self.assertEqual(reward, report["reward"])
        self.assertLess(reward, 1.0)

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


class TaskImageTests(unittest.TestCase):
    def test_every_task_dockerfile_pins_its_base_by_digest(self):
        """An unpinned FROM lets the base image change under a fixed task."""
        dockerfiles = sorted(HARBOR.glob("tasks/*/environment/Dockerfile")) + sorted(
            HARBOR.glob("tasks/*/tests/Dockerfile")
        )
        self.assertTrue(dockerfiles)
        pinned = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}(\s+AS\s+\S+)?\s*$", re.IGNORECASE)
        for dockerfile in dockerfiles:
            froms = [
                line.strip()
                for line in dockerfile.read_text(encoding="utf-8").splitlines()
                if line.strip().upper().startswith("FROM")
            ]
            with self.subTest(dockerfile=str(dockerfile.relative_to(HARBOR))):
                self.assertTrue(froms)
                for line in froms:
                    self.assertRegex(line, pinned)


if __name__ == "__main__":
    unittest.main()
