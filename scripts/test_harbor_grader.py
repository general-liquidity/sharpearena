"""Exercise the actual Harbor grader without Docker or the Harbor SDK."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

GRADER_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations/harbor/tasks/sa-replay-feasibility/tests/grade.py"
)


class HarborGraderTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("harbor_grader", GRADER_PATH)
        self.grader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.grader)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.grader.ARTIFACT = root / "actions.jsonl"
        self.grader.PRIVATE = root / "future_prices.json"
        self.grader.VERIFIER_DIR = root / "verifier"
        self.grader.REWARD_PATH = self.grader.VERIFIER_DIR / "reward.txt"
        self.grader.REPORT_PATH = self.grader.VERIFIER_DIR / "report.json"
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


if __name__ == "__main__":
    unittest.main()
