"""The cross-product tutorial is a deterministic executable contract fixture."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
TUTORIAL = REPOSITORY / "examples" / "forecast-quality"
FIELDS = (".", "withheld")
EVIDENCE = ("agent-alpha.json", "agent-beta.json")


def test_tutorial_regenerates_the_committed_evidence_byte_for_byte(tmp_path):
    subprocess.run(
        [sys.executable, str(TUTORIAL / "tutorial.py"), "--output-dir", str(tmp_path)],
        check=True,
    )

    for field in FIELDS:
        for name in (*EVIDENCE, "manifest.json"):
            generated = tmp_path / field / name
            committed = TUTORIAL / "fixtures" / field / name
            assert generated.read_bytes() == committed.read_bytes(), f"{field}/{name}"


@pytest.mark.parametrize("field", FIELDS)
def test_tutorial_manifest_binds_every_evidence_file(field):
    fixtures = TUTORIAL / "fixtures" / field
    manifest = json.loads((fixtures / "manifest.json").read_text("utf-8"))
    assert manifest["purpose"].endswith("not empirical agent evidence")
    assert set(manifest["files"]) == set(EVIDENCE)
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((fixtures / name).read_bytes()).hexdigest() == expected


def test_withheld_field_is_the_first_eight_questions_of_the_supported_field():
    for name in EVIDENCE:
        supported = json.loads((TUTORIAL / "fixtures" / name).read_text("utf-8"))
        withheld = json.loads((TUTORIAL / "fixtures" / "withheld" / name).read_text("utf-8"))
        assert len(supported["contracts"]) == 12
        assert len(withheld["contracts"]) == 8
        assert [c["contract_id"] for c in withheld["contracts"]] == [
            c["contract_id"] for c in supported["contracts"][:8]
        ]
        assert [r["outcome"] for r in withheld["resolutions"]] == [
            r["outcome"] for r in supported["resolutions"][:8]
        ]
        assert len({c["resolves_at"] for c in supported["contracts"]}) == 6
        assert len({c["resolves_at"] for c in withheld["contracts"]}) == 2
