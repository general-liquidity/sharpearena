"""The cross-product tutorial is a deterministic executable contract fixture."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
TUTORIAL = REPOSITORY / "examples" / "forecast-quality"


def test_tutorial_regenerates_the_committed_evidence_byte_for_byte(tmp_path):
    subprocess.run(
        [sys.executable, str(TUTORIAL / "tutorial.py"), "--output-dir", str(tmp_path)],
        check=True,
    )

    for name in ("agent-alpha.json", "agent-beta.json", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (TUTORIAL / "fixtures" / name).read_bytes()


def test_tutorial_manifest_binds_every_evidence_file():
    manifest = json.loads((TUTORIAL / "fixtures" / "manifest.json").read_text("utf-8"))
    assert manifest["purpose"].endswith("not empirical agent evidence")
    assert set(manifest["files"]) == {"agent-alpha.json", "agent-beta.json"}
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((TUTORIAL / "fixtures" / name).read_bytes()).hexdigest() == expected
