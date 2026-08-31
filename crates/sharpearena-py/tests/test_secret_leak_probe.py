"""Unique-needle leak probe for the sealed-seed salt.

``SealedSalt`` promises the salt cannot escape into anything an entrant can read;
the Rust side pins the type-level guarantees (redacting Debug, no Serialize). This
probe checks the promise END TO END on the operator workflow instead: a per-run
unique needle is used as the salt, the commit-reveal artifacts the sealed-eval
workflow actually persists (the sha256 commitment, the scored regression snapshot,
the revealed seed mapping) are written out, and every artifact plus the in-memory
result reprs are scanned for the literal needle. Scope: the sealed-evaluation
workflow is the only flow a salt enters; the local-field and paper arms never see
one.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from sharpearena.eval_seeds import evaluate_eval_set, sealed_eval_seeds


def test_salt_needle_never_reaches_persisted_artifacts(tmp_path) -> None:
    needle = f"NEEDLE-{uuid.uuid4().hex}"
    salt = needle.encode("utf-8")  # 39 bytes, clears MIN_SEALED_SALT_BYTES

    # The operator workflow, end to end: commit, run, report, reveal.
    commitment = hashlib.sha256(salt).hexdigest()
    (tmp_path / "commitment.txt").write_text(commitment, encoding="utf-8")
    snapshot = evaluate_eval_set(
        n_symbols=2, n_days=40, max_steps=8, n_trials=8, salt=salt
    )
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    seeds = sealed_eval_seeds(salt)
    (tmp_path / "revealed-seeds.json").write_text(json.dumps(seeds), encoding="utf-8")

    scanned = 0
    for artifact in sorted(tmp_path.rglob("*")):
        if artifact.is_file():
            scanned += 1
            data = artifact.read_bytes()
            assert needle.encode("utf-8") not in data, f"salt leaked into {artifact.name}"
            assert salt.hex().encode() not in data, f"salt hex leaked into {artifact.name}"
    assert scanned == 3, "the probe must scan every artifact the workflow persisted"

    # The in-memory objects an operator might json.dumps or log verbatim.
    for rendered in (repr(snapshot), repr(seeds), json.dumps(snapshot)):
        assert needle not in rendered
