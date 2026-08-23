"""The armed eval-seeds regression gate.

The committed snapshot at ``tests/data/eval_seeds_reference.json`` pins the scored
outcome of the frozen reference policy on every named held-out seed, per tier, under
the sharpebench 0.5.0 kernel. Any generator, env, or kernel change that moves a
pinned number fails here, which is the CI gate ``eval_seeds`` promised but never
armed. Regenerate the snapshot deliberately (scratch script mirrored in the file's
``config`` block) when a change is meant to move the numbers, and say why in the
commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from sharpearena.eval_seeds import (
        EVAL_SET_VERSION,
        SCHEMA_VERSION,
        assert_no_regression,
        evaluate_eval_set,
    )

    _HAVE_BINDING = True
except ImportError:  # pragma: no cover - native module not built
    _HAVE_BINDING = False

requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="sharpearena_py native module not built"
)

REFERENCE_PATH = Path(__file__).parent / "data" / "eval_seeds_reference.json"


def _load_reference() -> dict:
    with REFERENCE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def test_reference_snapshot_is_committed_and_versioned():
    ref = _load_reference()
    assert ref["tiers"].keys() == {"calm", "hard", "extreme"}
    for tier in ref["tiers"].values():
        assert len(tier) == 8, "the frozen set is eight named seeds"


@requires_binding
def test_snapshot_matches_module_versions():
    ref = _load_reference()
    assert ref["eval_set_version"] == EVAL_SET_VERSION
    assert ref["schema_version"] == SCHEMA_VERSION


@requires_binding
@pytest.mark.parametrize("tier", ["calm", "hard", "extreme"])
def test_no_regression_against_committed_snapshot(tier):
    ref = _load_reference()
    current = evaluate_eval_set(
        distribution_mode=tier, n_trials=int(ref["config"]["n_trials"])
    )
    assert_no_regression(ref["tiers"][tier], current)
