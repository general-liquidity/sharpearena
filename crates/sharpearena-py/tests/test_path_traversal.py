"""Path-traversal fixtures for the plan-embedded ``csv_path`` consumers.

The two places SharpeArena resolves a path found *inside* a handed-in document
against an implied root are ``local_field_cli.load_plan`` and
``strategy_cli.load_strategy_plan``: both resolve a dataset ``csv_path`` against
the plan file's directory. A plan is a document that travels (shared, downloaded,
committed), so its embedded paths are not operator keyboard input; without
containment a hostile plan reads an arbitrary file into ``csv_text`` and from
there into the evidence artifact. Three fixture classes per consumer: a ``..``
component, an absolute path where a relative one is expected, and a symlink
inside the plan directory pointing outside it.

Classes that do NOT apply elsewhere, checked and stated rather than fixtured:

* Whole-path CLI arguments (``--plan`` / ``--evidence`` / ``--output-dir`` on the
  field, strategy, paper and bench-bridge CLIs) are operator-typed complete paths
  with no expected root, so "``..``" and "absolute where relative expected" are
  meaningless for them by construction.
* Minari dataset ids never become filesystem paths in this repo; ``to_minari``
  hands the id to Minari's own library, which validates its id grammar and owns
  its storage layout.
* Silver/gold trace-promotion stores take operator-supplied whole paths, and
  candidate ids are sha256 digests that are never used as file names.
* The MCP server exposes no tool that takes a filesystem path.
* The Rust/wasm surfaces take CSV text and JSON blobs, never paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sharpearena.local_field_cli import load_plan
from sharpearena.strategy_cli import load_strategy_plan


def _field_plan(tmp_path: Path, csv_path: str) -> Path:
    plan_path = tmp_path / "field.json"
    plan_path.write_text(
        json.dumps(
            {
                "models": [{"model": "test-fixture:synthetic"}],
                "datasets": [{"dataset_id": "fixture", "csv_path": csv_path}],
                "seeds": [7],
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def _strategy_plan(tmp_path: Path, csv_path: str) -> Path:
    plan_path = tmp_path / "strategy.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": {"model": "test-fixture:synthetic"},
                "prompt": "Generate one strategy.",
                "requested_candidates": 1,
                "validation_dataset": {
                    "dataset_id": "validation",
                    "csv_path": csv_path,
                    "window_start": 0,
                    "window_end": 1,
                },
                "test_dataset": {
                    "dataset_id": "test",
                    "csv_path": csv_path,
                    "window_start": 1,
                    "window_end": 2,
                },
                "validation_seeds": [1],
                "test_seeds": [2],
            }
        ),
        encoding="utf-8",
    )
    return plan_path


_CSV = "date,symbol,close\n2026-01-01,AAA,100\n2026-01-02,AAA,101\n"

_LOADERS = [("field", _field_plan, load_plan), ("strategy", _strategy_plan, load_strategy_plan)]
_IDS = [name for name, _, _ in _LOADERS]


@pytest.fixture()
def outside_secret(tmp_path: Path) -> Path:
    # A file OUTSIDE the plan directory that a traversal would exfiltrate.
    secret = tmp_path / "outside" / "secret.csv"
    secret.parent.mkdir()
    secret.write_text(_CSV, encoding="utf-8")
    return secret


@pytest.mark.parametrize("name,mk_plan,loader", _LOADERS, ids=_IDS)
def test_dotdot_component_is_refused(tmp_path, outside_secret, name, mk_plan, loader):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    plan = mk_plan(plan_dir, "../outside/secret.csv")
    with pytest.raises(ValueError, match="escapes the plan directory"):
        loader(plan)


@pytest.mark.parametrize("name,mk_plan,loader", _LOADERS, ids=_IDS)
def test_absolute_path_is_refused(tmp_path, outside_secret, name, mk_plan, loader):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    plan = mk_plan(plan_dir, str(outside_secret))
    with pytest.raises(ValueError, match="must be relative to the plan directory"):
        loader(plan)


@pytest.mark.parametrize("name,mk_plan,loader", _LOADERS, ids=_IDS)
def test_symlink_escaping_the_plan_dir_is_refused(
    tmp_path, outside_secret, name, mk_plan, loader
):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    link = plan_dir / "prices.csv"
    try:
        os.symlink(outside_secret, link)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"cannot create symlinks here: {exc}")
    plan = mk_plan(plan_dir, "prices.csv")
    with pytest.raises(ValueError, match="escapes the plan directory"):
        loader(plan)


@pytest.mark.parametrize("name,mk_plan,loader", _LOADERS, ids=_IDS)
def test_contained_relative_path_still_loads(tmp_path, name, mk_plan, loader):
    # The negative control: containment must not break the legitimate shape,
    # including a subdirectory reference.
    plan_dir = tmp_path / "plans"
    (plan_dir / "data").mkdir(parents=True)
    (plan_dir / "data" / "prices.csv").write_text(_CSV, encoding="utf-8")
    plan = mk_plan(plan_dir, "data/prices.csv")
    loaded = loader(plan)
    csv_text = (
        loaded.datasets[0].csv_text if name == "field" else loaded.validation_dataset.csv_text
    )
    assert csv_text == _CSV
