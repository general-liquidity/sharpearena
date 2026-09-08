"""The Python leg of the cross-language conformance kit.

The kit itself lives at ``crates/sharpearena/contract/`` and is indexed by
``conformance-kit.v1.json``. Rust checks it in ``crates/sharpearena/tests/``, npm in
``npm/sharpearena/test/conformance.test.js``, and this file checks it from Python. All
three read the same versioned files by repository path, so none of the three packages
gains a dependency on either of the others and the fixture set cannot drift into being
checked by only one runtime.

The badge legs asserted here are the schema legs GOVERNANCE.md names: every fixture
observation validates against ``observation.schema.json``, every recorded legacy decision
validates against ``decision.schema.json``, and the legacy shape carries none of the
additive optional fields. The behavioural replay through a live agent stays in Rust.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/sharpearena-py/tests/test_wire_conformance.py -q
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "crates" / "sharpearena" / "contract"
KIT = json.loads((CONTRACT / "conformance-kit.v1.json").read_text(encoding="utf-8"))

# Fields the additive-only discipline says an older agent may omit entirely.
ADDITIVE_ORDER_FIELDS = ("confidence", "rationale")


def _schema(kind: str) -> dict:
    return json.loads((CONTRACT / KIT["schemas"][kind]).read_text(encoding="utf-8"))


def _fixture(name: str) -> dict:
    return json.loads((CONTRACT / "conformance" / name).read_text(encoding="utf-8"))


def test_kit_names_exactly_the_fixtures_on_disk():
    on_disk = sorted(p.name for p in (CONTRACT / "conformance").glob("*.json"))
    assert sorted(KIT["fixtures"]) == on_disk
    assert KIT["fixtures"], "the kit must list at least one fixture"


def test_kit_version_matches_the_schema_major():
    major = KIT["contract_version"].split(".")[0]
    for kind in ("observation", "decision"):
        assert f"/contract/v{major}/" in _schema(kind)["$id"]
    assert isinstance(KIT["kit_version"], int)


@pytest.mark.parametrize("name", KIT["fixtures"])
def test_observation_validates_against_the_published_schema(name):
    Draft202012Validator(_schema("observation")).validate(_fixture(name)["observation"])


@pytest.mark.parametrize("name", KIT["legacy_decision_fixtures"])
def test_recorded_decision_validates_against_the_published_schema(name):
    Draft202012Validator(_schema("decision")).validate(_fixture(name)["legacy_decision"])


def test_the_kit_exercises_every_additive_optional_field_as_omitted():
    # The additive-only rule is only proved by a fixture that actually leaves the field
    # out. Asserted over the kit as a whole rather than per fixture, because a fixture is
    # also allowed to carry the fields: the point is that the set covers both.
    decisions = [_fixture(name)["legacy_decision"] for name in KIT["legacy_decision_fixtures"]]
    orders = [order for decision in decisions for order in decision["orders"]]
    assert any("reasoning" not in decision for decision in decisions), (
        "no fixture omits `reasoning`, so its serde default is never exercised"
    )
    for field in ADDITIVE_ORDER_FIELDS:
        assert any(field not in order for order in orders), (
            f"no fixture order omits `{field}`, so its serde default is never exercised"
        )


def test_the_installed_binding_ships_this_contract_version_of_the_decision_schema():
    # The extension embeds the schema with include_str! from the same contract directory.
    # Comparing the installed copy against the file the fixtures were validated with is
    # what makes this a check on the shipped Python surface rather than on the repository
    # alone, and it needs no dependency on the Rust or npm packages.
    from sharpearena.sharpearena_py import decision_schema_json

    assert json.loads(decision_schema_json()) == _schema("decision")
