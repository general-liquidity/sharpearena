//! The conformance kit is checked by three language surfaces (Rust here, Python in
//! `crates/sharpearena-py/tests/test_wire_conformance.py`, npm in
//! `npm/sharpearena/test/conformance.test.js`). Each of them discovers the fixture set
//! from `contract/conformance-kit.v1.json` rather than from its own copy of the list, so
//! a fixture added for one runtime cannot be silently skipped by the other two.
//!
//! This file checks the index itself: that it names exactly the fixtures on disk, that the
//! schemas it points at exist, and that the wire version it certifies is the one the crate
//! actually compiles with. The behavioural replay stays in `conformance.rs`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use sharpearena::CONTRACT_VERSION;

fn contract_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("contract")
}

fn manifest() -> serde_json::Value {
    let path = contract_dir().join("conformance-kit.v1.json");
    let text = fs::read_to_string(&path).expect("conformance kit manifest is readable");
    serde_json::from_str(&text).expect("conformance kit manifest is valid JSON")
}

#[test]
fn kit_certifies_the_compiled_contract_version() {
    let kit = manifest();
    assert_eq!(
        kit["contract_version"].as_str(),
        Some(CONTRACT_VERSION),
        "the kit certifies a different wire version than the crate compiles with",
    );
    assert!(
        kit["kit_version"].as_u64().is_some(),
        "kit_version must be an integer that moves when the fixture set changes",
    );
}

#[test]
fn kit_names_exactly_the_fixtures_on_disk() {
    let kit = manifest();
    let listed: BTreeSet<String> = kit["fixtures"]
        .as_array()
        .expect("fixtures is an array")
        .iter()
        .map(|v| v.as_str().expect("fixture name is a string").to_owned())
        .collect();
    assert!(!listed.is_empty(), "the kit must list at least one fixture");

    let dir = contract_dir().join("conformance");
    let mut on_disk = BTreeSet::new();
    for entry in fs::read_dir(&dir).expect("conformance directory exists") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            on_disk.insert(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert_eq!(
        listed, on_disk,
        "conformance-kit.v1.json and contract/conformance/ disagree; every surface reads the manifest, so an unlisted fixture is checked nowhere",
    );
}

#[test]
fn every_legacy_fixture_named_by_the_kit_carries_a_legacy_decision() {
    let kit = manifest();
    let dir = contract_dir().join("conformance");
    let named = kit["legacy_decision_fixtures"]
        .as_array()
        .expect("legacy_decision_fixtures is an array");
    assert!(
        !named.is_empty(),
        "the additive-only discipline needs at least one legacy-shaped decision",
    );
    for name in named {
        let name = name.as_str().expect("fixture name is a string");
        let text = fs::read_to_string(dir.join(name)).expect("named legacy fixture exists");
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert!(
            value.get("legacy_decision").is_some(),
            "{name} is listed as a legacy fixture but carries no legacy_decision",
        );
    }
}

#[test]
fn kit_points_at_schemas_that_exist_and_declare_the_same_major() {
    let kit = manifest();
    let major = CONTRACT_VERSION
        .split('.')
        .next()
        .expect("contract version has a major");
    for key in ["observation", "decision"] {
        let file = kit["schemas"][key]
            .as_str()
            .unwrap_or_else(|| panic!("kit names a {key} schema"));
        let text = fs::read_to_string(contract_dir().join(file)).expect("named schema file exists");
        let schema: serde_json::Value = serde_json::from_str(&text).expect("schema is valid JSON");
        let id = schema["$id"].as_str().expect("schema declares an $id");
        assert!(
            id.contains(&format!("/contract/v{major}/")),
            "{file} $id {id} does not carry the v{major} contract major",
        );
    }
}
