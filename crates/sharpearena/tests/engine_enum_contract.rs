//! Emit the engine-output vocabulary the npm package restates, so the restatement can be
//! checked instead of trusted.
//!
//! `npm/sharpearena/src/types.ts` types the engine's JSON shapes for TypeScript consumers.
//! The two *wire contract* types are cross-checked against the published JSON Schemas by
//! `npm/sharpearena/test/conformance.test.js`, but the engine-output enums have no schema to
//! check against: they restate Rust declarations, and TypeScript types are erased before any
//! validator runs. That is the part of ARENA-REVIEW A15 left open, which said closing it
//! "would need a generated contract file rather than a test". This is that file's producer.
//!
//! Two guarantees, and only two:
//!
//! * **Variant drift fails to compile.** Each vocabulary is built through a match with no
//!   wildcard arm, so a variant added to `DistributionMode`, `Regime` or `BaselineAgent`
//!   stops this test compiling rather than going unnamed. Same device as the process-event
//!   contract. For the latter two the match lives in `sharpearena::vocabulary`, so the
//!   compile failure lands on the crate rather than on this file.
//! * **Restatement drift fails a test.** The committed artifact is what the npm suite reads,
//!   so a `types.ts` union or interface that disagrees with the engine fails there.
//!
//! What it does *not* establish: `DistributionMode`, `ObservationRichness` and `ScenarioSpec`
//! carry their wire form in serde, so their labels and field names here are the engine's own.
//! `Regime` and `BaselineAgent` cannot — the first is a foreign type this crate cannot derive
//! `Serialize` for, the second exists to name a dispatch rather than to be serialized — so
//! their labels come from `sharpearena::vocabulary`, which is also what the wasm export layer
//! reads. That shared authoring is the whole of the guarantee for those two: the labels below
//! are not a restatement of what the engine emits, they *are* what the engine emits, so a
//! renamed label moves this artifact and the committed `tag_regime` / `run_baseline` goldens
//! together and cannot be regenerated away. See the A15 disposition.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde_json::{json, Map, Value};
use sharpearena::{
    regime_label, BaselineAgent, DistributionMode, ObservationRichness, ScenarioSpec, REGIMES,
};

const CONTRACT: &str = "contract/engine-enums.v1.json";

/// Set `UPDATE_ENGINE_ENUM_CONTRACT=1` to rewrite the committed artifact after an
/// intentional change. Unset, the test compares and fails.
const UPDATE_ENV: &str = "UPDATE_ENGINE_ENUM_CONTRACT";

fn contract_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CONTRACT)
}

/// Every `DistributionMode`, under the label serde writes, in declaration order.
///
/// The match is exhaustive with no wildcard arm: a tier added to the enum stops this from
/// compiling, which is the whole point of generating the list instead of typing it.
fn distribution_modes() -> Vec<String> {
    [
        DistributionMode::Calm,
        DistributionMode::Hard,
        DistributionMode::Extreme,
        DistributionMode::CointegratedPairs,
        DistributionMode::RegimeShift,
    ]
    .into_iter()
    .map(|mode| match mode {
        carried @ (DistributionMode::Calm
        | DistributionMode::Hard
        | DistributionMode::Extreme
        | DistributionMode::CointegratedPairs
        | DistributionMode::RegimeShift) => serde_json::to_value(carried)
            .expect("a distribution mode serializes")
            .as_str()
            .expect("its wire form is a string")
            .to_string(),
    })
    .collect()
}

/// Every `Regime`, under the label the JSON surface writes.
///
/// `Regime` derives no `Serialize` and is a foreign type, so its labels cannot be read off
/// the type the way the modes above are. They come from `sharpearena::regime_label`, which
/// is the same function the wasm export layer's `tag_regime` emits through — so these are
/// the engine's own bytes rather than a second authoring of them. `regime_label`'s match is
/// wildcard-free, so an added variant is still a compile error.
fn regimes() -> Vec<String> {
    REGIMES
        .into_iter()
        .map(|regime| regime_label(regime).to_string())
        .collect()
}

/// Every `BaselineAgent`, under the label `run_baseline` dispatches on.
///
/// Same shape as `regimes` and for a related reason: the enum names a dispatch rather than
/// a serialized value, so `BaselineAgent::label` is the authoring and the wasm export
/// layer's `build_agent` resolves through it.
fn baseline_agents() -> Vec<String> {
    BaselineAgent::ALL
        .into_iter()
        .map(|agent| agent.label().to_string())
        .collect()
}

/// The serde field names of a struct, taken from a serialized value rather than restated.
fn serialized_fields<T: serde::Serialize>(value: &T) -> Vec<String> {
    let Value::Object(map) = serde_json::to_value(value).expect("the struct serializes") else {
        panic!("expected a JSON object");
    };
    map.keys().cloned().collect()
}

fn build_contract() -> Value {
    let mut enums = Map::new();
    enums.insert("BaselineAgent".into(), json!(baseline_agents()));
    enums.insert("DistributionMode".into(), json!(distribution_modes()));
    enums.insert("Regime".into(), json!(regimes()));

    let mut structs = BTreeMap::new();
    structs.insert(
        "ObservationRichness",
        serialized_fields(&ObservationRichness::default()),
    );
    structs.insert("ScenarioSpec", serialized_fields(&ScenarioSpec::default()));

    json!({
        "schema_version": 1,
        "note": "Generated by crates/sharpearena/tests/engine_enum_contract.rs. The engine-output \
    vocabulary that npm/sharpearena/src/types.ts restates as TypeScript unions and interfaces, \
    which no published JSON Schema covers (ARENA-REVIEW A15). Regenerate with \
    UPDATE_ENGINE_ENUM_CONTRACT=1; do not hand-edit.",
        "enums": Value::Object(enums),
        "structs": structs,
    })
}

fn rendered() -> String {
    serde_json::to_string_pretty(&build_contract()).expect("the contract renders") + "\n"
}

#[test]
fn the_committed_engine_enum_contract_is_what_this_tree_emits() {
    let path = contract_path();
    let expected = rendered();
    if std::env::var(UPDATE_ENV).is_ok() {
        fs::write(&path, &expected).expect("the contract artifact is writable");
        return;
    }
    let actual = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must exist and be readable: {e}", path.display()))
        .replace("\r\n", "\n");
    assert_eq!(
        actual, expected,
        "{CONTRACT} has fallen behind the engine; rerun with {UPDATE_ENV}=1 and check what \
         moved, because npm/sharpearena/src/types.ts is checked against this file"
    );
}

#[test]
fn the_contract_is_not_vacuous() {
    // A generator that emitted nothing would make every consumer's cross-check pass for free.
    let contract = build_contract();
    assert_eq!(
        contract["enums"]["DistributionMode"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert_eq!(contract["enums"]["Regime"].as_array().unwrap().len(), 3);
    assert_eq!(
        contract["enums"]["BaselineAgent"].as_array().unwrap().len(),
        4
    );
    assert!(contract["structs"]["ScenarioSpec"]
        .as_array()
        .unwrap()
        .contains(&json!("start_level")));
    assert!(contract["structs"]["ObservationRichness"]
        .as_array()
        .unwrap()
        .contains(&json!("lookback")));
}

/// A published label that the dispatch refuses would be worse than no contract: consumers
/// would type against a name `run_baseline` rejects. The artifact's `BaselineAgent` list is
/// therefore required to resolve, entry for entry, through the same function the wasm export
/// layer calls, and a name that is not in the list is required not to.
#[test]
fn every_published_baseline_label_resolves_through_the_dispatch() {
    let contract = build_contract();
    let labels = contract["enums"]["BaselineAgent"]
        .as_array()
        .unwrap()
        .clone();
    for label in &labels {
        let label = label.as_str().expect("a baseline label is a string");
        assert_eq!(
            BaselineAgent::parse(label).map(|agent| agent.label()),
            Ok(label),
            "{label} is published in the contract but does not resolve to itself"
        );
    }
    assert!(
        BaselineAgent::parse("not_a_baseline").is_err(),
        "the resolver accepts a name the contract does not publish, so resolving proves nothing"
    );
}
