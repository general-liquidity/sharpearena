//! Conformance kit — the executable definition of "conforms to the SharpeArena Agent
//! Interface v1.0". Every fixture under `contract/conformance/` is replayed through the
//! in-process reference agent ([`BuyAndHold`]) and the resulting [`Decision`] is checked
//! for well-formedness against the wire contract:
//!
//! - it deserializes through the protocol types,
//! - every `action` is one of the four enum variants,
//! - every `target_weight` is finite, and
//! - every order `symbol` is a subset of the symbols the observation actually offered.
//!
//! Fixtures that carry a `legacy_decision` (no `confidence` / `rationale` / `reasoning`)
//! also assert the additive-only `#[serde(default)]` discipline: the legacy shape still
//! parses and is itself well-formed. New fixtures are auto-discovered from the directory.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use sharpearena::{
    Action, Agent, BuyAndHold, Decision, DecisionCost, MarketObservation, Order, PositionState,
    SymbolSnapshot, CONTRACT_VERSION,
};

/// The frozen wire version this kit certifies against. A bump here is a deliberate,
/// reviewed event (see `GOVERNANCE.md`), not an accident of editing.
#[test]
fn contract_version_is_frozen() {
    assert_eq!(CONTRACT_VERSION, "1.0");
}

/// Assert a decision is well-formed against the observed universe.
fn assert_well_formed(decision: &Decision, observed: &BTreeSet<String>, ctx: &str) {
    for order in &decision.orders {
        assert!(
            matches!(
                order.action,
                Action::Buy | Action::Sell | Action::Hold | Action::Close
            ),
            "{ctx}: action outside the enum",
        );
        assert!(
            order.target_weight.is_finite(),
            "{ctx}: non-finite target_weight for {}",
            order.symbol,
        );
        assert!(
            observed.contains(&order.symbol),
            "{ctx}: order on unobserved symbol {}",
            order.symbol,
        );
        assert!(
            (0.0..=1.0).contains(&order.confidence),
            "{ctx}: confidence {} out of [0, 1] for {}",
            order.confidence,
            order.symbol,
        );
    }
}

#[test]
fn conformance_fixtures_yield_well_formed_decisions() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contract/conformance");
    let mut fixtures = 0usize;

    for entry in fs::read_dir(&dir).expect("conformance directory exists") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = fs::read_to_string(&path).expect("fixture readable");
        let v: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: invalid JSON: {e}"));

        // Parse the observation through the protocol types (leg 1 of the badge).
        let obs: MarketObservation = serde_json::from_value(v["observation"].clone())
            .unwrap_or_else(|e| panic!("{name}: observation does not parse: {e}"));
        let observed: BTreeSet<String> = obs.symbols.iter().map(|s| s.symbol.clone()).collect();

        // Run the in-process reference agent over the observation (leg 2).
        let mut agent = BuyAndHold;
        let decision = agent.decide(&obs);
        assert_well_formed(&decision, &observed, &name);
        assert_eq!(
            decision.orders.len(),
            obs.symbols.len(),
            "{name}: BuyAndHold should weight every observed symbol",
        );

        // A fixture may also pin the legacy decision shape — it must still parse and be
        // well-formed, proving the additive-only discipline survives (leg 3).
        if let Some(legacy) = v.get("legacy_decision") {
            let parsed: Decision = serde_json::from_value(legacy.clone())
                .unwrap_or_else(|e| panic!("{name}: legacy_decision does not parse: {e}"));
            assert_well_formed(&parsed, &observed, &name);
        }

        fixtures += 1;
    }

    assert!(
        fixtures >= 4,
        "expected the full conformance kit, found {fixtures}"
    );
}

// -- published schema vs the types it claims to describe ----------------------

/// Every key a fully-populated value serializes to, at the top level.
fn serialized_keys<T: serde::Serialize>(value: &T) -> BTreeSet<String> {
    match serde_json::to_value(value).expect("value serializes") {
        serde_json::Value::Object(map) => map.keys().cloned().collect(),
        other => panic!("expected a JSON object, got {other}"),
    }
}

/// The property names a schema object declares.
fn schema_properties(schema: &serde_json::Value, pointer: &str) -> BTreeSet<String> {
    schema
        .pointer(pointer)
        .unwrap_or_else(|| panic!("schema has no {pointer}"))
        .as_object()
        .unwrap_or_else(|| panic!("{pointer} is not an object"))
        .keys()
        .cloned()
        .collect()
}

/// `contract/decision.schema.json` is published as the authoritative wire contract, and it
/// sets `additionalProperties: false`. That makes any field present on the Rust type but
/// absent from the schema an interoperability break rather than a documentation gap: a
/// conforming non-Rust implementer validating a Rust-emitted decision would reject it.
/// `Decision::cost` drifted in exactly this way, and it was invisible because `cost` is
/// `skip_serializing_if = "Option::is_none"`, so only a cost-reporting agent tripped it.
/// This test populates every optional field so the comparison sees the full surface.
#[test]
fn published_decision_schema_matches_the_protocol_types() {
    let schema: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("contract/decision.schema.json").expect("schema is readable"),
    )
    .expect("schema is valid JSON");

    let order = Order {
        symbol: "TSLA".into(),
        action: Action::Sell,
        target_weight: -0.3,
        confidence: 0.7,
        rationale: "downtrend".into(),
    };
    let decision = Decision {
        orders: vec![order.clone()],
        reasoning: "tilt short".into(),
        cost: Some(DecisionCost {
            cost_usd: 0.014,
            tokens_in: 2048,
            tokens_out: 256,
            reasoning_tokens: 64,
        }),
    };

    assert_eq!(
        serialized_keys(&decision),
        schema_properties(&schema, "/properties"),
        "Decision fields and decision.schema.json properties have drifted",
    );
    assert_eq!(
        serialized_keys(&order),
        schema_properties(&schema, "/$defs/Order/properties"),
        "Order fields and the schema's Order definition have drifted",
    );
    assert_eq!(
        serialized_keys(&decision.cost.expect("cost was populated above")),
        schema_properties(&schema, "/$defs/DecisionCost/properties"),
        "DecisionCost fields and the schema's DecisionCost definition have drifted",
    );
}

/// The same guard for the observation half of the contract.
#[test]
fn published_observation_schema_matches_the_protocol_types() {
    let schema: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("contract/observation.schema.json").expect("schema is readable"),
    )
    .expect("schema is valid JSON");

    let snapshot = SymbolSnapshot {
        symbol: "TSLA".into(),
        close_history: vec![240.0, 235.5],
        fundamentals: [("pe".to_string(), 61.4)].into_iter().collect(),
        news: vec!["deliveries beat".into()],
    };
    let position = PositionState {
        symbol: "TSLA".into(),
        shares: 0.5,
        avg_price: 250.0,
    };
    let observation = MarketObservation {
        date: "2025-05-05".into(),
        cash: 1.0,
        symbols: vec![snapshot.clone()],
        portfolio: vec![position.clone()],
    };

    assert_eq!(
        serialized_keys(&observation),
        schema_properties(&schema, "/properties"),
        "MarketObservation fields and observation.schema.json properties have drifted",
    );
    assert_eq!(
        serialized_keys(&snapshot),
        schema_properties(&schema, "/$defs/SymbolSnapshot/properties"),
        "SymbolSnapshot fields and the schema's definition have drifted",
    );
    assert_eq!(
        serialized_keys(&position),
        schema_properties(&schema, "/$defs/PositionState/properties"),
        "PositionState fields and the schema's definition have drifted",
    );
}
