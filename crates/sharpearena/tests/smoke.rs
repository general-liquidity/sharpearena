//! M0 smoke tests — prove the re-exported SharpeArena surface is usable end-to-end from
//! outside the crate: the engine runs an agent in a market, and the language-agnostic wire
//! contract round-trips through the public types.

use serde::Deserialize;
use sharpearena::{
    run_backtest, Action, BuyAndHold, CostModel, Dataset, Decision, MarketObservation, Order,
    TradingEnv, Window,
};

#[derive(Deserialize)]
struct ForwardCommitmentFixture {
    agent_id: String,
    target_window: String,
    artifact_digest: String,
    salt: String,
    commit_hash: String,
}

/// The engine surface: run a baseline agent over a synthetic point-in-time dataset and get
/// per-period returns + a decision trace back.
#[test]
fn reexported_engine_runs_a_backtest() {
    let data = Dataset::synthetic(4, 120, 11);
    let mut agent = BuyAndHold;
    let run = run_backtest(
        &data,
        &mut agent,
        Window {
            start: 20,
            end: 120,
        },
        1,
        CostModel::default(),
    );
    assert_eq!(run.returns.len(), 100);
    assert!(!run.trace.events.is_empty());
}

/// The wire contract: the language-agnostic observation/decision JSON deserializes through
/// the re-exported protocol types — including the legacy decision shape (no `confidence` /
/// `rationale`), proving the additive-only `#[serde(default)]` discipline survives re-export.
#[test]
fn reexported_wire_contract_parses() {
    let obs_json = r#"{
        "date": "2025-01-02",
        "cash": 1.0,
        "symbols": [{ "symbol": "AAPL", "close_history": [187.2, 188.0, 190.4] }],
        "portfolio": []
    }"#;
    let obs: MarketObservation = serde_json::from_str(obs_json).expect("observation parses");
    assert_eq!(obs.symbols.len(), 1);

    let legacy_decision =
        r#"{ "orders": [{ "symbol": "AAPL", "action": "buy", "target_weight": 0.5 }] }"#;
    let decision: Decision = serde_json::from_str(legacy_decision).expect("legacy decision parses");
    assert_eq!(decision.orders.len(), 1);
}

/// The environment advertises signed target weights, so the public composed surface must
/// open an actual short rather than accepting the wire value and silently clamping it to
/// zero inside the shared engine.
#[test]
fn signed_target_opens_a_short_through_the_public_environment() {
    let data = Dataset::synthetic(1, 40, 19);
    let symbol = data.symbols().into_iter().next().expect("one symbol");
    let mut env = TradingEnv::new(data, Window { start: 10, end: 30 }, CostModel::default(), 3);
    let observation = env.reset();
    let decision = Decision {
        orders: vec![Order {
            symbol: symbol.clone(),
            action: Action::Sell,
            target_weight: -0.5,
            confidence: 1.0,
            rationale: "regression: signed target".to_string(),
        }],
        reasoning: String::new(),
        cost: None,
    };
    decision
        .validate_for(&observation)
        .expect("the canonical contract accepts the signed target");

    let stepped = env.step(decision);
    let position = stepped
        .observation
        .portfolio
        .iter()
        .find(|position| position.symbol == symbol)
        .expect("the short position is present in the next observation");
    assert!(
        position.shares < 0.0,
        "a negative target must create negative shares, not a flat hold"
    );
}

/// Python and Rust both consume this fixture. Calling the published attestation crate
/// here makes a delimiter or field-order change fail in SharpeArena before release.
#[test]
fn forward_commitment_matches_the_published_attestation_primitive() {
    let fixture: ForwardCommitmentFixture = serde_json::from_str(include_str!(
        "../contract/attestation/forward-commitment.json"
    ))
    .expect("shared commitment fixture parses");
    let commitment = sharpebench_attest::make_commitment(
        &fixture.agent_id,
        &fixture.target_window,
        &fixture.artifact_digest,
        &fixture.salt,
    );
    assert_eq!(commitment.commit_hash, fixture.commit_hash);
}
