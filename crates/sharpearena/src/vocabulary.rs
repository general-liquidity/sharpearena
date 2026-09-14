//! The JSON label vocabularies the export layers write, each authored exactly once.
//!
//! Most of the engine's wire vocabulary needs no module like this. `DistributionMode`,
//! `ObservationRichness` and `ScenarioSpec` carry their wire form in serde, so
//! `contract/engine-enums.v1.json` reads their labels and field names straight off the
//! type and there is no second authoring to drift from. Two vocabularies could not do
//! that, for different reasons, and both live here so each has one authoring site:
//!
//! * **`BaselineAgent` had no Rust declaration at all.** The four names the JSON surfaces
//!   dispatch on were string literals in the wasm crate's `build_agent`, matched against a
//!   `BaselineConfig.agent` field typed `String`, so `npm/sharpearena/src/types.ts`
//!   restated a vocabulary with nothing to restate it *from*. The enum belongs to this
//!   crate rather than the wasm crate because the contract generator is a test of this
//!   crate and cannot depend on its own dependent.
//!
//! * **`Regime` cannot carry its own labels.** It is re-exported from `sharpebench-sim`
//!   and derives no `Serialize`; this crate cannot add one to a foreign type. Its labels
//!   were therefore authored twice, once in the wasm export layer's `tag_regime` and once
//!   in the contract generator. Those two authorings are not equally defended: the wasm
//!   side is pinned by the committed `tag_regime` goldens in
//!   `contract/attestation/kernel-goldens.json`, so renaming a label there fails a test,
//!   while the generator side was pinned by nothing. A rename there, followed by the
//!   documented `UPDATE_ENGINE_ENUM_CONTRACT=1` regeneration and a matching edit to
//!   `types.ts`, left the whole suite green while publishing a TypeScript label the engine
//!   does not emit. [`regime_label`] is the single site both now read, which puts the
//!   generator's output under the goldens' pin: a rename changes what `tag_regime` emits,
//!   so regenerating the contract no longer buys silence.

use crate::Regime;

/// The baseline agents the JSON surfaces dispatch by name.
///
/// This is the declaration `npm/sharpearena/src/types.ts`'s `BaselineAgent` union restates
/// and `contract/engine-enums.v1.json` is generated from. It deliberately derives no
/// `Serialize`: a serde representation would be a *second* authoring of the same four
/// labels, which is the failure mode this module exists to remove. [`label`] is the only
/// place a label is written.
///
/// [`label`]: BaselineAgent::label
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BaselineAgent {
    BuyAndHold,
    Hold,
    Momentum,
    Random,
}

impl BaselineAgent {
    /// Every variant, in the order a parse failure lists them.
    ///
    /// Order is part of the contract: it is the order of the generated artifact's label
    /// list and of the `expected ...` clause in [`BaselineAgent::parse`]'s error.
    pub const ALL: [Self; 4] = [Self::BuyAndHold, Self::Hold, Self::Momentum, Self::Random];

    /// The wire label, and the only place in the tree it is written.
    ///
    /// The match carries no wildcard arm, so a variant added to the enum stops this
    /// compiling rather than reaching a caller unnamed. That is the same device the
    /// contract generator uses for `DistributionMode`.
    pub const fn label(self) -> &'static str {
        match self {
            Self::BuyAndHold => "buy_and_hold",
            Self::Hold => "hold",
            Self::Momentum => "momentum",
            Self::Random => "random",
        }
    }

    /// Resolve a wire label, or refuse it by naming every label that exists.
    ///
    /// The `expected ...` clause is built from [`Self::ALL`] rather than typed out, so a
    /// variant added to the enum is offered to the caller without a second edit. The
    /// wording is the refusal the npm suite's `an unknown baseline agent throws` asserts
    /// on and the wasm bundle gate's fixed battery compares byte for byte; it is contract,
    /// not diagnostics.
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|agent| agent.label() == name)
            .ok_or_else(|| {
                let expected: Vec<&str> = Self::ALL.iter().map(|agent| agent.label()).collect();
                format!(
                    "unknown baseline agent {name:?} (expected {})",
                    expected.join(" | ")
                )
            })
    }
}

/// Every [`Regime`], in the order the generated contract lists them.
pub const REGIMES: [Regime; 3] = [Regime::Bull, Regime::Bear, Regime::Chop];

/// The wire label for a [`Regime`], and the only place in the tree it is written.
///
/// Both the wasm export layer's `tag_regime` and the contract generator read it here, so
/// the generated artifact and the bytes the engine actually emits cannot disagree. See the
/// module documentation for what that replaced.
pub const fn regime_label(regime: Regime) -> &'static str {
    match regime {
        Regime::Bull => "bull",
        Regime::Bear => "bear",
        Regime::Chop => "chop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_baseline_label_parses_back_to_its_own_variant() {
        for agent in BaselineAgent::ALL {
            assert_eq!(BaselineAgent::parse(agent.label()), Ok(agent));
        }
    }

    #[test]
    fn the_baseline_labels_are_the_four_the_surfaces_publish() {
        let labels: Vec<&str> = BaselineAgent::ALL.iter().map(|a| a.label()).collect();
        assert_eq!(labels, ["buy_and_hold", "hold", "momentum", "random"]);
    }

    /// The refusal wording is asserted on by `an unknown baseline agent throws` in the npm
    /// suite and compared byte for byte by `scripts/check-wasm-bundle.mjs`, so it is pinned
    /// here in full rather than matched loosely.
    #[test]
    fn an_unknown_baseline_label_is_refused_with_the_published_wording() {
        assert_eq!(
            BaselineAgent::parse("nope"),
            Err(
                "unknown baseline agent \"nope\" (expected buy_and_hold | hold | momentum | random)"
                    .to_string()
            )
        );
    }

    /// A near miss is a miss: the resolver matches the whole label, not a prefix.
    #[test]
    fn a_near_miss_is_not_resolved() {
        assert!(BaselineAgent::parse("buy_and_hol").is_err());
        assert!(BaselineAgent::parse("BUY_AND_HOLD").is_err());
        assert!(BaselineAgent::parse("").is_err());
    }

    #[test]
    fn the_regime_labels_are_the_three_the_surfaces_publish() {
        let labels: Vec<&str> = REGIMES.iter().map(|r| regime_label(*r)).collect();
        assert_eq!(labels, ["bull", "bear", "chop"]);
    }

    /// `REGIMES` is an array, so unlike [`regime_label`]'s match it cannot be made
    /// exhaustive by the compiler. It can at least be required to be free of duplicates,
    /// which is what a careless edit to it would produce.
    #[test]
    fn the_regime_list_names_each_variant_once() {
        let mut labels: Vec<&str> = REGIMES.iter().map(|r| regime_label(*r)).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), REGIMES.len());
    }
}
