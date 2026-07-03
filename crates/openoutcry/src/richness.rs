//! Information-disclosure difficulty: the axis orthogonal to [`DistributionMode`].
//!
//! [`DistributionMode`](crate::DistributionMode) governs *how adversarial the price path
//! is* (volatility, jumps, regime structure). It says nothing about *how much of that
//! market the agent is allowed to see*. Two agents on the identical Calm panel can face
//! very different problems: one shown a 50-bar history plus fundamentals and news
//! headlines, the other shown only the last 3 closes and nothing else. The second is
//! genuinely harder despite the identical (calm) data-generating process; it measures
//! **information efficiency** and robustness to **data poverty**, a capability the regime
//! tiers structurally cannot probe. A scenario is therefore a point on a 2-D difficulty
//! grid: `(DistributionMode × ObservationRichness)`.
//!
//! Every field is additive and defaults to the pre-existing behavior ([`DEFAULT_LOOKBACK`]
//! trailing closes, no optional fields), so a scenario built without touching richness
//! produces a byte-identical observation stream. Richer disclosure only ever surfaces
//! **more past / contextual** information; it never reveals a future bar, so the leak-free
//! point-in-time invariant is untouched.

use serde::{Deserialize, Serialize};

/// The default number of trailing closes surfaced per symbol: the historical lookback the
/// harness has always emitted. Both [`ObservationRichness::default`] and
/// [`RichnessTier::Standard`] resolve to this, so the default observation path is unchanged.
pub const DEFAULT_LOOKBACK: usize = 20;

/// How much of the market an agent is shown at each decision point. Independent of the
/// price-path regime: a data-poor observation of a calm panel can be harder than a
/// data-rich observation of an extreme one. All fields are additive; [`Default`] is exactly
/// the pre-existing disclosure (a [`DEFAULT_LOOKBACK`]-bar history with no optional fields),
/// so the default path is byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationRichness {
    /// Trailing closes surfaced per symbol (overrides the historical fixed lookback). Fewer
    /// bars is a data-poorer, harder observation; more bars a richer, easier one.
    pub lookback: usize,
    /// Whether each symbol snapshot's optional `fundamentals` map is populated with
    /// point-in-time derived context (all `<= t`), or left empty as before.
    pub fundamentals: bool,
    /// Whether each symbol snapshot's optional `news` headlines are populated with
    /// point-in-time derived context (all `<= t`), or left empty as before.
    pub news: bool,
}

impl Default for ObservationRichness {
    fn default() -> Self {
        Self {
            lookback: DEFAULT_LOOKBACK,
            fundamentals: false,
            news: false,
        }
    }
}

/// A small set of named richness presets spanning the disclosure axis. They compose with
/// [`DistributionMode`](crate::DistributionMode) to form the `(regime × richness)` grid:
/// pick a regime for the price path and a tier for how much of it the agent sees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RichnessTier {
    /// The data-poor extreme: a 3-bar history, no fundamentals, no news.
    DataPoor,
    /// The historical default disclosure: a [`DEFAULT_LOOKBACK`]-bar history, no optional
    /// fields. Byte-identical to a scenario built without a richness setting.
    #[default]
    Standard,
    /// The data-rich extreme: a 50-bar history plus fundamentals and news headlines.
    DataRich,
}

impl RichnessTier {
    /// The concrete [`ObservationRichness`] this preset resolves to.
    pub fn richness(self) -> ObservationRichness {
        match self {
            RichnessTier::DataPoor => ObservationRichness {
                lookback: 3,
                fundamentals: false,
                news: false,
            },
            RichnessTier::Standard => ObservationRichness::default(),
            RichnessTier::DataRich => ObservationRichness {
                lookback: 50,
                fundamentals: true,
                news: true,
            },
        }
    }

    /// The three tiers in disclosure order (poorest first), for sweeping the whole axis.
    pub fn all() -> [RichnessTier; 3] {
        [
            RichnessTier::DataPoor,
            RichnessTier::Standard,
            RichnessTier::DataRich,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_historical_disclosure() {
        let d = ObservationRichness::default();
        assert_eq!(d.lookback, DEFAULT_LOOKBACK);
        assert!(!d.fundamentals);
        assert!(!d.news);
    }

    #[test]
    fn standard_tier_equals_default() {
        // Standard must be byte-for-byte the pre-existing behavior, so a scenario tagged
        // Standard and one built with no richness setting are indistinguishable.
        assert_eq!(
            RichnessTier::Standard.richness(),
            ObservationRichness::default()
        );
        assert_eq!(RichnessTier::default(), RichnessTier::Standard);
    }

    #[test]
    fn tiers_span_the_axis() {
        let poor = RichnessTier::DataPoor.richness();
        let std = RichnessTier::Standard.richness();
        let rich = RichnessTier::DataRich.richness();
        // Lookback strictly increases along the axis.
        assert!(poor.lookback < std.lookback);
        assert!(std.lookback < rich.lookback);
        // Only the rich extreme populates the optional contextual fields.
        assert!(!poor.fundamentals && !poor.news);
        assert!(!std.fundamentals && !std.news);
        assert!(rich.fundamentals && rich.news);
    }

    #[test]
    fn tier_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&RichnessTier::DataPoor).unwrap(),
            "\"data_poor\""
        );
        assert_eq!(
            serde_json::to_string(&RichnessTier::DataRich).unwrap(),
            "\"data_rich\""
        );
    }

    #[test]
    fn richness_round_trips_through_json() {
        let r = RichnessTier::DataRich.richness();
        let j = serde_json::to_string(&r).unwrap();
        let back: ObservationRichness = serde_json::from_str(&j).unwrap();
        assert_eq!(r, back);
    }
}
