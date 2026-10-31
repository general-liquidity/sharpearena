//! Opt-in scenario redraw for recycled vector lanes.
//!
//! By default a recycled lane replays the scenario its construction seed selected, so a
//! run sees exactly as many scenarios as lanes. [`AutoresetScenarios::FreshV1`] instead
//! gives each recycled episode a new scenario, drawn from a per-lane stream that is a
//! pure function of `(root_seed, lane, episode)`. Nothing is sampled from ambient state,
//! so a run replays from its constructor arguments alone.
//!
//! Every draw lands in the training band `[0, EVAL_SEED_BASE)`. The bound is structural:
//! [`fresh_scenario_seed`] maps into that interval, so no draw can reach a held-out band
//! whatever the root seed. Lanes constructed outside the band are refused rather than
//! silently mixing held-out and training scenarios in one run.
//!
//! The stream is frozen under its label. A change to [`fresh_scenario_seed`] needs a new
//! label, never an edit under `fresh_v1`, because a recorded run names the label and
//! nothing else about the derivation.
//!
//! Draws are with replacement within a lane and across lanes. The band holds a million
//! seeds, so a repeat is rare but possible, and two draws are not promised to differ.
//!
//! This module is deliberately outside `SPEC_FILES`: it chooses which scenario a lane
//! runs and defines no tape bytes.

use crate::vec_env::LaneConfig;
use crate::{Dataset, EVAL_SEED_BASE};

/// The half-open training band every fresh draw lands in.
pub const FRESH_SEED_BAND: (u64, u64) = (0, EVAL_SEED_BASE);

/// Where a recycled lane's next scenario comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoresetScenarios {
    /// Restart the lane's current scenario (the historical behaviour).
    #[default]
    Replay,
    /// Draw a new scenario per recycled episode from the `fresh_v1` seed stream.
    FreshV1,
}

impl AutoresetScenarios {
    /// Parse the wire label (`"replay" | "fresh_v1"`).
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "replay" => Some(Self::Replay),
            "fresh_v1" => Some(Self::FreshV1),
            _ => None,
        }
    }

    /// The wire label for this policy.
    pub fn label(self) -> &'static str {
        match self {
            Self::Replay => "replay",
            Self::FreshV1 => "fresh_v1",
        }
    }
}

/// SplitMix64 finalizer. Restated rather than shared because the `scenario_gen` copy
/// lives in a `SPEC_FILES` input and a frozen stream must not depend on a private helper.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// `"FreshSV1"`: separates this stream from every other seed derivation in the crate.
const DOMAIN_V1: u64 = 0x4672_6573_6853_5631;

/// The `episode`-th fresh scenario seed of `lane`, always inside [`FRESH_SEED_BAND`].
///
/// Rejection sampling keeps the draw uniform over the band instead of biasing it by a
/// bare modulo. The rejection walk is itself deterministic.
pub fn fresh_scenario_seed(root_seed: u64, lane: u64, episode: u64) -> u64 {
    let (start, end) = FRESH_SEED_BAND;
    let span = end - start;
    let accept_below = u64::MAX - (u64::MAX % span);

    let mut s = mix64(DOMAIN_V1 ^ root_seed);
    s = mix64(s ^ lane.wrapping_add(1).wrapping_mul(GOLDEN));
    s = mix64(s ^ episode.wrapping_add(1).wrapping_mul(GOLDEN.rotate_left(17)));
    while s >= accept_below {
        s = mix64(s.wrapping_add(GOLDEN));
    }
    start + (s % span)
}

/// Why a batch cannot run under [`AutoresetScenarios::FreshV1`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshScenarioError {
    /// One template per lane is required.
    LaneCount { lanes: usize, templates: usize },
    /// A lane's template disagrees with the seed the lane is running.
    TemplateSeedMismatch {
        lane: usize,
        lane_seed: u64,
        template_seed: u64,
    },
    /// A lane starts outside the training band.
    SeedOutsideTrainBand { lane: usize, seed: u64 },
}

impl std::fmt::Display for FreshScenarioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LaneCount { lanes, templates } => write!(
                f,
                "fresh autoreset scenarios need one template per lane: {lanes} lanes, {templates} templates"
            ),
            Self::TemplateSeedMismatch {
                lane,
                lane_seed,
                template_seed,
            } => write!(
                f,
                "lane {lane} runs seed {lane_seed} but its template names seed {template_seed}"
            ),
            Self::SeedOutsideTrainBand { lane, seed } => write!(
                f,
                "lane {lane} seed {seed} is outside the training band [{}, {}); fresh autoreset scenarios are training-only",
                FRESH_SEED_BAND.0, FRESH_SEED_BAND.1
            ),
        }
    }
}

impl std::error::Error for FreshScenarioError {}

/// One lane's redraw state.
pub(crate) struct FreshLane {
    pub(crate) template: LaneConfig,
    pub(crate) root_seed: u64,
    pub(crate) drawn: Vec<u64>,
    pub(crate) current: Dataset,
}

impl FreshLane {
    pub(crate) fn next_seed(&self, lane: usize) -> u64 {
        fresh_scenario_seed(self.root_seed, lane as u64, self.drawn.len() as u64)
    }
}

/// The per-lane redraw state of a batch running under `fresh_v1`.
pub struct FreshScenarios {
    pub(crate) lanes: Vec<FreshLane>,
}

impl FreshScenarios {
    pub(crate) fn new(
        templates: Vec<LaneConfig>,
        seeds: &[u64],
    ) -> Result<Self, FreshScenarioError> {
        if templates.len() != seeds.len() {
            return Err(FreshScenarioError::LaneCount {
                lanes: seeds.len(),
                templates: templates.len(),
            });
        }
        let (start, end) = FRESH_SEED_BAND;
        for (lane, (template, &seed)) in templates.iter().zip(seeds).enumerate() {
            if !(start..end).contains(&seed) {
                return Err(FreshScenarioError::SeedOutsideTrainBand { lane, seed });
            }
            if template.seed != seed {
                return Err(FreshScenarioError::TemplateSeedMismatch {
                    lane,
                    lane_seed: seed,
                    template_seed: template.seed,
                });
            }
        }
        let lanes = templates
            .into_iter()
            .map(|template| FreshLane {
                root_seed: template.seed,
                drawn: Vec::new(),
                current: template.dataset(),
                template,
            })
            .collect();
        Ok(Self { lanes })
    }

    /// The seed that opened `lane`'s stream.
    pub fn root_seed(&self, lane: usize) -> u64 {
        self.lanes[lane].root_seed
    }

    /// Every seed `lane` has drawn so far, in episode order.
    pub fn drawn_seeds(&self, lane: usize) -> &[u64] {
        &self.lanes[lane].drawn
    }

    /// The dataset `lane` is running now, for a fingerprint of the live tape rather than
    /// of a regenerated shadow.
    pub fn current_dataset(&self, lane: usize) -> &Dataset {
        &self.lanes[lane].current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Freezes `fresh_v1`. A failure here means the stream moved: restore it or ship a
    /// new label, because every recorded run depends on these values.
    #[test]
    fn fresh_v1_stream_is_frozen() {
        let drawn: Vec<u64> = (0..4).map(|e| fresh_scenario_seed(0, 0, e)).collect();
        assert_eq!(drawn, FROZEN_ROOT0_LANE0);
        assert_eq!(fresh_scenario_seed(7, 3, 11), FROZEN_ROOT7_LANE3_EPISODE11);
    }

    #[test]
    fn extreme_inputs_stay_inside_the_training_band() {
        for root in [0, 1, EVAL_SEED_BASE, u64::MAX] {
            for lane in [0, 1, u64::MAX] {
                for episode in [0, 1, 1 << 40, u64::MAX] {
                    assert!(fresh_scenario_seed(root, lane, episode) < EVAL_SEED_BASE);
                }
            }
        }
    }

    #[test]
    fn lane_and_episode_each_move_the_draw() {
        assert_ne!(fresh_scenario_seed(5, 0, 0), fresh_scenario_seed(5, 1, 0));
        assert_ne!(fresh_scenario_seed(5, 0, 0), fresh_scenario_seed(5, 0, 1));
        assert_ne!(fresh_scenario_seed(5, 0, 0), fresh_scenario_seed(6, 0, 0));
    }

    #[test]
    fn only_the_two_labels_parse_and_replay_is_the_default() {
        assert_eq!(AutoresetScenarios::default(), AutoresetScenarios::Replay);
        for policy in [AutoresetScenarios::Replay, AutoresetScenarios::FreshV1] {
            assert_eq!(AutoresetScenarios::from_label(policy.label()), Some(policy));
        }
        assert_eq!(AutoresetScenarios::from_label("fresh"), None);
    }

    const FROZEN_ROOT0_LANE0: [u64; 4] = [579_153, 783_213, 915_425, 650_791];
    const FROZEN_ROOT7_LANE3_EPISODE11: u64 = 181_547;
}
