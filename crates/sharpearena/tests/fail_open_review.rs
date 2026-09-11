//! Characterization tests for the 2026-09-09 adversarial review of SharpeArena
//! (`docs/audits/2026-09-09/ARENA-REVIEW.md`). Each test pins the *current*
//! permissive behaviour so a later repair has something to invert. No production
//! code is changed by this file.

use sharpearena::{
    level_seed, mandate_breach, perturb_action, train_test_split, AdaptiveCurriculum, ExecNoise,
    Mandate, MandateStyle, ScenarioSpec,
};

fn long_only(max_drawdown: Option<f64>, max_inventory: Option<f64>) -> Mandate {
    Mandate {
        style: MandateStyle::LongOnly,
        max_drawdown,
        max_inventory,
        benchmark: None,
        text: String::new(),
    }
}

/// R1. `mandate_breach` combines its breach sources with `f64::max` and guards each one
/// with a `>` comparison; both discard NaN. A weight vector or return series carrying a
/// NaN therefore scores a *clean* mandate (0.0) rather than being refused, on every
/// breach source: the long-only rule, the market-neutral rule, the inventory cap and
/// the drawdown cap.
#[test]
fn r1_nan_weights_and_returns_score_a_clean_mandate() {
    // Long-only: `fold(f64::INFINITY, f64::min)` over an all-NaN bar stays at INFINITY,
    // so the bar is never counted as holding a short.
    let structural = long_only(None, None);
    assert_eq!(
        mandate_breach(&structural, &[], &vec![vec![-0.5, 0.2]; 4]),
        1.0
    );
    assert_eq!(
        mandate_breach(&structural, &[], &vec![vec![f64::NAN, f64::NAN]; 4]),
        0.0,
        "an all-NaN book scores a clean long-only mandate"
    );

    // Market-neutral: `gross > EPS` is false for a NaN gross, so the bar contributes 0.
    let neutral = Mandate {
        style: MandateStyle::MarketNeutral,
        ..long_only(None, None)
    };
    assert!(mandate_breach(&neutral, &[], &[vec![0.5, 0.5]]) > 0.0);
    assert_eq!(mandate_breach(&neutral, &[], &[vec![0.5, f64::NAN]]), 0.0);

    // Inventory cap: `gross > cap` is false for a NaN gross.
    let capped = long_only(None, Some(1.0));
    assert!(mandate_breach(&capped, &[], &[vec![5.0, 5.0]]) > 0.0);
    assert_eq!(mandate_breach(&capped, &[], &[vec![5.0, f64::NAN]]), 0.0);

    // Drawdown cap: one NaN return poisons the equity curve, and `dd > mdd` is then
    // false for every later bar, so the realized drawdown reads as zero.
    let braked = long_only(Some(0.10), None);
    assert!(mandate_breach(&braked, &[-0.30, -0.30], &[vec![0.5]]) > 0.0);
    assert_eq!(
        mandate_breach(&braked, &[f64::NAN, -0.30, -0.30], &[vec![0.5]]),
        0.0,
        "a leading NaN return hides the whole drawdown"
    );
}

/// R2. `exec_noise::perturb` validates neither knob, and no caller on any surface
/// validates them either, so a malformed integrity setting is never refused. Negative
/// knobs take the "no knobs" pass-through and report a run with no execution noise at
/// all; a probability above one makes every step sticky; NaN skips every guard and
/// poisons the realized action instead.
#[test]
fn r2_malformed_execution_noise_is_never_refused() {
    let requested = [0.2, -0.5, 0.7];
    let previous = [9.9, 9.9, 9.9];

    let live = ExecNoise {
        delay_prob: 0.0,
        slippage_bps: 100.0,
    };
    assert_ne!(
        perturb_action(5, 1, &requested, &previous, &live),
        requested.to_vec(),
        "a live slippage knob must perturb"
    );

    // Negative: silent pass-through, indistinguishable from "no noise configured".
    assert_eq!(
        perturb_action(
            5,
            1,
            &requested,
            &previous,
            &ExecNoise {
                delay_prob: -1.0,
                slippage_bps: -100.0
            }
        ),
        requested.to_vec()
    );
    // A probability above one is unchecked: every step replays the previous action, so
    // the agent's own decisions never reach the market.
    assert_eq!(
        perturb_action(
            5,
            1,
            &requested,
            &previous,
            &ExecNoise {
                delay_prob: 4.0,
                slippage_bps: 0.0
            }
        ),
        previous.to_vec()
    );
    // NaN: no guard fires and the realized action becomes NaN rather than a refusal.
    for cfg in [
        ExecNoise {
            delay_prob: f64::NAN,
            slippage_bps: f64::NAN,
        },
        ExecNoise {
            delay_prob: 0.0,
            slippage_bps: f64::NAN,
        },
    ] {
        let out = perturb_action(5, 1, &requested, &previous, &cfg);
        assert!(out.iter().all(|x| x.is_nan()), "{out:?}");
    }
}

/// R3. `AdaptiveCurriculum::with_prior` documents `prior` in `[0, 1]` but does not
/// enforce it. A NaN prior makes every unseen level's ZPD weight NaN, and `select_next`
/// compares with `>`, so the scan never displaces its seed: the curriculum degenerates to
/// "always the first level" without complaint. An out-of-range prior yields a negative
/// weight, which the same scan also cannot rank.
#[test]
fn r3_curriculum_prior_is_unvalidated() {
    let nan = AdaptiveCurriculum::with_prior([5u64, 6, 7], f64::NAN);
    assert!(nan.weight(6).is_nan());
    assert_eq!(nan.select_next(), 5);

    let out_of_range = AdaptiveCurriculum::with_prior([5u64, 6, 7], 5.0);
    assert_eq!(out_of_range.weight(6), 5.0 * (1.0 - 5.0));
    assert!(out_of_range.weight(6) < 0.0);
}

/// R4. `train_test_split(train, 0, gap)` is documented as carving `n_test` held-out
/// levels. `n_test == 0` instead produces an *unbounded* family (`num_levels == 0` is
/// Procgen's "unlimited"), so asking for zero held-out levels yields a test band that
/// spans the rest of the seed space.
#[test]
fn r4_zero_test_levels_yields_an_unbounded_test_family() {
    let train = ScenarioSpec {
        start_level: 100,
        num_levels: 50,
        ..ScenarioSpec::default()
    };
    let (_, test) = train_test_split(train, 0, 10_000).expect("a bounded train band splits");
    assert_eq!(test.num_levels, 0);
    // An "empty" test family still yields distinct seeds for every index.
    let a = level_seed(&test, 0);
    let b = level_seed(&test, 1);
    let far = level_seed(&test, 1_000_000_000);
    assert_ne!(a, b);
    assert_ne!(a, far);
}

/// R5. The disjointness guarantee of `train_test_split` used to be carried by
/// `debug_assert!`, so it was compiled out of every release build: the published crate,
/// the maturin wheel and the wasm bundle. It is now a typed refusal (`SplitError`), which
/// survives `[profile.release]`. This test is compiled only in the configuration that
/// actually ships, because that is the configuration the defect lived in; CI runs it via
/// the `cargo test --release --test fail_open_review` step.
///
/// The `Err` is the isolation: nothing else in the function can return one, so the
/// assertion cannot be satisfied by an unrelated cause. The no-overlap leg then pins what
/// the refusal buys, on the same inputs that previously produced a fully overlapping
/// family.
#[cfg(not(debug_assertions))]
#[test]
fn r5_release_build_refuses_an_overlapping_train_test_split() {
    use sharpearena::SplitError;

    let unbounded = ScenarioSpec {
        start_level: 100,
        num_levels: 0, // unbounded: the train band is [100, u64::MAX)
        ..ScenarioSpec::default()
    };
    assert_eq!(
        train_test_split(unbounded, 64, 10_000),
        Err(SplitError::UnboundedTrain { start_level: 100 }),
        "an unbounded train band must be refused in the shipped profile"
    );

    // The wrapping sum that could land the test band below the train band is refused
    // in release too, where `+` does not panic.
    let near_max = ScenarioSpec {
        start_level: u64::MAX - 10,
        num_levels: 5,
        ..ScenarioSpec::default()
    };
    assert_eq!(
        train_test_split(near_max, 64, 10_000),
        Err(SplitError::BandOverflow {
            start_level: u64::MAX - 10,
            num_levels: 5,
            gap: 10_000,
        })
    );

    // A bounded band still splits, and the split it returns is genuinely disjoint.
    let bounded = ScenarioSpec {
        start_level: 100,
        num_levels: 50,
        ..ScenarioSpec::default()
    };
    let (train, test) = train_test_split(bounded, 64, 10_000).expect("bounded train splits");
    assert_eq!(test.start_level, 10_150);
    let train_seeds: std::collections::HashSet<u64> =
        (0..50u64).map(|i| level_seed(&train, i)).collect();
    let overlapping = (0..64u64)
        .map(|i| level_seed(&test, i))
        .filter(|s| train_seeds.contains(s))
        .count();
    assert_eq!(overlapping, 0, "no held-out seed may be a train seed");
}
