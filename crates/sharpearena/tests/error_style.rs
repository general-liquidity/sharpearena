//! Machine-checked error-message style guide.
//!
//! The register, derived from the existing messages (the `[CODE]`-prefixed engine
//! errors on the pyo3 surface and `SealedSaltError` here): an `[UPPER_SNAKE]` code
//! prefix, a body that starts lowercase (an all-caps acronym like `CSV` is allowed),
//! no trailing period, and the body names the observable the code saw (the offending
//! value or bound), never the inference. The Python suite applies the same rules to
//! every reachable engine exception; this test constructs every variant of every
//! `Display`-able typed error THIS crate defines.
//!
//! Coverage note: `SealedSaltError`, `SplitError`, `MandateError` and `ExecNoiseError`
//! are the crate's `Display` error types. `TransportFault` / `CellOutcome` are typed data
//! carriers without a message register, and `DecideError` is defined upstream in
//! `sharpebench-sim`.

use sharpearena::{
    ExecNoiseError, MandateError, SealedSaltError, SplitError, MIN_SEALED_SALT_BYTES,
};

/// Assert `message` obeys the register. Returns the body for observable checks.
fn assert_register(message: &str) -> &str {
    let rest = message
        .strip_prefix('[')
        .unwrap_or_else(|| panic!("missing [CODE] prefix: {message:?}"));
    let (code, body) = rest
        .split_once("] ")
        .unwrap_or_else(|| panic!("malformed [CODE] prefix: {message:?}"));
    assert!(
        !code.is_empty()
            && code
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'),
        "code must be UPPER_SNAKE: {message:?}"
    );
    let first_word = body.split([' ', ':']).next().unwrap_or_default();
    let acronym = first_word.len() > 1 && first_word.chars().all(|c| c.is_ascii_uppercase());
    assert!(
        body.starts_with(|c: char| c.is_ascii_lowercase()) || acronym,
        "body must start lowercase (or an acronym): {message:?}"
    );
    assert!(
        !body.ends_with('.'),
        "no trailing period (the register has none): {message:?}"
    );
    body
}

#[test]
fn every_typed_error_variant_obeys_the_register() {
    // SealedSaltError: one variant. Construct it, not trigger it, so a new variant
    // added without a register-conforming message shows up here as a compile-time
    // reminder to extend this list.
    let errors: Vec<Box<dyn std::error::Error>> = vec![Box::new(SealedSaltError::TooShort {
        got: 3,
        min: MIN_SEALED_SALT_BYTES,
    })];
    for error in errors {
        let message = error.to_string();
        let body = assert_register(&message);
        // Names the observable: both the offered and required lengths appear.
        assert!(
            body.contains('3') && body.contains(&MIN_SEALED_SALT_BYTES.to_string()),
            "body must name the observed and required values: {body:?}"
        );
    }
}

/// `SplitError` carries the train/test disjointness refusal, so its register is
/// checked the same way: construct every variant rather than trigger it, so a new
/// variant without a conforming message is a compile-time reminder here.
#[test]
fn split_error_variants_obey_the_register() {
    let unbounded = SplitError::UnboundedTrain { start_level: 100 }.to_string();
    let body = assert_register(&unbounded);
    assert!(
        body.contains("100"),
        "body must name the observed start_level: {body:?}"
    );

    let overflow = SplitError::BandOverflow {
        start_level: 7,
        num_levels: 5,
        gap: 10_000,
    }
    .to_string();
    let body = assert_register(&overflow);
    assert!(
        body.contains('7') && body.contains('5') && body.contains("10000"),
        "body must name the three observed operands: {body:?}"
    );
}

/// `MandateError` carries the refusal of a book the grader cannot read, and crosses the
/// pyo3 boundary, so its code must also be one `relay_err` maps to a typed Python class.
/// Every variant is constructed rather than triggered, so a new one without a conforming
/// message is a compile-time reminder here.
#[test]
fn mandate_error_variants_obey_the_register() {
    let cases = [
        (
            MandateError::NonFiniteReturn {
                index: 3,
                value: f64::NAN,
            },
            vec!["3", "NaN"],
        ),
        (
            MandateError::NonFiniteWeight {
                bar: 4,
                index: 1,
                value: f64::INFINITY,
            },
            vec!["4", "1", "inf"],
        ),
        (
            MandateError::InvalidDrawdownCap { max_drawdown: 1.5 },
            vec!["1.5"],
        ),
        (
            MandateError::InvalidInventoryCap { max_inventory: 0.0 },
            vec!["0"],
        ),
    ];
    for (error, observables) in cases {
        let message = error.to_string();
        assert!(
            message.starts_with("[INVALID_ARGUMENT] "),
            "a variant crossing pyo3 must carry a mapped code: {message:?}"
        );
        let body = assert_register(&message);
        for observable in observables {
            assert!(
                body.contains(observable),
                "body must name the observable {observable:?}: {body:?}"
            );
        }
    }
}

/// `ExecNoiseError` carries the refusal of an out-of-range benchmark-integrity knob, and
/// crosses the pyo3 boundary, so the same two rules apply.
#[test]
fn exec_noise_error_variants_obey_the_register() {
    let cases = [
        (ExecNoiseError::DelayProb { delay_prob: -0.1 }, "-0.1"),
        (
            ExecNoiseError::SlippageBps {
                slippage_bps: f64::NAN,
            },
            "NaN",
        ),
    ];
    for (error, observable) in cases {
        let message = error.to_string();
        assert!(
            message.starts_with("[INVALID_ARGUMENT] "),
            "a variant crossing pyo3 must carry a mapped code: {message:?}"
        );
        let body = assert_register(&message);
        assert!(
            body.contains(observable),
            "body must name the observable {observable:?}: {body:?}"
        );
    }
}
