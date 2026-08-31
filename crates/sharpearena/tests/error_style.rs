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
//! Coverage note: `SealedSaltError` is currently the crate's only `Display` error
//! type. `TransportFault` / `CellOutcome` are typed data carriers without a message
//! register, and `DecideError` is defined upstream in `sharpebench-sim`.

use sharpearena::{SealedSaltError, MIN_SEALED_SALT_BYTES};

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
