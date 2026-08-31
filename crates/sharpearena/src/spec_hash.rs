//! The cross-surface tape-semantics fingerprint (see `build.rs` for what is hashed
//! and why FNV-1a). The engine compiles the hash in; each FFI wrapper (npm/TS over
//! the committed wasm, the pure-Python package over the pyo3 extension) records the
//! hash it was generated against and refuses to construct against an engine that
//! reports a different one — a stale surface is diagnosed by name ("engine spec
//! 0x…, wrapper built against 0x…") instead of producing a wrong number.
//!
//! The committed record lives at `contract/attestation/spec-hash.json`; a test below
//! keeps it bound to the compiled value, so changing any spec file forces a visible
//! rebind commit across every surface's pin.

/// The spec hash as the 16-hex-digit string `build.rs` emitted. This is the exact
/// form the wrappers pin and the FFI surfaces return.
pub const SPEC_HASH_HEX: &str = env!("SHARPEARENA_SPEC_HASH");

/// The spec hash as a `u64`, parsed from [`SPEC_HASH_HEX`] at compile time.
pub const SPEC_HASH: u64 = {
    let s = SPEC_HASH_HEX.as_bytes();
    let mut v = 0u64;
    let mut i = 0;
    while i < s.len() {
        let d = match s[i] {
            b'0'..=b'9' => s[i] - b'0',
            b'a'..=b'f' => s[i] - b'a' + 10,
            _ => panic!("SHARPEARENA_SPEC_HASH must be lowercase hex"),
        };
        v = v * 16 + d as u64;
        i += 1;
    }
    v
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed cross-surface record every wrapper pin is derived from.
    const COMMITTED: &str = include_str!("../contract/attestation/spec-hash.json");

    #[test]
    fn hex_and_u64_forms_agree() {
        assert_eq!(SPEC_HASH_HEX.len(), 16);
        assert_eq!(format!("{SPEC_HASH:016x}"), SPEC_HASH_HEX);
    }

    /// The compiled hash must equal the committed record. When a spec file changes
    /// this fails; rebind `contract/attestation/spec-hash.json` AND the wrapper pins
    /// (`npm/sharpearena/src/specHash.ts`, `crates/sharpearena-py/python/sharpearena/
    /// _spec_hash.py`) to the new value, and rebuild the committed wasm bundle.
    #[test]
    fn committed_spec_hash_record_is_current() {
        let doc: serde_json::Value = serde_json::from_str(COMMITTED).unwrap();
        assert_eq!(
            doc["spec_hash"].as_str().expect("record needs spec_hash"),
            SPEC_HASH_HEX,
            "spec files changed: rebind spec-hash.json + the npm/python wrapper pins \
             and rebuild the committed wasm bundle"
        );
        // The recorded file set and epoch must match what build.rs actually hashes,
        // so the record documents the real provenance of the value.
        assert_eq!(
            doc["epoch"].as_str(),
            Some("spec-epoch-1"),
            "epoch drifted from build.rs SPEC_EPOCH"
        );
        let build_rs = include_str!("../build.rs");
        for file in doc["files"].as_array().expect("record needs files") {
            let file = file.as_str().unwrap();
            assert!(
                build_rs.contains(&format!("\"{file}\"")),
                "spec-hash.json lists {file} but build.rs does not hash it"
            );
        }
        assert_eq!(
            doc["files"].as_array().unwrap().len(),
            7,
            "spec-hash.json file count drifted from build.rs SPEC_FILES"
        );
    }
}
