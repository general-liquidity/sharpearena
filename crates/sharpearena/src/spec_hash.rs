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
        // The recorded inputs and epoch must equal what build.rs actually folded into
        // the value. Both come from the build script itself: `SHARPEARENA_SPEC_INPUTS`
        // is accumulated by the hashing loop as each input is hashed, and
        // `SHARPEARENA_SPEC_EPOCH` is the `SPEC_EPOCH` bytes that seed it. Searching
        // build.rs's source text instead would accept a name that appears only in a
        // comment, and a hardcoded count does not move with the array it claims to
        // track; comparing the declared list to the hashed list, in order, does.
        assert_eq!(
            doc["epoch"].as_str(),
            Some(env!("SHARPEARENA_SPEC_EPOCH")),
            "spec-hash.json epoch drifted from the epoch build.rs seeded the hash with"
        );
        let recorded: Vec<&str> = doc["files"]
            .as_array()
            .expect("record needs files")
            .iter()
            .map(|file| file.as_str().expect("each recorded input is a string"))
            .collect();
        let hashed: Vec<&str> = env!("SHARPEARENA_SPEC_INPUTS").split(',').collect();
        assert_eq!(
            recorded, hashed,
            "spec-hash.json lists inputs build.rs does not hash, in that order"
        );
    }

    #[test]
    fn suite_dependencies_that_feed_tape_semantics_are_exact_pinned() {
        let manifest = include_str!("../Cargo.toml");
        let manifest: toml::Value = manifest.parse().unwrap();
        for dependency in [
            "sharpebench-sim",
            "sharpebench-protocol",
            "sharpebench-core",
            "sharpebench-attest",
        ] {
            let entry = manifest
                .get("dependencies")
                .and_then(|table| table.get(dependency))
                .or_else(|| {
                    manifest
                        .get("dev-dependencies")
                        .and_then(|table| table.get(dependency))
                })
                .unwrap_or_else(|| panic!("missing {dependency}"));
            let requirement = entry
                .as_str()
                .or_else(|| entry.get("version").and_then(toml::Value::as_str))
                .unwrap();
            assert!(
                requirement.starts_with('='),
                "{dependency} must be exact-pinned because the manifest's dependency contract, not the resolved registry \
                 source, is the dependency input bound into SPEC_HASH; found {requirement}"
            );
        }
    }
}
