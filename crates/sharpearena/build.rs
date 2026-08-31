//! Emit `SHARPEARENA_SPEC_HASH`: a fingerprint of the source files that define the
//! tape semantics — what bytes a generated scenario, a cleared bar, a matched fill
//! tape, an execution perturbation, or a sampled mandate contains for a given
//! `(spec, seed)`. The three published surfaces (the crates.io native crate, the
//! committed wasm/npm bundle, the maturin-built pyo3 wheel) agree today only by CI
//! discipline; this constant makes the agreement structural. Each FFI wrapper records
//! the hash it was generated against and compares it to the engine's at construction,
//! so a stale bundle driven by a newer spec fails loudly by name instead of computing
//! a silently wrong number.
//!
//! FNV-1a (not `DefaultHasher`) because the value must reproduce across toolchains
//! and platforms: a wrapper pinned on Linux CI and an engine built on a Windows
//! workstation must agree. For the same reason every hashed byte stream is normalized
//! by dropping `\r` before hashing (this repo has CRLF churn history; `.gitattributes`
//! pins LF, but an editor-saved CRLF working tree must still hash identically to the
//! LF checkout the wrapper pins were computed from).
//!
//! What is deliberately NOT hashed: `curriculum.rs` (seed-selection policy, not tape
//! bytes), `leaderboard_ci.rs` (post-hoc statistics), `vec_env.rs` (batching over the
//! same per-step body), `transport_gate.rs` (fault classification), `lib.rs`
//! (re-exports), and the `sharpebench-sim` engine dependency (out of tree). A
//! semantics change arriving through a dependency bump or any other path this file
//! set misses is what the manual `SPEC_EPOCH` escape hatch is for.

/// The tape-semantics-defining sources. Keep alphabetical; adding a file here moves
/// the spec hash, which is the point.
const SPEC_FILES: [&str; 7] = [
    "src/contract.rs",
    "src/exec_noise.rs",
    "src/lob_market.rs",
    "src/mandate.rs",
    "src/market.rs",
    "src/richness.rs",
    "src/scenario_gen.rs",
];

/// Manual epoch: bump to force a new spec hash for a semantics change the file set
/// cannot see (an engine-dependency bump, a data-layer change in `sharpebench-sim`).
const SPEC_EPOCH: &[u8] = b"spec-epoch-1";

fn fnv1a(seed: u64, bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = seed;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() {
    let mut h = fnv1a(0xcbf2_9ce4_8422_2325, SPEC_EPOCH.iter().copied());
    for f in SPEC_FILES {
        println!("cargo:rerun-if-changed={f}");
        let bytes = std::fs::read(f)
            .unwrap_or_else(|e| panic!("spec-hash input {f} must exist and be readable: {e}"));
        // CRLF -> LF normalization: drop every carriage return before hashing.
        h = fnv1a(h, bytes.into_iter().filter(|&b| b != b'\r'));
    }
    println!("cargo:rustc-env=SHARPEARENA_SPEC_HASH={h:016x}");
}
