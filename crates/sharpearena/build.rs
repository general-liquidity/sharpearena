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
//! workstation must agree. For the same reason CRLF source line endings are normalized
//! to LF before hashing (this repo has CRLF churn history; `.gitattributes` pins LF,
//! but an editor-saved Windows working tree must still agree with CI). A lone carriage
//! return is retained because it is source content, not a line-ending encoding.
//!
//! Each file is framed as `name NUL canonical_length_le64 canonical_bytes`. Hashing
//! names and lengths makes the file boundary part of the fingerprint: moving bytes
//! between adjacent inputs cannot preserve the stream by accident.
//!
//! The crate manifest IS hashed because the Calm generator starts in the published
//! `sharpebench-sim::Dataset::synthetic` implementation. Those suite dependencies are
//! exact-pinned: a dependency upgrade edits the manifest and therefore moves the hash
//! automatically instead of relying on someone to remember the manual epoch.
//!
//! What is deliberately NOT hashed: `curriculum.rs` (seed-selection policy, not tape
//! bytes), `leaderboard_ci.rs` (post-hoc statistics), `vec_env.rs` (batching over the
//! same per-step body), `transport_gate.rs` (fault classification), `lib.rs`
//! (re-exports). A semantics change arriving through any path this file set and the
//! exact dependency pins miss is what the manual `SPEC_EPOCH` escape hatch is for.

/// The tape-semantics-defining sources. Keep alphabetical; adding a file here moves
/// the spec hash, which is the point.
const SPEC_FILES: [&str; 8] = [
    "Cargo.toml",
    "src/contract.rs",
    "src/exec_noise.rs",
    "src/lob_market.rs",
    "src/mandate.rs",
    "src/market.rs",
    "src/richness.rs",
    "src/scenario_gen.rs",
];

/// Manual epoch: bump to force a new spec hash for a semantics change the file set
/// and the exact suite-dependency pins cannot see (for example, a toolchain codegen
/// property that becomes part of the wire contract).
const SPEC_EPOCH: &[u8] = b"spec-epoch-1";

fn fnv1a(seed: u64, bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = seed;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn canonical_source_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut canonical = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            canonical.push(b'\n');
            index += 2;
        } else {
            canonical.push(bytes[index]);
            index += 1;
        }
    }
    canonical
}

fn main() {
    let mut h = fnv1a(0xcbf2_9ce4_8422_2325, SPEC_EPOCH.iter().copied());
    for f in SPEC_FILES {
        println!("cargo:rerun-if-changed={f}");
        let bytes = std::fs::read(f)
            .unwrap_or_else(|e| panic!("spec-hash input {f} must exist and be readable: {e}"));
        let canonical = canonical_source_bytes(&bytes);
        h = fnv1a(h, f.bytes().chain([0]));
        h = fnv1a(h, (canonical.len() as u64).to_le_bytes());
        h = fnv1a(h, canonical);
    }
    println!("cargo:rustc-env=SHARPEARENA_SPEC_HASH={h:016x}");
}
