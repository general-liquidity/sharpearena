# smolvm assessment for SharpeArena

This is the engineering ledger for the local smolvm snapshot reviewed while
hardening SharpeArena. It records what was transferable, what was implemented,
and what was deliberately rejected. It is not a claim that SharpeArena embeds or
depends on smolvm.

## Scope of the review

The local snapshot identifies itself as smolvm 1.13.1 and contains 492 files:
222 Rust files (about 188,000 lines), 72 shell files (about 17,600 lines), 33
Markdown files, plus TypeScript, Python, Nix, packaging, deployment, and release
assets. The review covered all top-level product slices:

- 109 files in `src/`;
- 131 files across the 15 workspace crates;
- 42 integration-test files and 23 benchmark files;
- 45 SDK files; and
- build, packaging, deployment, Nix, examples, demo, and repository metadata.

The snapshot is an exported tree without Git metadata. Coordinates below are
therefore `path:line` references into that snapshot, not commit-stable upstream
links.

## Why smolvm has so much at the repository root

smolvm is a multi-deliverable infrastructure monorepo, not one library with a
large source directory. Its root separates the host CLI/runtime (`src/`), 15
internal crates (`crates/`), real-VM acceptance suites (`tests/`), benchmarks,
examples, demos, deployment manifests, OS packaging, Nix expressions, language
SDKs, and release tooling.

Three apparently duplicated roots are Git submodules:

| Root | Role |
|---|---|
| `libkrun/` | The virtual-machine monitor used to enter and run the guest. |
| `libkrunfw/` | Firmware plus the guest Linux kernel consumed by libkrun. |
| `smolvm-sdk/` | A separately versioned SDK repository. |

`sdks/` contains language wrappers that are developed in the monorepo, while
`lib/` contains prebuilt cross-platform runtime artifacts tracked through Git
LFS. In this local export the three submodule directories are empty and the
dynamic libraries are LFS pointer files, so the source can be reviewed but a VM
cannot be booted from this snapshot.

This physical organization would not improve SharpeArena. The useful part to
copy is smolvm's task-first README order—install, quick start, use cases,
architecture, security model, limitations—not its root folder count.

## Adopted ideas

| smolvm evidence | SharpeArena extraction | Correctness assessment |
|---|---|---|
| `bench/bench.sh:146-205` reads the daemon's effective share mode and rejects a mislabeled benchmark arm. | `effective_config.py`, native `TradingEnv.effective_config`, and every scenario-producing evidence script compare requested configuration with values read from the environment that consumed it. | Correct. Dimensions/window/seed come from the native instance; scenario knobs are independently regenerated and fingerprinted rather than echoed from the driver. The implementation documents what that independent path still cannot detect. |
| `crates/smolvm-cuda/build.rs:8-37` computes a deterministic FNV-1a hash over the files defining its wire protocol. | `crates/sharpearena/build.rs`, `spec_hash.rs`, and pins in Python/npm/WASM expose and check `SPEC_HASH`. | Strengthened. Inputs are sorted and length-framed, CRLF is normalized without collapsing lone CR bytes, `Cargo.toml` and exact SharpeBench engine pins are included, and stale or pre-handshake wrappers refuse at construction. A committed record and cross-surface tests pin the value. |
| `src/secrets.rs:502-544` wraps secret text, redacts `Debug`, and avoids casual serialization/display. | `SealedSalt` centralizes the 16-byte floor, exposes no `Serialize`, `Display`, or `Deref`, redacts `Debug`, and scrubs its buffer on drop. | Appropriate. The published Rust API can no longer bypass the entropy floor. The scrub is explicitly best-effort because the crate forbids unsafe code and did not add a crypto dependency merely to promise a volatile erase. Seed derivation and historical goldens are unchanged. |
| `src/cli/cleanup_ephemeral.rs:52-128` separates cleanup ordering from effects and retains registration if deletion fails. | Release cleanup is a pure, injected sequence; path deletion is guarded by a non-empty, strictly-inside-root, non-root, non-symlink predicate. | Correct for the release worktree problem. Failure preserves recoverable state instead of forgetting a directory that still exists. |
| `src/artifact_cache.rs:159-181` uses exclusive create, file sync, rename, and parent-directory sync. | `paper/src/make-provenance.py` atomically publishes the manifest in its destination directory and syncs the file and parent where supported. | Correct. The checker sees either the old complete manifest or the new complete manifest, never an intended partial file. |
| `crates/smolvm-smolfile/src/lib.rs` and API inputs use `deny_unknown_fields`. | Closed caller-input structs reject unknown scenario, mandate, market, and WASM configuration fields. Engine output structs remain forward-open. | Correctly scoped. A typo in a safety/config input fails; an older consumer can still read a newer output with additive fields. |
| `tests/test_network.sh:22-64` distinguishes an immediate policy denial from a timeout and asserts the probe client actually ran. | Generated-fixture and negative-control tests require non-empty inputs and name the false-pass they exclude. | Correctly generalized. SharpeArena has no container egress boundary of its own, so importing the network test literally would test a property the product does not provide. |
| `crates/smolvm-pack/src/extract.rs` and asset tests reject traversal/symlink escape. | Plan-embedded `csv_path` values are contained inside the plan directory; absolute paths, `..`, and escaping symlinks are refused. A unique-needle scan protects sealed-seed artifacts. | Correct and found a real read primitive, not a hypothetical test. It prevents a shared plan from incorporating an arbitrary host file into evidence. |
| Error conventions in `src/data/errors.rs` and protocol fallbacks keep machine codes stable while preserving unknown values. | The pyo3 boundary maps stable `[CODE]` failures to typed Python exceptions; unknown codes degrade to the base exception with the code intact. `tests/error_style.rs` checks the register. | Correct compatibility behavior. Codes can be added but should not be renamed. Existing `ValueError` handlers continue to catch the base type. |
| smolvm packaging and SDK scripts motivate consumer-side artifact tests. | CI packs the npm tarball and Python wheel, installs each into a clean offline environment, imports it, and exercises a public symbol; the wheel includes `py.typed` and a drift-checked native stub. | Stronger than a repository-local import. It tests the artifact a registry consumer receives. |
| smolvm release and cleanup code isolates mutable operations from the developer tree. | `scripts/release.py` rehearses and executes from a freshly fetched throwaway worktree, refuses local-only or empty changelog notes, rebinds provenance, validates an annotated tag, and atomically pushes branch plus tag. | Correctly adapted to SharpeArena's provenance cycle; it does not copy smolvm's non-atomic two-push release behavior. |
| Deterministic fixtures and compatibility tests in the pack/config code retain readable pre-hash material. | Every scenario golden is backed by committed canonical JSON before the compact fingerprint assertion. | Correct. A drift failure produces a readable structural diff instead of only two opaque hashes. |

## Existing equivalents that did not need another port

- SharpeArena already had deterministic raw-decision replay, cross-runtime
  goldens, exact package-version checks, a source/artifact provenance manifest,
  and a language-neutral schema. smolvm did not justify parallel mechanisms.
- SharpeBench's shared protocol provides additive compatibility and schema drift
  tests; the `SPEC_HASH` supplements it for tape semantics instead of replacing
  the public contract version.
- The committed canonical golden JSON is stronger for diagnosis than adding a
  second cryptographic hash over the same bytes.

## Deliberately not integrated

| smolvm subsystem | Decision |
|---|---|
| libkrun/libkrunfw, microVM launch, snapshots, forkpoints, and per-episode VMs | Rejected. SharpeArena is a deterministic market environment, not a containment runtime. Per-episode VM startup would dominate large seeded fields, and the product explicitly delegates untrusted entrant containment to SharpeBench. |
| `.smolmachine` | Rejected as a research-evidence format. Its compatible footer remains CRC32-based; current smolvm adds a full-artifact SHA-256 marker for shared extraction, but that cache identity is not a source/provenance manifest. SharpeArena's exact source and artifact digests are the better fit. |
| CUDA/NVML remoting, GPU shims, VNC/framebuffer, S3/FUSE, OCI registry/cache, Kubernetes/containerd, fleet admission, and cloud control-plane code | Not relevant to the environment or its published interfaces. Adding them would enlarge the trusted computing base without strengthening a stated SharpeArena property. |
| smolvm binaries or distribution artifacts | Rejected. SharpeArena does not need them, and redistribution would inherit libkrunfw/LGPL and bundled GPL-kernel obligations described in smolvm's `Licenses.md`. |
| Node/Python SDK implementation | Used only as interface-design prior art. The local Node build references a missing `crates/smolvm-napi`, and neither SDK is exercised in the main CI. SharpeArena's published wrappers and pack/install tests are stronger evidence. |
| Artifact cache eviction, registry retry, and P2P distribution | Conditional only. Revisit if SharpeArena later owns a shared model/dataset blob cache; there is no such cache to harden today. |

## Conclusion

All smolvm ideas that materially strengthen SharpeArena's current claims have an
implemented equivalent, and each was translated to the product's actual boundary
rather than copied wholesale. The integration improved determinism,
cross-surface compatibility, evidence integrity, input safety, packaging, and
release isolation. It did **not** turn SharpeArena into a process sandbox, and it
would be inaccurate to say otherwise.
