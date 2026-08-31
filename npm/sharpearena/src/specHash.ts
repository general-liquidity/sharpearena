/**
 * The tape-semantics spec hash this wrapper (and its committed `pkg/` wasm bundle)
 * was generated against — the value in
 * `crates/sharpearena/contract/attestation/spec-hash.json`.
 *
 * At module load, `index.ts` compares this pin to the hash the wasm engine itself
 * reports (`spec_hash()`, compiled in by the core crate's `build.rs` over the
 * tape-defining sources) and refuses to run on a mismatch. That makes "a wasm
 * bundle built from commit A driven by a wrapper from commit B" a named error
 * instead of a silently wrong number. Rebind this constant (and rebuild `pkg/`)
 * whenever the committed spec-hash record moves.
 */
export const SPEC_HASH = "2d912913eedd333e";

/**
 * Compare the engine-reported spec hash against the wrapper's pin, throwing the
 * named mismatch diagnosis on disagreement. `engineHash === undefined` means the
 * loaded engine predates the handshake entirely — the one frame of this exchange
 * that is decoded leniently, precisely so a stale surface is diagnosed by name
 * rather than dropped as malformed.
 */
export function checkSpecHash(
  engineHash: string | undefined,
  wrapperHash: string = SPEC_HASH,
): void {
  if (engineHash === wrapperHash) return;
  if (engineHash === undefined) {
    throw new Error(
      `SpecHashMismatch: the loaded wasm engine predates the SPEC_HASH handshake ` +
        `(no spec_hash export); wrapper built against 0x${wrapperHash}. ` +
        `Rebuild pkg/ from the same commit as this wrapper.`,
    );
  }
  throw new Error(
    `SpecHashMismatch: engine spec 0x${engineHash}, wrapper built against ` +
      `0x${wrapperHash}. The committed wasm bundle and this wrapper come from ` +
      `different tape-semantics revisions; rebuild pkg/ from the same commit.`,
  );
}
