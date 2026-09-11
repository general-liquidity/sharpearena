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
export const SPEC_HASH = "5518afd039aa5317";

/**
 * Compare an engine-reported spec hash against **this wrapper's pin**, throwing the
 * named mismatch diagnosis on disagreement. `engineHash === undefined` means the
 * loaded engine predates the handshake entirely, the one frame of this exchange
 * that is decoded leniently, precisely so a stale surface is diagnosed by name
 * rather than dropped as malformed.
 *
 * The pin is deliberately not a parameter. An exported `check(a, b)` reads as
 * "compare these two", not "compare against the pin", and `check(h, h)` passes for
 * any `h`. This is the only spec-hash entry point the package exposes, so a caller
 * cannot supply the value it is being checked against.
 */
export function checkSpecHash(engineHash: string | undefined): void {
  compareSpecHash(engineHash, SPEC_HASH);
}

/**
 * The two-sided comparison behind {@link checkSpecHash}. Internal: it is not
 * re-exported from the package entry point, because a caller-supplied `wrapperHash`
 * turns the handshake into a tautology. The stale-surface regression drives this
 * function directly, which is why it is named rather than inlined.
 */
export function compareSpecHash(
  engineHash: string | undefined,
  wrapperHash: string,
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
