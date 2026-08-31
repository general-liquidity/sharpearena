/**
 * The cross-surface SPEC_HASH handshake gate for the published npm artifact.
 *
 * Three pins must agree: the committed spec-hash record
 * (crates/sharpearena/contract/attestation/spec-hash.json), the wrapper's compiled-in
 * pin (src/specHash.ts), and what the committed pkg/sharpearena_bg.wasm actually
 * reports. A committed bundle rebuilt from a different tape-semantics revision than
 * this wrapper fails here (and at wrapper load) by name instead of shipping.
 */
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const kernel = require("../pkg/sharpearena.js");

const REPO = path.resolve(__dirname, "../../..");
const RECORD = JSON.parse(
  fs.readFileSync(
    path.join(REPO, "crates/sharpearena/contract/attestation/spec-hash.json"),
    "utf8",
  ),
);

/** The wrapper pin, read from the TS source (tests run without a tsc build). */
function wrapperPin() {
  const src = fs.readFileSync(path.join(__dirname, "../src/specHash.ts"), "utf8");
  const match = /export const SPEC_HASH = "([0-9a-f]{16})";/.exec(src);
  assert.ok(match, "src/specHash.ts must pin a 16-hex-digit SPEC_HASH");
  return match[1];
}

test("the committed .wasm, the wrapper pin, and the attestation record agree", () => {
  assert.equal(
    typeof kernel.spec_hash,
    "function",
    "the committed pkg/ predates the spec_hash export; rebuild it",
  );
  assert.equal(
    kernel.spec_hash(),
    RECORD.spec_hash,
    "the committed pkg/sharpearena_bg.wasm reports a different spec hash than the attestation record",
  );
  assert.equal(
    wrapperPin(),
    RECORD.spec_hash,
    "src/specHash.ts pin drifted from the attestation record",
  );
});

test("the wrapper entry point runs the handshake at load", () => {
  // The refusal helper only protects consumers if index.ts actually invokes it at
  // module load (tests drive pkg/ directly, so the wiring needs its own pin).
  const src = fs.readFileSync(path.join(__dirname, "../src/index.ts"), "utf8");
  assert.match(src, /checkSpecHash\(/, "index.ts must invoke checkSpecHash at load");
});

test("a stale wrapper is refused with the named mismatch diagnosis", () => {
  // Simulate the stale-surface pairing: a wrapper pinned on an old hash driving the
  // current engine. Transpile the real specHash.ts (the same code index.ts runs at
  // load) and drive its checkSpecHash directly, without the module-level handshake.
  const ts = require("typescript");
  const src = fs.readFileSync(path.join(__dirname, "../src/specHash.ts"), "utf8");
  const js = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText;
  const mod = { exports: {} };
  new Function("exports", "module", "require", js)(mod.exports, mod, require);
  const { checkSpecHash } = mod.exports;
  assert.equal(typeof checkSpecHash, "function");

  const stale = "00000000deadbeef";
  assert.throws(
    () => checkSpecHash(kernel.spec_hash(), stale),
    (err) =>
      err.message.includes("SpecHashMismatch") &&
      err.message.includes(`engine spec 0x${kernel.spec_hash()}`) &&
      err.message.includes(`wrapper built against 0x${stale}`),
    "a mismatch must name both hashes",
  );

  // The lenient leg: an engine with no spec_hash export at all is diagnosed by name.
  assert.throws(
    () => checkSpecHash(undefined, stale),
    (err) =>
      err.message.includes("SpecHashMismatch") &&
      err.message.includes("predates the SPEC_HASH handshake"),
    "a pre-handshake engine must be named, not dropped as malformed",
  );

  // And the matched pair passes.
  checkSpecHash(kernel.spec_hash(), kernel.spec_hash());
});
