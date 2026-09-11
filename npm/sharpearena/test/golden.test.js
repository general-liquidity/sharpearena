/**
 * The published-artifact gate.
 *
 * `npm test` is the only gate that touches the .wasm binary this package actually ships.
 * The Rust suite asserts the goldens against a host-compiled kernel and `wasm-pack test`
 * asserts them against a freshly built wasm32 module; neither one reads
 * `pkg/sharpearena_bg.wasm`. These tests do, so a committed binary that has fallen behind
 * the engine fails here instead of being published.
 *
 * The pins are read from the same committed contract file the Rust and wasm32 tests read,
 * so a generator change cannot be absorbed by editing one runtime's copy of the numbers.
 */
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const kernel = require("../pkg/sharpearena.js");

const REPO = path.resolve(__dirname, "../../..");
const GOLDENS = JSON.parse(
  fs.readFileSync(
    path.join(REPO, "crates/sharpearena/contract/attestation/scenario-goldens.json"),
    "utf8",
  ),
);

/** Dependency-free FNV-1a/64, the fingerprint the goldens are recorded in. */
function fnv1a64(bytes) {
  let h = 0xcbf29ce484222325n;
  for (const b of bytes) {
    h ^= BigInt(b);
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h;
}

test("the committed .wasm reproduces every cross-runtime scenario golden", () => {
  const names = GOLDENS.scenarios.map((s) => s.name);
  for (const required of ["calm_4x120_seed7", "hard_clustered_4x120_seed7"]) {
    assert.ok(names.includes(required), `scenario-goldens.json must keep pinning ${required}`);
  }

  // Committed pre-hash canonical JSON per golden name: compared BEFORE the fingerprint,
  // so a canonicalization regression fails with a readable diff, not two hex numbers.
  const PRE_HASH_FIXTURES = {
    calm_4x120_seed7: "scenario-calm-4x120-seed7.json",
    hard_clustered_4x120_seed7: "scenario-hard-clustered-4x120-seed7.json",
  };

  for (const scenario of GOLDENS.scenarios) {
    const out = kernel.generate_scenario(JSON.stringify(scenario.input));
    assert.ok(!out.startsWith('{"error"'), `${scenario.name}: the wasm export failed: ${out}`);
    const fixture = PRE_HASH_FIXTURES[scenario.name];
    assert.ok(fixture, `${scenario.name}: golden has no committed pre-hash fixture`);
    const expected = fs.readFileSync(
      path.join(REPO, "crates/sharpearena/contract/attestation/pre-hash", fixture),
      "utf8",
    );
    assert.equal(
      out,
      expected,
      `${scenario.name}: the committed .wasm's bytes drifted from the pre-hash fixture`,
    );
    assert.equal(
      fnv1a64(Buffer.from(out, "utf8")).toString(16),
      scenario.fnv1a64,
      `${scenario.name}: the committed pkg/sharpearena_bg.wasm has drifted from the golden`,
    );
  }
});

test("every golden is reachable through the wrapper, not only through pkg/", () => {
  // A3: `generate_scenario` used to be the one export `src/index.ts` did not wrap, so
  // the function the cross-runtime byte-identity claim is written about was reachable
  // only by the deep import that skips the spec-hash handshake. Driving the goldens
  // through the public API pins that it is no longer.
  //
  // The fingerprint is asserted on the kernel bytes above; a JS re-serialization of the
  // parsed value is not those bytes (`100.0` round-trips to `100`), so this leg pins the
  // wrapper to the kernel's own decoded output for the same input rather than re-hashing.
  const api = require("../dist/index.js");
  for (const scenario of GOLDENS.scenarios) {
    assert.deepEqual(
      api.generateScenario(scenario.input),
      JSON.parse(kernel.generate_scenario(JSON.stringify(scenario.input))),
      `${scenario.name}: the wrapper's generateScenario is not the golden export`,
    );
  }
});

test("the shipped wasm package carries the crate version", () => {
  const wrapper = JSON.parse(fs.readFileSync(path.join(__dirname, "../package.json"), "utf8"));
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, "../pkg/package.json"), "utf8"));
  const cargo = fs.readFileSync(path.join(REPO, "Cargo.toml"), "utf8");
  const workspaceVersion = /\[workspace\.package\][^[]*?\bversion\s*=\s*"([^"]+)"/s.exec(cargo);

  assert.ok(workspaceVersion, "could not read [workspace.package] version from Cargo.toml");

  // The binary's own stamp, first. The three text comparisons below are three manifests
  // agreeing with each other; none of them reads the .wasm, so before `crate_version()`
  // existed a stale binary beside a current pkg/package.json passed this test. See A4 in
  // docs/audits/2026-09-09/ARENA-REVIEW.md.
  assert.equal(
    typeof kernel.crate_version,
    "function",
    "the committed pkg/ predates the crate_version stamp; rebuild it with wasm-pack",
  );
  assert.equal(
    kernel.crate_version(),
    workspaceVersion[1],
    "the committed pkg/sharpearena_bg.wasm was built from a different crate version than the workspace is at; rebuild pkg/",
  );

  assert.equal(
    pkg.version,
    workspaceVersion[1],
    "pkg/package.json is the wasm-pack output; its version must be the crate version it was built from",
  );
  assert.equal(
    wrapper.version,
    workspaceVersion[1],
    "the npm wrapper version must track the crate version it wraps",
  );
});
