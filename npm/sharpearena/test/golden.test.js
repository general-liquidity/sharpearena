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

  for (const scenario of GOLDENS.scenarios) {
    const out = kernel.generate_scenario(JSON.stringify(scenario.input));
    assert.ok(!out.startsWith('{"error"'), `${scenario.name}: the wasm export failed: ${out}`);
    assert.equal(
      fnv1a64(Buffer.from(out, "utf8")).toString(16),
      scenario.fnv1a64,
      `${scenario.name}: the committed pkg/sharpearena_bg.wasm has drifted from the golden`,
    );
  }
});

test("the shipped wasm package carries the crate version", () => {
  const wrapper = JSON.parse(fs.readFileSync(path.join(__dirname, "../package.json"), "utf8"));
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, "../pkg/package.json"), "utf8"));
  const cargo = fs.readFileSync(path.join(REPO, "Cargo.toml"), "utf8");
  const workspaceVersion = /\[workspace\.package\][^[]*?\bversion\s*=\s*"([^"]+)"/s.exec(cargo);

  assert.ok(workspaceVersion, "could not read [workspace.package] version from Cargo.toml");
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
