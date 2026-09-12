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

const BACKTEST = JSON.parse(
  fs.readFileSync(
    path.join(REPO, "crates/sharpearena/contract/attestation/backtest-goldens.json"),
    "utf8",
  ),
);

/**
 * The fingerprint as the backtest goldens record it: a fixed-width 16-hex-digit word.
 * `BigInt.toString(16)` drops leading zeros, so an unpadded comparison is a one-in-sixteen
 * flake against a real fingerprint that happens to start with a zero nibble — which is
 * exactly what `momentum_2x40_seed3` does. Rust reads these with `from_str_radix`, which
 * is indifferent to the padding.
 */
const fingerprint = (text) => fnv1a64(Buffer.from(text, "utf8")).toString(16).padStart(16, "0");

/** Read a committed pre-hash fixture by file name. */
const preHash = (name) =>
  fs.readFileSync(
    path.join(REPO, "crates/sharpearena/contract/attestation/pre-hash", name),
    "utf8",
  );

test("the committed .wasm reproduces every cross-runtime backtest golden", () => {
  // T1: the scenario goldens above pin generation only. Nothing pinned the execution or
  // replay arithmetic across runtimes — the wasm32 test compared one module against
  // itself and the npm smoke suite compared replay against its own output, both of which
  // stay green if native and wasm32 disagree. These entries are the same ones the native
  // suite (`backtest_goldens_reproduce_natively`) and the wasm32 suite
  // (`exported_backtest_goldens_reproduce_under_wasm32`) drive, read from the same file.
  const names = [...BACKTEST.runs, ...BACKTEST.replays].map((e) => e.name);
  for (const required of [
    "momentum_2x40_seed3",
    "fixed_weight_2x40_seed7_costed",
    "long_short_rotation_2x40_seed13_frictionless",
  ]) {
    assert.ok(names.includes(required), `backtest-goldens.json must keep pinning ${required}`);
  }
  assert.ok(BACKTEST.replays.length > 0, "a backtest golden set with no replay entry is the T1 gap");

  for (const entry of BACKTEST.runs) {
    const out = kernel.run_baseline(JSON.stringify(entry.config));
    assert.ok(!out.startsWith('{"error"'), `${entry.name}: the wasm export failed: ${out}`);
    assert.equal(
      out,
      preHash(entry.pre_hash),
      `${entry.name}: the committed .wasm's run bytes drifted from the pre-hash fixture`,
    );
    assert.equal(
      fingerprint(out),
      entry.fnv1a64,
      `${entry.name}: the committed pkg/sharpearena_bg.wasm has drifted from the backtest golden`,
    );
  }

  for (const entry of BACKTEST.replays) {
    // The dataset argument is the kernel's own dataset_synthetic output, passed verbatim,
    // so what is pinned is the replay arithmetic and not a JS re-serialization of the panel.
    const dataset = kernel.dataset_synthetic(JSON.stringify(entry.dataset));
    assert.ok(!dataset.startsWith('{"error"'), `${entry.name}: dataset_synthetic failed: ${dataset}`);
    const out = kernel.replay_run(
      dataset,
      JSON.stringify(entry.trajectory),
      JSON.stringify(entry.costs),
    );
    assert.ok(!out.startsWith('{"error"'), `${entry.name}: the wasm export failed: ${out}`);
    assert.equal(
      out,
      preHash(entry.pre_hash),
      `${entry.name}: the committed .wasm's replay bytes drifted from the pre-hash fixture`,
    );
    assert.equal(
      fingerprint(out),
      entry.fnv1a64,
      `${entry.name}: the committed pkg/sharpearena_bg.wasm has drifted from the backtest golden`,
    );
  }
});

test("every backtest golden is reachable through the wrapper, not only through pkg/", () => {
  const api = require("../dist/index.js");
  for (const entry of BACKTEST.runs) {
    assert.deepEqual(
      api.runBaseline(entry.config),
      JSON.parse(kernel.run_baseline(JSON.stringify(entry.config))),
      `${entry.name}: the wrapper's runBaseline is not the golden export`,
    );
  }
  for (const entry of BACKTEST.replays) {
    const dataset = JSON.parse(kernel.dataset_synthetic(JSON.stringify(entry.dataset)));
    assert.deepEqual(
      api.replayRun(dataset, entry.trajectory, entry.costs),
      JSON.parse(
        kernel.replay_run(
          JSON.stringify(dataset),
          JSON.stringify(entry.trajectory),
          JSON.stringify(entry.costs),
        ),
      ),
      `${entry.name}: the wrapper's replayRun is not the golden export`,
    );
  }
});

const KERNEL = JSON.parse(
  fs.readFileSync(
    path.join(REPO, "crates/sharpearena/contract/attestation/kernel-goldens.json"),
    "utf8",
  ),
);

test("the committed .wasm reproduces every cross-runtime kernel golden", () => {
  // The remainder of T1. The backtest goldens above cover run_baseline and replay_run;
  // walk_forward, stress_suite and tag_regime were named in the same finding and had no
  // committed cross-runtime fixture. smoke.test.js checks their shapes, which stays green
  // while every number behind the shape differs between runtimes. These entries are the
  // same ones the native suite (`kernel_goldens_reproduce_natively`) and the wasm32 suite
  // (`exported_kernel_goldens_reproduce_under_wasm32`) drive, read from the same file.
  const names = [...KERNEL.walk_forward, ...KERNEL.stress_suite, ...KERNEL.tag_regime].map(
    (e) => e.name,
  );
  for (const required of [
    "wf_200d_warmup20_test60_step60",
    "stress_suite_seed0",
    "regime_2x120_seed0_full",
  ]) {
    assert.ok(names.includes(required), `kernel-goldens.json must keep pinning ${required}`);
  }

  const check = (entry, out) => {
    assert.ok(!out.startsWith('{"error"'), `${entry.name}: the wasm export failed: ${out}`);
    assert.equal(
      out,
      preHash(entry.pre_hash),
      `${entry.name}: the committed .wasm's bytes drifted from the pre-hash fixture`,
    );
    assert.equal(
      fingerprint(out),
      entry.fnv1a64,
      `${entry.name}: the committed pkg/sharpearena_bg.wasm has drifted from the kernel golden`,
    );
  };

  for (const entry of KERNEL.walk_forward) {
    check(entry, kernel.walk_forward(JSON.stringify(entry.params)));
  }
  for (const entry of KERNEL.stress_suite) {
    check(entry, kernel.stress_suite(JSON.stringify(entry.params)));
  }
  for (const entry of KERNEL.tag_regime) {
    // The dataset argument is the kernel's own dataset_synthetic output, passed verbatim,
    // so what is pinned is the regime tagging and not a JS re-serialization of the panel.
    const dataset = kernel.dataset_synthetic(JSON.stringify(entry.dataset));
    assert.ok(!dataset.startsWith('{"error"'), `${entry.name}: dataset_synthetic failed: ${dataset}`);
    check(
      entry,
      kernel.tag_regime(`{"dataset":${dataset},"window":${JSON.stringify(entry.window)}}`),
    );
  }
});

test("every kernel golden is reachable through the wrapper, not only through pkg/", () => {
  const api = require("../dist/index.js");
  for (const entry of KERNEL.walk_forward) {
    assert.deepEqual(
      api.walkForward(entry.params),
      JSON.parse(kernel.walk_forward(JSON.stringify(entry.params))),
      `${entry.name}: the wrapper's walkForward is not the golden export`,
    );
  }
  for (const entry of KERNEL.stress_suite) {
    assert.deepEqual(
      api.stressSuite(entry.params.seed),
      JSON.parse(kernel.stress_suite(JSON.stringify(entry.params))),
      `${entry.name}: the wrapper's stressSuite is not the golden export`,
    );
  }
  for (const entry of KERNEL.tag_regime) {
    const dataset = kernel.dataset_synthetic(JSON.stringify(entry.dataset));
    assert.deepEqual(
      api.tagRegime(JSON.parse(dataset), entry.window),
      JSON.parse(
        kernel.tag_regime(`{"dataset":${dataset},"window":${JSON.stringify(entry.window)}}`),
      ).regime,
      `${entry.name}: the wrapper's tagRegime is not the golden export`,
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
