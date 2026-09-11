#!/usr/bin/env node
// The committed npm WASM bundle is the PUBLISHED bundle: check it still answers like the
// bundle this source tree builds.
//
// Before this check, the release workflow deleted npm/sharpearena/pkg and rebuilt it, and
// published the rebuild without running the npm suite over it; CI did not run on tags, so
// no gate ever saw the artifact users receive (A4 in docs/audits/2026-09-09/ARENA-REVIEW.md).
// The release now publishes the committed bundle and runs the full npm suite against those
// exact bytes first, which is what makes the published artifact the tested one. This script
// carries the other half: that the committed bundle has not fallen behind source.
//
// WHAT THIS ASSERTS, AND WHY IT IS NOT BYTE EQUALITY.
//
// The first version of this gate required the rebuild to be byte-identical to the committed
// bundle. That held on the host it was measured on and does not hold in CI: an ubuntu-latest
// runner and a Windows host, on the same pinned rustc 1.96.0 and wasm-pack 0.15.0, produce
// bundles that agree on the spec hash, the crate version and every committed scenario golden
// and still differ in bytes. Byte equality is therefore a property of the build environment,
// not of the artifact, and asserting it made the gate a report about which machine ran it.
//
// What is asserted instead is DIFFERENTIAL BEHAVIORAL EQUIVALENCE: both bundles are loaded
// and driven through every export on a fixed battery of inputs, and their returned JSON is
// compared byte for byte. That is host-independent, and it covers the surface the stamps do
// not: `SPEC_HASH` fingerprints seven tape-defining sources, and `crate_version` moves only
// when the version does, so neither one notices a change in `run_baseline`, `replay_run`,
// `walk_forward`, `stress_suite`, `tag_regime` or the cost model. The battery drives all of
// them. The committed bundle is separately anchored against the committed scenario goldens,
// which is an absolute check rather than a comparison against a peer.
//
// WHAT THIS NO LONGER CATCHES. Two bundles agreeing on every input in the battery are taken
// as equivalent, so a divergence only reachable by an input the battery does not contain
// survives this gate. Byte equality would have caught that and this does not. The battery is
// therefore the gate's real scope: extend it when an export grows a branch. Byte equality is
// still computed and reported, so a bundle that does match is said to match, but a mismatch
// is reported as an environment difference rather than failed.
//
// A failure is classified before it is reported, because a stale committed artifact and a
// build that diverges for another reason produce the same inequality and need different
// remedies.
//
// Usage: node scripts/check-wasm-bundle.mjs [bundle dir]   (from the repository root)
//
// The optional argument names the bundle to check instead of npm/sharpearena/pkg. CI and
// the release workflow pass nothing; it exists so the classification can itself be
// mutation-checked against a known-stale bundle without touching the worktree.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const REPO = process.cwd();
const COMMITTED = path.resolve(REPO, process.argv[2] ?? path.join("npm", "sharpearena", "pkg"));
const require_ = createRequire(import.meta.url);

// wasm-pack writes a `.gitignore` that is never committed, so it is not part of the
// comparison; every other file it emits is.
const IGNORED = new Set([".gitignore"]);

const sha256 = (file) => createHash("sha256").update(readFileSync(file)).digest("hex");

function digests(dir) {
  const out = new Map();
  for (const name of readdirSync(dir).sort()) {
    if (IGNORED.has(name)) continue;
    out.set(name, sha256(path.join(dir, name)));
  }
  return out;
}

function fnv1a64(bytes) {
  let h = 0xcbf29ce484222325n;
  for (const b of bytes) {
    h ^= BigInt(b);
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h;
}

const GOLDENS = JSON.parse(
  readFileSync(
    path.join(REPO, "crates/sharpearena/contract/attestation/scenario-goldens.json"),
    "utf8",
  ),
);

/**
 * The fixed input battery: every export, on inputs chosen to reach the branches that the
 * spec hash and the version stamp do not cover, including the error paths (an unknown
 * agent, an unknown field, an unsatisfiable window), which are part of the surface a
 * consumer sees. Every entry is deterministic and takes no clock, no randomness and no
 * file system. `label` is what a failure names, so it has to say which call diverged.
 */
function battery() {
  const calls = [];
  const push = (label, fn, arg) => calls.push({ label, fn, arg });

  push("spec_hash", "spec_hash", null);
  push("crate_version", "crate_version", null);

  for (const params of [
    "",
    '{"n_symbols":3,"n_days":40,"seed":99}',
    '{"n_symbols":1,"n_days":5,"seed":0}',
    '{"n_symbols":8,"n_days":260,"seed":4242}',
    '{"n_symbols":2,"n_days":40,"bogus_field":1}',
  ]) {
    push(`dataset_synthetic(${params || "<default>"})`, "dataset_synthetic", params);
  }

  // The committed goldens, plus one spec per distribution mode and per opt-in knob family,
  // so a generator change outside the two pinned families is a difference here.
  for (const scenario of GOLDENS.scenarios) {
    push(`generate_scenario(golden ${scenario.name})`, "generate_scenario", JSON.stringify(scenario.input));
  }
  const base = { start_level: 0, num_levels: 0, n_symbols: 3, n_days: 90 };
  for (const mode of ["calm", "hard", "extreme", "cointegrated_pairs", "regime_shift"]) {
    push(`generate_scenario(mode ${mode})`, "generate_scenario", JSON.stringify({ spec: { ...base, distribution_mode: mode }, seed: 13 }));
  }
  push(
    "generate_scenario(richness)",
    "generate_scenario",
    JSON.stringify({
      spec: { ...base, distribution_mode: "calm", obs_richness: { lookback: 3, fundamentals: true, news: true } },
      seed: 5,
    }),
  );
  push(
    "generate_scenario(clustering + jump bursts)",
    "generate_scenario",
    JSON.stringify({
      spec: {
        ...base,
        distribution_mode: "calm",
        vol_clustering: 0.4,
        jump_burst_probability: 0.05,
        jump_burst_persistence: 0.6,
        jump_burst_size: 0.07,
      },
      seed: 21,
    }),
  );
  push("generate_scenario(unknown field)", "generate_scenario", '{"spec":{"start_level":0,"num_levels":0,"n_symbols":2,"n_days":30,"distribution_mode":"calm"},"nope":1}');
  push("generate_scenario(partial spec)", "generate_scenario", '{"spec":{"n_symbols":2},"seed":1}');

  for (const agent of ["buy_and_hold", "hold", "momentum", "random"]) {
    push(`run_baseline(${agent})`, "run_baseline", JSON.stringify({ agent, seed: 3 }));
    push(
      `run_baseline(${agent}, costs)`,
      "run_baseline",
      JSON.stringify({
        agent,
        dataset: { synthetic: { n_symbols: 5, n_days: 150, seed: 17 } },
        window: { start: 20, end: 140 },
        seed: 11,
        costs: { fee_bps: 3, slippage_bps: 7, impact_bps: 2, financing_bps: 1, max_participation: 0.25 },
      }),
    );
  }
  push("run_baseline(momentum lookback)", "run_baseline", JSON.stringify({ agent: "momentum", seed: 3, momentum_lookback: 5 }));
  push(
    "run_baseline(csv)",
    "run_baseline",
    JSON.stringify({
      agent: "buy_and_hold",
      dataset: {
        csv: "date,symbol,close\n2025-01-01,AAA,10\n2025-01-02,AAA,11\n2025-01-03,AAA,12\n2025-01-04,AAA,13",
      },
      window: { start: 1, end: 4 },
      seed: 0,
    }),
  );
  push("run_baseline(unknown agent)", "run_baseline", '{"agent":"nope"}');
  push("run_baseline(unknown field)", "run_baseline", '{"agent":"momentum","seed":3,"momentum_lookbak":5}');

  for (const seed of [0, 1, 2, 7]) {
    push(`stress_suite(${seed})`, "stress_suite", JSON.stringify({ seed }));
  }

  for (const params of [
    { n_days: 200, warmup: 20, test: 60, step: 60 },
    { n_days: 1000, warmup: 50, test: 100, step: 25 },
    { n_days: 10, warmup: 20, test: 60, step: 60 },
    { n_days: 200, warmup: 20, test: 0, step: 60 },
  ]) {
    push(`walk_forward(${JSON.stringify(params)})`, "walk_forward", JSON.stringify(params));
  }

  return calls;
}

/**
 * The two exports that take a whole Dataset as input are driven separately, because the
 * input has to be produced by the bundle under test: feeding both bundles a dataset built
 * by one of them would hide a divergence in the generator behind an identical replay.
 */
function datasetDrivenCalls(kernel) {
  const out = [];
  const dataset = JSON.parse(kernel.dataset_synthetic('{"n_symbols":4,"n_days":120,"seed":11}'));
  const symbol = Object.keys(dataset.closes)[0];

  for (const window of [
    { start: 0, end: 120 },
    { start: 20, end: 60 },
    { start: 100, end: 120 },
  ]) {
    out.push({
      label: `tag_regime(${JSON.stringify(window)})`,
      value: kernel.tag_regime(JSON.stringify({ dataset, window })),
    });
  }

  const steps = Array.from({ length: 10 }, (_, i) => ({
    step: i,
    observation_id: dataset.dates[20 + i],
    decision: {
      orders: [{ symbol, action: "buy", target_weight: 0.25, confidence: 0.6, rationale: "gate" }],
      reasoning: "fixed allocation",
    },
  }));
  const trajectory = { window_start: 20, window_end: 30, seed: 7, steps };
  out.push({
    label: "replay_run(no costs)",
    value: kernel.replay_run(JSON.stringify(dataset), JSON.stringify(trajectory), ""),
  });
  out.push({
    label: "replay_run(costs)",
    value: kernel.replay_run(
      JSON.stringify(dataset),
      JSON.stringify(trajectory),
      '{"fee_bps":5,"slippage_bps":10,"impact_bps":3,"financing_bps":2,"max_participation":0.1}',
    ),
  });
  out.push({
    label: "replay_run(short window)",
    value: kernel.replay_run(
      JSON.stringify(dataset),
      JSON.stringify({ ...trajectory, window_start: 0, window_end: 0, steps: [] }),
      "",
    ),
  });
  return out;
}

/** Every answer a bundle gives, as `label -> returned string`. */
function interrogate(dir) {
  const kernel = require_(path.join(dir, "sharpearena.js"));
  const answers = new Map();
  for (const { label, fn, arg } of battery()) {
    if (typeof kernel[fn] !== "function") {
      answers.set(label, "<export absent>");
      continue;
    }
    answers.set(label, arg === null ? kernel[fn]() : kernel[fn](arg));
  }
  for (const { label, value } of datasetDrivenCalls(kernel)) answers.set(label, value);
  return {
    answers,
    specHash: typeof kernel.spec_hash === "function" ? kernel.spec_hash() : null,
    crateVersion: typeof kernel.crate_version === "function" ? kernel.crate_version() : null,
    goldens: GOLDENS.scenarios.map((s) => ({
      name: s.name,
      ok:
        fnv1a64(Buffer.from(kernel.generate_scenario(JSON.stringify(s.input)), "utf8")).toString(16) ===
        s.fnv1a64,
    })),
  };
}

const name = path.relative(REPO, COMMITTED) || COMMITTED;
const workDir = mkdtempSync(path.join(tmpdir(), "sharpearena-bundle-"));
try {
  const fresh = path.join(workDir, "pkg");
  execFileSync(
    "wasm-pack",
    ["build", "crates/sharpearena-wasm", "--target", "nodejs", "--out-dir", fresh, "--out-name", "sharpearena"],
    { cwd: REPO, stdio: ["ignore", "inherit", "inherit"] },
  );

  const committedDigests = digests(COMMITTED);
  const freshDigests = digests(fresh);
  const names = [...new Set([...committedDigests.keys(), ...freshDigests.keys()])].sort();
  const differingFiles = names.filter((n) => committedDigests.get(n) !== freshDigests.get(n));

  const before = interrogate(COMMITTED);
  const after = interrogate(fresh);

  const stampMismatch = [];
  if (before.specHash !== after.specHash) {
    stampMismatch.push(`spec_hash: committed ${before.specHash} vs built ${after.specHash}`);
  }
  if (before.crateVersion !== after.crateVersion) {
    stampMismatch.push(`crate_version: committed ${before.crateVersion} vs built ${after.crateVersion}`);
  }
  const missedGoldens = before.goldens.filter((g) => !g.ok).map((g) => g.name);
  const divergent = [...after.answers.keys()].filter(
    (label) => before.answers.get(label) !== after.answers.get(label),
  );

  if (stampMismatch.length === 0 && missedGoldens.length === 0 && divergent.length === 0) {
    console.log(`ok   ${name} answers identically to this tree's wasm-pack build`);
    console.log(`     ${after.answers.size} calls across every export, byte-identical returns`);
    console.log(`     spec_hash=${before.specHash} crate_version=${before.crateVersion}`);
    console.log(`     ${before.goldens.length} committed scenario golden(s) reproduced by the committed bundle`);
    if (differingFiles.length === 0) {
      console.log(`     the two bundles are also byte-identical:`);
      for (const n of names) console.log(`       ${committedDigests.get(n)}  ${n}`);
    } else {
      console.log(
        `     note: the two bundles differ in bytes (${differingFiles.join(", ")}) while answering`,
      );
      console.log(
        `     identically. wasm-pack output is not byte-reproducible across operating systems;`,
      );
      console.log(`     that is an environment difference and is reported, not failed.`);
    }
    process.exit(0);
  }

  console.error("");
  console.error(`FAIL ${name} does not answer like the bundle this tree builds`);
  for (const line of stampMismatch) console.error(`     ${line}`);
  for (const golden of missedGoldens) {
    console.error(`     committed golden ${golden}: MISSED by the committed bundle`);
  }
  for (const label of divergent.slice(0, 12)) {
    const a = String(before.answers.get(label) ?? "<absent>");
    const b = String(after.answers.get(label) ?? "<absent>");
    console.error(`     ${label}:`);
    console.error(`       committed ${a.length > 160 ? `${a.slice(0, 160)}... (${a.length} chars)` : a}`);
    console.error(`       built     ${b.length > 160 ? `${b.slice(0, 160)}... (${b.length} chars)` : b}`);
  }
  if (divergent.length > 12) console.error(`     ... and ${divergent.length - 12} more divergent call(s)`);
  console.error("");
  if (stampMismatch.length > 0 || missedGoldens.length > 0) {
    console.error("     diagnosis: THE COMMITTED BUNDLE IS STALE. It reports a different revision");
    console.error("     than this source tree builds, or fails a committed golden, so it is the");
    console.error("     artifact that is wrong. Rebuild and commit it:");
    console.error("       wasm-pack build crates/sharpearena-wasm --target nodejs \\");
    console.error("         --out-dir ../../npm/sharpearena/pkg --out-name sharpearena");
  } else {
    console.error("     diagnosis: THE COMMITTED BUNDLE ANSWERS DIFFERENTLY while reporting the");
    console.error("     same spec hash and crate version and reproducing every committed golden.");
    console.error("     The spec hash covers the seven tape-defining sources only, so a change in");
    console.error("     the export layer, the baselines, the replay path or the cost model lands");
    console.error("     here and nowhere else. Rebuild and commit the bundle; if the committed");
    console.error("     bundle is the intended one, this source tree is what moved.");
  }
  process.exit(1);
} finally {
  rmSync(workDir, { recursive: true, force: true, maxRetries: 5 });
}
