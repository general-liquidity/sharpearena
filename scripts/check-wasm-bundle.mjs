#!/usr/bin/env node
// The committed npm WASM bundle is the PUBLISHED bundle: check it is what this source
// tree builds.
//
// Before this check, the release workflow deleted npm/sharpearena/pkg and rebuilt it, and
// published the rebuild without running the npm suite over it; CI did not run on tags, so
// no gate ever saw the artifact users receive (A4 in docs/audits/2026-09-09/ARENA-REVIEW.md).
// The release now publishes the committed bundle, which CI does test, and this script is
// what keeps that bundle from falling behind source.
//
// A bare "the bytes differ" is not a diagnosis: a stale committed bundle and a build that
// is not reproducible on this runner produce the same mismatch. So a difference is
// classified before it is reported, by driving both bundles:
//
//   - a different spec_hash or crate_version, or a golden the committed bundle misses,
//     means the COMMITTED BUNDLE IS STALE: rebuild and commit it;
//   - identical stamps and identical golden output with different bytes means the build
//     IS NOT BYTE-REPRODUCIBLE here: a toolchain or environment difference, not a stale
//     artifact.
//
// Both fail, because either one breaks the claim that the published bundle is the tested
// one; only the remedy differs.
//
// Usage: node scripts/check-wasm-bundle.mjs [bundle dir]   (from the repository root)
//
// The optional argument names the bundle to check instead of npm/sharpearena/pkg. CI and
// the release workflow pass nothing; it exists so the classification above can itself be
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

/** Everything a bundle can say about which revision it is. */
function interrogate(dir) {
  const kernel = require_(path.join(dir, "sharpearena.js"));
  const goldens = JSON.parse(
    readFileSync(
      path.join(REPO, "crates/sharpearena/contract/attestation/scenario-goldens.json"),
      "utf8",
    ),
  );
  return {
    specHash: typeof kernel.spec_hash === "function" ? kernel.spec_hash() : null,
    crateVersion: typeof kernel.crate_version === "function" ? kernel.crate_version() : null,
    goldens: goldens.scenarios.map((s) => ({
      name: s.name,
      ok: fnv1a64(Buffer.from(kernel.generate_scenario(JSON.stringify(s.input)), "utf8"))
        .toString(16) === s.fnv1a64,
    })),
  };
}

const workDir = mkdtempSync(path.join(tmpdir(), "sharpearena-bundle-"));
try {
  const fresh = path.join(workDir, "pkg");
  execFileSync(
    "wasm-pack",
    [
      "build",
      "crates/sharpearena-wasm",
      "--target",
      "nodejs",
      "--out-dir",
      fresh,
      "--out-name",
      "sharpearena",
    ],
    { cwd: REPO, stdio: ["ignore", "inherit", "inherit"] },
  );

  const committedDigests = digests(COMMITTED);
  const freshDigests = digests(fresh);

  const names = [...new Set([...committedDigests.keys(), ...freshDigests.keys()])].sort();
  const differing = names.filter((n) => committedDigests.get(n) !== freshDigests.get(n));

  if (differing.length === 0) {
    console.log(
      `ok   ${path.relative(REPO, COMMITTED) || COMMITTED} is byte-identical to this tree's wasm-pack build`,
    );
    for (const name of names) console.log(`     ${committedDigests.get(name)}  ${name}`);
    process.exit(0);
  }

  const before = interrogate(COMMITTED);
  const after = interrogate(fresh);
  const staleStamp =
    before.specHash !== after.specHash || before.crateVersion !== after.crateVersion;
  const staleGolden = before.goldens.some((g) => !g.ok);

  console.error("");
  console.error(
    `FAIL ${path.relative(REPO, COMMITTED) || COMMITTED} is not what this tree builds`,
  );
  for (const name of differing) {
    console.error(
      `     ${name}: committed ${committedDigests.get(name) ?? "(absent)"} vs built ${
        freshDigests.get(name) ?? "(absent)"
      }`,
    );
  }
  console.error("");
  console.error(`     committed: spec_hash=${before.specHash} crate_version=${before.crateVersion}`);
  console.error(`     built    : spec_hash=${after.specHash} crate_version=${after.crateVersion}`);
  for (const g of before.goldens) {
    console.error(`     committed golden ${g.name}: ${g.ok ? "reproduced" : "MISSED"}`);
  }
  console.error("");
  if (staleStamp || staleGolden) {
    console.error("     diagnosis: THE COMMITTED BUNDLE IS STALE. It reports a different");
    console.error("     revision than this source tree builds, so it is the artifact that is");
    console.error("     wrong, not the comparison. Rebuild and commit it:");
    console.error("       wasm-pack build crates/sharpearena-wasm --target nodejs \\");
    console.error("         --out-dir ../../npm/sharpearena/pkg --out-name sharpearena");
  } else {
    console.error("     diagnosis: THE BUILD IS NOT BYTE-REPRODUCIBLE on this runner. Both");
    console.error("     bundles report the same spec hash and crate version and reproduce every");
    console.error("     committed golden, so this is a toolchain or environment difference, not");
    console.error("     a stale artifact. Compare rustc, wasm-pack and wasm-opt versions against");
    console.error("     the pins in rust-toolchain.toml and .github/workflows/release.yml before");
    console.error("     committing the rebuild.");
  }
  process.exit(1);
} finally {
  rmSync(workDir, { recursive: true, force: true, maxRetries: 5 });
}
