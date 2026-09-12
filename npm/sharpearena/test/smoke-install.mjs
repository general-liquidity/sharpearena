// Pack-then-offline-install smoke test for the published artifact.
//
// `npm test` exercises the working tree by relative path; nothing exercised the
// *tarball*, and nothing exercised what a consumer can resolve **by package name**.
// A `files` list that drops `dist` or `pkg`, a `main` that points at a file the pack
// excludes, a wasm loader path that only resolves in-repo, and an `exports` map that
// is wrong in either direction all pass the unit tests and then decide what every
// installer gets. This script tests exactly that: `npm pack` -> a throwaway project
// -> an OFFLINE install from the tarball with an isolated cache (so the registry
// cannot paper over a missing file) -> resolution by package name.
//
// The subpath probe is the A3 gate. Proving a bypass is closed means attempting it
// and reading *which* refusal fired: a deep import can also fail because the file is
// absent from the tarball, which would leave this green while `exports` did nothing.
// So the probe asserts Node's `ERR_PACKAGE_PATH_NOT_EXPORTED` by code, and separately
// asserts the same file IS in the tarball and IS reachable by relative path from
// inside the installed package. Only the exports map can produce that combination.
//
// npm is spawned as `node npm-cli.js ...`: on Node >= 20.12 `execFileSync` refuses
// `npm.cmd` on Windows without `shell: true` (CVE-2024-27980), and a shell means
// quoting; the JS entry point needs neither.

import assert from "node:assert";
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function npmCliPath() {
  const nodeDir = path.dirname(process.execPath);
  const candidates = [
    // Windows: npm ships beside node.exe.
    path.join(nodeDir, "node_modules", "npm", "bin", "npm-cli.js"),
    // Unix: node lives in <prefix>/bin, npm in <prefix>/lib/node_modules.
    path.join(nodeDir, "..", "lib", "node_modules", "npm", "bin", "npm-cli.js"),
  ];
  const found = candidates.find(existsSync);
  assert(found, `cannot find npm-cli.js near ${process.execPath}; tried:\n${candidates.join("\n")}`);
  return found;
}

const npmCli = npmCliPath();
function npm(args, options) {
  return execFileSync(process.execPath, [npmCli, ...args], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
    ...options,
  });
}

// The pack ships prebuilt artifacts and runs no build of its own (no prepack script),
// so an unbuilt tree must be a loud refusal here, not an empty pack that installs fine.
for (const artifact of ["dist/index.js", "pkg/sharpearena.js", "pkg/sharpearena_bg.wasm"]) {
  assert(
    existsSync(path.join(packageDir, artifact)),
    `${artifact} is missing: build first (npm run build; the pkg/ wasm comes from wasm-pack)`,
  );
}

const workDir = mkdtempSync(path.join(tmpdir(), "sharpearena-smoke-"));
try {
  // Pack from a staged copy in publish-time state. wasm-pack drops a `.gitignore`
  // containing `*` into pkg/, and npm honors a nested .gitignore even for a directory
  // the `files` list names, so packing the working tree as-is ships no wasm. At release
  // `prepublishOnly` deletes that file before `npm publish`; the stage mirrors exactly
  // that (and skips node_modules, which the pack never reads).
  const stageDir = path.join(workDir, "stage");
  cpSync(packageDir, stageDir, {
    recursive: true,
    filter: (source) => {
      const relative = path.relative(packageDir, source);
      return relative !== "node_modules" && relative !== path.join("pkg", ".gitignore");
    },
  });
  // npm 11 prints an array of packed tarballs; npm 12 prints an object keyed by
  // package name. The release job pins npm 12 and CI uses the Node-bundled npm
  // 11, so reading `.length` off the parsed value passed every CI run and failed
  // every release, which is how v0.27.0 reached crates.io and PyPI and not npm.
  // Measured on this package: npm@11.15.0 gives `[{...}]`, npm@12.0.2 gives
  // `{"@general-liquidity/sharpearena": {...}}`.
  const packedRaw = JSON.parse(
    npm(["pack", "--json", "--pack-destination", workDir], { cwd: stageDir }),
  );
  const packed = Array.isArray(packedRaw) ? packedRaw : Object.values(packedRaw);
  assert.strictEqual(
    packed.length,
    1,
    `npm pack must produce exactly one tarball, got ${packed.length}`,
  );
  const [tarball] = packed;
  assert(tarball.files.length > 0, "npm pack produced an empty tarball");
  const packedPaths = new Set(tarball.files.map((file) => file.path));
  for (const artifact of [
    "package.json",
    "dist/index.js",
    "dist/index.d.ts",
    "pkg/sharpearena.js",
    "pkg/sharpearena_bg.wasm",
  ]) {
    assert(
      packedPaths.has(artifact),
      `${artifact} is not in the tarball; check the files list in package.json`,
    );
  }
  // The `files` list is an allowlist, so the sources and the suite must NOT ship. This
  // is also what makes the subpath probe below meaningful: `pkg/` ships and `src/` does
  // not, so a refusal on `pkg/...` cannot be a missing file.
  for (const excluded of ["src/index.ts", "test/smoke.test.js", "bench/throughput.js", "tsconfig.json"]) {
    assert(
      !packedPaths.has(excluded),
      `${excluded} is in the tarball; the files list is supposed to exclude it`,
    );
  }
  const tarballPath = path.join(workDir, tarball.filename);
  assert(existsSync(tarballPath), `npm pack reported ${tarball.filename} but it is not on disk`);

  const projectDir = path.join(workDir, "project");
  mkdirSync(projectDir);
  writeFileSync(
    path.join(projectDir, "package.json"),
    JSON.stringify({ name: "sharpearena-smoke", private: true }),
  );
  // --offline + a fresh empty cache: everything must come from the tarball.
  npm(
    [
      "install",
      "--offline",
      "--no-audit",
      "--no-fund",
      "--cache",
      path.join(workDir, "npm-cache"),
      tarballPath,
    ],
    { cwd: projectDir },
  );

  // Not just "require resolves": call into the wasm kernel, so a tarball whose JS shims
  // packed but whose .wasm did not still fails here.
  const probe = `
    const assert = require("node:assert");
    const fs = require("node:fs");
    const path = require("node:path");

    const arena = require("@general-liquidity/sharpearena");
    for (const name of ["runBaseline", "replayRun", "datasetSynthetic", "generateScenario", "stressSuite", "walkForward", "tagRegime", "checkSpecHash"]) {
      assert.equal(typeof arena[name], "function", name + " is not exported");
    }
    const run = arena.runBaseline({ agent: "buy_and_hold", seed: 1 });
    assert.equal(run.returns.length, 100, "packed kernel did not run a baseline");
    const scenario = arena.generateScenario({ spec: { start_level: 0, num_levels: 0, n_symbols: 2, n_days: 40, distribution_mode: "calm" }, seed: 7 });
    assert.equal(scenario.dates.length, 40, "packed generateScenario is not the kernel export");
    assert.equal(Object.keys(scenario.closes).length, 2);

    // The handshake pin is not caller-supplied, and the wrapper refuses a foreign hash.
    assert.equal(arena.checkSpecHash.length, 1, "checkSpecHash must take exactly the engine hash");
    assert.equal(arena.compareSpecHash, undefined, "the two-sided comparison must stay internal");
    assert.throws(() => arena.checkSpecHash("00000000deadbeef"), /SpecHashMismatch/);

    // --- A3: the deep import that skipped the handshake -----------------------------
    const installed = path.dirname(require.resolve("@general-liquidity/sharpearena/package.json"));
    const kernelFile = path.join(installed, "pkg", "sharpearena.js");
    assert.ok(fs.existsSync(kernelFile), "pkg/sharpearena.js must be installed; otherwise the probe below proves nothing");

    let code = null;
    try {
      require("@general-liquidity/sharpearena/pkg/sharpearena.js");
    } catch (error) {
      code = error.code;
    }
    assert.equal(code, "ERR_PACKAGE_PATH_NOT_EXPORTED", "the deep subpath import must be refused by the exports map, not by anything else; got " + code);

    for (const subpath of ["./pkg/sharpearena.js", "./pkg/sharpearena_bg.wasm", "./dist/index.js", "./dist/specHash.js"]) {
      let sub = null;
      try { require("@general-liquidity/sharpearena/" + subpath.slice(2)); } catch (error) { sub = error.code; }
      assert.equal(sub, "ERR_PACKAGE_PATH_NOT_EXPORTED", subpath + " resolved by name; the exports map is incomplete");
    }

    // The refusal is the exports map and nothing else: the same file loads fine by path.
    const direct = require(kernelFile);
    assert.equal(typeof direct.spec_hash, "function", "the installed kernel file is intact, so the refusal above was the exports map");

    // package.json stays reachable by name; tooling and the sibling smoke test read it.
    assert.equal(require("@general-liquidity/sharpearena/package.json").version, require(path.join(installed, "package.json")).version);

    console.log("smoke-install ok: the tarball installs offline, the wasm kernel answers, and the deep kernel path is ERR_PACKAGE_PATH_NOT_EXPORTED");
  `;
  const output = execFileSync(process.execPath, ["-e", probe], {
    cwd: projectDir,
    encoding: "utf8",
  });
  process.stdout.write(output);
} finally {
  rmSync(workDir, { recursive: true, force: true, maxRetries: 5 });
}
