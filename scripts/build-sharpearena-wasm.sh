#!/usr/bin/env bash
# Build the SharpeArena trading-agent environment (sharpearena, via sharpearena-wasm) to
# WASM so TypeScript/Bun can run baselines + recompute-to-verify with the IDENTICAL
# engine as the Rust harness.
#
# Output: ./npm/sharpearena/pkg with a JS module exposing
# `run_baseline(configJson)` / `replay_run(datasetJson, trajectoryJson, costsJson)`
# (+ dataset/scenario/stress/walk-forward helpers), which the npm wrapper imports.
#
# This must be wasm-pack, not bare wasm-bindgen: the committed bundle is the PUBLISHED
# bundle, and `scripts/check-wasm-bundle.mjs` gates it byte for byte against exactly this
# command. wasm-pack runs wasm-opt after wasm-bindgen, so a bare wasm-bindgen build
# produces a different binary and turns that gate red.
set -euo pipefail

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack is required (CI pins wasm-pack 0.15.0 with Rust 1.96.0):" >&2
  echo "  cargo install wasm-pack --version 0.15.0" >&2
  exit 1
fi

rustup target add wasm32-unknown-unknown

# The npm wrapper is CommonJS and supports Node >=18, so its checked-in WASM bindings
# must use Node's loader, matching the release workflow.
wasm-pack build crates/sharpearena-wasm \
  --target nodejs \
  --out-dir ../../npm/sharpearena/pkg \
  --out-name sharpearena

echo "wrote ./npm/sharpearena/pkg  (require('./pkg/sharpearena'))"
echo "verify with: node scripts/check-wasm-bundle.mjs"
