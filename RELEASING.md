# Releasing SharpeArena

SharpeArena ships from **one Rust engine** to three surfaces: the `sharpearena` crate
(crates.io), the `@general-liquidity/sharpearena` npm package (the WASM build), and the
`sharpearena` PyPI wheel (the pyo3 binding + Gymnasium adapter). It depends on the
**published** `sharpebench-*` crates (the simulator engine) rather than vendoring them.

## Cutting a version

```bash
# green checks (cargo-release will not run these)
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo deny check

cargo release patch            # DRY RUN
cargo release patch --execute  # bump shared version + rewrite pins + tag vX.Y.Z + push
```

`release.toml` sets `publish = false`: the local machine never publishes. The `v*`
tag triggers CI, which publishes via **OIDC Trusted Publishing** (no stored tokens).

Never hand-edit a version. `Cargo.toml` is the only place one is authored: the npm
`package.json` and the two `crates/sharpearena-py` manifests (that crate is excluded from
the Cargo workspace) are rewritten from the workspace version by `pre-release-replacements`
in **`crates/sharpearena/release.toml`**; they live on the crate, not at the workspace
root, because cargo-release resolves `file` relative to each crate being processed.

Three guards keep the surfaces in lockstep, because every publish step is skip-if-present
and a stale manifest therefore ships nothing while still reporting green:

1. `pre-release-replacements` rewrite all three non-inherited manifests on every bump.
2. The npm and PyPI jobs assert their manifest equals the tag *before* publishing.
3. The `verify` job queries crates.io, npm and PyPI after the fact and fails the run if
   any of them is not serving the tag version.

## The standing pipeline

The packages are live on all three registries and every publish runs through
`release.yml` on a `v*` tag. No tokens are stored anywhere:

- **crates.io**: `sharpearena` and `sharpearena-wasm` publish via OIDC trusted
  publishing (owner `general-liquidity`, repo `sharpearena`, workflow `release.yml`).
- **npm**: `@general-liquidity/sharpearena` publishes via its trusted publisher.
- **PyPI**: `sharpearena` publishes via its trusted publisher (workflow `release.yml`,
  environment `pypi`); maturin builds and uploads the wheel.

Each publish job is gated by a repo variable (`PUBLISH_CRATES`, `PUBLISH_NPM`,
`PUBLISH_PYPI`) and runs in its GitHub Environment (`crates` / `npm` / `pypi`), so a
registry can be paused by flipping its variable without touching the workflow. After
the publish jobs, the `verify` job queries all three registries and fails the run if
any of them is not serving the tag version.
