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

`release.toml` sets `publish = false` — the local machine never publishes. The `v*`
tag triggers CI, which publishes via **OIDC Trusted Publishing** (no stored tokens).

Never hand-edit a version. `Cargo.toml` is the only place one is authored: the npm
`package.json` and the two `crates/sharpearena-py` manifests (that crate is excluded from
the Cargo workspace) are rewritten from the workspace version by `pre-release-replacements`
in **`crates/sharpearena/release.toml`** — they live on the crate, not at the workspace
root, because cargo-release resolves `file` relative to each crate being processed.

Three guards keep the surfaces in lockstep, because every publish step is skip-if-present
and a stale manifest therefore ships nothing while still reporting green:

1. `pre-release-replacements` rewrite all three non-inherited manifests on every bump.
2. The npm and PyPI jobs assert their manifest equals the tag *before* publishing.
3. The `verify` job queries crates.io, npm and PyPI after the fact and fails the run if
   any of them is not serving the tag version.

## One-time publishing setup (pending)

Before the first CI publish, each registry needs its trusted publisher configured —
mirroring the SharpeBench process:

- **crates.io** — `sharpearena`, `sharpearena-wasm`: a crate must exist before a trusted
  publisher can be added, so the **first** publish of each name needs a token
  (`cargo publish -p sharpearena` then `-p sharpearena-wasm`); then add the trusted
  publisher (owner `general-liquidity`, repo `sharpearena`, workflow `release.yml`) and
  never use a token again.
- **npm** — `@general-liquidity/sharpearena`: claim once (`npm publish --access public`),
  then add the trusted publisher.
- **PyPI** — `sharpearena`: configure a trusted publisher (GitHub → repo `sharpearena`,
  workflow `release.yml`, environment `pypi`); maturin builds + uploads the wheel.

Then set repo variables `PUBLISH_CRATES=true`, `PUBLISH_NPM=true`, `PUBLISH_PYPI=true`
and create the `crates` / `npm` / `pypi` GitHub Environments (with whatever review
protection you want on the gate). `release.yml` is added as part of the first publish.
