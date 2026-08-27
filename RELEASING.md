# Releasing SharpeArena

SharpeArena ships from **one Rust engine** to three surfaces: the `sharpearena` crate
(crates.io), the `@general-liquidity/sharpearena` npm package (the WASM build), and the
`sharpearena` PyPI wheel (the pyo3 binding + Gymnasium adapter). It depends on the
**published** `sharpebench-*` crates (the simulator engine) rather than vendoring them.

## Cutting a version

```bash
# Green checks. CI runs the same release rehearsal before a tag can publish.
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo deny check

# Exercise the exact bump -> clean rebind -> validation -> tag sequence in an
# isolated local clone. It creates no tag and pushes nothing from this checkout.
python scripts/release.py rehearse patch

# Cut the release. This is the only supported path: it atomically pushes main and
# the annotated tag only after the prospective tag passes every provenance check.
python scripts/release.py execute patch
```

`release.toml` sets `publish = false`: the local machine never publishes. The `v*`
tag triggers CI, which publishes via **OIDC Trusted Publishing** (no stored tokens).

## Why the checked driver owns the tag

`release.toml` sets both `tag = false` and `push = false`. Running cargo-release
directly can create local version commits, but it cannot publish an unchecked tag.
`scripts/release.py` owns the complete operation and the order matters for the
tamper-evidence manifest.

The version bump rewrites six files inside the provenance source scope, so a manifest
bound before the release is stale the moment cargo-release commits. The driver waits
until that commit is clean, regenerates the manifest with
`generated_at_head_dirty: false`, and commits only `paper/evidence/provenance.json`.
The annotated tag points at that provenance-only child.

Before pushing, `verify-tag` requires all of the following:

1. the tag commit has one parent and changes only the provenance manifest;
2. the manifest names that parent as a clean generation;
3. the source, artifact, model-artifact, exclusion and identity-field rules equal the
   canonical rules in `provenance_common.py`;
4. every recorded source and artifact digest and every model identity summary matches
   the tagged tree;
5. the snapshot digest recomputes;
6. the Rust, Python and npm version surfaces all equal the tag; and
7. a published tag commit is an ancestor of `origin/main`. The driver's private
   pre-push check additionally accepts the prospective tag as a child of
   `origin/main`; the public `verify-tag` command does not.

The manifest cannot contain the hash of its own commit without becoming
self-referential. This construction avoids that problem: `provenance.json` is outside
its own scope, and the tag commit is byte-identical to its named parent everywhere the
manifest binds. Main and the tag are pushed together with `git push --atomic`, so an
interrupted client cannot expose only half of the release state.

The release workflow passes the selected tag through the `RELEASE_TAG` environment
variable and quotes every shell use. GitHub expressions never enter a `run:` body;
this is checked structurally because a manually dispatched tag is operator input, not
shell syntax. Validation emits the exact commit object it checked, and every publishing
job checks out that immutable SHA rather than resolving the tag again. The workflow's
actions and downloaded publishing tools are pinned to immutable revisions or exact
versions, including the manylinux container digest.

## Correcting the v0.19.0 provenance gap

Do not move or replace `v0.19.0`. Its packages are already public, and its tag points
at the version-bump commit rather than a clean provenance-only rebind. The tag's
manifest names an older commit and six in-scope version files have different digests;
`python scripts/release.py verify-tag v0.19.0` therefore fails, intentionally.

The honest correction is a patch release containing the disclosure and the checked
release machinery:

```bash
python scripts/release.py rehearse patch   # exercises prospective v0.19.1 locally
python scripts/release.py execute patch    # creates and publishes v0.19.1
```

Do not tag or publish while reviewing this change. The command above is the operator
path once main is green and a patch release is desired.

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
