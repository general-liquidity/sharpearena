# AGENTS.md for SharpeArena

Onboarding for coding agents opening this repository. `CLAUDE.md` is an alias
carrying the same rules. SharpeArena is a deterministic point-in-time trading
sandbox and governed agent contract: a Rust environment kernel with Gymnasium,
PettingZoo, vector and WASM surfaces, scenario generation, market models
including a price-time-priority limit-order book, capture and replay, and
checked external-agent execution.

## Current goal: close the gap programme

The active goal is to close every row of the gap programme across both
repositories, with no release until the owner approves it at the end. The row
list lives in the maintainer workspace (`research/product-context/acceptance-map.md`
and `implementation-gaps.md`), not in this repository. An agent without that
workspace takes its rows from the coordinator.

The audit ledgers are finished records. Rows G01 to G34 of the
[2026-09-09 ledger](docs/audits/2026-09-09/IMPLEMENTATION.md) are closed, and the
[2026-09-07 audit](docs/audits/2026-09-07/IMPLEMENTATION.md) is a completed
historical checkpoint. Neither is an active schedule. Do not silently reopen or
rewrite historical evidence.

## Goal rules

These come from the goal, not from general practice, and they override
convenience:

1. **No release or tag until the owner approves it** at the end of the
   programme. No force push or history rewrite at any time.
2. **Experiments run only when declared, protocol-frozen and approved.** A run
   needs a frozen protocol and analysis plan before it starts, and the owner's
   approval. Without both there are no model downloads, model calls, benchmark
   fields or market-data acquisition. Synthetic regressions, existing fixtures,
   package checks and finite diagnostics are always in scope.
3. **Published numerical evidence stays frozen.** Where a repair changes what a
   producer would compute, say so in the paper instead of regenerating the
   number. An approved experiment publishes a separately versioned result beside
   the frozen one, never in place of it.
4. **A suspected issue is not a confirmed bug.** Every row closes with an
   implementation or test reference, or with an evidence-backed disposition.
5. **Mutation-check every regression.** Revert the fix in an isolated temporary
   installed package, confirm the new test fails, restore. Never mutate the
   production worktree.
6. **A source-tree test does not establish installed-package parity.** The
   native extension, wheel and npm/WASM surfaces need the rebuilt artifact
   exercised.
7. **Verify against the committed tree**, not the working tree: `git show
   HEAD:<path>`. Check exit codes, not the tail of the output.
8. Repair branches merge into `main` by normal merge after all relevant checks
   pass on the exact pushed head; verify the resulting main tree.

## Repository rules

- **Commit author must be `Tiberiu Toca <tibi.toca@gmail.com>`.** Never add a
  `Co-Authored-By` trailer for any agent or model.
- Conventional-commit prefixes. Small commits over explicit paths; never
  `git add -A`. Preserve unrelated edits.
- No repo-wide destructive git: no `git checkout -- .`, `git reset --hard`,
  `git stash` or `git clean`.
- No em dashes in Markdown or paper prose.
- The environment kernel is deterministic: no ambient randomness, no system
  clock, fixed reduction order. `SPEC_HASH` identity must survive Cargo
  packaging; run the packaged-spec check after touching the manifest or
  build script.
- Arena consumes an exact-pinned registry SharpeBench: `sharpebench-core`,
  `sharpebench-sim`, `sharpebench-protocol` and `sharpebench-attest` at
  `=0.28.0`. Three of those four are inputs to `SPEC_HASH`: `build_support.rs`
  canonicalizes `sharpebench-core`, `sharpebench-protocol` and `sharpebench-sim`
  into `suite-dependencies.v1.toml`, so moving any of those pins rebinds the
  attestation record and every wrapper pin. `sharpebench-attest` is a
  dev-dependency and is **outside** the hash; `spec_hash.rs` asserts that all
  four are exact-pinned, which is a manifest-hygiene check, not hash coverage.
  The committed record's `note` names the three. A local Bench repair does not
  reach Arena without a Bench release and a pin bump here; record the pending
  propagation instead of claiming parity.
- Bind provenance on a clean candidate before pushing: `python
  paper/src/check-provenance.py`. The clean-tree check intentionally counts the
  modified manifest, so bind after committing.

## Commands

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
python scripts/check-packaged-spec.py
python scripts/check-lean-scope.py       # every Lean module declares Covers/Assumes
python -m pytest -q                      # needs maturin develop in a venv
python paper/src/check-provenance.py
```

`cargo test --workspace` includes the seeded kernel properties in
`crates/sharpearena/tests/kernel_properties.rs` (reset, step, terminal, fill
and accounting transitions executed against the shipped functions); the Lean
model in `formal/` covers the forecast protocol only.

Cross-platform Rust, the installed wheel, the offline npm consumer, wasm-pack,
`cargo deny` and provenance run in CI; a local pass is necessary, not
sufficient.

## Where things live

| Area | Path |
|---|---|
| Environment kernel, scenarios, markets | `crates/sharpearena/src/` |
| Limit-order book | `crates/sharpearena/src/lob_market.rs` |
| Python distribution, Gym/PettingZoo, task layer | `crates/sharpearena-py/` |
| WASM and npm surface | `crates/sharpearena-wasm/`, `npm/` |
| Wire contract, schemas, fixtures | `crates/sharpearena/contract/` |
| Contract governance | `crates/sharpearena/GOVERNANCE.md` |
| Paper, producers, frozen evidence | `paper/` |
| Documentation map | `docs/README.md` |
| Finished audit records | `docs/audits/` |
| Release operations | `RELEASING.md` |
