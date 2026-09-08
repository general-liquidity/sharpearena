# AGENTS.md for SharpeArena

Onboarding for coding agents opening this repository. `CLAUDE.md` is an alias
carrying the same rules. SharpeArena is a deterministic point-in-time trading
sandbox and governed agent contract: a Rust environment kernel with Gymnasium,
PettingZoo, vector and WASM surfaces, scenario generation, market models
including a price-time-priority limit-order book, capture and replay, and
checked external-agent execution.

## Current goal: 2026-09-07 audit repair

**The active engineering goal for this repository is the audit repair
checklist at [`docs/audits/2026-09-07/IMPLEMENTATION.md`](docs/audits/2026-09-07/IMPLEMENTATION.md).**
It is shared with SharpeBench and mirrored byte-for-byte there; edit both
copies together or neither. The chronological repair diary is
[`VERIFICATION-LOG.md`](docs/audits/2026-09-07/VERIFICATION-LOG.md) beside it.

Work the batches in the order the checklist gives:

| Batch | Scope |
|---|---|
| A | Paper pass. Historical-impact caveat, R11/R13/R14 text, snapshot complexity claim. |
| B | Publication gating. Package consumers exercised in CI; versioned cross-language conformance fixtures. |
| C | Run identity (BI3, Bench-led): keyed run/window/seed identity. |
| D | Producer rows that touch existing claims: AP5, AP3. |
| E | Shared mathematics and contracts: R06/AI1, R03 and R09 propagation through the pinned Bench dependency, R12, AR2. |
| F | Remaining Bench diagnostics (Bench repository). |
| G | Remaining Arena telemetry: AR1/AR3 strict optional telemetry, AR4 duration provenance. |
| H | Producer rows for the next field run: AP1, AP2, AP4, AP6. Not required for the current papers. |
| I | Bounded probes. One attempt each, then promote to a defect row or delete. |

Batches A and B change what the shipped product claims and come first. Five
items are explicitly deferred and listed at the end of the checklist; do not
start them without reopening the decision.

R11 is partly stale: `paper/sections/03-environment.tex` line 104 already
describes price-time priority and the call-auction uncross, and
`crates/sharpearena/src/lob_market.rs` implements FIFO per level. The residual
defect is line 106 contradicting line 104.

## Goal rules

These come from the goal, not from general practice, and they override
convenience:

1. **No release, tag, force push or history rewrite** under this goal.
2. **No new experiments**: no model downloads, model calls, benchmark fields or
   market-data acquisition. Synthetic regressions, existing fixtures, package
   checks and finite diagnostics are in scope.
3. **Published numerical evidence stays frozen.** Where a repair changes what a
   producer would compute, say so in the paper instead of regenerating the
   number.
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
- Arena consumes an exact-pinned registry SharpeBench. A local Bench repair
  does not reach Arena without a Bench release, which this goal does not
  authorize; record the pending propagation instead of claiming parity.
- Bind provenance on a clean candidate before pushing: `python
  paper/src/check-provenance.py`. The clean-tree check intentionally counts the
  modified manifest, so bind after committing.

## Commands

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
python scripts/check-packaged-spec.py
python -m pytest -q                      # needs maturin develop in a venv
python paper/src/check-provenance.py
```

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
| Audit goal | `docs/audits/2026-09-07/` |
| Release operations | `RELEASING.md` |
