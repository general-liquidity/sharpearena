# SharpeArena documentation

The root [`README`](../README.md) is the product entry point. Start here when a task
needs more detail.

## Current state

The active goal is to close the gap programme, shared with SharpeBench, with no
release until the owner approves it at the end. [`AGENTS.md`](../AGENTS.md)
carries that goal and its rules. The latest tag is `v0.31.0`, `[Unreleased]` in
[`CHANGELOG.md`](../CHANGELOG.md) lists the work after it, and the tree pins
SharpeBench at `=0.28.0`.

[Evidence and current status](evidence.md) says what has been measured, and
[support status](integrations/support-status.md) says what each interface has
been tested against. The audits under [`audits/`](audits/) are finished records,
not schedules.

## Use the product

| I want to | Read |
|---|---|
| Step, vectorize, or register a Python environment | [Gymnasium guide](gymnasium.md) |
| Connect an agent over stdio or HTTP | [Agent contract](agent-contract.md) |
| Get a runnable example for PettingZoo, `verifiers`, Minari, MCP, TorchRL, the JSON contract, or the SharpeBench bridge | [Interface examples](interfaces.md) |
| Train with `verifiers` or Prime RL | [Training guide](training.md) |
| Choose a market, wrapper, or operational surface | [Capability map](capabilities.md) |
| Run the local open-weight field | [Local-agent architecture](LOCAL_AGENT_ARCHITECTURE.md) |
| Compare supported local model runtimes | [Local model matrix](LOCAL_MODEL_MATRIX_2026.md) |
| Commit forecasts now and score them after resolution | [Prospective forecast evidence](forecast-evidence.md) |
| Interpret baseline scores, unavailable estimates and intervals | [Statistical confidence](statistical-confidence.md) |
| Tell an unavailable kernel score from a refused one, or classify a new kernel key | [Kernel field convention](kernel-field-convention.md) |
| Add a framework adapter, or check one against the engine | [Integration contracts](integrations/contracts.md) and [inventory](integrations/inventory.md) |
| See what's tested, specified, or ruled out per interface | [Support status](integrations/support-status.md) |
| Check the current CleanRL, native-bridge and HUD/Harbor scope decisions | [Integration dispositions](integrations/current-dispositions.md) |

Registry-specific instructions live with the [Rust crate](../crates/sharpearena/),
[Python distribution](../crates/sharpearena-py/), and
[npm package](../npm/sharpearena/).

## Understand the claims

| Question | Read |
|---|---|
| Which package owns each responsibility? | [Architecture](architecture.md) |
| What is guaranteed, and where does the boundary stop? | [Integrity and security](integrity-and-security.md) |
| What has been measured, and what has not? | [Evidence and current status](evidence.md) |
| Which artifact backs a paper claim, and what does it support today? | [Evidence register](evidence-register.md) |
| Which RL-contract behaviours have tests, and which are open? | [RL-contract coverage matrix](rl-contract-coverage.md) |
| Which seeds, gates, and reports define an evaluation? | [`EVALUATION.md`](../EVALUATION.md) |
| Where are the paper and committed evidence? | [`paper/`](../paper/) |

## Govern and operate

| Task | Document |
|---|---|
| Evolve the observation/decision wire | [Contract governance](../crates/sharpearena/GOVERNANCE.md) |
| Cut or verify a release | [`RELEASING.md`](../RELEASING.md) |

## Design records

These documents record prior comparisons and decisions. They are not required to use
the package:

- [Sandbox and generated-code research](SANDBOX_ENVIRONMENT_RESEARCH_2026.md)
- [Gordon port assessment](GORDON_PORT_ASSESSMENT.md)
- [Suite audit and complete repair checklist, 7 September 2026](audits/2026-09-07/README.md)

The package READMEs are intentionally self-contained so registry users do not need the
repository layout for first use.
