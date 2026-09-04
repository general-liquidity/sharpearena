# SharpeArena documentation

The root [`README`](../README.md) is the product entry point. Start here when a task
needs more detail.

## Use the product

| I want to | Read |
|---|---|
| Step, vectorize, or register a Python environment | [Gymnasium guide](gymnasium.md) |
| Connect an agent over stdio or HTTP | [Agent contract](agent-contract.md) |
| Train with `verifiers` or Prime RL | [Training guide](training.md) |
| Choose a market, wrapper, or operational surface | [Capability map](capabilities.md) |
| Run the local open-weight field | [Local-agent architecture](LOCAL_AGENT_ARCHITECTURE.md) |
| Compare supported local model runtimes | [Local model matrix](LOCAL_MODEL_MATRIX_2026.md) |
| Commit forecasts now and score them after resolution | [Prospective forecast evidence](forecast-evidence.md) |

Registry-specific instructions live with the [Rust crate](../crates/sharpearena/),
[Python distribution](../crates/sharpearena-py/), and
[npm package](../npm/sharpearena/).

## Understand the claims

| Question | Read |
|---|---|
| Which package owns each responsibility? | [Architecture](architecture.md) |
| What is guaranteed, and where does the boundary stop? | [Integrity and security](integrity-and-security.md) |
| What has been measured, and what has not? | [Evidence and current status](evidence.md) |
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

The package READMEs are intentionally self-contained so registry users do not need the
repository layout for first use.
