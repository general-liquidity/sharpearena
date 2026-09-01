# SharpeArena documentation

The root [`README`](../README.md) is the short product entry point. Use this map
for the deeper material.

## Build and use

| Topic | Document |
|---|---|
| System boundaries and package topology | [Architecture](architecture.md) |
| Environment, training, market, and agent-operation surfaces | [Capability map](capabilities.md) |
| Determinism, leak freedom, containment boundary, and provenance | [Integrity and security](integrity-and-security.md) |
| Current experimental evidence and what has not been run | [Evidence](evidence.md) |
| Training with Gymnasium, `verifiers`, and Prime RL | [Training](training.md) |
| Local open-weight field design | [Local-agent architecture](LOCAL_AGENT_ARCHITECTURE.md) |
| Model/runtime feasibility study | [Local model matrix](LOCAL_MODEL_MATRIX_2026.md) |
| Sandbox and generated-code research | [Sandbox research](SANDBOX_ENVIRONMENT_RESEARCH_2026.md) |

## Contracts and operations

| Topic | Document |
|---|---|
| Canonical evaluation contract | [`EVALUATION.md`](../EVALUATION.md) |
| Wire-governance rules | [`crates/sharpearena/GOVERNANCE.md`](../crates/sharpearena/GOVERNANCE.md) |
| Release process | [`RELEASING.md`](../RELEASING.md) |
| Gordon comparison | [Gordon port assessment](GORDON_PORT_ASSESSMENT.md) |
| Methodology paper and committed evidence | [`paper/`](../paper/) |

Package-specific READMEs live beside the Rust crate, Python distribution, npm
package, and Prime RL example so registry consumers do not need this repository
layout to use their surface.
