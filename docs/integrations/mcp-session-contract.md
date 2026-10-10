# MCP episode and refusal contract

`sharpearena.mcp_server.build_server` registers `reset`, `step` and `spec`.
Each server owns one lazily created episode, shared by all callers. Tool calls
are serialized. A reset discards that episode's progress; it does not create an
independent rollout for each MCP connection. Use a separate server instance for
each independent rollout. The `spec` tool reports these rules in
`session_contract`.

Call `reset` before `step`. After a returned termination or truncation, another
step returns `episode_done` until reset. Calling `spec` does not reset an episode.

Decisions use the public sparse order contract: omitted symbols retain their
current portfolio weights. Parsing uses the environment's configured maximum
weight, not an unrelated default. The environment then validates the entire
target vector against its advertised action space, including `allow_short`.
There is no clipping or invented fallback action.

Malformed decisions and typed input refusals return a JSON string containing
`error`, `detail` and `environment_advanced: false`. Wrapper/native input
refusals also include `reason_code`. These are JSON objects encoded in the tool
result string, not the MCP protocol's `structuredContent` field. A refused
decision has not advanced the environment; a valid decision may be retried.

Unrelated runtime failures and native `EngineFailure` exceptions are not
relabeled as agent refusals. A caller must record them as a failed rollout, not
as an agent decision or a scored transition. The builder does not configure
transport authentication, access policies or per-client isolation. Those depend
on the deployment; no general claim about transport authentication follows
from this episode contract.

The entry point uses streamable HTTP. The tests exercise the tools locally with
the installed native environment and a registration double, without a model,
network transport or any deployment authentication claim.
