# Local-agent build audit — 2026-08-26

Scope: commit `fe42e13` "feat(local-agents): add local fields, strategy search and paper
trading" plus release `6ffdf06` v0.16.0, audited against the original build list.

Method: read the code; run it. Commit messages, docstrings, CHANGELOG entries and
`docs/` were treated as claims to be tested, not as evidence. Every verdict below cites
either a file:line or a command that was actually executed. Probe scripts were written
to a scratch directory outside the repo; nothing in the tree was modified except this
file.

## Test runs (real numbers)

```
$ cargo test -p sharpearena
   unit          140 passed; 0 failed; 0 ignored
   conformance     4 passed; 0 failed
   smoke           3 passed; 0 failed
   doc-tests       0 passed
   exit 0

$ crates/sharpearena-py/.venv/Scripts/python.exe -m pytest crates/sharpearena-py -q
   783 passed, 2 skipped, 1596 warnings in 38.52s
   exit 0
```

The 2 skips are inverse guards, not gaps: `test_market.py:38` and
`test_pettingzoo.py:35` are `skipif(_HAS_PETTINGZOO)` — they run *only when pettingzoo
is absent*. Nothing in the new work is skipped. All six console entry points install and
respond to `--help`, and both shipped example plans run under `--inspect`.

## Verdict table

| # | Item | Verdict |
|---|---|---|
| A1 | Ollama stdio shim | **BUILT** |
| A2 | Scaffold-neutral prompt renderer | **BUILT** |
| A3 | Constrained decoding from committed schema | **BUILT** |
| A4 | Unification of the two decision formats | **BUILT**, one semantic split |
| A5 | Stop masking failures as holds | **PARTIAL — see Finding 1** |
| A6 | Batched inference driver | **BUILT** |
| A7 | Resumable, shardable cell scheduler | **BUILT** |
| A8 | Model identity in the evidence row | **BUILT**, two gaps |
| A9 | Cadence + thinking budget as config axes | **PARTIAL** |
| A10 | Evidence writer / SharpeBench artifact | **BUILT**, two defects |
| B1 | N candidates generated, every one recorded | **BUILT** |
| B2 | Candidate representation (constrained DSL) | **BUILT** |
| B3 | Observed trial count feeding deflation | **BUILT**, scope caveat |
| B4 | Backtest driver over candidates | **BUILT** |
| B5 | Selection protocol | **BUILT**, untested |
| D1 | Market data connector | **BUILT**, thinly tested |
| D2 | Paper broker, no real-capital path | **BUILT — safety check passes** |
| D3 | Separate non-replayable evidence class | **BUILT and enforced** |
| D5 | Arena commit-reveal wiring | **PARTIAL** |
| E1 | Provenance manifest extended to model artifacts | **PARTIAL** |
| E3 | Guarantee boundary documented | **BUILT** |
| E4 | CI: one-cell smoke only, no field runs | **BUILT** |

---

# Finding 1 (most important): the fault-masking premise in the build list is wrong

The build list states that "the Rust `external.rs` was written specifically to avoid
this and flags faults into TransportHealth instead." That is half true, and the half
that is false is the half that matters.

**The Python side is genuinely fail-closed.** This is real work and it is verified.
`decision_parser.parse_decision` no longer fabricates zero weights — the module docstring
at `decision_parser.py:9-13` promises fail-closed behaviour and the code delivers it. I
drove all six fault classes end to end through the real CLI subprocess against a stub
HTTP server on `127.0.0.1:11434`:

```
MODE=malformed  rc=1 stdout='' stderr='{"error":"model emitted an invalid Decision: completion does not contain a JSON decision object",...}'
MODE=legacy     rc=1 stdout='' stderr='{"error":"model emitted an invalid Decision: unknown decision field(s): weights",...}'
MODE=empty      rc=1 stdout='' stderr='{"error":"model emitted an invalid Decision: completion does not contain a JSON decision object",...}'
MODE=nocontent  rc=1 stdout='' stderr='{"error":"Ollama response has no message.content",...}'
MODE=http500    rc=1 stdout='' stderr='{"error":"Ollama /api/chat returned HTTP 500: ...",...}'
MODE=unknownsym rc=1 stdout='' stderr='{"error":"...symbol \'NOPE\' was not observed",...}'
```

Empty stdout in every case: no hold is ever emitted. In `LocalFieldRunner`, a failed
lane is deactivated (`local_agents.py:1037-1044`), the record is written with
`status: "failed"` and no `returns` key (`:1185-1198`), and `bench_bridge._validate_field`
refuses to compile a field containing any failed cell (`bench_bridge.py:136-141`). That
chain is intact and I verified each link.

**The stdio path the docs promote is not fail-closed.** `docs/LOCAL_AGENT_ARCHITECTURE.md:87-94`
directs the operator to the `ollama_shim` → `ExternalAgent` composition. In
`sharpebench-sim-0.11.0/src/external.rs`, that transport masks:

```rust
// external.rs:47-55
/// An empty-orders hold emitted when a decision could not be produced. The health
/// (not this value) carries whether it was a masked fault vs. a deliberate hold.
fn error_hold(reason: &str) -> Decision { Decision { orders: Vec::new(), ... } }

// external.rs:163-183
fn decide(&mut self, obs: &MarketObservation) -> Decision {
    if self.breaker.is_tripped() {
        self.health.record(DecideError::Transport, true);
        return error_hold("external agent circuit open → hold");
    }
    match self.decide_once(obs) {
        Ok(d) => { self.breaker.record_success(); d }
        Err(e) => {
            let tripped = self.breaker.record_fault();
            self.health.record(e, tripped);
            error_hold("external agent transport fault → hold")
        }
    }
}
```

Upstream is honest about this — the doc comment says the trait cannot signal an error
and that the hold is *flagged* in the health. But the consequence is concrete: the
shim's disciplined `exit(1)` breaks the pipe, and from that bar on every subsequent
`decide()` returns a hold, the run completes, and a full scoreable return series is
produced. `TransportHealth` is the only thing standing between that and publication.

**And SharpeArena does not re-export the health surface.** `crates/sharpearena/src/lib.rs:102-104`
re-exports `ExternalAgent`, `HoldAgent` and `HttpAgent`. A repo-wide grep for
`TransportHealth`, `TransportDiagnostics`, `FailureKind` or `health()` across
`crates/**/*.rs` and `crates/**/*.py` returns exactly two hits: that `ExternalAgent`
re-export line and the word "ExternalAgent" in the shim's own docstring. Neither the
Rust crate nor the Python package ever inspects transport health. A consumer who takes
SharpeArena's public Rust surface at face value and pairs `ExternalAgent` with the
shipped shim gets masked holds with no reachable way to detect them.

**Which way does a masked hold bias?** I measured it rather than assuming. Against the
native engine, `orders: []` is a true hold — the position persists:

```
after buy      portfolio: shares=0.004956843475965662  cash=0.4999
after empty    portfolio: shares=0.004956843475965662  cash=0.4999
after empty x2 portfolio: shares=0.004956843475965662  cash=0.4999
```

So the bias is not "flat return series" as the build list assumed. It is worse in one
respect and better in another: a wedged agent silently keeps riding its last position
and accrues its P&L for the remainder of the window, rather than going to cash.

**Required fixes, in order:** (a) re-export `TransportDiagnostics`/`TransportHealth`
from `crates/sharpearena/src/lib.rs`; (b) make any SharpeArena-side harness that uses
`ExternalAgent` assert zero recorded faults before a return series is scoreable;
(c) correct `docs/LOCAL_AGENT_ARCHITECTURE.md:32-33` and
`SANDBOX_ENVIRONMENT_RESEARCH_2026.md:330`, which both state the absolute "never
converted into a hold" — true of the direct-policy field, false of the stdio path the
same document recommends.

---

# Finding 2: "empty orders is a hold" means two different things

`decision_parser.parse_decision`'s docstring says:

> An empty ``orders`` array is the canonical hold action: it leaves the current
> portfolio unchanged rather than forcibly flattening it.

That is true on the `Decision`→native path (measured above). It is **false** on the
weight-vector path, which is what `parse_decision` itself returns. Measured:

```
after buy positions:                  [0.00496348 0.]
empty-orders weight vector:           [0. 0.]
after empty-orders step positions:    [-1.9557e-05 0.]   reward -0.00020382737664270678
```

`decision_to_weights` maps empty orders to `np.zeros(...)`, and the gym env treats a
zero target vector as *flatten*. So the documented hold liquidates the book and pays the
round-trip cost. This path is not a corner: `verifiers_env.py:316` and
`mcp_server.py:46-49` both call `parse_decision`, and `verifiers_env.py:223` and
`PromptRenderer.system_prompt` both *tell the model* "an empty orders array is a hold."
An RL agent trained through the verifiers env is therefore punished for taking the
action the prompt describes as neutral.

This is the same class of defect A5 was written to eliminate, arriving through a
different door: a deliberate hold silently becomes a forced liquidation.

On the narrow A4 question — yes, there is now one wire format. The legacy
`{"weights": ...}` dialect is rejected everywhere (verified: `unknown decision field(s):
weights`), and MCP, verifiers and the local-model path all call the same parser. The
*syntax* is unified; the *semantics* of the canonical hold are not.

---

# Finding 3: the k axis is degenerate, and the compiled artifact misdescribes it

`FieldPlan.repetitions` is the k axis. `LocalFieldRunner._run_group` builds the batch
env from `[cell.seed for cell in cells]` — the same `cell.seed` for every repetition —
so the market path is identical across k. Only `model.sampling.seed + repetition`
varies. With a deterministic model:

```
returns_sha256 per repetition: [(0,'9742158feac3'), (1,'9742158feac3'), (2,'9742158feac3')]
distinct return series: 1
```

`SamplingConfig.temperature` defaults to `0.0`, and the shipped
`examples/local-agents/field-plan-smoke.json` sets `"temperature": 0.0` with
`"repetitions": 2`. At temperature 0 the sampling seed has no effect either, so k
repetitions are k duplicates of one run for a real model too.

`bench_bridge` then tells SharpeBench those duplicates are Monte-Carlo execution
replicates. End-to-end run of the real compiler:

```
execution_seeds_per_window: 2   runs_per_agent: 4
score_command: sharpebench score ... --execution-seeds-per-window 2 --json
run return signatures (chunks of 2 = "execution replicates"):
  ['d9be949add', 'd9be949add', 'c2f27356b4', 'c2f27356b4']
```

The two "replicates" in each window block are byte-identical. `sharpebench-core-0.11.0/src/composite.rs:1433-1461`
`pooled_returns` averages replicates per bar, so this does not inflate the observation
count — averaging identical series is a no-op, and the run ordering
(`bench_bridge.py:232`, sorted by `(seed_index, repetition)`) correctly aligns the
`chunks_exact(width)` blocks. The damage is representational rather than arithmetic:
the artifact asserts an execution-noise averaging that did not happen, and any reader
of the manifest will believe the entry was replicated under execution noise when it was
not.

Either vary the execution seed across repetitions, or emit
`execution_seeds_per_window: 1` and describe k honestly as a sampling-seed axis.

---

# Finding 4: fabricated confidences enter the scored artifact

`local_agents.py:1075-1084`:

```python
confidence = float(order.get("confidence", 0.5))
...
confidences[index].append(
    sum(lane_confidences) / len(lane_confidences) if lane_confidences else 0.5
)
```

`Decision.Order.confidence` is optional in the schema, and `PromptRenderer.system_prompt`
never asks for it (verified: `'confidence' in system_prompt` → `False`). So in normal
operation every order arrives without a confidence and the harness substitutes `0.5`,
plus another `0.5` for every hold bar.

That stream is consumed downstream: `sharpebench-core-0.11.0/src/composite.rs:1176`
and `:1224-1227` use `confidences` for calibration and confidence-weighting, and fall
back to an unweighted mean only when the vector is *empty*. By synthesizing a constant,
the bridge converts the honest signal "this agent reported no confidence" into a
degenerate-but-scored calibration series. Either propagate an empty vector when the
model reported nothing, or ask for confidence in the prompt.

---

# Finding 5: a paper claim the code deliberately does not implement

`paper/sections/05-protocol.tex:37`:

> a host-side parser separately enforces episode semantics: symbols must have been
> observed, each symbol may occur once, weights must be finite and within the configured
> bound, **and action labels must agree with their signed targets**.

The first three are enforced. The fourth was explicitly removed, with a comment saying
so (`decision_parser.py:127-131`):

> The target is the authoritative desired position; ``action`` is an audit label, not
> the size or sign. [...] Those relations need the current portfolio and cannot be
> inferred here.

Executed:

```python
parse_decision('{"orders":[{"symbol":"A","action":"sell","target_weight":0.9},
                           {"symbol":"B","action":"buy","target_weight":-0.9}]}', ['A','B'])
# -> array([ 0.9, -0.9])   accepted
```

The code's reasoning is sound; the paper sentence is not. Delete that clause.

`paper/sections/05-protocol.tex:45` and `docs/LOCAL_AGENT_ARCHITECTURE.md:150` both say
"a fixed **native** risk guard". `PaperRiskGuard` is pure Python (`paper_trading.py:308`)
with no native component; the same wording appears in the module docstring at
`paper_trading.py:5`. In a paper whose central claim is a native byte-identity boundary,
"native" is a loaded word — change it to "fixed" or "host-side".

---

# D2 safety check: what this code can and cannot do with money

**It cannot place a real-capital order.** I looked for every route and found none.

- The only credentialed write in the codebase is `AlpacaPaperBroker.submit` →
  `POST {base_url}/v2/orders` (`paper_trading.py:538-546`).
- `base_url` is validated against `ALPACA_PAPER_ORIGIN = "https://paper-api.alpaca.markets"`
  (`:26`) in the constructor (`:506-512`), and then reassigned to that constant
  unconditionally at `:513`, so even a bypassed check cannot change the origin.
- `_NoRedirectHandler.redirect_request` returns `None` (`:45-49`), so a 3xx cannot move
  the request to another host.
- The CLI additionally gates remote submission behind `--allow-remote-paper-submit`
  (`paper_cli.py:207-211`); without it the run raises before a broker is constructed.
- `PaperOrder.__post_init__` rejects anything but `order_type == "market"` (`:246-249`).
- Market data is read-only: `BinancePublicData` is unauthenticated `GET /api/v3/klines`;
  `AlpacaMarketData` is `GET /v2/stocks/{sym}/bars` against `data.alpaca.markets`.
- Deny-first guard with kill switch, symbol allowlist, per-order notional, gross
  exposure, daily loss, drawdown and shorting controls (`PaperRiskGuard.assess`,
  `:317-359`), preflighted over the whole batch against a shadow portfolio before the
  first submission (`assess_batch`, `:363-392`).
- Credentials are read from `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` only
  (`paper_cli._required_environment`), never accepted in a plan file, and never written
  into evidence.

The safety posture is genuinely good. Two real weaknesses inside it:

1. **The risk gate reasons about a fiction when the remote broker is used.** `PaperAccount`
   is read from the plan JSON (`paper_cli.py:136`) and never reconciled with Alpaca —
   `GET /v2/account` and `GET /v2/positions` are never called anywhere.
   `InMemoryPaperBroker.submit` mutates the account; `AlpacaPaperBroker.submit` does not.
   So after the first remote fill, every subsequent gross-exposure, drawdown and
   daily-loss check is evaluated against a stale operator-declared portfolio, and each
   fresh CLI invocation resets to the plan's starting equity. The limits are real but
   they are measured against the wrong book.
2. **`--allow-remote-paper-submit` — the one switch between the CLI and a remote order —
   has no test.** Neither does `paper_cli._run_execute`'s live path or the `commit`
   subcommand.

Also: `PaperRiskGuard.assess`'s `"nonpositive-equity"` branch (`:326-327`) is dead —
`PaperAccount.__post_init__` (`:262-268`) already rejects non-positive equity.

---

# Remaining items, briefly

**A1 / A2 / A3 — BUILT, verified end to end.** I ran the real CLI as a subprocess
speaking stdin/stdout against a stub Ollama, exactly as `ExternalAgent::spawn` would:

```
$ python -m sharpearena.ollama_shim --model qwen3:4b --identity-out identity.json
returncode: 0
stdout: {"cost":{...},"orders":[{"action":"buy","confidence":0.5,"rationale":"",
         "symbol":"SYM00","target_weight":0.3}],"reasoning":"fake"}
identity: {"capabilities":["completion"],"context_length":40960,
           "digest":"sha256:deadbeef","family":"qwen3","format":"gguf",
           "license_sha256":"529fc91e...","modelfile_sha256":"18c3b6c9...",
           "offload":"size_vram=100; size=200", ...}
captured request contains `format` field: True
```

The renderer (`PromptRenderer`, `local_agents.py:334-364`) is a stateless
observation→messages function with no `verifiers` import — genuinely scaffold-neutral;
`OpenAICompatibleClient` reuses it unchanged. The constrained-decoding schema is the
committed contract, not a copy: `json.loads(decision_schema_json()) == json.loads(open('crates/sharpearena/contract/decision.schema.json'))`
→ `True`, and the Rust test `published_decision_schema_matches_the_protocol_types`
guards it against drift.

**A6 — BUILT.** `decide_many` fans requests across a `ThreadPoolExecutor` and the runner
feeds the results into one `env.step_batch` call (`local_agents.py:1017-1035`,
`:1097`). Note this is request-level concurrency, not batched forward passes — Ollama
has no batch API, so this is the correct reading of the requirement, but the evidence
field is `parallel_requests`, not a batch size.

**A7 — BUILT.** Sharding is `index % shard_count == shard_index` over the flattened
Cartesian list (`local_agents.py:930-935`); `plan_sha256` excludes `shard_index` and
`shard_count`, so shards share a plan hash. Resume is keyed on `status == "completed"`,
so failed cells are correctly retried. I verified the property that actually matters —
that resuming reproduces byte-identical results:

```
resume counts: {'completed': 2, 'failed': 0, 'skipped': 1}
IDENTICAL AFTER RESUME: True
```

"Sharded across cores" is process-level: the operator launches N processes with
different `shard_index`. Within one process `run()` is a serial loop over
models × datasets.

**A8 — BUILT, two gaps.** Digest, quantization level, parameter size, family, format,
context length, server + version, VRAM/total offload split, and license / modelfile /
template / parameters hashes are all captured from `/api/tags`, `/api/show`, `/api/version`
and `/api/ps` (`local_agents.py:432-500`) and land in the evidence row. Gaps: (a) no
CUDA/driver/GPU identity is recorded anywhere; (b) `reasoning_tokens` is always zero for
Ollama — `local_agents.py:552-557` is dead code:

```python
reasoning_tokens = int(response.get("reasoning_count", 0) or 0)
thinking = message.get("thinking", "")
if reasoning_tokens == 0 and isinstance(thinking, str) and thinking:
    reasoning_tokens = 0        # assigns 0 to a value that is already 0
```

`reasoning_count` is not a field Ollama returns. The comment explains the intent
honestly, but the branch is a no-op and should be deleted or replaced with a real
measurement.

**A9 — PARTIAL.** `decision_cadence` is a genuine, recorded, tested axis (the cost is
attributed only to the bar that produced the decision — `local_agents.py:1058-1062`,
asserted in two tests). "Thinking budget" is not: `SamplingConfig.thinking` is a `bool`
(`:64`), passed as `think: true/false` to Ollama and `chat_template_kwargs.enable_thinking`
to the OpenAI-compatible backend. There is no token or level budget, and no measurement
of thinking tokens (see A8).

**A10 — BUILT, two defects.** The bridge validates a complete Cartesian field and emits
a conforming `AgentSubmission` per dataset; I ran it end to end and the output parses.
Defects: `"candidates": []` is hardcoded (`bench_bridge.py:253`), which zeroes
SharpeBench's selection-robustness metric
(`sharpebench-stats-0.11.0/src/selection.rs:59-64`); and the
`execution_seeds_per_window` mislabel of Finding 3.

**B1 / B2 / B3 / B4 / B5 — BUILT.** Verified end to end with a generator emitting 5 valid
and 1 invalid candidate:

```
status: completed
requested: 6   observed_n_trials: 6
accepted: 5    rejected: [{'candidate_id':'bad','index':5,'reason':'long_when.op is unsupported'}]
selected: s4   test seeds: [3]
generated_code_executed: False
```

`parse_generated_pool` returns `len(raw)` — counted before validation and before
deduplication (`strategy_generation.py:421`) — and that count is what reaches
`score_run(series, n_trials, ...)` (`:616`). The plumbing is correct end to end. B2 is a
closed non-executable JSON DSL over six indicators; no model text is ever executed, so
the Docker boundary is genuinely not needed on this path. Splits are enforced disjoint,
including window-level disjointness when both splits share the same bytes (`:535-552`).

Three caveats on B3:
- The count covers **one generation response**. `_write_evidence` uses
  `temporary.replace(path)` (`:757`), so re-running the search against the same evidence
  path *overwrites* the previous run. Trials do not accumulate across invocations, and
  the prior evidence is destroyed rather than appended. For a multiple-testing artifact
  this is the wrong write mode — it should be append-only JSONL like the field journal.
- `requested_candidates` is embedded in the generation prompt, so in practice observed
  will equal declared; the value is that it is *measured*, not that it will differ.
- The A-track field runner still uses `precommitted_n_trials` (declared, default 1),
  propagated by the bridge as `in_sample_trials` with no cross-check. It is labelled
  `n_trials_source: "precommitted-model-entry"`, so it is disclosed rather than hidden,
  but it is on the honour system.

**D1 — BUILT, thinly tested.** Both connectors exist and are read-only. Neither is ever
exercised against a live endpoint; both tests inject a fixture transport that returns
the same canned payload for every URL. For Alpaca **no price field of the parsed bar is
asserted at all** — only `bars[0].source == "alpaca-market-data"` — so an `o`→`close`
mis-mapping would pass. For Binance only `close` is asserted, and the URL is checked
with `startswith(...)`, so the `BTC/USDT`→`BTCUSDT` normalisation the test's own input
was written to exercise is never actually verified. `AlpacaMarketData.recent_bars`
also silently returns `[]` on a malformed payload (`paper_trading.py:213`), where
`BinancePublicData` raises; the CLI catches it, but the asymmetry is real.

**D3 — BUILT and enforced, not just labelled.** Forward records carry
`evidence_class: "forward_paper_trading"`, `deterministic: False`,
`replay_guarantee: "none-live-market-and-paper-broker-state"`. The separation is
mechanical, not documentary — I tried to feed forward evidence to the bridge:

```
bridge refused forward evidence: a record has the wrong evidence_class
```

**D5 — PARTIAL.** Only the commit half exists (`prepare_forward_window_commitment`,
`paper_cli commit`); there is no reveal command. On the "byte for byte" claim at
`paper_trading.py:614`: I read the upstream source rather than trusting the docstring.
`sharpebench-attest-0.10.0/src/lib.rs:60-76` hashes `agent_id | target_window |
artifact_digest | salt |` with a trailing delimiter after every field, which is exactly
what the Python does. **The claim is true.** But it is not verified by anything in this
repo: `sharpebench-attest` is not a dependency of either crate, no `.rs` file mentions
`make_commitment`, and the only test pins a literal digest that I reproduced with
`hashlib` from the Python implementation itself — a regression pin against Python-side
drift, carrying no cross-language evidence. If upstream changes the delimiter
convention, every test here still passes.

**E1 — PARTIAL.** `paper/evidence/provenance.json` validates:

```
$ python paper/src/check-provenance.py
OK: 113 sources and 29 artifacts match the tree
exit 0
```

All nine new modules are in `source_files`. But the manifest was **not** extended to
model artifacts — there is no model digest, quantization or server-version section, and
`make-provenance.py` hashes source files only. Model identity lives exclusively in the
per-cell evidence journal (A8). That is a defensible design, but it is not what E1 asked
for, and the paper's provenance manifest cannot currently bind a model.

**E3 — BUILT.** The boundary is stated in the evidence schema itself
(`deterministic_environment: True`, `deterministic_agent: False`,
`replay_guarantee: "environment-and-recorded-decisions-only"`), in
`docs/LOCAL_AGENT_ARCHITECTURE.md:26-38`, and in
`paper/sections/09-limitations.tex:19`, which states plainly that the local-agent field
"is built, and still unrun" and that no number in the paper comes from it. That candour
is the right call and should be preserved.

**E4 — BUILT.** `.github/workflows/ci.yml` runs `cargo test --workspace` and
`python -m pytest -q`. Nothing in CI reaches a model server, downloads weights, or
touches live market data — every network boundary is stubbed. The one-cell smoke exists
as `test_field_runner_batches_scores_and_records_repetition_seed` rather than as a
separate job, which satisfies the requirement. Minor: CI installs
`pip install --upgrade pip maturin pytest numpy gymnasium verifiers` with no pins and
without `pettingzoo`/`minari`/`mcp`, so those conformance suites silently skip in CI
while `SANDBOX_ENVIRONMENT_RESEARCH_2026.md:55` warns against exactly that pattern.

---

# Placeholders, dead code, hardcoded values, and empty tests

No `TODO`, `FIXME`, `XXX` or `NotImplementedError` appears in any of the nine new
modules. The genuine items are:

1. **Dead branch** — `local_agents.py:552-557`, the reasoning-token no-op (A8 above).
2. **Hardcoded `"candidates": []`** — `bench_bridge.py:253`, disabling a downstream metric.
3. **Hardcoded `0.5` confidence** — `local_agents.py:1079-1084` (Finding 4).
4. **Dead refusal branch** — `paper_trading.py:326-327`, `"nonpositive-equity"`,
   unreachable behind `PaperAccount.__post_init__`.
5. **Tautological assertions against producer literals.** A cluster of tests asserts that
   a string constant in the module equals itself: `test_local_agents.py:133-143`
   (`evidence_class`, `deterministic_environment`, `deterministic_agent`,
   `n_trials_source`, `cost`, `cost_unit`), `test_paper_trading.py:188-195`
   (`evidence_class`, `deterministic`, `replay_guarantee`),
   `test_strategy_generation.py:152-153`. The last is the worst:
   `assert evidence["generated_code_executed"] is False` reads as the B2 safety
   guarantee but verifies a self-declared boolean. Nothing in the suite would catch a
   change that executed generated code and left the flag `False`.
6. **A credential-leak test that cannot fail.** `test_paper_trading.py:145` asserts
   `"secret" not in json.dumps(response)`. `AlpacaPaperBroker.submit` builds its return
   dict from six explicit keys and never touches `self.headers`, so the secret is
   structurally unreachable — a stub returning `{}` passes.
7. **A test whose name overstates it.**
   `test_search_selects_on_validation_and_tests_only_the_winner`
   (`test_strategy_generation.py:129-154`) never reads
   `evidence["selection"]["selected_candidate_id"]`. A runner that picked the *worst*
   candidate, or `ranking[-1]`, or always the first, passes. B5's selection rule — the
   multiple-testing object being measured — is untested, including its tie-break.
8. **Truthiness-only assertions** — `test_strategy_generation.py:218,251`,
   `test_paper_trading.py:350` assert that a sha256 hex string is non-empty.
9. **Self-consistency assertions** — `inspected["plan_sha256"] == plan.plan_sha256`
   (three CLI tests) computes both sides with the same property. No golden pins any
   `plan_sha256`, so silently dropping `datasets` from the hashed payload would pass
   every test.
10. **A blocklist standing in for schema/validator agreement.**
    `test_strategy_generation.py:85-90` asserts the substrings `"python"`, `"command"`,
    `"tool"` are absent from `STRATEGY_GENERATION_SCHEMA`. More importantly, that schema
    is only ever sent to the model as `format=`; enforcement is a separate hand-rolled
    validator (`_closed_object` / `_validate_value` / `_validate_condition`). Nothing
    asserts the two agree, so they can drift silently.

## Coverage holes worth naming

- **The real HTTP transports have zero coverage.** `OllamaClient._request`,
  `OpenAICompatibleClient._request` and `UrlLibJsonTransport` are wholly replaced by
  stubs in every test. That leaves untested: `HTTPError`→`LocalAgentError` mapping,
  `URLError`/`TimeoutError`/`JSONDecodeError` handling, and — notably —
  `_NoRedirectHandler.redirect_request` in **both** modules, the stated control that
  "paper credentials and orders never change origin."
- **`decide_many` is never executed.** Both test doubles override it wholesale, so the
  `ThreadPoolExecutor` fan-out, the `futures[future]` index→lane reordering (a
  silent-corruption risk under concurrency), and the per-future exception capture are
  untested in both near-identical copies.
- **Sharding is tested only as set arithmetic.** No test runs a sharded field, writes two
  journals and recombines them via `compile_benchmark_evidence([a, b])` — which is the
  entire purpose of `_read_journals` accepting a sequence. Every bridge test passes a
  single journal and a 1×1 field, so the model/dataset terms of `FieldCell.ordinal` are
  never exercised and roughly fifteen `_validate_field` error branches (including the
  `returns_sha256` mismatch check) never fire.
- **Most of the strategy DSL is never evaluated.** Only `momentum` is exercised, and only
  for its sign. `price`, `sma`, `ema`, `rsi` and `volatility` have no numeric assertion
  anywhere, and the `and` / `or` / `not` operators are never evaluated.
- **`openai_compatible_shim.py` has zero test coverage** — no test imports it.
- **`_run_group`'s three failure branches** — `NonFiniteReward`, `InvalidProcessTrace`,
  `InsufficientReturns` — are all untested, and `record["termination"]` is never asserted.

The strongest test in the new work, worth saying plainly, is
`test_field_runner_preserves_the_native_engine_trace_exactly`
(`test_local_agents.py:160-173`): it rebuilds a real `VecTradingEnv` from the recorded
seed, replays the recorded decisions, and asserts the journalled `trace.events` equal
the freshly-stepped events. A stub cannot fake that.

## Export coverage

`test_exports.py` spot-checks six of the new symbols (`LocalFieldRunner`, `OllamaClient`,
`StrategySearchRunner`, `PaperRiskGuard`, `PaperTradingSession`,
`prepare_forward_window_commitment`). It is not exhaustive, and it misses a real bug:

```
OpenAICompatibleClient   importable=False   in __all__=False
load_identity_manifest   importable=False   in __all__=False
```

Both are missing from `sharpearena/__init__.py` **and** from `local_agents.__all__`
(`local_agents.py:1222-1237`), even though `local_field_cli.py:23,25` imports them and
they are the entire OpenAI-compatible backend. Also unexported: `parse_generated_pool`,
`evaluate_condition`, `strategy_decision`, `target_weights_to_orders`,
`make_forward_commitment`, `LocalAgentError`, `FieldCell`, `StrategyProtocolError`.

## Documentation discrepancies

Beyond the paper claims in Finding 5:

| Location | Claim | Reality |
|---|---|---|
| `LOCAL_AGENT_ARCHITECTURE.md:32-33`, `SANDBOX:330` | fault "never converted into a hold" | true of the direct field; false of the `ExternalAgent` stdio path the same doc promotes (Finding 1) |
| `LOCAL_AGENT_ARCHITECTURE.md:182-188` | llama.cpp/vLLM/SGLang adapter is "future" work | `OpenAICompatibleClient`, `load_identity_manifest`, `openai_compatible_shim.py`, `--backend openai-compatible` and the `sharpearena-openai-shim` script all ship. Zero hits for `openai-compatible` across `README.md`, `docs/`, `CHANGELOG.md` |
| `LOCAL_AGENT_ARCHITECTURE.md:36-38` | untraceable models are "never independent field evidence" | `entry_class` is validated in `ModelRunConfig` but `bench_bridge` never reads it; an `unverified-local` field compiles identically |
| `LOCAL_AGENT_ARCHITECTURE.md:129-130` | "Docker absence is an error, never an unsandboxed fallback" | `sharpebench-arena/src/sandbox.rs` supports `allow_unsandboxed` + `unsandboxed_command`. `SANDBOX:69` states the opt-in correctly; the architecture doc drops the qualifier |
| `LOCAL_AGENT_ARCHITECTURE.md:190-196` | CI "validates sharding/resume … commitment compatibility" | resume yes; sharding only as set arithmetic; commitment only against a self-generated Python digest |
| `LOCAL_AGENT_ARCHITECTURE.md:110-115` | strategy smoke command | `strategy-search-smoke.json` has `csv_path: "../../../sharpebench/data/us-indices-1d.csv"`, outside the repo. Resolves on this host only because a sibling `sharpebench` checkout exists; fails on any standalone clone |
| `LOCAL_AGENT_ARCHITECTURE.md:127` | "only a bare loopback origin" | `OllamaClient` accepts hostname `localhost` and scheme `https` (`local_agents.py:390-392`) — name-based, not an IP-loopback assertion |
| `SANDBOX:333`, `:404` | MCP parser unification "must be done" / listed as an exit criterion | already done — `mcp_server.py:46-49` calls `parse_decision`; the same doc says so at `:97` |
| `SANDBOX:76`, `:402` | "no hostile-fixture suite"; "add a readiness command" | both shipped upstream (`sandbox.rs` `HOSTILE_PROBE`, `check_sandbox_readiness`, `sharpebench sandbox-check`). Still true that the probe covers 7 checks and §5.6 items 1/3/6/8/9 are unimplemented |
| `SANDBOX:329` | typed fault carries "raw-output hash" | on `DecisionParseError` the raw text is discarded (`local_agents.py:543-549`); `raw_response_sha256` is computed only on success. All fault classes also collapse to `failure.type == "LocalAgentError"`, so `LOCAL_MODEL_MATRIX:158`'s "record faults distinctly" holds only in the prose `detail` |
| `SANDBOX:94`, `README.md:97` | SharpeArena does Docker execution of untrusted images | SharpeArena has no Docker code and does not depend on `sharpebench-arena`, where the sandbox lives |
| `LOCAL_MODEL_MATRIX:32` | "an initial 8K–16K cap is preferable" | no `num_ctx` anywhere; `OllamaClient.decide` sets only temperature/top_p/seed/num_predict. `context_length` is read, never capped |
| `LOCAL_MODEL_MATRIX:146-156` | required per-row provenance | publisher revision, quantizer/converter version, engine commit, constrained-decoding backend, KV-cache dtype, tensor parallelism, batch size, CUDA/driver/GPU identity, raw completion and retry count are all absent. The raw completion *is* retained on the strategy path (`strategy_generation.py:681`), so the asymmetry is inconsistency rather than policy. There is no retry mechanism at all |
| `GORDON_PORT_ASSESSMENT.md` | — | the three load-bearing defect claims were re-verified and are all **true** (compactionHandler prefix slice, `withResultCache.isEnabled()` returning `true` unconditionally, `drawdownRoom` mixing a position limit with a drawdown limit). But ~9 cited paths are wrong (`signing.ts:340-466` is past EOF in a 191-line file; `src/core/strategies/strategySandbox.ts` does not exist), which breaks the traceability standard the doc sets for itself at `:35` |

Model references check out: `Qwen/Qwen3.8-27B` in the example plans is a real August 2026
release, and the `source_url` resolves.

---

# What is genuinely well done

Worth stating plainly, because most of this report is about defects:

- The Python fail-closed parser is real, complete, and I could not break it across six
  fault classes driven through the actual CLI subprocess.
- The shim, both clients, both CLIs and the bridge all work end to end against real HTTP
  round-trips. This is not scaffolding.
- Constrained decoding uses the committed contract byte-identically, with a Rust
  conformance test guarding drift.
- Observed-trial counting (B3) is correctly plumbed from `len(raw)` straight into
  `score_run`, counted before validation and before dedup. That is the subtle part of
  the build list and it is right.
- Resume produces byte-identical results, which I verified rather than assumed.
- The evidence-class separation (D3) is mechanically enforced, not merely labelled.
- The paper's limitations section states without hedging that the local-agent field is
  built and unrun and that no number in the manuscript comes from it.
- No real-capital path exists, and the layered defences around the paper endpoint are
  the work of someone who took the requirement seriously.

# Recommended order of work

1. Re-export `TransportDiagnostics`/`TransportHealth`; gate any `ExternalAgent`-based
   scoring on zero recorded faults; correct the two absolute "never a hold" statements.
2. Reconcile the `parse_decision` hold semantics with the Decision-path semantics, or
   stop telling the model that empty orders is a hold on the weight-vector path.
3. Vary the execution seed across repetitions, or stop passing
   `--execution-seeds-per-window`.
4. Stop synthesizing `0.5` confidences; emit an empty vector when none were reported.
5. Delete the "action labels must agree with their signed targets" clause from
   `05-protocol.tex:37` and drop "native" from the risk-guard descriptions.
6. Make the strategy evidence append-only rather than overwriting.
7. Export `OpenAICompatibleClient` and `load_identity_manifest`; document the
   OpenAI-compatible backend.
8. Add: a sharded run recombined through the bridge; a selection-rule test that asserts
   *which* candidate wins; coverage for the real transports and `decide_many`.
