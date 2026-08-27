# SharpeArena v0.17.0 to HEAD (a2442b2) independent audit

Scope: the 10 commits `git log v0.17.0..HEAD`, none of which had been independently
checked. Everything below was verified by reading source and running it. Where a claim
could not be executed it is marked UNVERIFIED with what would be needed.

Working-tree note: `git status` in this repo shows 28 modified files while
`git diff` is empty. Confirmed CRLF churn from `core.autocrlf=true`; ignored throughout
except where it breaks a gate (Finding 6).

---

## Gate results, measured

| Gate | Baseline (v0.17.0) | Measured at HEAD | Verdict |
|---|---|---|---|
| `cargo test -p sharpearena` | 145 | **153** (145 lib + 4 conformance + 4 smoke), 0 failed, 0 ignored | pass, +8 |
| `cargo test --workspace` | not stated | **164** (adds 11 in `sharpearena-wasm`), 0 failed, 0 ignored | pass |
| `python -m pytest crates/sharpearena-py -q` | 868 passed, 2 skipped | **892 passed, 2 skipped**, exit 0 | pass, +24 |
| `python paper/src/check-provenance.py` | OK, 118 sources, 29 artifacts | **FAIL: 20 problems over 118 sources, 29 artifacts, 0 model artifact manifests** | see Finding 6 |
| `latexmk -pdf` (TeX Live 2026) | 51 pages, 0 overfull, 0 undefined | **51 pages, 0 overfull, 0 undefined references, 0 undefined citations**, 33 underfull hboxes | pass |

The 2 Python skips are inverted-condition negative tests
(`test_market.py:38` and `test_pettingzoo.py:35`, reason "pettingzoo is installed"),
not missing-dependency skips.

---

## Findings, ordered by severity

### 1. HIGH: nothing in CI or any test protects the WASM leg of the byte-identity guarantee

The central claim is byte-identical determinism across native, WASM and Python.
I verified by execution that **the claim is currently TRUE**:

| Runtime | canonical golden | clustered golden |
|---|---|---|
| Rust source pin (`crates/sharpearena-wasm/src/lib.rs:527,565`) | `0xb7cf976c71219c52` | `0xa1d231f7e114a381` |
| Native via pyo3 `generate_scenario_json(7,4,120,'calm',...)` | `0xb7cf976c71219c52` | `0xa1d231f7e114a381` |
| Committed `npm/sharpearena/pkg/sharpearena_bg.wasm` under Node 24 | `0xb7cf976c71219c52` | `0xa1d231f7e114a381` |

Both WASM outputs were also byte-identical strings to the native serialization
(10246 bytes canonical), not merely equal-fingerprint.

What is wrong is that nothing enforces this:

- The two golden tests are plain `#[test]`, not `wasm_bindgen_test`. `grep -c
  wasm_bindgen_test crates/sharpearena-wasm/src/lib.rs` returns **0**. They compile
  for the host target and exercise `generate_scenario_json`, the shared kernel; the
  `#[cfg(target_arch = "wasm32")] mod wasm` export layer is never compiled during
  `cargo test`.
- CI (`.github/workflows/ci.yml:56`) only runs `cargo build -p sharpearena-wasm
  --target wasm32-unknown-unknown --release`. It never executes wasm32.
- The npm job runs `npm run build` = `tsc` (no wasm rebuild) then `npm test`.
  `npm/sharpearena/test/smoke.test.js` contains **no golden or fingerprint assertion**
  at all; `grep -rn "golden\|fingerprint\|fnv" npm/sharpearena/{src,test,bench}` is
  empty. Its strongest determinism claim is `assert.deepEqual(again.returns,
  run.returns)`, WASM against itself.
- The committed artifact is stale and mis-versioned: `npm/sharpearena/pkg/package.json`
  says `"version": "0.0.6"` while the wrapper says `0.18.0`, and
  `git diff --stat v0.17.0..HEAD -- npm/sharpearena/pkg/` is empty. The .wasm last
  changed at `f8840f2`, long before this release.

So the published npm 0.18.0 ships a WASM binary that no gate in this release
recompiled or fingerprint-checked. It happens to still match. A future Rust change that
altered the generator would ship a divergent .wasm silently, and the paper's
byte-identity sentence would become false with no test turning red.

The paper's wording in `paper/sections/04-contract.tex:27` reads "The Rust core, its
WebAssembly build and the scenario bytes returned through the Python bindings are pinned
to the byte by committed FNV-1a goldens, which the native and Python suites assert in
full and the WebAssembly suite asserts for the canonical and clustered scenarios." What
the WebAssembly suite asserts is the wasm crate's Rust source compiled natively. The
WebAssembly *build* is asserted by nothing. That sentence should say so.

Severity is HIGH because it is one of the three named central guarantees and the
protection is absent, not because the guarantee is currently broken. It is not.

### 2. HIGH: the tamper-evidence gate is not run by CI

`grep -n "provenance" .github/workflows/ci.yml` returns nothing. The CI jobs are
`rust` (fmt, clippy, rustdoc, `cargo test --workspace`, wasm build), `supply-chain`
(cargo-deny), `npm`, and `python` (fmt, clippy, maturin, pytest). Neither
`make-provenance.py` nor `check-provenance.py` runs anywhere, and the paper is never
built in CI either.

`paper/sections/A-commands.tex` states that "`check-provenance.py` recomputes every
recorded digest, re-expands both artifact scopes and exits nonzero on any mismatch, so a
manifest that has fallen behind its tree is a detectable failure rather than a silent
one." Detectable only by a human who remembers to run it. As of this release nobody has
automated the one gate that carries the recompute-from-decisions tamper-evidence story
into the build.

### 3. MEDIUM: the model-artifact provenance feature is untested code over zero data, and its validator is weaker than its writer

The 0.18.0 CHANGELOG advertises "a separately validated model-artifact scope that
exposes checkpoint digest, quantization, server and server version". Measured:

- The manifest contains **0 model artifacts** (`len(manifest['model_artifacts']) == 0`).
- The scope directory holds only `paper/evidence/model-artifacts/.gitkeep`. The glob is
  `*.json`, so `expand()` returns an empty list and the entire branch in
  `check-provenance.py` (lines ~100-120) is dead. `check_group("model artifact", [])`
  is a no-op. This is a gate whose condition is currently constant.
- `grep -rn "make-provenance\|check-provenance\|model_summaries\|model_artifact"
  crates/sharpearena-py/tests/ .github/workflows/` returns **nothing**. Neither script
  has a single test.

I exercised the untested path directly and found a real asymmetry:

- `make-provenance.py:model_artifact_records` rejects a field whose value is
  `"unknown"` or `"unresolved"` (raises `ValueError ... lacks explicit fields`).
- `check-provenance.py:model_summaries` does not. Feeding it
  `{"model":"m","digest":"unresolved","quantization":"q","server":"s","server_version":"1"}`
  returns the summary cleanly with `digest: "unresolved"`.

So a hand-edited manifest carrying an unresolved checkpoint digest would pass the
validator that the generator would have refused to write. That is exactly backwards for
a tamper-evidence artifact, where the checker is the trusted half.

The paper is honest here even though the CHANGELOG is not: `A-commands.tex` says "An
empty model-artifact list therefore means no model result has entered the paper, rather
than that model provenance was omitted." The CHANGELOG bullet reads as if the scope is
populated.

### 4. MEDIUM: the optional-dependency declaration is incomplete, and the CI-fix pattern was not what it looked like

The four CI commits (`9a764e7` Pillow, `744b9f7` h5py, `7adde43` JAX, each followed by a
provenance rebind) do **not** correspond to newly added tests.
`git log v0.17.0..HEAD -- crates/sharpearena-py/tests/test_minari.py` is empty; the file
last changed at `80ff730` (the OpenOutcry rename). What happened is that CI previously
installed only `pytest numpy gymnasium verifiers`, so every Minari test had been
silently skipping in CI for its whole life. `.github/workflows/ci.yml:137` now installs
`crates/sharpearena-py/ci-requirements.txt`, which unmasked them, and each missing
transitive dependency surfaced one at a time. The CHANGELOG line
("eliminating environment-dependent silent skips") is accurate and this is a genuine
improvement.

Degradation behavior is correct, not silent: `test_data_collector_works_on_live_env`
(`test_minari.py:264`) calls `minari.DataCollector`, whose `add_step_data` needs jax and
whose default storage needs h5py. With minari installed and jax absent it raises, it does
not skip. That is why the fixes were forced.

The dishonest part is the declaration. `pyproject.toml` declares
`[project.optional-dependencies] minari = ["minari"]`. minari 0.5.3's own metadata puts
`h5py>=3.8.0` behind `extra == "hdf5"` and `jax[cpu]` behind `extra == "create"`. Neither
is a base dependency. A user running `pip install sharpearena[minari]` gets exactly the
environment CI had before this release. The correct declaration is
`minari = ["minari[hdf5,create]"]` plus pillow. `ci-requirements.txt` is the only honest
record and it is not shipped in the wheel.

Two further gaps: nothing asserts that `ci-requirements.txt` is complete or that the
suite produced no unexpected skips, so the next optional dependency reintroduces the same
silent-skip hole; and local pillow is 11.2.1 against the CI pin of 12.1.1.

Also worth noting: `minari_export.py:39` states "`minari.create_dataset_from_buffers` is
the only entry used, so the heavy `jax` dependency of `EpisodeBuffer.add_step_data` is
avoided by assembling buffers by direct field assignment." That is true of `to_minari`
but the test suite needs jax anyway via `DataCollector`, so the module docstring reads
more reassuring than the build is.

### 5. MEDIUM: prior-audit tautologies are only partly fixed, and the worst one survives

`paper/review/local-agent-audit-2026-08-26.md` listed ten items. Status now:

Fixed:
- #1 dead reasoning-token no-op: removed; replaced with an honest
  `reasoning_tokens_available = "reasoning_count" in response` flag.
- #2 `bench_bridge.py` hardcoded `"candidates": []`: now reads from
  `representative["model_config"]` (`bench_bridge.py:288`).
- #3 hardcoded `0.5` confidence: gone. `grep -n "0\.5" local_agents.py` is empty.
- #4 dead `nonpositive-equity` refusal branch: removed. I checked that this is safe
  rather than a weakened guard. `AccountSnapshot.__post_init__` (`paper_trading.py:537`)
  enforces finite and strictly positive equity, so `reconcile_account`, the only mutator
  of `account.equity`, cannot write a non-positive or NaN value; `_projected_gross`
  retains its own `if account.equity <= 0.0: return math.inf`; and even at equity 0 the
  daily-loss branch fires because `max_daily_loss` is constrained to `[0, 1)`. Verified
  safe, but the removal is not in the CHANGELOG.
- #7 selection test that never read the winner: fixed.
  `test_strategy_generation.py:322` now asserts
  `evidence["selection"]["selected_candidate_id"] == "contrarian"` and `:323` pins the
  call sequence `[["trend","contrarian"], ["contrarian"]]`.
- Most of the #5 cluster in `test_local_agents.py:133-143` is gone, replaced by
  `test_embedded_decision_schema_is_the_published_closed_contract`, a real schema pin.

Not fixed:
- **#5, the one the prior audit called "the worst":**
  `test_strategy_generation.py:274` still reads
  `assert evidence["generated_code_executed"] is False`, and
  `strategy_generation.py:861` and `:924` still write the literal `False`. The
  assertion verifies a self-declared boolean. Mitigating: I confirmed the claim is
  substantively true. `grep -nE "\b(eval|exec|compile|__import__|subprocess|os\.system)\s*\("`
  over `crates/sharpearena-py/python/sharpearena/*.py` matches only `re.compile` in four
  places. The DSL is genuinely parse-only. But nothing in the suite would catch a change
  that added execution and left the flag at `False`.
- #6, the credential-leak test that cannot fail:
  `test_paper_trading.py:168`, `assert "secret" not in json.dumps(response)`.
  `AlpacaPaperBroker.submit` builds its return dict from six explicit keys off the broker
  response and never touches `self.headers`, so the secret is structurally unreachable.
  A stub returning `{}` passes.
- #8, truthiness-only assertions on hex strings: still four of them,
  `test_strategy_generation.py:270,387,420` and `test_paper_trading.py:467`.
- #9, self-consistency `plan_sha256`: still unpinned.
  `test_local_agents.py:804` and `test_strategy_generation.py:486` compute both sides
  from the same property, and `grep -rnoE '"[0-9a-f]{64}"' crates/sharpearena-py/tests/`
  finds pinned digests only in `test_adverse_selection.py`. Dropping a field from the
  hashed payload would still pass every test.

New weak assertion introduced in this release:
- `test_every_dsl_indicator_and_boolean_operator_has_numeric_semantics`
  (`test_strategy_generation.py:203`). The CHANGELOG describes this as "every
  strategy-DSL indicator and Boolean operator is numerically pinned". It is not pinned;
  every assertion is a one-sided inequality. I computed the actual values against the
  thresholds:

  | indicator | actual | threshold | op | slack |
  |---|---|---|---|---|
  | price | 8.0 | 7.0 | gt | 1.0 |
  | sma | 4.666666666666667 | 4.666666666666667 | gte | 0.0 |
  | ema | 5.5 | 5.49 | gt | 0.01 |
  | momentum | 3.0 | 2.99 | gt | 0.01 |
  | rsi | 100.0 | 99.0 | gt | 1.0 |
  | volatility | 0.04714045207910321 | 0.01 | gt | 0.037 |

  `sma` is tight. `ema` and `momentum` are within 1 percent. `price`, `rsi` and
  especially `volatility` (4.7x the threshold) are loose: a volatility implementation
  returning variance instead of standard deviation, or population instead of sample, or
  an annualized figure, passes unchanged. There is no false-side case for any indicator
  except `trend.long_when`. Equality assertions on the computed values would make the
  CHANGELOG sentence true.

### 6. MEDIUM: one numeric claim in the paper is wrong by a factor of two

`paper/sections/07-findings.tex:161`, final sentence:

> On this mandate-blind reference field, roughly half of the failures are process
> failures that a return-ranked board does not record.

Recomputed from `paper/evidence/f7-failures.json` by counting `episodes[*].mode`
(confirmed against the committed `rollup_overall`):

```
total 384   clean 190   failures 194
  mandate_structural 103
  mandate_drawdown    57
  mandate_inventory   29
  stopped_out          5
process (mandate_*) = 189
189 / 194 failures  = 97.4%
189 / 384 episodes  = 49.2%
```

Process failures are **97.4% of failures**, not half. The arithmetic works exactly for
episodes, so the sentence almost certainly means "roughly half of the *episodes* end in
a process failure" (49.2%). As printed it understates the paper's own headline point by
a factor of two, in the direction that weakens it.

This is pre-existing, not introduced in this release: `07-findings.tex` is not in
`git diff v0.17.0..HEAD -- paper/sections/`. It is still an error shipping in the
built 51-page PDF.

Two smaller provenance-adjacent items found alongside it:

- `paper/evidence/f1-baselines.json` records `"package_version": "0.9.0"` while the
  workspace is at 0.18.0. Committed evidence carrying a nine-versions-stale package
  label undercuts the appendix's claim that the content snapshot identifies the source
  state the numbers came from.
- `06-validation.tex` (F6 paragraph) states "mean 239 filled units over the 48
  exogenous-arm episodes" and "a parent order fills a mean 22.4 of its 60 offered
  units". **UNVERIFIABLE**: `f6-adverse-selection.json` commits `filled_qty` only for
  `detail_episode_makers` at `detail_seed` 0, which sums to 227, a single episode.
  Consistent with a 48-episode mean of 239 but not evidence for it. Verifying needs the
  F6 producer to emit per-episode filled quantity across all 48 arm episodes, or a
  `mean_filled_qty` field. Every derived quantity that depends on these (V=240,
  0.042 price units per unit, 0.81% displacement) does check out against committed keys.
- `witness.json`'s own note `noise_replicates.config.crossing_point` says
  "half-width = bisect_resolution / 2" with `bisect_resolution` 0.005, implying 0.0025.
  The paper says 0.0016 and the paper is right: every attained bracket is 0.00312 wide.
  The stale annotation is in the producer's JSON, not the paper.

Everything else with a machine-readable source reconciles. All 18 throughput figures in
`05-protocol.tex` match `throughput.json` medians exactly (native 747,468 steps/s, WASM
508,127, Python 14,240, the ~52x ratio, 4,608 episodes / 552,960 steps / 302 s /
~1,830 steps/s); every seed count in F1 through F8 plus predictability, sealed-seeds and
the witness matches its `config.seeds` length; and all of F2, F3, F5, F6, F7 and F8's
reported statistics match their committed keys, including recomputed paired t-tests
(Calm t(15)=3.360, p=0.0043) and the 7-of-23 / 0-of-23 concave sweep slot counts.

### 7. LOW (environmental, but it breaks the gate as given): provenance fails on any `core.autocrlf=true` checkout

`check-provenance.py` reports FAIL with 20 digest mismatches. All 20 are the files git
also lists as modified. This is entirely CRLF: `sha256(path.read_bytes())` hashes
on-disk bytes, and the manifest was generated on LF bytes. I verified this directly.

For all three files I sampled, the recorded digest equals the LF git blob exactly:

```
crates/sharpearena/src/transport_gate.rs
  git blob (LF)  ee10655f66533fd7885101b263e21140d0c2361ccdd1e5de0842c53c38b29ba4
  recorded       ee10655f66533fd7885101b263e21140d0c2361ccdd1e5de0842c53c38b29ba4
  on disk (CRLF) 2e1dca4b40ce7a71f28fdb19c8a7178b2bbda5469fe76f6d5923535152fe1c4c
```

Recomputing every entry against `git show HEAD:<path>` gives:
**118 sources, 0 mismatches, 0 missing; 29 artifacts, 0 problems; source snapshot
recomputes to `b87a383feeead3ee7563e57f8ecc93ce285a8a4ad4327b96064920196bfa808e`,
matching the recorded value.**

So there is **no real provenance drift**. The manifest is exactly correct against the
committed tree. But the appendix claims "the same tree yields the same snapshot hash on
a different machine" and that is false on Windows without a `.gitattributes`
(the repo has none). Fixing this needs either `* text=auto eol=lf` in `.gitattributes`
or normalizing newlines in `sha256()`.

On the specific provenance questions asked:

- **No digests are quoted in the paper.**
  `grep -rnoE "[0-9a-f]{16,64}" paper/sections/ paper/main.tex README.md` returns
  nothing. The known self-reference issue is therefore moot; the appendix describes the
  manifest without quoting it.
- `generated_at_head` is `7adde439...` while HEAD is `a2442b21...`. That is the
  expected one-behind: the manifest records the commit it was generated at, and the
  commit that carries it is necessarily the child. `generated_at_head_dirty` is `false`.
- Commit `7fad11b` "bind the CI dependency fix" shows 634 changed lines but changes
  **zero** semantics: I diffed the parsed manifests at `21836d0` and `7fad11b` and got
  0 changed hashes, 0 added, 0 removed paths, identical snapshot digest. It is a
  whole-file line-ending rewrite. Cosmetic noise in the history of the tamper-evidence
  artifact, which is a bad place for noise.
- The three "bind the CI fix" commits cannot do what they say. `SOURCE_SCOPE` in
  `make-provenance.py` covers `*.toml`, `*.rs`, `python/**/*.py`, `paper/src/*.py`,
  `*.tex`, `refs.bib`. Neither `.github/workflows/ci.yml` nor `ci-requirements.txt` is in
  scope, so the manifest provably cannot bind a CI change. All three commits changed only
  `generated_at_head`.
- `21836d0` "record a clean v0.18.0 generation head" changed exactly two lines,
  `generated_at_head` and `generated_at_head_dirty: true -> false`, with no hash
  regeneration. That is consistent with a regeneration on a clean checkout of `bc730c3`,
  but it is indistinguishable from hand-editing the manifest's own honesty flag.
  UNVERIFIED. Distinguishing them would need a reproducible-generation check in CI,
  which is Finding 2.

### 8. LOW: things in `git log v0.17.0..HEAD` the CHANGELOG does not mention

- **`__version__` was wrong and was silently fixed.**
  `crates/sharpearena-py/python/sharpearena/__init__.py` went `"0.16.0"` to `"0.18.0"`.
  At the v0.17.0 tag the shipped package reported itself as 0.16.0 while
  `pyproject.toml` said 0.17.0. That is a released-artifact identity bug and its fix
  deserves a Fixed line.
- **About 25 new public exports** added to `__all__`: `LocalAgentError`,
  `ModelHttpError`, `ModelResponseError`, `ModelTransportError`, `DecisionResponseError`,
  `DecisionModel`, `FieldCell`, `InferenceOutcome`, `InferenceResult`,
  `OpenAICompatibleClient`, `load_identity_manifest`, `CandidateRejection`,
  `GenerationResult`, `StrategyGenerator`, `StrategyProtocolError`, `evaluate_condition`,
  `parse_generated_pool`, `strategy_decision`, `make_forward_commitment`,
  `prepare_forward_window_reveal`, `target_weights_to_orders`. A public API expansion of
  that size with no CHANGELOG line is a support liability.
- **MCP behavioral change**: `mcp_server.py` now tracks `last_observation` and returns a
  new `{"error": "episode_not_reset", "environment_advanced": false}` state when `step`
  precedes `reset`. New error contract on a shipped tool surface, unmentioned.
- **New CLI flag** `--context-tokens` on `sharpearena-ollama-shim`.
- **New dev-dependency** `sharpebench-attest = "0.12.0"` on `crates/sharpearena`,
  adding 30 crates to `Cargo.lock` including `ed25519-dalek`, `curve25519-dalek`,
  `sha2`, `getrandom`. Dev-only, so the shipped crate is unaffected, but note that
  `paper/refs.bib:128` still cites SharpeBench as "Version 0.11.0" while the repo now
  consumes a 0.12.0 crate alongside `sharpebench-core 0.11.0`.
- **Removal of the `nonpositive-equity` refusal** from the money-path risk guard
  (verified safe, see Finding 5, but a deleted deny-list branch belongs in a changelog).
- `paper/evidence/model-artifacts/.gitkeep` added (Finding 3).

---

## Things that are correct and well done

**The transport gate is intact and honestly documented.** `transport_gate.rs` still
converts a recorded fault into `CellOutcome::Failed(TransportFault)`;
`scored()` returns `None` on a failed cell rather than a default `Run`;
`lib.rs:82-83` re-exports `DecideError`, `TransportDiagnostics`, `TransportHealth`,
`run_backtest_checked`, `transport_fault`, `CellOutcome` and `TransportFault`. Five
tests cover it including the characterization test
`a_wedged_agent_yields_a_scoreable_series_through_the_unguarded_engine`, which keeps the
defect visible in the suite rather than only in prose.

On consumers: a repo-wide grep still finds **zero production callers** outside the
module's own tests, docs and the changelog. The project says so itself, in both places
it matters. `paper/sections/04-contract.tex:21`: "the type system does not force a
wire-transport consumer to choose the checked entry point; repository consumers outside
the tests do not yet call it." `README.md:88`: "the type system does not force a wire
consumer to choose the checked function, so callers must do so deliberately." That is
the honest framing, and the Python local-field scheduler does enforce the equivalent rule
directly (`local_agents.py:1275-1330` marks a failed lane and stops it rather than
holding). This is a disclosed gap, not an overclaim.

**The paper-execution state machine has no real-capital path.** Verified line by line:

- Origin is validated *and then unconditionally reassigned*:
  `paper_trading.py:1017-1025` rejects any base URL whose
  `scheme://netloc` is not `https://paper-api.alpaca.markets` or whose path is not
  `""`/`"/"`, then sets `self.base_url = ALPACA_PAPER_ORIGIN` regardless of input.
- Redirects are refused: `_NoRedirectHandler.redirect_request` returns `None`, installed
  via `build_opener(_NoRedirectHandler())`, so a 3xx surfaces as `HTTPError`. A live
  loopback test (`test_local_transports.py:96`) asserts
  `pytest.raises(ModelHttpError, match="HTTP 302")` against a real redirecting server.
- Market-only: `PaperOrder.__post_init__` raises
  `"the forward safety profile permits market paper orders only"` on any
  `order_type != "market"`; side is restricted to `{buy, sell}`; quantity must be finite
  and positive.
- Deny-first with kill switch: `PaperRiskGuard.assess` checks `kill_switch` first, then
  allowlist, then price validity, then daily-loss, drawdown, notional, gross exposure,
  shorting. Missing or invalid price forces `gross = math.inf` before any comparison,
  so absence of data denies rather than passes. `assess_batch` walks a shadow portfolio
  so a batch cannot exceed the gross cap by passing order-by-order.
- Both new remote calls are GETs: `find_order` GETs
  `/v2/orders:by_client_order_id`, `account_snapshot` GETs `/v2/account` and
  `/v2/positions`. The only POST is `/v2/orders`, to the reassigned paper origin, gated
  by the risk verdict. An unanswerable query raises `SubmissionUnknown` rather than
  resolving to a confirmed absence.

What the code can do with money: submit, query and cancel-adjacent reads against
`https://paper-api.alpaca.markets` only, market orders only, subject to the guard.
What it cannot do: reach any live venue (origin is overwritten, not merely checked),
follow a redirect to one, place a limit or stop order, short without
`allow_shorting`, or act at all with the kill switch set. No override path exists.

**`target_weights_to_orders` sparse-order fix is real.** `paper_trading.py:1157-1173`
now skips symbols absent from `decision["orders"]` instead of flattening them to zero,
and computes `delta = target - current` against `account.positions`. Before this, an
omitted symbol produced a liquidation order. The `current_weights=[0.0]*len(symbols)`
argument looks alarming but is neutralized by the skip. The same call in
`local_agents.py:699,952` is validation-only (return value discarded); the field runner
hands the raw `Decision` JSON to the native `env.step_batch`, so sparse semantics are
resolved in Rust, not Python.

**The cross-language commitment fixture is a genuine pin.**
`crates/sharpearena/contract/attestation/forward-commitment.json` holds a committed
`commit_hash`. `smoke.rs:forward_commitment_matches_the_published_attestation_primitive`
regenerates it through the published `sharpebench_attest::make_commitment`;
`test_paper_trading.py:306` regenerates it through the Python
`make_forward_commitment` and also asserts the delimiter rejection. Both read the same
file; neither can drift without the other failing.

**The new loopback transport tests are real, not fixtures.**
`test_local_transports.py` stands up a `ThreadingHTTPServer` on 127.0.0.1 and covers
invalid JSON, HTTP 503, HTTP 302 refusal, an unreachable port, and concurrent lane
ordering with a per-lane fault. These can fail.

**Local-model provenance is honest by construction.** `ModelIdentity` now defaults 14
backend fields to the literal `"unresolved"` rather than inventing values,
`_local_gpu_identity()` returns `{}` on any failure instead of guessing, a numeric
thinking budget on Ollama raises before a request is sent, `source_revision` rejects
`main`/`master`/`latest`/`head` as mutable, and `LocalAgentError` is split into
`ModelHttpError` / `ModelTransportError` / `ModelResponseError` /
`DecisionResponseError` carrying the raw-response hash.

**The containment wording is correct.** `README.md:114`: "SharpeArena provides no
process or container isolation and does not claim any... The fail-closed OCI containment
path for *untrusted* entrants lives in the sibling `sharpebench-arena` crate."
`09-limitations.tex` attributes it the same way, states "this crate contains no
containment code at all", and correctly narrowed the previous "never been observed to
hold" to "A continuous-integration job on a Docker-enabled runner has since executed
both inside a live container. That is the first evidence the boundary holds at all, and
it is narrow evidence: one runner image, one pinned fixture tag, and no run on the
development machine." It still says "Nothing here should yet be read as evidence that
hostile code is contained." That is the right calibration.

**UNVERIFIED, cross-repo:** the claim that the sibling `sharpebench-arena` container
tests have now run in CI, and the claim in
`docs/SANDBOX_ENVIRONMENT_RESEARCH_2026.md:75` that "The shipped hostile probe covers
seven boundary checks", cannot be checked from this repository. Verifying them needs a
green run link or workflow log from the SharpeBench repo. Nothing in SharpeArena
overclaims on the basis of them.

**The 23-of-24 headline still matches its source.** Independently recomputed from
`paper/evidence/f4-realism.json`:

- `config.seeds == [0,1,2,3,4,5,6,7]` (8 seeds) x 3 tiers = 24 panels. Matches
  "24 seeded panels" and every "8 seeds" claim in the F4 sections.
- Unclustered `tiers` pass rates: calm 0.0, hard 0.0, extreme 0.125 -> 0/8, 0/8, 1/8 =
  1 pass of 24, i.e. **23 of 24 fail**. Matches
  `00-abstract.tex`, `01-introduction.tex:31`, `03-environment.tex:112`,
  `06-validation.tex:5`, `07-findings.tex:5` and `README.md:47`, including the
  README's explicit "(0/8 Calm, 0/8 Hard, 1/8 Extreme)".
- Clustered (`vol_clustering: 0.5`) pass rates: 0.0, 1.0, 0.5 -> 0/8, 8/8, 4/8. Matches
  `README.md:47` exactly.
- No test count appears anywhere in the paper or README, so there is nothing to drift.

---

## Summary

No central guarantee was quietly weakened in this release. Leak-freedom is untouched,
the transport gate is intact and its zero-consumer status is disclosed in both the paper
and the README, the money path still has no real-capital route, containment is still
attributed to the sibling crate, and I confirmed by execution that native, WASM and
Python all reproduce both committed golden fingerprints byte for byte.

The defects are mostly about what protects those guarantees rather than the guarantees
themselves: the WASM leg has no enforcing test and ships a stale, mis-versioned artifact;
the provenance gate is not in CI and fails outright on a Windows checkout; the
model-artifact provenance feature is untested code over an empty scope whose validator is
laxer than its writer; the optional-dependency extras are incomplete; and the worst
tautological assertion the previous audit found is still in the suite, along with three
other unfixed items from that list and one new overclaimed test.

The one substantive content error is in the paper itself and predates this release:
`07-findings.tex:161` says process failures are "roughly half of the failures" when the
committed artifact says 189 of 194, or 97.4%. It is 49.2% of *episodes*, so the fix is
one word, but it is currently wrong in the built PDF.
