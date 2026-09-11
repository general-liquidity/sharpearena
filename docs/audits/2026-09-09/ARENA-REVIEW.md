# Adversarial review of SharpeArena

Scope: SharpeArena as it stands on `main` at `6953538`. SharpeBench was reviewed
adversarially twice this week; Arena consumes the same kernel and publishes its own
leaderboard but had nothing equivalent. This was a review, not a repair: at `6953538` no
production code was changed, and one test file was added to demonstrate five of the
findings, `crates/sharpearena/tests/fail_open_review.rs`. A1 and A2 have since been
repaired on top of this branch; each carries a **Disposition** paragraph recording what
was changed and what it cost. Every other finding stands as written and is open.

The two patterns the sibling reviews kept finding were looked for specifically:
fail-open shapes (a missing, malformed or unknown value producing a permissive default
rather than a refusal) and tests that pass for a cause other than the one they name
(`VERIFICATION.md`, "Three tests that would have passed while the thing they name was
not what refused"). Both are present here.

The single most consequential finding is A1 and A2, the same defect in both languages:
the train/test disjointness that the whole overfitting argument rests on is carried by
an assertion that is compiled out of every configuration that ships.

## Findings

Severity is about what a wrong answer costs on a leaderboard: **high** means a published
number can be wrong or an identity guarantee can be bypassed without anything refusing;
**medium** means a permissive default or an unguarded duplicate that has not yet been
shown to move a published number; **low** means a documentation or hygiene gap.

### A1. The train/test disjointness guarantee is absent from every shipped build (high)

`crates/sharpearena/src/scenario_gen.rs:452-472`. `train_test_split` carries both of its
guarantees in `debug_assert!`:

```rust
    debug_assert!(
        train.num_levels > 0,
        "an unbounded train interval admits no disjoint test split"
    );
    let test_start = train.start_level + train.num_levels + gap;
```

`[profile.release]` in the workspace `Cargo.toml:25-27` sets only `lto` and
`codegen-units`, so `debug-assertions` and `overflow-checks` are off in release. Every
configuration a user actually gets is a release build: the crates.io crate, the maturin
wheel, the wasm bundle. In those builds an unbounded train interval (`num_levels == 0`,
which `level_seed` reads as Procgen's "unlimited", `scenario_gen.rs:440-447`) produces a
"held-out" family that lies entirely inside the train band, and nothing refuses. The
second `debug_assert!` at `:467`, the one that would catch it, is gone in the same build.
The sum at `:461` also wraps rather than panicking in release, so a large `start_level`
can land the test band below the train band.

The function's own doc comment says "a **provably disjoint** test family", and
`EVALUATION.md` says the bands are "**provably disjoint**".

Reproduce (against `6953538`, before the repair):

```
cargo test --release --test fail_open_review r5_release_build_admits_an_overlapping_train_test_split
```

The test asserted that all 64 seeds of the "held-out" family are also train seeds. It is
gated `#[cfg(not(debug_assertions))]` because CI ran `cargo test --workspace` in debug,
where the assertion still fires; that gap between the tested configuration and the
shipped one is itself part of the finding.

**Disposition: repaired.** `train_test_split` returns
`Result<(ScenarioSpec, ScenarioSpec), SplitError>`, with `SplitError::UnboundedTrain` for
`num_levels == 0` and `SplitError::BandOverflow` for a sum that does not fit in a `u64`
(`checked_add`, so the wrap at `:461` is gone as well). A typed refusal rather than a
`panic!` follows the crate's own convention for an input that cannot produce a valid
result: `SealedSalt::new` returns `Result<_, SealedSaltError>`, and `error_style.rs`
applies one `[CODE]` message register to every `Display` error the crate defines, which
`SplitError`'s two variants now join. The signature change is breaking and is recorded
under `### Breaking` in the changelog; `train_test_split` has no pyo3 or wasm export, so
the break is confined to direct Rust callers.

The release-only test is kept release-only and now asserts the refusal, plus the
disjointness of a split that is accepted, on the same inputs that previously produced a
fully overlapping family: `r5_release_build_refuses_an_overlapping_train_test_split`. The
`Err` is what isolates the cause, since nothing else in the function returns one. Two
config-agnostic unit tests in `scenario_gen.rs` cover the same two refusals, and
`.github/workflows/ci.yml` gained a `cargo test --workspace --release` step, so the
configuration the defect lived in is now a gate rather than a manual run.

`scenario_gen.rs` is one of the seven `SPEC_FILES`, so the repair rebinds `SPEC_HASH`
from `d22da4be7f050c5d` to `2eca39c3ad45a5f7` with no `SPEC_EPOCH` change; the attestation
record, the Python and npm wrapper pins and the committed wasm bundle rebind together, as
they did for the 0.20.0 and 0.21.0 pin moves. No golden, no snapshot and nothing under
`paper/evidence/` was regenerated.

### A2. The Python twin of A1, stripped by `python -O` (high)

`crates/sharpearena-py/python/sharpearena/generalization.py:40-44`:

```python
    assert n_train >= 0 and n_test >= 0 and gap >= 0
    train = list(range(seed_start, seed_start + n_train))
    test_start = seed_start + n_train + gap
    test = list(range(test_start, test_start + n_test))
    assert set(train).isdisjoint(test), "train/test seed bands overlap"
```

The docstring at `:38` says "Disjointness is asserted, not assumed", and `EVALUATION.md`
names this function as the thing that produces the split and asserts disjointness. Bare
`assert` statements are removed entirely by the CPython compiler under `-O` or
`PYTHONOPTIMIZE=1`. Under that flag both guards are gone at once: the range check that
would reject a negative `gap` and the disjointness check that would catch its effect.
`train_test_seeds(256, 256, 0, -256)` then returns two identical 256-seed bands labelled
train and test, and the generalization gap computed over them is zero by construction,
which reads as "generalizes perfectly".

This cannot be demonstrated from a test inside the package's own pytest run, because
pytest does not run under `-O`. It is a property of the language, not of this code, and
the source above is the whole evidence.

**Disposition: repaired.** Both guards are `raise ValueError` statements, which follows
the package's own convention for an argument it cannot use (`_salt_bytes` in
`eval_seeds.py`, the `n_windows` and `mode` checks in `dataset.py`). The docstring no
longer says disjointness is asserted; it says what is refused and why the check is not an
assertion. The exception type changes from `AssertionError` to `ValueError`, which is
caller-visible and recorded under `### Breaking`.

The same sweep found three more `assert` statements guarding published guarantees in the
same package, all converted: the named eval seeds' held-out band membership and uniqueness
at import time in `eval_seeds.py` (factored into `_require_held_out_band` so the guard is
callable under the flag), the sealed-seed band membership inside `sealed_eval_seeds`, and
the train-seed range check in `dataset.build_dataset`. The native-versus-Python
`EVAL_SEED_BASE` cross-check became a `RuntimeError` for the same reason.

Demonstrated in the configuration the defect lives in, not in the one pytest offers:
`crates/sharpearena-py/tests/test_optimized_guards.py` re-enters `python -O` in a
subprocess, confirms the flag reached the child without using an assertion to do it, and
compares a single printed refusal token, so a child that died for any other reason fails
the comparison. `scripts/check-optimized-guards.py` drives the same guards from a `-O`
interpreter with no pytest at all, and runs as a CI step; running pytest itself under `-O`
would be worse than useless, because the tests' own assertions would be stripped and every
one of them would pass vacuously.

A fourth surface was carrying the guarantee rather than enforcing it:
`paper/src/make-f3-generalization.py:156` recomputed the bands inline as
`range(N_TRAIN + SEED_GAP, ...)` instead of calling `train_test_seeds`. It now calls it.
The arithmetic is identical (train `[0, 16)`, test `[10016, 10032)`), so no published
number moves.

### A1/A2 follow-through: the rest of the assertion sweep

Every `debug_assert!` in the Rust crates and every bare `assert` in the Python package was
read, to separate the ones carrying a published claim from the ones carrying an internal
precondition. The published-claim ones are listed in the two dispositions above and are
repaired. These are the rest, left as they are:

- `curriculum.rs:110`, `debug_assert!(false, "recorded outcome for off-schedule level")`.
  The doc comment above it says "Unknown levels are ignored (guarded by a debug
  assertion)", so the shipped behaviour is the documented behaviour and nothing published
  claims a refusal. Recorded as A17, which stands.
- `leaderboard_ci.rs:288`, `debug_assert!(n > 0)` in the private `SplitMix64::below`. A
  precondition on a private helper whose only callers pass a non-empty length. No public
  claim rests on it.
- `check_env.py`, 18 asserts. The asserts *are* the conformance checker's body, so under
  `-O` the function reports conformance it did not check. It is a diagnostic a user runs
  deliberately rather than a guarantee the library enforces on its own operations, and
  converting it would rewrite the module rather than fix a guard, so it is recorded here
  and not changed.
- Constructor and argument checks in `pairs.py` (window, delta, obs_var), `wrappers.py`
  and `wrappers_vector.py` (`num_stack`, observation-space shape), `risk.py`,
  `obs_extra.py`, `forecast.py`, `indicators.py` and `news.py` (observation-space shape).
  These reject a caller's bad argument; none of them is a published guarantee, and four
  test files assert on the `AssertionError` they raise.
- Internal narrowing and numeric invariants: `deferred.py:609`, `edge_manifest.py:659`
  and `:669`, `local_agents.py:1390`, `trace_promotion.py:260`,
  `strategy_generation.py:846-847`, `manipulation.py:373` and `:386-387`. Not-None
  narrowing and a sum-to-one check on values the module just computed.
- `assert_no_regression` in `eval_seeds.py` already uses `raise AssertionError`, not
  `assert`, so the eval-seed regression gate survives `-O` unchanged.

**Should `[profile.release]` enable `overflow-checks`?** Not as the fix for A1, and the
repair does not depend on it. Arguments for: it would turn the wrapping sum at
`scenario_gen.rs:461` and every other release wrap into a panic, and the crate's
determinism claim means a silent wrap is a wrong number rather than a crash. Arguments
against, which win here: it changes the cost of the kernel's hot path, and the published
throughput figure under `paper/evidence/` describes a build without it, so flipping it
would make the frozen number describe a binary nobody ships; a panic is a worse outcome
than a typed refusal at a library entry point, which is the convention A1 was repaired
under; and the checks would apply to code that wraps on purpose (`wrapping_mul` and
`wrapping_add` are explicit in `mix64`, `sealed_seed` and `SplitMix64`, so those are
unaffected, but nothing guarantees the next such site will be). The narrower action is the
one taken: `checked_add` at the site where the sum carries a guarantee. Revisit the
profile flag as a deliberate, measured change with its own throughput rerun, not as a side
effect of a repair.

### A3. The npm spec-hash handshake is bypassable by a documented import path (high)

`npm/sharpearena/package.json` declares `"main": "dist/index.js"` and
`"files": ["dist", "pkg"]` and has **no `"exports"` field**. Without an `exports` map Node
permits deep subpath imports, and `pkg/package.json` declares its own
`"main": "sharpearena.js"`. So:

```js
require("@general-liquidity/sharpearena/pkg/sharpearena.js")
```

yields the raw wasm kernel with no handshake at all. The guard lives only in the module
body of `npm/sharpearena/src/index.ts:35-40`, which the deep path never executes.

This is not a hypothetical path. The package's own tests and benchmark take it:
`npm/sharpearena/test/golden.test.js:18`, `npm/sharpearena/test/spec-hash.test.js:15`,
`npm/sharpearena/bench/throughput.js:6` all `require("../pkg/sharpearena.js")`.

Worse, it is the *only* way a consumer can reach `generate_scenario`. That export exists
on the wasm (`crates/sharpearena-wasm/src/lib.rs:441`, `npm/sharpearena/pkg/sharpearena.d.ts`)
but `src/index.ts` wraps only six of the seven non-`spec_hash` exports, so the one function
the cross-runtime byte-identity claim is written about is reachable only through the
unguarded path.

Related, smaller: `checkSpecHash(engineHash, wrapperHash = SPEC_HASH)` is publicly
re-exported with the pin as a caller-supplied parameter, so `checkSpecHash(h, h)` passes
for any `h`. That does not weaken the load-time check, but the exported helper is
"compare these two", not "compare against the pin".

Reproduce: read `npm/sharpearena/package.json` for the absent `exports`, then
`node -e "const k = require('./npm/sharpearena/pkg/sharpearena.js'); console.log(k.spec_hash())"`
from the repository root, which loads and runs the kernel without `index.ts` ever
executing.

### A4. The published wasm bundle is never the one that was tested (high)

`.github/workflows/ci.yml:186-202` runs the npm job as `npm ci && npm run build && npm test`,
and `npm run build` is `tsc` only (`package.json:30`), so CI genuinely tests the committed
`pkg/`. That part is right, and is the opposite of the sibling's stale-bundle defect.

But `.github/workflows/release.yml:129-132` deletes and rebuilds the bundle:

```yaml
        run: |
          rm -rf npm/sharpearena/pkg
          wasm-pack build crates/sharpearena-wasm --target nodejs --out-dir ../../npm/sharpearena/pkg --out-name sharpearena
```

and the publish step at `:163-171` then runs `npm ci`, `npm run build`, `npm publish`.
`npm test` is never run after the rebuild. CI triggers on `push: branches: [main]` and
`pull_request` only (`ci.yml:3-6`), never on tags. So the artifact that ships is a
different binary from the one any gate saw, and it is published with neither the scenario
goldens nor the spec-hash cross-check applied to it. If the fresh build's spec hash
differed from the pin in `src/specHash.ts:13`, every consumer would get a
`SpecHashMismatch` at load and nothing in the release workflow would have noticed.

A second-order gap makes this harder to detect after the fact: the `.wasm` carries no
crate version stamp. Grepping the committed `pkg/sharpearena_bg.wasm` finds
`d22da4be7f050c5d` and `sharpebench-sim-0.21.0` but not `0.25.0`. The version test
(`test/golden.test.js:73-89`) compares `pkg/package.json` text to `Cargo.toml` text, so a
stale binary beside a correct `pkg/package.json` passes.

### A5. `mandate_breach` scores a NaN book as a clean mandate (medium-high)

`crates/sharpearena/src/mandate.rs:239-287`. Every breach source is folded in with
`worst.max(...)`, and `f64::max` discards a NaN operand; every guard is a `>` comparison,
which is false for NaN. The result is that a NaN anywhere in the weights or the returns
turns a breach into a clean 0.0 rather than a refusal, on all four sources:

- long-only: `w.iter().cloned().fold(f64::INFINITY, f64::min)` over an all-NaN bar stays
  at `INFINITY`, so the bar never counts as holding a short (`:245-249`);
- market-neutral and pairs-convergence: `if gross > EPS { net / gross } else { 0.0 }`
  contributes zero for a NaN gross (`:256`);
- inventory cap: `if gross > cap` is false for a NaN gross (`:268`);
- drawdown cap: one NaN return poisons the equity curve, and `dd > mdd` is then false for
  every later bar, so the realized drawdown reads as zero (`:197-213`).

`mandate_breach` is the graded objective. The reward layer turns it into `1 - breach`, so
a NaN book collects full credit.

Reproduce: `cargo test --test fail_open_review r1_nan_weights_and_returns_score_a_clean_mandate`.

### A6. A malformed mandate is graded as no mandate, which is full credit (medium-high)

`crates/sharpearena-py/python/sharpearena/verifiers_env.py:141-152` resolves the mandate
through `validate_mandate` and returns `None` when it does not validate;
`:154-166` then reads `None` as vacuous satisfaction:

```python
    m = _mandate_from_state(state)
    if m is None:
        return 1.0
```

`validate_mandate` (`mandate.py:104-119`) returns `False` for a present-but-unparseable
payload, an unrecognized style, `max_drawdown` outside `(0, 1]` or a non-positive
`max_inventory`. None of those is "there is no mandate". The two cases are conflated at
the point where they become a reward, and the permissive one wins.

### A7. The Python style list is an unchecked hand copy of the Rust enum (medium)

`crates/sharpearena-py/python/sharpearena/mandate.py:34`:

```python
STYLES = ("long_only", "market_neutral", "momentum", "unconstrained", "pairs_convergence")
```

against `crates/sharpearena/src/mandate.rs:36-42`. The pyo3 module
(`crates/sharpearena-py/src/lib.rs:1358-1380`) exports `sample_mandate_json`,
`mandate_breach`, `spec_hash`, `EVAL_SEED_BASE` and `MIN_SEALED_SALT_BYTES`, but no style
list, so the tuple cannot be derived and is maintained by hand. Nothing asserts
`set(STYLES)` equals the label set Rust can emit: `mandate.rs:320-335` pins the labels
Rust-side only, and `tests/test_verifiers.py:161` and `:237` are single-sample membership
checks in one direction.

Composed with A6 this is the sibling's pricing-table defect in another register: a sixth
Rust style would deserialize, fail `validate_mandate`, and be graded 1.0.

### A8. The execution-noise integrity knobs are unvalidated on every surface (medium)

`crates/sharpearena/src/exec_noise.rs:68-98` validates neither knob, and neither does any
caller: the pyo3 binding passes them straight through
(`crates/sharpearena-py/src/lib.rs:958-977`, unlike `score_run` at `:742-747` and
`validate_impact_exponent` at `:1000`, which do validate at the same boundary), the Python
wrapper does `float(delay_prob)` with no range or finiteness check
(`execution_noise.py:63-64`), and `ExecutionNoiseConfig` has no `__post_init__` while its
sibling `PreprocessingConfig` does (`preprocessing.py:41-58` vs `:97-106`). The three
behaviours:

- negative either knob: the `<= 0.0` fast path at `exec_noise.rs:76` returns the requested
  action, identical to "no execution noise configured";
- `delay_prob > 1.0`: unchecked, so `rng.next_unit() < delay_prob` always holds and the
  agent's own decisions never reach the market;
- NaN either knob: every guard is false, `scale` becomes NaN, and the realized action is
  NaN. `execution_noise.py:99` then `np.clip`s it, which propagates NaN into `env.step`.

These are declared reportable benchmark-integrity settings (`execution_noise.py:21-23`,
`preprocessing.py:208-209` marks them `[DISCLOSE]`), so the failure shape is a run that
discloses `delay_prob=-0.1` and was in fact noise-free. Also note
`PreprocessingConfig.enabled` is `self.delay_prob != 0.0 or self.slippage_bps != 0.0`, and
`nan != 0.0` is `True`, so NaN *enables* the wrapper.

Reproduce: `cargo test --test fail_open_review r2_malformed_execution_noise_is_never_refused`.

### A9. One of the five mandate styles is graded by nothing (medium)

`crates/sharpearena/src/mandate.rs:260`:

```rust
            MandateStyle::Momentum | MandateStyle::Unconstrained => {}
```

`Unconstrained` is the declared permissive control. `Momentum` is not: it renders as
"Momentum mandate: lean into recent winners, cut losers" in the prompt text
(`:120`) and is drawn with the same probability as the others, but carries no structural
rule, so an agent that ignores it entirely scores a clean mandate. The doc comment at
`:223-224` states this, but the module header at `:5-6` says each scenario draws an
objective "the episode is graded against", and `EVALUATION.md` does not qualify it. About
a fifth of sampled mandates therefore present a constraint that is scored by nothing.

### A10. Leaderboard ties are broken by declaration order and rendered as ranks (medium)

`crates/sharpearena-py/python/sharpearena/baselines.py:608-611`:

```python
    ordered = sorted(rows, key=lambda r: (
        is_kernel_score_unavailable(score(r)),
        0.0 if is_kernel_score_unavailable(score(r)) else -float(score(r)),
    ))
```

The docstring at `:594-595` says "The sort key is deflated Sharpe and *only* deflated
Sharpe". That is true, and the consequence is that `sorted`, being stable, resolves ties
by the order the rows arrive, which is `BASELINE_POLICIES` declaration order. The renderer
then prints `str(i)` as the rank (`:622`) with no tie marker and formats the score to four
decimals (`:624`). `EVALUATION.md`'s Calm table shows `kelly_vol_target` and
`equal_weight_long` both at `1.0000`, ranked 1 and 2. Either the displayed equality hides
a real difference in the fifth decimal, in which case the rank is sound but unsupported by
what is printed, or the floats tie, in which case the published rank order is an artifact
of a list literal. Which of the two it is could not be established here (see "Not
established").

### A11. The significance ranker defaults a missing score to zero (medium)

`crates/sharpearena-py/python/sharpearena/confidence.py:152`:

```python
    ordered = sorted(usable, key=lambda r: r.get("deflated_sharpe", 0.0), reverse=True)
```

A row without the key sorts as a mid-table zero rather than being refused. The `usable`
filter on the preceding lines requires a usable kernel score and mitigates this, but the
default is the wrong shape for a ranking key and does not need to be there: the same
module's `kernel_score.py` exists precisely to make "no score" unrepresentable as a number.

Adjacent, in the renderer: `baselines.py:636-638` uses `r.get("passed_k_rate", 0.0)` and
`float(r.get("mean_return", 0.0))`, so a row with no pass^k rate prints `0.00`, and
`:624` renders an unidentified entry's policy as `"?"`.

### A12. Kernel unavailability is detected by matching a field-name suffix (medium)

`crates/sharpearena-py/python/sharpearena/kernel_score.py:60-64`:

```python
        if str(key).endswith("_error") and value not in (None, "")
```

This is the gate that decides whether a kernel score may be published. It recognizes the
kernel's typed errors by naming convention rather than by contract, and the docstring at
`:56-59` explicitly future-proofs by suffix. A renamed or restructured `CompositeScore`
field yields zero detected errors, and `_kernel_value` then returns the no-skill floor as
a score, which is exactly the flattering substitution the module's own header says the
kernel refused to make. The pin is an Arena-side string convention against a pinned
dependency's field names, with no test tying the two together.

### A13. Process blocks are detected by substring (medium)

`crates/sharpearena-py/python/sharpearena/episode_outcomes.py:15-20`:

```python
    name = str(event.get("event", "")).lower()
    return (
        "manipulative" in name
        or name == "protocol_error"
        or str(event.get("severity", "")).lower() == "block"
    )
```

Any event whose name merely contains `manipulative` blocks; a manipulation event named
without that token does not. This gates `reward_eligible` and `process_check_reward`, and
`EVALUATION.md` makes process-check cleanliness part of rank eligibility.

### A14. `sharpebench-attest` is documented as a `SPEC_HASH` input and is not one (low-medium)

`AGENTS.md` states that Arena pins "`sharpebench-core`, `sharpebench-sim`,
`sharpebench-protocol` and `sharpebench-attest` at `=0.21.0` (the pin is an input to
`SPEC_HASH`, so moving it rebinds the attestation record and every wrapper pin)".
`crates/sharpearena/build_support.rs:16-20` canonicalizes only three:

```rust
    for name in [
        "sharpebench-core",
        "sharpebench-protocol",
        "sharpebench-sim",
    ] {
```

`sharpebench-attest` is a dev-dependency (`crates/sharpearena/Cargo.toml`) and is outside
the hash entirely. The committed record says so correctly
(`contract/attestation/spec-hash.json`, the `note` names three crates), so this is a
documentation defect, not a hash defect. It is easy to believe the doc because
`spec_hash.rs:82-110` asserts that all four are exact-pinned, which looks like the
enforcement of the claim and is not.

### A15. The npm `Decision` type has drifted from the published contract (low-medium)

`crates/sharpearena/contract/decision.schema.json:19-32` defines an optional `cost` object
(`cost_usd`, `tokens_in`, `tokens_out`, `reasoning_tokens`) and documents that "The
scoring kernel accumulates it into the run cost that drives the cost-normalized
leaderboard columns". `npm/sharpearena/src/types.ts:23-28` has no `cost` field and the
package has no `DecisionCost` type. `test/conformance.test.js:140-152` validates the JSON
*fixtures* against the schemas, never the TS types, which are erased before any validator
sees them. The same file's enum restatements (`Action` at `types.ts:9`, `BaselineAgent` at
`:138`, `Regime` at `:163`) are likewise hand copies with no cross-check.

### A16. `SealedSalt` enforces length, and is framed as enforcing entropy (low-medium)

`crates/sharpearena/src/scenario_gen.rs:543-546` says the sealed-seed argument "rests on
two properties of the salt, entropy and secrecy, and before this type both were
conventions a caller could quietly break". The type then plumbs secrecy properly (no
`Serialize`, no `Display`, redacting `Debug`, scrub on drop, a single `expose`) and
enforces a 16-byte *length* floor for entropy (`:561-567`). Length is a bound on entropy,
not entropy: `SealedSalt::new(b"password12345678")` is accepted, and the module's own test
at `:1339` passes `&[0u8; MIN_SEALED_SALT_BYTES]`. The `sealed_seed` doc comment is
otherwise unusually honest about what the construction is not, which is why this one
sentence stands out.

### A17. `AdaptiveCurriculum` prior is unvalidated, and off-schedule records vanish in release (low)

`crates/sharpearena/src/curriculum.rs:43-62`: `with_prior` documents `prior` in `[0, 1]`
and does not check it. A NaN prior makes every unseen level's ZPD weight NaN, and
`select_next` compares with `>` (`:121`), so the scan never displaces its seed and the
curriculum degenerates to "always the first level". An out-of-range prior yields a negative
weight the same scan cannot rank. Separately, `record` for a level outside the candidate
set is `debug_assert!(false, ...)` at `:110`, so in release the outcome is silently
dropped and the level keeps its prior forever.

Reproduce: `cargo test --test fail_open_review r3_curriculum_prior_is_unvalidated`.

### A18. Asking for zero held-out levels yields an unbounded test family (low)

`train_test_split(train, 0, gap)` sets `num_levels = 0`, which `level_seed` reads as
unlimited, so an "empty" held-out family spans the rest of the seed space. Consistent with
the documented Procgen convention for `ScenarioSpec`, and inconsistent with
`train_test_split`'s own parameter being named `n_test`.

Reproduce: `cargo test --test fail_open_review r4_zero_test_levels_yields_an_unbounded_test_family`.

### A19. Constants restated across surfaces without a cross-check (low)

- `PERIODS_PER_YEAR = 252.0` exists as `leaderboard_ci.rs:49` and as three bare literals in
  pyo3 signature defaults (`crates/sharpearena-py/src/lib.rs:740`, `:781`, `:807`, which
  import only `KERNEL_BASE_TRIALS` and `TRIALS_SR_STD_DEFAULT` from the const module) and
  again as `confidence.py:48`. Same for `DEFAULT_N_BOOT`, `DEFAULT_ALPHA` and
  `DEFAULT_RESAMPLE_SEED` (`confidence.py:45-47` against `lib.rs:781`).
  `tests/test_confidence.py:114` asserts a literal against a literal.
- `EVAL_SEED_BASE` has five Python copies (`dataset.py:24`, `gym.py:38`, `vector.py:52`,
  `effective_config.py:45`, `minari_export.py:58-61`). The native cross-check at
  `eval_seeds.py:91-96` binds only `dataset.EVAL_SEED_BASE`. `gym._EVAL_SEED_BASE`, which
  computes the seed offset every env actually uses (`gym.py:85`), is not covered.

### A20. Two event-to-weights adapters disagree (low)

`mandate.py:135` feeds the breach kernel from any event dict carrying a `weights` key;
`rewards.py:148` and `reward_misspecification.py:56` feed the turnover penalty only from
events whose `event` is `target_weights`. The two can read the same trace differently, and
nothing cross-checks them.

### A21. The spec-hash record's file list is checked by substring-searching build.rs (low)

`crates/sharpearena/src/spec_hash.rs:66-78` validates the committed record's `files` array
by asking whether `build.rs`'s *source text* contains the quoted file name, and asserts a
hardcoded count of `8` rather than `SPEC_FILES.len() + 1`. A name that appeared only in a
comment in `build.rs` would satisfy the first check, and the count does not move with the
array it claims to track. In practice the `spec_hash` equality assertion at `:53-58` is
what makes this leg hard to fool, which is the point: the file-set leg is not carrying the
weight its message implies.

## Tests that could pass for a cause other than the one they name

### T1. `exported_run_baseline_and_replay_agree_under_wasm32`

`crates/sharpearena-wasm/src/lib.rs:702-717`. Named for `run_baseline` and `replay`
agreeing under wasm32. It never calls `replay_run`. And its three assertions are that the
returns array has length 100, that calling `run_baseline` twice in one module gives the
same string, and that calling `dataset_synthetic` twice gives the same string. Nothing
binds the wasm32-compiled backtest to the native engine or to any committed fixture, so an
arithmetic divergence introduced only on the wasm32 target would leave it green.

The scenario leg does not have this problem and is the model for fixing it:
`scenario_kernel_matches_native_and_every_committed_golden` (`:644-668`, host) pins
native == kernel == committed pre-hash fixture, and
`exported_generate_scenario_reproduces_every_committed_golden` (`:681-695`, wasm32) pins
the wasm32 bytes to the same committed fixture, which makes native == wasm32 transitive.
`wasm_kernel_matches_native_engine` (`:601-624`) compares native to the JSON kernel but
both sides are host-compiled, so it does not close the wasm32 leg for runs.

Net effect: the cross-runtime byte-identity claim is enforced for `generate_scenario` and
unenforced for `run_baseline`, `replay_run`, `walk_forward`, `stress_suite` and
`tag_regime`. The npm side has the same shape: `golden.test.js` exercises only
`generate_scenario`, and `smoke.test.js` asserts shape and determinism rather than values.

### T2. The spec-hash record's file-set leg

See A21. `committed_spec_hash_record_is_current` is named for the record being current.
Its first assertion (the compiled hash equals the recorded hash) genuinely establishes
that. Its file-set assertions establish something weaker than they read as.

### T3. `test_confidence.py:114`

`assert daily == deflated_sharpe_ci(per_seed, 6, periods_per_year=252.0)` pins the default
by restating it, so it passes for any value of the default as long as the same literal is
written on both sides. Named for the daily default; tests that `252.0 == 252.0`.

## Claims checked and found sound

- **The negative-dispersion fail-open the sibling had is closed here.**
  `leaderboard_ci.rs:222`: `expected_max_sharpe` refuses a negative, NaN or infinite
  dispersion and refuses zero trials, and only returns the zero bar for the two cases where
  there is genuinely nothing to deflate (`n <= 1` or `trials_sr_std == 0.0`). The doc
  comment states the reasoning and names the sibling's rule (R02). The refusal strings are
  hoisted to constants and pinned by
  `confidence_boundaries_refuse_missing_support_and_computed_overflow` (`:634`), which
  exercises empty rows, a single row, a short row, a NaN observation, an overflowing
  observation, zero and one bootstrap draws, and NaN/0/1 alphas.
- **`checked_dsr` checks its intermediates before saturation can hide them**
  (`leaderboard_ci.rs:366`): mean, standard deviation, Sharpe, skew, kurtosis, the PSR
  denominator and the bar are all finiteness-checked *before* `norm_cdf` can clamp a bad
  value into `[0, 1]`. The comment says exactly that.
- **`kernel_score.py` returns no default on any path**, as its header claims. `_kernel_value`
  raises on any typed error, on a non-numeric value, on a bool, and on a non-finite number;
  `kernel_score_or_unavailable` converts that to a string that no arithmetic or sort can
  silently consume; `kernel_score_difference` refuses a one-sided gap rather than
  presenting the withheld side as zero. `leaderboard_markdown`'s `score()` helper routes a
  missing `deflated_sharpe` key through it and correctly ends up marking the row
  unavailable.
- **`_validated_interval` raises rather than clamping**: `baselines.py:631-633` rejects
  interval bounds outside `[0, 1]` instead of coercing them.
- **The Python spec-hash handshake covers every public entry point.**
  `python/sharpearena/__init__.py:18-25` calls `check_spec_hash(engine_spec_hash(...))` as
  the first executable statement, not wrapped in `try`, and
  `pyproject.toml`'s `module-name = "sharpearena.sharpearena_py"` places the compiled
  extension inside the package, so even `from sharpearena.sharpearena_py import ...`
  executes it first. That covers the Gym/PettingZoo envs, the market and LOB envs,
  `score_run`, the mandate and noise entry points, the bootstrap primitives, the
  `gymnasium.make` registrations, and all seven console scripts. `check_spec_hash(None)`
  raises, and `tests/test_spec_hash.py:64-65` pins that. It is an import-time check rather
  than a per-construction one, which is equivalent for a normal process; the only bypass is
  loading the `.pyd` out of the package by file path, which nothing in the repo does.
- **The npm spec-hash pin is cross-checked three ways.** `test/spec-hash.test.js:33-49`
  binds the committed `.wasm`'s reported hash, `contract/attestation/spec-hash.json`, and
  the literal in `src/specHash.ts:13`. The check itself fails closed on a missing
  `spec_hash` export. The gap is the import path (A3), not the comparison.
- **CI tests the committed wasm bundle as shipped, not a rebuild.** `ci.yml:186-202` runs
  `npm test` with `tsc` as the only build step, and the `rust` job's
  `wasm-pack test --node` (`ci.yml:129`) builds into its own temporary directory. I ran the
  committed `pkg/sharpearena_bg.wasm` directly: it reports spec hash `d22da4be7f050c5d`,
  matching both the attestation record and the TS pin, and reproduces both committed
  scenario goldens byte-exactly against their pre-hash fixtures. The sibling's stale-bundle
  defect does not exist here on `main`. The release path is the exposure (A4).
- **`clear_bar` accounting is bound exactly, not approximately.**
  `crates/sharpearena/tests/kernel_properties.rs:564-602` recomputes cash, shares, NAV and
  reward from the reported fills in the kernel's own per-symbol fold order and asserts
  exact float equality, over randomized shapes. Its stated scope is bookkeeping
  consistency; it does not claim to pin fill *pricing*, and its doc comment does not
  overstate it.
- **The terminal-state empty date is deliberate, not an unnoticed `unwrap_or_default`.**
  `market.rs:1095` and `:663` use `dates.get(..).cloned().unwrap_or_default()`, and
  `kernel_properties.rs:361` asserts `obs.date.is_empty()` past the end of the path, so the
  behaviour is pinned rather than incidental. The similar-looking
  `data.closes.get(s).cloned().unwrap_or_default()` at `market.rs:522` is also not a
  fail-open: the key comes from `data.symbols()`, and an empty series would panic at the
  next indexing operation rather than silently producing a flat tape.
- **`transport_gate` fails closed.** `CellOutcome::scored()` returns `None` for a failed
  cell rather than a default-valued `Run` (`transport_gate.rs:44-49`), and the module's
  characterization test at `:180-197` first demonstrates the unguarded engine producing a
  full 70-bar series for a wedged agent, which is a well-isolated demonstration of the
  defect the gate exists to close. Worth noting as scope rather than as a defect: nothing
  inside Arena calls `run_backtest_checked`; it is a type consumers must choose, and the
  wasm baseline path uses plain `run_backtest` (`sharpearena-wasm/src/lib.rs:217`), which is
  correct there because no wire transport is involved.
- **`sealed_seed`'s cryptographic claims are correctly bounded.**
  `scenario_gen.rs:619-628` states that FNV-1a is not collision-resistant, that the
  SplitMix64 finalizer is a public bijection, that an adversary with several `(slot, seed)`
  pairs may recover the salt digest algebraically, and what the construction does buy. The
  same honesty appears in `build.rs:28-34` about what `SPEC_HASH` does not cover. The one
  overreach is A16.
- **`no_short_never_yields_market_neutral` and `no_short_never_yields_pairs_convergence`
  are isolated.** Both scan 64 seeds. Removing the style from the filter leaves three
  styles where there were two excluded, so a false pass needs the draw to miss the style on
  all 64 seeds, probability about `(4/5)^64`, roughly `6e-7`. The assertion rests on the
  filter, not on a lucky seed range.
- **Unknown wasm config fields are refused, not defaulted.** Every input struct in
  `sharpearena-wasm/src/lib.rs` carries `#[serde(deny_unknown_fields)]`, and
  `baseline_config_refuses_unknown_fields` (`:453-470`) tests both a misspelled live field
  and a field the crate never had. The comment names the failure mode it closes.
- **The CI legs do what their names say.** Three-OS `cargo test` matrix for the determinism
  claim, `wasm-pack test --node` so the `#[wasm_bindgen]` layer actually executes on wasm32,
  `check-packaged-spec.py` for `SPEC_HASH` survival through Cargo packaging,
  `check-packaged-consumer.py` for a throwaway crate built against the extracted archive,
  an offline `npm install` of the packed tarball into a fresh project, a wheel built and
  installed into a clean venv and imported from outside the checkout, and a release
  rehearsal that performs a real version bump and tag in an isolated clone on every PR.
- **`lookahead_guard.py:72`'s environment escape hatch fails closed**
  (`os.environ.get(ENV_ALLOW_FULL_SERIES, "") == "1"`), and `edge_manifest.py:114`'s digest
  prefix check refuses rather than accepting on a short or unprefixed value.
- **`_action_validation.validated_action` refuses** shape, dtype, finiteness and bound
  violations rather than clipping them, which is the correct direction for an agent-facing
  boundary.

## Not established

Each of these needs something this review deliberately did not do, per the goal's rules
against new experiments and regenerating published evidence.

- **Whether the two `1.0000` Calm rows in `EVALUATION.md` are exactly equal floats.**
  Settling A10 as "the rank is arbitrary" rather than "the rank is real but the rendering
  hides it" requires `maturin develop` and a `run_baselines` execution. The wheel is not
  built in this worktree, and running the producer would regenerate a published number.
- **Whether rebuilding `crates/sharpearena-wasm` from the current tree reproduces the
  committed `pkg/sharpearena_bg.wasm` byte for byte.** I verified the committed bundle's
  behaviour (spec hash, both goldens) but not binary reproducibility, which needs a
  `wasm-pack build` writing into the worktree.
- **Whether A3's deep import is reachable from the published tarball exactly as from the
  repository.** `files` includes `pkg` and `prepublishOnly` removes `pkg/.gitignore`, so it
  should be, but I did not `npm pack` and unpack the result to confirm.
- **Whether the wasm32-compiled engine actually agrees with the native engine on backtest
  returns.** T1 shows nothing asserts it; whether it is in fact true needs a wasm-pack run
  that compares wasm32 output to a committed native fixture, which does not exist to
  compare against.
- **Whether `kernel_errors`'s suffix convention currently matches every field
  `sharpebench-core 0.21.0`'s `CompositeScore` can carry.** That requires reading the pinned
  dependency's source, which is outside this repository.
- **Anything needing a second host, a network file system, credentials or a live registry.**
  In particular, whether the release workflow's rebuilt bundle has ever differed from the
  committed one, which would need the published tarballs.

## Verification

Run from the worktree with
`CARGO_TARGET_DIR=C:/Users/adria/Downloads/gordon-cli-alpha/sharpearena/target`.

| Command | Exit |
|---|---|
| `cargo fmt --all --check` | 0 |
| `cargo clippy --all-targets --all-features -- -D warnings` | 0 |
| `cargo test --workspace` | 0 (166 + 4 + 4 + 1 + 4 + 9 + 1 + 4 + 2 + 13 passed, 0 failed) |
| `cargo test --release --test fail_open_review` | 0 (5 passed, including the release-only R5) |
| `python scripts/check-lean-scope.py` | 0 |
| `python scripts/check-packaged-spec.py` | 0 |
| `python paper/src/check-provenance.py` | 0 |
