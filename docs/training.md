<!-- prettier-ignore -->
<div align="center">

# Training on SharpeArena

### From `vf-eval` baseline to a GRPO run, on a leak-free trading floor

</div>

---

SharpeArena ships a real, trainable [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
environment: `sharpearena.load_environment()` returns an `SharpeArenaVerifiersEnv`
(`vf.MultiTurnEnv`) that drives the point-in-time market one bar per turn and scores the
realized return series with the SharpeBench kernel. This guide covers running it under
`vf-eval` and training on it with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

For a copy-pasteable end-to-end config, see [`examples/prime-rl/`](../examples/prime-rl/).

## The env-package layout

The env is a standard installed Python package; the agent contract and Rust engine live
behind a pyo3 wheel built by [maturin](https://www.maturin.rs).

```
crates/sharpearena-py/
  pyproject.toml                     # package metadata + [tool.verifiers.eval] + tags
  python/sharpearena/
    __init__.py
    verifiers_env.py                 # load_environment(), the rubric, the multi-turn rollout
    dataset.py                       # build_scenario_dataset() - the multi-row taskset
    gym.py                           # the underlying gymnasium.Env stepped per turn
    sharpearena_py...pyd              # the compiled Rust kernel (SharpeBench scorer)
```

`load_environment(**args)` is the entry point `verifiers` and prime-rl call. It accepts
`n_symbols`, `n_days`, `n_windows`, `max_episode_bars`, `max_turns`, `max_weight`, and
`allow_short`. `args.n_windows` sets the dataset length, which is the number of GRPO
tasks (`num_tasks`) - it must be `> 1` so within-group reward variance has something to
vary over.

## Install and discover

```bash
pip install -e "crates/sharpearena-py[verifiers]"   # editable, with the verifiers extra
```

Once published to the [PrimeIntellect](https://app.primeintellect.ai) Environments-Hub,
the env is installable by org-qualified id:

```bash
prime env install general-liquidity/sharpearena
```

The `[project].tags` list in `pyproject.toml` feeds Hub discoverability (the `prime` CLI
reads it at push time); the `[tool.verifiers.eval]` table sets the default
`num_examples` / `rollouts_per_example` for a bare `vf-eval sharpearena`.

## Baseline with `vf-eval`

```bash
vf-eval sharpearena \
  -m Qwen/Qwen3-1.7B -n 20 \
  -a '{"n_windows": 20, "n_symbols": 4, "n_days": 120, "max_episode_bars": 16}'
```

The rubric's dense `realized_return_reward` is the GRPO objective; the real
`deflated_sharpe_reward` is a secondary objective; `pass_k_reward`,
`process_check_reward`, and `format_reward` are zero-weight diagnostics.

### Episode eligibility and failed rollouts

The weighted rubric first checks the runner-owned `state["episode"]` record.
It records the requested cap, available native window, planned horizon
(`min(requested_bars, available_bars)`), realized bars, terminal status and reason.
The available window is read back from the environment that consumes it.

Only a complete, process-clean episode earns the configured rewards. Protocol
errors, bankruptcy, block-severity process events, incomplete horizons, framework
cutoffs and inconsistent evidence receive a composite reward of **-1** under
all eight schemes. The primary contribution is -1; Sharpe and mandate
contributions are zero. Zeroing the whole reward instead would allow an abort
to beat a valid losing episode. Raw return/event traces remain available for
diagnosis; the runner does not fabricate holds or unobserved returns.

The environment closes once on completion, failure or framework cleanup. A
terminal state cannot start a fresh market, and the final observation does not
request another model response. Raw helpers such as `realized_return_reward`
still describe partial traces, but are not substitutes for the gated training
rubric. Hand-built rubric inputs without episode accounting now receive the
failure floor. Time-varying volatility aversion uses the recorded planned
horizon unless an explicit diagnostic horizon is supplied.

These rules constrain reward accounting. They do not establish training
convergence, positive within-group variance or resistance to every reward exploit.

### Reward-misspecification diagnostics

`reward_misspecification.py` keeps incomplete research rewards outside the production
registry. Its comparisons use hand-written policies, not agents trained to maximize
those rewards. A gap between them does not establish a causal effect of reward design,
an out-of-sample generalization gap, or guaranteed underperformance.

The indicator proxy now follows the sign of the last three **realized portfolio
returns**, matching `indicator_shaped`'s signal and window. The runner delivers each
observed return through `observe_return()` before the next action. The recency proxy
instead follows the latest per-symbol price change. A synthetic reversal fixture
requires opposite actions from the two policies. The indicator policy aligns the
current signal; it does not optimize the future sequence of rewards.

`misspecification_gap` reports `comparison_kind="heuristic_policy_comparison"`,
`optimization_performed=false`, and `clean_reward_role="label_only"`. The legacy
`demonstrate_punishment` name remains, but its output is a diagnostic table, not an
eligibility decision. Existing evidence is not regenerated by this repair.

## v1 taskset and the subprocess runtime

Under the verifiers v1 contract the env is a **taskset** (`taskset = { id = "sharpearena" }`)
composed with a harness and a runtime. With no container image declared, it runs on the
**subprocess runtime**: prime-rl spawns a local env-server subprocess that imports the
installed package and serves rollouts over the worker pool. That is the right default
for a pure-Python + pyo3 env with no external services. Drive it from prime-rl with
either the v1 `taskset = { id = "sharpearena" }` form or the legacy `id = "sharpearena"`
form shown in [`examples/prime-rl/rl.toml`](../examples/prime-rl/rl.toml).

## Leak-freedom: point-in-time is ours, the split is yours

Two distinct guarantees:

1. **Within a scenario - structural, ours.** The data layer has no API to read a future
   bar; the environment owns the time cursor. An agent cannot peek, by construction.
2. **Across train and eval - experimental, yours.** A trustworthy training run must
   evaluate on a **strictly held-out seed band**. `build_scenario_dataset` supports this
   directly: `mode="train"` draws seeds from base `0`, `mode="eval"` from
   `EVAL_SEED_BASE = 1_000_000`, and the two ranges are asserted disjoint
   (`seed_ranges_disjoint`).

`load_environment` forwards `mode` and `seed_start` to `build_scenario_dataset`, so
held-out eval works from config alone: passing `mode = "eval"` through a prime-rl eval
`args` block builds the eval dataset from the disjoint `EVAL_SEED_BASE` band, genuinely
held out from the train band at base `0`. The eval block in the example config uses
exactly this. Two things remain yours to keep straight: keep `mode = "eval"` on the eval
env only (a train env left on the default `mode="train"` draws from base `0`), and if
you pass an explicit `dataset=` into `load_environment`, `mode`/`seed_start` are ignored
and the split is whatever your dataset encodes.

## Two disjointness mechanisms, and which is used where

The repo ships two independent train/eval split conventions; they do not interact.

- **The generalization-gap band split** (`generalization.py`): `train_test_seeds` places
  train at `[seed_start, seed_start + n_train)` and test at a far-separated band
  starting `gap = 10_000` seeds later (with `n_train = n_test = 256`, that is
  `[0, 256)` vs `[10256, 10512)`). Used by `generalization_gap` and the Rust
  `train_test_split`; the gap absorbs later growth of the train band. Both entry points
  refuse the inputs that would break disjointness (`ValueError` in Python, a typed
  `SplitError` in Rust) in the configuration that ships, not only in a debug build or an
  unoptimized interpreter.

  **Zero held-out levels is not an empty family in Rust.** The two surfaces spell the
  count differently and the difference is deliberate. `train_test_seeds(n_train, 0, ...)`
  in Python returns an empty test list, because the band is materialized as a `range`.
  `train_test_split(train, 0, gap)` in Rust returns a `ScenarioSpec` with
  `num_levels == 0`, and `num_levels == 0` is `ScenarioSpec`'s documented Procgen
  convention for *unlimited*: the held-out family spans `[train_end + gap, u64::MAX)`.
  That family is still provably disjoint from the train band, which is the guarantee
  `SplitError` protects, so zero is accepted rather than refused; it just means "every
  seed above the gap", not "no seeds". Pass the count you want held out. A Rust caller
  that wants an empty evaluation set should skip the evaluation, not ask for zero levels.
- **The `EVAL_SEED_BASE = 1_000_000` offset** (`dataset.py`): `mode="eval"` datasets,
  the `-Eval-v1` Gymnasium IDs, and the frozen named seeds in `eval_seeds.py` all live
  at or above `EVAL_SEED_BASE`, provably disjoint from the train band
  `[0, EVAL_SEED_BASE)`. This is the split the `verifiers`/prime-rl loop and the
  regression eval set use.

Both make train and eval disjoint by construction; the band split is a measurement
instrument (one gap number), the offset is the operational convention every eval-mode
dataset draws from.

### Sealed eval seeds (opt-in)

Disjointness says the agent never trained on an eval seed; it does not say the eval
seeds are unknown. The public `EVAL_SEEDS` are fixed constants, and a bounded public
band is recoverable from one observed bar by a table scan (the paper's predictability
probe). `sealed_eval_seeds(salt)` derives the same named slots from a secret salt via
`sharpearena::sealed_seed` (a keyed derivation; every result is still
`>= EVAL_SEED_BASE`, so the disjointness check needs no salt), and
`evaluate_eval_set(..., salt=salt)` runs the unchanged protocol on them. Workflow:
generate `salt = os.urandom(32)` and keep it outside the repo and the agent's reach;
publish `sha256(salt)` before the run (for example in the SharpeBench forward
attestation); evaluate; reveal the salt afterwards so anyone can recompute the seeds
and replay the run. A revealed salt is spent. The public set and its
`EVAL_SET_VERSION` are unchanged.

**The 16-byte floor is a length check, not an entropy measurement.** `SealedSalt::new`
(and every surface that reaches it) refuses fewer than `MIN_SEALED_SALT_BYTES = 16`
bytes, which rules out the short passphrase that would put the salt back inside an
enumeration budget. It cannot inspect how those bytes were produced: sixteen zero bytes,
or sixteen bytes of a memorable phrase, are accepted by the constructor and are worth
nothing against an adversary who guesses them. Nothing about passing the length check
establishes that a salt is random. Entropy and secrecy remain the operator's obligation:
draw the salt from a cryptographic RNG (`os.urandom(32)`, `secrets.token_bytes(32)`),
never derive it from a passphrase, a timestamp or a run label, keep it out of the
repository, the agent's reach, logs and evidence artifacts until the reveal, and never
reuse a revealed salt. Operators who need resistance to salt recovery from disclosed
seeds should derive the salt per evaluation from a real KDF and treat `sealed_seed` as
the band-mapping step only; the construction is a keyed PRF-*style* derivation built from
the primitives the crate already carries, not a certified cryptographic MAC.
