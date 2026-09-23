# INT-15: EnvPool feasibility assessment (I11)

Status: **feasibility investigation closed with a no-go under the current authority.** No
adapter was built and none is claimed. This report is the "precise, evidenced
incompatibility report" branch of the acceptance criterion in
[the integration plan, section 14](../../SHARPE-HANDOFF-SUPPORT/product-planning/SHARPEARENA-INTEGRATION-PLAN-2026-09-22.md).

Feasibility, working integration and performance improvement are reported separately and
none stands in for another.

Date of evidence: 2026-09-23. Quotations were fetched from primary sources in that
session.

## 1. The project is alive, which makes the assessment worth doing

| Fact | Value | Source |
|---|---|---|
| Latest release | `1.2.7`, uploaded 2026-09-15 | `https://pypi.org/pypi/envpool/json` |
| Release cadence in 2026 | 1.1.1 (04-07), 1.2.0 (04-14), 1.2.2 (05-09), 1.2.3 (05-11), 1.2.4 (05-16), 1.2.5 (05-20), 1.2.6 (08-28), 1.2.7 (09-15) | same |
| Python floor | `requires_python: >=3.12` | same |
| Wheels for 1.2.7 | 12 files: `win_amd64`, `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_13_0_arm64`, each for cp312, cp313, cp314 | same |
| Repository | last push 2026-09-15, 16 open issues | `https://api.github.com/repos/sail-sg/envpool` |

Note for anyone carrying an older impression of this project: a `win_amd64` wheel is
published for 1.2.7, and no "Windows unsupported" statement appears in the current README
or build docs. Consuming EnvPool from Python on this machine's platform is not the
obstacle.

## 2. What the custom-environment interface actually requires

From `docs/content/new_env.rst` on `main`, which renders as
`https://envpool.readthedocs.io/en/latest/content/new_env.html`.

The environment is a C++ class pair. The spec side declares configuration and the
observation and action layout as compile-time dictionaries:

```cpp
class CartPoleEnvFns {
  static decltype(auto) DefaultConfig() {
    return MakeDict("reward_threshold"_.Bind(195.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<float>({4}, {{...}, {...}})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, 1})));
  }
};
using CartPoleEnvSpec = EnvSpec<CartPoleEnvFns>;
```

The env side must override the constructor `XxxEnv(const Spec& spec, int env_id)`, plus
`bool IsDone()`, `void Reset()` and `void Step(const Action& action)`, and is pooled with
`using XxxEnvPool = AsyncEnvPool<XxxEnv>;`.

Registration is three more layers: a Bazel `cc_library` plus `pybind_extension` plus
`py_library` in a `BUILD` file under the `//envpool/...` package path, a
`PYBIND11_MODULE(...) { REGISTER(m, XxxEnvSpec, XxxEnvPool) }` translation unit, and a
`registration.py` calling `register(task_id=..., import_path=..., spec_cls=...,
dm_cls=..., gymnasium_cls=...)`.

The documented review checklist for a new environment family also requires deterministic
replay tests, seed and randomisation acceptance checks, step-level oracle alignment tests
against a pinned upstream implementation, render tests where rendering exists, an entry in
`envpool/make_test.py`, and updates to the docs index, README support list and release
packaging.

## 3. The explicit prohibition on calling the official Python environment

The first line of the new-environment review checklist, verbatim:

> The runtime implementation is native C++; do not call or embed the official
> Python environment from C++ runtime code.

This forecloses the cheapest bridge. A C++ `Step()` that reached back into
`sharpearena.TradingEnv` through the CPython API would violate the documented contract and
would also take the GIL on every transition inside a thread pool whose entire purpose is to
step environments off the Python interpreter.

## 4. Memory ownership across the boundary

EnvPool does not hand the environment a buffer to fill at its discretion. `Reset()` and
`Step()` must end by calling `Allocate()` and writing through the returned handle:

> EnvPool has carefully designed the data movement to achieve zero-copy with the lowest
> overhead. We create a simple API to make it be more user-friendly.

> At the end of `Reset` and `Step` function, you need to call `Allocate` method to
> allocate state for writing.

> You do not pass this state to any other functions or return. Instead, AsyncEnvPool will
> automatically process the data and pack it to the python interface.

Ownership lives in `envpool/core/array.h`, where `Array` holds `std::shared_ptr<char> ptr_`
and supports both an owning construction and an aliasing construction with an empty owner
that wraps a borrowed pointer without allocating. `Truncate` returns "a new Array that
shares the same memory". The pool owns the batch buffers; the environment writes into a
slice of them for the duration of one call and must not retain it.

The docs also warn that element types are not checked at compile time:

> Assigning int to a float array or assigning double to an uint64_t array will not
> generate any compilation error, but in the actual runtime, the data is wrong. Please use
> `static_cast` to convert the type correctly.

A Rust-to-C++-to-pool bridge would therefore be carrying f64 engine output into a
statically declared `Spec<...>` layout with no compiler help, on a determinism-critical
engine whose byte identity is an acceptance property. That is a real correctness risk, not
a stylistic one.

## 5. Build and cross-platform packaging burden

From `docs/content/build.rst`:

> We use bazel to build EnvPool.

> The default build and test shortcuts in this repo use **Bazel 9.2.0** via `bazelisk`.
> Bazel dependencies are configured as modules in `MODULE.bazel`.

The declared prerequisites are Python >= 3.12, Java 17, Go >= 1.22 plus bazelisk, SWIG and
Qt 6 (Qt 5 accepted for source builds), with per-platform additions: GCC/G++ on Ubuntu
24.04, Xcode Command Line Tools plus Homebrew packages on macOS 13 or newer, Visual Studio
2022 with the C++ workload plus Qt 6.11.1 on Windows. No build-time estimate appears in
the docs.

Two consequences for packaging. First, SharpeArena's release process today produces
maturin wheels and a wasm bundle; an EnvPool environment adds a Bazel, Java, Go, SWIG and
Qt toolchain to CI on three platforms, for one supplemental integration. Second, the docs
describe no out-of-tree or plugin mechanism at all. Every registration example lives at a
`//envpool/...` Bazel package path, and the review checklist is written for opening a pull
request against the upstream repository. Distributing a SharpeArena environment therefore
means either upstreaming it into sail-sg/envpool and accepting their release cadence and
oracle-alignment test requirements, or maintaining a fork and publishing a differently
named wheel. Neither is a thin adapter. The absence of a documented plugin path is an
absence of evidence rather than an explicit prohibition, and is recorded as such.

## 6. What the bridge would need from our side

The same structural fact that blocks INT-14 applies here. SharpeArena's engine exposes no
C ABI: no `extern "C"`, no `#[no_mangle]` anywhere in the tree. The only native artifacts
are the pyo3 extension in `crates/sharpearena-py` and the wasm-bindgen module in
`crates/sharpearena-wasm`, and both move data as JSON strings.
`PyTradingEnv::step` is `fn step(&mut self, decision_json: &str) -> PyResult<(String, f64,
bool, String)>` (`crates/sharpearena-py/src/lib.rs:439`).

A C++ `Step()` cannot use either surface. Using the pyo3 one is the thing the checklist
forbids. Using the wasm one is not applicable. So the bridge requires a new flat-numeric
`extern "C"` surface on the Rust engine, with its own panic-unwind discipline, its own
ownership rules for returned buffers and its own conformance and determinism tests. That
is the native ABI expansion this ticket has no authority to perform, and it would need an
ADR and a maintenance owner regardless of which trainer motivated it.

The alternative, reimplementing the market, fill and accounting mechanics as C++ inside
`Step()`, is the second finance engine the plan rejects: "Do not port finance mechanics
merely to claim support."

## 7. Threading comparison against the existing Rayon engine

EnvPool's asynchronous pool is configured by `num_envs`, `batch_size` and `num_threads`,
where `num_threads` is "the maximum thread number for executing the actual `env.step`",
and "when the finished stepping thread number >= `batch_size`, we return the result". The
low-level API is `send(action, env_id)` and `recv()`, with `step()` defined as
`send(...); return recv()`.

SharpeArena's `VecTradingEnv::step_batch` is synchronous and fully parallel: it maps
`par_iter_mut` over every lane and returns when all lanes have stepped
(`crates/sharpearena/src/vec_env.rs:348`). EnvPool's advantage over that design is
asynchrony, which hides straggler lanes by returning the first `batch_size` completions
rather than waiting for the slowest. That advantage is real for environments with
high per-step variance.

It is not the constraint this project is under. The INT-13 baseline
([INT-13-BASELINE-PERFORMANCE.md](INT-13-BASELINE-PERFORMANCE.md)) measures where
SharpeArena's throughput is actually going, and an asynchronous C++ pool does not address
it. Replacing a synchronous parallel batch with an asynchronous one also changes the
lane-to-step correspondence that the deterministic, byte-identical batch semantics depend
on, which would need its own parity argument before it could be considered an
optimisation rather than a semantic change.

## 8. Verdict

**Feasibility: no-go under current authority.** EnvPool is maintained, ships wheels for
this platform, and has a documented custom-environment path, so the obstacle is not the
project's health. The obstacle is that its path requires a C++ environment registered
through Bazel and pybind11 inside the EnvPool tree, explicitly forbidden from calling the
Python environment, writing into pool-owned buffers under a statically declared type
layout. Meeting it from SharpeArena requires a new `extern "C"` surface on the finance
engine plus a Bazel, Java, Go, SWIG and Qt build on three platforms, or a C++
reimplementation of the market mechanics. The first is out of scope for this ticket; the
second is rejected by the plan.

**Working integration: not attempted, not claimed.**

**Performance improvement: not measured, not claimed.** No throughput benefit was
established, and the plan's rule applies: a no-benefit result closes the feasibility
investigation and cannot be reported as a shipped adapter.

### What would reopen this

- A documented out-of-tree environment registration path in EnvPool, removing the fork or
  upstream requirement.
- A separately approved ADR authorising a flat-numeric C ABI on the Rust engine, with
  conformance tests and a named owner, justified by more than one trainer.
- A measured workload where per-lane step variance is large enough that asynchronous
  batching beats the synchronous Rayon batch, which the INT-13 baseline does not show.
