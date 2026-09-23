# INT-13: native and Python-boundary baseline measurement

Status: **harness and environment record complete; the throughput and scaling
measurements themselves are PENDING, not reported.** Per
[the integration plan, section 16](../../SHARPE-HANDOFF-SUPPORT/product-planning/SHARPEARENA-INTEGRATION-PLAN-2026-09-22.md),
this baseline exists to decide whether a native bridge (INT-14, INT-15) would address a
real bottleneck. That question needs a number measured on a quiet machine, and this
machine was not quiet when the check was made. A contended run is worse than no run: it
looks like evidence and is not one. See section 5.

Feasibility, working integration and performance improvement remain three separate
statuses, per the plan. This report speaks only to the third, and only speaks to it once
the measurement below actually runs.

## 1. What this measures and why

INT-14 and INT-15 each conclude with a no-go on the currently documented native-bridge
route (`INT-14-PUFFERLIB-FEASIBILITY.md`, `INT-15-ENVPOOL-FEASIBILITY.md`). Both verdicts
lean on a claim this document is supposed to substantiate with numbers rather than
assert: that `crates/sharpearena/src/vec_env.rs` already parallelises transitions with
Rayon, so the question a native bridge would need to answer is not "is Rust fast", it is
"where does the time actually go between a scalar step, a vectorised step, and the
Python boundary, and does either upstream project's own parallelism model address that,
given ours already exists." That is a measurement question, not a documentation
question, which is why it was deferred to its own harness instead of asserted in the
feasibility reports.

## 2. Step definition

From `docs/integration/int13/workload-manifest.json`, `step_definition`, verbatim:

> One step is one full market tick for one lane: the engine applies that lane's Decision
> at the current daily bar, clears fills under the cost model, marks the book and
> advances the cursor by one bar. A batched call over B lanes counts B steps. This unit is
> not a token, not a tool call and not an agent turn.

The workload is a synthetic 4-symbol, 120-day panel (`panel.n_symbols = 4`,
`panel.n_days = 120`, `distribution_mode: "calm"`), stepped under a constant
`target_weight = 0.25` buy policy and `CostModel::default()`, in `f64` throughout, with
`autoreset_mode: "next_step"`. Both harnesses read this one manifest so the native and
Python-boundary halves measure the same workload rather than two workloads that happen
to share a name.

## 3. Harness design

Two harnesses, committed and building cleanly, neither of which has run its measured
cells on a quiet machine yet:

- `crates/sharpearena/examples/bench-int13.rs` (native, Rust). Measures, in order:
  `native_scalar_episode_loop` (full episodes, construction included, the honest scalar
  cost), `native_scalar_transition_only` (construction hoisted out, isolating transition
  cost), `native_construct_and_first_reset`, `native_reset_existing_env`,
  `native_clone_restore_state_roundtrip` (checkpoint cost at mid-episode depth), then a
  vector scaling curve (`native_vector_step_batch`) over
  `scaling.lane_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]` at the ambient Rayon
  thread count, and a thread-scaling curve (`native_vector_thread_scaling`) at a fixed
  256 lanes over `scaling.thread_counts = [1, 2, 4, 8, 16, 32, 64]` using an explicit
  `rayon::ThreadPoolBuilder` pool per thread count, so each cell records the thread
  allocation it actually ran under rather than inheriting an ambient pool.
- `scripts/bench/int13_python_boundary.py` (Python, against the installed `sharpearena`
  wheel, not a source-tree import; a debug build would measure the wrong thing).
  Decomposes the boundary into: `py_scalar_binding_no_python_json` (pyo3 call plus
  Rust-side `serde_json` parse/serialize only, decision string prebuilt), then
  `py_scalar_binding_with_python_json` (adds `json.dumps`/`json.loads` on the Python
  side), then `py_json_encode_decode_only` (the Python JSON cost alone, no engine call at
  all, isolating what fraction of cell 2 is JSON rather than the call), then
  `py_gymnasium_env_step` (the path a user actually trains against:
  `SharpeArenaEnv.step` with a numpy action, JSON boundary, numpy observation decode),
  then reset and checkpoint cost at the boundary
  (`py_reset_existing_env`, `py_construct_and_first_reset`,
  `py_clone_restore_state_roundtrip`), then a vector boundary scaling curve at the same
  lane counts as the native harness, both with and without the Python-side JSON step
  (`py_vector_step_batch_no_python_json`, `py_vector_step_batch_with_python_json`), so
  the marginal cost of the Python JSON layer over the raw batched call is visible at
  every lane count rather than only at one. It also records payload sizes: decision and
  observation JSON bytes, the `clone_state` snapshot size, and the ratio of JSON bytes
  moved to the actual numeric payload the Gymnasium space exposes
  (`json_to_numeric_ratio`), which bears directly on whether the boundary cost is call
  overhead or serialization volume.

Both harnesses use the same uncertainty contract (`measurement` block in the manifest):
15 repetitions per cell, 2 discarded warm-up repetitions immediately before the timed
repetitions of that cell on the same constructed objects, and each report carries
median, mean, sample standard deviation, `rel_sd`, a 95 percent normal-approximation
interval on the mean, and `n`, so the interval can be re-derived rather than trusted.
Steps-per-second is timed with `std::time::Instant` on the Rust side and
`time.perf_counter` on the Python side.

## 4. Recorded environment

Captured 2026-09-23, independent of whether the machine was quiet, because this part of
the record does not depend on load:

| Field | Value |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 5975WX, 32 physical cores, 64 logical processors, 3600 MHz base (`Win32_Processor`) |
| OS | Microsoft Windows 11 Pro Insider Preview, 10.0.26220 |
| rustc | 1.96.0 (`ac68faa20`, 2026-05-25) |
| cargo | 1.96.0 (`30a34c682`, 2026-05-25) |
| cargo profile | `release` (the harness binary was built with `cargo run --release`; `bench-int13.exe` records `cfg!(debug_assertions)` into its own output so a debug run would self-report as such) |
| rayon | 1.12.0 (`Cargo.lock`) |
| serde_json | 1.0.151 (`Cargo.lock`) |
| sharpearena (Rust crate) | 0.31.0 |
| pyo3 | 0.29.2 (`crates/sharpearena-py/Cargo.lock`) |
| C compiler for the Python extension | none observed: `crates/sharpearena-py` has no `build.rs`, and the pyo3 cdylib links through rustc/the Python import library, not a separate C compilation step |
| Python | 3.12.6 |
| numpy | 2.5.1 |
| sharpearena (installed wheel) | 0.31.0, matching the workspace crate version; installed at `...\Python312\site-packages\sharpearena\` |
| Thread allocation | ambient Rayon pool (`rayon::current_num_threads()`, recorded into the native harness's own JSON output as `ambient_rayon_threads`) for the lane-scaling cells; an explicit `rayon::ThreadPoolBuilder` pool sized to each of `[1, 2, 4, 8, 16, 32, 64]` for the thread-scaling cells |
| Warm-up policy | 2 discarded repetitions per measured cell, run immediately before the timed repetitions of that same cell on the same constructed objects (both harnesses; see manifest `measurement.warmup`) |

`bench-int13.exe` (release) is already built at
`target/release/examples/bench-int13.exe`; the Python wheel (`sharpearena` 0.31.0) is
already installed. Producing a result requires running them, not building them.

## 5. Why no throughput number appears in this document

At 14:39 on the measurement day, this machine was running approximately 40 competing
Python processes at roughly 61 percent aggregate CPU load, including a 32-worker Monte
Carlo job and three additional simulation runs from an unrelated workstream. A direct
re-check while writing this report (same day) showed the condition had not cleared:

| Check | Result |
|---|---|
| `Win32_Processor.LoadPercentage` | 97 percent |
| Running `python`/`python3`/`pythonw` processes (`Get-Process`) | 45 |

Both harnesses are CPU-bound wall-clock benchmarks on a machine with 64 logical
processors; the vector scaling and thread scaling cells specifically vary the Rayon
thread pool up to 64 threads, so contention from other CPU-bound processes on the same
box invalidates exactly the numbers this report exists to produce. A number produced
under that load would not show SharpeArena's throughput, it would show SharpeArena
competing with a Monte Carlo job for cores, and reporting it as a baseline would be
worse than reporting nothing, because it would be read as evidence.

No `int13-native.json` or `int13-python.json` result file exists in this worktree. The
harness was built and the wheel is installed, but neither harness's measured cells have
been executed and recorded under a quiet machine. Nothing measured under the contended
window is quoted anywhere in this document or in INT-14/INT-15.

**What "quiet" means for the rerun:** `Win32_Processor.LoadPercentage` at or below
approximately 10 percent immediately before the run, and no competing
`python`/`python3`/`pythonw` processes beyond the harness's own interpreter, checked
immediately before starting and spot-checked during the longest-running cells (the
lane-count and thread-count scaling sweeps). Both checks should be re-recorded into this
document's environment table alongside whichever run actually produces the numbers,
because "was the machine quiet" is part of the result, not a precondition to omit once
satisfied.

## 6. What this leaves open

The measurement itself: run
`cargo run --release -p sharpearena --example bench-int13 -- --out int13-native.json`
and `python scripts/bench/int13_python_boundary.py --out int13-python.json` back to
back on a verified-quiet machine, append the resulting cells to this document with their
uncertainty intervals, and only then answer the questions this baseline was
commissioned to answer: how much of total latency is the native transition versus the
Python/JSON boundary versus reset/checkpoint overhead, whether vector throughput scales
sublinearly with lane count or thread count in a way that indicates a real Rayon
contention point, and whether that point is one a native bridge (INT-14, INT-15) could
plausibly move. Until that run exists, INT-14's and INT-15's references to this
document's findings should be read as references to the methodology above, not to a
number.

**Feasibility:** not this document's subject; see INT-14 and INT-15.

**Working integration:** not attempted, not claimed.

**Performance improvement:** not measured, not claimed. Pending a quiet-machine rerun of
the harness committed alongside this report.
