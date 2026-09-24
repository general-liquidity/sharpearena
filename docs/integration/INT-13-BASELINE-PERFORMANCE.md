# INT-13: native and Python-boundary baseline measurement

Status: **the native half is measured and reported below, on a machine verified quiet
before, during and after the run. The Python-boundary half is still PENDING: it was
attempted on the same day, ran for 55 minutes, and was aborted unfinished when concurrent
activity on this machine made the remainder of it a contended run.** Per
[the integration plan, section 16](../../SHARPE-HANDOFF-SUPPORT/product-planning/SHARPEARENA-INTEGRATION-PLAN-2026-09-22.md),
this baseline exists to decide whether a native bridge (INT-14, INT-15) would address a
real bottleneck. That question needs a number measured on a quiet machine. Half of it now
has one. A contended run is worse than no run: it looks like evidence and is not one, so
the aborted Python half is reported as an abort with its observed load, not as a number.
See sections 5 and 7.

Feasibility, working integration and performance improvement remain three separate
statuses, per the plan. This report speaks only to the third, and speaks to it only for
the cells that actually ran quiet.

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

The native half now answers part of it, and the answer is not the one the feasibility
reports assumed. Section 6 states what changes and what does not.

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

Two harnesses, committed and building cleanly:

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
  allocation it actually ran under rather than inheriting an ambient pool. This harness
  ran to completion; its raw output is committed at
  [`int13/int13-native.json`](int13/int13-native.json).
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
  overhead or serialization volume. This harness has not yet produced a quiet-machine
  result; see section 7.

Both harnesses use the same uncertainty contract (`measurement` block in the manifest):
15 repetitions per cell, 2 discarded warm-up repetitions immediately before the timed
repetitions of that cell on the same constructed objects, and each report carries
median, mean, sample standard deviation, `rel_sd`, a 95 percent normal-approximation
interval on the mean, and `n`, so the interval can be re-derived rather than trusted.
Steps-per-second is timed with `std::time::Instant` on the Rust side and
`time.perf_counter` on the Python side. The tables below additionally quote the observed
minimum and maximum across the 15 repetitions, because on the thread-scaling cells the
spread is wide enough that a median alone would overstate what is known.

## 4. Recorded environment

Captured 2026-09-23, alongside the run that produced the section 5 numbers.

| Field | Value |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 5975WX, 32 physical cores, 64 logical processors, 3600 MHz base (`Win32_Processor`) |
| OS | Microsoft Windows 11 Pro Insider Preview, 10.0.26220 |
| rustc | 1.96.0 (`ac68faa20`, 2026-05-25) |
| cargo | 1.96.0 (`30a34c682`, 2026-05-25) |
| cargo profile | `release`. The harness was built with `cargo build --release`, and `bench-int13.exe` records `cfg!(debug_assertions)` into its own output; `int13-native.json` reads `"profile": "release"`, so this is the binary's own report rather than an assertion about how it was invoked |
| rayon | 1.12.0 (`Cargo.lock`) |
| serde_json | 1.0.151 (`Cargo.lock`) |
| sharpearena (Rust crate) | 0.31.0 |
| pyo3 | 0.29.2 (`crates/sharpearena-py/Cargo.lock`) |
| C compiler for the Python extension | none observed: `crates/sharpearena-py` has no `build.rs`, and the pyo3 cdylib links through rustc/the Python import library, not a separate C compilation step |
| Python | 3.12.6 |
| numpy | 2.5.1 |
| sharpearena (installed wheel) | 0.31.0, matching the workspace crate version; installed at `...\Python312\site-packages\sharpearena\` |
| Thread allocation | ambient Rayon pool for the lane-scaling cells, which `int13-native.json` records as `ambient_rayon_threads: 64`, matching the logical processor count; an explicit `rayon::ThreadPoolBuilder` pool sized to each of `[1, 2, 4, 8, 16, 32, 64]` for the thread-scaling cells |
| Warm-up policy | 2 discarded repetitions per measured cell, run immediately before the timed repetitions of that same cell on the same constructed objects, as `int13-native.json` records in `warmup_repetitions` |
| Repetitions | 15 timed repetitions per cell (`repetitions` in the same file) |
| Native run wall time | 475.8 s, exit code 0 |

### Observed load during the native run

"Was the machine quiet" is part of the result, not a precondition to omit once satisfied,
so the load record travels with the numbers.

| Check | Result |
|---|---|
| `Win32_Processor.LoadPercentage`, six samples at 4 s intervals ending immediately before the run | 13, 4, 6, 9, 5, 3 percent |
| `Win32_Processor.LoadPercentage`, four samples at 3 s intervals in the last 15 s before launch | 1, sample unavailable, 1, 0 percent |
| Competing `python`/`python3`/`pythonw` processes immediately before launch | 0 |
| Competing `node`/`bun` processes immediately before launch | 0 |
| Aggregate load during the run | 78 samples at 5 s intervals, ranging 0 to 85 percent. These are the harness's own load: it drives a Rayon pool of up to 64 threads, and the low samples align with the single-thread and two-thread cells at the end of the sweep. This figure is not a contention measure |
| Competing `python` processes observed on the first post-run check | 2. One (`python -B -m unittest paper/src/test_joint_gate_power.py`, from the unrelated `wt-p12a-sb` worktree) had started at 22:38:29, inside the run window, and had accumulated 4.0 CPU seconds in total by 22:45:27. The other started at 22:45:04, after the run ended at approximately 22:44:46 |
| Worst-case external contention during the run | approximately 4 CPU seconds against 475.8 s x 64 logical processors = 30,451 core seconds available, or 0.013 percent of machine capacity, from a single-threaded process |
| Post-run `LoadPercentage` | 2 percent |

Two internal consistency checks support the load record rather than only asserting it.
The 256-lane, 64-thread configuration is measured twice by independent cells roughly five
minutes apart in the run, once as the last point of the lane sweep at the ambient pool and
once as the last point of the thread sweep at an explicit 64-thread pool; they agree to
2.2 percent (208,234 against 203,735 steps/s). And `rel_sd` is at or below 4.3 percent on
every cell except the 1-, 2-, 4- and 8-thread scaling cells, which is not the signature of
sporadic heavy contention. The exception is stated as an exception in section 5.

## 5. Native results

All figures from [`int13/int13-native.json`](int13/int13-native.json), n = 15 timed
repetitions per cell after 2 discarded warm-up repetitions, release profile.

### 5.1 Scalar, reset and checkpoint

| Cell | Lanes | Threads | Median | Min to max | 95% CI on mean | rel sd |
|---|---|---|---|---|---|---|
| `native_scalar_episode_loop` | 1 | 1 | 614,833 steps/s | 603,411 to 647,494 | 612,802 to 628,081 | 2.43 % |
| `native_scalar_transition_only` | 1 | 1 | 699,049 steps/s | 680,395 to 705,104 | 692,706 to 700,633 | 1.12 % |
| `native_construct_and_first_reset` | 1 | 1 | 8.80 us/env | 8.77 to 9.26 | 8.84 to 9.02 | 2.06 % |
| `native_reset_existing_env` | 1 | 1 | 0.87 us/reset | 0.86 to 0.89 | 0.87 to 0.88 | 1.28 % |
| `native_clone_restore_state_roundtrip` | 1 | 1 | 1.24 us/roundtrip | 1.24 to 1.32 | 1.25 to 1.28 | 2.32 % |

One scalar transition costs 1.43 us on one core. A 120-bar episode is 120 of those, about
172 us, and the per-episode construction and first reset add 8.80 us, which is why the
full-episode loop (614,833 steps/s) sits 12 percent below the transition-only figure
(699,049 steps/s): construction is roughly 5 percent of an episode and the remainder is
the first reset and loop bookkeeping. Neither reset nor checkpoint is a bottleneck at this
panel size: resetting an existing env costs 0.87 us, well under one transition, and a
full `clone_state` / `restore_state` roundtrip at mid-episode depth costs 1.24 us, which
is less than the transition it would let you replay.

### 5.2 Vector scaling by lane count, ambient 64-thread pool

| Cell | Lanes | Threads | Median | Min to max | 95% CI on mean | rel sd |
|---|---|---|---|---|---|---|
| `native_vector_step_batch` | 1 | 64 | 510,445 steps/s | 500,646 to 514,303 | 508,125 to 511,694 | 0.69 % |
| `native_vector_step_batch` | 2 | 64 | 43,744 steps/s | 40,282 to 44,174 | 42,669 to 43,877 | 2.76 % |
| `native_vector_step_batch` | 4 | 64 | 42,727 steps/s | 41,246 to 44,642 | 42,401 to 43,484 | 2.49 % |
| `native_vector_step_batch` | 8 | 64 | 46,153 steps/s | 43,411 to 46,437 | 45,499 to 46,253 | 1.62 % |
| `native_vector_step_batch` | 16 | 64 | 46,915 steps/s | 44,615 to 50,502 | 46,453 to 47,624 | 2.46 % |
| `native_vector_step_batch` | 32 | 64 | 55,140 steps/s | 50,605 to 60,500 | 53,802 to 56,041 | 4.03 % |
| `native_vector_step_batch` | 64 | 64 | 85,043 steps/s | 76,453 to 91,320 | 82,717 to 86,319 | 4.21 % |
| `native_vector_step_batch` | 128 | 64 | 140,894 steps/s | 137,566 to 143,150 | 140,117 to 141,900 | 1.25 % |
| `native_vector_step_batch` | 256 | 64 | 208,234 steps/s | 204,809 to 212,537 | 206,935 to 209,044 | 1.00 % |
| `native_vector_step_batch` | 512 | 64 | 258,185 steps/s | 242,830 to 263,620 | 253,826 to 259,400 | 2.15 % |

Reading the same rows as cost per batched call, against the 1.43 us per step the scalar
cell establishes as the work content:

| Lanes | Cost per batched call | Step work in that call | Residual per call | Residual per lane |
|---|---|---|---|---|
| 1 | 2.0 us | 1.4 us | 0.5 us | 0.5 us |
| 2 | 45.7 us | 2.9 us | 42.9 us | 21.4 us |
| 4 | 93.6 us | 5.7 us | 87.9 us | 22.0 us |
| 8 | 173.3 us | 11.4 us | 161.9 us | 20.2 us |
| 16 | 341.0 us | 22.9 us | 318.2 us | 19.9 us |
| 32 | 580.3 us | 45.8 us | 534.6 us | 16.7 us |
| 64 | 752.6 us | 91.6 us | 661.0 us | 10.3 us |
| 128 | 908.5 us | 183.1 us | 725.4 us | 5.7 us |
| 256 | 1,229.4 us | 366.2 us | 863.2 us | 3.4 us |
| 512 | 1,983.1 us | 732.4 us | 1,250.7 us | 2.4 us |

The one-lane cell is a different shape from the rest: at a single lane Rayon runs the work
on the calling thread without splitting, so that row costs 0.5 us over the scalar step and
is the batched call with its parallelism effectively switched off. From two lanes upward
the residual jumps by roughly two orders of magnitude, to about 21 us per lane, and then
decays as the fixed part of the dispatch amortises, reaching 2.4 us per lane at 512 lanes.
It does not reach zero, and it does not fall below the 1.43 us of work: at every lane
count from 2 to 512 the batched call spends more time on something other than the
transition than it spends on the transition.

The residual column is the batched call minus the scalar work, so it bounds rather than
isolates the Rayon dispatch. The harness allocates a fresh `Vec<Decision>` per call and
`step_batch` returns a freshly allocated `BatchStep`, and both scale with lane count, so
some of the residual is allocation. The thread sweep below separates the two, because it
varies nothing but the pool size.

### 5.3 Thread scaling at a fixed 256 lanes, explicit Rayon pool

| Cell | Lanes | Threads | Median | Min to max | 95% CI on mean | rel sd |
|---|---|---|---|---|---|---|
| `native_vector_thread_scaling` | 256 | 1 | 398,492 steps/s | 279,017 to 488,228 | 360,817 to 423,270 | 15.74 % |
| `native_vector_thread_scaling` | 256 | 2 | 376,090 steps/s | 282,467 to 473,813 | 335,234 to 409,511 | 19.71 % |
| `native_vector_thread_scaling` | 256 | 4 | 580,908 steps/s | 466,049 to 684,064 | 542,177 to 605,739 | 10.94 % |
| `native_vector_thread_scaling` | 256 | 8 | 571,292 steps/s | 515,251 to 684,480 | 555,616 to 601,267 | 7.80 % |
| `native_vector_thread_scaling` | 256 | 16 | 488,685 steps/s | 473,010 to 514,095 | 485,999 to 497,550 | 2.32 % |
| `native_vector_thread_scaling` | 256 | 32 | 307,557 steps/s | 291,273 to 314,379 | 302,814 to 309,526 | 2.17 % |
| `native_vector_thread_scaling` | 256 | 64 | 203,735 steps/s | 200,070 to 205,545 | 202,727 to 204,393 | 0.81 % |

This is the cleanest cell in the report because it is the same code, the same allocations,
the same 256 lanes and the same panel at every row; only the Rayon pool size changes. The
curve rises to a peak somewhere in the 4 to 8 thread range and then falls monotonically,
losing a factor of 2.85 between 4 threads (580,908 steps/s) and 64 (203,735 steps/s). Past
4 threads, adding threads to this workload costs throughput.

The four low-thread rows carry the widest dispersion in the report, 7.8 to 19.7 percent
relative standard deviation against 0.8 to 4.3 percent everywhere else, so their point
estimates are the softest numbers here and should not be quoted to three figures. The
ordering survives that dispersion: the 4-thread interval on the mean (542,177 to 605,739)
and the 64-thread interval (202,727 to 204,393) do not come close to overlapping, and the
16-, 32- and 64-thread rows are tight and monotone. A plausible source of the low-thread
spread on this part in particular is where the OS places a 1-, 2-, 4- or 8-thread pool
across a 32-core, multi-die package between repetitions; the harness does not pin
affinity, and this report does not claim to have identified the cause.

### 5.4 What the native half establishes

Three things, none of which were established before this run.

1. **The scalar path is the fastest path on this workload.** One core stepping one lane
   does 699,049 steps/s. The best vectorised configuration measured, 256 lanes on a
   4-thread pool, does 580,908 steps/s, using four cores. There is no lane count and no
   thread count in the swept range at which `VecTradingEnv::step_batch` beats the scalar
   loop on throughput per machine. On the ambient 64-thread pool its best lane count is
   512 lanes at 258,185 steps/s, which is 2.71 times below the scalar loop while
   occupying the whole box, and its worst is 4 lanes at 42,727 steps/s, 16.4 times below
   it.
2. **The cost is in the batched call, not in the transition.** The transition is 1.43 us
   and the residual per call is 42.9 us at 2 lanes and 1,250.7 us at 512. `step_batch`
   issues one `par_iter_mut` dispatch per batched call
   (`crates/sharpearena/src/vec_env.rs:349`) with no chunking and no small-batch
   threshold, so each call pays a fork-join barrier to distribute work units of 1.43 us
   across a pool of up to 64 threads. On this workload the unit of parallel work is
   smaller than the cost of handing it to another thread.
3. **Reset and checkpoint are not the bottleneck.** At 0.87 us and 1.24 us they are below
   the cost of the transition they bracket, so neither autoreset churn nor
   checkpoint-heavy use changes the picture.

The measurement does not establish where the limit moves under a larger panel. Every
figure here is a 4-symbol, 120-day panel, and the 1.43 us work unit is what makes the
dispatch look expensive; a panel with more symbols or a heavier cost model would raise the
work per lane and shift the crossover. Nothing here should be read as a statement about
`step_batch` at a workload the manifest does not describe.

## 6. What this does and does not support for INT-14 and INT-15

Stated plainly, because it cuts against how the feasibility reports used this document.

**It undercuts the rationale both reports leaned on.** INT-15 says SharpeArena's
`step_batch` is "synchronous and fully parallel" and that EnvPool's asynchrony is "not the
constraint this project is under". The first half is accurate as a description of the code
and misleading as a description of its effect: the batch is fully parallel and, on this
workload, parallelising it is what costs the throughput. INT-14 says this baseline "shows
where this project's own throughput is actually lost, and that bottleneck is reachable
without any of the work above". That sentence was written before the number existed; it
now turns out to be correct, and for a sharper reason than it was offered. The bottleneck
is a per-call fork-join barrier over work units of 1.43 us in our own `step_batch`.

**It does not overturn either verdict, and the reason matters.** Both no-go verdicts rest
on scope and authority, not on performance: PufferLib 5.x accepts environments only as C
compiled in-tree against vendored dependencies, and EnvPool requires a C++ environment
registered through Bazel inside its own tree, forbidden from calling the Python
environment. A measurement cannot move either of those. What the measurement changes is
the argument available to a future reopening. It is no longer true that our existing Rayon
parallelism makes a native batching runtime redundant; it is true instead that the
cheapest fix for the measured bottleneck is in our own tree, not behind a C ABI. Chunking
lanes so each Rayon task carries many transitions instead of one, or stepping serially
below a lane threshold, are both local changes to `vec_env.rs` that address exactly the
residual measured in section 5.2, and neither needs a fork of anything. That is a stronger
version of INT-14's "reachable without any of the work above", so INT-14's verdict stands
on firmer ground than it did, not weaker.

**One INT-15 claim is now unsupported as written and has been narrowed.** Its reopening
criterion asked for "a measured workload where per-lane step variance is large enough that
asynchronous batching beats the synchronous Rayon batch, which the INT-13 baseline does not
show." This baseline does not measure per-lane step variance at all: it measures aggregate
throughput per batched call, and a lane's individual completion time is not recorded. The
criterion is fine; the claim that this document rules it out was never something this
document could support, and INT-15 now says so.

## 7. The Python-boundary half, attempted and aborted

The Python harness was run on the same day, from the same quiet start as the native half,
with the same manifest, against the installed 0.31.0 wheel. It ran 3,324.6 s (55.4 min)
and was aborted before it wrote its results file, so no Python-boundary cell is reported
here and none is quoted in INT-14 or INT-15. The abort was not a harness failure. Partway
through the run, unrelated concurrent activity on this machine grew from nothing to a
sustained multi-process load, and the remaining cells would have been measured under it.

Accounting taken over the run window, by differencing per-process CPU time before and
after and excluding the harness's own process:

| Check | Result |
|---|---|
| Wall time before abort | 3,324.6 s |
| Machine capacity over that window | 212,774 core seconds (3,324.6 s x 64 logical processors) |
| External CPU consumed by other processes | 5,748.9 core seconds |
| External contention | 2.70 percent of machine capacity, against 0.013 percent for the native run |
| Largest external consumers | 1,723 CPU s (`claude`), 1,454 and 675 CPU s (`Cursor`), plus browser and shell processes |
| Aggregate `LoadPercentage`, 660 samples at 5 s intervals | mean 20.2 percent, median 16 percent, with a tail reaching 100 percent; the first third of the run sits at 0 to 5 percent and the last third does not |

A 2.70 percent average understates the problem, because it is an average over a window
whose first third was genuinely quiet and whose later parts were not, and because the
cells were not measured simultaneously: whichever cells happened to fall in the loaded
stretch would have carried the whole of it. That is precisely the failure this document
was deferred to avoid, so the run was stopped rather than completed and qualified.

**What the rerun needs.** `python scripts/bench/int13_python_boundary.py --out
int13-python.json`, from the repository root, on a machine with
`Win32_Processor.LoadPercentage` at or below approximately 10 percent immediately before
the run and no competing `python`/`python3`/`pythonw` process beyond the harness's own
interpreter, with the same check spot-sampled during the run and the external CPU-second
accounting above repeated, since a single pre-run check cannot see contention that arrives
later. Budget approximately 90 minutes of uninterrupted quiet: the run reached its vector
sweep at 55 minutes and had not finished it. Append the cells to section 5 with their
intervals, and record the load alongside them as section 4 does for the native half.

## 8. What this leaves open

- The Python-boundary cells, as section 7 describes. Until they exist, the split between
  native transition cost and boundary cost is unmeasured, and this report can say only
  that the native transition is 1.43 us, not what fraction of a Python training step that
  is.
- Whether the section 5.2 residual is dominated by the Rayon fork-join or by the per-call
  allocations. The thread sweep shows dispatch cost is real and large, but it does not
  partition the residual, and the harness does not have a cell that holds the pool fixed
  while removing the allocations.
- Whether the crossover moves under a larger panel, more symbols or a heavier cost model,
  none of which this manifest varies.
- The cause of the 7.8 to 19.7 percent dispersion on the 1- to 8-thread cells. Thread
  placement across the package is a hypothesis, not a finding; the harness does not pin
  affinity and this run did not test it.

**Feasibility:** not this document's subject; see INT-14 and INT-15.

**Working integration:** not attempted, not claimed.

**Performance improvement:** not claimed. This report measures a baseline and identifies a
bottleneck; it does not implement or measure a fix, and the chunking and threshold changes
named in section 6 are candidates, not results.
