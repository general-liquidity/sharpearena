# INT-14: PufferLib feasibility assessment (I10)

Status: **feasibility investigation closed with a no-go on the currently documented
route.** No adapter was built and none is claimed. This report is the "precise, evidenced
incompatibility report" branch of the acceptance criterion in
[the integration plan, section 14](../../SHARPE-HANDOFF-SUPPORT/product-planning/SHARPEARENA-INTEGRATION-PLAN-2026-09-22.md).

Three statuses are kept separate throughout, per the plan: **feasibility**, **working
integration**, **performance improvement**. This report establishes only the first, and
it establishes it negatively for the supported path.

Date of evidence: 2026-09-23. Every quotation below was fetched from the primary source
in that session; none is from recall.

## 1. Which release is the real one

The plan asks to "first select an actual maintained release". That selection is not
clean, because the published package and the published documentation describe different
architectures.

| Source | What it says | Retrieved from |
|---|---|---|
| PyPI | Latest release `3.0.0`, uploaded 2025-06-23. Version list ends at `3.0.0` (previous: `3.0.0a1`, `2.0.6`). `requires_python: >=3.9`. The only file for `3.0.0` is `pufferlib-3.0.0.tar.gz`, a source distribution: no wheels. | `https://pypi.org/pypi/pufferlib/json` |
| GitHub | `"default_branch": "5.0"`, `"pushed_at": "2026-09-13T18:33:52Z"`, 6470 stars, 191 open issues. | `https://api.github.com/repos/PufferAI/PufferLib` |
| puffer.ai/docs.html | Describes 5.0 only: "PuffeRL: Up to 60,000,000 step/second training in only ~10k lines of CUDA C." | `https://puffer.ai/docs.html` |

So `pip install pufferlib` today installs 3.0.0, the Python/Gymnasium-era package, while
every sentence of the public documentation describes the 5.0 branch, which has never been
published to PyPI. The documentation does not disclose this gap. Any integration decision
has to name which of the two it targets; they do not share an environment interface.

The 5.0 branch is the actually maintained line (pushed 2026-09-13), so it is the one
assessed here. Targeting 3.0.0 instead would mean building against a package whose upstream
has been superseded and whose environment modules the maintainer has removed, which is not
a defensible foundation for a supported adapter.

## 2. What the supported custom-environment path requires

Confirmed: the route is C, in-tree, compiled into the PufferLib binary.

The 5.0 repository root contains no Python package at all. Its top level is
`config`, `ocean`, `resources`, `src`, `tests`, `vendor`, plus `README.md`,
`SKILL_ISSUES.md`, `build.sh` and `profile.sh`
(`https://api.github.com/repos/PufferAI/PufferLib/contents/?ref=5.0`). There is no
`pufferlib/` directory, therefore no `emulation.py` and no `vector.py`.

The documentation states the authoring model directly:

> Ocean environments are written in C. Mostly very simple C. Like first 2 weeks of an
> intro systems course C. Compile standalone builds with --cpu.

> Observations, actions, rewards, and terminals are each allocated as big chunks of
> memory that are contiguous across all (usually thousands) of environment instances.

> The Minimal environment is a clean template with comments that walk you through our
> API. Read and understand it first. To create your own environment, first copy and
> rename all the files. PufferLib explicitly looks for your_env.h inside of
> ocean/your_env.

`ocean/minimal` contains exactly three files: `minimal.cu`, `minimal.h`,
`minimal_net.h` (`https://api.github.com/repos/PufferAI/PufferLib/contents/ocean/minimal?ref=5.0`).

The C contract is in `src/pufferenv.h` on branch 5.0, quoted verbatim:

```c
typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

// Shared env API. CPU: per-env Env*. GPU: Env* is device batch base; step/reset
// run the full vector inside the env. puf_bind_stream / puf_vec_create are GPU
// create-path hooks (CPU builds get stubs in pufferl).
void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
// CPU: host Env*. GPU: device batch base; implementation D2Hs what it needs and draws.
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
```

Two points follow from the header rather than from prose. The environment author does not
allocate the observation, action, reward or terminal buffers: they arrive as raw pointers
on an `Agent` struct that the runtime owns, and `puf_step` writes into them in place. And
`pufferenv.h` includes `raylib.h`, so the environment translation unit is compiled inside
the PufferLib tree against PufferLib's vendored dependencies, not linked as an independent
module.

One documented detail does not match the source. The debugging checklist says "Ensure you
have defined your observation and action metadata (space/size/type) correctly in your
binding.c", but no file named `binding.c` exists in the 5.0 tree; `ocean/minimal/minimal.h`
and `ocean/template/template.h` declare that metadata with preprocessor macros
(`#define OBS_SIZE`, `#define ACT_SIZES`, `#define NUM_ATNS`) in the environment header.
The discrepancy is recorded because it affects how much of the documentation can be relied
on when estimating integration effort; it is not itself a blocker.

## 3. The Python wrapper route is gone, and the maintainer says so

The FAQ on puffer.ai/docs.html, verbatim:

> Where did all the Python/third-party stuff go? It was all 100x+ slower than PufferLib
> is now. We do plan on hooking the C/C++ for Atari and maybe ProcGen into our low-level
> interface when we have time.

That is an explicit removal with an explicit reason, and the stated future plan is to hook
already-compiled C/C++ engines into the low-level C interface, not to restore a Python
environment wrapper. The structural evidence in section 2 agrees: there is no Python
package on the branch to wrap anything with.

The "100x+ slower" figure carries no benchmark script, workload or methodology in the
docs. It is recorded here as the maintainer's stated rationale, not as a measurement this
project has verified or relies on.

## 4. What a bridge to the canonical Rust engine would actually require

SharpeArena's engine has no C ABI. Searching the tree for `extern "C"` and `#[no_mangle]`
returns nothing; the only native artifacts are `crates/sharpearena-py` (`crate-type =
["cdylib"]`, a pyo3 extension module) and `crates/sharpearena-wasm` (`crate-type =
["cdylib", "rlib"]`, a wasm-bindgen module). Both of those surfaces move data as JSON
strings: `PyTradingEnv::step` has signature
`fn step(&mut self, decision_json: &str) -> PyResult<(String, f64, bool, String)>`
(`crates/sharpearena-py/src/lib.rs:439`), and every exported wasm function in
`crates/sharpearena-wasm/src/lib.rs` takes and returns `String`.

So a PufferLib 5.0 bridge needs all of the following, in order:

1. **A new C ABI on the Rust engine.** `puf_step` writes into `float*` buffers it does not
   own; it cannot consume a `String`. Bridging means exporting `extern "C"` entry points
   over a flat numeric layout, with manual lifetime and panic-unwind discipline at the
   boundary. This is precisely the "new native ABI expansion" that this ticket is not
   authorised to perform, and it is not a thin shim: it is a second public surface of the
   engine with its own versioning, its own determinism obligations and its own conformance
   tests.
2. **An in-tree C environment inside a fork of PufferLib.** `PufferLib explicitly looks
   for your_env.h inside of ocean/your_env`, and the header includes raylib. The
   environment is not a separately distributable artifact; it lives in a PufferLib
   checkout and is built by `build.sh`. Distribution therefore means maintaining a fork,
   because there is no PyPI package for 5.0 to depend on.
3. **A static or dynamic link from that C environment to a Rust cdylib.** Nothing in the
   docs or in `pufferenv.h` forbids calling an external library from `puf_step`; nothing
   sanctions it either, and no shipped Ocean environment does it. It is mechanically
   possible and architecturally unattested.
4. **A GPU.** The FAQ: "Can I run PufferLib without an Nvidia GPU? As of 5.0, we have a
   solid --cpu eval mode but no CPU training option." The documented training path, which
   is the reason to integrate at all, requires CUDA.

Item 1 alone is out of scope by the terms of this ticket. Items 2 and 3 convert a
supported integration into a maintained fork with an unattested linkage pattern.

The alternative, writing the market dynamics in C so they live inside `puf_step` as the
Ocean template expects, is a second finance engine. The plan rejects that by default and
requires a separate architecture decision, ADR and parity protocol before it could be
reconsidered. This report does not request one.

## 5. Build and platform reality on this machine

`build.sh` on branch 5.0 invokes the compiler directly (`${CC:-clang}`, `nvcc`,
`-std=c++17`) with no `setup.py`, no Cython and no pip step. The documented install is
`curl -fsSL .../PufferTank/refs/heads/5.0/install.sh | bash` or a prebuilt CUDA Docker
image. The documentation makes no statement about Windows or WSL in either direction;
`build.sh` has no Windows branch. SharpeArena's own development and CI matrix would have
to absorb a clang/nvcc Unix toolchain and a CUDA device to exercise the adapter at all.

## 6. Verdict

**Feasibility: no-go on the supported route, for reasons that are structural rather than
budgetary.** The currently maintained PufferLib line accepts custom environments only as C
compiled in-tree against its own vendored dependencies, writing into runtime-owned flat
buffers, with training gated on CUDA, and it is not installable from PyPI. Reaching it
from SharpeArena requires creating a C ABI on the finance engine and maintaining a
PufferLib fork. The plan's stop condition applies: stop at a feasibility report and request
a separate architecture decision.

**Working integration: not attempted, not claimed.**

**Performance improvement: not measured, not claimed.** The INT-13 baseline
([INT-13-BASELINE-PERFORMANCE.md](INT-13-BASELINE-PERFORMANCE.md)) has since measured its
native half, and it shows where this project's own throughput is actually lost: one scalar
transition costs 1.43 us, while `VecTradingEnv::step_batch` spends 42.9 us per call beyond
that work at 2 lanes and 1,250.7 us at 512, because it issues one `par_iter_mut` dispatch
per batched call over work units smaller than the cost of handing them to another thread.
At a fixed 256 lanes, throughput peaks at 580,908 steps/s on a 4-thread Rayon pool and
falls to 203,735 steps/s at 64 threads, so the ambient default is 2.85 times worse than
the best measured pool size and worse than the 699,049 steps/s the scalar loop reaches on
one core. That bottleneck is reachable without any of the work above: chunking lanes per
Rayon task, or stepping serially below a lane threshold, are local changes to
`crates/sharpearena/src/vec_env.rs` and need no C ABI and no fork. The baseline's Python
boundary half is still pending a quiet machine, so no claim here rests on it.

### What would reopen this

- PufferLib publishing a 5.x package with a documented out-of-tree environment plugin
  interface, so an adapter is not a fork.
- A separately approved ADR that authorises a flat-numeric C ABI on the Rust engine, with
  its own conformance and determinism tests, for reasons broader than one trainer.
