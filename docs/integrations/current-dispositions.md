# Integration dispositions rechecked on 2026-10-10

This review supplements the dated feasibility reports. It does not replace
their historical results, authorize a new native ABI, or claim an adapter was
built. Arena source was reviewed at `5b5d8ab`; no training or performance
measurement was run for this review.

## CleanRL: dependency conflict remains for the reviewed route

The [upstream manifest at fe8d8a03](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/pyproject.toml)
declares Python `>=3.8,<3.11` and `gymnasium==0.29.1`. Arena declares
`gymnasium>=1.0` in its Python distribution manifest. These Gymnasium ranges
have no intersection, independently of which interpreter is used. Arena's
cp312 wheel additionally cannot be installed on CleanRL's declared Python range.
This is a metadata compatibility finding, not an attempted training run.

Do not relax Arena's dependency floor to make an installation succeed. A
modernized or vendored CleanRL example is separate work, with licensing,
API-conversion, installed-artifact parity and maintenance obligations. The
existing SB3 route is a separate alternative, not evidence that CleanRL works.

Update, 2026-10-30: that separate work took the fresh-write form.
`examples/cleanrl-style/ppo_vector.py` is a CleanRL-style single file with attribution
and no vendored code. The conflict above still stands, and the floor did not move.
[The support-status page](support-status.md) records how far the example was checked.

## PufferLib and EnvPool: native bridge decisions remain separate

PufferLib's [5.0 environment header at 6ffa5b10](https://github.com/PufferAI/PufferLib/blob/6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2/src/pufferenv.h)
still defines C lifecycle calls and raw observation/action/reward/terminal
buffers. This does not describe a Gymnasium adapter. A bridge from Arena's
canonical engine needs explicit ownership, numeric layout, panic handling and
parity contracts. See the dated [INT-14 assessment](../integration/INT-14-PUFFERLIB-FEASIBILITY.md)
for the route and its original scope decision. No PufferLib performance claim
is established by this header review.

EnvPool's [environment guide at 9c31c547](https://github.com/sail-sg/envpool/blob/9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3/docs/content/new_env.rst)
still starts from a fork with a C++ environment, Bazel build definition,
pybind11 binding and registration. It does not establish a shipped, out-of-tree
Arena plugin. The [INT-15 assessment](../integration/INT-15-ENVPOOL-FEASIBILITY.md)
remains a no-go for that reviewed route under its stated scope, not a proof
that every future integration is impossible.

A source search over `crates/sharpearena/src`, `crates/sharpearena-py/src` and
`crates/sharpearena-wasm/src` found no authored `extern "C"` or `no_mangle`
entrypoints exposing a flat numeric engine interface. Generated Python/WASM
binding glue is not that public ABI. `vec_env.rs` still uses Rayon
`par_iter_mut` for lane stepping; this search establishes neither optimal
parallel scheduling nor a speedup from an alternative runtime.

Reopening either native bridge requires an architecture and maintenance
decision. The unfinished Python-boundary measurement and unmeasured individual-
lane completion variance are not supplied by the native aggregate baseline.

## HUD and Harbor: fixture evidence is not adapter support

The Python package source and its dependency manifest contain no HUD or Harbor
imports or declared dependencies. The fixtures remain in `examples/hud/` and
`integrations/harbor/`, outside the distributed adapter surface.

HUD's local/subprocess checks do not establish filesystem or network isolation,
hosted runtime behavior or control-channel authentication. Harbor's recorded
job exercised scripted tape-replay agents, not financial RL training. Its
historical accepted/refused/crashed trial tally remains the one documented in
[INT-09](INT-09-harbor-local-feasibility.md); the malformed-byte grader repairs
do not constitute a new container-job run.

A supported adapter or additional isolation evidence would be a new scoped
deliverable. Fixture tests and logos do not establish it. Keep the
[support-status table](support-status.md) authoritative about these distinctions.

Update, 2026-10-30: Harbor gained a job metric, a pre-collection hook and a corrected
fixture, and CI gained a non-blocking job that runs the HUD fixture with `hud`
installed. None of this builds an adapter, and no container job was rerun.
