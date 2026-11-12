# Kernel field convention

`score_run` returns the pinned SharpeBench kernel's `CompositeScore` as JSON.
Some of its top-level keys say why the kernel did not give a score. Arena reads
those keys in one place, `sharpearena.kernel_score`, and classifies each key by
name from an explicit allowlist. A key's suffix alone decides nothing.

## The three classes

| Class | Keys | What the kernel did | What Arena does |
|---|---|---|---|
| Unavailable | `deflation_error`, `bootstrap_error`, `selection_error`, `dsr_ci_error`, `exposure_path_error` | Could not compute a figure. The ranked numbers beside the key are the no-skill floor or a sentinel. | Raises `KernelScoreUnavailable`; a recorded row reads `unavailable_scoring_kernel_error: <key>: <message>`. |
| Refused | `standard_error_refusal`, `deflation_refusal`, `exposure_path_refusal` | Computed the track and a gate declined to rank it. The deflated Sharpe beside the key is an estimate, and `rank_eligible` is false. | Raises `KernelScoreRefused`; a recorded row reads `refused_scoring_kernel_gate: <key>: <message>`. |
| Not withholding | `psr_standard_error` | Scored the track and labels a methodology choice: `"benchmark"` when the opt-in null-evaluated PSR standard error was used, absent for the default. | Reads the composite as if the key were absent; the key is never a reason. |
| Unrecognized | any other top-level key ending in `_error` or `_refusal` | Unknown to this reader. | Raises `KernelFieldUnrecognized` naming the keys. It is never recorded as a reason, and no number is read from the composite. |

The allowlists come from the `CompositeScore` struct in
`crates/sharpebench-core/src/composite.rs`. SharpeBench main (checked at
2969abc) carries all nine suffixed keys in the table: the refusal taxonomy adds
`deflation_refusal` (a track refused for its shape: constant, every window flat,
no window with observed risk, or no bar a loss), `exposure_path_refusal` (the
exposure-path gate) and `dsr_ci_error` (the DSR interval could not be
estimated) to `deflation_error`, `bootstrap_error`, `selection_error`,
`standard_error_refusal` and `exposure_path_error`, and the opt-in PSR
standard-error convention adds `psr_standard_error`. `standard_error_refusal`
holds `NOT_CERTIFIED_UNDER_SERIAL_CORRELATION` or
`NOT_CERTIFIED_WITHOUT_MOMENT_CREDIT`. The pinned `=0.28.0` kernel emits only
`deflation_error`, `bootstrap_error` and `selection_error`.
`scripts/check-against-bench-main.py` reads the struct on the Bench checkout it
overlays and fails when a suffixed key is in none of the three lists, so the
nightly job names a new key before a pin bump has to.

### The two meanings of `exposure_path_error`

Before the refusal taxonomy, `exposure_path_error` carried both the
exposure-path gate's refusal and its rare numerical failures. On SharpeBench
main it carries only the failures, and the refusal moves to
`exposure_path_refusal`. Arena accepts both: `exposure_path_error` is unavailability unless its message
is exactly the gate's refusal requirement, `EXPOSURE_PATH_UNPROVEN`, either bare
or in the `returns <requirement>` form main renders. That message is read as a
refusal. Matching is exact; any other text, including a near miss, stays
unavailability. The accepted messages are in `LEGACY_REFUSAL_MESSAGES`.

## Rules a reader can rely on

- A key whose value is null or empty carries no claim and is ignored, in either
  allowlist. An unrecognized key fails closed whatever its value, because its
  presence alone means the kernel's vocabulary has moved.
- When unavailability and refusal keys are both present, the outcome is
  unavailable and the reason names every key; the refusals stay on the
  exception's `refusals` map.
- A refused or unavailable score is never a number. `is_kernel_score_unavailable`
  is true for both recorded forms, and `is_kernel_score_refused` picks out the
  refusal form. `KernelScoreWithheld` is the common base class of both
  exceptions.
- `seed_pass_tally` counts a refused seed as unscorable and not passed, the same
  as an unavailable one, so the tally's keys are unchanged.

### The constant track under each kernel

`score_run` withholds a track with no Sharpe ratio. The pinned `=0.29.0` kernel
refuses such a track under `deflation_refusal`, and Arena adds nothing beside it,
so a constant track is recorded as `refused_scoring_kernel_gate:
deflation_refusal: returns must not be constant: a constant series has no Sharpe
ratio`. On a kernel from before 0.29.0, which withholds it under
`deflation_error` or not at all, Arena's own boundary writes the same key and
reason. The pinned kernel also refuses two shapes that 0.28.0 and earlier
scored: a track whose moments show no risk in any window ("must show observed
risk"), and a track with no losing bar in any window ("returns must show an
observed loss").

## Known limits

Against a kernel from before the refusal taxonomy, 0.28.0 and earlier,
`deflation_error` also carries the shape refusals that 0.29.0 moves to
`deflation_refusal`. Arena cannot tell those from a numerical
failure without matching message text, so on such a kernel it reports them as
unavailable. `selection_error` is advisory in
SharpeBench (it does not remove rank eligibility), and Arena still withholds the
deflated Sharpe beside it, as it did before this convention was written.

## Moving the pin

Before the SharpeBench pin moves, list the new release's top-level
`CompositeScore` keys that end in `_error` or `_refusal`, and add each to one
list with a test in `crates/sharpearena-py/tests/test_kernel_score.py`.
`scripts/bump-bench-pin.py` runs that listing against the new release's source
and refuses to bump while a key is unclassified. A
missed key makes every read of an affected composite raise
`KernelFieldUnrecognized` rather than publish a guess. A withholding field whose
name ends in neither suffix is invisible to this check, so name new kernel
refusal fields with a `_refusal` suffix.
