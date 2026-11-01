# Kernel field convention

`score_run` returns the pinned SharpeBench kernel's `CompositeScore` as JSON.
Some of its top-level keys say why the kernel did not give a score. Arena reads
those keys in one place, `sharpearena.kernel_score`, and classifies each key by
name from an explicit allowlist. A key's suffix alone decides nothing.

## The three classes

| Class | Keys | What the kernel did | What Arena does |
|---|---|---|---|
| Unavailable | `deflation_error`, `bootstrap_error`, `selection_error` | Could not compute a figure. The ranked numbers beside the key are the no-skill floor or a sentinel. | Raises `KernelScoreUnavailable`; a recorded row reads `unavailable_scoring_kernel_error: <key>: <message>`. |
| Refused | `standard_error_refusal`, `exposure_path_error` | Computed the track and a gate declined to rank it. The deflated Sharpe beside the key is an estimate, and `rank_eligible` is false. | Raises `KernelScoreRefused`; a recorded row reads `refused_scoring_kernel_gate: <key>: <message>`. |
| Unrecognized | any other top-level key ending in `_error` or `_refusal` | Unknown to this reader. | Raises `KernelFieldUnrecognized` naming the keys. It is never recorded as a reason, and no number is read from the composite. |

The allowlists come from the `CompositeScore` struct on SharpeBench main
(`crates/sharpebench-core/src/composite.rs`). The refusal keys carry the typed
refusal constants of `sharpebench-stats`: `standard_error_refusal` holds
`NOT_CERTIFIED_UNDER_SERIAL_CORRELATION` or
`NOT_CERTIFIED_WITHOUT_MOMENT_CREDIT`, and `exposure_path_error` holds
`EXPOSURE_PATH_UNPROVEN`. The pinned `=0.28.0` kernel emits only the three
unavailability keys; the two refusal keys arrive with the next pin.

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

## Known limits

`deflation_error` also carries deliberate shape refusals (a track with no window
that varies, no resolved motion, or no observed loss). Arena cannot tell those
from a numerical failure without matching message text, so it reports them as
unavailable until SharpeBench gives them their own key. `exposure_path_error`
likewise carries the exposure-path gate's rare numerical failures, which Arena
reports as refusals. `selection_error` is advisory in SharpeBench (it does not
remove rank eligibility), and Arena still withholds the deflated Sharpe beside
it, as it did before this convention was written.

## Moving the pin

Before the SharpeBench pin moves, list the new release's top-level
`CompositeScore` keys that end in `_error` or `_refusal`, and add each to one
allowlist with a test in `crates/sharpearena-py/tests/test_kernel_score.py`. A
missed key makes every read of an affected composite raise
`KernelFieldUnrecognized` rather than publish a guess. A withholding field whose
name ends in neither suffix is invisible to this check, so name new kernel
refusal fields with a `_refusal` suffix.
