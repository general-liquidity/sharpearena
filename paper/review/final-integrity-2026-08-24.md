# Stage 4.5 Final Integrity Report — 2026-08-24

## Verdict

**PASS** for the candidate manuscript in this version-controlled working tree.
This is a coverage-bounded audit of the registered paper surfaces, not a
certificate that the simulator or its synthetic markets are universally valid.

## Registered-surface inventory

| Surface | Denominator | Result |
|---|---:|---|
| Bibliography entries | 30 | 30/30 are cited; all carry author, title, and year metadata. |
| Citation commands | 67 | 67/67 resolve to registered keys; 0 undefined and 0 uncited entries. |
| New concave-impact citations | 3 | Directly checked against the source metadata for Huberman--Stanzl, Gatheral, and Almgren et al.; the manuscript uses them only for the scoped no-arbitrage and impact-shape claims stated in the text. |
| Numerical evidence files | 9 | F1--F8 plus `predictability.json` are present and parse as JSON. |
| New numerical surfaces | 2 | The concave experiment contains two 8-seed arms (0.5 and 0.7), neither profitable anywhere; the seed search recovers 16/16 bounded-band seeds with no collision and extrapolates the full-$2^{64}$ search beyond one million CPU-years. |

All paper tables spell out repeated values rather than using ditto cells. The
PDF rebuilt with Tectonic and has no undefined references, errors, or overfull
boxes. Underfull-box warnings are layout elasticity warnings only.

## Claim and evidence check

The F1--F8 tables and prose are reconciled against their committed JSON
evidence. The new F5 ablation reports its null as a confidence-interval result,
not a theorem: the $\kappa=0.5$ canonical interval crosses zero and the text
states that fact, the small-flow crossover, and the absent asymmetric-schedule
positive control. The predictability probe states both sides of the result: an
in-band AR adversary gains little beyond disclosed Calm momentum, while a public
bounded seed band permits 16/16 recovery. The limitations preserve the
generator-inversion threat and do not claim cryptographic hiding.

The companion citation now records SharpeBench version 0.6.0. The Arena PDF was
rebuilt after that publication, and its rendered bibliography contains that
exact version; this is the completed narrow cross-reference replay before the
Arena 0.11.0 release.

## AI research failure-mode checklist

| Mode | Status | Evidence |
|---|---|---|
| Implementation bug accepted as result | CLEAR | Rust, Python, WASM, and npm test gates were rerun; the default $\kappa=1$ path remains identity-tested. |
| Hallucinated citation | CLEAR | Full citation-key inventory resolves; new impact citations were source-checked. |
| Hallucinated experimental result | CLEAR | Every reported new result has an on-disk JSON input and an executable generator. |
| Shortcut reliance | CLEAR / bounded | The paper measures and scopes the public-seed shortcut rather than calling the interface leak-free in that stronger sense. |
| Bug reframed as insight | CLEAR | Negative and reversal results are retained with their replication counts and caveats. |
| Methodology fabrication | CLEAR | Commands, source entry points, evidence paths, and current code surfaces agree. |
| Frame-lock | CLEAR / bounded | The paper distinguishes interface point-in-time access from generator secrecy and names the remaining attack. |

No paid LLM evaluation was executed or used by this audit. The incomplete
SharpeBench frontier-model field is outside this manuscript and is not evidence
for any Arena claim.

## Release-readiness note

This PASS applies to the present candidate working tree. It is followed by the
mandatory author confirmation before source commits, tags, registry publication,
and the final companion-citation stitch.
