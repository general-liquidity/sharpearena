# Prospective capability rereview, 2026-09-04

## Scope

This rereview treats the F1 through F8 evidence as the only empirical result set
and the prospective forecast path as a software and protocol capability. It does
not admit a current-model result. The superseded convenience-sample pilot is an
archived lifecycle trace and supports no comparison, leaderboard, or paper
finding.

## Checks performed

- Compared every new paper claim with `forecast_contract.py`,
  `forecast_evidence.py`, the cross-product tutorial, and the Lean model.
- Ran the focused forecast and tutorial tests and built the Lean project.
- Ran the tutorial through the sibling SharpeBench scorer and required its
  committed report to reproduce exactly.
- Built the paper and checked for overfull boxes, undefined references, and
  undefined citations.
- Scanned the manuscript for private execution context, obsolete model names,
  copied current-version claims, and long embedded digests.

## Findings and dispositions

1. The earlier draft named a particular inference backend and described a local
   model setup. Those details were irrelevant to the method and were removed.
2. Current package versions and test counts had been copied into prose. They
   would become stale at the next release, so the manuscript now delegates the
   current engineering identity to the provenance manifest.
3. The archived pilot could be mistaken for an empirical model field. The
   abstract, scope, limitations, reproducibility statement, and command appendix
   now state that it is superseded and excluded from every result and rank.
4. The forecast-score paragraph distinguished directional accuracy from proper
   scores without a source. It now cites the general scoring-rule literature.
5. The formal claims could be read as implementation verification. Every such
   claim now states that Lean proves a small abstract integer model and that
   executable conformance tests bridge, but do not identify, that model with the
   Python implementation.

## Residual boundaries

- Logical clocks and digests bind the recorded protocol but do not establish
  neutral custody or independently observed wall time.
- The resolution-clock block is a declared dependence unit, not proof that
  longer dependence is absent.
- SharpeArena provides point-in-time information isolation, not hostile-process
  containment.
- No current-model performance claim is available until a separately declared
  field is completed and admitted.

