# Studies

Frozen protocols and producers for experiments that run on the merged engine. Nothing here
is evidence until a final stage has run, and smoke outputs are plumbing checks only.

- `realism/`: acceptance rows P13-R and P14-R.
- `learner/`: acceptance rows P18-E and P18-S.
- `COSTS.md`: measured throughput and expected core-hours.

Outputs go under each study's `runs/<stage>/`, which is outside the paper's provenance scope.
A later coordinator step decides what enters `paper/evidence`.

Tests: `python -m pytest studies/tests` with the built package on the path.
