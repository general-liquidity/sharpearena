# Approved vector reset and episode-cap contracts

Owner approved both reset reseeding and enforced caps on 2026-10-10.
Baseline: `d3318f1`. This is engineering verification, not a training study.

## Contract

- An integer reset seed rebuilds native lanes on consecutive seeds. A list
  supplies one nonnegative int64 seed per lane, literally even in eval mode.
- Native construction options survive reset, including an explicit execution
  seed; the default execution seed follows each scenario seed.
- Unseeded and automatic resets replay current scenarios. Explicit reset clears
  queued actions, pending reset state and counters. Invalid seed input is atomic.
- A positive executed-step cap truncates each lane using its selected autoreset
  mode. Disabled capped lanes freeze; a cap does not alter the data window or
  rewards before it. None retains the native horizon.
- Default registered scenario/fill semantics and existing exact registry pins
  remain unchanged. Custom caps belong in run configuration, not implicit parity.

## Verification performed

- Old installed package: the new finite contract suite had 22 failures and
  2 characterization passes. Reseeding/cap assertions failed on actual runtime
  values, not just import errors. The newly added native keyword was absent;
  that signature failure alone is not treated as behavior evidence.
- Candidate release wheel in an isolated consumer: 61 tests passed across
  vector contract, vector, registration and vector episode-statistics suites.
- Native core: 196 library tests passed, including unchanged scalar/vector
  bytes and the new unequal-native-horizon/per-lane cap test.
- Removing reseeding in the isolated installed wrapper made both seed-rebuild
  cases fail on observation equality. Restoring the wrapper passed.
- An isolated native cap mutant failed the new core regression. A release
  wheel built from that mutant failed all five installed cap cases: three
  autoreset modes, the direct native binding and the registered factory.
- The isolated native source was restored and the core cap test passed in a
  fresh target directory, avoiding reuse of a stale compiled mutant.
- Native and standalone PyO3 Clippy passed with warnings denied. The final
  installed artifact, committed-tree provenance and remote CI remain delivery
  gates, not implied by the source tests.

No released package, scientific experiment or frozen evidence was generated.
