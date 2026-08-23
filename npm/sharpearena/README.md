# @general-liquidity/sharpearena

A typed TypeScript/JavaScript API over the SharpeArena engine: the leak-free, point-in-time market environment for trading agents. The identical Rust engine that powers the [SharpeBench](https://www.npmjs.com/package/@general-liquidity/sharpebench) harness, compiled to WebAssembly, so a number computed here is byte-identical to the one computed from Rust or Python.

## Install

```bash
npm i @general-liquidity/sharpearena
```

Node >= 18. No native build step: the engine ships as WASM.

## Quickstart

```ts
import { runBaseline } from "@general-liquidity/sharpearena";

// The identical Rust engine, in the browser or Bun: run a baseline over a seeded panel.
const run = runBaseline({
  agent: "momentum",
  dataset: { synthetic: { n_symbols: 4, n_days: 120, seed: 1 } },
  seed: 7,
});
console.log(run.returns.length, run.cost);   // per-period returns + realized execution cost
```

## Tamper-evident replay

A run records only the agent's raw decisions. `replayRun` replays them through the identical engine against the frozen dataset and recomputes the result:

```ts
import { replayRun } from "@general-liquidity/sharpearena";

const replayed = replayRun(dataset, trajectory, costs);
```

The replay is byte-identical to the originally captured run if and only if nothing was altered. A tampered trajectory recomputes to different returns, so an agent cannot lie about what it earned.

Also exported: `datasetSynthetic`, `stressSuite`, `walkForward`, `tagRegime`, and the full wire-contract types (`MarketObservation`, `Decision`, `Run`, `RunTrajectory`, ...).

## Links

- Repository: [general-liquidity/sharpearena](https://github.com/general-liquidity/sharpearena)
- Scoring: [@general-liquidity/sharpebench](https://www.npmjs.com/package/@general-liquidity/sharpebench)

## License

MIT OR Apache-2.0
