// Throughput benchmark for the WebAssembly surface: scenario generation and a full
// in-kernel baseline episode. Run with `node bench/throughput.js` from the package root.
// The panel matches the native (`cargo run --release -p sharpearena --example bench-steps`)
// and Python (`paper/src/make-throughput.py`) benchmarks so the surfaces are comparable.

const kernel = require("../pkg/sharpearena.js");

const N_SYMBOLS = 4;
const N_DAYS = 120;
const EPISODES = 500;
const REPEATS = 3;
// `run_baseline` defaults to a 20-bar warm-up on a 120-day panel.
const STEPS_PER_EPISODE = 100;

function benchGeneration(episodes) {
  const start = process.hrtime.bigint();
  let total = 0;
  for (let seed = 0; seed < episodes; seed += 1) {
    const out = kernel.generate_scenario(
      JSON.stringify({
        spec: {
          start_level: 0,
          num_levels: 0,
          n_symbols: N_SYMBOLS,
          n_days: N_DAYS,
          distribution_mode: "calm",
        },
        seed,
      }),
    );
    total += out.length;
  }
  const elapsed = Number(process.hrtime.bigint() - start) / 1e9;
  if (total <= 0) throw new Error("empty scenario");
  return episodes / elapsed;
}

function benchBaseline(episodes) {
  const start = process.hrtime.bigint();
  for (let seed = 0; seed < episodes; seed += 1) {
    kernel.run_baseline(
      JSON.stringify({
        agent: "buy_and_hold",
        dataset: { synthetic: { n_symbols: N_SYMBOLS, n_days: N_DAYS, seed } },
        seed,
      }),
    );
  }
  const elapsed = Number(process.hrtime.bigint() - start) / 1e9;
  return {
    stepsPerSecond: (episodes * STEPS_PER_EPISODE) / elapsed,
    episodeMs: (1e3 * elapsed) / episodes,
  };
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// Warm the JIT and the wasm instance before timing.
benchGeneration(8);
benchBaseline(8);

const gen = [];
const steps = [];
const epMs = [];
for (let i = 0; i < REPEATS; i += 1) {
  gen.push(benchGeneration(EPISODES));
  const b = benchBaseline(EPISODES);
  steps.push(b.stepsPerSecond);
  epMs.push(b.episodeMs);
}

console.log(
  JSON.stringify(
    {
      panel: { n_symbols: N_SYMBOLS, n_days: N_DAYS },
      episodes_per_repeat: EPISODES,
      repeats: REPEATS,
      steps_per_episode: STEPS_PER_EPISODE,
      wasm_generation_scenarios_per_s: { median: median(gen), runs: gen },
      wasm_steps_per_s: { median: median(steps), runs: steps },
      wasm_episode_wall_ms: { median: median(epMs), runs: epMs },
      node: process.version,
    },
    null,
    2,
  ),
);
