const test = require("node:test");
const assert = require("node:assert");
const oo = require("../dist/index.js");

// A JS string reaches the engine as UTF-8. A lone UTF-16 surrogate has no UTF-8
// encoding: `TextEncoder` would hand the engine U+FFFD in its place and the run would
// succeed on text the caller never wrote. The wrapper refuses it by name instead.

const HIGH = "\uD800";
const LOW = "\uDC00";

const csv = (name) =>
  `date,symbol,close\n2025-01-01,${name},10\n2025-01-02,${name},11\n2025-01-03,${name},12\n`;
const config = (name) => ({
  agent: "hold",
  dataset: { csv: csv(name) },
  window: { start: 0, end: 3 },
  seed: 1,
});

function assertRefused(f, where) {
  assert.throws(f, (e) => {
    assert.ok(e instanceof oo.InvalidUtf8Error, `expected InvalidUtf8Error, got ${e.name}: ${e.message}`);
    assert.match(e.message, /lone UTF-16 surrogate/);
    assert.ok(e.message.includes(where), `message names ${where}: ${e.message}`);
    return true;
  });
}

test("a lone high surrogate in a string value is refused by name", () => {
  assertRefused(() => oo.runBaseline(config("AAA" + HIGH)), "config.csv");
});

test("a lone low surrogate in a string value is refused by name", () => {
  assertRefused(() => oo.runBaseline(config(LOW + "AAA")), "config.csv");
});

test("a surrogate pair in the wrong order is refused", () => {
  assertRefused(() => oo.runBaseline(config("AAA" + LOW + HIGH)), "config.csv");
});

test("a lone surrogate in an object key is refused", () => {
  const dataset = oo.datasetSynthetic({ n_symbols: 2, n_days: 30, seed: 3 });
  const [first] = Object.keys(dataset.closes);
  dataset.closes["X" + HIGH] = dataset.closes[first];
  assertRefused(() => oo.tagRegime(dataset, { start: 0, end: 30 }), "input key");
});

test("every entry point that takes text checks it", () => {
  const dataset = oo.datasetSynthetic({ n_symbols: 2, n_days: 30, seed: 3 });
  const bad = { ...dataset, dates: [...dataset.dates] };
  bad.dates[0] = "2025" + HIGH;
  assertRefused(() => oo.tagRegime(bad, { start: 0, end: 30 }), "input.0");
  assertRefused(() => oo.replayRun(bad, { steps: [] }), "dataset.0");
  assertRefused(() => oo.replayRun(dataset, { steps: [], note: HIGH }), "trajectory.note");
  assertRefused(() => oo.generateScenario({ spec: { name: HIGH } }), "input.name");
});

test("well-formed text, including a real pair and a literal U+FFFD, still runs", () => {
  for (const name of ["AAA", "A😀A", "A�A"]) {
    const run = oo.runBaseline(config(name));
    assert.equal(run.returns.length, 3, name);
  }
});
