/**
 * `@general-liquidity/sharpearena` — the trading-agent environment (a leak-free,
 * point-in-time market simulator + reference baselines + recompute-to-verify) as a
 * typed JS API over the *identical* Rust engine that powers the SharpeBench harness
 * (compiled to WebAssembly).
 *
 * The harness sends a {@link MarketObservation}, the agent returns a {@link Decision},
 * repeat — and look-ahead is structurally impossible. A run's returns are recomputed
 * from raw decisions ({@link replayRun}), never trusted from the agent's word.
 */
import * as kernel from "../pkg/sharpearena.js";

import { checkSpecHash } from "./specHash.js";

import type {
  BaselineConfig,
  CostModel,
  Dataset,
  Regime,
  Run,
  RunTrajectory,
  ScenarioInput,
  StressScenario,
  SyntheticParams,
  WalkForwardParams,
  Window,
} from "./types.js";

export * from "./types.js";
// `compareSpecHash` stays internal on purpose: see its doc comment in specHash.ts.
export { checkSpecHash, SPEC_HASH } from "./specHash.js";

// Refuse-on-mismatch spec-hash handshake, at load: the committed wasm bundle must
// report the tape-semantics hash this wrapper was generated against, or every number
// it would produce is suspect. `spec_hash` is read leniently so an engine that
// predates the handshake is diagnosed by name, not dropped as malformed.
checkSpecHash(
  typeof (kernel as { spec_hash?: () => string }).spec_hash === "function"
    ? (kernel as { spec_hash: () => string }).spec_hash()
    : undefined,
);

/** Parse a kernel JSON string, surfacing the kernel's `{error}` as a thrown Error. */
function parse<T>(json: string): T {
  const value = JSON.parse(json) as unknown;
  if (value && typeof value === "object" && "error" in value) {
    throw new Error(String((value as { error: unknown }).error));
  }
  return value as T;
}

/**
 * A lone UTF-16 surrogate: a high surrogate not followed by a low one, or a low
 * surrogate not preceded by a high one. Such a string has no UTF-8 encoding.
 */
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

/** The refusal for text that cannot cross into the engine as UTF-8. */
export class InvalidUtf8Error extends Error {
  constructor(where: string) {
    super(
      `InvalidUtf8: ${where} contains a lone UTF-16 surrogate, which has no UTF-8 ` +
        `encoding. Refusing it rather than letting it reach the engine as U+FFFD.`,
    );
    this.name = "InvalidUtf8Error";
  }
}

function checkText(text: string, where: string): void {
  if (LONE_SURROGATE.test(text)) throw new InvalidUtf8Error(where);
}

/**
 * Serialize a kernel argument, refusing any key or string value the engine could only
 * receive as replacement characters. Without this the refusal would depend on which
 * path the text took: `JSON.stringify` escapes a lone surrogate as `\ud800`, which
 * serde_json rejects with an unrelated parse message, while a string handed to the
 * glue directly is encoded by `TextEncoder`, which substitutes U+FFFD and succeeds.
 * Each export returns a Rust `String`, valid UTF-8 by construction, so the other
 * direction cannot carry an invalid sequence.
 */
function toJson(value: unknown, what: string): string {
  return JSON.stringify(value, function (key: string, v: unknown) {
    checkText(key, `${what} key`);
    if (typeof v === "string") checkText(v, key === "" ? what : `${what}.${key}`);
    return v;
  });
}

/** Empty/absent config → `""`, which the kernel reads as "use defaults". */
function optJson(value: object | undefined, what: string): string {
  if (!value || Object.keys(value).length === 0) return "";
  return toJson(value, what);
}

/**
 * Run a named in-process baseline (`buy_and_hold` | `hold` | `momentum` | `random`)
 * over a dataset for a window + seed + cost model, returning the {@link Run} — its
 * per-period returns, decision trace, and per-step confidences/outcomes.
 */
export function runBaseline(config: BaselineConfig): Run {
  return parse(kernel.run_baseline(toJson(config, "config")));
}

/**
 * Replay a captured {@link RunTrajectory}'s raw decisions through the identical
 * engine to regenerate its {@link Run} — the tamper-evidence path. The result is
 * byte-identical to the originally captured run iff `dataset` and `costs` match what
 * it was captured against.
 */
export function replayRun(
  dataset: Dataset,
  trajectory: RunTrajectory,
  costs?: CostModel,
): Run {
  return parse(
    kernel.replay_run(
      toJson(dataset, "dataset"),
      toJson(trajectory, "trajectory"),
      optJson(costs, "costs"),
    ),
  );
}

/** Build a deterministic synthetic {@link Dataset} from `{n_symbols, n_days, seed}`. */
export function datasetSynthetic(params?: SyntheticParams): Dataset {
  return parse(kernel.dataset_synthetic(optJson(params, "params")));
}

/**
 * Draw one level of a procedural scenario family: the {@link Dataset} for
 * `input.spec` under `input.seed`. Deterministic, and the export the cross-runtime
 * byte-identity claim is written about (its output is pinned against the committed
 * goldens in `contract/attestation/scenario-goldens.json` from Rust, from wasm32 and
 * from this package's own suite).
 *
 * It is wrapped here, rather than left to `pkg/`, because a consumer who has to reach
 * past the wrapper to call it also reaches past the spec-hash handshake that runs at
 * this module's load. See finding A3 in the 2026-09-09 Arena review.
 */
export function generateScenario(input?: ScenarioInput): Dataset {
  return parse(kernel.generate_scenario(optJson(input, "input")));
}

/** The named adversarial stress suite (flash-crash, whipsaw, …) for a seed. */
export function stressSuite(seed = 0): StressScenario[] {
  return parse(kernel.stress_suite(toJson({ seed }, "params")));
}

/**
 * Generate walk-forward out-of-sample {@link Window}s. The windows are disjoint only
 * when `step >= test`; with `step < test` consecutive windows share `test - step` bars.
 */
export function walkForward(params: WalkForwardParams): Window[] {
  return parse(kernel.walk_forward(toJson(params, "params")));
}

/** Tag a window's coarse market regime (bull / bear / chop). */
export function tagRegime(dataset: Dataset, window: Window): Regime {
  const out = parse<{ regime: Regime }>(
    kernel.tag_regime(toJson({ dataset, window }, "input")),
  );
  return out.regime;
}
