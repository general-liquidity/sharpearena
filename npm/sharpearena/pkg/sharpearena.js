/* @ts-self-types="./sharpearena.d.ts" */
import * as wasm from "./sharpearena_bg.wasm";
import { __wbg_set_wasm } from "./sharpearena_bg.js";

__wbg_set_wasm(wasm);
wasm.__wbindgen_start();
export {
    dataset_synthetic, generate_scenario, replay_run, run_baseline, stress_suite, tag_regime, walk_forward
} from "./sharpearena_bg.js";
