r"""Try to break the zero-mean claim on seeds that played no part in the choice.

    cargo run --release -p sharpearena --example dump-scenario-panels -- <panels>/stress 40000 4096 4 120 martingale_v1
    python studies/generator-calibration/stress.py <panels>/stress

Report seeds gave a mean-return interval whose lower end sat at zero and an equal-weight
Sharpe of +0.21. Either that is sampling noise or the design claim is wrong, and 128 seeds
cannot say which. This band is 32 times larger, disjoint from the fit, report and frozen
realism bands, and changes no parameter: its only output is whether the claim survived.
Writes ``stress.json`` beside this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from stats import annualized_sharpe, lag1_autocorr, load_panels, simple_returns, strategy_returns, summarize

SEEDS = {"start": 40000, "count": 4096}


def main() -> None:
    panels = load_panels(Path(sys.argv[1]) / "martingale_v1.f64", 120, 4)
    if len(panels) != SEEDS["count"]:
        raise SystemExit(f"expected {SEEDS['count']} panels, got {len(panels)}")
    rows: dict[str, list[float]] = {}
    for closes in panels:
        r = simple_returns(closes)
        rows.setdefault("mean_daily_return", []).append(float(r.mean()))
        rows.setdefault("lag1_autocorr", []).append(lag1_autocorr(r))
        for name, series in strategy_returns(r).items():
            rows.setdefault(f"sharpe_{name}", []).append(float(annualized_sharpe(series[:, None])[0]))
    out = {
        "seeds": SEEDS,
        "note": "falsification only; no parameter was or may be changed from this band",
        "finite_sample_lag1_expectation": -1.0 / 119.0,
        "stats": {k: summarize(np.array(v)) for k, v in rows.items()},
    }
    path = Path(__file__).with_name("stress.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for key, s in out["stats"].items():
        print(f"{key:36s} {s['mean']:+.6f} [{s['ci95_low']:+.6f},{s['ci95_high']:+.6f}]")


if __name__ == "__main__":
    main()
