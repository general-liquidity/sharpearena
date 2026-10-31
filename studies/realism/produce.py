#!/usr/bin/env python3
"""Producer for the realism control study. See PROTOCOL.md; constants are in protocol.json.

    python studies/realism/produce.py --stage smoke --workers 4
    python studies/realism/produce.py --stage diagnostic --sharpebench-data <dir> --workers 32
    python studies/realism/produce.py --stage confirmation --sharpebench-data <dir> --workers 32

A job is one arm on one band. Finished jobs are skipped, so a killed run resumes by being
started again. Real observations never reach an output file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
from common import Job, StudyRefused  # noqa: E402

PROTOCOL_PATH = HERE / "protocol.json"
ARM_TAGS = {"pos_garch": 1, "neg_gauss": 2, "neg_iid_t": 3, "neg_shuffled_real": 4}
FIXTURE_NAME = "smoke-fixture.csv"


# -- panels -------------------------------------------------------------------


def garch_returns(rng, n_returns: int, n_symbols: int, spec: dict) -> np.ndarray:
    """Independent GARCH(1,1) Student-t lanes with unit unconditional variance."""
    alpha, beta, nu = spec["alpha"], spec["beta"], spec["nu"]
    omega = 1.0 - alpha - beta
    if omega <= 0.0:
        raise StudyRefused("a GARCH control needs alpha + beta below one")
    var = np.ones(n_symbols)
    scale = math.sqrt(nu / (nu - 2.0))
    out = np.zeros((n_returns, n_symbols))
    for t in range(-spec["burn"], n_returns):
        shock = np.sqrt(var) * rng.standard_t(nu, n_symbols) / scale
        if t >= 0:
            out[t] = shock
        var = omega + alpha * shock * shock + beta * var
    return out * spec["scale"]


def shuffled_window(rng, window: np.ndarray) -> np.ndarray:
    """Each symbol permuted on its own: marginals kept, time order and co-movement gone."""
    return np.column_stack([rng.permutation(window[:, c]) for c in range(window.shape[1])])


def read_closes(path: Path) -> tuple[list[str], np.ndarray]:
    """A long ``date,symbol,close`` file as sorted symbols and a (bars, symbols) array."""
    by_date: dict[str, dict[str, float]] = {}
    symbols: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_date.setdefault(row["date"], {})[row["symbol"]] = float(row["close"])
            symbols.add(row["symbol"])
    ordered = sorted(symbols)
    complete = [d for d in sorted(by_date) if len(by_date[d]) == len(ordered)]
    return ordered, np.array([[by_date[d][s] for s in ordered] for d in complete])


def real_windows(
    path: Path, sha256: str | None, n_returns: int, n_symbols: int, stride: int
) -> list[np.ndarray]:
    """Consecutive non-overlapping return windows on the first ``n_symbols`` sorted symbols.

    ``sha256`` is the protocol's hash for the file. ``None`` is the smoke fixture, which has
    no identity to check.
    """
    if sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha256:
            raise StudyRefused(f"{path.name} hash {actual} is not the protocol's {sha256}")
    symbols, closes = read_closes(path)
    if len(symbols) < n_symbols:
        raise StudyRefused(f"{path.name} has {len(symbols)} symbols, needs {n_symbols}")
    returns = np.diff(np.log(closes[:, :n_symbols]), axis=0)
    if stride < n_returns:
        raise StudyRefused("overlapping windows are refused")
    count = 1 + (returns.shape[0] - n_returns) // stride
    return [returns[i * stride : i * stride + n_returns] for i in range(count)]


def tier_panel(seed: int, tier: str, vol_clustering: float, geometry: dict, check: bool) -> np.ndarray:
    """A flat-policy price panel, collected as F4 collects it."""
    from sharpearena import SharpeArenaEnv, check_env_effective_config

    n_symbols, n_days = geometry["n_symbols"], geometry["n_returns"]
    env = SharpeArenaEnv(
        n_symbols=n_symbols,
        n_days=n_days,
        seed=seed,
        distribution_mode=tier,
        vol_clustering=vol_clustering,
    )
    if check:
        check_env_effective_config(
            env,
            seed=seed,
            n_symbols=n_symbols,
            n_days=n_days,
            distribution_mode=tier,
            vol_clustering=vol_clustering,
        )
    obs, _ = env.reset()
    closes = [np.asarray(obs["closes"], dtype=np.float64).reshape(-1)]
    flat = np.zeros(n_symbols, dtype=np.float32)
    while True:
        obs, _, terminated, truncated, _ = env.step(flat)
        closes.append(np.asarray(obs["closes"], dtype=np.float64).reshape(-1))
        if terminated or truncated:
            break
    panel = np.stack(closes, axis=0)
    if panel.shape != (n_days + 1, n_symbols):
        raise StudyRefused(f"tier panel has shape {panel.shape}")
    return panel


def grade(panel: np.ndarray, kind: str, lags: int) -> dict:
    from sharpearena import certify_realism

    report = certify_realism(panel, kind=kind, abs_acf_lags=lags)
    return {
        "passed": bool(report.passed),
        "checks": {k: bool(v) for k, v in sorted(report.checks.items())},
        "facts": {
            k: (float(v) if math.isfinite(v) else None) for k, v in sorted(report.facts.items())
        },
    }


# -- jobs ---------------------------------------------------------------------


def run_price_job(p: dict) -> dict:
    """Grade one arm on one band. Rows hold statistics only."""
    geometry, arm = p["geometry"], p["arm"]
    lags = geometry["abs_acf_lags"]
    n_returns, n_symbols = geometry["n_returns"], geometry["n_symbols"]
    rows = []
    if p["family"] == "tier":
        for index, seed in enumerate(p["seeds"]):
            panel = tier_panel(seed, p["tier"], p["vol_clustering"], geometry, index == 0)
            rows.append({"id": seed, **grade(panel, "price", lags)})
    elif p["family"] == "synthetic":
        spec = p["spec"]
        for seed in p["seeds"]:
            rng = np.random.default_rng((ARM_TAGS[arm], seed))
            if arm == "pos_garch":
                panel = garch_returns(rng, n_returns, n_symbols, spec)
            elif arm == "neg_gauss":
                panel = rng.standard_normal((n_returns, n_symbols)) * spec["scale"]
            else:
                panel = rng.standard_t(spec["nu"], (n_returns, n_symbols)) * spec["scale"]
            rows.append({"id": seed, **grade(panel, "return", lags)})
    elif p["family"] == "real":
        windows = real_windows(Path(p["path"]), p["sha256"], n_returns, n_symbols, p["stride"])
        for index, window in enumerate(windows):
            rows.append({"id": index, **grade(window, "return", lags)})
    elif p["family"] == "shuffled":
        pools = [
            real_windows(Path(src["path"]), src["sha256"], n_returns, n_symbols, p["stride"])
            for src in p["sources"]
        ]
        for index, seed in enumerate(p["seeds"]):
            pool = pools[index % len(pools)]
            rng = np.random.default_rng((ARM_TAGS[arm], seed))
            rows.append(
                {"id": seed, **grade(shuffled_window(rng, pool[index % len(pool)]), "return", lags)}
            )
    else:
        raise StudyRefused(f"unknown family {p['family']}")
    return {"arm": arm, "band": p["band"], "rows": rows}


def _quote(pair: Sequence[int]) -> np.ndarray:
    return np.asarray(pair, dtype=np.float32)


def seat_fill_share(seed: int, spec: dict, priority: str) -> dict:
    """E1: fill share of seat 0 between two identical quoters."""
    from sharpearena.lob_env import LOBMarketEnv

    class Counting(LOBMarketEnv):
        def _apply_fills(self, fills, ladder):
            for fill in fills:
                owner = self._owner(fill["maker_agent"])
                if owner is not None:
                    self.study_fills[owner] += fill["qty"]
            return super()._apply_fills(fills, ladder)

    env = Counting(
        spec["n_agents"], n_steps=spec["n_steps"], seed=seed, mark=spec["mark"], priority=priority
    )
    env.study_fills = [0] * spec["n_agents"]
    env.reset(seed=seed)
    env.study_fills = [0] * spec["n_agents"]
    quote = _quote(spec["quote"])
    while env.agents:
        env.step({agent: quote for agent in env.agents})
    total = sum(env.study_fills)
    if total == 0:
        return {"seed": seed, "share": None, "fills": env.study_fills}
    return {"seed": seed, "share": env.study_fills[0] / total, "fills": env.study_fills}


def mark_walk(seed: int, spec: dict, walk: dict, mark: str) -> dict:
    """E2: a lone quoter walks the reference mid, then tightens, with no inventory penalty.

    ``no_fill_pnl`` is the absolute reward booked on steps where the agent traded with
    nobody. ``book_mid`` books its own quote walk there; ``ex_own_mid`` should not.
    """
    from sharpearena.lob_env import LOBMarketEnv

    class Traded(LOBMarketEnv):
        def _apply_fills(self, fills, ladder):
            self.study_traded = any(
                self._owner(f["maker_agent"]) == 0 or self._owner(f["taker_agent"]) == 0
                for f in fills
            )
            return super()._apply_fills(fills, ladder)

    env = Traded(1, n_steps=spec["n_steps"], seed=seed, inventory_penalty=0.0, mark=mark)
    env.study_traded = False
    env.reset(seed=seed)
    total, absolute, no_fill, step = 0.0, 0.0, 0.0, 0
    while env.agents:
        quote = walk["walk_quote"] if step < walk["walk_steps"] else walk["tight_quote"]
        reward = float(env.step({"agent_0": _quote(quote)})[1]["agent_0"])
        total += reward
        absolute += abs(reward)
        if not env.study_traded:
            no_fill += abs(reward)
        step += 1
    value = env._cash["agent_0"] + env._inventory["agent_0"] * env._mid
    return {"seed": seed, "reward": total, "value": float(value),
            "absolute_pnl": absolute, "no_fill_pnl": no_fill}


def run_execution_job(p: dict) -> dict:
    spec = p["execution"]
    lob = spec["lob"]
    check = p["check"]
    if check == "e1":
        rows = [seat_fill_share(s, lob, p["priority"]) for s in p["seeds"]]
        return {"check": check, "priority": p["priority"], "mark": lob["mark"], "controllers": "per_seat", "rows": rows}
    if check == "e2":
        rows = [mark_walk(s, lob, spec["mark_walk"], p["mark"]) for s in p["seeds"]]
        return {"check": check, "mark": p["mark"], "priority": "agent_index", "controllers": "per_seat", "rows": rows}
    if check == "e3":
        from sharpearena.impact_diagnostics import meta_order_impact_shape

        rows = []
        for seed in p["seeds"]:
            shape = meta_order_impact_shape(impact_exponent=p["impact_exponent"], seed=seed)
            rows.append(
                {
                    "seed": seed,
                    "execution_exponent": float(shape.execution_exponent),
                    "duration_exponent": float(shape.duration_exponent),
                    "max_relaxation_ratio": float(max(shape.relaxation_ratios)),
                    "min_relaxation_ratio": float(min(shape.relaxation_ratios)),
                }
            )
        return {"check": check, "impact_exponent": p["impact_exponent"], "rows": rows}
    raise StudyRefused(f"unknown check {check}")


def run_job(p: dict) -> dict:
    return run_price_job(p) if p["kind"] == "price" else run_execution_job(p)


def build_jobs(protocol: dict, stage: str, sources: list[dict]) -> list[Job]:
    geometry = protocol["geometry"]
    bands_spec = protocol["bands"]
    if stage == "smoke":
        bands = {"smoke": bands_spec["smoke"]}
    elif stage == "diagnostic":
        bands = {"diagnostic": bands_spec["diagnostic"]}
    elif stage == "confirmation":
        bands = {"confirmation": bands_spec["confirmation"]}
    else:
        raise StudyRefused(f"unknown stage {stage}")
    jobs: list[Job] = []
    stride = protocol["real_controls"]["stride_returns"]
    synthetic = protocol["synthetic_controls"]
    for name, spec in bands.items():
        seeds = common.band(spec["start"], spec["count"])
        base = {"kind": "price", "geometry": geometry, "band": name, "seeds": seeds}
        for arm, arm_spec in synthetic.items():
            jobs.append(Job(f"price/{name}/{arm}", {**base, "family": "synthetic", "arm": arm, "spec": arm_spec}))
        for tier in protocol["generator"]["tiers"]:
            jobs.append(
                Job(f"price/{name}/{tier}", {**base, "family": "tier", "arm": tier, "tier": tier,
                                             "vol_clustering": protocol["generator"]["primary"]["vol_clustering"]})
            )
            if tier in protocol["generator"]["descriptive_tiers"]:
                jobs.append(
                    Job(f"price/{name}/{tier}_vc05", {**base, "family": "tier", "arm": f"{tier}_vc05", "tier": tier,
                                                      "vol_clustering": protocol["generator"]["descriptive"]["vol_clustering"]})
                )
        jobs.append(
            Job(f"price/{name}/neg_shuffled_real", {**base, "family": "shuffled", "arm": "neg_shuffled_real",
                                                    "sources": sources, "stride": stride})
        )
        execution = protocol["execution"]
        lob = execution["lob"]
        for priority in (lob["priority"], lob["negative_priority"]):
            jobs.append(Job(f"execution/{name}/e1_{priority}", {"kind": "execution", "check": "e1", "band": name,
                                                                "seeds": seeds, "priority": priority, "execution": execution}))
        for mark in (lob["mark"], lob["negative_mark"]):
            jobs.append(Job(f"execution/{name}/e2_{mark}", {"kind": "execution", "check": "e2", "band": name,
                                                            "seeds": seeds, "mark": mark, "execution": execution}))
        prefix = seeds[: execution["impact_shape"]["seeds_from_band_prefix"]]
        for exponent in execution["impact_shape"]["impact_exponents"]:
            jobs.append(Job(f"execution/{name}/e3_{exponent}", {"kind": "execution", "check": "e3", "band": name,
                                                                "seeds": prefix, "impact_exponent": exponent, "execution": execution}))
    if stage in ("smoke", "diagnostic"):
        for src in sources:
            jobs.append(
                Job(f"price/real/{src['name']}", {"kind": "price", "family": "real", "geometry": geometry, "band": "real",
                                                  "arm": f"pos_real:{src['name']}", "path": src["path"],
                                                  "sha256": src["sha256"], "stride": stride})
            )
    return jobs


# -- the decision rule --------------------------------------------------------


def rate_row(rows: list[dict], confidence: float, family_size: int = 1) -> dict:
    n = len(rows)
    passes = sum(1 for r in rows if r["passed"])
    alpha = (1.0 - confidence) / family_size
    fact_passes: dict[str, int] = {}
    for row in rows:
        for fact, ok in row["checks"].items():
            fact_passes[fact] = fact_passes.get(fact, 0) + int(ok)
    return {
        "n": n,
        "passes": passes,
        "rate": passes / n,
        "lower": common.cp_lower(passes, n, 1.0 - alpha),
        "upper": common.cp_upper(passes, n, 1.0 - alpha),
        "fact_passes": dict(sorted(fact_passes.items())),
        "failing_facts": sorted(k for k, v in fact_passes.items() if v < n),
    }


def gate_validity(price: dict, real: dict, protocol: dict, bands: Sequence[str]) -> dict:
    """Rule 1. Anything missing makes the answer unavailable, never a pass."""
    t = protocol["thresholds"]
    reasons: list[str] = []
    for band_name in bands:
        arms = price.get(band_name, {})
        pos = arms.get("pos_garch")
        if pos is None:
            reasons.append(f"{band_name}:pos_garch:missing")
        elif pos["lower"] < t["tau_market_like"]:
            reasons.append(f"{band_name}:pos_garch:lower_below_tau")
        for arm in ("neg_gauss", "neg_iid_t", "neg_shuffled_real"):
            neg = arms.get(arm)
            if neg is None:
                reasons.append(f"{band_name}:{arm}:missing")
            elif neg["upper"] > t["tau_not_market_like_upper"]:
                reasons.append(f"{band_name}:{arm}:upper_above_limit")
    if not real:
        reasons.append("pos_real:missing")
    for name, row in real.items():
        if row["rate"] < t["tau_market_like"]:
            reasons.append(f"pos_real:{name}:share_below_tau")
    return {"valid": not reasons, "reasons": reasons}


def tier_verdict(tier: str, price: dict, validity: dict, protocol: dict, bands: Sequence[str]) -> str:
    """Rule 2 and rule 3."""
    if not validity["valid"]:
        return "unavailable:gate_failed_control"
    tau = protocol["thresholds"]["tau_market_like"]
    rows = [price.get(b, {}).get(tier) for b in bands]
    if any(r is None for r in rows):
        return "unavailable:missing_band"
    if all(r["lower"] >= tau for r in rows):
        return "market_like_under_gate"
    if all(r["upper"] < tau for r in rows):
        return "not_market_like_under_gate"
    return "inconclusive"


def execution_block(results: dict[str, dict], protocol: dict, band_name: str) -> dict:
    """E1 to E3, reported alone. No line here enters a price verdict."""
    t = protocol["thresholds"]
    lob = protocol["execution"]["lob"]
    margin = t["exchangeability_margin"]

    def share(priority: str) -> dict | None:
        job = results.get(f"execution/{band_name}/e1_{priority}")
        if job is None:
            return None
        rows = job["rows"]
        kept = [r["share"] - 0.5 for r in rows if r["share"] is not None]
        interval = common.t_interval(kept, 0.90)
        return {"priority": priority, "mark": job["mark"], "controllers": job["controllers"],
                "n": len(rows), "n_with_fills": len(kept), "difference": interval}

    good, bad = share(lob["priority"]), share(lob["negative_priority"])

    def inside(row: dict | None) -> bool:
        d = (row or {}).get("difference", {})
        return "lo" in d and d["lo"] >= -margin and d["hi"] <= margin

    def above(row: dict | None) -> bool:
        d = (row or {}).get("difference", {})
        return "lo" in d and d["lo"] > margin

    def walk(mark: str) -> dict | None:
        job = results.get(f"execution/{band_name}/e2_{mark}")
        if job is None:
            return None
        rows = job["rows"]
        absolute = float(sum(r["absolute_pnl"] for r in rows))
        return {
            "mark": mark,
            "n": len(rows),
            "no_fill_pnl_share": float(sum(r["no_fill_pnl"] for r in rows)) / absolute if absolute > 0.0 else None,
            "mean_reward_minus_value": float(np.mean([r["reward"] - r["value"] for r in rows])),
            "mean_value": float(np.mean([r["value"] for r in rows])),
        }

    good_walk, bad_walk = walk(lob["mark"]), walk(lob["negative_mark"])
    limit = t["no_fill_pnl_share_max"]

    def within(row: dict | None) -> bool | None:
        if row is None or row["no_fill_pnl_share"] is None:
            return None
        return row["no_fill_pnl_share"] <= limit

    shapes = {}
    low, high = t["impact_exponent_band"]
    for exponent in protocol["execution"]["impact_shape"]["impact_exponents"]:
        job = results.get(f"execution/{band_name}/e3_{exponent}")
        if job is None:
            shapes[str(exponent)] = {"unavailable": "missing"}
            continue
        exec_exp = float(np.mean([r["execution_exponent"] for r in job["rows"]]))
        relax = float(np.mean([r["max_relaxation_ratio"] for r in job["rows"]]))
        shapes[str(exponent)] = {
            "n": len(job["rows"]),
            "mean_execution_exponent": exec_exp,
            "mean_max_relaxation_ratio": relax,
            "meets_descriptive_shape": bool(low <= exec_exp <= high and relax <= t["relaxation_ratio_max"]),
        }
    return {
        "e1": {"good": good, "bad": bad, "exchangeable": inside(good), "bad_arm_detected": above(bad)},
        "e2": {"good": good_walk, "bad": bad_walk, "good_within_tolerance": within(good_walk),
               "bad_exceeds_tolerance": None if within(bad_walk) is None else not within(bad_walk)},
        "e3": shapes,
    }


def summarize(protocol: dict, jobs: list[Job], out: Path, stage: str, prior: dict | None) -> dict:
    confidence = protocol["thresholds"]["confidence_one_sided"]
    results, absent = common.collect(out, [j.key for j in jobs])
    price: dict[str, dict] = {}
    real: dict[str, dict] = dict(prior["real"]) if prior else {}
    tiers = set(protocol["generator"]["tiers"])
    for key, result in results.items():
        if not key.startswith("price/"):
            continue
        band_name, arm = result["band"], result["arm"]
        if band_name == "real":
            real[arm.split(":", 1)[1]] = rate_row(result["rows"], confidence)
        else:
            family = protocol["thresholds"]["tier_family_size"] if arm in tiers else 1
            price.setdefault(band_name, {})[arm] = rate_row(result["rows"], confidence, family)
    if prior:
        price = {**prior["price"], **price}
    bands = ["smoke"] if stage == "smoke" else ["diagnostic", "confirmation"]
    validity = gate_validity(price, real, protocol, bands)
    final = stage == "confirmation"
    verdicts = {
        t: tier_verdict(t, price, validity, protocol, bands) if final else "unavailable:not_final_stage"
        for t in protocol["generator"]["tiers"]
    }
    band_for_execution = bands[-1] if stage != "diagnostic" else "diagnostic"
    execution = dict(prior["execution_realism"]) if prior else {}
    execution[band_for_execution] = execution_block(results, protocol, band_for_execution)
    return {
        "stage": stage,
        "price_realism": {"gate": validity, "verdicts": verdicts},
        "price": price,
        "real": real,
        "execution_realism": execution,
        "absent_jobs": absent,
        "jobs_expected": len(jobs),
    }


def write_fixture(path: Path, bars: int = 400) -> None:
    """A seeded random walk for the smoke run, so no real series enters the gate early."""
    rng = np.random.default_rng(0x5EED)
    path.parent.mkdir(parents=True, exist_ok=True)
    closes = 100.0 * np.exp(np.cumsum(rng.standard_normal((bars, 5)) * 0.01, axis=0))
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("date,symbol,close\n")
        for t in range(bars):
            for c in range(5):
                handle.write(f"d{t:05d},S{c},{closes[t, c]:.8f}\n")


def resolve_sources(args, protocol: dict, out: Path) -> list[dict]:
    if args.stage == "smoke":
        fixture = out / FIXTURE_NAME
        write_fixture(fixture)
        return [{"name": "fixture", "path": str(fixture), "sha256": None}]
    if args.sharpebench_data is None:
        raise StudyRefused("this stage needs --sharpebench-data")
    root = Path(args.sharpebench_data)
    sources = []
    for name, spec in protocol["real_controls"]["datasets"].items():
        path = root / name
        windows = real_windows(
            path, spec["sha256"], protocol["geometry"]["n_returns"], protocol["geometry"]["n_symbols"],
            protocol["real_controls"]["stride_returns"],
        )
        if len(windows) != spec["expected_windows"]:
            raise StudyRefused(f"{name} gives {len(windows)} windows, protocol says {spec['expected_windows']}")
        sources.append({"name": name, "path": str(path), "sha256": spec["sha256"]})
    return sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "diagnostic", "confirmation"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sharpebench-data", default=None)
    parser.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        metavar="REASONS.json",
        help="retry the failed keys named in this {key: infrastructure reason} file",
    )
    args = parser.parse_args(argv)
    retry_reasons = common.load_retry_reasons(args.retry_failed)

    protocol, sha = common.load_protocol(PROTOCOL_PATH)
    root = args.out if args.out is not None else HERE / "runs"
    out = root / args.stage
    common.require_disjoint(
        {k: common.band(v["start"], v["count"]) for k, v in protocol["bands"].items()}
    )
    prior = None
    if args.stage != "smoke":
        common.require_clean_tree(args.stage)
    if args.stage == "confirmation":
        path = root / "diagnostic" / "summary.json"
        if not path.exists():
            raise StudyRefused("confirmation needs a finished diagnostic summary")
        prior = common.read_json(path)
        if prior.get("absent_jobs"):
            raise StudyRefused("the diagnostic stage has failed or missing jobs")
    sources = resolve_sources(args, protocol, out)
    jobs = build_jobs(protocol, args.stage, sources)
    counts = common.run_jobs(jobs, run_job, out, args.workers, retry_reasons=retry_reasons)
    ledger = common.attempt_ledger(out)
    summary = summarize(protocol, jobs, out, args.stage, prior)
    summary["attempt_ledger"] = ledger
    common.write_json(out / "summary.json", summary)
    common.write_json(
        out / "manifest.json",
        common.manifest(
            protocol["protocol_id"], protocol["version"], sha, args.stage,
            {"datasets": {s["name"]: s["sha256"] for s in sources}, "attempt_ledger": ledger},
        ),
    )
    print(f"{args.stage}: {counts}; core-seconds {common.core_seconds(out):.1f}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
