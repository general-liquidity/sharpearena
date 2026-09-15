#!/usr/bin/env python3
"""Re-render every paper figure from the committed evidence JSON.

Reads paper/evidence/*.json and reproduces the figures the make-* scripts emit,
without re-running any experiment. Every bar and point is a reduction over the
committed records; no number is typed into this script.

The dispatch is an explicit registry: one entry per renderer, naming the
evidence file it reads and every PDF it writes. Every figure's layout lives in
its producer, and each entry calls that producer's own figure function on the
committed record, so a rebuild cannot drift from a full run and no layout is
maintained twice. ``main`` closes the registry against the
figure directory and prints an omission notice naming any committed PDF no
entry claims, so an incomplete rebuild says so instead of quietly leaving stale
plots in place. Evidence files that do not exist yet are skipped with a notice
that names the figures they would have produced.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")

PAPER = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"


def _load(name: str) -> dict | None:
    path = EVIDENCE / name
    if not path.exists():
        print(f"skip: {path} (evidence not generated yet)")
        return None
    return json.loads(path.read_text())


def producer(stem: str):
    """Import a hyphenated producer module by path, once.

    The producers guard their ``sharpearena`` import so this stays a
    frozen-input path: importing one to reach its figure function does not
    require the native bindings, only the committed JSON it is handed.
    """
    name = stem.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SRC / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def f1(data: dict) -> None:
    producer("make-f1-baselines").make_figure(data)


def f2(data: dict) -> None:
    producer("make-f2-regret").make_figure(data)


def f4(data: dict) -> None:
    producer("make-f4-realism")._plot_realism(data["tiers"])


def f6(data: dict) -> None:
    producer("make-f6-adverse-selection").markouts_figure(data["comparison"])


def f7(data: dict) -> None:
    producer("make-f7-failures").make_figure(data)


def f8(data: dict) -> None:
    producer("make-f8-ecology").make_figures(data)


def f4_calm_calibration(data: dict) -> None:
    producer("make-f4-realism")._plot_calm_calibration(data["calm_calibration"])


def f5(data: dict) -> None:
    producer("make-f5-manipulation").make_figures(data)


def f6_endogenous(data: dict) -> None:
    producer("make-f6-adverse-selection").endogenous_figure(data["endogenous"])


def predictability(data: dict) -> None:
    producer("make-predictability").make_figure(data["tiers"])


def witness(data: dict) -> None:
    producer("make-witness").make_figure(data)


class Renderer(NamedTuple):
    """One dispatch entry: the record it reads and every PDF it writes.

    ``figures`` is the claim this registry is checked against, so an entry that
    silently stops writing one of its plots is a defect the coverage check can
    name, not a gap the reader has to notice.
    """

    evidence: str
    figures: tuple[str, ...]
    render: Callable[[dict], None]


REGISTRY: tuple[Renderer, ...] = (
    Renderer("f1-baselines.json", ("f1-baselines.pdf",), f1),
    Renderer("f2-regret.json", ("f2-regret.pdf",), f2),
    Renderer("f4-realism.json", ("f4-realism.pdf",), f4),
    Renderer("f4-realism.json", ("f4-calm-calibration.pdf",), f4_calm_calibration),
    Renderer(
        "f5-manipulation.json",
        (
            "f5-boundaries.pdf",
            "f5-size-response.pdf",
            "f5-concave.pdf",
            "f5-positive-control.pdf",
            "f5-extended-sweeps.pdf",
        ),
        f5,
    ),
    Renderer("f6-adverse-selection.json", ("f6-markouts.pdf",), f6),
    Renderer("f6-adverse-selection.json", ("f6-endogenous.pdf",), f6_endogenous),
    Renderer("f7-failures.json", ("f7-failures.pdf",), f7),
    Renderer(
        "f8-ecology.json", ("f8-ecology-control.pdf", "f8-ecology-shocked.pdf"), f8
    ),
    Renderer("predictability.json", ("predictability.pdf",), predictability),
    Renderer("witness.json", ("witness.pdf",), witness),
)

REGISTERED_FIGURES: frozenset[str] = frozenset(
    name for entry in REGISTRY for name in entry.figures
)


def unregistered_figures() -> list[str]:
    """Committed PDFs that no registry entry claims to rebuild.

    Reported rather than raised: a figure this script cannot regenerate is a
    fact about the registry that the operator needs told, and the rebuild of
    everything else is still worth doing.
    """
    return sorted({p.name for p in FIGURES.glob("*.pdf")} - REGISTERED_FIGURES)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for entry in REGISTRY:
        data = _load(entry.evidence)
        if data is None:
            print(f"  not rendered: {', '.join(entry.figures)}")
            continue
        entry.render(data)
        print(f"wrote {', '.join(entry.figures)} from {entry.evidence}")
    missing = unregistered_figures()
    if missing:
        print(
            "omission: no registry entry rebuilds "
            + ", ".join(missing)
            + "; those files were left as they were"
        )
    else:
        print(f"registry covers every PDF in {FIGURES} ({len(REGISTERED_FIGURES)})")


if __name__ == "__main__":
    main()
