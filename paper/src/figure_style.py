"""Shared conventions for every paper figure.

Importing this module embeds fonts as TrueType (``pdf.fonttype`` 42), so no
figure PDF carries a Type 3 font. ``save_pdf`` writes no creation or
modification date, so re-rendering the same committed evidence yields
byte-identical PDFs. Colours are Okabe-Ito; a contrast a figure's claim rests
on also differs by marker, line style or hatch, so it survives greyscale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
OKABE_ITO = (BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, YELLOW, BLACK)

# One style per scenario tier, reused by every figure that compares tiers.
TIER_COLOR = {"calm": BLUE, "hard": ORANGE, "extreme": VERMILLION}
TIER_MARKER = {"calm": "o", "hard": "s", "extreme": "^"}
TIER_HATCH = {"calm": "", "hard": "///", "extreme": "..."}

PDF_METADATA = {"CreationDate": None, "ModDate": None}


def save_pdf(fig, path: Path) -> None:
    fig.savefig(path, metadata=PDF_METADATA)
