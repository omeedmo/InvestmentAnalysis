"""
Equinix: FFO as the company reports it.

Equinix labels the Nareit subtotal "FFO attributable to common stockholders",
earlier "NAREIT FFO attributable to common stockholders", and spells the last
word "shareholders" in some years — all three are matched.

Scale is the trap here rather than wording: Equinix reported in thousands
through the late 2010s ($992,363 for FY2017) and switched to millions ($2,668
for FY2025). Taking either at face value is out by a factor of a thousand,
which is why the shared extractor reads the declared unit and then checks the
table's net income against the XBRL figure before accepting anything.
"""
from __future__ import annotations

import re

from . import _reit_ffo

# The FFO row this plugin publishes is the company's own reported figure, so
# the app's quarterly NI+D&A proxy stands down rather than putting a derived
# quarter beside as-reported years.
REPORTED_FFO = True

TOTAL = re.compile(
    r"(?<!Core )(?<!Adjusted )(?:NAREIT )?"
    r"FFO attributable to common (?:stock|share)holders"
    r"\s*\$?\s*(?=\(?[\d,]{3,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    financials["_reported_ffo"] = _reit_ffo.walk_filings(
        filings, {**ctx, "net_income": financials.get("net_income")}, TOTAL,
        # Equinix elected REIT status effective 2015. A FY2014 line under the
        # same wording is a pre-conversion measure, and at $153M against $629M
        # the year after it would read as a 4x jump that never happened.
        from_year=2015)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))
