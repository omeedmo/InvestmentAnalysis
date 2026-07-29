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

# Equinix names this line three ways across its history: "NAREIT FFO
# attributable to common stockholders", "FFO attributable to common
# shareholders", and — through FY2018-FY2021 — a bare "FFO". All three are
# matched. The (?<![A-Za-z]) guard is what keeps the bare form from also
# firing inside "AFFO", which sits in the same table carrying a much larger
# number; "FFO from unconsolidated joint ventures" is excluded already, since
# a word follows rather than a figure.
TOTAL = re.compile(
    r"(?<!Core )(?<!Adjusted )(?:NAREIT )?(?<![A-Za-z])"
    r"FFO(?: attributable to common (?:stock|share)holders)?"
    r"\s*\$?\s*(?=\(?[\d,]{3,})", re.I)


# Equinix publishes AFFO alongside FFO. Its FFO pattern already carries the
# (?<![A-Za-z]) guard, so the bare "FFO" form cannot fire inside "AFFO"; this
# pattern is the deliberate match for that line.
AFFO_TOTAL = re.compile(
    r"(?<!Core )(?<!Normalized )"
    r"AFFO(?: attributable to common (?:stock|share)holders)?"
    r"\s*\$?\s*(?=\(?[\d,]{3,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    financials["_reported_ffo"] = _reit_ffo.walk_filings(
        filings, {**ctx, "net_income": financials.get("net_income")}, TOTAL,
        # Equinix elected REIT status effective 2015. A FY2014 line under the
        # same wording is a pre-conversion measure, and at $153M against $629M
        # the year after it would read as a 4x jump that never happened.
        from_year=2015,
        # The FY2018-FY2021 tables print three fiscal years at once.
        ncols=3)
    financials["_reported_affo"] = _reit_ffo.walk_filings(
        filings, ctx, AFFO_TOTAL, anchor_re=TOTAL,
        anchor_series=financials["_reported_ffo"], from_year=2015, ncols=3)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))
    _reit_ffo.publish_affo(financials, financials.pop("_reported_affo", {}))


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    _reit_ffo.quarterly(financials, quarter_end_dates, quarter_filing_links,
                        ctx, TOTAL)
    _reit_ffo.quarterly_affo(financials, quarter_end_dates,
                             quarter_filing_links, ctx, AFFO_TOTAL, TOTAL)
