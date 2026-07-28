"""
AvalonBay: FFO as the company reports it.

AvalonBay's wording moved over the history — "FFO attributable to common
stockholders" through the late 2010s, a bare "FFO" line in the most recent
filings — so both forms are matched. "Core FFO", its own adjusted measure,
is deliberately excluded: it sits directly beneath the Nareit line in the same
table and is a different definition.
"""
from __future__ import annotations

import re

from . import _reit_ffo

# The FFO row this plugin publishes is the company's own reported figure, so
# the app's quarterly NI+D&A proxy stands down rather than putting a derived
# quarter beside as-reported years.
REPORTED_FFO = True

TOTAL = re.compile(
    r"(?<!Core )(?<!Adjusted )(?<!Normalized )"
    r"FFO(?: attributable to common stockholders)?"
    r"\s*\$?\s*(?=\(?[\d,]{5,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    financials["_reported_ffo"] = _reit_ffo.walk_filings(
        filings, {**ctx, "net_income": financials.get("net_income")}, TOTAL)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    _reit_ffo.quarterly(financials, quarter_end_dates, quarter_filing_links,
                        ctx, TOTAL)
