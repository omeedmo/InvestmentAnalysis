"""
Realty Income: FFO as the company reports it.

Realty Income prints a Nareit FFO reconciliation in every 10-K MD&A, closing on
"FFO available to common stockholders". That figure is not XBRL-tagged
anywhere, so the app's generic row was net income plus D&A — a proxy that for
FY2025 gives a materially different answer from the $3.860B Realty Income
actually reported.

The subtotal has three near-identical neighbours in the same table: "Diluted
FFO" (adds back FFO allocable to dilutive noncontrolling interests),
"Normalized FFO" (Realty Income's own further adjustments) and "Normalized FFO
available to common stockholders". Only the plain Nareit line is taken, since
mixing the variants across years would change what the row means partway down
the series.
"""
from __future__ import annotations

import re

from . import _reit_ffo

# The FFO row this plugin publishes is the company's own reported figure, so
# the app's quarterly NI+D&A proxy stands down rather than putting a derived
# quarter beside as-reported years.
REPORTED_FFO = True

TOTAL = re.compile(
    r"(?<!Normalized )(?<!Adjusted )(?<!Core )(?<!Diluted )"
    r"FFO available to common stockholders\s*\$?\s*(?=\(?[\d,]{5,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    financials["_reported_ffo"] = _reit_ffo.walk_filings(
        filings, {**ctx, "net_income": financials.get("net_income")}, TOTAL)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    _reit_ffo.quarterly(financials, quarter_end_dates, quarter_filing_links,
                        ctx, TOTAL)
