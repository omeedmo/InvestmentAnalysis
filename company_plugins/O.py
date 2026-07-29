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

TOTAL = re.compile(
    r"(?<!Normalized )(?<!Adjusted )(?<!Core )(?<!Diluted )(?<![A-Za-z])"
    r"FFO available to common stockholders\s*\$?\s*(?=\(?[\d,]{5,})", re.I)


# Realty Income also publishes AFFO, on the same wording one line down. The
# (?<![A-Za-z]) guard on TOTAL above is what keeps the FFO pattern from firing
# inside this one; without it "AFFO available to common stockholders" matches
# both, and only document order kept FFO on the right number.
AFFO_TOTAL = re.compile(
    r"(?<!Normalized )(?<!Core )"
    r"AFFO available to common stockholders\s*\$?\s*(?=\(?[\d,]{5,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    ffo = _reit_ffo.walk_filings(
        filings, {**ctx, "net_income": financials.get("net_income")}, TOTAL)
    financials["_reported_ffo"] = ffo
    # AFFO is checked against the FFO the reconciliation starts from, which is
    # a tighter anchor than net income: the two should agree to the dollar.
    financials["_reported_affo"] = _reit_ffo.walk_filings(
        filings, ctx, AFFO_TOTAL, anchor_re=TOTAL, anchor_series=ffo)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))
    _reit_ffo.publish_affo(financials, financials.pop("_reported_affo", {}))


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    _reit_ffo.quarterly(financials, quarter_end_dates, quarter_filing_links,
                        ctx, TOTAL)
    _reit_ffo.quarterly_affo(financials, quarter_end_dates,
                             quarter_filing_links, ctx, AFFO_TOTAL, TOTAL)
