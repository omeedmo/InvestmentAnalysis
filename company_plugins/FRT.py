"""
Federal Realty: FFO as the company reports it.

Federal Realty's MD&A reconciles net income to "Funds from operations", the
Nareit measure, and then onward to two derived lines: "Funds from operations
available for common shareholders" (after preferred dividends and amounts
attributable to downREIT units and unvested shares) and "Core FFO" (its own
further adjustments). The Nareit subtotal is the one taken, so the row means
the same thing here as it does for the other REITs with a plugin.

FY2025 reconciles exactly: 423,648 - 12,571 - 150,111 + 7,425 + 320,311
+ 42,671 = 631,373, the printed total.
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
    r"Funds from operations(?! available)(?! per)(?! attributable)"
    r"\s*\$?\s*(?=\(?[\d,]{5,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    financials["_reported_ffo"] = _reit_ffo.walk_filings(
        filings, {**ctx, "net_income": financials.get("net_income")}, TOTAL,
        # Federal Realty's table opens on net income AFTER preferred dividends
        # and noncontrolling interests -- $304M for FY2024 where XBRL carries
        # $411M. That 26% gap is correct in both places but sits just outside
        # the default gate, which cost the most recent year. Widened only as
        # far as this filer needs; at half the series picks up matches its own
        # filings contradict.
        tolerance=0.35)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    _reit_ffo.quarterly(financials, quarter_end_dates, quarter_filing_links,
                        ctx, TOTAL, tolerance=0.35)
