"""
Brandywine Realty Trust: FFO and NOI as the company reports them.

Two separate reconciliations in the 10-K, each verified the same way — the
table's own net-income row has to agree with the net income already held from
XBRL for that year, which confirms the right table, the right scale and the
right column alignment together.

FFO, from MD&A. The table runs from "Net loss attributable to common
unitholders" through the impairments, gains and depreciation add-backs to a
Nareit subtotal, and then one line further to a figure after amounts allocable
to unvested restricted shareholders. The Nareit subtotal is the one taken, so
the row means here what it means for Federal Realty and Simon. FY2025
reconciles exactly: -180,015 + 1,231 + 227 - 9,396 + 63,392 + 4,149 + 154,009
+ 19,130 + 41,959 - 88 = 94,598, the printed total. FY2024 likewise:
-197,670 + 1,178 - 63,696 - 2,297 + 44,101 + 147,184 + 154,945 + 19,746
+ 47,013 - 9 = 150,495.

NOI, from Note 18 (Segment Information), where Brandywine reconciles
consolidated net loss up to "Consolidated net operating income" — total
property revenue less property operating expenses, real estate taxes and
third-party management expenses. Three fiscal years to a column rather than
two. The figures agree with the separate comparative table in MD&A, which
prints the same $299.2M and $318.2M rounded to millions.

AFFO is deliberately absent. Brandywine does not publish one: the string
appears nowhere in the 10-K, in any spelling, and what it reports instead is
"cash available for distribution", a different measure with different
adjustments. Deriving something and calling it AFFO is exactly what was
removed from this app, so the row stays empty.

The anchor is the Operating Partnership's net loss attributable to common
unitholders rather than the Trust's -- Brandywine files as a dual registrant
and the FFO table is presented on the partnership basis. The two differ by
about 0.3% (FY2025: -180,015 against XBRL's -179,478), comfortably inside the
default gate, so no widened tolerance is needed here.
"""
from __future__ import annotations

import re

from . import _reit_ffo

# The Nareit subtotal, not the line below it. "Funds from operations allocable
# to unvested restricted shareholders" and "Funds from operations available to
# common share and unit holders (FFO)" both open with the same three words, so
# both are excluded explicitly; without the `allocable` guard the -1,212
# adjustment line matches first and is read as the year's FFO.
TOTAL = re.compile(
    r"(?<!Core )(?<!Adjusted )(?<!Normalized )"
    r"Funds from operations(?! available)(?! per)(?! allocable)(?! attributable)"
    r"\s*\$?\s*(?=\(?[\d,]{5,})", re.I)

# Note 18's subtotal. "Consolidated" is part of the printed label and is what
# separates it from the two dozen other places the 10-K writes "net operating
# income" in prose, and from the per-segment NOI columns above it.
NOI_TOTAL = re.compile(
    r"Consolidated net operating income\s*\$?\s*(?=\(?[\d,]{5,})", re.I)


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    c = {**ctx, "net_income": financials.get("net_income")}
    financials["_reported_ffo"] = _reit_ffo.walk_filings(filings, c, TOTAL)
    # Three columns: Note 18 carries a third fiscal year that MD&A's FFO table
    # does not, so each filing contributes one more year of NOI than of FFO.
    financials["_reported_noi"] = _reit_ffo.walk_filings(filings, c, NOI_TOTAL,
                                                        ncols=3)


def postprocess(financials: dict) -> None:
    _reit_ffo.publish(financials, financials.pop("_reported_ffo", {}))

    noi_by_year = financials.pop("_reported_noi", {})
    if not noi_by_year:
        return
    # Same date-key convention the FFO publisher uses, so the NOI row lines up
    # with every other annual row rather than sitting on bare year strings.
    key_of = {d[:4]: d for d in financials.get("net_income", {})
              if not str(d).startswith("Q")}
    noi = {key_of.get(y, y): v for y, v in noi_by_year.items()}
    financials["noi"] = noi

    def by_year(series):
        return {str(k)[:4]: v for k, v in (series or {}).items()
                if v is not None and not str(k).startswith("Q")}

    noi_y, rev_y = by_year(noi), by_year(financials.get("revenue"))
    if rev_y:
        financials["noi_margin"] = {
            key_of.get(y, y): noi_y[y] / rev_y[y]
            for y in noi_y if rev_y.get(y)}


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    # FFO only. Brandywine's 10-Qs carry the FFO reconciliation in MD&A, but
    # the NOI reconciliation lives in the segment note, whose quarterly columns
    # the same leading-column rule does not describe -- left for a separate
    # pass rather than guessed at.
    _reit_ffo.quarterly(financials, quarter_end_dates, quarter_filing_links,
                        ctx, TOTAL)
