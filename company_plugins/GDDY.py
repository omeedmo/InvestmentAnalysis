"""
GoDaddy: shares outstanding for the years its XBRL leaves blank.

FY2019 through FY2023 have no share count anywhere in companyfacts. Not the
wrong tag being tried -- there is no share-count concept at all in the shares
unit for those years, and the frames API agrees: GoDaddy is present at
CY2018Q4I with 174,803,000 and at CY2024Q4I with 141,208,000, and absent from
every quarter between while five thousand other filers are there. It carried
two share classes over that stretch and presented the counts per class, so
every fact is dimensional, and companyfacts publishes only the undimensioned
context. The same is true of its EPS denominators, which is why the weighted
average is blank across the same span.

The rendered balance-sheet parenthetical does carry them, one row per class.
Two obstacles rule out a binding, which is why this is a plugin: the fact base
skips parentheticals by design, and both class rows share one element with no
dimension, so it would keep the first and drop the other in any case.

Summing the classes is a judgement, and it is the one this filer's own numbers
support. The FY2019 10-K reports its prior year as 168,549,000 Class A plus
6,254,000 Class B, and companyfacts carries FY2018 as 174,803,000 -- the sum
to the share. That is not a rule to generalise: a filer whose classes have
different economics (Berkshire's A against its B) must be converted, not
added, which is exactly why this lives in a per-company plugin.
"""
from __future__ import annotations

from typing import Optional

import statement_store

CIK = "1609711"

# A parsed year is accepted only if every year in the same filing that is
# already known agrees within this. Loose enough for a restatement, far tighter
# than picking up the wrong row.
_TOL = 0.005


def _outstanding(accession: str) -> dict:
    """{column index: total shares outstanding} for one 10-K, or {}."""
    st = statement_store.filing_statements(CIK, accession)
    par = next((v for k, v in st.items()
                if "balance sheet" in k.lower() and "parenthetical" in k.lower()),
               None)
    if not par:
        return {}
    rows = [r.get("by_col") or [] for r in par["rows"]
            if (r.get("element") or "").endswith("CommonStockSharesOutstanding")]
    out: dict = {}
    for i in range(len(par.get("periods") or [])):
        vals = [r[i] for r in rows if i < len(r) and r[i]]
        if not vals:
            continue
        # Some filings print the combined total BESIDE the two classes rather
        # than instead of them -- the FY2018 10-K lists 174,803,000 and then
        # 168,549,000 and 6,254,000, its Class A and Class B. A plain sum
        # doubles that year. Where one value is the sum of the rest, it is the
        # total and the others are its components.
        total = next((v for v in vals
                      if abs(v - (sum(vals) - v)) <= max(1.0, abs(v) * 0.001)), None)
        out[i] = total if total is not None else sum(vals)
    return out


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    # Anchors must be real share counts. The row already contains one value
    # that is not: where no count existed, the app fell back to that year's
    # diluted weighted average, which for FY2023 put 151,452,000 in a
    # period-end column against an actual 142,310,000. Left in, it is a false
    # anchor that rejects every filing reporting FY2023 correctly and severs
    # the top of the range from its FY2024 anchor. A year whose count equals
    # the weighted average to the share is that substitution, not a coincidence.
    wtd = {str(k)[:4]: v
           for k, v in (financials.get("shares_diluted_wtd") or {}).items()
           if v and not str(k).startswith("Q")}
    known = {str(k)[:4]: v
             for k, v in (financials.get("shares_outstanding_end") or {}).items()
             if v and not str(k).startswith("Q")
             and v != wtd.get(str(k)[:4])}

    parsed = []
    for f in filings:
        fy = f.get("fiscal_year")
        if not fy or int(fy) < ctx.get("min_year", 0):
            continue
        try:
            cols = _outstanding(f["accession"])
        except Exception:            # noqa: BLE001 - an unreadable filing is a gap
            continue
        if cols:
            parsed.append((int(fy), cols))
    parsed.sort(key=lambda p: -p[0])          # newest first, so newest wins

    # Each filing has to agree with something already established before the
    # year only it covers is believed. The anchors are the two years
    # companyfacts does hold, and acceptance spreads outward from them through
    # the overlapping columns -- FY2018 confirms the FY2019 filing, which
    # establishes 2019, which in turn confirms the FY2020 filing. Repeated to a
    # fixed point because that spread runs in both directions and no single
    # ordering reaches every filing: the top of the range anchors on FY2024 and
    # walks back, the bottom on FY2018 and walks forward, and the FY2021 10-K
    # (whose parenthetical does not parse) leaves a gap between them.
    out: dict = {}
    resolved = dict(known)
    accepted: set = set()
    changed = True
    while changed:
        changed = False
        for fy, cols in parsed:
            if fy in accepted:
                continue
            checks = [(resolved[str(fy - i)], v) for i, v in cols.items()
                      if str(fy - i) in resolved]
            if not checks:
                continue                      # nothing to verify against yet
            if any(abs(v - k) > max(1.0, abs(k) * _TOL) for k, v in checks):
                continue                      # wrong rows, or a real conflict
            accepted.add(fy)
            for i, v in cols.items():
                # setdefault, and `parsed` is newest-first, so where two
                # filings report the same year the newer one has already
                # claimed it.
                out.setdefault(str(fy - i), v)
                resolved.setdefault(str(fy - i), v)
            changed = True

    if out:
        financials["_gddy_shares"] = out


def postprocess(financials: dict) -> None:
    shares = financials.pop("_gddy_shares", {})
    if not shares:
        return
    series = financials.setdefault("shares_outstanding_end", {})
    key_of = {str(d)[:4]: d for d in series if not str(d).startswith("Q")}
    for year, value in shares.items():
        # Written for every verified year, not only the blank ones, so the row
        # carries one definition end to end. FY2023 in particular held a
        # weighted-average diluted figure that had been filled in as if it were
        # a period-end count.
        series[key_of.get(year, year)] = value
