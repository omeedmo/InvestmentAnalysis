"""
GoDaddy: the share counts its XBRL leaves blank, and the per-share rows built
on them.

Two series are missing and they are missing over different spans. Shares
outstanding is absent FY2019-FY2023; the diluted weighted average, which every
per-share row divides by, is absent FY2017-FY2022. Neither is a wrong tag
being tried -- for those years companyfacts holds no share-count concept at
all in the shares unit, and the frames API agrees: GoDaddy appears at
CY2018Q4I with 174,803,000 and CY2024Q4I with 141,208,000 and is absent from
every quarter between, while five thousand other filers are present. It
carried two share classes over that stretch and presented both the counts and
the per-share figures per class, so the facts are dimensional and companyfacts
publishes only the undimensioned context.

Both live in the rendered statements, and they come from different places for
a reason.

  outstanding      balance-sheet parenthetical, one row per class, SUMMED.
                   A binding cannot reach it twice over: the fact base skips
                   parentheticals by design, and both class rows share one
                   element with no dimension, so it would keep the first and
                   drop the other regardless.

  diluted average  income statement, a single row labelled "Weighted-average
                   shares of Class A common stock outstanding". Class A ALONE,
                   not a sum. GoDaddy is an Up-C: Class B carries votes
                   against LLC units and does not share in the earnings that
                   "net income attributable to GoDaddy Inc." measures, so the
                   company divides by Class A and so does this. Dividing by
                   the summed count instead would be about 5% out and would
                   not reproduce the company's own printed EPS.

Every year is verified before it is believed. A filing is accepted only where
some year it reports is already established and agrees, so belief spreads
outward from the years companyfacts does hold rather than resting on a single
parse. That has to run to a fixed point rather than in one pass: the top of
each range anchors on the recent years and walks back, the bottom on the old
ones and walks forward, and the FY2021 10-K's parenthetical does not parse,
leaving a gap between them.

Summing classes is specific to this filer and deliberately not generalised.
It is right for the outstanding count because GoDaddy's own FY2018 total,
174,803,000, is exactly its 168,549,000 Class A plus 6,254,000 Class B. A
filer whose classes differ in economics -- Berkshire's A against its B -- has
to be converted, not added, which is the same reason the diluted average here
is Class A alone.
"""
from __future__ import annotations

import statement_store

CIK = "1609711"

# A parsed year is accepted only if every year in the same filing that is
# already known agrees within this. Loose enough for a restatement, far tighter
# than picking up the wrong row.
_TOL = 0.005


def _outstanding(accession: str) -> dict:
    """{column index: total shares outstanding} from the BS parenthetical."""
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


def _diluted(accession: str) -> dict:
    """{column index: diluted weighted-average Class A shares} from the P&L."""
    st = statement_store.filing_statements(CIK, accession)
    io = next((v for k, v in st.items()
               if "operations" in k.lower() and "parenthetical" not in k.lower()),
              None)
    if not io:
        return {}
    row = next((r for r in io["rows"]
                if (r.get("element") or "")
                .endswith("WeightedAverageNumberOfDilutedSharesOutstanding")), None)
    if not row:
        return {}
    return {i: v for i, v in enumerate(row.get("by_col") or []) if v}


def _parse(filings: list, ctx: dict, reader) -> list:
    out = []
    for f in filings:
        fy = f.get("fiscal_year")
        if not fy or int(fy) < ctx.get("min_year", 0):
            continue
        try:
            cols = reader(f["accession"])
        except Exception:            # noqa: BLE001 - an unreadable filing is a gap
            continue
        if cols:
            out.append((int(fy), cols))
    out.sort(key=lambda p: -p[0])     # newest first, so newest wins on overlap
    return out


def _resolve(parsed: list, known: dict) -> dict:
    """Years a filing established, each confirmed against one already known."""
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
                out.setdefault(str(fy - i), v)
                resolved.setdefault(str(fy - i), v)
            changed = True
    return out


def _by_year(series: dict) -> dict:
    return {str(k)[:4]: v for k, v in (series or {}).items()
            if v and not str(k).startswith("Q")}


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    wtd = _by_year(financials.get("shares_diluted_wtd"))

    # Anchors must be real share counts. The outstanding row already holds one
    # that is not: where no count existed the app fell back to that year's
    # diluted weighted average, which for FY2023 put 151,452,000 into a
    # period-end column against an actual 142,310,000. Left in, it is a false
    # anchor that rejects every filing reporting FY2023 correctly and severs
    # the top of the range from its FY2024 anchor. A count equal to the
    # weighted average to the share is that substitution, not a coincidence.
    known_out = {y: v for y, v in _by_year(financials.get("shares_outstanding_end")).items()
                 if v != wtd.get(y)}

    # Published straight away rather than in postprocess: the per-share rows are
    # rebuilt mid-run, before postprocess is called, and nothing reassigns this
    # series after this hook. shares_outstanding_end is the opposite case -- it
    # IS rebuilt wholesale later, so writing it here would be discarded.
    diluted = _resolve(_parse(filings, ctx, _diluted), wtd)
    if diluted:
        series = financials.setdefault("shares_diluted_wtd", {})
        key_of = {str(d)[:4]: d for d in series if not str(d).startswith("Q")}
        for year, value in diluted.items():
            series[key_of.get(year, f"{year}-12-31")] = value

    outstanding = _resolve(_parse(filings, ctx, _outstanding), known_out)
    if outstanding:
        financials["_gddy_shares"] = outstanding


def postprocess(financials: dict) -> None:
    shares = financials.pop("_gddy_shares", {})
    if not shares:
        return
    series = financials.setdefault("shares_outstanding_end", {})
    key_of = {str(d)[:4]: d for d in series if not str(d).startswith("Q")}
    for year, value in shares.items():
        # Written for every verified year, not only the blank ones, so the row
        # carries one definition end to end. FY2023 in particular held a
        # weighted-average diluted figure filled in as if it were a period-end
        # count.
        series[key_of.get(year, f"{year}-12-31")] = value

    # Book value per share divides by the period-end count, and the app's own
    # recompute of it has already run by the time this hook is called, so the
    # years just filled would stay blank without redoing it here.
    eq = _by_year(financials.get("equity"))
    so = _by_year(series)
    if eq and so:
        bvps = financials.setdefault("book_value_per_share", {})
        for d in list(financials.get("equity", {})):
            if str(d).startswith("Q"):
                continue
            y = str(d)[:4]
            if eq.get(y) and so.get(y):
                bvps[d] = eq[y] / so[y]
