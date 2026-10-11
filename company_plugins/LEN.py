"""
Lennar: the consolidated debt total for the periods whose balance sheet
prints only the Homebuilding line.

Lennar's whole borrowing is one balance-sheet line, "Senior notes and other
debts payable, net". For FY2019, FY2020, FY2024 and FY2025 the filing tags a
consolidated figure for it and the ordinary paths resolve it. For FY2021
through FY2023 -- and for Q3 FY2026 -- no consolidated figure is tagged
anywhere on the balance sheet: the face prints the Homebuilding segment's
line, and Financial Services' borrowing (the larger half of the gap years'
debt) rides inside a single lump-sum liabilities line. Those periods came
back blank, which reads as missing data. It is not missing; it is filed one
statement over.

Every one of those filings reports the total in its operating-segments
schedule -- "Schedule of Segment Assets and Liabilities" in the 10-Ks,
"Disclosure Of Assets and Liabilities" in the 10-Q -- broken out by segment
under us-gaap:LongTermDebt (the 10-Q uses Lennar's own extension element,
len:LongTermDebtSegmentAllocation, for the same line). build_factbase
never sees it: statement_store takes FilingSummary's "Statements" category
alone, and the schedule is a note. A binding therefore cannot reach it,
which is why this is a plugin rather than a third candidate in
bindings/companies/LEN.json -- the same division of labour as GDDY's
parentheticals, which the fact base also skips by design.

A period is filled only where the schedule corroborates itself, never on
a single unverified read:

  * where the schedule prints both a total row and the segment rows, the
    total must equal the sum of the segments (Q3 FY2026: Homebuilding
    $4,297,251k + Financial Services $1,584,448k = the printed total
    $5,881,699k, exactly); or
  * the same period must come back with the same figure from a second
    filing's schedule. Lennar's schedules carry the comparative year, so
    FY2021-FY2023 are each reported twice and agree to the dollar:
    FY2021 $6,378,364k, FY2022 $6,199,136k, FY2023 $4,984,028k.

The chain also anchors on the years that are already established, and the
plugin checks rather than assumes: the FY2021 10-K's schedule reports
FY2020 as $7,421,583k, the FY2024 and FY2025 schedules report their years
as $4,189,239k and $5,874,995k -- each exactly the figure the app already
carries. Only blanks are filled; an established value is never overwritten.

Nothing before FY2021 is attempted. The older filings' schedules do not
break the line out in this form, and their balance-sheet figure is the
Homebuilding segment alone (FY2018's $8,543.9M is Homebuilding, not the
company), so those years stay blank rather than wearing a segment's
number as the total.

The pass is deliberately narrow, for speed as much as scope: only filings
from the FY2021 10-K onward are consulted, and only while a year they
could report is actually blank. An analysis runs inside a 30-second
action budget in the deployed app, and every schedule read is an SEC
fetch through statement_store's rate limiter -- reading eighteen
filings' summaries to fill three years would spend most of that budget
on sleeps.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime

import statement_store as ss

CIK = "920760"

_ELEMENTS = ("us-gaap:LongTermDebt", "len:LongTermDebtSegmentAllocation")
_TOTAL_DIM = "Operating Segments"
_TOL = 0.001          # agreement between two reads of the same period
_ANCHOR_TOL = 0.005   # agreement with a value the app already carries
_CACHE_V = 1


def _period_iso(label: str):
    try:
        return datetime.strptime(label.strip(), "%b. %d, %Y").strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def _segment_of(dim):
    """Segment name for a dimensional row, None for the total row, or
    False for a sub-cut that is not a segment total (a VIE slice of a
    segment, say) and must not enter the sum."""
    if dim is None or dim == _TOTAL_DIM:
        return None
    parts = [p for p in dim.split(" | ") if p != _TOTAL_DIM]
    return parts[0] if len(parts) == 1 else False


def _schedule_totals(accession: str) -> dict:
    """{period_iso: (total, self_verified)} from one filing's segment
    schedule. self_verified means the filing's own segment rows sum to
    its total row. Extracted schedules are cached on disk; filings do
    not change."""
    acc = accession.replace("-", "")
    cache_path = os.path.join(ss.CACHE_DIR, f"len_sched_v{_CACHE_V}_{acc}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return {k: tuple(v) for k, v in json.load(f).items()}
        except Exception:  # noqa: BLE001 - a bad cache entry is just a miss
            pass

    base = ss.SEC_ARCHIVE.format(cik=CIK, accession=acc)
    try:
        summary = ss._get(f"{base}/FilingSummary.xml")
    except Exception:  # noqa: BLE001 - no schedule is a blank, not an error
        return {}
    totals: dict = {}
    for rep in re.findall(r"<Report[^>]*>(.*?)</Report>", summary, re.S):
        sn = re.search(r"<ShortName>(.*?)</ShortName>", rep)
        fn = re.search(r"<HtmlFileName>(.*?)</HtmlFileName>", rep)
        if not sn or not fn:
            continue
        name = sn.group(1)
        if "Segment" not in name or "Assets and Liabilities" not in name:
            continue
        try:
            st = ss.parse_report(ss._get(f"{base}/{fn.group(1)}"))
        except Exception:  # noqa: BLE001
            continue
        if not st:
            continue
        total_rows: dict = {}
        segments: dict = {}
        for r in st.get("rows", []):
            if r.get("element") not in _ELEMENTS:
                continue
            seg = _segment_of(r.get("dimension"))
            if seg is False:
                continue
            for label, value in (r.get("values") or {}).items():
                iso = _period_iso(label)
                if iso is None or value is None:
                    continue
                if seg is None:
                    total_rows[iso] = value
                else:
                    prev = segments.setdefault(iso, {})
                    if seg in prev and abs(prev[seg] - value) > _TOL * max(abs(value), 1.0):
                        prev[seg] = None  # duplicate rows disagree: unusable
                    else:
                        prev[seg] = value
        for iso, total in total_rows.items():
            segs = segments.get(iso) or {}
            seg_vals = [v for v in segs.values() if v is not None]
            verified = bool(seg_vals) and len(seg_vals) == len(segs) and \
                abs(sum(seg_vals) - total) <= _TOL * max(abs(total), 1.0)
            totals[iso] = (total, verified)
    try:
        os.makedirs(ss.CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(totals, f)
    except Exception:  # noqa: BLE001 - caching is an optimisation only
        pass
    return totals


def _trusted(cands: list) -> float | None:
    """One period's candidate reads [(value, self_verified), ...] across
    filings -> the value to believe, or None. A read is believed when its
    own filing ties out, or when a second filing reports the same figure;
    believed reads must agree with each other."""
    believed = []
    for value, verified in cands:
        if verified or sum(1 for v, _ in cands
                            if abs(v - value) <= _TOL * max(abs(value), 1.0)) >= 2:
            believed.append(value)
    if not believed:
        return None
    if max(believed) - min(believed) > _TOL * max(abs(believed[0]), 1.0):
        return None
    return believed[0]


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    series = financials.get("long_term_debt")
    if not series:
        return
    known = {k: v for k, v in series.items()
             if v is not None and not str(k).startswith("Q")}
    # Nothing to do -- and therefore nothing to fetch -- unless a year the
    # modern schedules could report is blank. Lennar's fiscal years end
    # Nov 30, so a fillable blank is a missing year from 2021 on.
    known_years = {str(k)[:4] for k in known}
    latest_fy = max((int(y) for y in known_years if y.isdigit()), default=0)
    if all(str(y) in known_years for y in range(2021, latest_fy + 1)):
        return
    cands: dict = {}
    for f in filings:
        acc = f.get("accession")
        # The FY2021 10-K (filed 2022-01-28) is the earliest filing whose
        # schedule breaks the line out in this form; older ones do not,
        # and fetching their summaries is pure cost.
        if not acc or (f.get("filing_date") or "") < "2022-01-01":
            continue
        for iso, cand in _schedule_totals(acc).items():
            cands.setdefault(iso, []).append(cand)
    key_of = {str(k)[:4]: k for k in series if not str(k).startswith("Q")}
    for iso, reads in cands.items():
        if iso in known:
            continue  # established already; the plugin never overwrites
        value = _trusted(reads)
        if value is None:
            continue
        series[key_of.get(iso[:4], iso)] = value


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    series = financials.get("long_term_debt")
    if not series:
        return
    annual = {k: v for k, v in series.items()
              if v is not None and not str(k).startswith("Q")}
    for qk, qdate in (quarter_end_dates or {}).items():
        if series.get(qk) is not None:
            continue
        url = (quarter_filing_links or {}).get(qk) or ""
        m = re.search(r"/data/\d+/(\d{18})/", url)
        if not m:
            continue
        totals = _schedule_totals(m.group(1))
        cand = totals.get(qdate)
        if cand is None:
            continue
        # Anchor: every period this filing reports that the app already
        # carries (its comparative column is the last fiscal year end)
        # must agree with the carried value, or the filing is not believed
        # for the new quarter either. A total row that ties to its own
        # segment rows needs no anchor; a lone total row is believed only
        # through one.
        anchored = False
        distrust = False
        for iso, (v, _) in totals.items():
            if iso in annual:
                anchored = True
                if abs(v - annual[iso]) > _ANCHOR_TOL * max(abs(annual[iso]), 1.0):
                    distrust = True
        if distrust:
            continue
        if cand[1] or anchored:
            series[qk] = cand[0]
