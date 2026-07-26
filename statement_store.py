"""
Statements as the company actually filed them.

The rest of this app reads SEC's `companyfacts` API, which is a convenient but
LOSSY view of a filing. Three things it drops, all of which matter:

  1. Company extension elements. It carries only dei/srt/us-gaap, so a REIT's
     own `spg:NetOperatingIncome` simply does not exist there.
  2. Dimensional facts. Anything a company reports only inside a breakdown —
     by segment, by class of stock, by subsidiary — is absent. Berkshire's
     income statement line "Insurance premiums earned" ($88.9B in FY2025,
     us-gaap:PremiumsEarnedNet) is missing from its companyfacts entirely for
     exactly this reason.
  3. The statement itself. companyfacts is a bag of concepts with no order, no
     subtotal structure, and only the standard taxonomy label — never the
     wording the company printed.

The filing's own rendered statements have all of it. Every EDGAR filing ships a
FilingSummary.xml listing its reports, and each statement report is an R*.htm
rendering with the company's line order, the company's labels, the element
behind each line, and the values.

Reading whole statements rather than scattered concepts is also what makes
verification possible: the company prints its own subtotals, so a parse can be
checked against them instead of trusted. See `check_totals`.

Filings are immutable, so everything here is cached on disk by accession.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".cache", "statements")

HEADERS = {
    "User-Agent": "InvestmentAnalysis research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Scale stated in the report header ("$ in Millions").
_SCALES = {"thousands": 1_000.0, "millions": 1_000_000.0, "billions": 1_000_000_000.0}

# Per-share amounts are printed unscaled even when the statement is in
# millions, so they must never take the header multiplier.
_PER_SHARE = re.compile(r"PerShare|PerBasicShare|PerDilutedShare|PerUnit", re.I)
_SHARE_COUNT = re.compile(
    r"SharesOutstanding|SharesIssued|WeightedAverageNumberOf|SharesAuthorized", re.I)


# Recognises a financial statement by the name the filer gave it, for filings
# that predate FilingSummary's MenuCategory.
_STATEMENT_NAME = re.compile(
    r"^consolidated\s+(balance sheets?|statements? of)", re.I)

# The three statements that are period-over-period and must carry period
# columns; the equity roll-forward is laid out by component instead.
_CORE_STATEMENT = re.compile(
    r"balance sheets?|financial position"
    r"|statements? of (earnings|operations|income)(?!.*comprehensive)"
    r"|cash flows?", re.I)


def _get(url: str, retries: int = 3) -> str:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:      # noqa: BLE001 - retried below
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url} ({last})")


# ── Parsing one rendered statement ───────────────────────────────────────────

def _clean(html: str) -> str:
    txt = re.sub(r"<[^>]+>", "", html)
    txt = txt.replace("&#160;", " ").replace("&nbsp;", " ")
    txt = txt.replace("&#8217;", "'").replace("&#8216;", "'")
    txt = txt.replace("&#8220;", '"').replace("&#8221;", '"')
    txt = txt.replace("&amp;", "&").replace("&#38;", "&")
    # Hex entities (&#xA0;) as well as decimal — missing the hex form left raw
    # markup inside labels and dimension names.
    txt = re.sub(r"&#x[0-9a-fA-F]+;|&#\d+;|&[a-zA-Z]+;", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _parse_value(cell_html: str, cell_class: str) -> Optional[float]:
    """A rendered cell to a number, or None when the company printed nothing."""
    txt = _clean(cell_html)
    if not txt or txt in {"$", "%"}:
        return None
    negative = txt.startswith("(") or cell_class == "num"
    digits = re.sub(r"[^\d.]", "", txt)
    if not digits or digits == ".":
        return None
    try:
        val = float(digits)
    except ValueError:
        return None
    return -val if negative else val


def parse_report(html: str) -> Optional[dict]:
    """
    One R*.htm rendering to {name, scale, periods, rows}.

    Each row keeps the element it came from and the label the company printed
    for it, which is what lets a metric binding say "this line, in these years"
    and what lets the UI show the filer's own wording.
    """
    table = re.search(r'<table class="report".*?</table>', html, re.S)
    if not table:
        return None
    body = table.group(0)

    title_m = re.search(r'<th class="tl"[^>]*>(.*?)</th>', body, re.S)
    title_txt = _clean(title_m.group(1)) if title_m else ""
    # Header wording changed over the years: newer filings print
    # "$ in Millions", filings before ~2015 print "(USD $)In Millions, unless
    # otherwise specified". Missing this made every value 1,000,000x too small,
    # which the companyfacts cross-check is what surfaced.
    share_scale = 1.0
    share_m = re.search(r"shares? (?:data )?in (thousands|millions|billions)",
                        title_txt, re.I)
    if share_m:
        share_scale = _SCALES[share_m.group(1).lower()]
    money_txt = re.sub(r"shares? (?:data )?in (thousands|millions|billions)", " ",
                       title_txt, flags=re.I)
    scale = 1.0
    money_m = re.search(r"\bin (thousands|millions|billions)\b", money_txt, re.I)
    if money_m:
        scale = _SCALES[money_m.group(1).lower()]
    name = re.split(r"\s+-\s+USD|\s+-\s+\$", title_txt)[0].strip()

    # Period columns. The header also carries span labels like "12 Months
    # Ended" that are not periods, so keep only the date headers.
    periods = []
    for th in re.findall(r'<th class="th"[^>]*>(.*?)</th>', body, re.S):
        t = _clean(th)
        # Some filings annotate the column with its unit
        # ("Dec. 31, 2021 USD ($)"), so match the date and drop the rest.
        dm = re.match(r"^([A-Z][a-z]{2}\.? \d{1,2}, \d{4})\b", t)
        if dm:
            periods.append(dm.group(1))

    rows = []
    # A statement can stack a consolidated section and then repeat the same
    # elements under dimensional sections — Berkshire's balance sheet prints
    # consolidated Goodwill of $83.1B, then Goodwill again under "Insurance and
    # Other [Member]" ($55.9B) and "Railroad, Utilities and Energy [Member]"
    # ($27.1B). Those are three different facts sharing one element name, so
    # every row has to carry the section it sits under or they collide. This
    # dimensional detail is exactly what companyfacts drops.
    current_dim: Optional[str] = None
    for tr in re.findall(r'<tr class="r[a-z]*">(.*?)</tr>', body, re.S):
        header = re.match(r'\s*<th class="tl"', tr) or 'class="rh"' in tr
        label_m = re.search(
            r"defref_([A-Za-z0-9_.\-]+)'[^>]*>(.*?)</a>", tr, re.S)
        plain = _clean(tr)
        if plain.endswith("[Member]") or "[Member]" in plain and not label_m:
            current_dim = plain.replace("[Member]", "").strip()
            continue
        if not label_m:
            continue
        element = label_m.group(1).replace("_", ":", 1)
        raw_label = label_m.group(2)
        label = _clean(raw_label)
        abstract = element.endswith("Abstract") or "<strong>" in raw_label

        # A cash flow statement's opening-balance line is an instant fact dated
        # at the PRIOR period end, even though it is printed in the current
        # period's column. Left unmarked it silently shifts cash by a year.
        opening = bool(re.search(r"beginning of (the )?(year|period)", label, re.I))

        indent = 0
        pl = re.search(r'<td class="pl[^"]*"[^>]*style="[^"]*padding-left:\s*(\d+)px', tr)
        if pl:
            indent = int(pl.group(1)) // 10

        values: dict[str, float] = {}
        cells = re.findall(r'<td class="(nump|num|text)"[^>]*>(.*?)</td>', tr, re.S)
        for idx, (cls, cell) in enumerate(cells):
            if idx >= len(periods):
                break
            v = _parse_value(cell, cls)
            if v is None:
                continue
            if _PER_SHARE.search(element):
                pass                       # printed unscaled
            elif _SHARE_COUNT.search(element):
                v *= share_scale
            else:
                v *= scale
            values[periods[idx]] = v

        rows.append({
            "element":   element,
            "label":     label,
            "indent":    indent,
            "abstract":  abstract,
            # None = the consolidated figure; otherwise the section this line
            # was printed under (a segment, a class of stock, a subsidiary).
            "dimension": current_dim,
            # True when the printed column is one period ahead of the fact's
            # actual date (cash flow opening balances).
            "opening_balance": opening,
            "values":    values,
        })

    return {"name": name, "scale": scale, "periods": periods, "rows": rows}


# ── Fetching a filing's statements ───────────────────────────────────────────

def filing_statements(cik: str, accession: str,
                      use_cache: bool = True) -> dict:
    """
    {report_name: parsed statement} for one filing's financial statements.

    Cached on disk: a filed 10-K never changes, so this is fetch-once.
    """
    cik = str(int(cik))
    acc = accession.replace("-", "")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cik}_{acc}.json")
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    base = SEC_ARCHIVE.format(cik=cik, accession=acc)
    summary = _get(f"{base}/FilingSummary.xml")

    out: dict[str, dict] = {}
    # Filings before ~2015 have no <MenuCategory> at all, so falling back to
    # the report's own name is what keeps the older half of a 15-year history
    # from silently coming back empty.
    has_categories = "<MenuCategory>" in summary
    for rep in re.findall(r"<Report[^>]*>(.*?)</Report>", summary, re.S):
        sn_probe = re.search(r"<ShortName>(.*?)</ShortName>", rep)
        if has_categories:
            cat = re.search(r"<MenuCategory>(.*?)</MenuCategory>", rep)
            if not cat or cat.group(1) != "Statements":
                continue
        elif not (sn_probe and _STATEMENT_NAME.search(_clean(sn_probe.group(1)))):
            continue
        fn = re.search(r"<HtmlFileName>(.*?)</HtmlFileName>", rep)
        sn = re.search(r"<ShortName>(.*?)</ShortName>", rep)
        if not fn or not sn:
            continue
        short = _clean(sn.group(1))
        # Parentheticals carry par values and share authorisations, not the
        # statement itself; keep them, they are where share-class detail lives.
        try:
            time.sleep(0.12)               # stay under SEC's rate limit
            parsed = parse_report(_get(f"{base}/{fn.group(1)}"))
        except Exception:
            continue
        if parsed:
            parsed["report"] = fn.group(1)
            out[short] = parsed

    if out:
        try:
            with open(cache_path, "w") as f:
                json.dump(out, f)
        except Exception:
            pass
    return out


# ── Verification ─────────────────────────────────────────────────────────────

def check_totals(statements: dict) -> list[dict]:
    """
    Check a parse against the subtotals the company itself printed.

    This is the point of reading whole statements: the balance sheet states
    Assets and Liabilities-and-Equity, so a parse that scaled a column wrongly
    or dropped a row stops tying. A failure here means the data is not
    trustworthy and should not be displayed, rather than displayed with a
    caveat.
    """
    problems = []
    for name, st in statements.items():
        if "parenthetical" in name.lower():
            continue
        # A statement of changes in equity is a roll-forward whose columns are
        # equity COMPONENTS, not periods, so it legitimately has no period
        # columns. Only the three core statements are checked for that.
        if not _CORE_STATEMENT.search(name):
            continue
        # A statement with no period columns yields no values at all, and an
        # empty statement would otherwise sail through every check below —
        # a vacuous pass is the most dangerous kind.
        if not st.get("periods"):
            problems.append({"statement": name, "check": "no period columns parsed"})
            continue
        if not any(r["values"] for r in st["rows"]):
            problems.append({"statement": name, "check": "no values parsed"})
            continue
        by_el: dict[str, dict] = {}
        for row in st["rows"]:
            if row.get("dimension"):
                continue          # a segment column, not the consolidated total
            by_el.setdefault(row["element"], row["values"])
        assets = by_el.get("us-gaap:Assets")
        lae = by_el.get("us-gaap:LiabilitiesAndStockholdersEquity")
        if not assets or not lae:
            continue
        for period in st["periods"]:
            a, l = assets.get(period), lae.get(period)
            if a is None or l is None:
                continue
            if abs(a - l) > max(1.0, abs(a) * 1e-6):
                problems.append({
                    "statement": name, "period": period,
                    "check": "Assets = Liabilities + Equity",
                    "assets": a, "liabilities_and_equity": l, "diff": a - l,
                })
    return problems


def cross_check_companyfacts(statements: dict, facts: dict,
                             tolerance: float = 0.005,
                             accession: Optional[str] = None) -> dict:
    """
    Compare parsed statement values against companyfacts where they overlap.

    companyfacts is lossy but it is not wrong, so every concept present in both
    must agree. This is what catches a misread scale factor ("$ in Millions"
    applied to a statement filed in thousands) — the failure mode that would
    otherwise silently produce numbers off by 1000x.
    """
    ug = (facts.get("facts") or {}).get("us-gaap") or {}
    agree = disagree = 0
    examples = []
    sign_flips = []
    for st in statements.values():
        for row in st["rows"]:
            if not row["element"].startswith("us-gaap:"):
                continue
            if row.get("dimension"):
                # companyfacts holds only undimensioned facts, so a segment
                # row legitimately differs from it and must not be compared.
                continue
            if row.get("opening_balance"):
                continue      # dated at the prior period end, not this column
            concept = row["element"].split(":", 1)[1]
            if concept not in ug:
                continue
            units = ug[concept].get("units") or {}
            unit_key = next((u for u in units if u in ("USD", "shares", "USD/shares")), None)
            if not unit_key:
                continue
            # Restatements mean one period-end can carry several values from
            # different filings. Compare against THIS filing's own numbers
            # where possible, otherwise a legitimate restatement reads as a
            # parse error.
            entries = units[unit_key]
            if accession:
                acc_dash = accession if "-" in accession else \
                    f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
                same = [e for e in entries if e.get("accn") == acc_dash]
                if same:
                    entries = same
            # A duration fact keyed on `end` alone is ambiguous: Q4 and the
            # full year both end 31 December. These are annual statements, so
            # prefer the longest period ending on that date.
            by_end: dict[str, float] = {}
            best_span: dict[str, int] = {}
            for e in entries:
                end = e.get("end")
                if not end:
                    continue
                span = 0
                if e.get("start"):
                    try:
                        from datetime import date
                        y1, m1, d1 = (int(x) for x in e["start"].split("-"))
                        y2, m2, d2 = (int(x) for x in end.split("-"))
                        span = (date(y2, m2, d2) - date(y1, m1, d1)).days
                    except Exception:
                        span = 0
                if end not in by_end or span > best_span.get(end, -1):
                    by_end[end] = e["val"]
                    best_span[end] = span
            for period, val in row["values"].items():
                iso = _period_to_iso(period)
                if not iso or iso not in by_end:
                    continue
                ref = by_end[iso]
                if abs(ref) < 1e-9:
                    continue
                # A rendered statement applies negatedLabel (treasury stock
                # prints negative) while companyfacts keeps the raw value, so a
                # sign difference is not necessarily an error. It is not
                # necessarily fine either — it is exactly how an income
                # statement line can come out backwards — so magnitude
                # agreement counts as agreement but the flip is REPORTED, never
                # silently swallowed.
                if abs(abs(val) - abs(ref)) / abs(ref) <= tolerance:
                    agree += 1
                    if (val < 0) != (ref < 0):
                        sign_flips.append({"element": row["element"],
                                           "period": period,
                                           "parsed": val, "companyfacts": ref})
                else:
                    disagree += 1
                    if len(examples) < 8:
                        examples.append({"element": row["element"], "period": period,
                                         "parsed": val, "companyfacts": ref})
    return {"agree": agree, "disagree": disagree, "examples": examples,
            "sign_flips": sign_flips}


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _period_to_iso(period: str) -> Optional[str]:
    m = re.match(r"^([A-Z][a-z]{2})\.? (\d{1,2}), (\d{4})", period)
    if not m:
        return None
    mon = _MONTHS.get(m.group(1))
    if not mon:
        return None
    return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"


# ── Audit CLI ────────────────────────────────────────────────────────────────

def audit(cik: str, years: int = 15, form: str = "10-K") -> dict:
    """
    Parse every statement in a filer's last `years` filings and verify them.

    Two independent checks, because either alone can pass on bad data: the
    company's own printed subtotals (catches dropped rows) and agreement with
    companyfacts wherever both have the concept (catches misread scale, the
    failure that would otherwise be off by 1000x and look plausible).
    """
    import app                      # local: avoids a circular import at module load

    subs = app.fetch_submissions(str(cik).zfill(10))
    filings = app.all_filing_infos_from_submissions(subs, {form}, max_count=years + 1)
    facts = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
        headers=HEADERS, timeout=90).json()

    cutoff = max(int(f["fiscal_year"]) for f in filings if f["fiscal_year"]) - years + 1
    report = {"filings": [], "agree": 0, "disagree": 0, "tie_out_failures": []}
    for f in filings:
        fy = f.get("fiscal_year")
        if not fy or int(fy) < cutoff:
            continue
        try:
            sts = filing_statements(cik, f["accession"])
        except Exception as e:      # noqa: BLE001
            report["filings"].append({"fy": fy, "error": str(e)[:80]})
            continue
        cc = cross_check_companyfacts(sts, facts, accession=f["accession"])
        tt = check_totals(sts)
        report["agree"] += cc["agree"]
        report["disagree"] += cc["disagree"]
        report["tie_out_failures"] += tt
        report["filings"].append({
            "fy": fy, "statements": len(sts),
            "agree": cc["agree"], "disagree": cc["disagree"],
            "examples": cc["examples"][:3], "tie_out": len(tt),
        })
    total = report["agree"] + report["disagree"]
    report["agreement_pct"] = round(100 * report["agree"] / total, 2) if total else None
    return report


if __name__ == "__main__":
    import sys

    cik_arg = sys.argv[1] if len(sys.argv) > 1 else "1067983"
    rep = audit(cik_arg)
    for f in rep["filings"]:
        if "error" in f:
            print(f"  FY{f['fy']}: ERROR {f['error']}")
            continue
        print(f"  FY{f['fy']}: statements={f['statements']:2d} "
              f"agree={f['agree']:4d} disagree={f['disagree']:3d} "
              f"tie_out_failures={f['tie_out']}")
        for e in f["examples"]:
            print(f"       ! {e['element']} {e['period']}: "
                  f"parsed={e['parsed']:,.0f} companyfacts={e['companyfacts']:,.0f}")
    print(f"\nagreement: {rep['agreement_pct']}%  "
          f"({rep['agree']} agree / {rep['disagree']} disagree)")
    print(f"tie-out failures: {len(rep['tie_out_failures'])}")
