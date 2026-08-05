"""
CVS: the three operating measures it reports and XBRL does not carry.

  Medical benefit ratio     health care costs over premium revenues, for the
                            Health Care Benefits segment
  Pharmacy claims processed Pharmacy & Consumer Wellness volume
  Prescriptions filled      the same, on a 90-day-adjusted basis

None is XBRL-tagged, so the MD&A tables are the only source. MBR is the one
that matters most, because the app already shows a `loss_ratio` for this filer
and the two are different measures that look like the same thing: loss_ratio
is consolidated PolicyholderBenefitsAndClaimsIncurredNet over consolidated
PremiumsEarnedNet, 89.7% for Q2 FY2026, while the MBR CVS reports for the
segment was 87.4%. Both are right; only one is the company's.

Two parsing details cost real time and are worth stating.

The MBR row cannot be anchored on its own name. "Medical benefit ratio"
appears five times in the FY2025 10-K -- in the forward-looking-statements
list, in risk factors, and twice in the commentary defining it -- and not once
immediately before the figures. The row label wraps, so what actually precedes
the three percentages is the tail of its parenthetical: "...premium revenues)
91.2 % 92.5 % 86.2%". That tail is the anchor.

The count rows carry a footnote marker between the label and the first figure
("Pharmacy claims processed (5) 1,900.7 ..."), which has to be skipped or the
marker is read as the year's value.

Verification is by agreement across filings, which is the check available when
a measure has no XBRL counterpart to reconcile against. Each 10-K prints three
years, so consecutive filings overlap on two, and a year is accepted only once
some filing reporting it agrees with a year already established. A misread
column does not survive that, because the neighbouring filing puts a different
number in the same place. The oldest filing in the window is the seed and is
taken on its own; everything after it is confirmed.
"""
from __future__ import annotations

import re

_PCT = r"([\d.]+)\s*%"
# Anchored on the tail of the wrapped label, not the label itself -- see above.
MBR = re.compile(r"premium revenues\)\s*" + _PCT + r"\s*" + _PCT + r"\s*" + _PCT)

_NUM = r"([\d,]+\.?\d*)"
# (\(\d+\)\s*)? skips the footnote marker that sits between label and figures.
CLAIMS = re.compile(r"Pharmacy claims processed\s*(?:\(\d+\)\s*)?"
                    + _NUM + r"\s+" + _NUM + r"\s+" + _NUM)
SCRIPTS = re.compile(r"Prescriptions filled\s*(?:\(\d+\)\s*)?"
                     + _NUM + r"\s+" + _NUM + r"\s+" + _NUM)

_TOL = 0.005


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"&#\d+;|&[a-zA-Z]+;", " ", text))


def _read(text: str) -> dict:
    """{metric: {column index: value}} for one filing, columns newest first."""
    flat = _flat(text)
    out: dict = {}
    for name, pat, scale in (("mbr", MBR, 0.01),
                             ("pharmacy_claims", CLAIMS, 1e6),
                             ("prescriptions", SCRIPTS, 1e6)):
        m = pat.search(flat)
        if not m:
            continue
        try:
            out[name] = {i: float(g.replace(",", "")) * scale
                         for i, g in enumerate(m.groups())}
        except ValueError:
            continue
    return out


def _resolve(parsed: list) -> dict:
    """{fiscal year: value}, each confirmed by a second filing where one exists."""
    out: dict = {}
    accepted: set = set()
    changed = True
    while changed:
        changed = False
        for fy, cols in parsed:
            if fy in accepted:
                continue
            known = [(out[str(fy - i)], v) for i, v in cols.items()
                     if str(fy - i) in out]
            if known and any(abs(v - k) > max(abs(k) * _TOL, 1e-9) for k, v in known):
                continue                      # contradicts an established year
            if not known and out:
                continue                      # no overlap yet; try again later
            accepted.add(fy)
            for i, v in cols.items():
                out.setdefault(str(fy - i), v)
            changed = True
    return out


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    get_text, min_year = ctx["get_text"], ctx.get("min_year", 0)
    per_metric: dict = {}
    for f in filings:
        fy = f.get("fiscal_year")
        if not fy or int(fy) < min_year:
            continue
        try:
            got = _read(get_text(f))
        except Exception:                     # noqa: BLE001 - a bad parse is a gap
            continue
        for name, cols in got.items():
            per_metric.setdefault(name, []).append((int(fy), cols))

    key_of = {d[:4]: d for d in financials.get("revenue", {})
              if not str(d).startswith("Q")}
    for name, parsed in per_metric.items():
        parsed.sort(key=lambda p: p[0])       # oldest first: the seed anchors it
        series = _resolve(parsed)
        if series:
            financials[f"{name}_reported"] = {
                key_of.get(y, f"{y}-12-31"): v for y, v in series.items()}


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    """
    The same three rows for the quarter columns, from the 10-Q MD&A.

    A 10-Q prints four columns where the 10-K prints three: this quarter, the
    same quarter a year ago, then the two year-to-date figures. The patterns
    capture three groups and the first is the three-month figure in both forms,
    so no separate quarterly regex is needed -- Q2 FY2026 reads 87.4% against
    the 89.9% / 86.0% / 88.6% beside it, which is the ratio CVS reported.

    The counts carry their own check and it is worth taking, because a column
    misread is exactly the failure this app has hit before. Quarterly claims
    and prescriptions must sum to the year-to-date column printed alongside
    them, so where the earlier quarter is already known the two are added and
    compared. A year-to-date figure mistaken for a quarter fails it by roughly
    a factor of the quarter number. MBR gets no such check -- a ratio does not
    add -- so it rests on the column position alone, which is why the anchor is
    pinned to the label's own parenthetical rather than a loose search.
    """
    get_text_url = ctx.get("get_text_url")
    if not get_text_url:
        return

    for qk in sorted(quarter_end_dates):
        url = quarter_filing_links.get(qk)
        if not url:
            continue
        try:
            flat = _flat(get_text_url(url))
        except Exception:                     # noqa: BLE001 - a fetch failure is a gap
            continue
        for name, pat, scale in (("mbr", MBR, 0.01),
                                 ("pharmacy_claims", CLAIMS, 1e6),
                                 ("prescriptions", SCRIPTS, 1e6)):
            m = pat.search(flat)
            if not m:
                continue
            try:
                vals = [float(g.replace(",", "")) * scale for g in m.groups()]
            except ValueError:
                continue
            series = financials.setdefault(f"{name}_reported", {})
            if name != "mbr" and len(vals) >= 3:
                # vals[2] is the year-to-date column. Quarters before this one
                # must account for the difference, or the columns were misread.
                prior = [series[k] for k in series
                         if str(k).startswith("Q") and k < qk]
                if prior and abs(sum(prior) + vals[0] - vals[2]) > vals[2] * 0.005:
                    continue
            series[qk] = vals[0]
