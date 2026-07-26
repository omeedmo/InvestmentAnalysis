"""
Simon Property Group (SPG) company plugin.

Declared by company_templates/SPG.json via `"plugin": "SPG"`.

Why SPG needs code rather than declarative config:

  Simon's FFO (Funds From Operations) is THE headline number for a mall REIT,
  and it is a non-GAAP measure with no XBRL tag at all — it exists only as
  prose/table in the 10-K's MD&A. The prior version of this template computed
  a DERIVED proxy from GAAP components (net income + depreciation - gains) and
  said so honestly in its own caveat: "will NOT tie exactly to Simon's
  reported FFO... treat it as directional." A proxy that admits it doesn't tie
  is worse than reading the real number, and Simon prints the real number in a
  clean, itemized, self-verifying bridge every year:

      Consolidated Net Income  $X
      Adjustments to Arrive at FFO:
        Depreciation and amortization from consolidated properties   +
        Our share of depreciation from unconsolidated entities        +
        Gain on sale/disposal/impairment, net                         -
        Net income attributable to noncontrolling interests in properties -
        Noncontrolling interests' portion of depreciation             -
        Preferred distributions and dividends                         -
      FFO of the Operating Partnership   $X  <- reconciles to this total
      FFO allocable to limited partners  $X
      Dilutive FFO allocable to common stockholders  $X  <- ties to SPG's own
                                                             per-share FFO

extract_spg_ffo parses that bridge, sums the adjustment rows, and returns a
value ONLY when the sum reconciles to Simon's own printed FFO total — same
discipline as Berkshire's operating-earnings and float extractors. Table
format is stable back to FY2013's 10-K, and it gives three years per filing,
so overlapping filings cross-check each other the same way.
"""
from __future__ import annotations

import re
from itertools import combinations, product
from typing import Optional


def normalize_to_fiscal_years(*args, **kwargs):
    import app
    return app.normalize_to_fiscal_years(*args, **kwargs)


# ── FFO bridge parsing ───────────────────────────────────────────────────────

# Two known headings for the total row across the years: pre-2017ish filings
# say "Funds from Operations"; later ones say "FFO of the Operating
# Partnership" once the OP-unit / common-stockholder split is broken out.
_FFO_ANCHOR = re.compile(
    r"Consolidated Net Income\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)"
    r"\s*Adjustments to Arrive at FFO:?(.+?)"
    r"(Funds from Operations|FFO of the Operating Partnership)"
    r"(?:\s*\([A-Za-z0-9]+\))*"          # zero or more footnote markers, e.g. "(A) (B)"
    r"\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)",
    re.IGNORECASE | re.DOTALL,
)

# The path from the FFO total to "Dilutive FFO allocable to..." varies by
# year -- sometimes it's the very next line (via an intervening "FFO
# allocable to limited partners" row), sometimes several Real-Estate-FFO
# adjustment rows sit in between. Rather than model every variant in the
# main anchor, this is a second, independent, shorter-range search for
# whatever three values follow that specific phrase.
_COMMON_ALLOCABLE = re.compile(
    r"Dilutive FFO [Aa]llocable to (?:Simon(?: Property)?|common stockholders)"
    r"(?:\s*\([A-Za-z0-9]+\))*"
    r"\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)",
    re.IGNORECASE,
)

_NUM = re.compile(r"\(\s*[\d,]+\s*\)|[\d,]*\d")


def _clean(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"&#x[0-9a-fA-F]+;|&#\d+;|&[a-zA-Z]+;", " ", t)
    return re.sub(r"\s+", " ", t)


def _num(tok: str) -> float:
    tok = tok.strip()
    if tok.startswith("("):
        return -float(tok[1:-1].replace(",", ""))
    return float(tok.replace(",", ""))


def _rows(body: str) -> list[tuple[str, list[float]]]:
    """Split the adjustments block into (label, [values]) pairs, one per year
    column. A row boundary follows a digit/paren/percent OR a dash placeholder
    ("--", meaning zero/not applicable that year) -- missing the dash case
    silently merged a row ending in one into the next label entirely, e.g.
    "Loss on extinguishment of debt 116,256 -- Unrealized losses..." parsed as
    one row with three values instead of two rows with one value each, which
    both mis-sums the total and eats a real adjustment line."""
    out = []
    for part in re.split(r"(?:(?<=[\d)%])|(?<=-))\s+(?=[A-Za-z])", body):
        m = re.search(r"[\d(]", part)
        if not m:
            continue
        label, tail = part[:m.start()], part[m.start():]
        # A footnote reference like "Other expenses (1) 818 320" leaves "(1)"
        # as the very first token in tail, which _num() would otherwise read
        # as a genuine "-1" value. Real figures in this table are always in
        # the hundreds of thousands at minimum, so a leading 1-2-digit or
        # single-letter parenthetical is always a footnote marker, not data.
        tail = re.sub(r"^\(\s*(?:\d{1,2}|[A-Za-z])\s*\)\s*", "", tail)
        vals = []
        for tk in _NUM.findall(tail):
            tk = tk.strip()
            if not tk:
                continue
            if re.fullmatch(r"(19|20)\d\d", tk):   # a year, not a value
                continue
            vals.append(_num(tk))
        if label.strip() and vals:
            out.append((label.strip(), vals))
    return out


def _solve(rows: list, totals: list[float], seed: list[float],
          skip_labels: Optional[re.Pattern] = None) -> Optional[list[float]]:
    """
    Assign each adjustment row's values to year columns so every column sums
    to Simon's own printed total, starting from the seed (Consolidated Net
    Income). Works for however many year-columns `totals`/`seed` have (FFO's
    bridge gives 3, NOI's gives 2).

    `skip_labels` excludes rows that are themselves a running SUBTOTAL rather
    than a new adjustment — SPG's NOI bridge prints "Operating Income Before
    Other Items" partway through, which restates the cumulative sum so far
    rather than adding to it; summing it in would double it.

    A row missing a value in one column (a one-off item, or a line that only
    existed in some years) is genuinely ambiguous once flattened to text, so
    every placement is tried and only an assignment that reconciles to the
    printed total is accepted — same principle as the Berkshire extractors:
    if the table was misread, the sums will not tie, and this returns None
    rather than a plausible-looking wrong number.
    """
    n = len(seed)
    fixed = list(seed)
    ragged = []
    for label, vals in rows:
        if skip_labels and skip_labels.search(label):
            continue
        if len(vals) == n:
            for i in range(n):
                fixed[i] += vals[i]
        elif 1 <= len(vals) < n:
            ragged.append(vals)
        else:
            return None
    if len(ragged) > 5:
        return None
    choices = [list(combinations(range(n), len(v))) for v in ragged]
    for combo in (product(*choices) if choices else [()]):
        cols = list(fixed)
        for vals, slots in zip(ragged, combo):
            for v, i in zip(vals, slots):
                cols[i] += v
        if all(abs(cols[i] - totals[i]) < 1.0 for i in range(n)):
            return cols
    return None


# ── NOI bridge parsing ───────────────────────────────────────────────────────
# A separate, two-column bridge headed "Reconciliation of NOI of consolidated
# entities:", stable back to FY2018's 10-K. It has an internal subtotal row,
# "Operating Income Before Other Items", which restates the running sum so far
# rather than adding to it -- _NOI_SKIP excludes it from the sum.
_NOI_ANCHOR = re.compile(
    r"Reconciliation of NOI of consolidated entities:\s*"
    r"Consolidated Net Income\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)"
    r"(.+?)"
    r"NOI of consolidated entities\s*\$?\s*([\d,()]+)\s*\$?\s*([\d,()]+)",
    re.IGNORECASE | re.DOTALL,
)
_NOI_SKIP = re.compile(r"Operating Income Before Other Items", re.IGNORECASE)


def extract_spg_noi(text: str) -> dict[str, dict[str, float]]:
    """
    Simon's own "NOI of consolidated entities" -- the first, most directly
    reconciled checkpoint in Simon's NOI bridge (which continues on, through
    unconsolidated joint ventures and international investments, to a
    "Beneficial interest of Portfolio NOI" headline figure Simon emphasizes
    more -- that further chain is NOT attempted here: each additional stage
    is a fresh set of one-off adjustment lines and footnotes, and this
    stopped at the point where the reconciliation is still a single, stable,
    cleanly self-verifying bridge). Consolidated NOI is still Simon's own
    reported figure for the properties it controls, not an EBITDA + G&A
    proxy computed from unrelated line items.

    Two years per filing (not three, like FFO), so cross-filing confirmation
    is one year shallower, but still present.
    """
    txt = _clean(text)
    m = _NOI_ANCHOR.search(txt)
    if not m:
        return {}

    ni = [_num(m.group(1)), _num(m.group(2))]
    body = m.group(3)
    totals = [_num(m.group(4)), _num(m.group(5))]

    rows = _rows(body)
    cols = _solve(rows, totals, seed=ni, skip_labels=_NOI_SKIP)
    if cols is None:
        return {}

    header = re.search(r"(\d{4})\s+(\d{4})\s*\(in thousands\)",
                       txt[max(0, m.start() - 200):m.start()])
    if not header:
        return {}
    years = [int(header.group(1)), int(header.group(2))]

    return {"noi_consolidated": {f"{y}-12-31": v * 1000 for y, v in zip(years, cols)}}


def extract_spg_ffo(text: str) -> dict[str, dict[str, float]]:
    """
    Simon's own FFO bridge: {"ffo_operating_partnership": {date: value},
    "ffo_allocable_to_common": {date: value}}, keyed by the calendar year each
    column covers. The filing gives three years at a time; every value
    returned here reconciled to Simon's own printed subtotal.
    """
    txt = _clean(text)
    m = _FFO_ANCHOR.search(txt)
    if not m:
        return {}

    ni = [_num(m.group(i)) for i in (1, 2, 3)]
    body = m.group(4)
    ffo_totals = [_num(m.group(i)) for i in (6, 7, 8)]

    rows = _rows(body)
    cols = _solve(rows, ffo_totals, seed=ni)
    if cols is None:
        return {}

    # Recover the calendar years the three columns cover from the filing's own
    # nearby "For the Year Ended December 31, YYYY YYYY YYYY" header, which
    # always immediately precedes this anchor.
    header = re.search(r"(\d{4})\s+(\d{4})\s+(\d{4})\s*\(in thousands\)",
                       txt[max(0, m.start() - 200):m.start()])
    if not header:
        return {}
    years = [int(header.group(i)) for i in (1, 2, 3)]

    out = {"ffo_operating_partnership": {}}
    for y, v in zip(years, cols):
        out["ffo_operating_partnership"][f"{y}-12-31"] = v * 1000  # values are in thousands

    # Independent second pass, searched only in the text after the FFO total
    # (within a generous window -- Real Estate FFO's extra adjustment rows can
    # sit between the two on some years).
    common_m = _COMMON_ALLOCABLE.search(txt[m.end():m.end() + 2500])
    if common_m:
        try:
            common_totals = [_num(common_m.group(i)) for i in (1, 2, 3)]
            out["ffo_allocable_to_common"] = {
                f"{y}-12-31": v * 1000 for y, v in zip(years, common_totals)}
        except ValueError:
            pass

    return out


# ── Hooks called by the core app (see company_templates.call_hook) ──────────

def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    """Walk 10-Ks newest-first, filling FFO/NOI wherever not already covered."""
    fy_get = ctx["fy_get"]
    min_year = ctx["min_year"]
    get_text = ctx["get_text"]

    for filing in filings:
        fy = filing.get("fiscal_year", "")
        if not fy or int(fy) < min_year:
            break
        fy_int = int(fy)
        ffo_covered = all(
            fy_get(financials.get("ffo_operating_partnership", {}), str(fy_int - i)) is not None
            for i in range(3))
        noi_covered = all(
            fy_get(financials.get("noi_consolidated", {}), str(fy_int - i)) is not None
            for i in range(2))
        if ffo_covered and noi_covered:
            continue
        text = get_text(filing)
        for extractor in (extract_spg_ffo, extract_spg_noi):
            try:
                result = extractor(text)
            except Exception:
                continue
            for key, series in result.items():
                merged = dict(financials.get(key, {}))
                for d, v in series.items():
                    merged.setdefault(d, v)
                financials[key] = merged


def postprocess(financials: dict) -> None:
    """
    Surface Simon's own FFO as the app's "ffo" row and recompute FFO/share.

    "Dilutive FFO allocable to common stockholders" is the figure that ties to
    Simon's own reported per-share FFO (verified: dividing it by basic/diluted
    weighted average shares reproduces the disclosed "Diluted FFO per share"
    to the cent). It replaces the generic proxy (Net Income + D&A - RE gains)
    that the app computes for every other REIT and that this template's own
    prior version admitted "will NOT tie exactly to Simon's reported FFO".

    ffo_per_share has to be recomputed here rather than left alone: the app
    computes it from the proxy INSIDE build_financials, before this hook or
    the authored history ever run, so once "ffo" is corrected the per-share
    figure would otherwise keep reflecting the old proxy.
    """
    ffo = financials.get("ffo_allocable_to_common")
    if not ffo:
        return
    financials["ffo"] = dict(ffo)

    shares = financials.get("shares_outstanding_end", {})
    if not shares:
        return

    def by_year(series):
        return {k[:4]: v for k, v in series.items()
                if len(k) >= 4 and k[:4].isdigit() and v is not None}

    ffo_y, shares_y = by_year(ffo), by_year(shares)
    financials["ffo_per_share"] = {
        f"{y}-12-31": ffo_y[y] / shares_y[y]
        for y in ffo_y
        # Simon's own XBRL mistags CommonStockSharesOutstanding as 8,000 for
        # FY2011/2012 (confirmed against companyfacts directly -- almost
        # certainly the ~8,000-share Series J preferred count, tagged to the
        # wrong concept in that filing). Dividing by it would put FFO/share at
        # ~$250,000, a number that would visibly announce itself as broken
        # rather than merely be imprecise, so those two years are left blank
        # instead of wrong. No REIT this size has under a million common
        # shares outstanding.
        if shares_y.get(y, 0) > 1_000_000
    }

    # NOI: same principle, but this one is a REPLACEMENT, not a merge. The
    # reconciled figure only reaches back to FY2017 (the earliest comparative
    # in the oldest filing with this bridge); splicing it over the generic
    # EBITDA + G&A proxy for the years it doesn't cover would silently change
    # what the row means partway across the series. A row with a gap before
    # 2017 is preferable to one that means two different things depending on
    # which year you're looking at.
    noi = financials.get("noi_consolidated")
    if noi:
        financials["noi"] = dict(noi)
        noi_margin_base = financials.get("revenue", {})

        def by_year2(series):
            return {k[:4]: v for k, v in series.items()
                    if len(k) >= 4 and k[:4].isdigit() and v is not None}

        noi_y, rev_y = by_year2(noi), by_year2(noi_margin_base)
        financials["noi_margin"] = {
            f"{y}-12-31": noi_y[y] / rev_y[y] for y in noi_y if rev_y.get(y)}
        financials["noi_per_share"] = {
            f"{y}-12-31": noi_y[y] / shares_y[y]
            for y in noi_y if shares_y.get(y, 0) > 1_000_000
        }
