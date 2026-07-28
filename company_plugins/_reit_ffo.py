"""
Shared machinery for reading a REIT's own reported FFO out of its 10-K.

FFO is not XBRL-tagged. Checked across Simon, Realty Income, Federal Realty,
AvalonBay, Equinix and American Tower, the entire label linkbase of a REIT's
10-K contains no element labelled "funds from operations" — so companyfacts and
the rendered statements alike have nothing to read, and the figure exists only
as a prose table in MD&A. The app's generic row is a proxy (net income plus
D&A) that no REIT actually publishes.

Each company plugin supplies its own regex for the subtotal line it prints, and
this module does the rest. What it will NOT do is guess: a generic version of
this, matching any "funds from operations" total in any REIT's filing, was
built first and produced two classes of silent error —

  * Kimco's FY2013 filing stacks Q4-2013, Q4-2012, FY2013 and FY2012 in one
    table, so reading the leftmost columns as fiscal years reported a QUARTER
    as the year, and the adjustments summed to it perfectly.
  * Tables print in thousands or in millions, and Equinix switched between
    them mid-history, so a figure taken at face value is out by 1000x.

Both are caught here by verifying against data from outside the text. The
table's own net-income row must agree with the net income the app already
holds from XBRL for the same fiscal year. That single check confirms three
things at once: the right table was found, the scale is right, and column 0
really is the filing's own fiscal year. A block that fails it yields nothing,
so a parse error becomes a gap rather than a wrong number on screen.
"""
from __future__ import annotations

import re
from typing import Optional

_NUM = re.compile(r"\(?\s*-?[\d,]+(?:\.\d+)?\s*\)?")

_SCALES = {"thousand": 1e3, "thousands": 1e3,
           "million": 1e6, "millions": 1e6,
           "billion": 1e9, "billions": 1e9}
_SCALE_DECL = re.compile(r"in (thousands|millions|billions|thousand|million|billion)", re.I)

# The net income line a reconciliation opens on. The tolerance on the check is
# deliberately loose because the variants are genuinely different numbers: some
# REITs start from consolidated net income, others from the amount available to
# common after preferred dividends and noncontrolling interests. Federal
# Realty's table shows $304M for FY2024 where XBRL carries $411M, a 26% gap
# that is correct in both places. What the check has to catch is categorically
# larger: a 1000x scale error, a different table entirely, or a quarterly
# column read as a year (about 75% adrift). Half is comfortably inside all
# three and outside the legitimate variation.
NET_INCOME_LABEL = re.compile(r"net (?:income|earnings|loss)\b", re.I)


def normalise(text: str) -> str:
    """
    Flatten to one line and make the table's placeholders parseable.

    HTML entities survive the app's fast text extraction, and an em-dash
    written as `&#8212;` reads as the number 8212 — Federal Realty's
    "Impairment charge 7,425 &#8212; &#8212;" parsed as three real values. The
    entity's trailing semicolon also suppressed the row split, swallowing the
    next adjustment whole.
    """
    t = re.sub(r"&#8?2[01][12];|&[mn]dash;", " — ", text)
    t = re.sub(r"&#x[0-9a-fA-F]+;|&#\d+;|&[a-zA-Z]+;", " ", t)
    t = re.sub(r"\s+", " ", t)
    # A dash BETWEEN FIGURES is the tables' nil placeholder and has to become a
    # zero, or the remaining figures shift a column left into the wrong year.
    # Requiring a digit or closing paren before it keeps it from firing inside
    # a label ("Depreciation and amortization — real estate related").
    return re.sub(r"(?<=[\d)]) [‒-―−-](?= )", " 0", t)


def num(tok: str) -> Optional[float]:
    tok = tok.strip()
    neg = tok.startswith("(")
    tok = tok.strip("()").strip()
    # Must contain an actual digit. "[\d,]+" alone also matches a lone comma,
    # which then became float("") and raised — and because walk_filings treats
    # any exception as a gap, every filing after the first such token silently
    # produced nothing at all rather than reporting a problem.
    if not re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", tok):
        return None
    v = float(tok.replace(",", ""))
    return -v if neg else v


def values_after(tail: str, want: int) -> list[float]:
    """The first `want` figures in a row's tail, skipping year headings."""
    out: list[float] = []
    for tok in _NUM.findall(tail):
        if re.fullmatch(r"\(?\s*(19|20)\d\d\s*\)?", tok.strip()):
            continue
        v = num(tok)
        if v is None:
            continue
        out.append(v)
        if len(out) >= want:
            break
    return out


def scale_before(flat: str, pos: int, window: int = 900) -> Optional[float]:
    """
    The unit the table declares, read from the text just above it.

    Federal Realty prints FFO as 631,373 meaning $631M; Equinix prints 2,668
    meaning $2.67B, having reported in thousands earlier in its history. A
    block with no declared unit is refused rather than guessed at.
    """
    decls = _SCALE_DECL.findall(flat[max(0, pos - window):pos])
    return _SCALES[decls[-1].lower()] if decls else None


def extract(text: str, total_re: re.Pattern, ni_by_year: dict, fiscal_year: int,
            ncols: int = 2, tolerance: float = 0.25) -> dict[int, float]:
    """
    {column index: FFO} for the company's own subtotal, verified.

    `ni_by_year` is {"YYYY": net income} as the app already holds it from XBRL.
    Column 0 is the filing's own fiscal year, column 1 the year before it.
    """
    flat = normalise(text)
    for m in total_re.finditer(flat):
        vals = values_after(flat[m.end() - 1:], ncols) if m.groups() else []
        if not vals:
            vals = values_after(flat[m.end():], ncols)
        if len(vals) < ncols:
            continue
        scale = scale_before(flat, m.start())
        if scale is None:
            continue

        # Verify against XBRL: some net-income row in the block above this
        # subtotal must agree with the net income already known for this
        # fiscal year.
        #
        # ANY of them, not the nearest. These tables carry several lines
        # opening "net income", and the one closest to the subtotal is usually
        # an adjustment rather than the starting figure — Federal Realty's is
        # "Net income attributable to noncontrolling interests (12,571)", so
        # anchoring on the nearest match compared -$12.6M against an expected
        # $423.6M and discarded a correctly read table. Scanning them all is no
        # weaker a check, since a wrong table or a 1000x scale error has no row
        # that agrees, but it tolerates where a company puts its opening line.
        #
        # The tolerance stays tight. Loosening it to half, to accommodate
        # filings whose table opens on income available to common rather than
        # consolidated net income, let genuinely wrong tables through and put
        # Federal Realty's series back to a shape its own filings contradict.
        # A year that cannot be confirmed at this tolerance is left blank.
        expected = ni_by_year.get(str(fiscal_year))
        if not expected:
            continue
        head = flat[max(0, m.start() - 2600):m.start()]
        ok = False
        for ni_m in NET_INCOME_LABEL.finditer(head):
            row = values_after(head[ni_m.end():], 1)
            if row and abs(row[0] * scale - expected) <= abs(expected) * tolerance:
                ok = True
                break
        if not ok:
            continue                 # wrong table, wrong scale, or wrong column

        return {i: v * scale for i, v in enumerate(vals)}
    return {}


def walk_filings(filings: list, ctx: dict, total_re: re.Pattern,
                 ncols: int = 2, tolerance: float = 0.25,
                 from_year: Optional[int] = None) -> dict[str, float]:
    """
    {fiscal year: FFO} across a filer's 10-K history, newest filing winning.

    Each 10-K carries two or three years, so a year is reported by several
    filings; the most recent view of it is the one kept, matching how the
    scorecard fact base resolves the same overlap.
    """
    get_text, min_year = ctx["get_text"], ctx["min_year"]
    ni_by_year = {k[:4]: v for k, v in (ctx.get("net_income") or {}).items()
                  if v is not None and not str(k).startswith("Q")}
    out: dict[str, float] = {}
    for f in filings:
        fy = f.get("fiscal_year")
        if not fy or int(fy) < min_year:
            break
        try:
            cols = extract(get_text(f), total_re, ni_by_year, int(fy),
                           ncols=ncols, tolerance=tolerance)
        except Exception:            # noqa: BLE001 - a bad parse is a gap
            continue
        for i, v in cols.items():
            year = int(fy) - i
            # A REIT that converted mid-history has no comparable Nareit FFO
            # before it converted; Equinix elected REIT status for 2015 and its
            # FY2014 figure is a different measure under the same words.
            if from_year is not None and year < from_year:
                continue
            out.setdefault(str(year), v)
    return out


def publish(financials: dict, ffo_by_year: dict, label_key: str = "ffo") -> None:
    """
    Write the reported series onto the app's FFO row and rebuild what hangs
    off it.

    A REPLACEMENT, not a merge. The reported figure and the app's net-income-
    plus-D&A proxy are different definitions, and a row that is reported for
    some years and derived for others changes meaning partway across without
    saying so. Years the verification could not confirm stay blank.
    """
    if not ffo_by_year:
        return
    key_of = {d[:4]: d for d in financials.get("net_income", {})
              if not str(d).startswith("Q")}
    ffo = {key_of.get(y, y): v for y, v in ffo_by_year.items()}
    financials[label_key] = ffo

    def by_year(series):
        return {str(k)[:4]: v for k, v in (series or {}).items()
                if v is not None and not str(k).startswith("Q")}

    ffo_y = by_year(ffo)
    shares_y = by_year(financials.get("shares_diluted_wtd")
                       or financials.get("shares_outstanding_end"))
    if shares_y:
        financials["ffo_per_share"] = {
            key_of.get(y, y): ffo_y[y] / shares_y[y]
            for y in ffo_y if shares_y.get(y)}

    # AFFO and the payout ratio were derived from the proxy inside
    # build_financials, before this hook ran. Rebuild them on the reported
    # figure rather than leave them standing on the number it replaced.
    slr_y = by_year(financials.get("straight_line_rent"))
    rcx_y = by_year(financials.get("recurring_capex"))
    if slr_y or rcx_y:
        affo_y = {y: v - abs(slr_y.get(y, 0.0)) - abs(rcx_y.get(y, 0.0))
                  for y, v in ffo_y.items()}
        financials["affo"] = {key_of.get(y, y): v for y, v in affo_y.items()}
        if shares_y:
            financials["affo_per_share"] = {
                key_of.get(y, y): affo_y[y] / shares_y[y]
                for y in affo_y if shares_y.get(y)}
    else:
        for k in ("affo", "affo_per_share"):
            financials.pop(k, None)

    div_y = by_year(financials.get("dividends_paid"))
    financials["ffo_payout_ratio"] = {
        key_of.get(y, y): abs(div_y[y]) / ffo_y[y]
        for y in ffo_y if ffo_y[y] > 0 and div_y.get(y)}
