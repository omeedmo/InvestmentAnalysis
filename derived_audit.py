"""
Which derived rows were computed before the values they derive from?

Read-only. Nothing here changes behaviour.

build_financials derives a row like FCF near the top of the run, from whatever
companyfacts alone returned. For a filer with an authored binding most of its
inputs arrive later — from the scorecard overlay, a template, a plugin, or the
authored history — and unless something recomputes the row afterwards it keeps
the value it had before any of that landed. The row is not blank because the
data is missing; it is blank because it was worked out too early.

That is invisible from the row itself. What gives it away is the combination:
every input present for a year, the derived row empty for the same year. Lumen
showed sixteen years of operating cash flow and a complete capex line above an
FCF row that stopped at 2013, and separately had a UNTA row that never moved
when its inputs changed. Both were found by someone looking at a blank cell.

So this checks the arithmetic against the finished series rather than reading
the code: for each derived metric and each fiscal year, if every input has a
value, does the row have one too, and is it the value the inputs imply?

  STALE        inputs all present, derived row empty. The signature above.
  MISMATCH     both present, disagree beyond tolerance. Not automatically a
               fault — a binding, template or plugin may supply the row on
               purpose and outrank the formula, which is why the check reports
               where the value came from rather than asserting the formula
               wins. Berkshire's owner earnings and Simon's FFO are supplied,
               not computed.

Run against filers with a company binding by default, because those are the
ones whose inputs arrive after the derivation; pass tickers to widen it.

Usage:  python3 derived_audit.py [TICKER ...]      (exit 0 = clean)
"""
from __future__ import annotations

import glob
import os
import sys

import app

# metric -> (inputs, fn). Kept to rows whose formula is stated in one place and
# is genuinely a function of other rows on the same page. Ratios that are only
# meaningful for one sector are included; they simply never fire for filers
# that have neither side.
DERIVED = {
    "fcf":            (("operating_cash_flow", "capex"),   lambda o, c: o - abs(c or 0)),
    "total_cash":     (("cash", "short_term_investments"), lambda c, s: c + (s or 0)),
    "total_debt":     (("long_term_debt", "current_debt"), lambda l, c: (l or 0) + (c or 0)),
    "net_cash":       (("total_cash", "total_debt"),       lambda c, d: c - d),
    "ebitda":         (("operating_income", "depreciation"), lambda o, d: o + (d or 0)),
    "operating_margin": (("operating_income", "revenue"),  lambda o, r: o / r if r else None),
    "net_margin":     (("net_income", "revenue"),          lambda n, r: n / r if r else None),
    "roe":            (("net_income", "equity"),           lambda n, e: n / e if e else None),
    "effective_tax_rate": (("income_tax", "pretax_income"), lambda t, p: t / p if p else None),
    "revenue_per_share": (("revenue", "shares_diluted_wtd"), lambda r, s: r / s if s else None),
    "fcf_per_share":  (("fcf", "shares_diluted_wtd"),      lambda f, s: f / s if s else None),
    "book_value_per_share": (("equity", "shares_outstanding_end"), lambda e, s: e / s if s else None),
    # The post-overlay derivations. These are rebuilt after the overlay writes,
    # in a fixed order that nothing enforces, which is how UNTA came to be
    # computed against a total_debt of zero: the debt recompute was added later
    # and placed after the UNTA block that consumes it. Upbound read
    # -262,540,000 where its inputs give +1,307,344,000, the sign inverted, and
    # nothing flagged it because unta was not on this list.
    "unta": (("equity", "total_debt", "total_cash", "goodwill", "intangibles"),
             lambda e, d, c, g, i: e + d - c - g - i),
    # Guarded in app.py: a ratio over negative tangible capital is misleading,
    # so those years are deliberately absent rather than negative. Returning
    # None here matches, so the guard does not read as a stale row.
    "economic_goodwill": (("nopat", "unta"),
                          lambda n, u: n / u if u and u > 0 else None),
    "roic": (("operating_income", "effective_tax_rate",
              "equity", "total_debt", "total_cash"),
             lambda oi, t, e, d, c: (oi * (1 - min(t or 0.21, 0.5)) / (e + d - c))
             if (e + d - c) > 0 else None),
    # The rest of the margin family and the SBC-adjusted rows. Absent from the
    # first version of this file, which is the only reason it reported CLEAN
    # while Lumen's FCF margin stood at three years of fifteen: a row the audit
    # does not name is a row it cannot check, and the gap looks identical to
    # having no gap.
    "gross_margin":   (("gross_profit", "revenue"),        lambda g, r: g / r if r else None),
    "ebitda_margin":  (("ebitda", "revenue"),              lambda e, r: e / r if r else None),
    "fcf_margin":     (("fcf", "revenue"),                 lambda f, r: f / r if r else None),
    "adj_fcf":        (("fcf", "stock_based_compensation"), lambda f, s: f - abs(s)),
    "adj_fcf_margin": (("adj_fcf", "revenue"),             lambda a, r: a / r if r else None),
    "adj_fcf_roe":    (("adj_fcf", "equity"),              lambda a, e: a / e if e and e > 0 else None),
}

TOL = 0.01          # 1%: past rounding, well short of a different definition


def _annual(fin: dict, metric: str) -> dict:
    return {str(k)[:4]: v for k, v in (fin.get(metric) or {}).items()
            if not str(k).startswith("Q") and v is not None}


def _bound_tickers() -> list:
    return sorted(os.path.basename(p)[:-5]
                  for p in glob.glob("bindings/companies/*.json"))


def audit(ticker: str, client) -> list:
    d = client.get(f"/api/analyze?ticker={ticker}").get_json()
    if not d or d.get("error"):
        return [(ticker, "-", "-", "ERROR", d.get("error", "no data") if d else "no data")]
    fin = d.get("financials") or {}
    applied = ((d.get("scorecard") or {}).get("applied") or {})
    out = []
    for metric, (inputs, fn) in DERIVED.items():
        got = _annual(fin, metric)
        series = [_annual(fin, i) for i in inputs]
        years = set(series[0])
        for s in series[1:]:
            years &= set(s)
        src = "overlay" if metric in applied else "formula"

        # Split the blanks two ways before reporting them, because they are
        # different faults with different fixes. A row with nothing in it was
        # derived too early and never revisited. A row with SOME years in it,
        # blank in others where the inputs are present, is a row half-written
        # by one source and never completed by the other -- Lumen's FCF margin
        # is a binding expression the overlay could only resolve for four
        # years, and the refresh that should have filled the rest skipped the
        # metric entirely on the grounds that it was "supplied". Whether the
        # remedy is fill-only or overwrite depends on which of these it is, so
        # collapsing them into one bucket hides the answer.
        blanks, mismatches = [], []
        for y in sorted(years):
            try:
                want = fn(*[s[y] for s in series])
            except Exception:
                continue
            if want is None:
                continue
            have = got.get(y)
            if have is None:
                # A row the overlay supplies covers the years its binding
                # covers, and no others -- for a derived binding metric that is
                # the clearing working as designed, not a stale row. Berkshire
                # binds its own UNTA and 2014 falls outside it. Judging a
                # supplied row against the app's formula is the same category
                # error the margin refresh had to avoid.
                if src != "overlay":
                    blanks.append((y, f"implied {want:,.4g}"))
            elif abs(want) > 1e-9 and abs(have - want) / max(abs(want), 1e-9) > TOL:
                mismatches.append((y, f"row {have:,.4g} vs inputs {want:,.4g}  [{src}]"))

        kind = "PARTIAL" if (blanks and got) else "STALE"
        for y, why in blanks:
            out.append((ticker, metric, y, kind,
                        f"{why}; row has {len(got)} other year(s)" if kind == "PARTIAL"
                        else f"{why}; row is empty throughout"))
        for y, why in mismatches:
            out.append((ticker, metric, y, "MISMATCH", why))
    return out


def main() -> int:
    tickers = sys.argv[1:] or _bound_tickers()
    client = app.app.test_client()
    print(f"checking {len(DERIVED)} derived rows across {len(tickers)} filers: "
          f"{', '.join(tickers)}\n")

    findings = []
    for tk in tickers:
        findings += audit(tk, client)

    stale = [f for f in findings if f[3] == "STALE"]
    part  = [f for f in findings if f[3] == "PARTIAL"]
    mism  = [f for f in findings if f[3] == "MISMATCH"]

    def report(title, rows):
        print(f"── {title} ({len(rows)})")
        by = {}
        for tk, metric, y, _, why in rows:
            by.setdefault((tk, metric), []).append((y, why))
        for (tk, metric), items in sorted(by.items()):
            yrs = ", ".join(y for y, _ in items)
            print(f"     {tk:7} {metric:22} {len(items):>2} yr  {yrs[:46]}")
            print(f"             {items[0][1]}")

    report("derived row empty throughout while its inputs have values", stale)
    print()
    report("derived row filled for some years, blank for others its inputs cover", part)
    print()
    report("derived row disagrees with its inputs", mism)

    bad = stale + part
    print(f"\n{'CLEAN' if not bad else 'INCOMPLETE DERIVATIONS FOUND'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
