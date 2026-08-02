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
        for y in sorted(years):
            try:
                want = fn(*[s[y] for s in series])
            except Exception:
                continue
            if want is None:
                continue
            have = got.get(y)
            src = "overlay" if metric in applied else "formula"
            if have is None:
                out.append((ticker, metric, y, "STALE",
                            f"inputs present, row empty (implied {want:,.0f})"
                            if abs(want) > 1 else "inputs present, row empty"))
            elif abs(want) > 1e-9 and abs(have - want) / max(abs(want), 1e-9) > TOL:
                out.append((ticker, metric, y, "MISMATCH",
                            f"row {have:,.4g} vs inputs {want:,.4g}  [{src}]"))
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

    report("derived row empty while every input has a value", stale)
    print()
    report("derived row disagrees with its inputs", mism)

    print(f"\n{'CLEAN' if not stale else 'STALE DERIVATIONS FOUND'}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
