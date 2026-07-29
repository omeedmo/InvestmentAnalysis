"""
Which REITs can we read FFO and AFFO for, as the filer reported them?

Read-only. This writes no bindings and changes no behaviour; it answers a
sizing question — is the as-reported REIT universe forty names or four
hundred? — so the answer can decide whether the screener moves onto reported
figures or stays on proxies with honest labelling.

The approach separates the two halves of what the hand-written plugins do:

    company-specific   the label a filer prints for its subtotal
    generic            scale, columns, and the verification gate

Only the label is bespoke, and labels come from a small closed set: across
Simon, Realty Income, Federal Realty, AvalonBay and Equinix there are six
spellings between them. So the label is DISCOVERED by trying the library
below, and every candidate is then put through the same gate the plugins use —
the table's own net income must agree with the net income already known from
XBRL for that fiscal year.

That gate is the whole reason this is worth attempting. An earlier generic
extractor verified arithmetic self-consistency instead (do the adjustments sum
to the printed subtotal?) and was abandoned because Kimco's FY2013 filing
stacks quarterly and annual columns in one table: the quarter summed to its own
subtotal perfectly and was reported as the year. Against XBRL that same figure
is 74% adrift of the annual value, well outside the 25% tolerance, so it is now
rejected. The check moved from inside the text to outside it, which is what
makes discovery safe rather than merely convenient.

A pattern is accepted for a filer only when it clears the gate across several
filings AND the resulting series is free of unexplained discontinuities. What
it produces is a proposal to be reviewed and committed as a binding, never a
number rendered straight to a page.

Usage:
    python3 reit_discovery.py --tickers O FRT AVB EQIX SPG
    python3 reit_discovery.py --universe sp500 --limit 100 --json out.json
    python3 reit_discovery.py --cached          # offline, over cached filings
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import app
import screener
from company_plugins import _reit_ffo as R

# ── The label library ────────────────────────────────────────────────────────
# Every spelling seen across the filers read so far. The (?<![A-Za-z]) guard on
# the bare forms is what stops "FFO" matching inside "AFFO" a few lines below
# it, which carries a materially different number; the lookbehinds keep the
# Nareit subtotal apart from the Core/Normalized/Adjusted variants printed
# beside it, which are each a different definition.
_EXCL = r"(?<!Core )(?<!Normalized )(?<!Adjusted )(?<!Diluted )"
_NUM_AHEAD = r"\s*\$?\s*(?=\(?[\d,]{3,})"

FFO_PATTERNS: list[tuple[str, str]] = [
    ("ffo_available_common",
     _EXCL + r"(?<![A-Za-z])FFO available to common (?:stock|share)holders"),
    ("ffo_attributable_common",
     _EXCL + r"(?:NAREIT |Nareit )?(?<![A-Za-z])FFO attributable to common (?:stock|share)holders"),
    ("ffo_allocable_common",
     _EXCL + r"(?<![A-Za-z])FFO allocable to common (?:stock|share)holders"),
    ("funds_from_operations",
     _EXCL + r"Funds from operations(?! available)(?! per)(?! attributable)(?! allocable)"),
    ("nareit_ffo",
     _EXCL + r"(?:NAREIT|Nareit) FFO(?! per)"),
    ("ffo_bare",
     _EXCL + r"(?<![A-Za-z])FFO(?! per)(?! from)(?! adjustments)(?! allocable to (?:limited|dilutive))"),
]

AFFO_PATTERNS: list[tuple[str, str]] = [
    ("affo_available_common",
     r"(?<!Core )(?<!Normalized )AFFO available to common (?:stock|share)holders"),
    ("affo_attributable_common",
     r"(?<!Core )(?<!Normalized )AFFO attributable to common (?:stock|share)holders"),
    ("adjusted_ffo",
     r"(?<!Core )Adjusted funds from operations(?! per)"),
    ("affo_bare",
     r"(?<!Core )(?<!Normalized )(?<![A-Za-z])AFFO(?! per)(?! from)"),
]


def _compile(body: str) -> re.Pattern:
    return re.compile(body + _NUM_AHEAD, re.I)


# ── Quality gates ────────────────────────────────────────────────────────────

def _discontinuities(series: dict, limit: float = 0.6) -> list:
    """Year-on-year moves beyond `limit`, which need a corporate event to explain.

    Not fatal on its own — Realty Income genuinely doubled on the VEREIT merger
    — but a series with several is more likely a pattern drifting between two
    different subtotals than a company that kept transforming itself.
    """
    ys = sorted(series)
    out = []
    for a, b in zip(ys, ys[1:]):
        if series[a] and abs(series[b] / series[a] - 1) > limit:
            out.append((b, round(series[b] / series[a] - 1, 2)))
    return out


def _score(series: dict) -> dict:
    jumps = _discontinuities(series)
    ys = sorted(series)
    return {
        "years": len(series),
        "first": ys[0] if ys else None,
        "last": ys[-1] if ys else None,
        "jumps": jumps,
        # Contiguous runs matter more than raw count: a series with holes is
        # a pattern that only sometimes matches, which is a weaker signal than
        # one that matches every filing in a stretch.
        "gaps": sum(1 for a, b in zip(ys, ys[1:]) if int(b) - int(a) > 1),
    }


def discover_company(filings: list, get_text, ni_by_year: dict,
                     min_years: int = 3, max_jumps: int = 2) -> dict:
    """
    The best-scoring FFO and AFFO pattern for one filer, or {} for neither.

    Candidates are ranked by how much history they cover, then by how few
    discontinuities and holes they leave. A pattern covering fewer than
    `min_years` is not proposed at all: one or two verified filings is too
    little to tell a stable reading from a coincidence.
    """
    ctx = {"get_text": get_text, "min_year": 0, "net_income": None}
    out: dict = {}

    best = None
    for name, body in FFO_PATTERNS:
        try:
            series = R.walk_filings(filings, ctx, _compile(body), ncols=3,
                                    anchor_series=ni_by_year)
        except Exception:                      # noqa: BLE001 - a bad pattern is a miss
            continue
        if len(series) < min_years:
            continue
        sc = _score(series)
        if len(sc["jumps"]) > max_jumps:
            continue
        rank = (sc["years"], -len(sc["jumps"]), -sc["gaps"])
        if best is None or rank > best[0]:
            best = (rank, name, series, sc)
    if not best:
        return out
    _, name, ffo_series, sc = best
    out["ffo"] = {"pattern": name, "series": ffo_series, **sc}

    # AFFO is anchored on the FFO just verified, not on net income: an AFFO
    # reconciliation starts from FFO, and the two should agree to the dollar.
    ffo_re = _compile(dict(FFO_PATTERNS)[name])
    best_a = None
    for name_a, body_a in AFFO_PATTERNS:
        try:
            series = R.walk_filings(filings, ctx, _compile(body_a), ncols=3,
                                    anchor_re=ffo_re, anchor_series=ffo_series)
        except Exception:                      # noqa: BLE001
            continue
        if len(series) < min_years:
            continue
        sc_a = _score(series)
        if len(sc_a["jumps"]) > max_jumps:
            continue
        rank = (sc_a["years"], -len(sc_a["jumps"]), -sc_a["gaps"])
        if best_a is None or rank > best_a[0]:
            best_a = (rank, name_a, series, sc_a)
    if best_a:
        _, name_a, affo_series, sc_a = best_a
        out["affo"] = {"pattern": name_a, "series": affo_series, **sc_a}
    return out


# ── Driving it over a universe ───────────────────────────────────────────────

def run_ticker(ticker: str, max_filings: int = 16) -> dict:
    cik = screener.ticker_cik_map().get(ticker.upper())
    if not cik:
        return {"ticker": ticker, "error": "no CIK"}
    try:
        subs = app.fetch_submissions(str(cik).zfill(10))
        if str(subs.get("sic", "")) != "6798":
            return {"ticker": ticker, "error": f"not SIC 6798 ({subs.get('sic')})"}
        filings = app.all_filing_infos_from_submissions(subs, {"10-K"},
                                                        max_count=max_filings)
        if not filings:
            return {"ticker": ticker, "error": "no 10-Ks"}
        facts = app.fetch_company_facts(str(cik).zfill(10))
        ni = app.extract_annual_series(facts, app.METRIC_TAGS["net_income"])
        ni_by_year = {str(k)[:4]: v for k, v in (ni or {}).items() if v}
        if not ni_by_year:
            return {"ticker": ticker, "error": "no XBRL net income to verify against"}
        found = discover_company(
            filings, lambda f: app.filing_text_cached(f["url"]), ni_by_year)
    except Exception as e:                     # noqa: BLE001
        return {"ticker": ticker, "error": f"{type(e).__name__}: {e}"[:120]}
    return {"ticker": ticker, "cik": cik, **found}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--universe")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--json")
    args = ap.parse_args()

    tickers = list(args.tickers)
    if args.universe:
        tickers += screener.get_universe(args.universe)[: args.limit]
    tickers = list(dict.fromkeys(tickers))[: args.limit]
    if not tickers:
        ap.error("give --tickers or --universe")

    results, covered = [], 0
    for tk in tickers:
        r = run_ticker(tk)
        results.append(r)
        if r.get("ffo"):
            covered += 1
            f = r["ffo"]
            a = r.get("affo")
            print(f"  {tk:6} FFO {f['years']:2}yr {f['first']}-{f['last']} "
                  f"[{f['pattern']}] jumps={len(f['jumps'])} gaps={f['gaps']}"
                  + (f"   AFFO {a['years']:2}yr [{a['pattern']}]" if a else ""))
        elif r.get("error"):
            print(f"  {tk:6} — {r['error']}")
        else:
            print(f"  {tk:6} — no pattern cleared the gate")

    eligible = [r for r in results if not r.get("error")]
    print(f"\n=== {covered} of {len(eligible)} REITs read as-reported FFO "
          f"({len(results) - len(eligible)} excluded before testing)")
    print(f"    of those, {sum(1 for r in results if r.get('affo'))} also read AFFO")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"    wrote {args.json}")


if __name__ == "__main__":
    main()
