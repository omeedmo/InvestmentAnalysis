"""
Coverage audit — find material facts a company reports that NO metric consumes.

The app maps XBRL tags to canonical metrics via app.METRIC_TAGS. That mapping is
necessarily incomplete: every company tags things a little differently, and a
material line item with a tag we don't list is silently dropped. The failure is
invisible — the metric still renders, just wrong. (CHTR reports $67.5B of
IndefiniteLivedFranchiseRights, 45% of its assets, which nothing consumed, so
UNTA stripped $0.5B of intangibles instead of ~$68B.)

This audit inverts the question: instead of asking "did we find tag X?", it asks
"what did this company report that we never looked at?" — turning silent
wrongness into a reviewable, ranked list.

Deterministic; no LLM. Usage:
    python3 coverage_audit.py CHTR JPM SPG
    python3 coverage_audit.py --universe sp500 --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import app
import screener


# ── What counts as "consumed" ────────────────────────────────────────────────

def mapped_tags(ticker: str = None) -> set[str]:
    """
    Every tag any metric maps to, plus tags used only in local fallbacks.

    When a ticker is given, the tags its authored bindings consume count too.
    A company that has been moved onto the bindings path reads its statements
    by element name there, not through METRIC_TAGS, so without this the audit
    reports the very lines someone just finished binding as blind spots —
    Lumen's net PP&E was flagged as its third-largest gap immediately after
    being bound. The check is per-ticker on purpose: a tag consumed by
    Berkshire's bindings is still a genuine gap for everyone else.
    """
    out: set[str] = set()
    for tags in app.METRIC_TAGS.values():
        for t in (tags or []):
            out.add(t)
    # Tags consumed by targeted fallbacks rather than METRIC_TAGS lists.
    out |= {"InterestIncomeExpenseNet", "InterestIncomeExpenseNonoperatingNet"}

    if ticker:
        import company_templates
        import scorecard
        binding = scorecard._read_json(os.path.join(
            scorecard.BINDING_DIR, "companies",
            f"{ticker.upper().replace('.', '-')}.json")) or {}
        for spec in scorecard.load_bindings(binding.get("sector"), ticker):
            for b in spec.get("bind", []):
                el = b.get("element") or ""
                out.add(el.split(":", 1)[-1])       # drop the namespace prefix
        # Tags a company template adds to the global mapping for this filer
        # only — the other per-company way of consuming a tag, and equally
        # invisible if the audit reads METRIC_TAGS alone.
        tmpl = company_templates.load_template(ticker) or {}
        for tags in (tmpl.get("add_tags") or {}).values():
            for t in (tags or []):
                out.add(t.split(":", 1)[-1])
    return out


# ── Filtering ────────────────────────────────────────────────────────────────
# A tag being unconsumed is only a *problem* if leaving it out corrupts a
# metric. Two large classes are unconsumed but harmless:
#
#   1. Footnote/disclosure detail — maturity schedules, fair-value hierarchies,
#      tax reconciliations, roll-forwards. Never belonged on the face of a
#      statement.
#   2. Roll-up members — components of a total we already capture (equity
#      components roll into equity; expense subtotals into operating income;
#      a Gross variant duplicates the mapped Net).
#
# Excluding both is what separates signal from noise. Anything genuinely
# ambiguous is deliberately left IN: a false positive costs a moment's review,
# a false negative is exactly the silent bug this exists to catch.

DISCLOSURE_PATTERNS = [
    r"^Deferred(Tax|IncomeTax)",          # deferred-tax component detail
    r"IncomeTaxReconciliation",
    r"FairValue",
    r"Maturit",
    r"AccumulatedDepreciation",
    r"AccumulatedAmortization",
    r"FutureAmortizationExpense",
    r"AmortizationExpense(Next|Year|After)",
    r"^ShareBasedCompensation.*(Vested|Forfeit|Grant|Exercise|Outstanding|Nonvested)",
    r"Concentration",
    r"^Allowance.*(Recover|WriteOff|Writeoff|Adjustment|Provision)",
    r"MinimumPayments",
    r"^Business(Acquisition|Combination)",
    r"RelatedParty",
    r"^Restructuring.*(Reserve|Accrual)",
    r"PeriodIncreaseDecrease",
    r"WeightedAverage",
    r"^Sale.*Leaseback",
    r"UnrecognizedTaxBenefits",
    r"^EquityMethodInvestment.*(Summarized|Ownership)",
    r"Pledged",
    r"^Defined(Benefit|Contribution)",
    r"^Debt(Instrument|Conversion)",
    r"^LineOfCredit",
    r"Covenant",
    r"^SegmentReporting",
    r"^ContractualObligation",
    r"^CashCashEquivalentsRestricted",
    r"^LesseeOperatingLease(Liability)?(Payments|Maturity)",
    # Supplemental cash-flow disclosure (cash taxes/interest paid) — not a
    # statement line item; the accrual figures are mapped separately.
    r"^IncomeTaxesPaid", r"^IncomeTaxPaid", r"^InterestPaid",
    # Tax-footnote components: current/deferred by jurisdiction, and the
    # domestic/foreign pre-tax split. The consolidated tax expense is mapped.
    r"^(Current|Deferred)(Federal|Foreign|State)",
    r"^CurrentIncomeTaxExpenseBenefit", r"^DeferredIncomeTaxExpenseBenefit",
    r"BeforeIncomeTaxes.*(Foreign|Domestic)",
    r"^UndistributedEarningsOfForeign",
    # Variants/subtotals of net income — net_income itself is mapped.
    r"^IncomeLossFromContinuingOperations",
    r"^NetIncomeLossAvailableToCommonStockholders",
    r"^NetIncomeLossAttributableToParentDiluted",
    r"^OtherComprehensiveIncomeLoss",
    # Roll-forward / equity-movement detail.
    r"^GoodwillAcquiredDuringPeriod",
    r"^AdjustmentsToAdditionalPaidInCapital",
    r"^ContractWithCustomerLiabilityRevenueRecognized",
    r"^StockRepurchaseProgramAuthorizedAmount",
    r"^LeaseCost$",
    r"AmortizedCostBasis$",
]

# Cash-flow movement lines: individually unconsumed, but the section totals and
# the specific flows the app needs (capex, buybacks, dividends) are mapped.
CASHFLOW_DETAIL_PATTERNS = [
    r"^ProceedsFrom", r"^PaymentsFor", r"^PaymentsTo", r"^RepaymentsOf",
    r"^IncreaseDecreaseIn", r"^StockIssuedDuringPeriod",
    r"^StockRepurchasedDuringPeriod", r"^NetCashProvidedByUsedIn",
]

# Aggregates that are either definitionally derivable or already captured via
# their parent (so they aren't standalone blind spots).
ROLLUP_TAGS = {
    # Balance-sheet totals
    "LiabilitiesAndStockholdersEquity",
    "AssetsNoncurrent", "LiabilitiesNoncurrent",
    # Equity components — equity itself is mapped
    "AdditionalPaidInCapitalCommonStock", "AdditionalPaidInCapital",
    "RetainedEarningsAccumulatedDeficit", "CommonStockValue",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
    "MinorityInterest",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CommonStockSharesAuthorized", "CommonStockParOrStatedValuePerShare",
    # Income-statement subtotals — operating income / net income are mapped
    "CostsAndExpenses", "OperatingExpenses", "BenefitsLossesAndExpenses",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "ProfitLoss",
    "ComprehensiveIncomeNetOfTax",
    "ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest",
    "OtherComprehensiveIncomeLossNetOfTax",
}

_DISCLOSURE_RE = re.compile("|".join(DISCLOSURE_PATTERNS))
_CASHFLOW_RE   = re.compile("|".join(CASHFLOW_DETAIL_PATTERNS))


def skip_reason(tag: str, consumed: set[str]) -> str | None:
    """Why this unconsumed tag is not a real blind spot (None = it IS one)."""
    if tag in ROLLUP_TAGS:
        return "rollup"
    if _DISCLOSURE_RE.search(tag):
        return "disclosure"
    if _CASHFLOW_RE.search(tag):
        return "cashflow-detail"
    # A Gross variant whose corresponding Net is already mapped is a duplicate
    # measurement of the same balance, not a missing one.
    if tag.endswith("Gross"):
        stem = tag[: -len("Gross")]
        if f"{stem}Net" in consumed or stem in consumed:
            return "gross-of-mapped-net"
    return None


# ── Fact extraction ──────────────────────────────────────────────────────────

def latest_annual_facts(facts: dict) -> dict[str, dict]:
    """
    {tag: {value, end, instant}} using each tag's most recent 10-K FY fact.

    Only 'frame'-bearing entries are used: SEC assigns a frame to the clean,
    consolidated (undimensioned) value, which filters out segment/member
    breakdowns that would otherwise dominate by count.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out: dict[str, dict] = {}
    for tag, concept in gaap.items():
        entries = concept.get("units", {}).get("USD", [])
        best = None
        for e in entries:
            if e.get("form") not in {"10-K", "10-K/A", "20-F", "20-F/A"}:
                continue
            if not e.get("frame"):
                continue
            end = e.get("end", "")
            if not end or e.get("val") is None:
                continue
            if best is None or end > best["end"]:
                best = {"value": float(e["val"]), "end": end,
                        "instant": not e.get("start")}
        if best:
            out[tag] = best
    return out


# ── The audit ────────────────────────────────────────────────────────────────

def audit(facts: dict, threshold: float = 0.01, include_noise: bool = False,
          ticker: str = None) -> dict:
    """
    Flag material facts no metric consumes.

    Balance-sheet (instant) facts are scored against total assets; flow
    (duration) facts against revenue, falling back to assets when a filer
    reports no revenue line (banks/REITs often don't tag one cleanly).
    """
    consumed = mapped_tags(ticker)
    all_facts = latest_annual_facts(facts)

    assets  = (all_facts.get("Assets") or {}).get("value") or 0.0
    revenue = 0.0
    for rt in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
               "SalesRevenueNet"):
        if all_facts.get(rt):
            revenue = all_facts[rt]["value"]
            break

    flagged, suppressed = [], []
    for tag, f in all_facts.items():
        if tag in consumed:
            continue
        base = assets if f["instant"] else (revenue or assets)
        if not base:
            continue
        pct = abs(f["value"]) / abs(base)
        if pct < threshold:
            continue
        row = {
            "tag": tag,
            "value": f["value"],
            "end": f["end"],
            "kind": "balance" if f["instant"] else "flow",
            "pct_of_base": round(pct * 100, 2),
            "base": "assets" if f["instant"] else ("revenue" if revenue else "assets"),
        }
        reason = skip_reason(tag, consumed)
        if reason and not include_noise:
            row["skipped_as"] = reason
            suppressed.append(row)
        else:
            if reason:
                row["skipped_as"] = reason
            flagged.append(row)
    flagged.sort(key=lambda x: x["pct_of_base"], reverse=True)
    suppressed.sort(key=lambda x: x["pct_of_base"], reverse=True)
    return {
        "assets": assets,
        "revenue": revenue,
        "facts_examined": len(all_facts),
        "flagged": flagged,
        "suppressed": suppressed,
    }


# ── History audit: era transitions ───────────────────────────────────────────
# The unmapped-fact audit above is point-in-time. But the failure mode that has
# actually bitten most often is temporal: a company switches tags mid-history
# (CHTR InterestExpense -> InterestIncomeExpenseNet in FY2014; ACN
# PaymentsOfDividendsCommonStock -> PaymentsOfOrdinaryDividends in FY2023; BAC
# charge-offs across three tagging eras). The metric then just stops, and a
# chart silently truncates. This finds those discontinuities by comparing each
# metric's populated years against the years the company actually filed.

def audit_history(facts: dict, min_run: int = 3) -> dict:
    """Metrics that cover part of a company's filing history but stop or gap."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei", {})

    def years_for(tags: list[str]) -> set[int]:
        out: set[int] = set()
        for t in tags or []:
            concept = gaap.get(t) or dei.get(t)
            if not concept:
                continue
            for unit_entries in concept.get("units", {}).values():
                for e in unit_entries:
                    if e.get("form") in {"10-K", "10-K/A", "20-F", "20-F/A"} \
                       and e.get("frame") and e.get("end"):
                        out.add(int(e["end"][:4]))
        return out

    # Filing years = years the company reported total assets (a universal tag).
    filed = years_for(["Assets"]) or years_for(["StockholdersEquity"])
    if not filed:
        return {"filed_years": [], "issues": []}
    fmin, fmax = min(filed), max(filed)

    issues = []
    for metric, tags in app.METRIC_TAGS.items():
        if not tags:
            continue
        ys = years_for(tags) & set(range(fmin, fmax + 1))
        if len(ys) < min_run:
            continue                      # never really covered; not a regression
        missing = sorted(set(range(min(ys), fmax + 1)) - ys)
        if not missing:
            continue
        # A trailing gap (metric stops and never resumes) is the classic
        # tag-switch signature and the most damaging, so call it out.
        trailing = [y for y in missing if y > max(ys)]
        issues.append({
            "metric": metric,
            "covered": f"{min(ys)}-{max(ys)}",
            "missing_years": missing,
            "stops_early": bool(trailing) or max(ys) < fmax,
            "last_covered": max(ys),
            "filing_through": fmax,
        })
    issues.sort(key=lambda i: (not i["stops_early"], -len(i["missing_years"])))
    return {"filed_years": [fmin, fmax], "issues": issues}


def audit_ticker(ticker: str, threshold: float = 0.01,
                 include_noise: bool = False) -> dict:
    cik = screener.ticker_cik_map().get(ticker.upper())
    if not cik:
        return {"ticker": ticker, "error": "CIK not found"}
    try:
        facts = app.fetch_company_facts(str(cik).zfill(10))
    except Exception as e:
        return {"ticker": ticker, "error": f"fetch failed: {e}"}
    res = audit(facts, threshold, include_noise, ticker=ticker)
    res["ticker"] = ticker.upper()
    res["history"] = audit_history(facts)
    return res


# ── Reporting ────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e9:  return f"${v/1e9:,.1f}B"
    if a >= 1e6:  return f"${v/1e6:,.1f}M"
    return f"${v:,.0f}"


def print_report(res: dict, top: int = 15, show_suppressed: bool = False) -> None:
    tk = res.get("ticker", "?")
    if res.get("error"):
        print(f"\n=== {tk}: ERROR {res['error']}")
        return
    fl = res["flagged"]
    print(f"\n=== {tk}  (assets {_fmt(res['assets'])}, "
          f"{res['facts_examined']} facts examined, {len(fl)} blind spots, "
          f"{len(res.get('suppressed', []))} filtered)")
    if not fl:
        print("    ✓ no material unmapped facts")
    for f in fl[:top]:
        print(f"    {f['pct_of_base']:6.2f}% of {f['base']:<7} {_fmt(f['value']):>12}  "
              f"{f['kind']:<7} {f['tag']}  [{f['end']}]")
    if len(fl) > top:
        print(f"    … {len(fl)-top} more")
    hist = res.get("history", {})
    stops = [i for i in hist.get("issues", []) if i["stops_early"]]
    if stops:
        fy = hist.get("filed_years", [None, None])
        print(f"    --- era gaps (filing {fy[0]}-{fy[1]}) ---")
        for i in stops[:top]:
            print(f"    {i['metric']:<32} covered {i['covered']}, "
                  f"missing {len(i['missing_years'])}y through {i['filing_through']}")
    if show_suppressed:
        print("    --- filtered (not blind spots) ---")
        for f in res.get("suppressed", [])[:top]:
            print(f"    {f['pct_of_base']:6.2f}% {_fmt(f['value']):>12}  "
                  f"{f['skipped_as']:<20} {f['tag']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit XBRL coverage gaps.")
    ap.add_argument("tickers", nargs="*", help="tickers to audit")
    ap.add_argument("--universe", help="screener universe key (e.g. sp500)")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="materiality cutoff as a fraction of the base (default 0.01 = 1%%)")
    ap.add_argument("--include-noise", action="store_true",
                    help="don't filter disclosure/roll-up tags")
    ap.add_argument("--show-filtered", action="store_true",
                    help="also list what was filtered out, with the reason")
    ap.add_argument("--sleep", type=float, default=0.5, help="delay between SEC calls")
    ap.add_argument("--json", help="write full results to this path")
    ap.add_argument("--rank", action="store_true",
                    help="aggregate: rank the tags most often flagged across companies")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]
    if args.universe:
        tickers += [t for t in screener.get_universe(args.universe)][: args.limit]
    if not tickers:
        ap.error("give tickers or --universe")
    tickers = list(dict.fromkeys(tickers))[: args.limit] if args.universe else tickers

    results = []
    for i, tk in enumerate(tickers):
        res = audit_ticker(tk, args.threshold, args.include_noise)
        results.append(res)
        print_report(res, show_suppressed=args.show_filtered)
        sys.stdout.flush()
        if i < len(tickers) - 1:
            time.sleep(args.sleep)

    if args.rank:
        counts: dict[str, dict] = {}
        for r in results:
            for f in r.get("flagged", []):
                c = counts.setdefault(f["tag"], {"companies": 0, "max_pct": 0.0, "examples": []})
                c["companies"] += 1
                c["max_pct"] = max(c["max_pct"], f["pct_of_base"])
                if len(c["examples"]) < 4:
                    c["examples"].append(f"{r['ticker']}:{f['pct_of_base']}%")
        print("\n\n=== MOST-FLAGGED TAGS ACROSS "
              f"{len([r for r in results if not r.get('error')])} COMPANIES ===")
        for tag, c in sorted(counts.items(),
                             key=lambda kv: (kv[1]["companies"], kv[1]["max_pct"]),
                             reverse=True)[:40]:
            print(f"    {c['companies']:3d} cos  max {c['max_pct']:6.2f}%  {tag}"
                  f"   ({', '.join(c['examples'])})")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
