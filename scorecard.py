"""
Metrics bound to the lines a company actually filed.

`statement_store` reads each filing's own statements. This turns those into a
scorecard, and the difference from the old METRIC_TAGS approach is the whole
point of the rebuild:

  METRIC_TAGS said  "premiums_earned is one of these five element names,
                     take whichever hits first, any year, any context".
  A binding says    "premiums_earned is us-gaap:PremiumsEarnedNet, under the
                     'Insurance and Other' segment column, from 2009 onward".

The cascading-fallback style is what silently produces wrong numbers: it cannot
tell a segment subtotal from a consolidated one, and when a company switches
tags mid-history it quietly changes which concept it is reporting without
saying so. A binding is explicit about element, dimension and years, and when
no binding covers a year the cell is blank rather than guessed.

Three tiers, matching how the gaps actually distribute — the coverage audit
found them to be sector-systematic far more than company-idiosyncratic:

  global    metrics essentially every filer reports
  sector    what a sector reports and others don't (REIT NOI, bank NII,
            insurance premiums and float)
  company   what this filer alone reports (Charter's franchise intangibles,
            Berkshire's B-equivalent share count)

Company overrides sector overrides global, and every resolved value carries the
filing, element, dimension and the label the company printed, so any number on
screen can be traced back to the line it came from.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import statement_store as ss

BINDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bindings")

# Statement preference when the same element appears in more than one place
# (cash shows up on the balance sheet and again in the cash flow statement).
_STATEMENT_RANK = [
    (re.compile(r"balance sheet|financial position", re.I), 0),
    (re.compile(r"statements? of (earnings|operations|income)", re.I), 1),
    (re.compile(r"cash flow", re.I), 2),
    (re.compile(r"equity|shareholders", re.I), 3),
    (re.compile(r"comprehensive", re.I), 4),
]


def _rank(statement_name: str) -> int:
    for pat, r in _STATEMENT_RANK:
        if pat.search(statement_name):
            return r
    return 9


def _period_year(period: str) -> Optional[int]:
    m = re.match(r"^[A-Z][a-z]{2}\.? \d{1,2}, (\d{4})$", period)
    return int(m.group(1)) if m else None


# ── Fact base ────────────────────────────────────────────────────────────────

def _is_restatement(current: float, earlier: float, tolerance: float = 0.001) -> bool:
    """
    Did the company actually restate this figure, or does it merely print
    differently?

    Two figures of the same magnitude and opposite sign are a presentation
    difference, not a restatement — a line taken from the income statement in
    one filing and from the cash flow reconciliation in another legitimately
    carries opposite signs. Flagging those as restatements marked every year of
    Berkshire's investment gains as restated, which is noise that would teach a
    reader to ignore the marker entirely.
    """
    if abs(current) < 1e-9 and abs(earlier) < 1e-9:
        return False
    scale = max(abs(current), abs(earlier))
    if abs(abs(current) - abs(earlier)) <= scale * tolerance:
        return False          # same magnitude: sign convention only
    return True


def _facts_index(facts: Optional[dict]) -> dict:
    """{(concept, end): raw signed value} from companyfacts, longest span wins."""
    if not facts:
        return {}
    from datetime import date
    idx: dict[tuple, float] = {}
    span_of: dict[tuple, int] = {}
    for concept, doc in ((facts.get("facts") or {}).get("us-gaap") or {}).items():
        for unit, entries in (doc.get("units") or {}).items():
            if unit not in ("USD", "shares", "USD/shares"):
                continue
            for e in entries:
                end = e.get("end")
                if not end:
                    continue
                span = 0
                if e.get("start"):
                    try:
                        y1, m1, d1 = (int(x) for x in e["start"].split("-"))
                        y2, m2, d2 = (int(x) for x in end.split("-"))
                        span = (date(y2, m2, d2) - date(y1, m1, d1)).days
                    except Exception:
                        span = 0
                key = (concept, end)
                if key not in idx or span > span_of.get(key, -1):
                    idx[key] = e["val"]
                    span_of[key] = span
    return idx


def build_factbase(cik: str, filings: list, years: int = 15,
                   verify: bool = True, facts: Optional[dict] = None) -> dict:
    """
    {fiscal_year: {(element, dimension): fact}} across a filer's history.

    Each 10-K restates two or three prior years, so most years are reported by
    several filings. The newest filing wins, which is what the reader wants:
    the company's current view of its own past. The superseded figure is kept
    on the fact as `restated_from` rather than dropped, because a restatement
    is information, not noise.

    A filing whose statements fail their own tie-out is skipped entirely —
    better a gap than a number we cannot stand behind.
    """
    factbase: dict[int, dict] = {}
    provenance: dict[str, dict] = {}
    skipped: list[dict] = []
    # A rendered statement applies negatedLabel, so its printed sign is a
    # presentation choice — Berkshire's "Investment gains (losses)" prints with
    # the opposite sign to the underlying fact in every year of its history.
    # Take structure, order and labels from the rendering, but take the SIGNED
    # VALUE from the raw XBRL wherever companyfacts has it. Dimensional lines
    # have no raw counterpart (that is the whole reason we parse statements),
    # so those keep the printed value.
    raw = _facts_index(facts)

    fys = [int(f["fiscal_year"]) for f in filings if f.get("fiscal_year")]
    cutoff = (max(fys) - years + 1) if fys else 0

    for filing in filings:                      # newest first
        fy = filing.get("fiscal_year")
        if not fy or int(fy) < cutoff:
            continue
        try:
            statements = ss.filing_statements(cik, filing["accession"])
        except Exception as e:                  # noqa: BLE001
            skipped.append({"fy": fy, "reason": f"fetch failed: {e}"})
            continue
        if not statements:
            skipped.append({"fy": fy, "reason": "no statements parsed"})
            continue
        if verify:
            problems = ss.check_totals(statements)
            if problems:
                skipped.append({"fy": fy, "reason": "tie-out failed",
                                "detail": problems[:2]})
                continue

        provenance[str(fy)] = {"accession": filing["accession"],
                               "filed": filing.get("filing_date", "")}

        for name, st in sorted(statements.items(), key=lambda kv: _rank(kv[0])):
            if "parenthetical" in name.lower():
                continue
            for row in st["rows"]:
                if row["abstract"] or not row["values"]:
                    continue
                key = (row["element"], row.get("dimension"))
                for period, value in row["values"].items():
                    year = _period_year(period)
                    if year is None or year < cutoff - 1:
                        continue
                    if row.get("opening_balance"):
                        continue
                    slot = factbase.setdefault(year, {})
                    if key in slot:
                        prior = slot[key]
                        if (prior["from_fy"] != str(fy)
                                and "restated_from" not in prior
                                and _is_restatement(prior["value"], value)):
                            # An older filing reported this year differently.
                            prior["restated_from"] = {
                                "value": value, "in_10k_fy": str(fy)}
                        continue
                    signed = value
                    source = "rendering"
                    if raw and row["element"].startswith("us-gaap:") \
                            and not row.get("dimension"):
                        iso = ss._period_to_iso(period)
                        rk = (row["element"].split(":", 1)[1], iso)
                        if iso and rk in raw:
                            ref = raw[rk]
                            # Only adopt the raw sign when the magnitudes agree;
                            # a genuine mismatch is a parse problem, not a sign
                            # convention, and must not be papered over here.
                            if abs(ref) > 0 and abs(abs(ref) - abs(value)) <= \
                                    max(1.0, abs(ref) * 0.005):
                                signed = ref
                                source = "xbrl"
                    slot[key] = {
                        "value":     signed,
                        "printed":   value,
                        "value_source": source,
                        "element":   row["element"],
                        "dimension": row.get("dimension"),
                        "label":     row["label"],
                        "statement": name,
                        "from_fy":   str(fy),
                        "accession": filing["accession"],
                    }

    return {"years": factbase, "provenance": provenance, "skipped": skipped}


# ── Bindings ─────────────────────────────────────────────────────────────────

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_bindings(sector: Optional[str] = None,
                  ticker: Optional[str] = None) -> list:
    """Global, then sector, then company — later tiers override earlier ones."""
    tiers = [("global", os.path.join(BINDING_DIR, "global.json"))]
    if sector:
        tiers.append(("sector", os.path.join(BINDING_DIR, "sectors", f"{sector}.json")))
    if ticker:
        tiers.append(("company", os.path.join(BINDING_DIR, "companies",
                                              f"{ticker.upper()}.json")))

    merged: dict[str, dict] = {}
    order: list[str] = []
    for tier, path in tiers:
        doc = _read_json(path)
        if not doc:
            continue
        for spec in doc.get("metrics", []):
            metric = spec.get("metric")
            if not metric:
                continue
            spec = {**spec, "tier": tier}
            if metric not in merged:
                order.append(metric)
            merged[metric] = spec
    return [merged[m] for m in order]


def _bindings_for_year(spec: dict, year: int) -> list:
    """
    The candidate bindings that apply to this year, in author order.

    Companies genuinely switch tags mid-history — Berkshire moved operating
    cash flow from NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
    to the plain concept in 2015, and bank charge-offs crossed three concepts
    between 2009 and 2021. Two mechanisms cover that: explicit `from`/`to`
    years when the switch date is known, and otherwise an ordered list of named
    alternatives.

    This is still not the old cascading-tag guessing. Every candidate is
    written down by a human, the list is short and closed, and the resolved
    cell records exactly which element it came from — so "which concept is
    this number?" always has an answer on screen.
    """
    out = []
    for b in spec.get("bind", []):
        lo, hi = b.get("from"), b.get("to")
        if lo is not None and year < lo:
            continue
        if hi is not None and year > hi:
            continue
        out.append(b)
    return out


def _match_fact(slot: dict, b: dict) -> Optional[dict]:
    """
    Find the fact a binding names.

    `dimension` matches a segment section exactly; `dimension_pattern` matches
    it by regex, which is what lets a SECTOR binding work across filers who
    label the same segment differently ("Insurance and Other" at Berkshire vs
    "Insurance Operations" elsewhere). Omitting both means the consolidated
    figure, and that is the default precisely because a segment column silently
    standing in for a consolidated total is the error this design exists to
    prevent.
    """
    element = b["element"]
    want_stmt = b.get("statement")

    def ok(fact) -> bool:
        # A binding may require which statement the line came from. This is not
        # decoration: the same element name can appear as a BALANCE SHEET stock
        # and an INCOME STATEMENT flow, and Berkshire's
        # LiabilityForClaimsAndClaimsAdjustmentExpensePropertyCasualtyLiability
        # does exactly that. Binding it without the constraint took the $63.8B
        # reserve balance as if it were the year's loss expense and produced a
        # combined ratio of 2.18 — a stock/flow mixup is the most dangerous
        # error available here, because the number still looks like a number.
        return (not want_stmt) or bool(re.search(want_stmt, fact["statement"], re.I))

    if b.get("dimension_pattern"):
        pat = re.compile(b["dimension_pattern"], re.I)
        matches = [f for (el, dim), f in slot.items()
                   if el == element and dim and pat.search(dim) and ok(f)]
        if len(matches) == 1:
            return matches[0]
        # Ambiguous: several segments matched, so we cannot know which the
        # metric meant. Refuse rather than pick one.
        return None
    fact = slot.get((element, b.get("dimension")))
    return fact if (fact is not None and ok(fact)) else None


def resolve(factbase: dict, bindings: list) -> dict:
    """
    {metric: {"label", "tier", "years": {year: {value, label, element, ...}}}}

    Only bound, present lines produce values. Nothing is inferred, and a year
    with no binding produces no cell.
    """
    years_data = factbase["years"]
    out: dict[str, dict] = {}

    for spec in bindings:
        metric = spec["metric"]
        series: dict[str, dict] = {}
        for year in sorted(years_data):
            fact = None
            for b in _bindings_for_year(spec, year):
                fact = _match_fact(years_data[year], b)
                if fact is not None:
                    break
            if fact is None:
                continue
            value = fact["value"]
            if b.get("negate"):
                value = -value
            series[str(year)] = {
                "value":       value,
                "as_reported": fact["label"],
                "element":     fact["element"],
                "dimension":   fact["dimension"],
                "statement":   fact["statement"],
                "from_10k_fy": fact["from_fy"],
                "accession":   fact["accession"],
                "restated_from": fact.get("restated_from"),
            }
        if series:
            out[metric] = {
                "label":   spec.get("label", metric),
                "tier":    spec.get("tier", "global"),
                "section": spec.get("section", ""),
                "note":    spec.get("note", ""),
                "years":   series,
            }
    return out


# ── Derived metrics ──────────────────────────────────────────────────────────

def compute_derived(resolved: dict, bindings: list) -> dict:
    """
    Ratios and differences built from resolved lines.

    A derived metric only computes for a year where every input resolved, so a
    ratio can never be built on a silently-missing denominator.
    """
    import company_templates            # reuses the audited AST evaluator

    values_by_year: dict[str, dict] = {}
    for metric, data in resolved.items():
        for year, cell in data["years"].items():
            values_by_year.setdefault(year, {})[metric] = cell["value"]

    for spec in bindings:
        expr = spec.get("expr")
        if not expr:
            continue
        metric = spec["metric"]
        # Some components legitimately do not exist in some years — Berkshire
        # only began breaking out retroactive reinsurance reserves in 2016, and
        # before that they sat inside the main reserve line. Treating those as
        # zero is correct; treating a component that FAILED TO RESOLVE as zero
        # would silently understate the total. So the author must name which
        # ones are genuinely optional, per metric, and everything else still
        # blanks the whole cell when missing.
        optional = set(spec.get("zero_if_absent", []))
        series: dict[str, dict] = {}
        for year, vals in values_by_year.items():
            if optional:
                vals = {**{k: 0.0 for k in optional if vals.get(k) is None}, **vals}
            v = company_templates.eval_expr(expr, vals)
            if v is None:
                continue
            series[year] = {"value": v, "as_reported": None, "element": None,
                            "dimension": None, "statement": "derived",
                            "from_10k_fy": None, "accession": None,
                            "restated_from": None, "expr": expr}
        if series:
            resolved[metric] = {
                "label":   spec.get("label", metric),
                "tier":    spec.get("tier", "global"),
                "section": spec.get("section", ""),
                "note":    spec.get("note", ""),
                "derived": True,
                "years":   series,
            }
            # Feed the result back so later metrics can build on it. Berkshire's
            # unlevered net tangible assets need total_debt, which is itself
            # derived (the sum of its segment debt columns), and float/equity
            # needs the derived float. Bindings are evaluated global -> sector
            # -> company, so a later tier can always build on an earlier one.
            for year, cell in series.items():
                values_by_year.setdefault(year, {})[metric] = cell["value"]
    return resolved


def build(cik: str, filings: list, sector: Optional[str] = None,
          ticker: Optional[str] = None, years: int = 15,
          facts: Optional[dict] = None) -> dict:
    """Fact base -> bindings -> resolved scorecard, with provenance intact."""
    fb = build_factbase(cik, filings, years=years, facts=facts)
    bindings = load_bindings(sector, ticker)
    resolved = resolve(fb, bindings)
    resolved = compute_derived(resolved, bindings)
    return {
        "metrics":    resolved,
        "provenance": fb["provenance"],
        "skipped":    fb["skipped"],
        "years":      sorted(fb["years"]),
    }
