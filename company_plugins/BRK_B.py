"""
Berkshire Hathaway (BRK.A / BRK.B) company plugin.

Declared by company_templates/BRK-B.json via `"plugin": "BRK_B"`. Everything
Berkshire-specific lives here so that app.py holds no per-company knowledge.

Why Berkshire needs code rather than declarative config:

  * It tags NO consolidated debt concept in XBRL at all — debt is reported by
    segment (BNSF, BHE, insurance/other) in 10-K text and segment tables only.
  * Its XBRL share counts, weighted diluted shares and EPS are on a CLASS-A
    basis (~1.4M shares, ~$40,000 EPS), which is useless for per-B-share math.
    Shares outstanding here are B-EQUIVALENTS: class-A x 1500 + class-B, which
    is how Berkshire itself presents them and how market cap reconciles.
  * Its cash is double-counted by the tagged data (a combined cash tag plus a
    separate short-term-T-bill tag), so cash/short-term investments are taken
    from the filing text instead.
  * Operating earnings — the figure Buffett tells shareholders to judge the
    company on, as opposed to GAAP net income dominated by mark-to-market
    swings since ASU 2016-01 — appear only in an MD&A table, never as a tag.

None of that can be expressed as a tag mapping, so it is real Python. The
core app reaches it only through the generic hooks in company_templates.
"""
from __future__ import annotations

import re
from datetime import datetime


def normalize_to_fiscal_years(*args, **kwargs):
    """Deferred import: the plugin is loaded during a request, so app is
    already fully initialised, but importing it at module scope would make
    the dependency circular."""
    import app
    return app.normalize_to_fiscal_years(*args, **kwargs)


# ── Extractors ───────────────────────────────────────────────────────────────

def extract_berkshire_equivalent_b_shares(text: str, filing_date: str = "") -> dict[str, float]:
    """
    Extract class-A-equivalent share counts from BRK 10-K text.
    Returns {fiscal_year_end_date: class_B_equivalent_shares}.
    """
    normalized = re.sub(r"\s+", " ", text.lower())
    results: dict[str, float] = {}

    # Pattern 1: "on an equivalent class a common stock basis, there were X shares outstanding as of DATE"
    # Gives one specific point-in-time entry per occurrence.
    p1 = (
        r"on an equivalent class a common stock basis,\s*there were\s*([0-9,]+)\s*shares?\s*outstanding"
        r"\s*as of\s*([a-z]+ \d{1,2},?\s*\d{4})"
    )
    for m in re.finditer(p1, normalized, re.IGNORECASE):
        try:
            shares_a = float(m.group(1).replace(",", ""))
            date_str = re.sub(r"\s+", " ", m.group(2)).strip()
            period = datetime.strptime(date_str, "%B %d, %Y").strftime("%Y-%m-%d")
            results[period] = shares_a * 1500
        except ValueError:
            pass

    # Pattern 2: table row "average equivalent class a shares outstanding X X X"
    # followed by 1-5 year columns. Non-greedy up to the next word sequence.
    # NB: Python repeating groups only capture the last iteration, so we capture
    # the entire trailing chunk and extract numbers with findall.
    p2 = r"average equivalent class a shares outstanding\s+([\d,\s]{5,80}?)(?:[a-z—\-]|$)"
    m = re.search(p2, normalized, re.IGNORECASE)
    if m and filing_date:
        nums_text = m.group(1).strip()
        nums = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+", nums_text)]
        # nums are most-recent-first; map to FY years.
        # 10-K is typically filed within 90 days of FY end.
        # BRK files in Feb for Dec FY end, so the first column = filing_year - 1.
        try:
            filing_year  = int(filing_date[:4])
            filing_month = int(filing_date[5:7])
            # If filed in first half of year, FY ended the prior December
            base_year = filing_year - 1 if filing_month <= 6 else filing_year
        except (ValueError, IndexError):
            base_year = datetime.now().year - 1
        # Berkshire FY ends Dec 31; Pattern 1 takes precedence for a given year
        for i, shares_a in enumerate(nums[:5]):
            year = base_year - i
            fiscal_end = f"{year}-12-31"
            if shares_a > 0 and fiscal_end not in results:
                results[fiscal_end] = shares_a * 1500

    return results




def extract_berkshire_total_debt(text: str, filing_date: str = "") -> dict[str, float]:
    """
    Extract BRK total notes payable from the fair-value disclosure table in 10-K text.

    BRK's fair-value note shows (dollars in millions):
      notes payable and other borrowings: insurance and other  N1 N2 N3 N4 N5
      railroad, utilities and energy                           M1 M2 M3 M4 M5
    where N1 / M1 are carrying amounts for the CURRENT year and the second
    occurrence (after 'december 31, {prior_year}') gives prior-year values.

    Returns {fiscal_year_end_date: total_debt_dollars}
    """
    txt = re.sub(r"<[^>]+>", " ", text)
    txt = re.sub(r"\s+", " ", txt).lower()
    txt = txt.replace("&#160;", " ").replace("&nbsp;", " ").replace("&#8212;", "—")

    try:
        fy = int(filing_date[:4])
        fm = int(filing_date[5:7])
        curr_year = fy - 1 if fm <= 6 else fy
        prior_year = curr_year - 1
    except (ValueError, IndexError):
        return {}

    # Find all occurrences of the fair-value table pattern (no block constraint —
    # block-based approach fails because many "december 31, {year}" appear earlier).
    # Pattern: "notes payable and other borrowings: insurance and other  N
    #           railroad, utilities and energy  M"
    # The FIRST occurrence corresponds to the current filing year; the SECOND to prior.
    pat = re.compile(
        r"notes payable and other borrowings:\s*insurance and other\s+([\d,]+)"
        r".{1,200}?"                               # skip any intervening columns
        r"railroad,\s*utilities and energy\s+([\d,]+)",
        re.IGNORECASE | re.DOTALL,
    )

    matches = list(pat.finditer(txt))
    results: dict[str, float] = {}
    years = [curr_year, prior_year]
    for idx, m in enumerate(matches[:2]):
        try:
            io_val  = float(m.group(1).replace(",", ""))
            rue_val = float(m.group(2).replace(",", ""))
            total   = (io_val + rue_val) * 1e6
            yr      = years[idx]
            results[f"{yr}-12-31"] = total
        except (ValueError, AttributeError):
            continue

    return results


def extract_berkshire_cash_components(text: str, filing_date: str = "") -> dict[str, dict[str, float]]:
    """
    Extract BRK consolidated cash and short-term Treasury bills from 10-K text.

    BRK's consolidated balance sheet lists two rows side-by-side (curr, prior):
      Insurance and other: cash and cash equivalents*  $ 47,719  $ 44,333
      Short-term investments in U.S. Treasury Bills**    321,434    286,472
      ...
      Railroad, utilities and energy: cash and cash equivalents*  4,158  3,396

    Strategy: search globally for these line-pair patterns (no block constraint —
    block-based search fails because many earlier "december 31, YYYY" anchors exist
    and re.search always picks the first one, which is rarely the balance sheet).
    """
    txt = re.sub(r"<[^>]+>", " ", text)
    txt = re.sub(r"\s+", " ", txt).lower()
    txt = txt.replace("&#160;", " ").replace("&nbsp;", " ").replace("&#8212;", "—")

    if not filing_date:
        return {"cash": {}, "short_term_investments": {}, "total_cash": {}}

    try:
        fy = int(filing_date[:4])
        fm = int(filing_date[5:7])
        curr_year = fy - 1 if fm <= 6 else fy
        prior_year = curr_year - 1
    except (ValueError, IndexError):
        return {"cash": {}, "short_term_investments": {}, "total_cash": {}}

    empty = {"cash": {}, "short_term_investments": {}, "total_cash": {}}

    # ── Pattern A: find the balance sheet block that contains BOTH cash and T-bills
    # The two rows always appear together; capture all four numbers at once.
    # Row 1 (insurance cash):  "insurance and other: cash and cash equivalents*  $ C1  $ P1"
    # Row 2 (T-bills):         "short-term investments in u.s. treasury bills**   C2    P2"
    # (The T-bills row may or may not have a $ prefix on its numbers.)
    combined_pat = re.compile(
        r"insurance and other:\s+cash and cash equivalents\*?\s+\$?\s*([\d,]+)"
        r"\s+\$?\s*([\d,]+)"                        # prior-year insurance cash
        r".{0,600}?"                                 # other balance sheet rows between
        r"short-term investments in u\.?s\.? treasury bills\*{0,3}"
        r"\s+\$?\s*([\d,]+)\s+\$?\s*([\d,]+)",      # curr + prior T-bills
        re.IGNORECASE | re.DOTALL,
    )
    m = combined_pat.search(txt)
    if m:
        ins_curr  = float(m.group(1).replace(",", ""))
        ins_prior = float(m.group(2).replace(",", ""))
        tb_curr   = float(m.group(3).replace(",", ""))
        tb_prior  = float(m.group(4).replace(",", ""))

        # Also try to add railroad cash (appears further down the same balance sheet)
        # Look in the text after the combined match
        tail = txt[m.end():]
        rr_pat = re.compile(
            r"railroad,\s*utilities and energy:?\s+cash and cash equivalents\*?\s+\$?\s*([\d,]+)"
            r"\s+\$?\s*([\d,]+)",
            re.IGNORECASE,
        )
        rr = rr_pat.search(tail[:2000])
        rr_curr  = float(rr.group(1).replace(",", "")) if rr else 0.0
        rr_prior = float(rr.group(2).replace(",", "")) if rr else 0.0

        cash_curr  = (ins_curr  + rr_curr)  * 1e6
        cash_prior = (ins_prior + rr_prior) * 1e6
        tb_curr_d  = tb_curr  * 1e6
        tb_prior_d = tb_prior * 1e6

        return {
            "cash": {
                f"{curr_year}-12-31":  cash_curr,
                f"{prior_year}-12-31": cash_prior,
            },
            "short_term_investments": {
                f"{curr_year}-12-31":  tb_curr_d,
                f"{prior_year}-12-31": tb_prior_d,
            },
            "total_cash": {
                f"{curr_year}-12-31":  cash_curr  + tb_curr_d,
                f"{prior_year}-12-31": cash_prior + tb_prior_d,
            },
        }

    return empty


def extract_brk_quarterly_debt(text: str, q_key: str) -> dict[str, float]:
    """
    Extract BRK total notes payable from a 10-Q fair-value disclosure table.

    Same layout as the 10-K: the FIRST occurrence of the two-segment pattern
    corresponds to the current quarter-end (the second, if present, is prior year-end).

    Returns {q_key: total_debt_dollars} or {} on failure.
    """
    txt = re.sub(r"<[^>]+>", " ", text)
    txt = re.sub(r"\s+", " ", txt).lower()
    txt = txt.replace("&#160;", " ").replace("&nbsp;", " ")

    pat = re.compile(
        r"notes payable and other borrowings:\s*insurance and other\s+([\d,]+)"
        r".{1,200}?"
        r"railroad,\s*utilities and energy\s+([\d,]+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(txt)
    if not m:
        return {}
    try:
        io_val  = float(m.group(1).replace(",", ""))
        rue_val = float(m.group(2).replace(",", ""))
        return {q_key: (io_val + rue_val) * 1e6}
    except (ValueError, AttributeError):
        return {}


def extract_brk_quarterly_cash(text: str, q_key: str, quarter_end_date: str) -> dict[str, dict[str, float]]:
    """
    Extract BRK cash components from a 10-Q balance sheet for one quarter.

    BRK's 10-Q consolidated balance sheet has the same two-column layout as the 10-K
    (current quarter-end | prior fiscal year-end).  We only need the current column.

    Returns {"cash": {q_key: val}, "short_term_investments": {q_key: val}, "total_cash": {q_key: val}}
    or empty dicts on failure.
    """
    empty = {"cash": {}, "short_term_investments": {}, "total_cash": {}}
    if not quarter_end_date:
        return empty
    txt = re.sub(r"<[^>]+>", " ", text)
    txt = re.sub(r"\s+", " ", txt).lower()
    txt = txt.replace("&#160;", " ").replace("&nbsp;", " ")

    # Pattern: same two rows as annual — grab first (current-period) number from each
    ins_pat = re.compile(
        r"insurance and other:\s+cash and cash equivalents\*?\s+\$?\s*([\d,]+)",
        re.IGNORECASE,
    )
    tb_pat = re.compile(
        r"short-term investments in u\.?s\.? treasury bills\*{0,3}\s+\$?\s*([\d,]+)",
        re.IGNORECASE,
    )
    rr_pat = re.compile(
        r"railroad,\s*utilities and energy:?\s+cash and cash equivalents\*?\s+\$?\s*([\d,]+)",
        re.IGNORECASE,
    )
    ins_m = ins_pat.search(txt)
    tb_m  = tb_pat.search(txt)
    if not ins_m or not tb_m:
        return empty

    ins_val = float(ins_m.group(1).replace(",", "")) * 1e6
    tb_val  = float(tb_m.group(1).replace(",", "")) * 1e6

    tail = txt[ins_m.end():]
    rr_m = rr_pat.search(tail[:3000])
    rr_val = float(rr_m.group(1).replace(",", "")) * 1e6 if rr_m else 0.0

    cash_val  = ins_val + rr_val
    total_val = cash_val + tb_val
    return {
        "cash":                 {q_key: cash_val},
        "short_term_investments": {q_key: tb_val},
        "total_cash":           {q_key: total_val},
    }







def extract_berkshire_operating_earnings(text: str, filing_date: str = "") -> dict[str, float]:
    """
    Extract BRK's operating earnings from the per-segment earnings attribution table
    that Berkshire includes in its 10-K MD&A section.

    The table (in millions, after-tax, excl. noncontrolling interests) looks like:
        2025    2024    2023
        Insurance – underwriting          $  7,258   $  9,020   $  5,428
        Insurance – investment income       12,513     13,670      9,567
        BNSF                                 5,476      5,031      5,087
        Berkshire Hathaway Energy (BHE)      3,979      3,730      2,331
        Manufacturing, service/retailing    13,647     13,072     13,362
        Investment gains (losses)           30,737     41,558     58,873   ← stop here
        ...
        Net earnings attributable…          66,968     88,995     96,223

    Operating earnings = sum of segment rows BEFORE "Investment gains".
    Returns {date_str: value_in_dollars} for curr, prior, and prior-prior year.
    """
    try:
        fy = int(filing_date[:4])
        fm = int(filing_date[5:7])
        curr_year  = fy - 1 if fm <= 6 else fy
        prior_year = curr_year - 1
        prior2_year = curr_year - 2
    except (ValueError, IndexError):
        return {}

    txt = re.sub(r"<[^>]+>", " ", text)
    txt = re.sub(r"&#160;|&nbsp;", " ", txt)
    txt = re.sub(r"&#8211;|&#8212;", "-", txt)
    txt = re.sub(r"&#\d+;", " ", txt)   # strip all remaining numeric HTML entities
    txt = re.sub(r"&[a-z]+;", " ", txt)
    txt = re.sub(r"\s+", " ", txt)

    # Anchor: find the full 3-year shareholder-earnings attribution table.
    # Capture everything from the year headers to "Net earnings attributable".
    anchor_pat = re.compile(
        r"earnings attributable to berkshire shareholders[^2]*"
        r"(?:in millions)[^2]*"
        r"(\d{4})\s+(\d{4})\s+(\d{4})"      # year headers
        r"(.+?)"                              # all segment rows
        r"net earnings attributable",         # stop at the net-earnings total
        re.IGNORECASE | re.DOTALL,
    )
    m = anchor_pat.search(txt)
    if not m:
        return {}

    yr0 = int(m.group(1))   # typically curr_year
    yr1 = int(m.group(2))   # prior_year
    yr2 = int(m.group(3))   # prior_prior_year
    body = m.group(4)

    # Rows to EXCLUDE from operating earnings (investment gains, impairments)
    exclude_pat = re.compile(
        r"investment gains|investment losses|impairment|unrealized",
        re.IGNORECASE,
    )

    # Value tokens: parenthesized negative "(1,234)", plain number "1,234",
    # or standalone dash " - " meaning zero/NA.
    val_tok = re.compile(r"\(\s*([\d,]+)\s*\)|([\d,]+)|\s(-)\s")

    def parse_tok(s: str) -> float:
        s = s.strip()
        if not s or s == "-":
            return 0.0
        if s.startswith("(") and s.endswith(")"):
            return -float(s[1:-1].replace(",", ""))
        return float(s.replace(",", ""))

    col = [0.0, 0.0, 0.0]

    # Walk through the body line-by-line (we re-split on keyword boundaries).
    # Each segment row looks like:  LABEL  VAL0  VAL1  VAL2
    # We extract runs of 3 consecutive value tokens after non-excluded labels.
    segments = re.split(
        r"(?="
        r"insurance\s*[-–]"
        r"|bnsf"
        r"|berkshire hathaway energy"
        r"|manufacturing"
        r"|investment gains"
        r"|impairment"
        r"|other\b"
        r")",
        body,
        flags=re.IGNORECASE,
    )

    for seg in segments:
        if not seg.strip():
            continue
        if exclude_pat.search(seg[:80]):     # check only the label portion
            continue
        # Strip the label (everything before the first digit or opening paren)
        # so that dashes in label text like "Insurance - underwriting" are ignored.
        num_start = re.search(r"[\d(]", seg)
        values_text = seg[num_start.start():] if num_start else seg

        # Extract up to first 3 value tokens from the numeric portion
        toks = val_tok.findall(values_text)
        vals = []
        for t in toks:
            if t[0]:                          # parenthesized negative
                vals.append(-float(t[0].replace(",", "")))
            elif t[1]:                        # plain number
                stripped = t[1].replace(",", "")
                if not stripped:
                    continue
                # Skip 4-digit years that appear in the label portion
                if len(stripped) == 4 and 2000 <= int(stripped) <= 2100:
                    continue
                vals.append(float(stripped))
            else:                             # standalone dash = zero
                vals.append(0.0)
            if len(vals) == 3:
                break
        if len(vals) == 3:
            col[0] += vals[0]
            col[1] += vals[1]
            col[2] += vals[2]

    if all(v == 0.0 for v in col):
        return {}

    result = {}
    for hdr_yr, canon_yr, v in [
        (yr0, curr_year,   col[0]),
        (yr1, prior_year,  col[1]),
        (yr2, prior2_year, col[2]),
    ]:
        if v > 0:
            result[f"{canon_yr}-12-31"] = v * 1_000_000
    return result


def extract_berkshire_equivalent_b_shares_from_facts(facts: dict) -> dict[str, float]:
    """
    Build year-by-year class B equivalent share count for Berkshire.
    BRK files class A equivalent weighted avg shares through ~2014 only.
    Multiply class A equivalent × 1500 to get class B equivalents.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei",  {})
    combined: dict[str, float] = {}

    # Strategy 1: look for explicit ClassA / ClassB outstanding tags
    class_a: dict[str, float] = {}
    class_b: dict[str, float] = {}
    for namespace in [dei, gaap]:
        for concept_name, concept in namespace.items():
            lowered = concept_name.lower()
            if "shares" not in lowered or "outstanding" not in lowered:
                continue
            is_a = "classa" in lowered or "class_a" in lowered
            is_b = "classb" in lowered or "class_b" in lowered
            if not is_a and not is_b:
                continue
            target = class_a if is_a else class_b
            for unit_key in ["shares", "pure"]:
                for e in concept.get("units", {}).get(unit_key, []):
                    if e.get("form") not in {"10-K", "10-K/A"}:
                        continue
                    end, val = e.get("end", ""), e.get("val")
                    if end and val is not None:
                        if end not in target or abs(val) > abs(target[end]):
                            target[end] = val

    if class_a or class_b:
        for end in sorted(set(class_a) | set(class_b)):
            total = class_b.get(end, 0.0) + class_a.get(end, 0.0) * 1500
            if total > 0:
                combined[end] = total

    # Strategy 2: WeightedAverageNumberOfSharesOutstandingBasic for BRK is
    # reported in class A equivalent units — multiply by 1500
    if not combined:
        wtd_tag = gaap.get("WeightedAverageNumberOfSharesOutstandingBasic", {})
        for unit_key in ["shares", "pure"]:
            for e in wtd_tag.get("units", {}).get(unit_key, []):
                if e.get("form") not in {"10-K", "10-K/A"} or e.get("fp") != "FY":
                    continue
                end, val = e.get("end", ""), e.get("val")
                if end and val and val < 10_000_000:  # sanity: class A shares are ~1-2M
                    b_equiv = val * 1500
                    if end not in combined or b_equiv > combined[end]:
                        combined[end] = b_equiv

    return normalize_to_fiscal_years(combined) if combined else {}



# ── Hooks called by the core app (see company_templates.call_hook) ───────────

def seed_from_facts(facts: dict, financials: dict) -> None:
    """Replace the class-A XBRL share count with B-equivalents (~2011-2014)."""
    equivalent_b = extract_berkshire_equivalent_b_shares_from_facts(facts)
    if equivalent_b:
        financials["shares_outstanding_end"] = equivalent_b


def apply_annual_filings(filings: list, financials: dict, ctx: dict) -> None:
    """
    Walk historical 10-Ks newest-first, filling shares, debt, cash and
    operating earnings from the filing text wherever the tagged data is
    missing, and stop fetching once every displayed year is covered.

    ctx supplies the core app's helpers: get_text(filing) -> str (cached),
    fy_get(series, year) and min_year.
    """
    get_text = ctx["get_text"]
    fy_get = ctx["fy_get"]
    min_year = ctx["min_year"]

    for filing in filings:
        fy = filing.get("fiscal_year", "")
        if not fy or int(fy) < min_year:
            break
        fy_int = int(fy)
        existing_sh = financials.get("shares_outstanding_end", {})
        existing_td = financials.get("total_debt", {})
        existing_tc = financials.get("total_cash", {})
        existing_cash = financials.get("cash", {})
        existing_st = financials.get("short_term_investments", {})
        # Each 10-K carries a 3-year share table but only 2 years of balances.
        sh_covered = all(fy_get(existing_sh, str(fy_int - i)) is not None
                         for i in range(3))
        td_covered = all(fy_get(existing_td, str(fy_int - i)) is not None
                         for i in range(2))
        tc_covered = all(fy_get(existing_tc, str(fy_int - i)) is not None
                         for i in range(2))
        cash_components_covered = all(
            fy_get(existing_cash, str(fy_int - i)) is not None and
            fy_get(existing_st, str(fy_int - i)) is not None
            for i in range(2)
        )
        if sh_covered and td_covered and tc_covered and cash_components_covered:
            continue
        try:
            text = get_text(filing)
            fdate = filing["filing_date"]

            if not sh_covered:
                equiv_b = extract_berkshire_equivalent_b_shares(text, fdate)
                if equiv_b:
                    financials["shares_outstanding_end"] = {**existing_sh, **equiv_b}
            if not td_covered:
                debt = extract_berkshire_total_debt(text, fdate)
                if debt:
                    merged_td = dict(existing_td)
                    for d, v in debt.items():
                        if d not in merged_td:
                            merged_td[d] = v
                    financials["total_debt"] = merged_td
                    financials["long_term_debt"] = merged_td
            if not tc_covered or not cash_components_covered:
                cash_parts = extract_berkshire_cash_components(text, fdate)
                if cash_parts.get("cash"):
                    merged_cash = dict(financials.get("cash", {}))
                    for d, v in cash_parts["cash"].items():
                        merged_cash[d] = v
                    financials["cash"] = merged_cash
                if cash_parts.get("short_term_investments"):
                    merged_st = dict(financials.get("short_term_investments", {}))
                    for d, v in cash_parts["short_term_investments"].items():
                        merged_st[d] = v
                    financials["short_term_investments"] = merged_st
                if cash_parts.get("total_cash"):
                    merged_tc = dict(existing_tc)
                    for d, v in cash_parts["total_cash"].items():
                        merged_tc[d] = v
                    financials["total_cash"] = merged_tc

            # Operating earnings (before investment gains) from the MD&A table.
            oe_reported = extract_berkshire_operating_earnings(text, fdate)
            if oe_reported:
                merged_oe = dict(financials.get("brk_operating_earnings", {}))
                for d, v in oe_reported.items():
                    if d not in merged_oe:
                        merged_oe[d] = v
                financials["brk_operating_earnings"] = merged_oe

        except Exception:
            continue

    # XBRL shares_diluted_wtd and eps_diluted are class-A basis (~1.4M shares,
    # ~$40,000/share). Drop them so the core recompute derives per-B-share
    # figures from net income / B-equivalent shares instead.
    financials.pop("shares_diluted_wtd", None)
    financials.pop("eps_diluted", None)


def apply_quarterly(financials: dict, quarter_end_dates: dict,
                    quarter_filing_links: dict, ctx: dict) -> None:
    """Same corrections against the 10-Qs: text-extract cash and debt, and
    forward-fill B-equivalent shares over the class-A counts XBRL files."""
    get_text_url = ctx["get_text_url"]

    for qk, qdate in quarter_end_dates.items():
        qurl = quarter_filing_links.get(qk)
        if not qurl:
            continue
        try:
            text = get_text_url(qurl)
            cash_parts = extract_brk_quarterly_cash(text, qk, qdate)
            for subkey in ("cash", "short_term_investments", "total_cash"):
                if cash_parts.get(subkey):
                    financials.setdefault(subkey, {}).update(cash_parts[subkey])
            qdebt = extract_brk_quarterly_debt(text, qk)
            if qdebt:
                for dk in ("total_debt", "long_term_debt"):
                    financials.setdefault(dk, {}).update(qdebt)
        except Exception:
            pass

    so = financials.get("shares_outstanding_end", {})
    annual_dates = sorted(d for d in so if not d.startswith("Q"))
    if annual_dates:
        last_annual = so[annual_dates[-1]]
        for qk in quarter_end_dates:
            existing = so.get(qk)
            # Replace if missing or clearly class-A scale (< 100M shares).
            if not existing or abs(existing) < 1e8:
                so[qk] = last_annual
