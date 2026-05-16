#!/usr/bin/env python3
"""
Value Line Style Investment Analysis – Flask Backend
Pulls 15 years of financial data from SEC EDGAR (free, no API key).
"""

import json
import math
import os
import re

from collections import Counter
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "InvestmentAnalysis research@example.com", "Accept": "application/json"}
MAX_YEARS = 15
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dash}/{document}"

# ── Buffett 1980-letter constants ─────────────────────────────────────────────
# Buffett used 28% capital-gains tax and ~12% inflation in the 1980 letter to
# show that most equity investors were earning negative real after-tax returns.
# Adjust INFLATION_RATE to the current environment as needed.
CAPITAL_GAINS_TAX_RATE = 0.15   # rate used in Buffett's original 1980 analysis
INFLATION_RATE         = 0.03   # approximate current long-run inflation assumption


def find_section(text: str, markers: list, char_limit: int = 60000) -> str:
    """Find first marker in text (case-insensitive) and return up to char_limit chars."""
    tl = text.lower()
    for marker in markers:
        idx = tl.find(marker.lower())
        if idx >= 0:
            return text[idx: idx + char_limit]
    return text[-char_limit:]


# ─── EDGAR helpers ────────────────────────────────────────────────────────────

def resolve_cik(ticker: str) -> Optional[str]:
    normalized = ticker.upper().strip()
    candidates = {
        normalized,
        normalized.replace(".", "-"),
        normalized.replace(".", ""),
        normalized.replace("-", "."),
        normalized.replace("-", ""),
    }

    # Strategy 1: bulk tickers file (~10K most active filers)
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        for entry in r.json().values():
            if entry["ticker"].upper() in candidates:
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass

    # Strategy 2: EDGAR company search — accepts ticker symbols directly and
    # returns a page whose URL contains the CIK (works for any SEC registrant)
    try:
        search_url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={normalized}&type=10-K"
            "&dateb=&owner=include&count=1&search_text="
        )
        r = requests.get(search_url, headers=HEADERS, timeout=15, allow_redirects=True)
        # The CIK appears in the response URL or body as a 10-digit string
        import re as _re
        m = _re.search(r"CIK=?(\d{7,10})", r.url + r.text)
        if m:
            return m.group(1).zfill(10)
    except Exception:
        pass

    # Strategy 3: EDGAR full-text search index
    try:
        for form in ("10-K", "20-F"):
            r = requests.get(
                f"https://efts.sec.gov/LATEST/search-index?q=%22{normalized}%22&forms={form}",
                headers=HEADERS, timeout=15,
            )
            hits = r.json().get("hits", {}).get("hits", [])
            for h in hits:
                ciks = h.get("_source", {}).get("ciks", [])
                syms = " ".join(h.get("_source", {}).get("display_names", []))
                if normalized in syms.upper() and ciks:
                    return ciks[0].zfill(10)
    except Exception:
        pass

    return None


def fetch_company_facts(cik: str) -> dict:
    r = requests.get(f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_submissions(cik: str) -> dict:
    r = requests.get(f"{EDGAR_BASE}/submissions/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def sec_get_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return r.text


def filing_html_to_text(html: str) -> str:
    """Full BeautifulSoup parse — accurate but slower."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def quick_filing_text(html: str) -> str:
    """Fast regex-based HTML→text for large 10-K files (no BeautifulSoup)."""
    # Drop script/style blocks
    text = re.sub(r"<(?:script|style)[^>]*?>.*?</(?:script|style)>", " ",
                  html, flags=re.DOTALL | re.IGNORECASE)
    # Drop all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace(
        "&lt;", "<").replace("&gt;", ">").replace("&#160;", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_earnings_materials(
    submissions: dict,
    quarter_end_dates: dict[str, str],
    ticker: str,
) -> dict[str, list]:
    """
    For each quarter-end date, collect SEC 8-K filings filed within 60 days
    afterwards that relate to earnings (item 2.02) or investor presentations
    (items 7.01 / 8.01).  Also attaches a Seeking Alpha transcript link.

    Returns {qk: [{"type", "label", "url", "index_url", "filing_date"}, …]}
    """
    if not quarter_end_dates:
        return {}

    recent       = submissions.get("filings", {}).get("recent", {})
    form_list    = recent.get("form", [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    items_list   = recent.get("items", [])
    cik_no_zero  = str(int(submissions.get("cik", "0")))

    # Collect all 8-K / 8-K/A filings once
    eight_ks: list[dict] = []
    for idx, form in enumerate(form_list):
        if form not in {"8-K", "8-K/A"}:
            continue
        fd = filing_dates[idx] if idx < len(filing_dates) else ""
        if not fd:
            continue
        eight_ks.append({
            "filing_date": fd,
            "items":       items_list[idx] if idx < len(items_list) else "",
            "accession":   accessions[idx],
            "primary_doc": primary_docs[idx] if idx < len(primary_docs) else "",
        })

    # Seeking Alpha ticker format (BRK.B → BRK-B)
    sa_ticker = ticker.replace(".", "-").replace("/", "-")

    result: dict[str, list] = {}
    for qk, qdate in quarter_end_dates.items():
        try:
            qdt = datetime.strptime(qdate, "%Y-%m-%d")
        except Exception:
            continue

        materials: list[dict] = []

        for ek in eight_ks:
            try:
                fdt = datetime.strptime(ek["filing_date"], "%Y-%m-%d")
            except Exception:
                continue
            days_after = (fdt - qdt).days
            if not (0 <= days_after <= 60):
                continue

            items = ek.get("items", "")
            if "2.02" in items:
                label    = "Earnings Press Release (8-K)"
                mat_type = "earnings_release"
            elif "7.01" in items:
                label    = "Investor Presentation / Reg-FD (8-K)"
                mat_type = "presentation"
            elif "8.01" in items:
                label    = "Other Earnings Event (8-K)"
                mat_type = "other"
            else:
                continue  # unrelated 8-K

            acc            = ek["accession"]
            acc_no_dash    = acc.replace("-", "")
            primary_url    = SEC_ARCHIVES.format(
                cik_no_zero=cik_no_zero,
                accession_no_dash=acc_no_dash,
                document=ek["primary_doc"],
            )
            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_zero}/{acc_no_dash}/{acc}-index.htm"
            )
            materials.append({
                "type":         mat_type,
                "label":        label,
                "url":          primary_url,
                "index_url":    index_url,
                "filing_date":  ek["filing_date"],
            })

        # Always add transcript links (free services)
        materials.append({
            "type":  "transcript",
            "label": "Transcripts (Motley Fool)",
            "url":   f"https://www.google.com/search?q={sa_ticker}+earnings+call+transcript+site:fool.com",
        })
        materials.append({
            "type":  "transcript",
            "label": "Transcripts (Seeking Alpha)",
            "url":   f"https://seekingalpha.com/symbol/{sa_ticker}/earnings/transcripts",
        })

        if materials:
            result[qk] = materials

    return result


def all_filing_infos_from_submissions(submissions: dict, forms: set[str],
                                      max_count: int = 25) -> list[dict]:
    """Return a list of all matching filings (most-recent first), up to max_count."""
    recent = submissions.get("filings", {}).get("recent", {})
    form_list    = recent.get("form", [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    cik_no_zero  = str(int(submissions.get("cik", "0")))

    results = []
    for idx, form in enumerate(form_list):
        if form not in forms:
            continue
        if len(results) >= max_count:
            break
        accession    = accessions[idx]
        primary_doc  = primary_docs[idx]
        filing_date  = filing_dates[idx] if idx < len(filing_dates) else ""
        report_date  = report_dates[idx] if idx < len(report_dates) else ""
        # Fiscal year = year of the report period end date
        if report_date:
            fy_year = report_date[:4]
        elif filing_date:
            # Heuristic: 10-K filed Jan-Jun covers the prior fiscal year
            fy_year = str(int(filing_date[:4]) - 1) if filing_date[5:7] <= "06" else filing_date[:4]
        else:
            fy_year = ""
        results.append({
            "form":         form,
            "filing_date":  filing_date,
            "report_date":  report_date,
            "fiscal_year":  fy_year,
            "url": SEC_ARCHIVES.format(
                cik_no_zero=cik_no_zero,
                accession_no_dash=accession.replace("-", ""),
                document=primary_doc,
            ),
        })
    return results


def filing_info_from_submissions(submissions: dict, forms: set[str]) -> Optional[dict]:
    """Return the single most-recent matching filing."""
    hits = all_filing_infos_from_submissions(submissions, forms, max_count=1)
    return hits[0] if hits else None


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




def get_proxy_filing_url(submissions: dict) -> tuple[str, str]:
    """Return (url, filing_date) of the most recent DEF 14A proxy."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    dates        = recent.get("filingDate", [])
    cik_no_zero  = str(int(submissions.get("cik", "0")))

    for i, form in enumerate(forms):
        if form == "DEF 14A":
            acc = accessions[i].replace("-", "")
            doc = primary_docs[i]
            url = SEC_ARCHIVES.format(
                cik_no_zero=cik_no_zero,
                accession_no_dash=acc,
                document=doc,
            )
            return url, dates[i]
    return "", ""


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


def market_ticker_candidates(ticker: str) -> list[str]:
    normalized = ticker.upper().strip()
    candidates = [
        normalized,
        normalized.replace(".", "-"),
        normalized.replace(".", ""),
        normalized.replace("-", "."),
        normalized.replace("-", ""),
    ]
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def get_market_data(ticker: str) -> dict:
    """Fetch ONLY current price and exchange-level statistics from Yahoo Finance.
    Share counts and market cap are derived from SEC EDGAR data instead."""
    data = {}
    for candidate in market_ticker_candidates(ticker):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{candidate}",
                             params={"interval": "1d", "range": "5d"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            result = r.json().get("chart", {}).get("result")
            if not result:
                continue
            meta = result[0]["meta"]
            data["price"] = meta.get("regularMarketPrice", meta.get("previousClose"))
            break
        except Exception:
            continue
    for candidate in market_ticker_candidates(ticker):
        try:
            r = requests.get(f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{candidate}",
                             params={"modules": "price,summaryDetail"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            result = r.json().get("quoteSummary", {}).get("result")
            if not result:
                continue
            payload = result[0]
            pi = payload.get("price", {})
            sd = payload.get("summaryDetail", {})
            data["beta"]    = sd.get("beta", {}).get("raw")
            data["52w_high"] = sd.get("fiftyTwoWeekHigh", {}).get("raw")
            data["52w_low"]  = sd.get("fiftyTwoWeekLow",  {}).get("raw")
            if "price" not in data:
                data["price"] = pi.get("regularMarketPrice", {}).get("raw")
            if data.get("price"):
                break
        except Exception:
            continue
    return data


# ─── XBRL tag priority lists ─────────────────────────────────────────────────

METRIC_TAGS: dict[str, list[str]] = {
    # Income Statement
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet", "NetSales",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": [
        "OperatingIncomeLoss",
        # Fallback for companies (e.g. BRK) that don't separately file OperatingIncomeLoss
        # but report pre-tax earnings — closest available proxy for conglomerates/insurers.
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],

    # Bank-specific income statement metrics
    "interest_income":      ["InterestAndDividendIncomeOperating", "InterestAndFeeIncomeLoansAndLeases",
                             "InterestIncomeOperating"],
    "net_interest_income":  ["InterestIncomeExpenseNet", "InterestIncomeExpenseAfterProvisionForLosses"],
    "noninterest_income":   ["NoninterestIncome"],
    "noninterest_expense":  ["NoninterestExpense"],
    "provision_for_losses": ["ProvisionForLoanAndLeaseLosses", "ProvisionForLoanLeaseAndOtherLosses",
                             "ProvisionForCreditLosses"],

    # Per Share (USD/shares unit)
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "dividends_per_share": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],

    # Cash Flow
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",  # DPZ 2014-2016 and similar
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",       # Seadrill (SDRL) and similar
        "PaymentsToAcquireOilAndGasPropertyAndEquipment",
        "PaymentsToAcquirePropertyPlantEquipmentAndIntangibleAssets",
        "SegmentExpenditureAdditionToLongLivedAssets",
        "PaymentsForCapitalImprovements",               # Noble Corporation (NE) and similar
    ],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
    "stock_based_compensation": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense",
                                  "StockBasedCompensation", "EmployeeBenefitsAndShareBasedCompensation"],
    # Unrealized investment gains / (losses) only — positive = gain, negative = loss.
    # Realized gains are kept in net income (they represent actual cash transactions).
    # Unrealized gains are stripped out because they are pure mark-to-market noise
    # that distorts recurring earning power (especially post-ASC 321, 2018+).
    "investment_gains": [
        "UnrealizedGainLossOnInvestments",
        "EquitySecuritiesFvNiUnrealizedGainLoss",         # most common post-ASC 321 tag
        "TradingSecuritiesUnrealizedHoldingGainLoss",
        "UnrealizedGainLossOnSecurities",
    ],

    # Balance Sheet – point-in-time
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        # ASC 230 / post-2017 standard tag; also used by BRK after 2017
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "short_term_investments": [
        "ShortTermInvestments", "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        "DebtAndFinanceLeaseObligationsNoncurrent",
    ],
    "current_debt": [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "CommercialPaper",
        "ShortTermDebt",
    ],
    "goodwill": ["Goodwill"],
    "intangibles": ["FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill"],
    "inventory": ["InventoryNet"],

    # Shares
    "shares_outstanding_end": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    "shares_diluted_wtd": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],

    # Capital Returns
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks_value": ["PaymentsForRepurchaseOfCommonStock"],
    "shares_repurchased": [
        "StockRepurchasedAndRetiredDuringPeriodShares",
        "StockRepurchasedDuringPeriodShares",
        "TreasuryStockSharesAcquired",
    ],

    # Buyback program remaining (best-effort; not all companies report via XBRL)
    "buyback_remaining": [
        "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
        "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount",
    ],

    # ── BDC / Investment Company metrics ─────────────────────────────────────
    # Net Investment Income (flow – income statement equivalent for BDCs)
    "net_investment_income": [
        "NetInvestmentIncome",
        "InvestmentIncomeOperatingAfterExpenseAndTax",
    ],
    # Gross Investment Income (top-line "revenue" equivalent for BDCs)
    "gross_investment_income": [
        "GrossInvestmentIncomeOperating",
        "InvestmentIncomeInterestAndDividend",
        "InvestmentIncomeInterest",
    ],
    # NAV per share (point-in-time, filed directly in XBRL)
    "nav_per_share": [
        "NetAssetValuePerShare",
    ],
    # NII per share (flow – per-share NII as reported in financial highlights)
    "nii_per_share": [
        "InvestmentCompanyInvestmentIncomeLossPerShare",
        "InvestmentCompanyInvestmentIncomeLossFromOperationsPerShare",
    ],
}


# ─── XBRL extraction ─────────────────────────────────────────────────────────

def _period_months(start: str, end: str) -> Optional[int]:
    try:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        return round((e - s).days / 30.44)
    except Exception:
        return None


def _discover_quarter_end_dates(facts: dict, tags: list[str], last_dt: datetime) -> list[str]:
    """
    Return up to 3 unique 10-Q end-dates strictly after last_dt, sorted ascending.
    Used to discover what quarter-end dates exist regardless of fiscal calendar.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei",    {})
    found: set[str] = set()
    for tag in tags:
        concept = gaap.get(tag) or dei.get(tag)
        if not concept:
            continue
        for unit_key in ["USD", "USD/shares", "shares", "pure"]:
            for e in concept.get("units", {}).get(unit_key, []):
                if e.get("form") not in {"10-Q", "10-Q/A"}:
                    continue
                end = e.get("end", "")
                if not end:
                    continue
                try:
                    end_dt = datetime.strptime(end, "%Y-%m-%d")
                except Exception:
                    continue
                if end_dt > last_dt:
                    # Only genuine quarter-end dates (period ~3 months)
                    start = e.get("start", "")
                    if start:
                        months = _period_months(start, end)
                        if months is not None and 2 <= months <= 4:
                            found.add(end)
                    else:
                        found.add(end)
            if found:
                break
    return sorted(found)[:3]


def extract_post_annual_quarters(
    facts: dict,
    tags: list[str],
    last_annual_date: str,
    is_balance_sheet: bool = False,
) -> dict[str, float]:
    """
    Extract up to 3 standalone quarterly values for 10-Q periods whose end date
    is strictly after last_annual_date (the fiscal year-end of the last 10-K).

    Works for any fiscal year calendar (Dec, Sep, Jun, Mar, etc.) because it
    does NOT filter by calendar month — it simply takes the first three 10-Q
    end-dates after the annual.

    Returns {"Q1": value, "Q2": value, "Q3": value}  (relative numbering).
    The caller should also store the matching end-dates for display labels.
    """
    try:
        last_dt = datetime.strptime(last_annual_date, "%Y-%m-%d")
    except Exception:
        return {}

    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei",    {})

    # Collect all 10-Q entries after the annual, keyed by end_date.
    # For flow metrics: store {end_date: {period_months: value}}
    # For balance sheet: store {end_date: value}
    flow_by_end: dict[str, dict[int, float]] = {}
    bs_by_end:   dict[str, float] = {}

    for tag in tags:
        concept = gaap.get(tag) or dei.get(tag)
        if not concept:
            continue
        for unit_key in ["USD", "USD/shares", "shares", "pure"]:
            entries = concept.get("units", {}).get(unit_key, [])
            if not entries:
                continue
            for e in entries:
                if e.get("form") not in {"10-Q", "10-Q/A"}:
                    continue
                end = e.get("end", "")
                val = e.get("val")
                if val is None or not end:
                    continue
                try:
                    end_dt = datetime.strptime(end, "%Y-%m-%d")
                except Exception:
                    continue
                if end_dt <= last_dt:
                    continue

                if is_balance_sheet:
                    if end not in bs_by_end or abs(val) > abs(bs_by_end[end]):
                        bs_by_end[end] = val
                else:
                    start = e.get("start", "")
                    if not start:
                        continue
                    months = _period_months(start, end)
                    if months is None or months < 2 or months > 10:
                        continue
                    bucket = flow_by_end.setdefault(end, {})
                    if months not in bucket or abs(val) > abs(bucket[months]):
                        bucket[months] = val
            break  # right unit type found; stop

    if is_balance_sheet:
        sorted_ends = sorted(bs_by_end)[:3]
        return {f"Q{i+1}": bs_by_end[e] for i, e in enumerate(sorted_ends)}

    if not flow_by_end:
        return {}

    # Sort quarter-end dates; take first 3 unique quarter-end dates
    sorted_ends = sorted(flow_by_end)[:3]

    # Compute standalone values.
    # standalone[end] = value if a ~3-month period exists
    # ytd_cum[end]    = (months, value) for the longest YTD period
    standalone: dict[str, float] = {}
    ytd_cum:    dict[str, tuple[int, float]] = {}

    for end, periods in flow_by_end.items():
        min_m = min(periods)
        max_m = max(periods)
        if min_m <= 4:
            standalone[end] = periods[min_m]
        if max_m >= 5:
            ytd_cum[end] = (max_m, periods[max_m])

    result: dict[str, float] = {}
    running_ytd = 0.0
    prev_ytd_end = None

    for i, end in enumerate(sorted_ends):
        label = f"Q{i+1}"
        if end in standalone:
            result[label] = standalone[end]
            # Update running YTD for potential future subtraction
            running_ytd += standalone[end]
        elif end in ytd_cum:
            ytd_m, ytd_v = ytd_cum[end]
            if i == 0:
                # First post-annual quarter: YTD == standalone
                result[label] = ytd_v
                running_ytd = ytd_v
            else:
                result[label] = ytd_v - running_ytd
                running_ytd = ytd_v
        # else: no data for this quarter

    return result


def extract_annual_series(facts: dict, tags: list[str]) -> dict[str, float]:
    """
    Return {end_date: value} scanning ALL candidate tags and keeping the
    largest absolute value per year-end date across all tags.
    This ensures that when a company reports both a subset metric and a total
    (e.g. BRK reports contract revenue AND total revenues), we always pick
    the total.  Only accepts 10-K/10-K/A FY entries spanning 10-14 months.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei",    {})

    merged: dict[str, float] = {}

    for tag in tags:
        concept = gaap.get(tag) or dei.get(tag)
        if not concept:
            continue
        units = concept.get("units", {})
        for unit_key in ["USD", "USD/shares", "shares", "pure"]:
            entries = units.get(unit_key, [])
            if not entries:
                continue
            for e in entries:
                if e.get("form") not in {"10-K", "10-K/A", "20-F", "20-F/A"} or e.get("fp", "") != "FY":
                    continue
                end = e.get("end", "")
                val = e.get("val")
                if val is None or not end:
                    continue
                start = e.get("start")
                if start:
                    months = _period_months(start, end)
                    if months is not None and not (10 <= months <= 14):
                        continue
                # Remap the end-date year when the company's fiscal year crosses a
                # calendar-year boundary (e.g. DPZ FY2025 ends 2026-01-04):
                #   fy=2025, end="2026-01-04" → int(end[:4]) - fy == 1 → remap to "2025-01-04"
                # This preserves the month-day so downstream year-prefix lookups still work.
                # Only fires when end_year is exactly fy+1 to avoid touching comparison-period
                # rows (e.g. fy=2026, end="2024-02-29" has a gap of 2, not 1).
                fy_field = e.get("fy")
                if fy_field and (int(end[:4]) - fy_field == 1):
                    end = f"{fy_field}{end[4:]}"
                # Keep largest absolute value across all tags for this date
                if end not in merged or abs(val) > abs(merged[end]):
                    merged[end] = val
            break  # once we find data in a unit type for this tag, stop checking other units

    return merged


def extract_point_in_time_series(facts: dict, tags: list[str]) -> dict[str, float]:
    """
    Return {end_date: value} for point-in-time (balance sheet) concepts.
    Scans ALL candidate tags and keeps the largest |value| per end-date, so
    that when a company switches tags mid-history (e.g. BRK switches cash tags
    in 2018) the full historical series is combined correctly.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})

    merged: dict[str, float] = {}

    for tag in tags:
        concept = gaap.get(tag) or dei.get(tag)
        if not concept:
            continue
        units = concept.get("units", {})
        for unit_key in ["shares", "USD", "pure", "USD/shares"]:
            entries = units.get(unit_key, [])
            if not entries:
                continue
            for e in entries:
                # Annual series only: 10-Q balance sheet dates are handled
                # separately by extract_post_annual_quarters and would otherwise
                # bleed the latest quarter-end date into a phantom annual column.
                if e.get("form") not in {"10-K", "10-K/A", "20-F", "20-F/A"}:
                    continue
                end = e.get("end", "")
                val = e.get("val")
                if val is None or not end:
                    continue
                # Same fiscal-year boundary remap as extract_annual_series
                fy_field = e.get("fy")
                if fy_field and (int(end[:4]) - fy_field == 1):
                    end = f"{fy_field}{end[4:]}"
                if end not in merged or abs(val) > abs(merged[end]):
                    merged[end] = val
            break  # found the right unit type for this tag
    return merged


def normalize_to_fiscal_years(data: dict[str, float]) -> dict[str, float]:
    """
    Collapse to one entry per fiscal year (calendar year of end date).
    When two dates share the same year, keeps the later one.
    """
    by_year: dict[str, tuple[str, float]] = {}
    for date_str, val in sorted(data.items()):
        by_year[date_str[:4]] = (date_str, val)
    return {d: v for d, v in by_year.values()}


def fy_get(data: dict[str, float], year: str) -> Optional[float]:
    """Look up by 4-digit year (matches the date key whose year prefix == year)."""
    for k, v in data.items():
        if k[:4] == year:
            return v
    return None


# ─── Derived metrics ─────────────────────────────────────────────────────────

def build_financials(facts: dict) -> dict[str, dict[str, float]]:
    raw: dict[str, dict[str, float]] = {}
    point_in_time_metrics = {
        "total_assets",
        "current_assets",
        "current_liabilities",
        "total_liabilities",
        "equity",
        "cash",
        "short_term_investments",
        "long_term_debt",
        "current_debt",
        "goodwill",
        "intangibles",
        "inventory",
        "shares_outstanding_end",
        "buyback_remaining",
        # BDC point-in-time
        "nav_per_share",
    }
    for key, tags in METRIC_TAGS.items():
        if not tags:
            continue
        if key in point_in_time_metrics:
            series = extract_point_in_time_series(facts, tags)
        else:
            series = extract_annual_series(facts, tags)
        if series:
            raw[key] = normalize_to_fiscal_years(series)

    # ── Derived ──────────────────────────────────────────────────────────────
    ocf     = raw.get("operating_cash_flow", {})
    capex   = raw.get("capex", {})
    dep     = raw.get("depreciation", {})
    ni      = raw.get("net_income", {})
    oi      = raw.get("operating_income", {})
    inv_gls = raw.get("investment_gains", {})
    rev   = raw.get("revenue", {})
    gp    = raw.get("gross_profit", {})

    # For banks: derive revenue = Net Interest Income + Non-interest Income
    # when no standard revenue tag is filed (WFC, JPM, BAC, etc.)
    if not rev:
        _nii_b = raw.get("net_interest_income", {})
        _noni  = raw.get("noninterest_income",  {})
        if _nii_b or _noni:
            _all_dates = set(_nii_b) | set(_noni)
            _bank_rev  = {}
            for d in _all_dates:
                _bank_rev[d] = (_nii_b.get(d) or 0) + (_noni.get(d) or 0)
            if _bank_rev:
                raw["revenue"] = _bank_rev
                rev = raw["revenue"]
    eq    = raw.get("equity", {})
    ta    = raw.get("total_assets", {})
    ltd   = raw.get("long_term_debt", {})
    ctd   = raw.get("current_debt", {})
    cash  = raw.get("cash", {})
    st    = raw.get("short_term_investments", {})
    tax   = raw.get("income_tax", {})
    so    = raw.get("shares_outstanding_end", {})
    sd    = raw.get("shares_diluted_wtd", {})
    ca    = raw.get("current_assets", {})
    cl    = raw.get("current_liabilities", {})

    def derive(anchor: dict, fn):
        """Apply fn(year) → value for each year in anchor that returns non-None."""
        result = {}
        for date_str in anchor:
            y = date_str[:4]
            v = fn(y)
            if v is not None:
                result[date_str] = v
        return result or None

    # FCF = OCF - |CapEx|
    if ocf:
        raw["fcf"] = {d: ocf[d] - abs(fy_get(capex, d[:4]) or 0) for d in ocf}

    # EBITDA = Operating Income + D&A
    if oi and dep:
        raw["ebitda"] = derive(oi, lambda y: (oi.get(next((k for k in oi if k[:4]==y), "")
                                               ) if any(k[:4]==y for k in oi) else None) and
                               oi.get(next((k for k in oi if k[:4]==y), ""))
                               + (fy_get(dep, y) or 0))
        # Cleaner implementation:
        ebitda = {}
        for d in oi:
            da = fy_get(dep, d[:4])
            if da is not None:
                ebitda[d] = oi[d] + da
        raw["ebitda"] = ebitda if ebitda else None

    # Total Cash = Cash + ST Investments
    if cash or st:
        all_d = set(cash) | set(st)
        raw["total_cash"] = {d: (fy_get(cash, d[:4]) or 0) + (fy_get(st, d[:4]) or 0)
                             for d in all_d}

    # Total Debt = LT Debt + Current Debt
    if ltd or ctd:
        all_d = set(ltd) | set(ctd)
        raw["total_debt"] = {d: (fy_get(ltd, d[:4]) or 0) + (fy_get(ctd, d[:4]) or 0)
                             for d in all_d}

    # Net Cash = Total Cash - Total Debt  (positive = net cash position)
    tc = raw.get("total_cash", {})
    td = raw.get("total_debt", {})
    if tc and td:
        all_d = set(tc) | set(td)
        raw["net_cash"] = {d: (fy_get(tc, d[:4]) or 0) - (fy_get(td, d[:4]) or 0)
                           for d in all_d}

    # Working Capital = Current Assets - Current Liabilities
    if ca and cl:
        raw["working_capital"] = {d: ca[d] - (fy_get(cl, d[:4]) or 0) for d in ca
                                  if fy_get(cl, d[:4]) is not None}

    # Book Value per Share = Equity / Shares Outstanding (period-end)
    if eq and so:
        bvps = {}
        for d in eq:
            s = fy_get(so, d[:4])
            if s and s > 0:
                bvps[d] = eq[d] / s
        raw["book_value_per_share"] = bvps or None

    # Revenue per Share = Revenue / Diluted Shares
    share_base_for_per_share = sd or so
    if rev and share_base_for_per_share:
        rps = {}
        for d in rev:
            s = fy_get(share_base_for_per_share, d[:4])
            if s and s > 0:
                rps[d] = rev[d] / s
        raw["revenue_per_share"] = rps or None

    # FCF per Share = FCF / Diluted Shares
    fcf = raw.get("fcf", {})
    if fcf and share_base_for_per_share:
        fps = {}
        for d in fcf:
            s = fy_get(share_base_for_per_share, d[:4])
            if s and s > 0:
                fps[d] = fcf[d] / s
        raw["fcf_per_share"] = fps or None

    # BDC: NII per share fallback = NII / Shares when direct tag absent
    nii = raw.get("net_investment_income", {})
    if nii and "nii_per_share" not in raw and share_base_for_per_share:
        nii_ps = {}
        for d in nii:
            s = fy_get(share_base_for_per_share, d[:4])
            if s and s > 0:
                nii_ps[d] = nii[d] / s
        raw["nii_per_share"] = nii_ps or None

    # EPS fallback = Net Income / Shares when direct diluted EPS is absent
    if "eps_diluted" not in raw and ni and share_base_for_per_share:
        eps = {}
        for d in ni:
            s = fy_get(share_base_for_per_share, d[:4])
            if s and s > 0:
                eps[d] = ni[d] / s
        raw["eps_diluted"] = eps or None

    # Effective Tax Rate = Tax / Operating Income (proxy for pre-tax income)
    if tax and oi:
        etr = {}
        for d in tax:
            o = fy_get(oi, d[:4])
            if o and o > 0:
                etr[d] = max(0.0, min(tax[d] / o, 0.99))
        raw["effective_tax_rate"] = etr or None

    # Margins
    def margin(numerator: dict, denominator: dict, key: str):
        if numerator and denominator:
            m = {}
            for d in numerator:
                den = fy_get(denominator, d[:4])
                if den and den > 0:
                    m[d] = numerator[d] / den
            if m:
                raw[key] = m

    margin(gp,               rev, "gross_margin")
    margin(oi,               rev, "operating_margin")
    margin(ni,               rev, "net_margin")
    margin(fcf,              rev, "fcf_margin")
    margin(raw.get("ebitda",{}), rev, "ebitda_margin")

    # Adjusted FCF = FCF - Stock-Based Compensation
    sbc = raw.get("stock_based_compensation", {})
    if fcf and sbc:
        adj_fcf = {}
        for d in fcf:
            s = fy_get(sbc, d[:4])
            if s is not None:
                adj_fcf[d] = fcf[d] - abs(s)
        if adj_fcf:
            raw["adj_fcf"] = adj_fcf
            margin(adj_fcf, rev, "adj_fcf_margin")

    # ROE = Net Income / Equity
    if ni and eq:
        roe = {}
        for d in ni:
            e = fy_get(eq, d[:4])
            if e and e != 0:
                roe[d] = ni[d] / e
        raw["roe"] = roe or None

    # FCF ROE = FCF / Equity
    if fcf and eq:
        fcf_roe = {}
        for d in fcf:
            e = fy_get(eq, d[:4])
            if e and e != 0:
                fcf_roe[d] = fcf[d] / e
        raw["fcf_roe"] = fcf_roe or None

    # NII ROE = Net Investment Income / Equity  (BDC equivalent of ROE)
    nii_series = raw.get("net_investment_income", {})
    if nii_series and eq:
        nii_roe = {}
        for d in nii_series:
            e = fy_get(eq, d[:4])
            if e and e != 0:
                nii_roe[d] = nii_series[d] / e
        raw["nii_roe"] = nii_roe or None

    # ── Bank-specific derived metrics ────────────────────────────────────────
    int_inc  = raw.get("interest_income",     {})
    nii_bank = raw.get("net_interest_income", {})
    noni     = raw.get("noninterest_income",  {})
    none_exp = raw.get("noninterest_expense", {})
    ta       = raw.get("total_assets",        {})

    # net_interest_income: prefer direct tag; fall back to interest_income − interest_expense
    int_exp = raw.get("interest_expense", {})
    if not nii_bank and int_inc and int_exp:
        nii_bank = {}
        for d in int_inc:
            ie = fy_get(int_exp, d[:4])
            if ie is not None:
                nii_bank[d] = int_inc[d] - abs(ie)
        if nii_bank:
            raw["net_interest_income"] = nii_bank

    # NIM = Net Interest Income / Total Assets  (proxy for earning assets)
    if nii_bank and ta:
        nim = {}
        for d in nii_bank:
            ta_v = fy_get(ta, d[:4])
            if ta_v and ta_v != 0:
                nim[d] = nii_bank[d] / ta_v
        raw["nim"] = nim or None

    # Efficiency Ratio = Non-interest Expense / (NII + Non-interest Income)
    if none_exp and nii_bank:
        eff = {}
        for d in none_exp:
            nii_v  = fy_get(nii_bank, d[:4]) or 0
            noni_v = fy_get(noni,     d[:4]) or 0
            denom  = nii_v + noni_v
            if denom and denom != 0:
                eff[d] = abs(none_exp[d]) / denom
        raw["efficiency_ratio"] = eff or None

    # Owner Earnings (Buffett, 1986 letter) = Net Income + D&A − CapEx − Investment Gains/Losses
    # Investment gains/losses are stripped out because they are non-cash, lumpy, and not
    # representative of the business's recurring earning power.
    # Real Owner Return (Buffett, 1980 letter) = Owner Earnings / Beginning-of-Year Equity
    # Beginning equity = prior fiscal year's ending equity (the capital at risk for the year)
    if ni and dep and capex and eq:
        oe: dict[str, float] = {}
        for d in ni:
            d_val  = fy_get(dep,   d[:4])
            cx_val = fy_get(capex, d[:4])
            if d_val is not None and cx_val is not None:
                ig_val = fy_get(inv_gls, d[:4]) or 0.0   # 0 if not reported / not applicable
                oe[d] = ni[d] + abs(d_val) - abs(cx_val) - ig_val
        if oe:
            raw["owner_earnings"] = oe
            # Sort equity by year so we can look up the prior year
            eq_by_year = {d[:4]: v for d, v in sorted(eq.items())}
            oe_ret: dict[str, float] = {}
            for d, v in oe.items():
                yr = int(d[:4])
                beg_eq = eq_by_year.get(str(yr - 1))   # beginning-of-year = end of prior year
                if beg_eq and beg_eq != 0:
                    oe_ret[d] = v / beg_eq
            if oe_ret:
                raw["owner_earnings_return"] = oe_ret
                # Real Owner Return (Buffett 1980 letter):
                # = Owner Earnings Return × (1 − cap-gains tax) − inflation
                # Shows what the equity owner actually keeps in real purchasing-power terms.
                raw["real_owner_return"] = {
                    d: v * (1 - CAPITAL_GAINS_TAX_RATE) - INFLATION_RATE
                    for d, v in oe_ret.items()
                }

    # ROA = Net Income / Total Assets
    if ni and ta:
        roa = {}
        for d in ni:
            a = fy_get(ta, d[:4])
            if a and a > 0:
                roa[d] = ni[d] / a
        raw["roa"] = roa or None

    # ROIC = NOPAT / Invested Capital
    # NOPAT = OI × (1 - ETR);  IC = Equity + Total Debt - Total Cash
    etr_d = raw.get("effective_tax_rate", {})
    if oi and eq:
        roic = {}
        for d in oi:
            y = d[:4]
            e    = fy_get(eq, y) or 0
            debt = fy_get(td, y) or 0
            cash_v = fy_get(tc, y) or 0
            t    = fy_get(etr_d, y) if etr_d else 0.21
            ic   = e + debt - cash_v
            if ic and ic > 0:
                roic[d] = oi[d] * (1 - min(t or 0.21, 0.5)) / ic
        raw["roic"] = roic or None

    # ── New capital-quality metrics ───────────────────────────────────────────
    gw  = raw.get("goodwill",    {})
    ia  = raw.get("intangibles", {})

    # ROTE = Net Income / Tangible Equity  (Tangible Equity = Equity − Goodwill − Intangibles)
    if ni and eq:
        rote = {}
        for d in ni:
            e    = fy_get(eq, d[:4]) or 0
            gw_v = fy_get(gw, d[:4]) or 0
            ia_v = fy_get(ia, d[:4]) or 0
            te   = e - gw_v - ia_v
            if te and te != 0:
                rote[d] = ni[d] / te
        raw["rote"] = rote or None

    # 1) Unleveraged Net Tangible Assets (UNTA)
    #    = Equity + Total Debt − Cash − Goodwill − Intangibles
    #    Removes leverage (adds back debt) and strips out financial assets (cash)
    #    and accounting intangibles to isolate the tangible operating capital base.
    if eq:
        unta: dict[str, float] = {}
        for d in eq:
            y      = d[:4]
            e      = fy_get(eq,   y) or 0
            debt   = fy_get(td,   y) or 0
            cash_v = fy_get(tc,   y) or 0
            gw_v   = fy_get(gw,   y) or 0
            ia_v   = fy_get(ia,   y) or 0
            unta[d] = e + debt - cash_v - gw_v - ia_v
        raw["unta"] = unta or None

    # 2) NOPAT = EBIT × (1 − effective tax rate)
    #    EBIT ≈ Operating Income; tax rate capped at 50 % and floored at 0 %.
    if oi:
        nopat: dict[str, float] = {}
        for d in oi:
            t = fy_get(etr_d, d[:4]) if etr_d else None
            nopat[d] = oi[d] * (1 - min(max(t or 0.21, 0.0), 0.5))
        raw["nopat"] = nopat or None

    # 3) Economic Goodwill = NOPAT / UNTA
    #    Measures returns generated per dollar of tangible capital deployed.
    #    A ratio > 1 implies the business earns more than its tangible book in
    #    a single year — a hallmark of a wide-moat franchise.
    _unta  = raw.get("unta",  {})
    _nopat = raw.get("nopat", {})
    if _unta and _nopat:
        eco_gw: dict[str, float] = {}
        for d in _nopat:
            u = fy_get(_unta, d[:4])
            if u and u != 0:
                eco_gw[d] = _nopat[d] / u
        raw["economic_goodwill"] = eco_gw or None

    # Clean up None entries
    return {k: v for k, v in raw.items() if v}


# ─── Year helpers ─────────────────────────────────────────────────────────────

def get_display_years(financials: dict, max_years: int = MAX_YEARS) -> list[str]:
    # Count how many metrics have a real (non-None) value for each year.
    # Years where only 1-2 stray metrics have data (e.g. a 10-Q shares filing
    # that creates a future year column with everything else empty) are excluded.
    year_counts: dict[str, int] = {}
    for series in financials.values():
        for d, v in series.items():
            if v is not None and not d.startswith("Q"):   # annual keys only
                y = d[:4]
                year_counts[y] = year_counts.get(y, 0) + 1

    min_metrics = max(3, len(financials) // 10)   # at least 10% of metrics must be present
    valid_years = {y for y, cnt in year_counts.items() if cnt >= min_metrics}
    return sorted(valid_years)[-max_years:]


def get_display_quarters(financials: dict) -> list[str]:
    """Return Q-prefixed keys that have at least 3 metrics populated."""
    q_counts: dict[str, int] = {}
    for series in financials.values():
        for d, v in series.items():
            if v is not None and d.startswith("Q"):
                q_counts[d] = q_counts.get(d, 0) + 1
    valid = {q for q, cnt in q_counts.items() if cnt >= 3}
    return sorted(valid)


def serialize(financials: dict, years: list[str], quarters: list[str] = None) -> dict:
    out: dict[str, dict[str, Optional[float]]] = {}
    for metric, series in financials.items():
        row: dict[str, Optional[float]] = {}
        for y in years:
            row[y] = fy_get(series, y)
        for q in (quarters or []):
            row[q] = series.get(q)   # exact key lookup for quarterly
        out[metric] = row
    return out


# ─── Reverse DCF (TV Multiple) ───────────────────────────────────────────────

def reverse_dcf(market_cap: float, current_fcf: Optional[float],
                discount_rate: float, tv_dollar: Optional[float],
                years: int) -> Optional[dict]:
    """
    Solve for the required average annual FCF over `years` to justify the market cap.

      market_cap = avg_FCF × annuity_factor  +  TV_PV
      avg_FCF    = (market_cap - TV_PV) / annuity_factor

    TV_PV = tv_dollar / (1+r)^N   if tv_dollar provided, else 0.
    """
    if not market_cap:
        return None

    r              = discount_rate
    annuity_factor = (1 - (1 + r) ** -years) / r
    tv_pv          = (tv_dollar / (1 + r) ** years) if tv_dollar else 0.0
    numerator      = market_cap - tv_pv

    if annuity_factor <= 0 or numerator <= 0:
        return None

    avg_fcf = numerator / annuity_factor

    return {
        "avg_fcf":        avg_fcf,
        "current_fcf":    current_fcf,
        "tv_pv":          tv_pv,
        "vs_current":     round(avg_fcf / current_fcf, 2) if current_fcf else None,
        "entry_multiple": round(market_cap / current_fcf, 1) if current_fcf else None,
        "annuity_pct":    round(numerator / market_cap * 100, 1),
        "tv_pct":         round(tv_pv / market_cap * 100, 1),
    }


# ─── Value Investors Club ────────────────────────────────────────────────────

VIC_BASE = "https://www.valueinvestorsclub.com"
VIC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.valueinvestorsclub.com/ideas",
    "Content-Type": "application/x-www-form-urlencoded",
}


def fetch_vic_ideas(ticker: str) -> list[dict]:
    """
    Search Value Investors Club for writeups matching a ticker.
    Returns [{"comp", "symbol", "date", "long", "url"}] sorted newest-first.

    VIC exposes a POST /search endpoint (discovered from their autocomplete JS).
    Results are filtered to exact ticker matches across all common format variants.
    """
    t = ticker.upper().strip()

    # Determine the search query (what we POST to VIC's autocomplete endpoint).
    # For BRK.B we search "BRK"; for plain MSFT we search "MSFT".
    stem = t.split(".")[0].split("-")[0]   # "BRK" from "BRK.B", "MSFT" from "MSFT"
    query = stem

    # Build the set of ticker symbols we'll ACCEPT in results.
    # We never include the bare stem when the ticker has a class/suffix (avoids
    # matching unrelated tickers — e.g. ASX:BRK "Brookside Energy" when searching BRK.B).
    has_suffix = t != stem
    if has_suffix:
        # Exact-match variants for tickers like BRK.B, BRK-B, BRK/B
        variants = {t,
                    t.replace(".", "-"), t.replace(".", "/"),
                    t.replace("-", "."), t.replace("-", "/")}
    else:
        # No suffix: also accept the bare stem and stripped forms
        variants = {t, t.replace(".", "-"), t.replace(".", "/"),
                    t.replace("-", "."), t.replace("-", "/"),
                    t.replace(".", ""), t.replace("-", "")}

    try:
        r = requests.post(
            f"{VIC_BASE}/search",
            data={"query": query, "tab": "ideas"},
            headers=VIC_HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    if not data.get("success"):
        return []

    ideas = []
    seen = set()
    for item in data.get("result", []):
        sym = (item.get("symbol") or "").upper().strip()
        if sym not in variants:
            continue
        link = item.get("link", "")
        if not link or link in seen:
            continue
        seen.add(link)
        ideas.append({
            "comp":   item.get("comp", "").strip(),
            "symbol": sym,
            "date":   item.get("add_date", ""),
            "long":   bool(item.get("l", 1)),
            "url":    VIC_BASE + link,
        })

    # Sort newest-first (dates are M/D/YYYY strings)
    def _date_key(idea):
        try:
            return datetime.strptime(idea["date"], "%m/%d/%Y")
        except ValueError:
            return datetime.min

    ideas.sort(key=_date_key, reverse=True)
    return ideas


# ─── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = app.make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/vic")
def vic():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    ideas = fetch_vic_ideas(ticker)
    return jsonify({"ticker": ticker, "ideas": ideas})


@app.route("/api/analyze")
def analyze():
    ticker        = request.args.get("ticker", "").upper().strip()
    discount_rate = float(request.args.get("discount_rate", 0.10))
    tv_raw        = request.args.get("tv")
    tv_dollar     = float(tv_raw) if tv_raw else None   # user passes raw dollars (already converted in JS)

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    if discount_rate <= 0 or discount_rate >= 1:
        return jsonify({"error": "Discount rate must be between 0 and 1 (e.g. 0.10)"}), 400
    if tv_dollar is not None and tv_dollar < 0:
        return jsonify({"error": "Terminal value must be a positive dollar amount"}), 400

    cik = resolve_cik(ticker)
    if not cik:
        return jsonify({"error": f"Ticker '{ticker}' not found in SEC database"}), 404

    try:
        submissions = fetch_submissions(cik)
        facts       = fetch_company_facts(cik)
    except requests.HTTPError as e:
        return jsonify({"error": f"EDGAR error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Fetch error: {e}"}), 500

    financials = build_financials(facts)
    if not financials:
        return jsonify({"error": "No XBRL financial data found for this company"}), 404

    # Build list of all 10-K filings (used for filing links and BRK share extraction)
    all_10k_filings = all_filing_infos_from_submissions(
        submissions, {"10-K", "10-K/A", "20-F", "20-F/A"}, max_count=25)
    all_10q_filings = all_filing_infos_from_submissions(
        submissions, {"10-Q", "10-Q/A"}, max_count=20)
    latest_10k = all_10k_filings[0] if all_10k_filings else None
    latest_10q = all_10q_filings[0] if all_10q_filings else None

    # Build {fiscal_year: 10-K URL} map for the frontend link headers
    filing_links: dict[str, str] = {
        f["fiscal_year"]: f["url"]
        for f in all_10k_filings
        if f.get("fiscal_year")
    }

    normalized_ticker = ticker.replace("-", ".")
    if normalized_ticker in {"BRK.A", "BRK.B"}:
        # Seed from XBRL facts (covers ~2011-2014 in class-A × 1500)
        equivalent_b_from_facts = extract_berkshire_equivalent_b_shares_from_facts(facts)
        if equivalent_b_from_facts:
            financials["shares_outstanding_end"] = equivalent_b_from_facts

        # Loop through ALL historical 10-K filings to fill every missing year.
        # Pattern 2 in each filing gives the 3-year table; Pattern 1 gives the
        # cover-page point-in-time count.  We stop when all display years covered.
        min_year = datetime.now().year - MAX_YEARS
        # Cache fetched HTML text keyed by filing URL so we don't fetch twice
        _brk_html_cache: dict[str, str] = {}

        def _get_brk_text(filing: dict) -> str:
            url = filing["url"]
            if url not in _brk_html_cache:
                _brk_html_cache[url] = quick_filing_text(sec_get_text(url))
            return _brk_html_cache[url]

        for filing in all_10k_filings:
            fy = filing.get("fiscal_year", "")
            if not fy or int(fy) < min_year:
                break
            fy_int = int(fy)
            existing_sh = financials.get("shares_outstanding_end", {})
            existing_td = financials.get("total_debt", {})
            existing_tc = financials.get("total_cash", {})
            existing_cash = financials.get("cash", {})
            existing_st = financials.get("short_term_investments", {})
            # Check what's missing for this filing's years (current + prior)
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
                text = _get_brk_text(filing)

                # ── Regex extraction ──────────────────────────────────────────
                if True:
                    if not sh_covered:
                        equiv_b = extract_berkshire_equivalent_b_shares(
                            text, filing["filing_date"])
                        if equiv_b:
                            financials["shares_outstanding_end"] = {**existing_sh, **equiv_b}
                    if not td_covered:
                        debt = extract_berkshire_total_debt(text, filing["filing_date"])
                        if debt:
                            merged_td = dict(existing_td)
                            for d, v in debt.items():
                                if d not in merged_td:
                                    merged_td[d] = v
                            financials["total_debt"]     = merged_td
                            financials["long_term_debt"] = merged_td
                    if not tc_covered or not cash_components_covered:
                        cash_parts = extract_berkshire_cash_components(text, filing["filing_date"])
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

                    # BRK operating earnings (before investment gains/taxes)
                    oe_reported = extract_berkshire_operating_earnings(text, filing["filing_date"])
                    if oe_reported:
                        existing_brk_oe = financials.get("brk_operating_earnings", {})
                        merged_brk_oe = dict(existing_brk_oe)
                        for d, v in oe_reported.items():
                            if d not in merged_brk_oe:
                                merged_brk_oe[d] = v
                        financials["brk_operating_earnings"] = merged_brk_oe

            except Exception:
                continue

        # The XBRL shares_diluted_wtd for BRK is in class-A units (~1-2M).
        # Remove it so the recompute below uses shares_outstanding_end (B-equiv) instead.
        financials.pop("shares_diluted_wtd", None)
        # XBRL EPS for BRK is also class-A basis (~$8000-$60000/share).
        # Clear it so recompute derives per-B-share EPS from NI / B-equiv shares.
        financials.pop("eps_diluted", None)

    shares_end_series = financials.get("shares_outstanding_end", {})
    shares_diluted_series = financials.get("shares_diluted_wtd", {})
    net_income_series = financials.get("net_income", {})
    eps_diluted_series = financials.get("eps_diluted", {})
    if shares_diluted_series:
        backfilled_shares = dict(shares_end_series)
        for date_str, diluted_shares in shares_diluted_series.items():
            year = date_str[:4]
            if fy_get(backfilled_shares, year) is None and diluted_shares:
                backfilled_shares[f"{year}-12-31"] = diluted_shares
        shares_end_series = backfilled_shares

    if net_income_series and eps_diluted_series:
        backfilled_shares = dict(shares_end_series)
        all_years = {d[:4] for d in net_income_series} | {d[:4] for d in eps_diluted_series}
        for year in sorted(all_years):
            if fy_get(backfilled_shares, year) is not None:
                continue
            ni = fy_get(net_income_series, year)
            eps = fy_get(eps_diluted_series, year)
            if ni is None or eps in (None, 0):
                continue
            derived_shares = ni / eps
            if derived_shares > 0:
                backfilled_shares[f"{year}-12-31"] = derived_shares
        shares_end_series = backfilled_shares

    # Override with EntityCommonStockSharesOutstanding (cover-page count, filed weeks
    # after fiscal year-end) where available.  It is more current and authoritative
    # than the balance-sheet-date CommonStockSharesOutstanding.  We key it by the
    # `fy` field so it lands in the right fiscal year regardless of the filing date.
    _entity_shares_tag = (
        facts.get("facts", {}).get("us-gaap", {}).get("EntityCommonStockSharesOutstanding")
        or facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding")
    )
    if _entity_shares_tag:
        for _e in _entity_shares_tag.get("units", {}).get("shares", []):
            if _e.get("form") not in {"10-K", "10-K/A", "20-F", "20-F/A"}:
                continue
            _fy = _e.get("fy"); _val = _e.get("val")
            if _fy and _val:
                shares_end_series[f"{_fy}-12-31"] = _val   # cover-page value wins

    financials["shares_outstanding_end"] = shares_end_series

    # Recompute per-share metrics now that the authoritative share series is set.
    # This matters for BRK and other companies where shares are injected after
    # build_financials() runs (which computed these with potentially empty shares).
    _so  = financials.get("shares_outstanding_end", {})
    _sd  = financials.get("shares_diluted_wtd", {})
    _sb  = _sd or _so   # share base
    _rev = financials.get("revenue", {})
    _fcf = financials.get("fcf", {})
    _ni  = financials.get("net_income", {})
    _eq  = financials.get("equity", {})
    if _sb:
        if _rev:
            financials["revenue_per_share"] = {
                d: _rev[d] / fy_get(_sb, d[:4])
                for d in _rev if fy_get(_sb, d[:4])
            } or financials.get("revenue_per_share")
        if _fcf:
            financials["fcf_per_share"] = {
                d: _fcf[d] / fy_get(_sb, d[:4])
                for d in _fcf if fy_get(_sb, d[:4])
            } or financials.get("fcf_per_share")
        if _ni:
            # Fill in EPS for any year not already covered, or overwrite if absent entirely
            existing_eps = financials.get("eps_diluted", {})
            existing_years = {k[:4] for k in existing_eps}
            extra_eps = {
                d: _ni[d] / fy_get(_sb, d[:4])
                for d in _ni
                if d[:4] not in existing_years and fy_get(_sb, d[:4])
            }
            if extra_eps:
                financials["eps_diluted"] = {**existing_eps, **extra_eps} or None
            elif not existing_eps:
                financials["eps_diluted"] = None
    if _so and _eq:
        financials["book_value_per_share"] = {
            d: _eq[d] / fy_get(_so, d[:4])
            for d in _eq if fy_get(_so, d[:4])
        } or financials.get("book_value_per_share")

    # Recompute total_cash from component series when available, then refresh net_cash.
    _cash = financials.get("cash", {})
    _st   = financials.get("short_term_investments", {})
    if _cash or _st:
        all_d = set(_cash) | set(_st)
        combined_cash = {
            d: (fy_get(_cash, d[:4]) or 0) + (fy_get(_st, d[:4]) or 0)
            for d in all_d
        }
        existing_tc = financials.get("total_cash", {})
        financials["total_cash"] = {**existing_tc, **combined_cash} if existing_tc else combined_cash

    # Recompute net_cash after any BRK debt/cash injections
    _tc = financials.get("total_cash", {})
    _td = financials.get("total_debt", {})
    if _tc and _td:
        all_d = set(_tc) | set(_td)
        financials["net_cash"] = {
            d: (fy_get(_tc, d[:4]) or 0) - (fy_get(_td, d[:4]) or 0)
            for d in all_d
        }

    years = get_display_years(financials)

    # ── Post-annual quarterly data ────────────────────────────────────────────
    # Anchor quarter discovery on the last display year (latY) so labels and
    # data are always consistent.  We look for an annual date whose year matches
    # latY in any series; fall back to latY-12-31 if none is found.
    _last_annual_date = ""
    _lat_y = years[-1] if years else ""
    for _ref_series in (financials.get("revenue", {}), financials.get("net_income", {}),
                        financials.get("equity", {}), financials.get("total_assets", {})):
        _annual_dates = [d for d in _ref_series if not d.startswith("Q") and d[:4] == _lat_y]
        if _annual_dates:
            _last_annual_date = max(_annual_dates)
            break
    if not _last_annual_date and _lat_y:
        # Fall back: construct a Dec-31 anchor from the display year
        _last_annual_date = f"{_lat_y}-12-31"

    # Metrics to extract for quarterly view
    _quarterly_flow_keys = {
        "revenue", "gross_profit", "operating_income", "net_income",
        "operating_cash_flow", "capex", "depreciation", "stock_based_compensation",
        "income_tax", "interest_expense", "investment_gains",
        "dividends_paid", "buybacks_value",
        # BDC flow metrics
        "net_investment_income", "gross_investment_income", "nii_per_share",
        # Bank flow metrics
        "interest_income", "net_interest_income", "noninterest_income",
        "noninterest_expense", "provision_for_losses",
    }
    _quarterly_bs_keys = {
        "total_assets", "current_assets", "current_liabilities", "equity",
        "cash", "short_term_investments", "long_term_debt", "current_debt",
        "total_liabilities", "goodwill", "inventory", "shares_outstanding_end",
        # BDC point-in-time
        "nav_per_share",
    }
    _point_in_time_metrics = {
        "total_assets", "current_assets", "current_liabilities", "total_liabilities",
        "equity", "cash", "short_term_investments", "long_term_debt", "current_debt",
        "goodwill", "intangibles", "inventory", "shares_outstanding_end", "buyback_remaining",
        # BDC point-in-time
        "nav_per_share",
    }

    # quarter_end_dates: {"Q1": "YYYY-MM-DD", ...}  for display labels
    # quarter_filing_links: {"Q1": "https://...", ...}  for header links
    quarter_end_dates:    dict[str, str] = {}
    quarter_filing_links: dict[str, str] = {}

    if _last_annual_date:
        _last_dt = datetime.strptime(_last_annual_date, "%Y-%m-%d")

        # Discover the actual quarter-end dates.
        # Try revenue first; fall back through a chain of common metrics so banks
        # (no revenue tag), BDCs (no revenue), and other non-standard filers all work.
        _rev_tags = METRIC_TAGS.get("revenue", [])
        _q_end_dates = _discover_quarter_end_dates(facts, _rev_tags, _last_dt)
        if not _q_end_dates:
            for _fallback_key in (
                "gross_investment_income", "net_investment_income",  # BDCs
                "net_income", "operating_income", "operating_cash_flow",  # banks / others
            ):
                _q_end_dates = _discover_quarter_end_dates(facts, METRIC_TAGS.get(_fallback_key, []), _last_dt)
                if _q_end_dates:
                    break
        quarter_end_dates = {f"Q{i+1}": d for i, d in enumerate(_q_end_dates[:3])}

        # Match each quarter-end date to its 10-Q filing URL
        # 10-Q report_date is typically within a few days of the quarter end
        _10q_by_date: dict[str, str] = {
            f["report_date"]: f["url"]
            for f in all_10q_filings
            if f.get("report_date")
        }
        quarter_filing_links: dict[str, str] = {}
        for qk, qdate in quarter_end_dates.items():
            # Try exact match first, then nearest date within 7 days
            if qdate in _10q_by_date:
                quarter_filing_links[qk] = _10q_by_date[qdate]
            else:
                try:
                    qdt = datetime.strptime(qdate, "%Y-%m-%d")
                    best = min(
                        _10q_by_date.items(),
                        key=lambda kv: abs((datetime.strptime(kv[0], "%Y-%m-%d") - qdt).days),
                        default=None,
                    )
                    if best and abs((datetime.strptime(best[0], "%Y-%m-%d") - qdt).days) <= 14:
                        quarter_filing_links[qk] = best[1]
                except Exception:
                    pass

        for key in _quarterly_flow_keys | _quarterly_bs_keys:
            tags = METRIC_TAGS.get(key)
            if not tags:
                continue
            is_bs = key in _point_in_time_metrics
            q_vals = extract_post_annual_quarters(facts, tags, _last_annual_date, is_bs)
            if q_vals:
                existing = financials.setdefault(key, {})
                for qk, v in q_vals.items():
                    existing[qk] = v

        # ── Cover-page shares override for quarterly periods ─────────────────
        # EntityCommonStockSharesOutstanding filed with 10-Q uses the *filing*
        # date as its end-date, not the quarter-end date, so we can't do an
        # exact match.  Instead, parse each candidate's end-date and assign it
        # to the nearest quarter-end within 90 days (filing dates are typically
        # 30-45 days after quarter-end).
        if _entity_shares_tag:
            _qdate_parsed = {
                qk: datetime.strptime(qd, "%Y-%m-%d")
                for qk, qd in quarter_end_dates.items()
            }
            _so_series = financials.setdefault("shares_outstanding_end", {})
            # Collect best (most recent end-date) cover-page value per Q-key
            _qk_best: dict[str, tuple[datetime, float]] = {}
            for _e in _entity_shares_tag.get("units", {}).get("shares", []):
                if _e.get("form") not in {"10-Q", "10-Q/A"}:
                    continue
                _end = _e.get("end", "")
                _val = _e.get("val")
                if not _end or not _val:
                    continue
                try:
                    _end_dt = datetime.strptime(_end, "%Y-%m-%d")
                except Exception:
                    continue
                # Find nearest quarter-end within 90 days (filing always after quarter-end)
                _best_qk, _best_diff = None, 999
                for _qk, _qdt in _qdate_parsed.items():
                    _diff = (_end_dt - _qdt).days
                    if 0 <= _diff <= 90 and _diff < _best_diff:
                        _best_qk, _best_diff = _qk, _diff
                if _best_qk:
                    prev = _qk_best.get(_best_qk)
                    if prev is None or _end_dt > prev[0]:
                        _qk_best[_best_qk] = (_end_dt, _val)
            for _qk, (_, _val) in _qk_best.items():
                _so_series[_qk] = _val   # cover-page quarterly wins

        # ── BRK-specific quarterly overrides ────────────────────────────────
        if normalized_ticker in {"BRK.A", "BRK.B"}:
            # 1. Cash: XBRL double-counts (combined tag + T-bill tag). Replace with
            #    text extraction from the 10-Q HTML, same as we do for 10-Ks.
            for qk, qdate in quarter_end_dates.items():
                qurl = quarter_filing_links.get(qk)
                if not qurl:
                    continue
                try:
                    _q_text = quick_filing_text(sec_get_text(qurl))
                    _q_cash_parts = extract_brk_quarterly_cash(_q_text, qk, qdate)
                    for _subkey in ("cash", "short_term_investments", "total_cash"):
                        if _q_cash_parts.get(_subkey):
                            financials.setdefault(_subkey, {}).update(_q_cash_parts[_subkey])
                    _q_debt = extract_brk_quarterly_debt(_q_text, qk)
                    if _q_debt:
                        for _dk in ("total_debt", "long_term_debt"):
                            financials.setdefault(_dk, {}).update(_q_debt)
                except Exception:
                    pass

            # 2. Shares: XBRL 10-Q files class-A counts only (~1.4M) — useless for
            #    per-B-share math. Forward-fill the last annual B-equivalent share count
            #    into every Q-key that doesn't already have a valid value.
            _brk_so = financials.get("shares_outstanding_end", {})
            _annual_so_dates = sorted(d for d in _brk_so if not d.startswith("Q"))
            if _annual_so_dates:
                _last_annual_so = _brk_so[_annual_so_dates[-1]]
                for qk in quarter_end_dates:
                    existing_v = _brk_so.get(qk)
                    # Replace if missing or clearly class-A scale (< 100M shares)
                    if not existing_v or abs(existing_v) < 1e8:
                        _brk_so[qk] = _last_annual_so

        # Quarterly derived balance sheet metrics
        _q_cash = {k: v for k, v in financials.get("cash", {}).items() if k.startswith("Q")}
        _q_st   = {k: v for k, v in financials.get("short_term_investments", {}).items() if k.startswith("Q")}
        _q_ltd  = {k: v for k, v in financials.get("long_term_debt", {}).items() if k.startswith("Q")}
        _q_ctd  = {k: v for k, v in financials.get("current_debt", {}).items() if k.startswith("Q")}
        _q_ca   = {k: v for k, v in financials.get("current_assets", {}).items() if k.startswith("Q")}
        _q_cl   = {k: v for k, v in financials.get("current_liabilities", {}).items() if k.startswith("Q")}

        if _q_cash or _q_st:
            _tc_q = financials.setdefault("total_cash", {})
            for qk in set(_q_cash) | set(_q_st):
                _tc_q[qk] = (_q_cash.get(qk) or 0) + (_q_st.get(qk) or 0)

        if _q_ltd or _q_ctd:
            _td_q = financials.setdefault("total_debt", {})
            for qk in set(_q_ltd) | set(_q_ctd):
                _td_q[qk] = (_q_ltd.get(qk) or 0) + (_q_ctd.get(qk) or 0)

        _q_tc_all = {k: v for k, v in financials.get("total_cash", {}).items() if k.startswith("Q")}
        _q_td_all = {k: v for k, v in financials.get("total_debt", {}).items() if k.startswith("Q")}
        if _q_tc_all or _q_td_all:
            _nc_q = financials.setdefault("net_cash", {})
            for qk in set(_q_tc_all) | set(_q_td_all):
                _nc_q[qk] = (_q_tc_all.get(qk) or 0) - (_q_td_all.get(qk) or 0)

        if _q_ca and _q_cl:
            _wc_q = financials.setdefault("working_capital", {})
            for qk in set(_q_ca) & set(_q_cl):
                _wc_q[qk] = _q_ca[qk] - _q_cl[qk]

        # Quarterly FCF = quarterly OCF - quarterly CapEx
        ocf_q = {k: v for k, v in financials.get("operating_cash_flow", {}).items() if k.startswith("Q")}
        cpx_q = {k: v for k, v in financials.get("capex", {}).items() if k.startswith("Q")}
        if ocf_q:
            fcf_existing = financials.setdefault("fcf", {})
            for qk, ocf_v in ocf_q.items():
                fcf_existing[qk] = ocf_v - abs(cpx_q.get(qk) or 0)

        # Quarterly margins
        rev_q  = {k: v for k, v in financials.get("revenue", {}).items() if k.startswith("Q")}
        # For banks: derive quarterly revenue from NII + non-interest income if absent
        if not rev_q:
            _q_nii_r = {k: v for k, v in financials.get("net_interest_income", {}).items() if k.startswith("Q")}
            _q_noni_r = {k: v for k, v in financials.get("noninterest_income", {}).items() if k.startswith("Q")}
            if _q_nii_r or _q_noni_r:
                _rev_bank_q = {}
                for qk in set(_q_nii_r) | set(_q_noni_r):
                    _rev_bank_q[qk] = (_q_nii_r.get(qk) or 0) + (_q_noni_r.get(qk) or 0)
                if _rev_bank_q:
                    financials.setdefault("revenue", {}).update(_rev_bank_q)
                    rev_q = _rev_bank_q
        for num_key, out_key in [
            ("gross_profit",    "gross_margin"),
            ("operating_income","operating_margin"),
            ("net_income",      "net_margin"),
            ("fcf",             "fcf_margin"),
        ]:
            num_q = {k: v for k, v in financials.get(num_key, {}).items() if k.startswith("Q")}
            if num_q and rev_q:
                mg = financials.setdefault(out_key, {})
                for qk, nv in num_q.items():
                    dv = rev_q.get(qk)
                    if dv and dv > 0:
                        mg[qk] = nv / dv

        # Quarterly bank metrics: NIM and Efficiency Ratio
        _q_nii_bank  = {k: v for k, v in financials.get("net_interest_income", {}).items() if k.startswith("Q")}
        _q_int_inc   = {k: v for k, v in financials.get("interest_income",     {}).items() if k.startswith("Q")}
        _q_int_exp   = {k: v for k, v in financials.get("interest_expense",    {}).items() if k.startswith("Q")}
        _q_noni      = {k: v for k, v in financials.get("noninterest_income",  {}).items() if k.startswith("Q")}
        _q_none_exp  = {k: v for k, v in financials.get("noninterest_expense", {}).items() if k.startswith("Q")}
        _q_ta_bank   = {k: v for k, v in financials.get("total_assets",        {}).items() if k.startswith("Q")}

        # Fill net_interest_income Q-keys if missing but components exist
        if not _q_nii_bank and _q_int_inc and _q_int_exp:
            _nii_bk = financials.setdefault("net_interest_income", {})
            for qk in set(_q_int_inc) & set(_q_int_exp):
                _nii_bk[qk] = _q_int_inc[qk] - abs(_q_int_exp[qk])
            _q_nii_bank = {k: v for k, v in _nii_bk.items() if k.startswith("Q")}

        if _q_nii_bank and _q_ta_bank:
            _nim_q = financials.setdefault("nim", {})
            for qk in set(_q_nii_bank) & set(_q_ta_bank):
                if _q_ta_bank[qk] and _q_ta_bank[qk] != 0:
                    _nim_q[qk] = (_q_nii_bank[qk] * 4) / _q_ta_bank[qk]  # annualised

        if _q_none_exp and _q_nii_bank:
            _eff_q = financials.setdefault("efficiency_ratio", {})
            for qk in set(_q_none_exp) & (set(_q_nii_bank) | set(_q_noni)):
                nii_v  = _q_nii_bank.get(qk) or 0
                noni_v = _q_noni.get(qk)      or 0
                denom  = nii_v + noni_v
                if denom and denom != 0:
                    _eff_q[qk] = abs(_q_none_exp[qk]) / denom

        # Quarterly ROE, FCF ROE, ROA (annualized: × 4)
        _q_ni  = {k: v for k, v in financials.get("net_income", {}).items() if k.startswith("Q")}
        _q_fcf = {k: v for k, v in financials.get("fcf", {}).items() if k.startswith("Q")}
        _q_eq  = {k: v for k, v in financials.get("equity", {}).items() if k.startswith("Q")}
        _q_ta  = {k: v for k, v in financials.get("total_assets", {}).items() if k.startswith("Q")}
        if _q_ni and _q_eq:
            _roe_q = financials.setdefault("roe", {})
            for qk in set(_q_ni) & set(_q_eq):
                if _q_eq[qk] and _q_eq[qk] != 0:
                    _roe_q[qk] = (_q_ni[qk] * 4) / _q_eq[qk]
        if _q_fcf and _q_eq:
            _fcfroe_q = financials.setdefault("fcf_roe", {})
            for qk in set(_q_fcf) & set(_q_eq):
                if _q_eq[qk] and _q_eq[qk] != 0:
                    _fcfroe_q[qk] = (_q_fcf[qk] * 4) / _q_eq[qk]
        if _q_ni and _q_ta:
            _roa_q = financials.setdefault("roa", {})
            for qk in set(_q_ni) & set(_q_ta):
                if _q_ta[qk] and _q_ta[qk] != 0:
                    _roa_q[qk] = (_q_ni[qk] * 4) / _q_ta[qk]
        if _q_ni and _q_eq:
            _q_gw_r = {k: v for k, v in financials.get("goodwill",   {}).items() if k.startswith("Q")}
            _q_ia_r = {k: v for k, v in financials.get("intangibles", {}).items() if k.startswith("Q")}
            _rote_q = financials.setdefault("rote", {})
            for qk in set(_q_ni) & set(_q_eq):
                te = (_q_eq.get(qk) or 0) - (_q_gw_r.get(qk) or 0) - (_q_ia_r.get(qk) or 0)
                if te and te != 0:
                    _rote_q[qk] = (_q_ni[qk] * 4) / te

        # Quarterly NII ROE (BDC)
        _q_nii_flow = {k: v for k, v in financials.get("net_investment_income", {}).items() if k.startswith("Q")}
        _q_eq_bdc   = {k: v for k, v in financials.get("equity", {}).items() if k.startswith("Q")}
        if _q_nii_flow and _q_eq_bdc:
            _nii_roe_q = financials.setdefault("nii_roe", {})
            for qk in set(_q_nii_flow) & set(_q_eq_bdc):
                if _q_eq_bdc[qk] and _q_eq_bdc[qk] != 0:
                    _nii_roe_q[qk] = (_q_nii_flow[qk] * 4) / _q_eq_bdc[qk]

        # Quarterly per-share metrics
        # Use diluted weighted-avg shares if available, else period-end shares
        _q_sd = {k: v for k, v in financials.get("shares_diluted_wtd", {}).items() if k.startswith("Q")}
        _q_so = {k: v for k, v in financials.get("shares_outstanding_end", {}).items() if k.startswith("Q")}
        _q_sb = _q_sd or _q_so  # share base for per-share calcs

        _q_rev = {k: v for k, v in financials.get("revenue", {}).items() if k.startswith("Q")}
        _q_fcf2 = {k: v for k, v in financials.get("fcf", {}).items() if k.startswith("Q")}
        _q_ni2  = {k: v for k, v in financials.get("net_income", {}).items() if k.startswith("Q")}
        _q_eq2  = {k: v for k, v in financials.get("equity", {}).items() if k.startswith("Q")}

        if _q_sb:
            if _q_rev:
                _rps_q = financials.setdefault("revenue_per_share", {})
                for qk in set(_q_rev) & set(_q_sb):
                    if _q_sb[qk]:
                        _rps_q[qk] = _q_rev[qk] / _q_sb[qk]
            if _q_fcf2:
                _fps_q = financials.setdefault("fcf_per_share", {})
                for qk in set(_q_fcf2) & set(_q_sb):
                    if _q_sb[qk]:
                        _fps_q[qk] = _q_fcf2[qk] / _q_sb[qk]
            if _q_ni2:
                existing_eps_q = financials.get("eps_diluted", {})
                _eps_q = {}
                for qk in set(_q_ni2) & set(_q_sb):
                    if _q_sb[qk] and qk not in existing_eps_q:
                        _eps_q[qk] = _q_ni2[qk] / _q_sb[qk]
                if _eps_q:
                    financials.setdefault("eps_diluted", {}).update(_eps_q)

        if _q_so and _q_eq2:
            _bvps_q = financials.setdefault("book_value_per_share", {})
            for qk in set(_q_eq2) & set(_q_so):
                if _q_so[qk]:
                    _bvps_q[qk] = _q_eq2[qk] / _q_so[qk]

        # Quarterly ROIC
        _q_oi  = {k: v for k, v in financials.get("operating_income", {}).items() if k.startswith("Q")}
        _q_td2 = {k: v for k, v in financials.get("total_debt", {}).items() if k.startswith("Q")}
        _q_eq3 = {k: v for k, v in financials.get("equity", {}).items() if k.startswith("Q")}
        if _q_oi and (_q_td2 or _q_eq3):
            _roic_q = financials.setdefault("roic", {})
            for qk in set(_q_oi) & (set(_q_td2) | set(_q_eq3)):
                ic = (_q_td2.get(qk) or 0) + (_q_eq3.get(qk) or 0)
                if ic and ic != 0:
                    _roic_q[qk] = (_q_oi[qk] * 4 * (1 - 0.21)) / ic

        # Quarterly UNTA, NOPAT, Economic Goodwill
        _q_tc_q  = {k: v for k, v in financials.get("total_cash", {}).items() if k.startswith("Q")}
        _q_gw    = {k: v for k, v in financials.get("goodwill",   {}).items() if k.startswith("Q")}
        _q_ia    = {k: v for k, v in financials.get("intangibles",{}).items() if k.startswith("Q")}
        if _q_oi and (_q_eq3 or _q_td2):
            _unta_q = financials.setdefault("unta", {})
            for qk in set(_q_eq3) | set(_q_td2):
                e      = _q_eq3.get(qk) or 0
                debt   = _q_td2.get(qk) or 0
                cash_v = _q_tc_q.get(qk) or 0
                gw_v   = _q_gw.get(qk)   or 0
                ia_v   = _q_ia.get(qk)   or 0
                _unta_q[qk] = e + debt - cash_v - gw_v - ia_v

            _nopat_q = financials.setdefault("nopat", {})
            for qk in set(_q_oi):
                _nopat_q[qk] = _q_oi[qk] * (1 - 0.21) * 4   # annualised

            _unta_vals = {k: v for k, v in financials.get("unta", {}).items() if k.startswith("Q")}
            if _unta_vals:
                _eco_q = financials.setdefault("economic_goodwill", {})
                for qk in set(_nopat_q) & set(_unta_vals):
                    if _unta_vals[qk] and _unta_vals[qk] != 0:
                        _eco_q[qk] = _nopat_q[qk] / _unta_vals[qk]

        # Quarterly BDC: NII per share (if not already from XBRL tag, compute from NII / shares)
        _q_nii = {k: v for k, v in financials.get("net_investment_income", {}).items() if k.startswith("Q")}
        _q_nii_ps = {k: v for k, v in financials.get("nii_per_share", {}).items() if k.startswith("Q")}
        if _q_nii and not _q_nii_ps and _q_sb:
            _nii_ps_q = financials.setdefault("nii_per_share", {})
            for qk in set(_q_nii) & set(_q_sb):
                if _q_sb[qk]:
                    _nii_ps_q[qk] = _q_nii[qk] / _q_sb[qk]

    quarters = [f"Q{i+1}" for i in range(len(quarter_end_dates))]
    fin_data = serialize(financials, years, quarters)

    # ── Market price (Yahoo) + market stats (beta, 52w) ─────────────────────
    market = get_market_data(ticker)
    price  = market.get("price")

    # Shares outstanding: prefer the most-recent quarterly cover-page figure
    # (Q3 > Q2 > Q1 of the latest fiscal year) over the annual 10-K figure,
    # since the quarterly count is more current.
    latest_yr = years[-1] if years else None
    so_series = financials.get("shares_outstanding_end", {})
    edgar_shares = None
    for _qk in ("Q3", "Q2", "Q1"):
        if so_series.get(_qk):
            edgar_shares = so_series[_qk]
            break
    if edgar_shares is None:
        edgar_shares = fy_get(so_series, latest_yr) if latest_yr else None

    # Market cap = current price × most-recent EDGAR share count
    mktcap = (price * edgar_shares) if (price and edgar_shares) else None

    # ── Valuation multiples ──────────────────────────────────────────────────
    latest = years[-1] if years else None
    def L(key): return fy_get(financials.get(key, {}), latest) if latest else None

    current_fcf  = L("fcf")
    current_ni   = L("net_income")
    current_rev  = L("revenue")
    current_ebitda = L("ebitda")
    current_tc   = L("total_cash")
    current_td   = L("total_debt")

    ev = (mktcap + (current_td or 0) - (current_tc or 0)) if mktcap else None

    def mult(num, den): return round(num / den, 2) if num and den and den > 0 else None

    current_eq = L("equity")
    multiples = {
        "pe":           mult(mktcap, current_ni),
        "p_fcf":        mult(mktcap, current_fcf),
        "p_s":          mult(mktcap, current_rev),
        "p_b":          mult(mktcap, current_eq),
        "ev_ebitda":    mult(ev, current_ebitda),
        "earnings_yield": round(current_ni / mktcap, 4) if current_ni and mktcap else None,
        "fcf_yield":    round(current_fcf / mktcap, 4) if current_fcf and mktcap else None,
    }

    # ── Buyback remaining ────────────────────────────────────────────────────
    br_series = financials.get("buyback_remaining", {})
    buyback_remaining = None
    if br_series:
        annual_br = [d for d in br_series if not d.startswith("Q")]
        buyback_remaining = br_series.get(max(annual_br)) if annual_br else None

    # ── Historical FCF average for context ───────────────────────────────────
    fcf_series = financials.get("fcf", {})

    def hist_avg_fcf(n):
        if not fcf_series:
            return None
        vals = [(d, v) for d, v in sorted(fcf_series.items())[-n:] if v is not None]
        return sum(v for _, v in vals) / len(vals) if vals else None

    # ── Dividend yield from EDGAR (dividends_per_share / price) ──────────────
    dps_series = financials.get("dividends_per_share", {})
    dividend_yield = None
    if dps_series and price:
        latest_dps = fy_get(dps_series, latest_yr) if latest_yr else None
        if latest_dps and price > 0:
            dividend_yield = latest_dps / price

    # ── Reverse DCF scenarios ────────────────────────────────────────────────
    dcf_scenarios = {}
    if mktcap:
        for h in [5, 10, 20]:
            res = reverse_dcf(mktcap, current_fcf, discount_rate, tv_dollar, h)
            if res:
                res["hist_avg_fcf"] = hist_avg_fcf(h)
                dcf_scenarios[str(h)] = res

    # ── Fiscal year end month ────────────────────────────────────────────────
    fy_month = "Dec"
    rev_series = financials.get("revenue", {})
    annual_rev_dates = [d for d in rev_series if not d.startswith("Q")]
    if annual_rev_dates:
        latest_rev_date = max(annual_rev_dates)
        m_num = int(latest_rev_date[5:7])
        fy_month = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"][m_num - 1]

    # BDC detection: true if XBRL data contains NetInvestmentIncome entries
    is_bdc  = bool(financials.get("net_investment_income"))
    is_bank = bool(financials.get("net_interest_income") or financials.get("noninterest_expense"))

    company_name = submissions.get("name", ticker)

    # ── Proxy statement link (DEF 14A) ────────────────────────────────────────
    proxy_url, proxy_date = get_proxy_filing_url(submissions)

    # ── Investor relations URL ─────────────────────────────────────────────────
    # EDGAR submissions often carry the company's website; we prefer that and
    # append /investors as a best-effort IR path, then fall back to a Google
    # search so there is always a working link.
    _edgar_website = (submissions.get("website") or "").strip().rstrip("/")
    if _edgar_website:
        ir_url = _edgar_website + "/investors"
    else:
        _ir_search_q = requests.utils.quote(f"{company_name} investor relations")
        ir_url = f"https://www.google.com/search?q={_ir_search_q}"

    # ── Earnings materials (8-Ks + Seeking Alpha links) per quarter ───────────
    earnings_materials = get_earnings_materials(submissions, quarter_end_dates, ticker)

    return jsonify({
        "company": {
            "name":             company_name,
            "ticker":           ticker,
            "cik":              cik,
            "sic":              submissions.get("sic", ""),
            "sic_description":  submissions.get("sicDescription", ""),
            "is_bdc":           is_bdc,
            "is_bank":          is_bank,
            "fiscal_year_end":  fy_month,
            "state":            submissions.get("stateOfIncorporation", ""),
            "latest_10k":       latest_10k,
            "latest_10q":       latest_10q,
        },
        "market": {
            "price":             price,
            "market_cap":        mktcap,
            "enterprise_value":  ev,
            "shares_outstanding": edgar_shares,   # from EDGAR, not Yahoo
            "beta":              market.get("beta"),
            "dividend_yield":    dividend_yield,   # computed from EDGAR DPS / price
            "52w_high":          market.get("52w_high"),
            "52w_low":           market.get("52w_low"),
            "buyback_remaining": buyback_remaining,
            **multiples,
        },
        "years":           years,
        "quarters":        quarters,
        "quarter_dates":   quarter_end_dates,
        "quarter_links":      quarter_filing_links,
        "earnings_materials": earnings_materials,
        "ir_url":             ir_url,
        "financials":         fin_data,
        "filing_links":       filing_links,
        "proxy": {
            "url":  proxy_url,
            "date": proxy_date,
        },
        "dcf": {
            "discount_rate": discount_rate,
            "tv_dollar":     tv_dollar,
            "current_fcf":   current_fcf,
            "market_cap":    mktcap,
            "scenarios":     dcf_scenarios,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050, host="0.0.0.0")
