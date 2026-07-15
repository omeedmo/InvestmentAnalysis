#!/usr/bin/env python3
"""
Value Line Style Investment Analysis – Flask Backend
Pulls 15 years of financial data from SEC EDGAR (free, no API key).
"""

import io
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

import screener

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

    # Strategy 1: cached bulk ticker→CIK map (shared with the screener; disk-cached
    # 24h with a bundled fallback). Avoids a fresh SEC fetch on every lookup, which
    # gets rate-limited from datacenter IPs (e.g. Railway) and made dotted symbols
    # like BRK.B fail there.
    try:
        cik_map = screener.ticker_cik_map()
        for c in candidates:
            if c in cik_map:
                return str(cik_map[c]).zfill(10)
    except Exception:
        pass

    # Strategy 2: EDGAR company search — try each symbol variant (e.g. BRK-B),
    # returns a page whose URL contains the CIK (works for any SEC registrant)
    import re as _re
    for cand in (normalized, normalized.replace(".", "-"), normalized.replace(".", "")):
        try:
            search_url = (
                "https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cand}&type=10-K"
                "&dateb=&owner=include&count=1&search_text="
            )
            r = requests.get(search_url, headers=HEADERS, timeout=15, allow_redirects=True)
            m = _re.search(r"CIK=?(\d{7,10})", r.url + r.text)
            if m:
                return m.group(1).zfill(10)
        except Exception:
            continue

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


def _parse_form4_purchases(cik_no_zero: str, accession_no_dash: str,
                           primary_doc: str, filing_date: str) -> list[dict]:
    """
    Fetch and parse one Form 4 XML, returning only open-market PURCHASES
    (non-derivative transaction code 'P' = bought with the insider's own money).
    Each purchase: owner, title, date, shares, price, value, and Form 4 URL.
    """
    # primaryDocument is the XSL-rendered path (e.g. "xslF345X06/wk-form4_*.xml");
    # the raw XML is the same filename without that prefix.
    raw_doc = primary_doc.split("/")[-1]
    if not raw_doc.lower().endswith(".xml"):
        return []
    xml_url = SEC_ARCHIVES.format(cik_no_zero=cik_no_zero,
                                  accession_no_dash=accession_no_dash, document=raw_doc)
    # Human-readable rendered Form 4 (nice link for the user)
    view_url = SEC_ARCHIVES.format(cik_no_zero=cik_no_zero,
                                   accession_no_dash=accession_no_dash, document=primary_doc)
    # Retry on 429 (SEC rate limit) with backoff — common from datacenter IPs
    # (e.g. Railway). Returning None signals a failed fetch so the caller can tell
    # the user to retry instead of showing a falsely-empty table.
    root = None
    for attempt in range(3):
        try:
            r = requests.get(xml_url, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                time.sleep(0.6 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            root = ET.fromstring(r.text)
            break
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    if root is None:
        return None

    def _t(el, path):
        x = el.find(path)
        return x.text if x is not None else None

    # Only count filings where THIS company is the issuer. An entity that also
    # invests (e.g. Berkshire) files Form 4s as the reporting owner of OTHER
    # companies; those appear in its submissions feed but are purchases of other
    # stocks, not of this company — skip them.
    issuer_cik = _t(root, ".//issuer/issuerCik")
    if issuer_cik and issuer_cik.lstrip("0") != str(cik_no_zero).lstrip("0"):
        return []

    owner = _t(root, ".//reportingOwner/reportingOwnerId/rptOwnerName") or ""
    rel   = root.find(".//reportingOwner/reportingOwnerRelationship")
    title = ""
    if rel is not None:
        if (_t(rel, "isDirector") or "") in ("1", "true"):
            title = "Director"
        if (_t(rel, "isOfficer") or "") in ("1", "true"):
            title = _t(rel, "officerTitle") or "Officer"
        if (_t(rel, "isTenPercentOwner") or "") in ("1", "true") and not title:
            title = "10% Owner"

    out = []
    # Parse both non-derivative and derivative code-P purchases. Some issuers file
    # convertible common (e.g. Berkshire Class A, convertible to Class B) as a
    # DERIVATIVE security, so an open-market buy like Greg Abel's shows up only in
    # <derivativeTransaction>. Both are own-money purchases when the code is 'P'.
    for tx in (root.findall(".//nonDerivativeTransaction")
               + root.findall(".//derivativeTransaction")):
        if _t(tx, "transactionCoding/transactionCode") != "P":
            continue
        try:
            shares = float(_t(tx, "transactionAmounts/transactionShares/value") or 0)
            price  = float(_t(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        except (TypeError, ValueError):
            continue
        if shares <= 0:
            continue
        out.append({
            "owner":  owner.title() if owner.isupper() else owner,
            "title":  title,
            "date":   _t(tx, "transactionDate/value") or filing_date,
            "shares": shares,
            "price":  price,
            "value":  shares * price,
            "security": _t(tx, "securityTitle/value") or "Common Stock",
            "url":    view_url,
        })
    return out


def get_insider_purchases(submissions: dict, months: int = 12,
                          max_filings: int = 300) -> dict:
    """
    Collect insider open-market purchases (Form 4, code 'P') over the last
    `months`, plus a per-year trend. Form 4 XMLs are fetched concurrently.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    accns  = recent.get("accessionNumber", [])
    docs   = recent.get("primaryDocument", [])
    dates  = recent.get("filingDate", [])
    cik_no_zero = str(int(submissions.get("cik", "0")))

    # Only Form 4s FILED within the window. Filing date >= transaction date, so
    # this captures every in-window purchase without pulling years of extra
    # filings (which made the "scanned" count meaningless — it just hit the cap).
    from datetime import timedelta
    horizon = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")

    jobs = []
    for i, f in enumerate(forms):
        if f not in ("4", "4/A"):
            continue
        fdate = dates[i] if i < len(dates) else ""
        if fdate and fdate < horizon:
            continue
        jobs.append((accns[i].replace("-", ""), docs[i], fdate))
        if len(jobs) >= max_filings:
            break

    # Per-filing cache: a Form 4 is immutable once filed, so its parsed result
    # never changes. We cache each successfully-parsed filing by accession and
    # only fetch the ones we don't have yet. If a run is rate-limited partway,
    # the filings we DID get are kept, and the next run resumes from there
    # (fetching only what's still missing) instead of starting over.
    cache = _load_filing_cache(cik_no_zero)          # {accession: [purchase dicts]}
    window_accns = {j[0] for j in jobs}
    to_fetch = [j for j in jobs if j[0] not in cache]

    failures = 0
    if to_fetch:
        # Keep concurrency modest — SEC rate-limits at ~10 requests/second.
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda j: _parse_form4_purchases(cik_no_zero, *j), to_fetch))
        for j, res in zip(to_fetch, results):
            if res is None:
                failures += 1          # fetch failed (rate limit / network) — retry next time
            else:
                cache[j[0]] = res      # parsed OK (may be [] — no P in this filing)

    # Prune to the current window and persist the accumulated progress.
    cache = {a: v for a, v in cache.items() if a in window_accns}
    _save_filing_cache(cik_no_zero, cache)

    # Aggregate purchases from every cached filing in the window.
    purchases: list[dict] = []
    for a in window_accns:
        if a in cache:
            purchases.extend(cache[a])

    # Keep only purchases whose transaction date is within the exact window.
    purchases = [p for p in purchases if p["date"] >= horizon]

    # Combine one insider's multiple lots on the same day into a single entry
    # (Form 4s often split a day's buying across several price lines). Shares and
    # value are summed; price is the volume-weighted average.
    grouped: dict[tuple, dict] = {}
    for p in purchases:
        key = (p["owner"], p["date"], p.get("security", ""))
        g = grouped.get(key)
        if g is None:
            grouped[key] = dict(p)
        else:
            g["shares"] += p["shares"]
            g["value"]  += p["value"]
    purchases = list(grouped.values())
    for g in purchases:
        g["price"] = (g["value"] / g["shares"]) if g["shares"] else 0.0
    purchases.sort(key=lambda p: p["date"], reverse=True)

    # Per-year trend: total value, shares, transaction count, unique buyers.
    trend: dict[str, dict] = {}
    for p in purchases:
        y = p["date"][:4]
        t = trend.setdefault(y, {"value": 0.0, "shares": 0.0, "count": 0, "buyers": set()})
        t["value"]  += p["value"]
        t["shares"] += p["shares"]
        t["count"]  += 1
        t["buyers"].add(p["owner"])
    trend_out = {y: {"value": round(v["value"]), "shares": round(v["shares"]),
                     "count": v["count"], "buyers": len(v["buyers"])}
                 for y, v in trend.items()}

    fetched_all = all(a in cache for a in window_accns)
    processed = len([a for a in window_accns if a in cache])
    return {
        "purchases": purchases[:60],   # cap the table
        "trend": trend_out,
        "total_value": round(sum(p["value"] for p in purchases)),
        "total_count": len(purchases),
        "months": months,
        # How many Form 4 filings were scanned to produce these results.
        "filings_processed": processed,
        "filings_total": len(window_accns),
        # True once every in-window Form 4 has been fetched and cached.
        "complete": fetched_all,
        # How many filings still need fetching (rate-limited/failed this run).
        "pending": len(window_accns) - processed,
        # Some Form 4 fetches failed — the UI shows a retry notice (with the
        # already-fetched purchases) rather than implying there were none.
        "rate_limited": not fetched_all,
    }


def _filing_cache_path(cik_no_zero: str) -> str:
    return os.path.join(screener.CACHE_DIR, f"insider_filings_{cik_no_zero}.json")


def _load_filing_cache(cik_no_zero: str) -> dict:
    """Load the per-accession Form 4 parse cache for a company (immutable data)."""
    try:
        with open(_filing_cache_path(cik_no_zero)) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_filing_cache(cik_no_zero: str, data: dict) -> None:
    try:
        with open(_filing_cache_path(cik_no_zero), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ─── Top shareholders (Schedule 13D/13G beneficial-ownership filings) ─────────

def _sc13_cache_path(cik_no_zero: str) -> str:
    return os.path.join(screener.CACHE_DIR, f"sc13_filings_{cik_no_zero}.json")


def _load_sc13_cache(cik_no_zero: str) -> dict:
    try:
        with open(_sc13_cache_path(cik_no_zero)) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_sc13_cache(cik_no_zero: str, data: dict) -> None:
    try:
        with open(_sc13_cache_path(cik_no_zero), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


_SC13_PCT_RE = re.compile(
    r"PERCENT OF CLASS REPRESENTED BY AMOUNT IN\s*ROW\s*\(?9\)?\s*(\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE)
_SC13_SHARES_RE = re.compile(
    r"AGGREGATE AMOUNT BENEFICIALLY OWNED BY\s*EACH REPORTING PERSON\s*([\d,]+)",
    re.IGNORECASE)
# Legacy terse plain-text layout (e.g. old FMR/Fidelity filings): cover page
# items are given as bare "Item 9: 998,190,803" / "Item 11: 4.069%" instead
# of the full verbose item-label text the two regexes above expect.
_SC13_ITEM9_RE  = re.compile(r"Item\s*9\s*:\s*([\d,]+)", re.IGNORECASE)
_SC13_ITEM11_RE = re.compile(r"Item\s*11\s*:\s*(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)


def _sc13_index_filers(cik_no_zero: str, accn_nodash: str, accn: str) -> list[dict]:
    """Parse an SC 13D/13G filing's own EDGAR index page for the reporting
    owner(s) ('Filed by' companyName spans) — reliable across all form eras,
    unlike scraping the cover page text for a name."""
    try:
        r = requests.get(
            f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accn_nodash}/{accn}-index.htm",
            headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
    except Exception:
        return []
    filers = []
    for m in re.finditer(r'<span class="companyName">(.*?)</span>', r.text, re.S):
        block = m.group(1)
        if "(Filed by)" not in block:
            continue
        name_m = re.match(r"\s*(.+?)\s*\(Filed by\)", block, re.S)
        cik_m = re.search(r"CIK=(\d+)", block)
        if name_m:
            filers.append({"name": re.sub(r"\s+", " ", name_m.group(1)).strip(),
                          "cik": int(cik_m.group(1)) if cik_m else None})
    return filers


# Sentinel distinguishing a permanent parse failure (fetched fine, but no
# regex/tag matched — retrying won't help until the parser itself changes)
# from a transient one (network error / non-200 / timeout — worth retrying).
# Conflating the two was the original bug: an unparseable filing looked
# identical to a pending network fetch, so it was counted as "pending"
# forever and the UI showed a permanent, misleading "rate limited" banner.
_SC13_UNPARSEABLE = object()


def _parse_sc13_ownership(cik_no_zero: str, accn_nodash: str, doc: str):
    """Extract the first reporting person's Item 9 (shares owned) / Item 11
    (% of class) from a Schedule 13D/13G cover page — the standardized cover
    page items every filer, old HTML or new, must complete. A joint filing
    lists one cover page per reporting person; the first is the lead/primary
    filer named on the index page, which is what we key the row to.

    Returns a dict on success, None on a transient fetch failure (retry
    later), or _SC13_UNPARSEABLE if the document fetched fine but matched
    none of the known cover-page layouts (don't retry — it'll never match)."""
    if not doc:
        return _SC13_UNPARSEABLE

    # Structured-XML era (Dec 2024+): the primary document is primary_doc.xml
    # (submissions.json prefixes it with an XSL viewer path). Parse the raw
    # XML's typed fields — cleaner than text heuristics, and it names the
    # reporting person directly.
    if doc.endswith("primary_doc.xml"):
        try:
            r = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accn_nodash}/primary_doc.xml",
                             headers=HEADERS, timeout=25)
            if r.status_code != 200:
                return None
            xml = re.sub(r'xmlns(:\w+)?="[^"]*"', "", r.text)   # drop namespace declarations
            xml = re.sub(r"<(/?)\w+:", r"<\1", xml)             # and tag prefixes (<sch:classPercent> → <classPercent>)
            persons = re.findall(r"<coverPageHeaderReportingPersonDetails>(.*?)</coverPageHeaderReportingPersonDetails>",
                                 xml, re.S)
            best = None
            for p in persons:
                name_m = re.search(r"<reportingPersonName>(.*?)</reportingPersonName>", p, re.S)
                pct_m = re.search(r"<classPercent>([\d.]+)</classPercent>", p)
                sh_m = re.search(r"<reportingPersonBeneficiallyOwnedAggregateNumberOfShares>([\d.,]+)<", p)
                if not (name_m and pct_m):
                    continue
                cand = {
                    "pct": float(pct_m.group(1)) / 100.0,
                    "shares": float(sh_m.group(1).replace(",", "")) if sh_m else None,
                    "name": re.sub(r"\s+", " ", name_m.group(1)).strip(),
                }
                # Joint filings list several reporting persons (parent entities
                # repeat the same stake); keep the largest as the lead row.
                if best is None or cand["pct"] > best["pct"]:
                    best = cand
            return best if best else _SC13_UNPARSEABLE
        except Exception:
            return None

    try:
        r = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accn_nodash}/{doc}",
                         headers=HEADERS, timeout=25)
        if r.status_code != 200:
            return None
        if doc.lower().endswith((".htm", ".html", ".txt")):
            text = BeautifulSoup(r.text, "html.parser").get_text(" ")
        else:
            text = r.text
        text = re.sub(r"\s+", " ", text)
    except Exception:
        return None
    pct_m = _SC13_PCT_RE.search(text)
    sh_m = _SC13_SHARES_RE.search(text)
    if not pct_m:
        pct_m = _SC13_ITEM11_RE.search(text)
        sh_m = _SC13_ITEM9_RE.search(text)
    if not pct_m:
        return _SC13_UNPARSEABLE
    return {
        "pct": float(pct_m.group(1)) / 100.0,
        "shares": float(sh_m.group(1).replace(",", "")) if sh_m else None,
    }


def _fetch_sc13_filing(cik_no_zero: str, accn: str, doc: str, fdate: str, form: str) -> Optional[dict]:
    """Returns a holder dict, None (transient failure — caller should retry),
    or {"_unparsed": True} (permanent — cache it so it stops being retried,
    but exclude it from results and don't count it against "complete")."""
    accn_nodash = accn.replace("-", "")
    own = _parse_sc13_ownership(cik_no_zero, accn_nodash, doc)
    if own is _SC13_UNPARSEABLE:
        return {"_unparsed": True}
    if not own:
        return None
    if own.get("name"):
        # Structured XML names the reporting person directly — no index-page
        # scrape needed (and joint parents share one 'Filed by' anyway).
        filer_name, filer_cik = own["name"], None
    else:
        filers = _sc13_index_filers(cik_no_zero, accn_nodash, accn)
        if not filers:
            return None
        filer_name, filer_cik = filers[0]["name"], filers[0]["cik"]
    return {
        "filer": filer_name,
        "filer_cik": filer_cik,
        "pct": own["pct"],
        "shares": own["shares"],
        "form": form,
        "date": fdate,
        "link": f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accn_nodash}/{accn}-index.htm",
    }


def get_top_shareholders(submissions: dict, months: int = 12, max_filings: int = 60) -> dict:
    """
    Schedule 13D/13G beneficial-ownership filings from the last `months` —
    each reporting owner's most recent filing in the window gives their
    current disclosed stake, since an amendment supersedes everything that
    owner filed before it.

    This is meant to be read alongside the company's Proxy Statement (shown
    in the same UI section), not as a standalone full holder list: the proxy
    already gives a point-in-time snapshot of top shareholders as of its own
    filing date (itself within the last 12 months), so a strict 12-month
    13D/13G scan is exactly the "what's changed since the proxy" delta —
    new stakes, exits, and updated percentages the proxy wouldn't yet show.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    accns  = recent.get("accessionNumber", [])
    docs   = recent.get("primaryDocument", [])
    dates  = recent.get("filingDate", [])
    cik_no_zero = str(int(submissions.get("cik", "0")))

    horizon = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")
    # Old-style form names plus the renamed types used by SEC's structured
    # (XML) 13D/G filing format mandated since Dec 2024 — new filings arrive
    # as "SCHEDULE 13G" / "SCHEDULE 13D/A", not "SC 13G" / "SC 13D/A".
    sc13_forms = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
                  "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A"}

    jobs = []
    for i, f in enumerate(forms):
        if f not in sc13_forms:
            continue
        fdate = dates[i] if i < len(dates) else ""
        if fdate and fdate < horizon:
            continue
        jobs.append((accns[i], docs[i], fdate, f))
        if len(jobs) >= max_filings:
            break

    # Per-accession cache — a filed 13D/13G cover page is immutable, so once
    # parsed it never needs re-fetching. Resumable like the Form 4 cache.
    cache = _load_sc13_cache(cik_no_zero)
    window_accns = {j[0] for j in jobs}
    to_fetch = [j for j in jobs if j[0] not in cache]

    if to_fetch:
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda j: _fetch_sc13_filing(cik_no_zero, j[0], j[1], j[2], j[3]), to_fetch))
        for j, res in zip(to_fetch, results):
            if res is not None:
                cache[j[0]] = res

    cache = {a: v for a, v in cache.items() if a in window_accns}
    _save_sc13_cache(cik_no_zero, cache)

    # Permanently-unparseable filings are cached (so they stop being retried)
    # but excluded from the usable rows — they don't count as "still pending"
    # either, since re-fetching them will never succeed.
    filings = [cache[a] for a in window_accns if a in cache and cache[a] and not cache[a].get("_unparsed")]

    latest_by_filer: dict = {}
    for f in filings:
        key = f.get("filer_cik") or f["filer"]
        cur = latest_by_filer.get(key)
        if cur is None or f["date"] > cur["date"]:
            latest_by_filer[key] = f
    # A 0% amendment is an exit notice (holder fell below the reporting
    # threshold) — it correctly supersedes that owner's stake above, but
    # isn't itself a shareholder row.
    holders = sorted((f for f in latest_by_filer.values() if f["pct"] > 0),
                     key=lambda f: (f["date"], f["pct"]), reverse=True)

    fetched_all = all(a in cache for a in window_accns)
    processed = len([a for a in window_accns if a in cache])
    return {
        "holders": holders[:20],
        "total_count": len(holders),
        "months": months,
        "filings_processed": processed,
        "filings_total": len(window_accns),
        "complete": fetched_all,
        "pending": len(window_accns) - processed,
        "rate_limited": not fetched_all,
    }


# ─── Institutional (13F) holders — curated value-investor roster ──────────────
# Sourced from ValueSider's ~92 tracked value-investing 13F filers (2026-07).
# This is a fixed, small universe (unlike EDGAR full-text search over ~6,000
# filers), so each fund's latest-quarter 13F is fetched and cached ONCE per
# quarter — shared across every stock's analyze — instead of re-searching for
# each stock. A stock lookup after the first population is a pure in-memory
# scan of the cached holdings data, no SEC calls at all.
GURUS: list[dict] = [
    {"guru": "Guy Spier", "fund": "Aquamarine Capital", "manager": "Aquamarine Capital Management, LLC", "cik": 1404599},
    {"guru": "Vitaliy Katsenelson", "fund": "IMA", "manager": "Investment Management Associates Inc /ADV", "cik": 52024},
    {"guru": "V. D. Dodge, E. M. Cox", "fund": "Dodge & Cox Stock Fund", "manager": "Dodge & Cox", "cik": 200217},
    {"guru": "Mark A. Hillman", "fund": "Hillman Value Fund", "manager": "Hillman Capital Management, Inc.", "cik": 1314620},
    {"guru": "Mason Hawkins", "fund": "Longleaf Partners", "manager": "Southeastern Asset Management Inc", "cik": 807985},
    {"guru": "Independent Franchise Partners", "fund": "US Equity Fund", "manager": "Independent Franchise Partners LLP", "cik": 1483866},
    {"guru": "John W. Rogers Jr.", "fund": "Ariel Appreciation Fund", "manager": "Ariel Investments, LLC", "cik": 936753},
    {"guru": "Christopher Davis", "fund": "Clipper Fund", "manager": "Davis Selected Advisers", "cik": 1036325},
    {"guru": "Jim Cullen", "fund": "Cullen Value Fund", "manager": "Cullen Capital Management, LLC", "cik": 1362535},
    {"guru": "Wallace Weitz", "fund": "Weitz Value Fund", "manager": "Weitz Investment Management, Inc.", "cik": 883965},
    {"guru": "Bill Nygren", "fund": "Oakmark Select Fund", "manager": "Harris Associates L P", "cik": 813917},
    {"guru": "Charles Bobrinskoy", "fund": "Ariel Focus Fund", "manager": "Ariel Investments, LLC", "cik": 936753},
    {"guru": "Ruane, Cunniff & Goldfarb", "fund": "Sequoia Fund", "manager": "Ruane, Cunniff & Goldfarb L.P.", "cik": 1720792},
    {"guru": "Charles Jigarjian", "fund": "7G Capital Management", "manager": "7G Capital Management, LLC", "cik": 1720350},
    {"guru": "Howard Marks", "fund": "Oaktree Capital Management", "manager": "Oaktree Capital Management LP", "cik": 949509},
    {"guru": "Duan Yongping", "fund": "H&H International Investment", "manager": "H&H International Investment, LLC", "cik": 1759760},
    {"guru": "Andrew Brenton", "fund": "Turtle Creek Asset Management", "manager": "Turtle Creek Asset Management Inc.", "cik": 1484148},
    {"guru": "David Einhorn", "fund": "Greenlight Capital", "manager": "Greenlight Capital Inc", "cik": 1079114},
    {"guru": "Christopher Bloomstran", "fund": "Semper Augustus Investments Group", "manager": "Semper Augustus Investments Group LLC", "cik": 1115373},
    {"guru": "Chris Hohn", "fund": "TCI Fund Management", "manager": "TCI Fund Management Ltd", "cik": 1647251},
    {"guru": "Ravenel Boykin Curry IV", "fund": "Eagle Capital Management", "manager": "Eagle Capital Management LLC", "cik": 945631},
    {"guru": "Ole Andreas Halvorsen", "fund": "Viking Global Investors", "manager": "Viking Global Investors LP", "cik": 1103804},
    {"guru": "Bill Miller", "fund": "Miller Value Partners", "manager": "Miller Value Partners, LLC", "cik": 1135778},
    {"guru": "Nelson Peltz", "fund": "Trian Fund Management", "manager": "Trian Fund Management, L.P.", "cik": 1345471},
    {"guru": "Quincy Lee", "fund": "Ancient Art (Teton Capital)", "manager": "Ancient Art, L.P.", "cik": 1426749},
    {"guru": "Nicolai Tangen", "fund": "AKO Capital", "manager": "AKO Capital LLP", "cik": 1376879},
    {"guru": "Greg Alexander", "fund": "Conifer Management", "manager": "Conifer Management, L.L.C.", "cik": 1773994},
    {"guru": "Francois Rochon", "fund": "Giverny Capital", "manager": "Giverny Capital Inc.", "cik": 1641864},
    {"guru": "Pat Dorsey", "fund": "Dorsey Asset Management", "manager": "Dorsey Asset Management, LLC", "cik": 1671657},
    {"guru": "Daniel Loeb", "fund": "Third Point", "manager": "Third Point LLC", "cik": 1040273},
    {"guru": "Fred Martin", "fund": "Disciplined Growth Investors", "manager": "Disciplined Growth Investors Inc /MN", "cik": 1050442},
    {"guru": "Dev Kantesaria", "fund": "Valley Forge Capital Management", "manager": "Valley Forge Capital Management, LP", "cik": 1697868},
    {"guru": "Stephen Mandel", "fund": "Lone Pine Capital", "manager": "Lone Pine Capital LLC", "cik": 1061165},
    {"guru": "David Tepper", "fund": "Appaloosa Management", "manager": "Appaloosa LP", "cik": 1656456},
    {"guru": "Henry Ellenbogen", "fund": "Durable Capital Partners", "manager": "Durable Capital Partners LP", "cik": 1798849},
    {"guru": "Robert Karr", "fund": "Joho Capital", "manager": "Joho Capital LLC", "cik": 1106500},
    {"guru": "Stuart Mclaughlin", "fund": "Triple Frond Partners", "manager": "Triple Frond Partners LLC", "cik": 1454502},
    {"guru": "Jeffrey Ubben", "fund": "ValueAct Holdings", "manager": "ValueAct Holdings, L.P.", "cik": 1418814},
    {"guru": "Chase Coleman III", "fund": "Tiger Global Management", "manager": "Tiger Global Management LLC", "cik": 1167483},
    {"guru": "Frederick (Shad) Rowe", "fund": "Greenbrier Partners Capital Management", "manager": "Greenbrier Partners Capital Management, LLC", "cik": 1532262},
    {"guru": "Bill Gates", "fund": "Bill & Melinda Gates Foundation Trust", "manager": "Gates Foundation Trust", "cik": 1166559},
    {"guru": "Warren Buffett", "fund": "Berkshire Hathaway", "manager": "Berkshire Hathaway Inc", "cik": 1067983},
    {"guru": "David Rolfe", "fund": "Wedgewood Partners", "manager": "Wedgewood Partners Inc", "cik": 859804},
    {"guru": "Alex Roepers", "fund": "Atlantic Investment Management", "manager": "Atlantic Investment Management, Inc.", "cik": 1063296},
    {"guru": "Glenn Greenberg", "fund": "Brave Warrior Advisors", "manager": "Brave Warrior Advisors, LLC", "cik": 1553733},
    {"guru": "Prem Watsa", "fund": "Fairfax Financial Holdings", "manager": "Fairfax Financial Holdings Ltd/Can", "cik": 915191},
    {"guru": "Terry Smith", "fund": "Fundsmith", "manager": "Fundsmith LLP", "cik": 1569205},
    {"guru": "Glenn W. Welling", "fund": "Engaged Capital", "manager": "Engaged Capital LLC", "cik": 1559771},
    {"guru": "Connor Haley", "fund": "Alta Fox Capital Management", "manager": "Alta Fox Capital Management, LLC", "cik": 1858353},
    {"guru": "Bill Ackman", "fund": "Pershing Square Capital Management", "manager": "Pershing Square Capital Management, L.P.", "cik": 1336528},
    {"guru": "Bryan R. Lawrence", "fund": "Oakcliff Capital Partners", "manager": "Oakcliff Capital Partners, LP", "cik": 1657335},
    {"guru": "Mark Massey", "fund": "AltaRock Partners", "manager": "AltaRock Partners LP", "cik": 1631014},
    {"guru": "Li Lu", "fund": "Himalaya Capital Management", "manager": "Himalaya Capital Management LLC", "cik": 1709323},
    {"guru": "Carl Icahn", "fund": "Icahn Capital Management", "manager": "Icahn Carl C", "cik": 921669},
    {"guru": "Bruce Berkowitz", "fund": "Fairholme Capital Management", "manager": "Fairholme Capital Management LLC", "cik": 1056831},
    {"guru": "Norbert Lou", "fund": "Punch Card Management", "manager": "Punch Card Management L.P.", "cik": 1631664},
    {"guru": "Adam Wyden", "fund": "ADW Capital Management", "manager": "ADW Capital Management, LLC", "cik": 1745214},
    {"guru": "Clifford Sosin", "fund": "CAS Investment Partners", "manager": "CAS Investment Partners, LLC", "cik": 1697591},
    {"guru": "Sarah Ketterer", "fund": "Causeway Capital Management", "manager": "Causeway Capital Management LLC", "cik": 1165797},
    {"guru": "Donald G. Smith", "fund": "Donald Smith & Co.", "manager": "Donald Smith & Co., Inc.", "cik": 814375},
    {"guru": "Francis Chou", "fund": "Chou Associates Management", "manager": "Chou Associates Management Inc.", "cik": 1389403},
    {"guru": "David M. Polen", "fund": "Polen Capital Management", "manager": "Polen Capital Management LLC", "cik": 1034524},
    {"guru": "Chuck Akre", "fund": "Akre Capital Management", "manager": "Akre Capital Management LLC", "cik": 1112520},
    {"guru": "Mohnish Pabrai", "fund": "Dalal Street", "manager": "Dalal Street, LLC", "cik": 1549575},
    {"guru": "Seth Klarman", "fund": "Baupost Group", "manager": "Baupost Group LLC/MA", "cik": 1061768},
    {"guru": "Nathaniel Simons", "fund": "Meritage Group", "manager": "Meritage Group LP", "cik": 1427119},
    {"guru": "Dennis Hong", "fund": "ShawSpring Partners", "manager": "ShawSpring Partners LLC", "cik": 1766908},
    {"guru": "David Abrams", "fund": "Abrams Capital Management", "manager": "Abrams Capital Management, L.P.", "cik": 1358706},
    {"guru": "Thomas Russo", "fund": "Gardner Russo & Quinn", "manager": "Gardner Russo & Quinn LLC", "cik": 860643},
    {"guru": "Marty Whitman", "fund": "Third Avenue Management", "manager": "Third Avenue Management LLC", "cik": 1099281},
    {"guru": "John Armitage", "fund": "Egerton Capital", "manager": "Egerton Capital (UK) LLP", "cik": 1581811},
    {"guru": "Andrew R. Adams", "fund": "Mairs & Power Growth Fund", "manager": "Mairs & Power Inc", "cik": 1070134},
    {"guru": "Paul Isaac", "fund": "Arbiter Partners Capital Management", "manager": "Arbiter Partners Capital Management LLC", "cik": 1513193},
    {"guru": "C.T. Fitzpatrick", "fund": "Vulcan Value Partners", "manager": "Vulcan Value Partners, LLC", "cik": 1556785},
    {"guru": "Rob Vinall", "fund": "RV Capital", "manager": "RV Capital AG", "cik": 1766596},
    {"guru": "Josh Tarasoff", "fund": "Greenlea Lane Capital Management", "manager": "Greenlea Lane Capital Management, LLC", "cik": 1766504},
    {"guru": "Thomas Graham, Alan, Irving Kahns", "fund": "Kahn Brothers Group", "manager": "Kahn Brothers Group Inc", "cik": 1039565},
    {"guru": "Harry Burn", "fund": "Sound Shore Management", "manager": "Sound Shore Management Inc /CT/", "cik": 820124},
    {"guru": "William Von Mueffling", "fund": "Cantillon Capital Management", "manager": "Cantillon Capital Management LLC", "cik": 1279936},
    {"guru": "Paul Lountzis", "fund": "Lountzis Asset Management", "manager": "Lountzis Asset Management, LLC", "cik": 1821168},
    {"guru": "B. Tweedy, Ch. Browne", "fund": "Tweedy, Browne Co All Funds (US)", "manager": "Tweedy, Browne Co LLC", "cik": 732905},
    {"guru": "Ronald Muhlenkamp", "fund": "Muhlenkamp & Co", "manager": "Muhlenkamp & Co Inc", "cik": 1133219},
    {"guru": "Parnassus Investments", "fund": "Parnassus Endeavor Fund", "manager": "Parnassus Investments, LLC", "cik": 948669},
    {"guru": "Eric H. Schoenstein", "fund": "Jensen Investment Management", "manager": "Jensen Investment Management Inc", "cik": 1106129},
    {"guru": "Donald Yacktman", "fund": "Yacktman Asset Management", "manager": "Yacktman Asset Management LP", "cik": 905567},
    {"guru": "Robert Torray", "fund": "Torray Fund", "manager": "Torray Investment Partners LLC", "cik": 98758},
    {"guru": "Michael Lindsell, Nick Train", "fund": "Lindsell Train", "manager": "Lindsell Train Ltd", "cik": 1484150},
    {"guru": "Charlie Munger", "fund": "Daily Journal Corp", "manager": "Daily Journal Corp", "cik": 783412},
    {"guru": "Edgar Wachenheim III", "fund": "Greenhaven Associates", "manager": "Greenhaven Associates Inc", "cik": 846222},
    {"guru": "Brian Bares", "fund": "Bares Capital Management", "manager": "Bares Capital Management, Inc.", "cik": 1340807},
    {"guru": "Michael Burry", "fund": "Scion Asset Management", "manager": "Scion Asset Management, LLC", "cik": 1649339},
    {"guru": "David Katz", "fund": "Matrix Advisors Value Fund", "manager": "Matrix Asset Advisors Inc/NY", "cik": 1016287},
    {"guru": "Phil Town", "fund": "Rule One Fund", "manager": "Rule One Partners, LLC", "cik": 2040263},
]


def _xml_tag(block: str, tag: str) -> Optional[str]:
    """Namespace-agnostic single-value extract from a 13F info-table block."""
    m = re.search(r"<(?:\w+:)?" + tag + r">\s*([^<]+?)\s*</(?:\w+:)?" + tag + ">", block)
    return m.group(1).strip() if m else None


def _latest_13f_quarter() -> str:
    """The most recent quarter-end whose 13F filing deadline (45 days) has passed
    — i.e. the latest fully-reported quarter. Used only as a display/target
    reference; each fund's cache is considered fresh once it holds this period
    (or the fund's own latest available filing, if they haven't filed yet)."""
    from datetime import date, timedelta
    d = datetime.now().date() - timedelta(days=46)   # 45-day deadline + 1 buffer
    qends = [date(y, m, dd)
             for y in (d.year, d.year - 1)
             for (m, dd) in ((3, 31), (6, 30), (9, 30), (12, 31))]
    passed = [q for q in qends if q <= d]
    return max(passed).isoformat() if passed else ""


def _guru_fund_cache_path(cik: int) -> str:
    return os.path.join(screener.CACHE_DIR, f"guru13f_{cik}.json")


def _load_guru_fund(cik: int) -> Optional[dict]:
    try:
        with open(_guru_fund_cache_path(cik)) as f:
            return json.load(f)
    except Exception:
        return None


def _save_guru_fund(cik: int, data: dict) -> None:
    try:
        with open(_guru_fund_cache_path(cik), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _fetch_fund_latest_13f_meta(cik: int) -> Optional[tuple]:
    """Return (accn_nodash, filing_date, period_ending) for a fund's most
    recent 13F-HR / 13F-HR/A filing, or None on failure / no such filing.
    (The submissions API's primaryDocument for a 13F is the cover page, not
    the information table — that's resolved separately, see _resolve_infotable_doc.)"""
    for attempt in range(2):
        try:
            r = requests.get(f"{EDGAR_BASE}/submissions/CIK{cik:010d}.json",
                             headers=HEADERS, timeout=20)
            if r.status_code == 429:
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            j = r.json()
            break
        except Exception:
            if attempt == 1:
                return None
            time.sleep(0.4)
    else:
        return None

    recent = j.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    accns  = recent.get("accessionNumber", [])
    dates  = recent.get("filingDate", [])
    periods = recent.get("reportDate", [])
    best = None   # (period, filing_date, accn)
    for i, f in enumerate(forms):
        if f not in ("13F-HR", "13F-HR/A"):
            continue
        period = periods[i] if i < len(periods) else ""
        fdate  = dates[i] if i < len(dates) else ""
        cand = (period, fdate, accns[i].replace("-", ""))
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if not best:
        return None
    return best[2], best[1], best[0]   # accn, filing_date, period


def _resolve_infotable_doc(cik: int, accn_nodash: str) -> Optional[str]:
    """A 13F filing's information table is a separate XML document from the
    cover-page primary_doc.xml, with a filer-chosen name (e.g. 'infotable.xml',
    'form13fInfoTable.xml', or a numeric hash). Find it via the filing's
    document index: the .xml file that isn't primary_doc.xml or an -index file."""
    try:
        r = requests.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/index.json",
                         headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        items = r.json().get("directory", {}).get("item", [])
    except Exception:
        return None
    for it in items:
        nm = (it.get("name") or "")
        low = nm.lower()
        if low.endswith(".xml") and low != "primary_doc.xml" and "-index" not in low:
            return nm
    return None


def _fetch_fund_13f_holdings(cik: int, accn_nodash: str) -> Optional[dict]:
    """Fetch and parse ALL holdings from one fund's 13F information table
    (not filtered to any single issuer — this fund's full portfolio)."""
    raw_doc = _resolve_infotable_doc(cik, accn_nodash)
    if not raw_doc:
        return None
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{raw_doc}"
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 429:
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            text = r.text
            break
        except Exception:
            if attempt == 1:
                return None
            time.sleep(0.4)
    else:
        return None

    holdings = []
    portfolio_val = 0.0
    for block in re.split(r"</(?:\w+:)?infoTable>", text):
        val_raw = _xml_tag(block, "value")
        if not val_raw:
            continue
        try:
            val = float(val_raw)
        except ValueError:
            continue
        nm = _xml_tag(block, "nameOfIssuer") or ""
        if not nm:
            continue
        sh_raw = _xml_tag(block, "sshPrnamt")
        try:
            sh = float(sh_raw) if sh_raw else 0.0
        except ValueError:
            sh = 0.0
        portfolio_val += val
        holdings.append({"name": nm.upper(), "cusip": _xml_tag(block, "cusip"),
                         "value": val, "shares": sh})
    return {"holdings": holdings, "portfolio_value": portfolio_val}


def _unique_guru_ciks() -> list:
    seen, out = set(), []
    for g in GURUS:
        if g["cik"] not in seen:
            seen.add(g["cik"])
            out.append(g["cik"])
    return out


def get_guru_universe(refresh: bool = False) -> dict:
    """
    Ensure every tracked fund's latest 13F is fetched and cached, then return
    the merged dataset. Only funds whose cache is missing (or, on refresh,
    purged) trigger a network fetch — once populated for a quarter, this is a
    zero-network operation, and get_institutional_holders() for ANY stock just
    scans the returned dict in memory.
    """
    target_q = _latest_13f_quarter()
    ciks = _unique_guru_ciks()

    if refresh:
        for c in ciks:
            try:
                os.remove(_guru_fund_cache_path(c))
            except OSError:
                pass

    to_fetch = []
    funds: dict = {}
    for c in ciks:
        cached = _load_guru_fund(c)
        # Fresh if it already reflects the target quarter, OR it's the most
        # recent filing we could find for a fund that hasn't filed the target
        # quarter yet (avoids hammering slow/late filers every request).
        if cached and (cached.get("period", "") >= target_q or cached.get("_no_newer")):
            funds[c] = cached
        else:
            to_fetch.append(c)

    failures = 0
    if to_fetch:
        def _job(cik):
            meta = _fetch_fund_latest_13f_meta(cik)
            if meta is None:
                return cik, None
            accn, fdate, period = meta
            data = _fetch_fund_13f_holdings(cik, accn)
            if data is None:
                return cik, None
            data.update({"period": period, "date": fdate,
                        "link": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/",
                        "_no_newer": period < target_q})
            return cik, data
        with ThreadPoolExecutor(max_workers=6) as ex:
            for cik, data in ex.map(_job, to_fetch):
                if data is None:
                    failures += 1
                else:
                    _save_guru_fund(cik, data)
                    funds[cik] = data

    # Drop funds whose latest 13F was filed more than 6 months ago — a fund
    # that's gone quiet that long is treated as no longer actively tracked
    # for holders/screener purposes (e.g. dropped below the $100M 13F
    # threshold, wound down, or otherwise stopped reporting).
    cutoff = (datetime.now() - timedelta(days=182)).strftime("%Y-%m-%d")
    fresh_funds = {c: d for c, d in funds.items() if (d.get("date") or "") >= cutoff}
    stale_count = len(funds) - len(fresh_funds)

    return {
        "funds": fresh_funds,
        "total": len(ciks),
        "scanned": len(funds),
        "stale_excluded": stale_count,
        "pending": len(ciks) - len(funds),
        "complete": len(funds) == len(ciks),
        "rate_limited": failures > 0,
        "target_period": target_q,
    }


def get_institutional_holders(submissions: dict, shares_out: Optional[float] = None,
                              refresh: bool = False) -> dict:
    """
    Institutional 13F holders of this stock, scanned from the cached
    value-investor universe (see get_guru_universe). No per-stock SEC calls
    once the universe is warm for the quarter.
    """
    name = submissions.get("name", "") or ""
    # Hyphens included: EDGAR says "CAL-MAINE FOODS" but 13F filers write
    # "CAL MAINE FOODS" — both must normalize identically.
    words = [w for w in re.sub(r"[.,/&\-]", " ", name.upper()).split() if w not in ("THE", "A")]
    name_kw = " ".join(words[:2])
    first_word = words[0] if words else ""
    empty = {"holders": [], "total_value": 0, "total_count": 0,
             "complete": True, "rate_limited": False, "pending": 0,
             "funds_scanned": 0, "funds_total": len(_unique_guru_ciks()), "period": ""}
    if not name_kw:
        return empty

    def _matches_issuer_name(holding_name: str) -> bool:
        # Normalize the holding's name text the same way as the target's, so
        # hyphen/punctuation differences don't break the prefix comparison.
        holding_name = " ".join(re.sub(r"[.,/&\-]", " ", holding_name.upper()).split())
        # Primary: precise two-word prefix match (e.g. "APPLE INC..." — avoids
        # false positives like "APPLE HOSPITALITY REIT").
        if holding_name.startswith(name_kw):
            return True
        # Fallback: some 13F filers drop the corporate suffix entirely and tag
        # just the first word (e.g. Lindsell Train tags Intuit as "INTUIT", not
        # "INTUIT INC"). An exact match on the first word alone is still safe —
        # it won't accidentally catch "APPLE HOSPITALITY" since that isn't an
        # exact match to "APPLE".
        return bool(first_word) and holding_name == first_word

    universe = get_guru_universe(refresh=refresh)

    # CUSIP is the standardized security identifier — far more reliable than
    # issuer-name text, which different filers format wildly differently (e.g.
    # "INTUIT INC", "INTUIT", "INTUIT COM" all refer to the same CUSIP
    # 461202103). Bootstrap the target's CUSIP(s) from whichever holdings match
    # by name, then do the real pass matching by CUSIP OR name — this catches
    # filers whose name text doesn't match any of our name heuristics as long
    # as at least one fund in the roster used a recognizable name.
    bootstrap_cusips: set = set()
    for data in universe["funds"].values():
        if not data:
            continue
        for h in data.get("holdings", []):
            if _matches_issuer_name(h["name"]) and h.get("cusip"):
                bootstrap_cusips.add(h["cusip"].upper())

    # Identifier-based seeding — the definitive path, independent of how any
    # filer spelled the name. Resolve this stock's ticker(s) to CUSIP(s) via:
    #  1. SEC's Fails-to-Deliver map inverted (ticker → all its CUSIPs,
    #     including share-class and CINS variants), and
    #  2. the screener's own CUSIP→ticker cache (built by the Guru Holdings
    #     universe resolution).
    own_tickers = {t.upper() for t in (submissions.get("tickers") or []) if t}
    if own_tickers:
        for c, tk in _load_ftd_cusip_map().items():
            if tk and tk.upper() in own_tickers:
                bootstrap_cusips.add(c.upper())
        for c, tk in _load_cusip_ticker_cache().items():
            if tk and tk.upper() in own_tickers:
                bootstrap_cusips.add(c.upper())

    def _matches(h: dict) -> bool:
        return _matches_issuer_name(h["name"]) or ((h.get("cusip") or "").upper() in bootstrap_cusips)

    # Guru/fund display label per CIK. SEC 13F filings are made at the manager
    # level, not per-fund, so when one manager runs several roster funds
    # (e.g. Ariel Investments files a single combined 13F covering both its
    # Appreciation and Focus funds) there's no way to attribute a holding to
    # one specific fund — label with the manager name instead of picking an
    # arbitrary individual guru's name, and note the funds it covers.
    ciks_by_cik: dict = {}
    for g in GURUS:
        ciks_by_cik.setdefault(g["cik"], []).append(g)
    label_by_cik: dict = {}
    for cik, entries in ciks_by_cik.items():
        if len(entries) == 1:
            g = entries[0]
            label_by_cik[cik] = (g["guru"], g["fund"], g["manager"])
        else:
            manager = entries[0]["manager"]
            funds = " / ".join(sorted({e["fund"] for e in entries}))
            label_by_cik[cik] = (manager, funds, manager)

    periods = [d.get("period", "") for d in universe["funds"].values() if d.get("period")]
    period = Counter(periods).most_common(1)[0][0] if periods else universe["target_period"]

    holders = []
    for cik, data in universe["funds"].items():
        if not data or not data.get("holdings"):
            continue
        val = 0.0
        sh  = 0.0
        for h in data["holdings"]:
            if _matches(h):
                val += h["value"]
                sh  += h["shares"]
        if val <= 0:
            continue
        guru, fund, manager = label_by_cik.get(cik, ("", "", ""))
        pv = data.get("portfolio_value") or 0.0
        holders.append({
            "fund": guru or manager,
            "funds": fund,
            "manager": manager,
            "cik": cik,
            "date": data.get("date", ""),
            "value": val,
            "shares": sh,
            "portfolio_value": pv,
            "portfolio_positions": len(data["holdings"]),
            "pct": (val / pv) if pv > 0 else 0.0,
            "own_pct": (sh / shares_out) if shares_out else None,
            "link": data.get("link", ""),
        })
    holders.sort(key=lambda h: (h.get("pct") or 0, h.get("value") or 0), reverse=True)

    return {
        "holders": holders[:100],
        "total_value": round(sum(h["value"] for h in holders)),
        "total_count": len(holders),
        "funds_scanned": len(universe["funds"]),
        "funds_total": universe["total"],
        "funds_stale_excluded": universe.get("stale_excluded", 0),
        "period": period,
        "shares_outstanding": shares_out,
        "sampled": False,   # fixed curated roster, not a sample of a larger universe
        "complete": universe["complete"],
        "pending": universe["pending"],
        "rate_limited": universe["rate_limited"],
    }


# ─── Guru Holdings universe (screener) ─────────────────────────────────────────
# Compile the union of every stock currently held by any of the ~92 tracked
# value-investor funds, so the screener can rank them by P/FCF / EV/EBIT like
# any other universe. 13F info tables identify securities by CUSIP, not ticker,
# so each CUSIP is resolved to a ticker via SEC EDGAR full-text search (no
# third party) and cached long-term — the mapping is effectively permanent, so
# this is a one-time cost amortized across quarters, same spirit as the
# guru-fund 13F cache itself.

def _cusip_ticker_cache_path() -> str:
    return os.path.join(screener.CACHE_DIR, "cusip_ticker_map.json")


def _load_cusip_ticker_cache() -> dict:
    try:
        with open(_cusip_ticker_cache_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cusip_ticker_cache(data: dict) -> None:
    try:
        with open(_cusip_ticker_cache_path(), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


_CUSIP_DISPLAY_NAME_RE = re.compile(
    r"^(.*?)\s+\(([A-Z0-9.\-]+(?:,\s*[A-Z0-9.\-]+)*)\)\s+\(CIK (\d+)\)$")


def _load_ftd_cusip_map() -> dict:
    """
    CUSIP → ticker dictionary from SEC's twice-monthly Fails-to-Deliver files —
    the only SEC dataset that is a direct CUSIP|SYMBOL table (a security only
    needs a single fail-to-deliver to appear, so coverage is near-total,
    including foreign CINS identifiers that trade in the US). Pulls the last
    few half-month files and merges them; 30-day disk cache.
    """
    cache_path = os.path.join(screener.CACHE_DIR, "ftd_cusip_map.json")
    try:
        if os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < 30 * 86400:
            with open(cache_path) as f:
                return json.load(f)
    except Exception:
        pass

    # Half-month files, newest first: cnsfails{YYYYMM}a (1st–15th) / b (16th–EOM).
    d = datetime.now().date()
    halves = []
    y, m = d.year, d.month
    for _ in range(4):   # current + previous month, both halves
        halves += [f"{y}{m:02d}b", f"{y}{m:02d}a"]
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    out: dict = {}
    fetched = 0
    for tag in halves:
        if fetched >= 3:   # 3 successful files is plenty of coverage
            break
        try:
            r = requests.get(f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{tag}.zip",
                             headers=HEADERS, timeout=45)
            if r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open(z.namelist()[0]) as f:
                for raw in f:
                    parts = raw.decode("utf-8", errors="replace").split("|")
                    if len(parts) < 3 or parts[1] == "CUSIP":
                        continue
                    cusip, symbol = parts[1].strip(), parts[2].strip()
                    if cusip and symbol and cusip not in out:
                        out[cusip] = symbol
            fetched += 1
        except Exception:
            continue

    if out:
        try:
            with open(cache_path, "w") as f:
                json.dump(out, f)
        except Exception:
            pass
    return out


def _resolve_cusips_ftd(cusips: list) -> dict:
    """Tier: direct CUSIP→ticker lookup in SEC's Fails-to-Deliver data."""
    ftd = _load_ftd_cusip_map()
    return {c: ftd[c.upper()] for c in cusips if ftd.get(c.upper())}


def _cik_ticker_title_map() -> dict:
    """CIK → (best ticker, registry title) from SEC's ticker registry
    (score-resolved so primary share classes win over -A/-WS variants;
    lower _ticker_score is better, matching screener's convention)."""
    out: dict = {}
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=HEADERS, timeout=20)
        entries = list(r.json().values())
    except Exception:
        return out
    for e in entries:
        tk, title, cik = e.get("ticker", ""), e.get("title", ""), int(e.get("cik_str") or 0)
        if not tk or not cik or screener._is_non_common(tk):
            continue
        cur = out.get(cik)
        if cur is None or screener._ticker_score(tk) < screener._ticker_score(cur[0]):
            out[cik] = (tk, title)
    return out


def _names_overlap(a: str, b: str) -> bool:
    """True if two issuer names share at least one distinctive (≥4-char) word —
    guards CIK-based resolution against matching the wrong filer entirely."""
    wa = {w for w in _norm_issuer_name(a) if len(w) >= 4}
    wb = {w for w in _norm_issuer_name(b) if len(w) >= 4}
    return bool(wa & wb)


def _resolve_cusip_edgar(cusip: str, cik_titles: Optional[dict] = None,
                         expected_name: str = "") -> Optional[str]:
    """
    Resolve a CUSIP to a ticker using SEC's own EDGAR full-text search — no
    third party involved. Two extraction paths per hit:
    1. Ownership-disclosure filings (SC 13D/13G, N-PX) index the subject
       company's display_name as "NAME (TICKER) (CIK NNNNNNNNNN)" — parse it.
    2. Issuer-made filings (424B prospectuses, 8-K, S-1, FWP) carry the CUSIP
       in the body and the issuer's own CIK in the hit — map CIK → ticker via
       SEC's ticker registry. Guarded by a name-overlap check against the
       expected issuer name: closed-end funds also file 424B prospectuses
       listing their *holdings'* CUSIPs, so without the check a fund like
       Source Capital would win the match for Heineken's CUSIP.
    """
    try:
        r = requests.get("https://efts.sec.gov/LATEST/search-index",
                         params={"q": cusip}, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return None
    issuer_forms = {"424B1", "424B2", "424B3", "424B4", "424B5", "8-K", "S-1",
                    "S-3", "FWP", "10-K", "10-Q", "S-4", "20-F", "6-K"}
    for h in hits[:8]:
        src = h.get("_source", {})
        dn = src.get("display_names", [])
        if dn:
            m = _CUSIP_DISPLAY_NAME_RE.match(dn[0])
            if m and (not expected_name or _names_overlap(expected_name, m.group(1))):
                return m.group(2).split(",")[0].strip()
        # Issuer-filing path: the filer IS the subject company.
        if cik_titles and set(src.get("root_forms") or []) & issuer_forms:
            for cik_s in src.get("ciks") or []:
                info = cik_titles.get(int(cik_s))
                if not info:
                    continue
                tk, title = info
                if not expected_name or _names_overlap(expected_name, title):
                    return tk
    return None


def _resolve_cusips_edgar(cusips: list, time_budget_s: float = 90.0,
                          names: Optional[dict] = None) -> dict:
    """Resolve a batch of CUSIPs via EDGAR full-text search (one request per
    CUSIP — SEC has no bulk CUSIP-lookup endpoint). Time-budgeted and
    resumable like every other SEC-fetch loop in this app. `names` maps
    CUSIP → expected issuer name, used to guard the issuer-CIK path."""
    out: dict = {}
    start = time.time()
    cik_titles = _cik_ticker_title_map()
    names = names or {}
    official = _load_13f_official_list()
    for cusip in cusips:
        if time.time() - start > time_budget_s:
            break
        expected = names.get(cusip) or official.get(cusip.upper()) or ""
        ticker = _resolve_cusip_edgar(cusip, cik_titles, expected_name=expected)
        if ticker:
            out[cusip] = ticker
        time.sleep(0.15)   # EDGAR full-text search rate limit is generous but not unlimited
    return out


def _norm_issuer_name(s: str) -> list:
    """Normalize an issuer/company name for matching: uppercase, split hyphens
    and punctuation into word boundaries, drop generic corporate-suffix words."""
    s = re.sub(r"[.,/&\-']", " ", s.upper())
    stop = {"THE", "A", "INC", "CORP", "CO", "LTD", "LP", "PLC", "SA", "NV", "LLC", "NEW"}
    return [w for w in s.split() if w not in stop]


def _squish(words: list) -> str:
    """Concatenate normalized words with no separator, for matching name
    variants that differ only in where SEC's two datasets put a space or
    punctuation (e.g. 13(f) list "OREILLY AUTOMOTIVE" vs ticker registry
    "O REILLY AUTOMOTIVE"; "EXXON MOBIL" vs "EXXONMOBIL HOLDINGS")."""
    return "".join(words)


def _load_13f_official_list() -> dict:
    """
    SEC's own quarterly "Official List of Section 13(f) Securities" — every
    CUSIP currently reportable on a 13F filing, with its authoritative issuer
    name. By definition this covers every CUSIP our guru funds' CURRENT
    holdings could report, so it's ground truth for CUSIP -> name, unlike
    free-text 13F filer name variants ("INTUIT" vs "INTUIT COM"). SEC only
    publishes the machine-readable .txt for the CURRENT quarter (older
    quarters are PDF-only), so we just try current-then-previous quarter
    (the latter covers the narrow window right after a quarter rolls before
    SEC has published the new list yet).
    """
    cache_path = os.path.join(screener.CACHE_DIR, "13f_official_list.json")
    try:
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 30 * 86400:
            with open(cache_path) as f:
                return json.load(f)
    except Exception:
        pass

    d = datetime.now().date()
    y, q = d.year, (d.month - 1) // 3 + 1
    prev_q, prev_y = (q - 1, y) if q > 1 else (4, y - 1)
    quarters = [f"{y}q{q}", f"{prev_y}q{prev_q}"]

    cusip_name: dict = {}
    for qtag in quarters:
        try:
            r = requests.get(f"https://www.sec.gov/files/investment/13flist{qtag}-txt.txt",
                             headers=HEADERS, timeout=30)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                if len(line) < 12:
                    continue
                cusip = line[:9].strip()
                name = line[10:40].strip()
                if cusip and name and cusip not in cusip_name:
                    cusip_name[cusip] = name
            break   # got a usable list — no need to also fetch the prior quarter
        except Exception:
            continue
    try:
        with open(cache_path, "w") as f:
            json.dump(cusip_name, f)
    except Exception:
        pass
    return cusip_name


def _load_ticker_name_index() -> dict:
    """
    Name -> best ticker index, built from SEC's bulk ticker file. Two lookup
    structures: an exact word-key index (fast dict lookup, handles the common
    case), and a squished (no-space) name list for prefix fallback matching —
    SEC's own ticker registry and its 13(f) securities list sometimes format
    the same company differently (e.g. "ExxonMobil Holdings Corp" vs the
    13(f) list's "EXXON MOBIL CORP"; "O REILLY" vs "OREILLY").
    """
    cache_path = os.path.join(screener.CACHE_DIR, "ticker_name_index.json")
    try:
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 86400:
            with open(cache_path) as f:
                return json.load(f)
    except Exception:
        pass

    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=20)
        entries = list(r.json().values())
    except Exception:
        entries = []

    word_idx: dict = {}
    squish_idx: dict = {}
    anyword_idx: dict = {}   # last-resort: any single distinctive word -> ticker
    for e in entries:
        tk = e.get("ticker", "")
        title = e.get("title", "")
        if not tk or not title or screener._is_non_common(tk):
            continue
        words = _norm_issuer_name(title)
        if not words:
            continue
        for key in (" ".join(words[:2]), words[0]):
            cur = word_idx.get(key)
            if cur is None or screener._ticker_score(tk) < screener._ticker_score(cur):
                word_idx[key] = tk
        sq = _squish(words)
        cur = squish_idx.get(sq)
        if cur is None or screener._ticker_score(tk) < screener._ticker_score(cur):
            squish_idx[sq] = tk
        # Some registry titles lead with a brand name before the formal name
        # ("PETROBRAS - PETROLEO BRASILEIRO SA"), so the 13(f) list's name
        # ("PETROLEO BRASILEIRO S A") never lines up on word #1. Index every
        # sufficiently distinctive word (not just the first two) as a fallback.
        # Same score-based conflict resolution as word_idx/squish_idx — this is
        # what correctly picks PBR over PBR-A when both share "PETROLEO" (two
        # share classes of the same company), rather than treating any repeat
        # as an unresolvable ambiguity.
        for w in words:
            if len(w) < 5:
                continue
            cur = anyword_idx.get(w)
            if cur is None or screener._ticker_score(tk) < screener._ticker_score(cur):
                anyword_idx[w] = tk

    result = {"word_idx": word_idx, "squish_list": sorted(squish_idx.items()),
             "anyword_idx": anyword_idx}
    try:
        with open(cache_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    return result


def _load_ticker_search_list() -> list:
    """Flat [{"ticker":, "name":}] list from SEC's bulk ticker file, for the
    ticker-input autocomplete. Includes preferred/warrant tickers too (unlike
    the CUSIP-resolution index) since a user might legitimately type one.
    Cached 24h — same source file as everything else that reads it."""
    cache_path = os.path.join(screener.CACHE_DIR, "ticker_search_list.json")
    try:
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 86400:
            with open(cache_path) as f:
                return json.load(f)
    except Exception:
        pass
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=20)
        entries = list(r.json().values())
    except Exception:
        entries = []
    seen = set()
    out = []
    for e in entries:
        tk, title = e.get("ticker", ""), e.get("title", "")
        if not tk or not title or tk in seen:
            continue
        seen.add(tk)
        out.append({"ticker": tk, "name": title})
    try:
        with open(cache_path, "w") as f:
            json.dump(out, f)
    except Exception:
        pass
    return out


@app.route("/api/ticker_list")
def ticker_list_route():
    """Full ticker+name list for the client-side autocomplete. Fetched once
    per browser session (not per keystroke) and filtered locally — a search
    endpoint hit on every keystroke was dominated by mobile round-trip
    latency, not server-side work. The list itself is ~9k small entries
    (a few hundred KB), so one cached fetch beats dozens of tiny slow ones.
    """
    entries = _load_ticker_search_list()
    resp = jsonify({"results": entries})
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


def _squish_prefix_lookup(squish_list: list, target: str) -> Optional[str]:
    """
    Find a ticker whose squished registry name is a prefix of `target` (or
    vice versa) — catches cases where SEC's two datasets tokenize the same
    name differently ("EXXON MOBIL CORP" vs "ExxonMobil Holdings Corp" both
    squish to strings sharing the "EXXONMOBIL" prefix). squish_list is sorted,
    so this is a linear scan bounded to the handful of CUSIPs that miss the
    exact-match tier — fine at this scale (thousands of entries, dozens of
    lookups).
    """
    if not target:
        return None
    for name, ticker in squish_list:
        if not name:
            continue
        if name.startswith(target) or target.startswith(name):
            # Require the shorter side to be a meaningfully long, specific
            # prefix (not just "A" or "CO") to avoid spurious collisions.
            if min(len(name), len(target)) >= 5:
                return ticker
    return None


def _resolve_cusips_13flist(cusips: list) -> dict:
    """
    Fast, in-memory, no-network CUSIP resolution: SEC's official 13(f)
    CUSIP->name list, cross-referenced against SEC's own ticker->name registry.
    Tried first since it resolves nearly everything in milliseconds; only
    genuine misses fall through to the slower per-CUSIP EDGAR full-text search.
    """
    cusip_name = _load_13f_official_list()          # keys are uppercase (SEC's own casing)
    idx = _load_ticker_name_index()
    word_idx = idx["word_idx"]
    squish_list = idx["squish_list"]
    anyword_idx = idx["anyword_idx"]
    out: dict = {}
    for c in cusips:
        # Some 13F filers write CUSIPs lowercase in their XML; the official
        # list and our lookups are case-normalized, so match case-insensitively
        # while still caching under the ORIGINAL casing (matches how the
        # CUSIP is recorded in the fund holdings we're resolving for).
        official_name = cusip_name.get(c.upper())
        if not official_name:
            continue
        words = _norm_issuer_name(official_name)
        if not words:
            continue
        ticker = word_idx.get(" ".join(words[:2])) or word_idx.get(words[0])
        if not ticker:
            ticker = _squish_prefix_lookup(squish_list, _squish(words))
        if not ticker:
            # Last resort: any single unambiguous distinctive word shared
            # between the two names (catches brand-name-first registry titles
            # like "PETROBRAS - PETROLEO BRASILEIRO SA" vs the 13(f) list's
            # "PETROLEO BRASILEIRO S A" — no word-#1 alignment, but "PETROLEO"
            # and "BRASILEIRO" both appear and are unambiguous).
            for w in sorted(words, key=len, reverse=True):
                if w in anyword_idx:
                    ticker = anyword_idx[w]
                    break
        if ticker:
            out[c] = ticker
    return out


def get_guru_holdings_tickers(refresh: bool = False, time_budget_s: float = 90.0) -> dict:
    """Union of every stock held by any tracked guru fund, as resolved tickers."""
    universe = get_guru_universe()
    cusips: set = set()
    for data in universe["funds"].values():
        if not data:
            continue
        for h in data.get("holdings", []):
            if h.get("cusip") and (h.get("value") or 0) > 0:
                cusips.add(h["cusip"])

    cache = {} if refresh else _load_cusip_ticker_cache()
    missing = [c for c in cusips if c not in cache]
    if missing:
        # Tier 1: direct CUSIP→ticker table from SEC's Fails-to-Deliver files.
        # Exact-identifier match, no name heuristics — runs FIRST because it can
        # never confuse similarly-named issuers (D R HORTON vs D-Wave, WW
        # GRAINGER vs WW International), and it covers foreign CINS identifiers
        # that trade in the US.
        resolved = _resolve_cusips_ftd(missing)
        cache.update(resolved)
        _save_cusip_ticker_cache(cache)

        # Tier 2: in-memory name matching via SEC's official 13(f) securities
        # list cross-referenced against SEC's ticker registry.
        still_missing = [c for c in missing if c not in resolved]
        if still_missing:
            resolved_13f = _resolve_cusips_13flist(still_missing)
            cache.update(resolved_13f)
            resolved.update(resolved_13f)
            _save_cusip_ticker_cache(cache)

        # Tier 3: whatever still couldn't resolve falls through to per-CUSIP
        # EDGAR full-text search (subject-company display names on ownership
        # filings, plus issuer CIK → ticker on issuer-made filings).
        still_missing = [c for c in missing if c not in resolved]
        if still_missing:
            holding_names = {}
            for data in universe["funds"].values():
                for h in (data or {}).get("holdings", []):
                    if h.get("cusip") in still_missing and h.get("name"):
                        holding_names.setdefault(h["cusip"], h["name"])
            resolved2 = _resolve_cusips_edgar(still_missing, time_budget_s=time_budget_s,
                                              names=holding_names)
            cache.update(resolved2)
            _save_cusip_ticker_cache(cache)

    # Keep only plain equity-ticker-shaped strings — some CUSIPs resolve to
    # bond/note identifiers when a fund holds fixed income; those can't be
    # screened as stocks anyway.
    def _looks_like_ticker(t: str) -> bool:
        return bool(t) and len(t) <= 6 and t.replace(".", "").replace("-", "").isalnum()

    tickers = sorted({cache[c] for c in cusips if cache.get(c) and _looks_like_ticker(cache[c])})
    scanned = len([c for c in cusips if c in cache])

    # Surface anything we couldn't map to a ticker — shown to the user so they
    # can look these up manually rather than silently dropping them. Include
    # the best-known name and which fund(s) hold it, so it's actually useful
    # to check offline, not just a bare CUSIP.
    unresolved_cusips = {c for c in cusips if not (cache.get(c) and _looks_like_ticker(cache[c]))}
    official_names = _load_13f_official_list()
    manager_by_cik = {g["cik"]: g["manager"] for g in reversed(GURUS)}
    unresolved: dict = {}
    for cik, data in universe["funds"].items():
        if not data:
            continue
        for h in data.get("holdings", []):
            c = h.get("cusip")
            if c not in unresolved_cusips:
                continue
            entry = unresolved.setdefault(c, {
                "cusip": c,
                "name": official_names.get(c.upper()) or h["name"],
                "funds": [],
            })
            entry["funds"].append({
                "cik": cik,
                "value": h.get("value") or 0,
                "manager": manager_by_cik.get(cik, str(cik)),
                "link": data.get("link", ""),
            })
    unresolved_list = []
    for c, entry in unresolved.items():
        total_value = sum(f["value"] for f in entry["funds"])
        entry["funds"].sort(key=lambda f: f["value"], reverse=True)
        unresolved_list.append({
            "cusip": c, "name": entry["name"],
            "held_by_funds": len(entry["funds"]),
            "total_value": round(total_value),
            "funds": [{"manager": f["manager"], "link": f["link"]} for f in entry["funds"]],
        })
    unresolved_list.sort(key=lambda x: x["total_value"], reverse=True)

    return {
        "tickers": tickers,
        "unresolved": unresolved_list,
        "total_cusips": len(cusips),
        "resolved_cusips": scanned,
        "pending_cusips": len(cusips) - scanned,
        "complete": scanned == len(cusips),
    }



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


def _short_interest_finra(ticker: str) -> dict:
    """Latest bi-monthly short-interest settlement from FINRA's own public
    Consolidated Short Interest API (Rule 4560 reporting, covers all
    exchanges — NYSE, Nasdaq, NYSE American, etc.). No auth required."""
    try:
        cutoff = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        r = requests.post(
            "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest",
            json={"compareFilters": [
                {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": ticker.upper()},
                {"compareType": "GTE", "fieldName": "settlementDate", "fieldValue": cutoff},
            ], "limit": 10},
            headers={"Accept": "application/json"}, timeout=15,
        )
        if r.status_code != 200:
            return {}
        rows = r.json() or []
        if not rows:
            return {}
        row = max(rows, key=lambda x: x.get("settlementDate", ""))
        shares_short = float(row.get("currentShortPositionQuantity") or 0)
        if shares_short <= 0:
            return {}
        return {
            "shares_short": shares_short,
            "settlement_date": row.get("settlementDate"),
            "days_to_cover": row.get("daysToCoverQuantity"),
        }
    except Exception:
        return {}


def _short_interest_nasdaq(ticker: str) -> dict:
    """Fallback: Nasdaq's public short-interest API (Nasdaq-listed names only)."""
    sym = ticker.replace(".", "/").replace("-", ".")   # BRK.B -> BRK/B for Nasdaq's URL scheme
    for candidate in (ticker.replace(".", "").replace("-", ""), ticker, sym):
        try:
            r = requests.get(f"https://api.nasdaq.com/api/quote/{candidate}/short-interest",
                             params={"assetclass": "stocks"},
                             headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                             timeout=15)
            if r.status_code != 200:
                continue
            rows = (r.json().get("data") or {}).get("shortInterestTable", {}).get("rows") or []
            if not rows:
                continue
            row = rows[0]   # most recent settlement date first
            shares_short = float((row.get("interest") or "0").replace(",", ""))
            m, d, y = (row.get("settlementDate") or "").split("/")
            return {
                "shares_short": shares_short,
                "settlement_date": f"{y}-{m}-{d}" if y else None,
                "days_to_cover": row.get("daysToCover"),
            }
        except Exception:
            continue
    return {}


def get_short_interest(ticker: str) -> dict:
    """Latest bi-monthly short-interest settlement (shares short, days to
    cover). FINRA's Consolidated Short Interest API is tried first (covers
    every exchange); Nasdaq's short-interest API is the backup for anything
    FINRA doesn't return. Short % is computed by the caller against EDGAR
    shares outstanding."""
    return _short_interest_finra(ticker) or _short_interest_nasdaq(ticker)


# ─── XBRL tag priority lists ─────────────────────────────────────────────────

METRIC_TAGS: dict[str, list[str]] = {
    # Income Statement
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet", "NetSales",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",  # AMR and similar mining/industrial
    ],
    "gross_profit": ["GrossProfit"],
    # Split cost-of-revenue components: some filers (e.g. INTU pre-2018) tag
    # CostOfGoodsSold and CostOfServices separately with no consolidated total.
    "cost_of_goods_component":    ["CostOfGoodsSold"],
    "cost_of_services_component": ["CostOfServices"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    # G&A as a separate line (useful for REIT NOI derivation: NOI = EBITDA + G&A)
    "general_admin_expense": [
        "GeneralAndAdministrativeExpense",
        # Some companies embed G&A in SGA — only use as fallback when G&A is not separately filed
    ],
    # Selling & marketing expense (filed separately from G&A by some companies, e.g. TMHC post-2021)
    "selling_marketing_expense": [
        "SellingAndMarketingExpense",
        "SellingExpense",
    ],
    # Operating expense lines used for the gross-profit add-back fallback
    # (GP = OI + S&M + R&D + G&A + amortization + restructuring; e.g. INTU post-2018)
    "amortization_of_intangibles": ["AmortizationOfIntangibleAssets"],
    "restructuring_charges": ["RestructuringCharges", "RestructuringCosts"],
    "operating_income": [
        "OperatingIncomeLoss",
        # Fallback for companies (e.g. BRK) that don't separately file OperatingIncomeLoss
        # but report pre-tax earnings — closest available proxy for conglomerates/insurers.
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",   # MCK FY2025+ and similar
        "InterestAndDebtExpense",
    ],
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
        "PaymentsToExploreAndDevelopOilAndGasProperties",  # E&P drilling capex (CHRD, APA, and similar)
        "PaymentsToAcquirePropertyPlantEquipmentAndIntangibleAssets",
        "SegmentExpenditureAdditionToLongLivedAssets",
        "PaymentsForCapitalImprovements",               # Noble Corporation (NE) and similar
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "CostOfGoodsAndServicesSoldDepreciationAndAmortization",  # NE (Noble) and similar drillers
    ],
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
        # Limited partnerships / MLPs report partners' capital, not stockholders'
        # equity (e.g. PAGP – Plains GP Holdings LP). Plain (parent) tag first,
        # then the including-NCI total, then LLC members' equity.
        "PartnersCapital",
        "PartnersCapitalIncludingPortionAttributableToNoncontrollingInterest",
        "MembersEquity",
        "MembersEquityIncludingPortionAttributableToNoncontrollingInterest",
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
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",  # INTU post-FY2023
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",       # DPZ securitization + leases (noncurrent)
        "LongTermDebtAndFinanceLeaseObligations",
        "DebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermNotesPayable",                         # ORCL (annual 10-K noncurrent)
        "LongTermNotesAndLoans",                        # ORCL (quarterly 10-Q noncurrent)
        "LongTermDebt",                                 # generic, may be partial (e.g. DPZ ~$14M only)
        "FinanceLeaseLiabilityNoncurrent",              # last: lease-financed cos (e.g. LIVE post-2022 sale-leasebacks)
    ],
    "current_debt": [
        "LongTermDebtAndCapitalLeaseObligationsCurrent",  # DPZ current portion of securitization
        "LongTermDebtCurrent",
        "DebtCurrent",
        "NotesPayableCurrent",                          # ORCL quarterly current debt
        "ShortTermBorrowings",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "CommercialPaper",
        "ShortTermDebt",
        "FinanceLeaseLiabilityCurrent",                 # last: lease-financed cos (e.g. LIVE)
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
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "buybacks_value": [
        "PaymentsForRepurchaseOfCommonStock",
        "StockRepurchasedAndRetiredDuringPeriodValue",
        "StockRepurchasedDuringPeriodValue",
    ],
    "treasury_stock": ["TreasuryStockCommonValue", "TreasuryStockValue"],
    "shares_repurchased": [
        "StockRepurchasedAndRetiredDuringPeriodShares",
        "StockRepurchasedDuringPeriodShares",
        "TreasuryStockSharesAcquired",
    ],
    "treasury_stock_shares": ["TreasuryStockCommonShares", "TreasuryStockShares"],

    # Buyback program remaining (best-effort; not all companies report via XBRL)
    "buyback_remaining": [
        "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
        "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount",
    ],

    # ── REIT-specific metrics ─────────────────────────────────────────────────
    # Gains / losses on real estate dispositions — subtracted in FFO derivation
    "gains_on_real_estate": [
        "GainLossOnSaleOfProperties",
        "GainsLossesOnSalesOfInvestmentRealEstate",
        "GainLossOnDispositionOfRealEstateAssets",
        "GainOnSaleOfProperties",
        "GainLossOnSaleOfPropertiesBeforeApplicableIncomeTaxes",  # SPG and similar
    ],
    # Real property depreciation — added back in FFO (may differ from total D&A).
    # Only REIT-specific tags here; general D&A fallback is applied in build_financials
    # but only when other REIT signals (real_estate_assets, straight_line_rent) are present.
    "real_estate_depreciation": [
        "DepreciationOfRealEstate",          # standard REIT tag (most REITs file this)
        "RealEstateDepreciationAndAmortization",
    ],
    # Straight-line rent adjustment — stripped out in AFFO derivation
    "straight_line_rent": [
        "StraightLineRent",
        "StraightLineRentAdjustments",
    ],
    # Net real estate assets on balance sheet
    "real_estate_assets": [
        "RealEstateInvestmentPropertyNet",
        "RealEstateAndAccumulatedDepreciation",  # alternative tag
    ],
    # Recurring (maintenance/tenant improvement) capex — used in AFFO
    "recurring_capex": [
        "PaymentsForTenantImprovements",
        "PaymentsForLeasingCosts",
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

    # ── Insurance (P&C / life) metrics ───────────────────────────────────────
    # Net premiums earned — the insurer's top-line "revenue" for ratio math
    "premiums_earned": [
        "PremiumsEarnedNet",
        "PremiumsEarnedNetPropertyAndCasualty",
        "SupplementaryInsuranceInformationPremiumRevenue",
    ],
    # Net premiums written (leading indicator of earned premium growth)
    "premiums_written": [
        "PremiumsWrittenNet",
        "SupplementaryInsuranceInformationPremiumsWritten",
    ],
    # Losses & loss-adjustment expenses incurred (numerator of the loss ratio)
    "losses_incurred": [
        "PolicyholderBenefitsAndClaimsIncurredNet",
        "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaims1",
        "SupplementaryInsuranceInformationBenefitsClaimsLossesAndSettlementExpense",
    ],
    # Total benefits, losses & expenses (numerator of the combined ratio)
    "benefits_losses_expenses": ["BenefitsLossesAndExpenses"],
    # Loss & LAE reserves (largest float component)
    "claims_reserve": [
        "LiabilityForClaimsAndClaimsAdjustmentExpense",
        "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseNet",
        "SupplementaryInsuranceInformationLiabilityForFuturePolicyBenefitsLossesClaimsAndLossExpenseReserves",
    ],
    # Unearned premium reserve (float component)
    "unearned_premiums": [
        "UnearnedPremiums",
        "SupplementaryInsuranceInformationUnearnedPremiums",
    ],
    # Premiums receivable (offsets float — money not yet collected)
    "premiums_receivable": ["PremiumsReceivableAtCarryingValue"],
    # Deferred policy acquisition costs (offsets float)
    "deferred_acquisition_costs": [
        "DeferredPolicyAcquisitionCosts",
        "SupplementaryInsuranceInformationDeferredPolicyAcquisitionCosts",
    ],
    # Reinsurance recoverables (offsets float)
    "reinsurance_recoverable": [
        "ReinsuranceRecoverablesOnPaidAndUnpaidLosses",
        "ReinsuranceRecoverableForUnpaidClaimsAndClaimsAdjustments",
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
                    # First tag wins per end-date (earlier/more-specific tags take priority)
                    if end not in bs_by_end:
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

    # First-tag-wins per date: earlier tags in the list take priority.
    # This prevents catch-all tags (e.g. CashCashEquivalentsRestrictedCash...)
    # from overriding more specific tags (e.g. CashAndCashEquivalentsAtCarryingValue).
    # Within a single tag, take the largest absolute value to handle comparatives.
    per_tag: list[dict[str, float]] = []

    for tag in tags:
        concept = gaap.get(tag) or dei.get(tag)
        if not concept:
            continue
        units = concept.get("units", {})
        tag_data: dict[str, float] = {}
        for unit_key in ["shares", "USD", "pure", "USD/shares"]:
            entries = units.get(unit_key, [])
            if not entries:
                continue
            for e in entries:
                if e.get("form") not in {"10-K", "10-K/A", "20-F", "20-F/A"}:
                    continue
                end = e.get("end", "")
                val = e.get("val")
                if val is None or not end:
                    continue
                fy_field = e.get("fy")
                if fy_field and (int(end[:4]) - fy_field == 1):
                    end = f"{fy_field}{end[4:]}"
                if end not in tag_data or abs(val) > abs(tag_data[end]):
                    tag_data[end] = val
            break
        if tag_data:
            per_tag.append(tag_data)

    # Merge: first tag that has a value for a given date wins
    merged: dict[str, float] = {}
    for tag_data in per_tag:
        for end, val in tag_data.items():
            if end not in merged:
                merged[end] = val
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
        "treasury_stock",
        "treasury_stock_shares",
        # BDC point-in-time
        "nav_per_share",
        # REIT point-in-time
        "real_estate_assets",
        # Insurance point-in-time (balance-sheet float components)
        "claims_reserve",
        "unearned_premiums",
        "premiums_receivable",
        "deferred_acquisition_costs",
        "reinsurance_recoverable",
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

    # OI fallback: derive from Gross Profit − operating expenses when OperatingIncomeLoss
    # is not tagged (e.g. TMHC post-2021 files GP + SellingAndMarketing + G&A separately).
    # Build a combined opex series covering all years where GP is known but OI is missing.
    if gp:
        _sga  = raw.get("sga_expense",             {})  # SellingGeneralAndAdministrativeExpense
        _ga   = raw.get("general_admin_expense",    {})  # GeneralAndAdministrativeExpense
        _sm   = raw.get("selling_marketing_expense",{})  # SellingAndMarketingExpense
        _oi_derived: dict[str, float] = {}
        for d in gp:
            if fy_get(oi, d[:4]) is not None:
                continue   # already have a direct OI value for this year
            # Total opex = SGA (combined) OR (S&M + G&A separately)
            sga_v = fy_get(_sga, d[:4])
            ga_v  = fy_get(_ga,  d[:4])
            sm_v  = fy_get(_sm,  d[:4])
            if sga_v is not None:
                opex = abs(sga_v)
            elif ga_v is not None or sm_v is not None:
                opex = abs(ga_v or 0) + abs(sm_v or 0)
            else:
                continue   # can't derive OI without any expense data
            _oi_derived[d] = gp[d] - opex
        if _oi_derived:
            if oi:
                oi.update(_oi_derived)
            else:
                oi = _oi_derived
            raw["operating_income"] = oi

        # Backfill sga_expense for years where the combined tag is absent but
        # the components (S&M + G&A) were filed separately (e.g. TMHC 2024+).
        _sga_existing = raw.get("sga_expense", {})
        _all_expense_dates = set(_sm) | set(_ga)
        _sga_fill: dict[str, float] = {}
        for d in _all_expense_dates:
            if fy_get(_sga_existing, d[:4]) is not None:
                continue   # combined tag already covers this year
            ga_v = fy_get(_ga, d[:4]) or 0.0
            sm_v = fy_get(_sm, d[:4]) or 0.0
            if ga_v or sm_v:
                _sga_fill[d] = abs(ga_v) + abs(sm_v)
        if _sga_fill:
            if _sga_existing:
                _sga_existing.update(_sga_fill)
            else:
                raw["sga_expense"] = _sga_fill

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

    # Combine split cost-of-revenue components (goods + services) per year.
    # The tag merge keeps the max single value, so a filer tagging both
    # CostOfGoodsSold and CostOfServices (e.g. INTU pre-2018) ends up with
    # only one component. The true total is the sum; a genuine total tag
    # always equals or exceeds the sum, so taking the max is safe.
    _cos_comp = raw.get("cost_of_services_component", {})
    if _cos_comp:
        _cog_comp = raw.get("cost_of_goods_component", {})
        _cogs_combined = dict(raw.get("cost_of_revenue", {}))
        for d, sv in _cos_comp.items():
            y = d[:4]
            combined = abs(sv) + abs(fy_get(_cog_comp, y) or 0)
            existing = fy_get(_cogs_combined, y)
            if existing is None:
                _cogs_combined[d] = combined
            elif combined > abs(existing):
                for k in list(_cogs_combined):
                    if k[:4] == y:
                        _cogs_combined[k] = combined
        raw["cost_of_revenue"] = _cogs_combined

    # Gross Profit = Revenue - Cost of Revenue for years not directly tagged.
    # Some filers switch away from GrossProfit while still reporting revenue/cost.
    if rev:
        cogs = raw.get("cost_of_revenue", {})
        if cogs:
            gp_filled = dict(gp or {})
            for d in rev:
                if fy_get(gp_filled, d[:4]) is not None:
                    continue
                c = fy_get(cogs, d[:4])
                if c is not None:
                    gp_filled[d] = rev[d] - abs(c)
            if gp_filled:
                raw["gross_profit"] = gp_filled
                gp = gp_filled

    # Gross Profit fallback #2: opex add-back, for filers that tag neither
    # GrossProfit nor a consolidated cost-of-revenue (e.g. INTU post-2018).
    # GP = OI + S&M + R&D + G&A (+ amortization of intangibles + restructuring),
    # i.e. add back every operating expense below the gross-profit line.
    # Requires SGA (or S&M + G&A) so we don't fabricate GP from OI alone.
    if rev and oi:
        _sga_gp = raw.get("sga_expense", {})
        _sm_gp  = raw.get("selling_marketing_expense", {})
        _ga_gp  = raw.get("general_admin_expense", {})
        _rd_gp  = raw.get("rd_expense", {})
        _am_gp  = raw.get("amortization_of_intangibles", {})
        _rst_gp = raw.get("restructuring_charges", {})
        gp_fill2 = dict(gp or {})
        _added = False
        for d in rev:
            y = d[:4]
            if fy_get(gp_fill2, y) is not None:
                continue
            o = fy_get(oi, y)
            if o is None:
                continue
            sm_v  = fy_get(_sm_gp, y)
            ga_v  = fy_get(_ga_gp, y)
            sga_v = fy_get(_sga_gp, y)
            if sm_v is not None and ga_v is not None:
                opex = abs(sm_v) + abs(ga_v)
            elif sga_v is not None:
                opex = abs(sga_v)
            else:
                continue
            opex += abs(fy_get(_rd_gp,  y) or 0)
            opex += abs(fy_get(_am_gp,  y) or 0)
            opex += abs(fy_get(_rst_gp, y) or 0)
            gp_fill2[d] = o + opex
            _added = True
        if _added:
            raw["gross_profit"] = gp_fill2
            gp = gp_fill2

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

    # EBIT = Operating Income (standard proxy; same XBRL source)
    if oi:
        raw["ebit"] = oi
        margin(oi, rev, "ebit_margin")

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

    # Buybacks fallback: derive from YoY change in Treasury Stock balance
    # Needed for companies (e.g. AMR) that file repurchase cash flows as
    # cumulative-since-inception, which the 10-14 month period filter rejects.
    bb = raw.get("buybacks_value", {})
    ts = raw.get("treasury_stock", {})
    if ts:
        # Build {year: (end_date, value)} keeping the latest date per year
        ts_by_year: dict[str, tuple[str, float]] = {}
        for d in sorted(ts):
            ts_by_year[d[:4]] = (d, ts[d])
        sorted_ts_years = sorted(ts_by_year)
        for i, y in enumerate(sorted_ts_years[1:], 1):
            prev_y = sorted_ts_years[i - 1]
            prev_val = ts_by_year[prev_y][1]
            curr_date, curr_val = ts_by_year[y]
            if fy_get(bb, y) is None and curr_val > prev_val:
                bb[curr_date] = curr_val - prev_val
        if bb:
            raw["buybacks_value"] = bb

    # Shares repurchased fallback: YoY change in treasury stock share count
    sh_bb = raw.get("shares_repurchased", {})
    ts_sh = raw.get("treasury_stock_shares", {})
    if ts_sh:
        ts_sh_by_year: dict[str, tuple[str, float]] = {}
        for d in sorted(ts_sh):
            ts_sh_by_year[d[:4]] = (d, ts_sh[d])
        sorted_ts_sh_years = sorted(ts_sh_by_year)
        for i, y in enumerate(sorted_ts_sh_years[1:], 1):
            prev_y = sorted_ts_sh_years[i - 1]
            prev_val = ts_sh_by_year[prev_y][1]
            curr_date, curr_val = ts_sh_by_year[y]
            if fy_get(sh_bb, y) is None and curr_val > prev_val:
                sh_bb[curr_date] = curr_val - prev_val
        if sh_bb:
            raw["shares_repurchased"] = sh_bb

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

    # Adj. FCF ROE = Adj. FCF (FCF - SBC) / Equity
    adj_fcf_for_roe = raw.get("adj_fcf", {})
    if adj_fcf_for_roe and eq:
        adj_fcf_roe = {}
        for d in adj_fcf_for_roe:
            e = fy_get(eq, d[:4])
            if e and e != 0:
                adj_fcf_roe[d] = adj_fcf_for_roe[d] / e
        raw["adj_fcf_roe"] = adj_fcf_roe or None

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

    # Pre-tax ROIC = EBIT / Invested Capital  (same IC as ROIC, no tax adjustment)
    if oi and eq:
        pretax_roic = {}
        for d in oi:
            y = d[:4]
            e      = fy_get(eq, y) or 0
            debt   = fy_get(td, y) or 0
            cash_v = fy_get(tc, y) or 0
            ic     = e + debt - cash_v
            if ic and ic > 0:
                pretax_roic[d] = oi[d] / ic
        raw["pretax_roic"] = pretax_roic or None

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
    # Skip years where UNTA is negative — dividing by negative tangible capital
    # produces a misleading negative ratio (same rationale as ROIC's IC > 0 guard).
    if _unta and _nopat:
        eco_gw: dict[str, float] = {}
        for d in _nopat:
            u = fy_get(_unta, d[:4])
            if u and u > 0:
                eco_gw[d] = _nopat[d] / u
        raw["economic_goodwill"] = eco_gw or None

    # Pre-tax Economic Goodwill = EBIT / UNTA
    # Same as economic_goodwill but uses pre-tax EBIT instead of NOPAT
    if _unta and oi:
        pretax_eco_gw: dict[str, float] = {}
        for d in oi:
            u = fy_get(_unta, d[:4])
            if u and u > 0:
                pretax_eco_gw[d] = oi[d] / u
        raw["pretax_economic_goodwill"] = pretax_eco_gw or None

    # ── REIT-specific derived metrics ─────────────────────────────────────────
    # Only compute REIT metrics when REIT-specific XBRL data is present.
    # If neither RealEstateInvestmentPropertyNet nor DepreciationOfRealEstate is
    # tagged, this is not a REIT and we skip FFO/AFFO derivation entirely.
    _has_reit_data = bool(
        raw.get("real_estate_assets") or
        raw.get("real_estate_depreciation") or
        raw.get("straight_line_rent")
    )
    # FFO (NAREIT definition) = Net Income + Real Estate Depreciation
    #                         − Gains on sale of real estate properties
    # Use DepreciationOfRealEstate when available; fall back to general D&A only
    # when real estate assets are present (confirming this is a REIT).
    re_dep_specific = raw.get("real_estate_depreciation", {})
    re_dep = re_dep_specific if re_dep_specific else (dep if _has_reit_data else {})
    re_gains = raw.get("gains_on_real_estate", {})
    if ni and re_dep:
        ffo: dict[str, float] = {}
        for d in ni:
            da_v = fy_get(re_dep, d[:4])
            if da_v is not None:
                gains_v = fy_get(re_gains, d[:4]) or 0.0
                ffo[d] = ni[d] + abs(da_v) - gains_v
        if ffo:
            raw["ffo"] = ffo
            # FFO per share
            if share_base_for_per_share:
                ffo_ps: dict[str, float] = {}
                for d in ffo:
                    s = fy_get(share_base_for_per_share, d[:4])
                    if s and s > 0:
                        ffo_ps[d] = ffo[d] / s
                raw["ffo_per_share"] = ffo_ps or None

    # AFFO (Adjusted FFO) = FFO − Straight-line Rent Adjustment − Recurring CapEx
    # Straight-line rent is a non-cash accrual that inflates FFO; recurring CapEx
    # (tenant improvements, leasing costs) is needed to maintain occupancy.
    ffo_series   = raw.get("ffo", {})
    slr          = raw.get("straight_line_rent", {})
    rec_cx       = raw.get("recurring_capex", {})
    if ffo_series:
        affo: dict[str, float] = {}
        for d in ffo_series:
            sl_v  = fy_get(slr,    d[:4]) or 0.0
            rc_v  = fy_get(rec_cx, d[:4]) or 0.0
            affo[d] = ffo_series[d] - abs(sl_v) - abs(rc_v)
        if affo:
            raw["affo"] = affo
            if share_base_for_per_share:
                affo_ps: dict[str, float] = {}
                for d in affo:
                    s = fy_get(share_base_for_per_share, d[:4])
                    if s and s > 0:
                        affo_ps[d] = affo[d] / s
                raw["affo_per_share"] = affo_ps or None

    # NOI (Net Operating Income) for REITs
    # Formula: NOI = EBITDA + G&A
    # Derivation: Income Statement = Revenue − PropertyOpEx − RETax − G&A − D&A
    #   → EBITDA (Operating Income + D&A) = Revenue − PropertyOpEx − RETax − G&A
    #   → NOI   = Revenue − PropertyOpEx − RETax = EBITDA + G&A
    # When G&A is not separately tagged, NOI ≈ EBITDA (acceptable for net-lease REITs
    # where G&A is a small fraction of revenue and tenants pay most property costs).
    ebitda_series = raw.get("ebitda", {})
    ga_series     = raw.get("general_admin_expense", {})
    if _has_reit_data and ebitda_series:
        noi: dict[str, float] = {}
        for d in ebitda_series:
            eb = ebitda_series[d]
            if eb is None:
                continue
            ga_v = fy_get(ga_series, d[:4]) or 0.0
            noi[d] = eb + abs(ga_v)
        if noi:
            raw["noi"] = noi
            margin(noi, rev, "noi_margin")
            # NOI per share
            if share_base_for_per_share:
                noi_ps: dict[str, float] = {}
                for d in noi:
                    s = fy_get(share_base_for_per_share, d[:4])
                    if s and s > 0:
                        noi_ps[d] = noi[d] / s
                raw["noi_per_share"] = noi_ps or None

    # FFO payout ratio = Dividends Paid / FFO
    divs = raw.get("dividends_paid", {})
    if ffo_series and divs:
        ffo_payout: dict[str, float] = {}
        for d in ffo_series:
            dv = fy_get(divs, d[:4])
            if dv is not None and ffo_series[d] and ffo_series[d] > 0:
                ffo_payout[d] = abs(dv) / ffo_series[d]
        raw["ffo_payout_ratio"] = ffo_payout or None

    # ── Insurance-specific derived metrics ────────────────────────────────────
    prem_earned = raw.get("premiums_earned", {})
    losses      = raw.get("losses_incurred", {})
    ble         = raw.get("benefits_losses_expenses", {})
    if prem_earned:
        # Loss Ratio = Losses & LAE Incurred / Net Premiums Earned
        if losses:
            loss_ratio: dict[str, float] = {}
            for d in prem_earned:
                pe = prem_earned[d]
                lo = fy_get(losses, d[:4])
                if pe and pe > 0 and lo is not None:
                    loss_ratio[d] = abs(lo) / pe
            raw["loss_ratio"] = loss_ratio or None

        # Combined Ratio = Total Benefits, Losses & Expenses / Net Premiums Earned
        # (approximate — includes any non-underwriting expense lines; for a
        # predominantly P&C insurer this tracks the reported combined ratio closely)
        if ble:
            combined_ratio: dict[str, float] = {}
            for d in prem_earned:
                pe = prem_earned[d]
                be = fy_get(ble, d[:4])
                if pe and pe > 0 and be is not None:
                    combined_ratio[d] = abs(be) / pe
            raw["combined_ratio"] = combined_ratio or None

            # Expense Ratio = Combined − Loss (underwriting expenses / premiums earned)
            _lr = raw.get("loss_ratio", {})
            _cr = raw.get("combined_ratio", {})
            if _lr and _cr:
                expense_ratio: dict[str, float] = {}
                for d in _cr:
                    lr = fy_get(_lr, d[:4])
                    if lr is not None:
                        expense_ratio[d] = _cr[d] - lr
                raw["expense_ratio"] = expense_ratio or None

    # Insurance Float ≈ Loss & LAE Reserves + Unearned Premiums
    #                   − Premiums Receivable − Deferred Acquisition Costs
    #                   − Reinsurance Recoverables
    # The pool of policyholder money the insurer holds and invests before paying claims.
    _claims = raw.get("claims_reserve", {})
    if _claims:
        _unearn = raw.get("unearned_premiums", {})
        _prem_r = raw.get("premiums_receivable", {})
        _dac    = raw.get("deferred_acquisition_costs", {})
        _reins  = raw.get("reinsurance_recoverable", {})
        flo: dict[str, float] = {}
        for d in _claims:
            y = d[:4]
            flo[d] = (
                _claims[d]
                + (fy_get(_unearn, y) or 0)
                - (fy_get(_prem_r, y) or 0)
                - (fy_get(_dac,    y) or 0)
                - (fy_get(_reins,  y) or 0)
            )
        raw["insurance_float"] = flo or None

        # Float per share
        if share_base_for_per_share and raw.get("insurance_float"):
            fps_i: dict[str, float] = {}
            for d in raw["insurance_float"]:
                s = fy_get(share_base_for_per_share, d[:4])
                if s and s > 0:
                    fps_i[d] = raw["insurance_float"][d] / s
            raw["float_per_share"] = fps_i or None

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


@app.route("/api/universes")
def universes():
    """Available named universes for the screener dropdown."""
    out = [{"key": "guru_holdings",
           "label": f"Guru Holdings ({len(_unique_guru_ciks())} Funds)"}]
    out += [{"key": k, "label": v} for k, v in screener.UNIVERSE_LABELS.items()]
    return jsonify({"universes": out})


@app.route("/api/screen")
def screen_route():
    """
    Valuation screener. Params:
      universe   — sp500 | nasdaq100 | dow30 | custom
      tickers    — comma-separated list (used when universe=custom)
      max_pfcf   — upper P/FCF cutoff (optional)
      max_ev_ebit— upper EV/EBIT cutoff (optional)
      fy         — fiscal year to value against (default: most recent complete)
    """
    universe = request.args.get("universe", "sp500").lower().strip()
    fy       = int(request.args.get("fy", 2025))

    def _f(name):
        v = request.args.get(name, "").strip()
        try:
            return float(v) if v else None
        except ValueError:
            return None
    max_pfcf    = _f("max_pfcf")
    max_ev_ebit = _f("max_ev_ebit")
    # Market-cap cutoffs arrive in $B from the UI; convert to dollars.
    _min_b      = _f("min_mktcap_b")
    _max_b      = _f("max_mktcap_b")
    min_mktcap  = _min_b * 1e9 if _min_b is not None else None
    max_mktcap  = _max_b * 1e9 if _max_b is not None else None

    refresh = request.args.get("refresh", "").strip() in ("1", "true", "yes")

    # Sector exclusions default to ON; pass "0"/"false" to include a sector.
    def _flag(name):
        return request.args.get(name, "1").strip() not in ("0", "false", "no")
    remove_insurance = _flag("remove_insurance")
    remove_banks     = _flag("remove_banks")
    remove_reits     = _flag("remove_reits")
    remove_20f       = _flag("remove_20f")

    guru_progress = None
    try:
        if universe == "custom":
            raw = request.args.get("tickers", "")
            tickers = [t.strip().upper() for t in re.split(r"[,\s]+", raw) if t.strip()]
            if not tickers:
                return jsonify({"error": "Provide tickers for a custom screen"}), 400
        elif universe == "guru_holdings":
            gh = get_guru_holdings_tickers(refresh=refresh)
            tickers = gh["tickers"]
            guru_progress = gh
            if not tickers:
                return jsonify({
                    "error": "Still resolving the guru holdings universe (CUSIP → ticker lookups) "
                             "— this can take a couple of minutes on a cold cache. Try again shortly.",
                    "results": [], "stats": {"pending": True},
                }), 200
        else:
            tickers = screener.get_universe(universe)
            if not tickers:
                return jsonify({"error": f"Unknown universe '{universe}'"}), 400

        result = screener.screen(universe, tickers, max_pfcf, max_ev_ebit, fy,
                                 min_mktcap=min_mktcap, max_mktcap=max_mktcap,
                                 refresh=refresh,
                                 remove_insurance=remove_insurance,
                                 remove_banks=remove_banks,
                                 remove_reits=remove_reits,
                                 remove_20f=remove_20f)
    except Exception as e:
        return jsonify({"error": f"Screen failed: {e}"}), 500

    # Attach company names from the SEC ticker map (cheap, already cached)
    if universe == "guru_holdings":
        result["stats"]["label"] = f"Guru Holdings ({len(_unique_guru_ciks())} Funds)"
        if guru_progress and not guru_progress["complete"]:
            result["stats"]["guru_note"] = (
                f"{guru_progress['resolved_cusips']} of {guru_progress['total_cusips']} "
                f"holdings resolved to tickers so far — reload to pick up more."
            )
        if guru_progress and guru_progress.get("unresolved"):
            result["guru_unresolved"] = guru_progress["unresolved"]
    else:
        result["stats"]["label"] = screener.UNIVERSE_LABELS.get(universe, universe.upper())
    return jsonify(result)


@app.route("/api/vic")
def vic():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    ideas = fetch_vic_ideas(ticker)
    return jsonify({"ticker": ticker, "ideas": ideas})


# ─── Company news (Google News RSS + Yahoo Finance, aggregated) ────────────────

_NEWS_PREMIUM_SITES = ("wsj.com", "nytimes.com", "fortune.com", "reuters.com",
                       "ft.com", "bloomberg.com", "cnbc.com", "barrons.com",
                       "economist.com", "forbes.com")


def _news_query_name(company_name: str) -> str:
    """Short, searchable company name: strip corporate suffixes and keep the
    leading distinctive words ('Apple Inc.' → 'Apple'; 'DXC Technology
    Company' → 'DXC Technology')."""
    words = re.sub(r"[.,]", " ", company_name).split()
    stop = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "PLC", "SA",
            "NV", "LLC", "LP", "GROUP", "HOLDINGS", "THE", "&"}
    kept = [w for w in words if w.upper() not in stop]
    return " ".join(kept[:2]) if kept else company_name


def _google_news_rss(query: str, limit: int = 30) -> list[dict]:
    try:
        r = requests.get("https://news.google.com/rss/search",
                         params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []
    from email.utils import parsedate_to_datetime
    out = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        src_el = item.find("source")
        source = src_el.text if src_el is not None else ""
        # Google News suffixes titles with " - Source"; strip the duplicate.
        if source and title.endswith(" - " + source):
            title = title[: -len(" - " + source)]
        ts = 0
        try:
            ts = int(parsedate_to_datetime(item.findtext("pubDate") or "").timestamp())
        except Exception:
            pass
        if title and link:
            out.append({"title": title.strip(), "url": link, "source": source, "ts": ts})
        if len(out) >= limit:
            break
    return out


def _yahoo_news(ticker: str, limit: int = 15) -> list[dict]:
    try:
        r = requests.get("https://query1.finance.yahoo.com/v1/finance/search",
                         params={"q": ticker, "newsCount": limit, "quotesCount": 0},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            return []
        news = r.json().get("news", []) or []
    except Exception:
        return []
    return [{"title": (n.get("title") or "").strip(),
             "url": n.get("link") or "",
             "source": n.get("publisher") or "",
             "ts": int(n.get("providerPublishTime") or 0)}
            for n in news if n.get("title") and n.get("link")]


def get_company_news(ticker: str, company_name: str) -> list[dict]:
    """Aggregated recent news for one company: a premium-source Google News
    query (WSJ/NYT/Fortune/Bloomberg/Reuters/CNBC/Barron's...), a general
    Google News query, and Yahoo Finance's feed — merged, deduped by
    normalized title, newest first. Cached 15 minutes per ticker."""
    cache_path = os.path.join(screener.CACHE_DIR, f"news_{ticker}.json")
    try:
        if os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < 15 * 60:
            with open(cache_path) as f:
                return json.load(f)
    except Exception:
        pass

    qname = _news_query_name(company_name) if company_name else ticker
    sites = " OR ".join(f"site:{s}" for s in _NEWS_PREMIUM_SITES)
    premium = _google_news_rss(f'"{qname}" ({sites}) when:30d')
    general = _google_news_rss(f'"{qname}" OR {ticker} stock when:14d', limit=20)
    # Yahoo's feed pads results with generic market stories about unrelated
    # companies — keep only items that actually name this company or ticker.
    name_words = [w.lower() for w in qname.split() if len(w) >= 3]
    yahoo = [n for n in _yahoo_news(ticker)
             if ticker.lower() in n["title"].lower()
             or any(w in n["title"].lower() for w in name_words)]

    def _norm_title(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()[:80]

    seen: set = set()
    merged = []
    # Premium sources first so they win the dedupe against syndicated copies.
    for item in premium + general + yahoo:
        key = _norm_title(item["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged.sort(key=lambda x: x["ts"], reverse=True)
    merged = merged[:30]
    premium_names = {"wsj", "wallstreetjournal", "thenewyorktimes", "nytimes", "fortune",
                     "reuters", "reutersvideos", "bloomberg", "bloombergcom", "cnbc",
                     "barrons", "barronscom", "financialtimes", "ft", "theeconomist", "forbes"}
    for m in merged:
        m["date"] = datetime.utcfromtimestamp(m["ts"]).strftime("%Y-%m-%d") if m["ts"] else ""
        m["premium"] = re.sub(r"[^a-z]", "", m["source"].lower()) in premium_names
    try:
        with open(cache_path, "w") as f:
            json.dump(merged, f)
    except Exception:
        pass
    return merged


@app.route("/api/news")
def company_news():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    cik = resolve_cik(ticker)
    name = _cached_company_name(cik) if cik else ""
    try:
        items = get_company_news(ticker, name)
        return jsonify({"ticker": ticker, "items": items})
    except Exception as e:
        return jsonify({"ticker": ticker, "items": [], "error": str(e)}), 200


@app.route("/api/insider")
def insider():
    """Insider open-market purchases (Form 4, code P). Loaded asynchronously by
    the frontend so the many Form 4 XML fetches don't slow the main analyze
    response. Cached 6h per CIK; a cache hit skips even the submissions fetch."""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    cik = resolve_cik(ticker)
    if not cik:
        return jsonify({"error": f"Ticker '{ticker}' not found"}), 404

    # refresh=1 purges this company's per-filing cache so everything is re-fetched.
    if request.args.get("refresh", "").strip() in ("1", "true", "yes"):
        try:
            os.remove(_filing_cache_path(str(int(cik))))
        except OSError:
            pass

    # get_insider_purchases caches each Form 4 individually (immutable filings)
    # and only fetches what's still missing, so repeat calls are cheap and a
    # rate-limited run resumes cumulatively on the next call.
    try:
        submissions = fetch_submissions(cik)
        return jsonify(get_insider_purchases(submissions))
    except Exception as e:
        return jsonify({"error": f"Insider fetch failed: {e}",
                        "purchases": [], "trend": {}, "total_value": 0,
                        "total_count": 0, "rate_limited": True}), 200


@app.route("/api/top_shareholders")
def top_shareholders():
    """Top shareholders derived from Schedule 13D/13G filings over the last
    12 months. Loaded asynchronously, same pattern as /api/insider — cached
    per accession so repeat calls are cheap and a rate-limited run resumes."""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    cik = resolve_cik(ticker)
    if not cik:
        return jsonify({"error": f"Ticker '{ticker}' not found"}), 404

    if request.args.get("refresh", "").strip() in ("1", "true", "yes"):
        try:
            os.remove(_sc13_cache_path(str(int(cik))))
        except OSError:
            pass

    try:
        submissions = fetch_submissions(cik)
        return jsonify(get_top_shareholders(submissions))
    except Exception as e:
        return jsonify({"error": f"Top shareholders fetch failed: {e}",
                        "holders": [], "total_count": 0, "rate_limited": True}), 200


def _cached_company_name(cik: str) -> str:
    """Company name for issuer-matching, cached long-term (names essentially
    never change) so /api/holders doesn't re-fetch the full submissions JSON
    from SEC on every call — that was hitting SEC's rate limit and turning
    transient throttling into a full request failure."""
    cache_path = os.path.join(screener.CACHE_DIR, f"company_name_{cik}.json")
    try:
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 30 * 86400:
            with open(cache_path) as f:
                return json.load(f)["name"]
    except Exception:
        pass
    submissions = fetch_submissions(cik)
    name = submissions.get("name", "") or ""
    try:
        with open(cache_path, "w") as f:
            json.dump({"name": name}, f)
    except Exception:
        pass
    return name


@app.route("/api/holders")
def holders():
    """Institutional 13F holders of the stock, scanned from a curated ~92-fund
    value-investor roster (see GURUS). The roster's 13F data is fetched and
    cached ONCE per quarter, shared across every stock — this endpoint mostly
    does zero SEC calls once that shared cache is warm."""
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    cik = resolve_cik(ticker)
    if not cik:
        return jsonify({"error": f"Ticker '{ticker}' not found"}), 404

    # refresh=1 purges the SHARED guru-fund cache, forcing the whole roster to
    # be refetched (expensive — an explicit user action, not automatic).
    refresh = request.args.get("refresh", "").strip() in ("1", "true", "yes")

    # Shares outstanding comes from the analyze flow (avoids a redundant fetch).
    try:
        shares_out = float(request.args.get("shares_out", "") or 0) or None
    except ValueError:
        shares_out = None

    try:
        name = _cached_company_name(cik)
        return jsonify(get_institutional_holders({"cik": cik, "name": name},
                                                   shares_out=shares_out, refresh=refresh))
    except Exception as e:
        return jsonify({"error": f"13F fetch failed: {e}", "holders": [],
                        "total_value": 0, "total_count": 0, "rate_limited": True}), 200


@app.route("/api/analyze")
def analyze():
    ticker        = request.args.get("ticker", "").upper().strip()
    discount_rate = float(request.args.get("discount_rate", 0.10))
    tv_raw        = request.args.get("tv")
    tv_dollar     = float(tv_raw) if tv_raw else None   # user passes raw dollars (already converted in JS)
    dcf_horizon   = int(request.args.get("horizon", 10))

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    if discount_rate <= 0 or discount_rate >= 1:
        return jsonify({"error": "Discount rate must be between 0 and 1 (e.g. 0.10)"}), 400
    if tv_dollar is not None and tv_dollar < 0:
        return jsonify({"error": "Terminal value must be a positive dollar amount"}), 400
    if dcf_horizon < 1 or dcf_horizon > 50:
        return jsonify({"error": "Horizon must be between 1 and 50 years"}), 400

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
    # Anchor quarter discovery on the true period-end date of the most recent
    # annual filing.  Using the FY-adjusted financial series dates is wrong for
    # non-December fiscal years (e.g. Salesforce's Jan 31 year-end): the
    # FY-adjustment maps "2026-01-31" → "2025-01-31", making the anchor a full
    # year too early and causing the most-recent quarters to be missed.
    # Priority: (1) report_date from the most recent 10-K filing in submissions,
    #           (2) FY-adjusted series date (existing logic), (3) Dec-31 fallback.
    _last_annual_date = ""
    _lat_y = years[-1] if years else ""

    # (1) True period-end from the most recent 10-K filing
    if all_10k_filings and all_10k_filings[0].get("report_date"):
        _last_annual_date = all_10k_filings[0]["report_date"]

    # (2) FY-adjusted series date — use if it's *newer* than the 10-K report_date
    #     (guards against stale 10-K lists) or if no 10-K date was found
    for _ref_series in (financials.get("revenue", {}), financials.get("net_income", {}),
                        financials.get("equity", {}), financials.get("total_assets", {})):
        _annual_dates = [d for d in _ref_series if not d.startswith("Q") and d[:4] == _lat_y]
        if _annual_dates:
            _candidate = max(_annual_dates)
            if not _last_annual_date or _candidate > _last_annual_date:
                _last_annual_date = _candidate
            break

    if not _last_annual_date and _lat_y:
        # (3) Fall back: construct a Dec-31 anchor from the display year
        _last_annual_date = f"{_lat_y}-12-31"

    # Metrics to extract for quarterly view
    _quarterly_flow_keys = {
        "revenue", "gross_profit", "cost_of_revenue", "operating_income", "net_income",
        "operating_cash_flow", "capex", "depreciation", "stock_based_compensation",
        "income_tax", "interest_expense", "investment_gains",
        "dividends_paid", "buybacks_value", "dividends_per_share",
        # BDC flow metrics
        "net_investment_income", "gross_investment_income", "nii_per_share",
        # Bank flow metrics
        "interest_income", "net_interest_income", "noninterest_income",
        "noninterest_expense", "provision_for_losses",
        # REIT flow metrics
        "gains_on_real_estate", "real_estate_depreciation", "straight_line_rent", "recurring_capex",
        "general_admin_expense", "selling_marketing_expense",
        # Opex lines for the gross-profit add-back fallback (INTU and similar)
        "sga_expense", "rd_expense", "amortization_of_intangibles", "restructuring_charges",
        # Insurance flow metrics
        "premiums_earned", "premiums_written", "losses_incurred", "benefits_losses_expenses",
    }
    _quarterly_bs_keys = {
        "total_assets", "current_assets", "current_liabilities", "equity",
        "cash", "short_term_investments", "long_term_debt", "current_debt",
        "total_liabilities", "goodwill", "intangibles", "inventory", "shares_outstanding_end",
        "treasury_stock", "treasury_stock_shares",
        # BDC point-in-time
        "nav_per_share",
        # REIT point-in-time
        "real_estate_assets",
        # Insurance point-in-time (float components)
        "claims_reserve", "unearned_premiums", "premiums_receivable",
        "deferred_acquisition_costs", "reinsurance_recoverable",
    }
    _point_in_time_metrics = {
        "total_assets", "current_assets", "current_liabilities", "total_liabilities",
        "equity", "cash", "short_term_investments", "long_term_debt", "current_debt",
        "goodwill", "intangibles", "inventory", "shares_outstanding_end", "buyback_remaining",
        "treasury_stock", "treasury_stock_shares",
        # BDC point-in-time
        "nav_per_share",
        # REIT point-in-time
        "real_estate_assets",
        # Insurance point-in-time (float components)
        "claims_reserve", "unearned_premiums", "premiums_receivable",
        "deferred_acquisition_costs", "reinsurance_recoverable",
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

        # Fallback for companies that don't file CommonStockSharesOutstanding or
        # EntityCommonStockSharesOutstanding (e.g. META): use the quarterly
        # WeightedAverageNumberOfSharesOutstandingBasic as a proxy.
        _so_series = financials.setdefault("shares_outstanding_end", {})
        _missing_qks = [qk for qk in quarter_end_dates if not _so_series.get(qk)]
        if _missing_qks:
            _wtd_avg_tag = (
                facts.get("facts", {}).get("us-gaap", {})
                    .get("WeightedAverageNumberOfSharesOutstandingBasic")
            )
            if _wtd_avg_tag:
                _qdate_parsed2 = {
                    qk: datetime.strptime(qd, "%Y-%m-%d")
                    for qk, qd in quarter_end_dates.items()
                    if qk in _missing_qks
                }
                _wtd_best: dict[str, tuple[int, float]] = {}  # qk -> (period_days, val)
                for _e in _wtd_avg_tag.get("units", {}).get("shares", []):
                    if _e.get("form") not in {"10-Q", "10-Q/A"}:
                        continue
                    _end = _e.get("end", "")
                    _start = _e.get("start", "")
                    _val = _e.get("val")
                    if not _end or not _start or not _val:
                        continue
                    try:
                        _end_dt   = datetime.strptime(_end,   "%Y-%m-%d")
                        _start_dt = datetime.strptime(_start, "%Y-%m-%d")
                    except Exception:
                        continue
                    _period_days = (_end_dt - _start_dt).days
                    # Only use quarterly (roughly 3-month) periods, not YTD
                    if not (60 <= _period_days <= 100):
                        continue
                    for _qk, _qdt in _qdate_parsed2.items():
                        if _end_dt == _qdt:   # exact quarter-end match
                            prev2 = _wtd_best.get(_qk)
                            if prev2 is None:
                                _wtd_best[_qk] = (_period_days, _val)
                for _qk, (_, _val) in _wtd_best.items():
                    _so_series[_qk] = _val

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

        # Quarterly OI fallback: GP − (SGA or S&M + G&A) when OI not tagged
        _q_gp_oi  = {k: v for k, v in financials.get("gross_profit",           {}).items() if k.startswith("Q")}
        _q_sga_oi = {k: v for k, v in financials.get("sga_expense",            {}).items() if k.startswith("Q")}
        _q_ga_oi  = {k: v for k, v in financials.get("general_admin_expense",  {}).items() if k.startswith("Q")}
        _q_sm_oi  = {k: v for k, v in financials.get("selling_marketing_expense",{}).items() if k.startswith("Q")}
        _q_oi_existing = {k: v for k, v in financials.get("operating_income",  {}).items() if k.startswith("Q")}
        if _q_gp_oi:
            _oi_fb = financials.setdefault("operating_income", {})
            for qk, gp_v in _q_gp_oi.items():
                if qk in _q_oi_existing:
                    continue
                sga_v = _q_sga_oi.get(qk)
                ga_v  = _q_ga_oi.get(qk)
                sm_v  = _q_sm_oi.get(qk)
                if sga_v is not None:
                    opex = abs(sga_v)
                elif ga_v is not None or sm_v is not None:
                    opex = abs(ga_v or 0) + abs(sm_v or 0)
                else:
                    continue
                _oi_fb[qk] = gp_v - opex

        # Quarterly SGA backfill: S&M + G&A when combined tag absent
        _q_sga_ex = {k: v for k, v in financials.get("sga_expense", {}).items() if k.startswith("Q")}
        _q_sga_fill: dict[str, float] = {}
        for qk in set(_q_sm_oi) | set(_q_ga_oi):
            if qk in _q_sga_ex:
                continue
            sm_v = _q_sm_oi.get(qk) or 0.0
            ga_v = _q_ga_oi.get(qk) or 0.0
            if sm_v or ga_v:
                _q_sga_fill[qk] = abs(sm_v) + abs(ga_v)
        if _q_sga_fill:
            financials.setdefault("sga_expense", {}).update(_q_sga_fill)

        # Quarterly EBITDA = quarterly OI + quarterly D&A
        _q_oi_eb  = {k: v for k, v in financials.get("operating_income", {}).items() if k.startswith("Q")}
        _q_dep_eb = {k: v for k, v in financials.get("depreciation",      {}).items() if k.startswith("Q")}
        if _q_oi_eb and _q_dep_eb:
            _ebitda_q = financials.setdefault("ebitda", {})
            for qk in set(_q_oi_eb) & set(_q_dep_eb):
                _ebitda_q[qk] = _q_oi_eb[qk] + abs(_q_dep_eb[qk])

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
            ("operating_income","ebit_margin"),   # EBIT = operating income
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

        # Quarterly Owner Earnings = Net Income + D&A - CapEx - Investment Gains
        _q_ni_oe   = {k: v for k, v in financials.get("net_income",       {}).items() if k.startswith("Q")}
        _q_dep_oe  = {k: v for k, v in financials.get("depreciation",     {}).items() if k.startswith("Q")}
        _q_cx_oe   = {k: v for k, v in financials.get("capex",            {}).items() if k.startswith("Q")}
        _q_ig_oe   = {k: v for k, v in financials.get("investment_gains", {}).items() if k.startswith("Q")}
        if _q_ni_oe and _q_dep_oe and _q_cx_oe:
            _q_oe = {}
            for qk, nv in _q_ni_oe.items():
                dv = _q_dep_oe.get(qk)
                cv = _q_cx_oe.get(qk)
                if dv is not None and cv is not None:
                    ig = _q_ig_oe.get(qk) or 0.0
                    _q_oe[qk] = nv + abs(dv) - abs(cv) - ig
            if _q_oe:
                financials.setdefault("owner_earnings", {}).update(_q_oe)

        # Quarterly Adj. FCF = FCF - SBC; Adj. FCF Margin = Adj. FCF / Revenue
        _q_fcf = {k: v for k, v in financials.get("fcf", {}).items() if k.startswith("Q")}
        _q_sbc = {k: v for k, v in financials.get("stock_based_compensation", {}).items() if k.startswith("Q")}
        if _q_fcf and _q_sbc:
            _q_adj_fcf = {}
            for qk, fv in _q_fcf.items():
                sv = _q_sbc.get(qk)
                if sv is not None:
                    _q_adj_fcf[qk] = fv - abs(sv)
            if _q_adj_fcf:
                financials.setdefault("adj_fcf", {}).update(_q_adj_fcf)
                for qk, av in _q_adj_fcf.items():
                    dv = rev_q.get(qk)
                    if dv and dv > 0:
                        financials.setdefault("adj_fcf_margin", {})[qk] = av / dv

        # Quarterly Gross Profit = Revenue - COGS (for companies that don't tag GrossProfit)
        _q_gp = {k: v for k, v in financials.get("gross_profit", {}).items() if k.startswith("Q")}
        if not _q_gp:
            _q_rev_gp = {k: v for k, v in financials.get("revenue", {}).items() if k.startswith("Q")}
            _q_cogs   = {k: v for k, v in financials.get("cost_of_revenue", {}).items() if k.startswith("Q")}
            if _q_rev_gp and _q_cogs:
                _q_gp_derived = {}
                for qk in _q_rev_gp:
                    if qk in _q_cogs:
                        _q_gp_derived[qk] = _q_rev_gp[qk] - abs(_q_cogs[qk])
                if _q_gp_derived:
                    financials.setdefault("gross_profit", {}).update(_q_gp_derived)
                    # Margin loop already ran above — fill gross_margin here too
                    for qk, gv in _q_gp_derived.items():
                        rv = rev_q.get(qk)
                        if rv and rv > 0:
                            financials.setdefault("gross_margin", {})[qk] = gv / rv

        # Quarterly GP fallback #2: opex add-back (GP = OI + S&M + R&D + G&A + amort + restructuring)
        _q_gp = {k: v for k, v in financials.get("gross_profit", {}).items() if k.startswith("Q")}
        if not _q_gp:
            _q_oi_gp  = {k: v for k, v in financials.get("operating_income",            {}).items() if k.startswith("Q")}
            _q_rd_gp  = {k: v for k, v in financials.get("rd_expense",                  {}).items() if k.startswith("Q")}
            _q_am_gp  = {k: v for k, v in financials.get("amortization_of_intangibles", {}).items() if k.startswith("Q")}
            _q_rst_gp = {k: v for k, v in financials.get("restructuring_charges",       {}).items() if k.startswith("Q")}
            _q_gp_fb2 = {}
            for qk, o in _q_oi_gp.items():
                sm_v  = _q_sm_oi.get(qk)
                ga_v  = _q_ga_oi.get(qk)
                sga_v = _q_sga_oi.get(qk)
                if sm_v is not None and ga_v is not None:
                    opex = abs(sm_v) + abs(ga_v)
                elif sga_v is not None:
                    opex = abs(sga_v)
                else:
                    continue
                opex += abs(_q_rd_gp.get(qk)  or 0)
                opex += abs(_q_am_gp.get(qk)  or 0)
                opex += abs(_q_rst_gp.get(qk) or 0)
                _q_gp_fb2[qk] = o + opex
            if _q_gp_fb2:
                financials.setdefault("gross_profit", {}).update(_q_gp_fb2)
                for qk, gv in _q_gp_fb2.items():
                    rv = rev_q.get(qk)
                    if rv and rv > 0:
                        financials.setdefault("gross_margin", {})[qk] = gv / rv

        # Quarterly buybacks fallback: delta between Q treasury stock and last year-end balance
        _q_bb = {k: v for k, v in financials.get("buybacks_value", {}).items() if k.startswith("Q")}
        if not _q_bb:
            _q_ts  = {k: v for k, v in financials.get("treasury_stock", {}).items() if k.startswith("Q")}
            _ann_ts = {k: v for k, v in financials.get("treasury_stock", {}).items() if not k.startswith("Q")}
            if _q_ts and _ann_ts:
                _base_ts = _ann_ts[max(_ann_ts)]
                _bb_q_derived = {}
                for qk in sorted(_q_ts):
                    _delta = _q_ts[qk] - _base_ts
                    if _delta > 0:
                        _bb_q_derived[qk] = _delta
                if _bb_q_derived:
                    financials.setdefault("buybacks_value", {}).update(_bb_q_derived)

        # Quarterly shares repurchased fallback: delta in treasury share count
        _q_sh_bb = {k: v for k, v in financials.get("shares_repurchased", {}).items() if k.startswith("Q")}
        if not _q_sh_bb:
            _q_ts_sh  = {k: v for k, v in financials.get("treasury_stock_shares", {}).items() if k.startswith("Q")}
            _ann_ts_sh = {k: v for k, v in financials.get("treasury_stock_shares", {}).items() if not k.startswith("Q")}
            if _q_ts_sh and _ann_ts_sh:
                _base_ts_sh = _ann_ts_sh[max(_ann_ts_sh)]
                _sh_bb_q_derived = {}
                for qk in sorted(_q_ts_sh):
                    _delta = _q_ts_sh[qk] - _base_ts_sh
                    if _delta > 0:
                        _sh_bb_q_derived[qk] = _delta
                if _sh_bb_q_derived:
                    financials.setdefault("shares_repurchased", {}).update(_sh_bb_q_derived)

        # Quarterly EBIT = quarterly operating income
        _q_oi = {k: v for k, v in financials.get("operating_income", {}).items() if k.startswith("Q")}
        if _q_oi:
            financials.setdefault("ebit", {}).update(_q_oi)

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
        _q_adj_fcf_r = {k: v for k, v in financials.get("adj_fcf", {}).items() if k.startswith("Q")}
        if _q_adj_fcf_r and _q_eq:
            _adjfcfroe_q = financials.setdefault("adj_fcf_roe", {})
            for qk in set(_q_adj_fcf_r) & set(_q_eq):
                if _q_eq[qk] and _q_eq[qk] != 0:
                    _adjfcfroe_q[qk] = (_q_adj_fcf_r[qk] * 4) / _q_eq[qk]
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

        # Quarterly FFO / AFFO (REIT)
        _q_ni_reit  = {k: v for k, v in financials.get("net_income",            {}).items() if k.startswith("Q")}
        _q_re_dep   = {k: v for k, v in financials.get("real_estate_depreciation",{}).items() if k.startswith("Q")}
        # Fall back to general D&A only when REIT-specific data (RE assets or SL rent) exists
        _has_reit_q = bool(
            {k for k in financials.get("real_estate_assets", {}) if k.startswith("Q")} or
            {k for k in financials.get("straight_line_rent", {}) if k.startswith("Q")} or
            {k for k in financials.get("real_estate_assets", {}) if not k.startswith("Q")}
        )
        if not _q_re_dep and _has_reit_q:
            _q_re_dep = {k: v for k, v in financials.get("depreciation", {}).items() if k.startswith("Q")}
        _q_re_gains = {k: v for k, v in financials.get("gains_on_real_estate",  {}).items() if k.startswith("Q")}
        _q_slr      = {k: v for k, v in financials.get("straight_line_rent",    {}).items() if k.startswith("Q")}
        _q_rec_cx   = {k: v for k, v in financials.get("recurring_capex",       {}).items() if k.startswith("Q")}
        if _q_ni_reit and _q_re_dep:
            _q_ffo = {}
            for qk in set(_q_ni_reit) & set(_q_re_dep):
                gains_v = _q_re_gains.get(qk) or 0.0
                _q_ffo[qk] = _q_ni_reit[qk] + abs(_q_re_dep[qk]) - gains_v
            if _q_ffo:
                financials.setdefault("ffo", {}).update(_q_ffo)
                # AFFO quarterly
                _q_affo = {}
                for qk, fv in _q_ffo.items():
                    sl_v = _q_slr.get(qk) or 0.0
                    rc_v = _q_rec_cx.get(qk) or 0.0
                    _q_affo[qk] = fv - abs(sl_v) - abs(rc_v)
                if _q_affo:
                    financials.setdefault("affo", {}).update(_q_affo)
                # FFO payout ratio quarterly (annualised FFO as denominator)
                _q_divs_reit = {k: v for k, v in financials.get("dividends_paid", {}).items() if k.startswith("Q")}
                if _q_divs_reit:
                    _ffo_payout_q = financials.setdefault("ffo_payout_ratio", {})
                    for qk, fv in _q_ffo.items():
                        dv = _q_divs_reit.get(qk)
                        ann_ffo = fv * 4
                        if dv is not None and ann_ffo and ann_ffo > 0:
                            _ffo_payout_q[qk] = abs(dv) * 4 / ann_ffo

        # Quarterly NOI = quarterly EBITDA + G&A
        _q_ebitda = {k: v for k, v in financials.get("ebitda", {}).items() if k.startswith("Q")}
        _q_ga     = {k: v for k, v in financials.get("general_admin_expense", {}).items() if k.startswith("Q")}
        if _has_reit_q and _q_ebitda:
            _q_noi = {}
            for qk, eb in _q_ebitda.items():
                if eb is None:
                    continue
                ga_v = _q_ga.get(qk) or 0.0
                _q_noi[qk] = eb + abs(ga_v)
            if _q_noi:
                financials.setdefault("noi", {}).update(_q_noi)
                for qk, nv in _q_noi.items():
                    rv = rev_q.get(qk)
                    if rv and rv > 0:
                        financials.setdefault("noi_margin", {})[qk] = nv / rv

        # Quarterly per-share FFO / AFFO
        _q_sb_reit = (
            {k: v for k, v in financials.get("shares_diluted_wtd", {}).items() if k.startswith("Q")}
            or {k: v for k, v in financials.get("shares_outstanding_end", {}).items() if k.startswith("Q")}
        )
        if _q_sb_reit:
            _q_ffo_vals  = {k: v for k, v in financials.get("ffo",  {}).items() if k.startswith("Q")}
            _q_affo_vals = {k: v for k, v in financials.get("affo", {}).items() if k.startswith("Q")}
            if _q_ffo_vals:
                _ffo_ps_q = financials.setdefault("ffo_per_share", {})
                for qk in set(_q_ffo_vals) & set(_q_sb_reit):
                    if _q_sb_reit[qk]:
                        _ffo_ps_q[qk] = _q_ffo_vals[qk] / _q_sb_reit[qk]
            if _q_affo_vals:
                _affo_ps_q = financials.setdefault("affo_per_share", {})
                for qk in set(_q_affo_vals) & set(_q_sb_reit):
                    if _q_sb_reit[qk]:
                        _affo_ps_q[qk] = _q_affo_vals[qk] / _q_sb_reit[qk]
            _q_noi_vals = {k: v for k, v in financials.get("noi", {}).items() if k.startswith("Q")}
            if _q_noi_vals:
                _noi_ps_q = financials.setdefault("noi_per_share", {})
                for qk in set(_q_noi_vals) & set(_q_sb_reit):
                    if _q_sb_reit[qk]:
                        _noi_ps_q[qk] = _q_noi_vals[qk] / _q_sb_reit[qk]

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

        # Quarterly ROIC and Pre-tax ROIC
        _q_oi  = {k: v for k, v in financials.get("operating_income", {}).items() if k.startswith("Q")}
        _q_td2 = {k: v for k, v in financials.get("total_debt", {}).items() if k.startswith("Q")}
        _q_eq3 = {k: v for k, v in financials.get("equity", {}).items() if k.startswith("Q")}
        if _q_oi and (_q_td2 or _q_eq3):
            _roic_q        = financials.setdefault("roic",        {})
            _pretax_roic_q = financials.setdefault("pretax_roic", {})
            for qk in set(_q_oi) & (set(_q_td2) | set(_q_eq3)):
                ic = (_q_td2.get(qk) or 0) + (_q_eq3.get(qk) or 0)
                if ic and ic != 0:
                    _roic_q[qk]        = (_q_oi[qk] * 4 * (1 - 0.21)) / ic
                    _pretax_roic_q[qk] = (_q_oi[qk] * 4) / ic

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
                    if _unta_vals[qk] and _unta_vals[qk] > 0:
                        _eco_q[qk] = _nopat_q[qk] / _unta_vals[qk]

                # Quarterly Pre-tax Economic Goodwill = EBIT / UNTA (annualised)
                _pretax_eco_q = financials.setdefault("pretax_economic_goodwill", {})
                for qk in set(_q_oi) & set(_unta_vals):
                    if _unta_vals[qk] and _unta_vals[qk] > 0:
                        _pretax_eco_q[qk] = (_q_oi[qk] * 4) / _unta_vals[qk]

        # Quarterly insurance ratios + float
        _q_pe   = {k: v for k, v in financials.get("premiums_earned",          {}).items() if k.startswith("Q")}
        _q_loss = {k: v for k, v in financials.get("losses_incurred",          {}).items() if k.startswith("Q")}
        _q_ble  = {k: v for k, v in financials.get("benefits_losses_expenses", {}).items() if k.startswith("Q")}
        if _q_pe:
            if _q_loss:
                _lr_q = financials.setdefault("loss_ratio", {})
                for qk in set(_q_pe) & set(_q_loss):
                    if _q_pe[qk] and _q_pe[qk] > 0:
                        _lr_q[qk] = abs(_q_loss[qk]) / _q_pe[qk]
            if _q_ble:
                _cr_q = financials.setdefault("combined_ratio", {})
                for qk in set(_q_pe) & set(_q_ble):
                    if _q_pe[qk] and _q_pe[qk] > 0:
                        _cr_q[qk] = abs(_q_ble[qk]) / _q_pe[qk]
                _exp_q = financials.setdefault("expense_ratio", {})
                for qk in _cr_q:
                    lr = financials.get("loss_ratio", {}).get(qk)
                    if lr is not None:
                        _exp_q[qk] = _cr_q[qk] - lr
        _q_claims = {k: v for k, v in financials.get("claims_reserve", {}).items() if k.startswith("Q")}
        if _q_claims:
            _q_un  = {k: v for k, v in financials.get("unearned_premiums",          {}).items() if k.startswith("Q")}
            _q_pr  = {k: v for k, v in financials.get("premiums_receivable",        {}).items() if k.startswith("Q")}
            _q_dac = {k: v for k, v in financials.get("deferred_acquisition_costs", {}).items() if k.startswith("Q")}
            _q_re  = {k: v for k, v in financials.get("reinsurance_recoverable",    {}).items() if k.startswith("Q")}
            _flo_q = financials.setdefault("insurance_float", {})
            for qk in _q_claims:
                _flo_q[qk] = (_q_claims[qk] + (_q_un.get(qk) or 0) - (_q_pr.get(qk) or 0)
                              - (_q_dac.get(qk) or 0) - (_q_re.get(qk) or 0))
            if _q_sb:
                _fps_q = financials.setdefault("float_per_share", {})
                for qk in set(_flo_q) & set(_q_sb):
                    if _q_sb[qk]:
                        _fps_q[qk] = _flo_q[qk] / _q_sb[qk]

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

    # Short interest (FINRA, Nasdaq backup — bi-monthly settlement) as % of
    # shares outstanding (EDGAR).
    short_data = get_short_interest(ticker)
    short_pct_shares_out = None
    if short_data.get("shares_short") and edgar_shares:
        short_pct_shares_out = short_data["shares_short"] / edgar_shares

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
    current_ffo  = L("ffo")
    current_affo = L("affo")
    multiples = {
        "pe":           mult(mktcap, current_ni),
        "p_fcf":        mult(mktcap, current_fcf),
        "p_s":          mult(mktcap, current_rev),
        "p_b":          mult(mktcap, current_eq),
        "ev_ebitda":    mult(ev, current_ebitda),
        "ev_ebit":      mult(ev, L("ebit")),
        "nopat_yield":  round(L("nopat") / ev, 4) if L("nopat") and ev else None,
        "earnings_yield": round(current_ni / mktcap, 4) if current_ni and mktcap else None,
        "fcf_yield":    round(current_fcf / mktcap, 4) if current_fcf and mktcap else None,
        # Cash Yield = (FCF + Net Interest) / EV
        # Adds net interest expense back to FCF so the numerator is capital-structure
        # neutral, matching EV in the denominator. Net interest = interest expense −
        # interest income (interest income is only present for some filers).
        "cash_yield": (
            round((current_fcf + (L("interest_expense") or 0) - (L("interest_income") or 0)) / ev, 4)
            if current_fcf and ev else None
        ),
        # REIT-specific multiples
        "p_ffo":        mult(mktcap, current_ffo),
        "p_affo":       mult(mktcap, current_affo),
        "ev_ffo":       mult(ev, current_ffo),
        "ffo_yield":    round(current_ffo / mktcap, 4) if current_ffo and mktcap else None,
        # Cap Rate = NOI / EV (market-implied yield on the real estate portfolio)
        "cap_rate":     round(L("noi") / ev, 4) if L("noi") and ev else None,
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

    # ── Reverse DCF scenario (single, user-adjustable horizon) ────────────────
    dcf_scenarios = {}
    if mktcap:
        res = reverse_dcf(mktcap, current_fcf, discount_rate, tv_dollar, dcf_horizon)
        if res:
            res["hist_avg_fcf"] = hist_avg_fcf(dcf_horizon)
            dcf_scenarios[str(dcf_horizon)] = res

    # ── Fiscal year end month ────────────────────────────────────────────────
    fy_month = "Dec"
    rev_series = financials.get("revenue", {})
    annual_rev_dates = [d for d in rev_series if not d.startswith("Q")]
    if annual_rev_dates:
        latest_rev_date = max(annual_rev_dates)
        m_num = int(latest_rev_date[5:7])
        fy_month = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"][m_num - 1]

    _sic     = str(submissions.get("sic", "") or "")
    _sic_int = int(_sic) if _sic.isdigit() else 0

    # Insurance detection: SIC 6300-6399 (insurance carriers) or presence of
    # net premiums earned in XBRL.
    is_insurance = ((6300 <= _sic_int <= 6411) or
                    bool(financials.get("premiums_earned")))
    # BDC detection: NetInvestmentIncome present — but insurers also file that
    # tag for their investment portfolios, so exclude anything flagged insurance.
    is_bdc  = bool(financials.get("net_investment_income")) and not is_insurance
    # Bank detection: SIC 6000-6199 (depository institutions, credit agencies) OR
    # presence of NoninterestExpense (a tag essentially exclusive to bank filers).
    # We intentionally do NOT use net_interest_income alone — many non-banks (e.g.
    # homebuilders with mortgage subsidiaries) file InterestIncomeExpenseNet, which
    # would trigger false positives.
    is_bank  = (
        (6000 <= _sic_int <= 6199) or
        bool(financials.get("noninterest_expense"))
    )
    # REIT detection: SIC 6798 (Real Estate Investment Trusts) or SIC 6500-6552,
    # or presence of real estate asset / straight-line rent XBRL data.
    # Note: we do NOT use ffo presence to detect REITs because FFO is derived
    # and would produce false positives for non-REITs with D&A and NI.
    # REIT detection: a real-estate SIC is definitive. Otherwise require the
    # REIT-specific straight-line rent accrual (a lessor concept) — NOT merely
    # holding real estate, since operating companies across sectors own property
    # (e.g. FANG, an oil & gas E&P, tags RealEstateInvestmentPropertyNet for
    # surface land) and insurers hold it as an investment (e.g. MET).
    is_reit = (
        _sic in {"6798", "6500", "6512", "6552"} or
        (not is_insurance and not is_bank and not is_bdc and
         bool(financials.get("straight_line_rent")))
    )

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

    # Insider open-market purchases (Form 4) are fetched separately by the
    # frontend via /api/insider — those Form 4 XML fetches are slow and would
    # otherwise inflate this response's latency.

    # ── Recent SEC filings (last 10, all form types) ──────────────────────────
    _recent_f   = submissions.get("filings", {}).get("recent", {})
    _f_forms    = _recent_f.get("form", [])
    _f_accns    = _recent_f.get("accessionNumber", [])
    _f_docs     = _recent_f.get("primaryDocument", [])
    _f_dates    = _recent_f.get("filingDate", [])
    _f_reports  = _recent_f.get("reportDate", [])
    _f_sizes    = _recent_f.get("size", [])
    _cik_nz     = str(int(submissions.get("cik", "0")))
    recent_filings: list[dict] = []
    for _i, _form in enumerate(_f_forms):
        if len(recent_filings) >= 10:
            break
        _accn = _f_accns[_i] if _i < len(_f_accns) else ""
        _doc  = _f_docs[_i]  if _i < len(_f_docs)  else ""
        recent_filings.append({
            "form":        _form,
            "filing_date": _f_dates[_i]   if _i < len(_f_dates)   else "",
            "report_date": _f_reports[_i] if _i < len(_f_reports) else "",
            "size_kb":     round(_f_sizes[_i] / 1024) if _i < len(_f_sizes) and _f_sizes[_i] else None,
            "url": SEC_ARCHIVES.format(
                cik_no_zero=_cik_nz,
                accession_no_dash=_accn.replace("-", ""),
                document=_doc,
            ) if _accn and _doc else "",
            "index_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=include&count=40",
        })
    all_filings_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=&dateb=&owner=include&count=40"
    )

    return jsonify({
        "company": {
            "name":             company_name,
            "ticker":           ticker,
            "cik":              cik,
            "sic":              submissions.get("sic", ""),
            "sic_description":  submissions.get("sicDescription", ""),
            "is_bdc":           is_bdc,
            "is_bank":          is_bank,
            "is_reit":          is_reit,
            "is_insurance":     is_insurance,
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
            "shares_short":        short_data.get("shares_short"),
            "short_pct_shares_out": short_pct_shares_out,
            "short_settlement_date": short_data.get("settlement_date"),
            "days_to_cover":       short_data.get("days_to_cover"),
            **multiples,
        },
        "years":           years,
        "quarters":        quarters,
        "quarter_dates":   quarter_end_dates,
        "quarter_links":      quarter_filing_links,
        "earnings_materials": earnings_materials,
        "recent_filings":     recent_filings,
        "all_filings_url":    all_filings_url,
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
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)), host="0.0.0.0")
