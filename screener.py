#!/usr/bin/env python3
"""
Valuation screener for stock universes (S&P 500, NASDAQ-100, Dow 30, custom).

Strategy — keep it to a handful of HTTP calls regardless of universe size:
  • SEC "frames" API returns ONE financial concept for *all* filers in a single
    response (e.g. every company's Operating Income for CY2024). We fetch ~8
    frames total instead of one multi-MB companyfacts file per ticker.
  • Yahoo v8 chart gives the current price per ticker; these are threaded and
    cached, so a 500-name screen is fast on repeat runs.

Per company we compute:
  market_cap = price × shares (cover-page shares from SEC dei frame)
  FCF        = Operating Cash Flow − |CapEx|
  EBIT       = Operating Income
  EV         = market_cap + total_debt − cash
  P/FCF      = market_cap / FCF
  EV/EBIT    = EV / EBIT
Then filter by the user's cutoffs and rank cheapest-first by P/FCF.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
from bs4 import BeautifulSoup

SEC_BASE = "https://data.sec.gov"
H_SEC = {"User-Agent": "InvestmentAnalysis research@example.com", "Accept": "application/json"}
H_YH  = {"User-Agent": "Mozilla/5.0"}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".screen_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ─── Disk cache ───────────────────────────────────────────────────────────────

def _cached(name: str, ttl_seconds: int, fetch_fn):
    """Return cached JSON if fresh, else call fetch_fn(), cache, and return it."""
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    data = fetch_fn()
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return data


# ─── Universe constituent lists ───────────────────────────────────────────────

_WIKI = {
    "sp500":     ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "constituents"),
    "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "constituents"),
    "dow30":     ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "constituents"),
}

UNIVERSE_LABELS = {
    "sp500":     "S&P 500",
    "nasdaq100": "NASDAQ-100",
    "dow30":     "Dow 30",
}


def _scrape_wiki_tickers(url: str, table_id: str) -> list[str]:
    r = requests.get(url, headers=H_YH, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": table_id})
    if not table:
        # Fall back to the first wikitable on the page
        table = soup.find("table", {"class": "wikitable"})
    tickers: list[str] = []
    if not table:
        return tickers
    # Header cells come from the first row; the symbol column is identified by name.
    head_row = table.find("tr")
    header = [c.get_text(strip=True).lower() for c in head_row.find_all(["th", "td"])]
    sym_col = 0
    for i, h in enumerate(header):
        if any(k in h for k in ("symbol", "ticker")):
            sym_col = i
            break
    for row in table.find_all("tr")[1:]:
        # Count th + td together so row-header company-name cells don't shift indices.
        cells = row.find_all(["th", "td"])
        if len(cells) <= sym_col:
            continue
        sym = cells[sym_col].get_text(strip=True).upper()
        sym = sym.replace("​", "").split()[0] if sym else sym
        sym = sym.split("[")[0]   # drop footnote markers like AAPL[1]
        if sym and len(sym) <= 6 and sym.replace(".", "").replace("-", "").isalnum():
            tickers.append(sym)
    return tickers


def get_universe(name: str) -> list[str]:
    """Return the ticker list for a named universe (cached 24h)."""
    key = name.lower().strip()
    if key not in _WIKI:
        return []
    url, table_id = _WIKI[key]
    return _cached(f"universe_{key}.json", 86400,
                   lambda: _scrape_wiki_tickers(url, table_id))


# ─── Ticker → CIK map ─────────────────────────────────────────────────────────

def ticker_cik_map() -> dict[str, int]:
    """Map every SEC ticker to its integer CIK (cached 24h)."""
    def fetch():
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=H_SEC, timeout=20)
        r.raise_for_status()
        return {e["ticker"].upper(): int(e["cik_str"]) for e in r.json().values()}
    return _cached("ticker_cik_map.json", 86400, fetch)


# ─── SEC frames ───────────────────────────────────────────────────────────────

def _frame(tag: str, unit: str, period: str) -> dict[str, float]:
    """Return {cik(str): val} for one concept/period. Empty on any failure."""
    def fetch():
        url = f"{SEC_BASE}/api/xbrl/frames/us-gaap/{tag}/{unit}/{period}.json"
        r = requests.get(url, headers=H_SEC, timeout=30)
        if r.status_code != 200:
            return {}
        return {str(d["cik"]): d["val"] for d in r.json().get("data", [])}
    # dei concepts live under a different namespace
    ns = "dei" if tag.startswith("Entity") else "us-gaap"
    def fetch_ns():
        url = f"{SEC_BASE}/api/xbrl/frames/{ns}/{tag}/{unit}/{period}.json"
        r = requests.get(url, headers=H_SEC, timeout=30)
        if r.status_code != 200:
            return {}
        return {str(d["cik"]): d["val"] for d in r.json().get("data", [])}
    return _cached(f"frame_{ns}_{tag}_{unit}_{period}.json", 43200, fetch_ns)


def _merge_frames(tags: list[str], unit: str, periods: list[str]) -> dict[str, float]:
    """
    Merge several tag/period candidates into one {cik: val} map.
    First hit per CIK wins, so list tags by priority and periods newest-first.
    """
    out: dict[str, float] = {}
    for period in periods:
        for tag in tags:
            frame = _frame(tag, unit, period)
            for cik, val in frame.items():
                if cik not in out:
                    out[cik] = val
    return out


# ─── Prices (threaded Yahoo v8) ───────────────────────────────────────────────

def _yahoo_price(ticker: str) -> Optional[float]:
    sym = ticker.replace(".", "-")
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": "1d"},
                             headers=H_YH, timeout=8)
            if r.status_code != 200:
                continue
            meta = r.json()["chart"]["result"][0]["meta"]
            return meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        except Exception:
            continue
    return None


def get_prices(tickers: list[str]) -> dict[str, float]:
    """Threaded price fetch with a 1-hour disk cache keyed by the universe set."""
    cache_key = "prices_" + str(abs(hash(",".join(sorted(tickers))))) + ".json"
    path = os.path.join(CACHE_DIR, cache_key)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < 3600:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for tk, px in zip(tickers, ex.map(_yahoo_price, tickers)):
            if px:
                out[tk] = px
    try:
        with open(path, "w") as f:
            json.dump(out, f)
    except Exception:
        pass
    return out


# ─── Screen ───────────────────────────────────────────────────────────────────

# Recent annual (duration) and instantaneous (balance-sheet) periods to try,
# newest first. Updated as fiscal years roll forward.
def _recent_periods(latest_fy: int):
    annual   = [f"CY{latest_fy}", f"CY{latest_fy - 1}"]
    instants = []
    for y in (latest_fy, latest_fy - 1):
        for q in ("Q4I", "Q3I", "Q2I", "Q1I"):
            instants.append(f"CY{y}{q}")
    return annual, instants


def screen(universe: str, tickers: list[str],
           max_pfcf, max_ev_ebit,
           latest_fy: int = 2025) -> dict:
    """
    Run the valuation screen. Returns {results: [...], stats: {...}}.
    Either cutoff may be None (no upper bound on that ratio).
    """
    annual, instants = _recent_periods(latest_fy)

    ocf  = _merge_frames(["NetCashProvidedByUsedInOperatingActivities",
                          "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
                         "USD", annual)
    capex = _merge_frames(["PaymentsToAcquirePropertyPlantAndEquipment",
                           "PaymentsToAcquireProductiveAssets"], "USD", annual)
    ebit = _merge_frames(["OperatingIncomeLoss"], "USD", annual)
    cash = _merge_frames(["CashAndCashEquivalentsAtCarryingValue"], "USD", instants)
    ltd  = _merge_frames(["LongTermDebtNoncurrent", "LongTermDebt"], "USD", instants)
    std  = _merge_frames(["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"], "USD", instants)
    shares = _merge_frames(["EntityCommonStockSharesOutstanding"], "shares", instants)

    cik_map = ticker_cik_map()
    prices = get_prices(tickers)

    results = []
    missing = 0
    for tk in tickers:
        cik = cik_map.get(tk.upper())
        if cik is None:
            cik = cik_map.get(tk.replace("-", ".").upper()) or cik_map.get(tk.replace(".", "-").upper())
        if cik is None:
            missing += 1
            continue
        c = str(cik)
        price = prices.get(tk)
        sh    = shares.get(c)
        o     = ocf.get(c)
        cx    = capex.get(c)
        eb    = ebit.get(c)
        if not (price and sh and o is not None):
            missing += 1
            continue
        mktcap = price * sh
        fcf    = o - abs(cx or 0)
        ev     = mktcap + (ltd.get(c) or 0) + (std.get(c) or 0) - (cash.get(c) or 0)

        p_fcf   = round(mktcap / fcf, 2) if fcf and fcf > 0 else None
        ev_ebit = round(ev / eb, 2) if eb and eb > 0 else None
        fcf_yld = round(fcf / mktcap, 4) if mktcap else None

        results.append({
            "ticker":   tk,
            "p_fcf":    p_fcf,
            "ev_ebit":  ev_ebit,
            "fcf_yield": fcf_yld,
            "market_cap": round(mktcap),
            "price":    round(price, 2),
        })

    # Apply cutoffs: a name passes only if it has the ratio AND it's within bound.
    def passes(r):
        if max_pfcf is not None:
            if r["p_fcf"] is None or r["p_fcf"] > max_pfcf:
                return False
        if max_ev_ebit is not None:
            if r["ev_ebit"] is None or r["ev_ebit"] > max_ev_ebit:
                return False
        # With no cutoffs, still require at least one valuation metric
        if max_pfcf is None and max_ev_ebit is None:
            return r["p_fcf"] is not None or r["ev_ebit"] is not None
        return True

    filtered = [r for r in results if passes(r)]
    # Rank cheapest-first: primary P/FCF, then EV/EBIT. None sorts last.
    filtered.sort(key=lambda r: (
        r["p_fcf"] if r["p_fcf"] is not None else 1e9,
        r["ev_ebit"] if r["ev_ebit"] is not None else 1e9,
    ))

    return {
        "results": filtered,
        "stats": {
            "universe":   universe,
            "total":      len(tickers),
            "evaluated":  len(results),
            "passed":     len(filtered),
            "missing":    missing,
            "fiscal_year": latest_fy,
        },
    }
