#!/usr/bin/env python3
"""
Intrinsic value calculator web app.

This app:
- pulls structured financial data from SEC EDGAR company facts
- pulls the latest annual filing text from the SEC
- extracts likely business-specific revenue / profitability drivers from MD&A
- performs a reverse DCF to show the FCF path implied by the current market cap
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from html import unescape
from typing import Optional

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

app = Flask(__name__)


HEADERS = {
    "User-Agent": "InvestmentAnalysis research@example.com",
    "Accept": "application/json, text/html",
}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dash}/{document}"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_SUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
YAHOO_QUOTE_PAGE = "https://finance.yahoo.com/quote/{ticker}/"


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "them",
    "there", "these", "they", "this", "to", "was", "we", "were", "which", "with", "within",
    "year", "years", "quarter", "quarters", "fiscal", "company", "business", "results",
    "increase", "increased", "decrease", "decreased", "change", "changes", "impact", "impacted",
    "driven", "primarily", "mainly", "related", "including", "include", "basis", "million",
    "billions", "billion", "percent", "compared", "prior", "current", "net", "total",
}

DRIVER_SIGNAL_WORDS = {
    "revenue", "sales", "margin", "profitability", "profit", "cash", "flow", "pricing", "price",
    "volume", "demand", "utilization", "occupancy", "retention", "churn", "backlog", "orders",
    "advertising", "subscriber", "subscription", "premium", "claims", "deposit", "loan", "aum",
    "assets", "spread", "yield", "traffic", "same-store", "same store", "comparable", "bookings",
    "capacity", "production", "shipments", "inventory", "royalty", "take rate", "engagement",
    "merchant", "seller", "buyer", "member", "arpu", "gpu", "compute", "seat", "license",
    "utilisation", "pipeline", "renewal", "mix", "gross", "opex", "cost", "expense", "capex",
}

METRIC_TAGS = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "NetSales",
    ],
    "Net Income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "Operating Income": ["OperatingIncomeLoss"],
    "Gross Profit": ["GrossProfit"],
    "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "CapEx": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "R&D Expense": ["ResearchAndDevelopmentExpense"],
    "SGA Expense": ["SellingGeneralAndAdministrativeExpense"],
    "Depreciation & Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "Shares Outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "Cash & Equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "Short-term Investments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
    ],
    "Long-term Debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "Total Debt": [
        "DebtAndFinanceLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        "ShortAndLongTermDebt",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "Shareholders Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}


def sec_get_json(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def sec_get_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def resolve_cik(ticker: str) -> Optional[str]:
    ticker = ticker.upper().strip()
    candidates = [
        ticker,
        ticker.replace(".", "-"),
        ticker.replace(".", ""),
        ticker.replace("-", "."),
        ticker.replace("-", ""),
    ]
    data = sec_get_json(SEC_TICKERS_URL)
    for entry in data.values():
        entry_ticker = entry["ticker"].upper()
        if entry_ticker in candidates:
            return str(entry["cik_str"]).zfill(10)
    return None


def fetch_submissions(cik: str) -> dict:
    return sec_get_json(EDGAR_SUBMISSIONS.format(cik=cik))


def fetch_company_facts(cik: str) -> dict:
    return sec_get_json(EDGAR_FACTS.format(cik=cik))


def market_ticker_candidates(ticker: str) -> list[str]:
    normalized = ticker.upper().strip()
    candidates = [
        normalized,
        normalized.replace(".", "-"),
        normalized.replace(".", ""),
        normalized.replace("-", "."),
        normalized.replace("-", ""),
    ]
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def get_market_data(ticker: str) -> dict:
    data: dict[str, float] = {}

    for candidate in market_ticker_candidates(ticker):
        try:
            chart = requests.get(
                YAHOO_CHART.format(ticker=candidate),
                params={"interval": "1d", "range": "5d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            chart.raise_for_status()
            result = chart.json().get("chart", {}).get("result")
            if result:
                meta = result[0]["meta"]
                data["price"] = meta.get("regularMarketPrice") or meta.get("previousClose")
                break
        except Exception:
            continue

    for candidate in market_ticker_candidates(ticker):
        try:
            summary = requests.get(
                YAHOO_SUMMARY.format(ticker=candidate),
                params={"modules": "price,defaultKeyStatistics"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            summary.raise_for_status()
            result = summary.json().get("quoteSummary", {}).get("result")
            if not result:
                continue
            payload = result[0]
            price_block = payload.get("price", {})
            stats_block = payload.get("defaultKeyStatistics", {})
            data["market_cap"] = (
                price_block.get("marketCap", {}).get("raw")
                or stats_block.get("marketCap", {}).get("raw")
            )
            data["shares_outstanding"] = (
                price_block.get("sharesOutstanding", {}).get("raw")
                or stats_block.get("sharesOutstanding", {}).get("raw")
            )
            if "price" not in data:
                data["price"] = price_block.get("regularMarketPrice", {}).get("raw")
            if data.get("market_cap") or data.get("price"):
                break
        except Exception:
            continue

    if not data.get("market_cap"):
        for candidate in market_ticker_candidates(ticker):
            try:
                quote_page = requests.get(
                    YAHOO_QUOTE_PAGE.format(ticker=candidate),
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15,
                )
                quote_page.raise_for_status()
                page_text = re.sub(r"\s+", " ", quote_page.text)
                market_cap_match = re.search(
                    r"Market Cap(?: \(intraday\))?.{0,80}?([0-9]+(?:\.[0-9]+)?[TMB])",
                    page_text,
                    re.IGNORECASE,
                )
                if market_cap_match:
                    data["market_cap"] = parse_suffix_number(market_cap_match.group(1))
                if not data.get("price"):
                    price_match = re.search(
                        r'"regularMarketPrice"\s*:\s*\{"raw"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                        quote_page.text,
                    )
                    if price_match:
                        data["price"] = float(price_match.group(1))
                if data.get("market_cap") or data.get("price"):
                    break
            except Exception:
                continue

    return data


def _period_months(start: str, end: str) -> Optional[int]:
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        return None
    return round((end_dt - start_dt).days / 30.44)


def extract_annual_values(facts: dict, tags: list[str]) -> dict[str, float]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})

    for tag in tags:
        concept = gaap.get(tag) or dei.get(tag)
        if not concept:
            continue

        for unit_key in ("USD", "shares", "USD/shares", "pure"):
            entries = concept.get("units", {}).get(unit_key, [])
            if not entries:
                continue

            annual: dict[str, float] = {}
            for entry in entries:
                if entry.get("form") not in {"10-K", "10-K/A", "20-F", "20-F/A"}:
                    continue
                if entry.get("fp") != "FY":
                    continue

                end = entry.get("end")
                val = entry.get("val")
                if not end or val is None:
                    continue

                start = entry.get("start")
                if start:
                    months = _period_months(start, end)
                    if months is not None and not (10 <= months <= 14):
                        continue

                if end not in annual or abs(val) > abs(annual[end]):
                    annual[end] = val

            if annual:
                return annual

    return {}


def build_financials(facts: dict) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric, tags in METRIC_TAGS.items():
        values = extract_annual_values(facts, tags)
        if values:
            result[metric] = dict(sorted(values.items()))

    ocf = result.get("Operating Cash Flow", {})
    capex = result.get("CapEx", {})
    if ocf:
        fcf: dict[str, float] = {}
        for period, ocf_value in ocf.items():
            fcf[period] = ocf_value - capex.get(period, 0.0)
        result["Free Cash Flow"] = fcf

    return result


def latest_metric(financials: dict[str, dict[str, float]], metric: str) -> Optional[float]:
    values = financials.get(metric, {})
    if not values:
        return None
    latest_period = sorted(values)[-1]
    return values[latest_period]


def years_available(financials: dict[str, dict[str, float]]) -> list[str]:
    periods: set[str] = set()
    for values in financials.values():
        periods.update(values.keys())
    return sorted(periods)


def compute_historical_summary(financials: dict[str, dict[str, float]], periods: list[str]) -> list[dict]:
    rows = []
    for metric in (
        "Revenue",
        "Operating Income",
        "Net Income",
        "Operating Cash Flow",
        "CapEx",
        "Free Cash Flow",
        "Shareholders Equity",
        "Shares Outstanding",
    ):
        data = financials.get(metric, {})
        if not data:
            continue
        row = {"metric": metric, "values": []}
        for period in periods:
            row["values"].append(data.get(period))
        rows.append(row)
    return rows


def format_large_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    amount = abs(value)
    if amount >= 1_000_000_000_000:
        return f"{sign}${amount / 1_000_000_000_000:.2f}T"
    if amount >= 1_000_000_000:
        return f"{sign}${amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.0f}K"
    return f"{sign}${amount:.0f}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_shares(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    amount = abs(value)
    sign = "-" if value < 0 else ""
    if amount >= 1_000_000_000:
        return f"{sign}{amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"{sign}{amount / 1_000_000:.1f}M"
    return f"{sign}{amount:,.0f}"


def annualized_growth(start_value: float, end_value: float, years: int) -> Optional[float]:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def reverse_dcf_growth(
    current_fcf: float,
    market_cap: float,
    discount_rate: float,
    years: int,
    terminal_growth: float = 0.03,
) -> Optional[float]:
    if current_fcf <= 0 or discount_rate <= terminal_growth:
        return None

    def value_for_growth(growth: float) -> float:
        present_value = 0.0
        fcf = current_fcf
        for year in range(1, years + 1):
            fcf *= 1 + growth
            present_value += fcf / ((1 + discount_rate) ** year)
        terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        present_value += terminal_value / ((1 + discount_rate) ** years)
        return present_value

    low, high = -0.75, 2.5
    for _ in range(220):
        mid = (low + high) / 2
        value = value_for_growth(mid)
        if value < market_cap:
            low = mid
        else:
            high = mid

    result = (low + high) / 2
    if math.isclose(result, low, abs_tol=1e-6) or math.isclose(result, high, abs_tol=1e-6):
        return result
    return result


def build_required_fcf_path(current_fcf: float, growth_rate: float, years: int) -> list[dict]:
    path = []
    fcf = current_fcf
    for year in range(1, years + 1):
        fcf *= 1 + growth_rate
        path.append({"year": year, "fcf": fcf})
    return path


def average_fcf(path: list[dict]) -> Optional[float]:
    if not path:
        return None
    return sum(point["fcf"] for point in path) / len(path)


def reverse_dcf_analysis(
    current_fcf: float,
    market_cap: float,
    discount_rate: float,
    terminal_growth: float = 0.03,
) -> list[dict]:
    scenarios = []
    for horizon in (5, 10, 20):
        growth = reverse_dcf_growth(
            current_fcf=current_fcf,
            market_cap=market_cap,
            discount_rate=discount_rate,
            years=horizon,
            terminal_growth=terminal_growth,
        )
        if growth is None:
            scenarios.append(
                {
                    "horizon": horizon,
                    "growth_rate": None,
                    "end_fcf": None,
                    "terminal_value": None,
                    "fcf_yield": None,
                    "path": [],
                }
            )
            continue

        path = build_required_fcf_path(current_fcf, growth, horizon)
        end_fcf = path[-1]["fcf"]
        terminal_value = end_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        scenarios.append(
            {
                "horizon": horizon,
                "growth_rate": growth,
                "end_fcf": end_fcf,
                "average_fcf": average_fcf(path),
                "terminal_value": terminal_value,
                "fcf_yield": end_fcf / market_cap if market_cap > 0 else None,
                "path": path,
            }
        )
    return scenarios


def get_latest_annual_filing(submissions: dict) -> Optional[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    for index, form in enumerate(forms):
        if form not in {"10-K", "10-K405", "20-F"}:
            continue
        accession = accessions[index]
        primary_doc = primary_docs[index]
        filing_date = filing_dates[index]
        cik_no_zero = str(int(submissions["cik"]))
        accession_no_dash = accession.replace("-", "")
        filing_url = SEC_ARCHIVES.format(
            cik_no_zero=cik_no_zero,
            accession_no_dash=accession_no_dash,
            document=primary_doc,
        )
        return {
            "form": form,
            "filing_date": filing_date,
            "url": filing_url,
        }
    return None


def get_latest_quarterly_filing(submissions: dict) -> Optional[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    for index, form in enumerate(forms):
        if form not in {"10-Q", "10-Q/A"}:
            continue
        accession = accessions[index]
        primary_doc = primary_docs[index]
        filing_date = filing_dates[index]
        cik_no_zero = str(int(submissions["cik"]))
        accession_no_dash = accession.replace("-", "")
        filing_url = SEC_ARCHIVES.format(
            cik_no_zero=cik_no_zero,
            accession_no_dash=accession_no_dash,
            document=primary_doc,
        )
        return {
            "form": form,
            "filing_date": filing_date,
            "url": filing_url,
        }
    return None


def filing_html_to_text(html: str, drop_tables: bool = True) -> str:
    soup = BeautifulSoup(html, "html.parser")
    removable_tags = ["script", "style", "ix:header"]
    if drop_tables:
        removable_tags.append("table")
    for tag in soup(removable_tags):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_money_value(text: str) -> Optional[float]:
    cleaned = re.sub(r"[\s\xa0]+", " ", text).strip()
    cleaned = cleaned.replace("$", "").replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cleaned)
    if not match:
        return None
    return float(match.group(1))


def parse_suffix_number(text: str) -> Optional[float]:
    cleaned = text.strip().replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)([TMB])", cleaned, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).upper()
    multipliers = {"T": 1_000_000_000_000, "B": 1_000_000_000, "M": 1_000_000}
    return value * multipliers[suffix]


def parse_numeric_cell(text: str) -> Optional[float]:
    cleaned = re.sub(r"[\s\xa0]+", " ", text).strip()
    if not cleaned:
        return None
    if cleaned in {"—", "-", "N/A", "nm"}:
        return None
    cleaned = cleaned.replace("$", "").replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", cleaned)
    if not match:
        return None
    return float(match.group(0))


def extract_buyback_authorization_from_html(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")
    target_phrase = "approximate dollar value of shares that may yet be purchased under the plans or programs"

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [cell.get_text(" ", strip=True) for cell in cells]
            if row:
                rows.append(row)
        if not rows:
            continue

        header_row_index = None
        column_index = None
        for row_index, row in enumerate(rows):
            for idx, cell_text in enumerate(row):
                normalized = re.sub(r"\s+", " ", cell_text).lower()
                if target_phrase in normalized:
                    header_row_index = row_index
                    column_index = idx
                    break
            if column_index is not None:
                break

        if column_index is None:
            continue

        best_value = None
        best_row_text = None
        for row in rows[header_row_index + 1:]:
            if column_index >= len(row):
                continue
            value = parse_money_value(row[column_index])
            if value is None:
                continue
            row_text = " | ".join(row)
            best_value = value
            best_row_text = row_text

        if best_value is not None:
            return {"amount": best_value, "evidence": best_row_text}

    return None


def extract_mda_section(filing_text: str) -> str:
    normalized = re.sub(r"\s+", " ", filing_text)
    patterns = [
        r"item\s+7\.?\s+management[’'`s\s]+discussion.*?analysis of financial condition and results of operations(.*?)(item\s+7a\.|item\s+8\.)",
        r"management[’'`s\s]+discussion.*?results of operations(.*?)(quantitative and qualitative disclosures|financial statements)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            section = match.group(1)
            return section[:60000]
    return normalized[:60000]


def extract_reported_segments_from_html(html: str, filing_date: Optional[str]) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    target_year = None
    if filing_date:
        target_year = filing_date[:4]

    segment_revenue: dict[str, float] = {}
    segment_profit: dict[str, float] = {}
    excluded_labels = {
        "total", "totals", "net sales", "revenue", "revenues", "operating income",
        "operating profit", "segment operating income", "segment profit", "other",
    }

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [cell.get_text(" ", strip=True) for cell in cells]
            if row:
                rows.append(row)
        if len(rows) < 2:
            continue

        table_text = " ".join(" ".join(row) for row in rows)
        lowered = re.sub(r"\s+", " ", table_text).lower()
        if "segment" not in lowered and "reportable" not in lowered:
            continue

        year_column = None
        header_row_index = None
        for row_index, row in enumerate(rows[:4]):
            for idx, cell in enumerate(row):
                if target_year and target_year in cell:
                    year_column = idx
                    header_row_index = row_index
                    break
            if year_column is not None:
                break
        if year_column is None:
            for row_index, row in enumerate(rows[:4]):
                for idx, cell in enumerate(row):
                    if re.search(r"20\d{2}", cell):
                        year_column = idx
                        header_row_index = row_index
                        break
                if year_column is not None:
                    break
        if year_column is None or year_column == 0:
            continue

        is_revenue_table = any(term in lowered for term in ("net sales", "revenue", "revenues"))
        is_profit_table = any(term in lowered for term in ("operating income", "operating profit", "segment profit"))
        if not is_revenue_table and not is_profit_table:
            continue

        for row in rows[(header_row_index or 0) + 1:]:
            if len(row) <= year_column:
                continue
            label = re.sub(r"\s+", " ", row[0]).strip()
            label_lower = label.lower()
            if not label or label_lower in excluded_labels:
                continue
            if any(token in label_lower for token in ("note", "years ended", "services", "products", "reconciliation")):
                continue

            value = parse_numeric_cell(row[year_column])
            if value is None:
                continue

            if is_profit_table:
                segment_profit[label] = value
            elif is_revenue_table:
                segment_revenue[label] = value

    segments = []
    for name, revenue in segment_revenue.items():
        profit = segment_profit.get(name)
        margin = profit / revenue if profit is not None and revenue not in (0, None) else None
        segments.append(
            {
                "name": name,
                "revenue": revenue,
                "profit": profit,
                "margin": margin,
            }
        )

    segments.sort(key=lambda item: abs(item["revenue"]), reverse=True)
    return segments[:8]


def extract_balance_sheet_row_value_from_html(html: str, filing_date: Optional[str], label_pattern: str) -> Optional[float]:
    soup = BeautifulSoup(html, "html.parser")
    target_year = filing_date[:4] if filing_date else None

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [cell.get_text(" ", strip=True) for cell in cells]
            if row:
                rows.append(row)
        if len(rows) < 2:
            continue

        table_text = " ".join(" ".join(row) for row in rows)
        lowered = re.sub(r"\s+", " ", table_text).lower()
        if "total assets" not in lowered or "shareholders' equity" not in lowered:
            continue

        year_column = None
        for row in rows[:4]:
            for idx, cell in enumerate(row):
                if target_year and target_year in cell:
                    year_column = idx
                    break
            if year_column is not None:
                break

        for row in rows:
            if not row:
                continue
            label = re.sub(r"\s+", " ", row[0]).strip()
            if not re.search(label_pattern, label, re.IGNORECASE):
                continue

            if year_column is not None and year_column < len(row):
                value = parse_numeric_cell(row[year_column])
                if value is not None:
                    return value

            for cell in row[1:]:
                value = parse_numeric_cell(cell)
                if value is not None:
                    return value

    return None


def extract_berkshire_equivalent_b_shares(text: str) -> dict[str, float]:
    normalized = re.sub(r"\s+", " ", text)
    results: dict[str, float] = {}

    patterns = [
        r"on an equivalent class a common stock basis, there were ([0-9,]+) shares outstanding as of ([a-z]+ \d{1,2}, \d{4}) and ([0-9,]+) shares outstanding as of ([a-z]+ \d{1,2}, \d{4})",
        r"on an equivalent class a common stock basis, there were ([0-9,]+) shares outstanding as of ([a-z]+ \d{1,2}, \d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        pairs = []
        if len(groups) == 4:
            pairs = [(groups[0], groups[1]), (groups[2], groups[3])]
        elif len(groups) == 2:
            pairs = [(groups[0], groups[1])]

        for share_count_text, date_text in pairs:
            try:
                period = datetime.strptime(date_text, "%B %d, %Y").strftime("%Y-%m-%d")
                equivalent_a_shares = float(share_count_text.replace(",", ""))
                results[period] = equivalent_a_shares * 1500
            except ValueError:
                continue
        if results:
            break

    return results


def extract_berkshire_balance_sheet_value(text: str, label_pattern: str) -> Optional[float]:
    normalized = re.sub(r"\s+", " ", text)
    match = re.search(
        rf"{label_pattern}\*?\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1).replace(",", "")) * 1_000_000


def sentence_split(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = []
    for part in raw_parts:
        cleaned = part.strip()
        if 60 <= len(cleaned) <= 450:
            sentences.append(cleaned)
    return sentences


def extract_candidate_phrases(sentence: str) -> list[str]:
    lowered = sentence.lower().replace("/", " ")
    lowered = re.sub(r"[^a-z0-9\-\s]", " ", lowered)
    tokens = [token for token in lowered.split() if token and token not in STOPWORDS]
    phrases = []
    for size in (2, 3, 4):
        for index in range(len(tokens) - size + 1):
            phrase_tokens = tokens[index:index + size]
            if any(token.isdigit() for token in phrase_tokens):
                continue
            if all(len(token) <= 2 for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            phrases.append(phrase)
    return phrases


def classify_driver(snippet: str, phrase: str) -> str:
    sample = f"{phrase} {snippet}".lower()
    if any(word in sample for word in ("margin", "cost", "expense", "pricing", "mix", "claims", "spread", "yield")):
        return "Profitability"
    if any(word in sample for word in ("capex", "capacity", "inventory", "working capital", "production")):
        return "Capital intensity"
    return "Revenue"


def identify_key_drivers(mda_text: str, limit: int = 6) -> list[dict]:
    sentences = sentence_split(mda_text)
    phrase_scores: Counter[str] = Counter()
    evidence: dict[str, str] = {}

    for sentence in sentences:
        lowered = sentence.lower()
        if not any(signal in lowered for signal in DRIVER_SIGNAL_WORDS):
            continue

        for phrase in extract_candidate_phrases(sentence):
            if phrase in STOPWORDS or len(phrase) < 8:
                continue
            score = 1
            if any(signal in phrase for signal in DRIVER_SIGNAL_WORDS):
                score += 3
            if any(signal in lowered for signal in ("increase", "decrease", "growth", "margin", "profit", "demand", "pricing", "volume")):
                score += 2
            phrase_scores[phrase] += score
            evidence.setdefault(phrase, sentence)

    drivers = []
    used_roots: set[str] = set()
    for phrase, score in phrase_scores.most_common(30):
        root = " ".join(phrase.split()[:2])
        if root in used_roots:
            continue
        snippet = evidence[phrase]
        if len(set(phrase.split())) == 1:
            continue
        drivers.append(
            {
                "name": phrase.title(),
                "category": classify_driver(snippet, phrase),
                "score": score,
                "evidence": snippet,
            }
        )
        used_roots.add(root)
        if len(drivers) >= limit:
            break

    return drivers


def extract_section(text: str, start_patterns: list[str], end_patterns: list[str], limit: int = 30000) -> str:
    normalized = re.sub(r"\s+", " ", text)
    start_index = None
    for pattern in start_patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            start_index = match.start()
            break
    if start_index is None:
        return normalized[:limit]

    remainder = normalized[start_index:]
    end_index = None
    for pattern in end_patterns:
        match = re.search(pattern, remainder, re.IGNORECASE)
        if match and match.start() > 100:
            end_index = match.start()
            break

    if end_index is None:
        return remainder[:limit]
    return remainder[:end_index]


def extract_buyback_authorization(text: str) -> Optional[dict]:
    normalized = extract_section(
        text,
        start_patterns=[
            r"unregistered sales of equity securities and use of proceeds",
            r"issuer purchases of equity securities",
            r"capital return program",
        ],
        end_patterns=[
            r"item\s+\d+\.",
            r"defaults upon senior securities",
            r"mine safety disclosures",
            r"other information",
            r"exhibits",
        ],
    )
    table_total_match = re.search(
        r"approximate dollar value of\s+shares that may yet be purchased\s+under the plans or programs.*?total\s+[0-9,]+\s+\$ ?([0-9,]+(?:\.[0-9]+)?)",
        normalized,
        re.IGNORECASE,
    )
    if table_total_match:
        raw_amount = float(table_total_match.group(1).replace(",", ""))
        multiplier = 1_000_000
        if re.search(r"in billions", normalized, re.IGNORECASE):
            multiplier = 1_000_000_000
        start = max(table_total_match.start() - 120, 0)
        end = min(table_total_match.end() + 120, len(normalized))
        return {"amount": raw_amount * multiplier, "evidence": normalized[start:end].strip()}

    remaining_match = re.search(
        r"remaining availability under the .*? program was \$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million)",
        normalized,
        re.IGNORECASE,
    )
    if remaining_match:
        amount = float(remaining_match.group(1))
        scale = remaining_match.group(2).lower()
        multiplier = 1_000_000_000 if scale == "billion" else 1_000_000
        start = max(remaining_match.start() - 120, 0)
        end = min(remaining_match.end() + 120, len(normalized))
        return {"amount": amount * multiplier, "evidence": normalized[start:end].strip()}

    utilization_match = re.search(
        r"repurchase up to \$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million).*?\$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million)\s+of the .*? program had been utilized",
        normalized,
        re.IGNORECASE,
    )
    if utilization_match:
        authorized = float(utilization_match.group(1))
        authorized_scale = utilization_match.group(2).lower()
        utilized = float(utilization_match.group(3))
        utilized_scale = utilization_match.group(4).lower()
        authorized_multiplier = 1_000_000_000 if authorized_scale == "billion" else 1_000_000
        utilized_multiplier = 1_000_000_000 if utilized_scale == "billion" else 1_000_000
        remaining = (authorized * authorized_multiplier) - (utilized * utilized_multiplier)
        start = max(utilization_match.start() - 120, 0)
        end = min(utilization_match.end() + 120, len(normalized))
        return {"amount": remaining, "evidence": normalized[start:end].strip()}

    full_text_match = re.search(
        r"remaining availability under the .*? program was \$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million)",
        re.sub(r"\s+", " ", text),
        re.IGNORECASE,
    )
    if full_text_match:
        amount = float(full_text_match.group(1))
        scale = full_text_match.group(2).lower()
        multiplier = 1_000_000_000 if scale == "billion" else 1_000_000
        start = max(full_text_match.start() - 120, 0)
        end = min(full_text_match.end() + 120, len(re.sub(r"\s+", " ", text)))
        normalized_full_text = re.sub(r"\s+", " ", text)
        return {"amount": amount * multiplier, "evidence": normalized_full_text[start:end].strip()}

    table_label_patterns = [
        r"approximate dollar value of shares that may yet be purchased under the plans or programs[^$]{0,120}\$ ?([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"approximate dollar value of shares that may yet be purchased under the plans or programs[^0-9]{0,120}([0-9][0-9,]*(?:\.[0-9]+)?)",
    ]
    for pattern in table_label_patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue
        raw_value = float(match.group(1).replace(",", ""))
        start = max(match.start() - 160, 0)
        end = min(match.end() + 160, len(normalized))
        return {"amount": raw_value, "evidence": normalized[start:end].strip()}

    patterns = [
        r"\$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million)?\s+(?:remained|remain|was remaining|available)\s+(?:available\s+)?(?:for\s+)?(?:repurchase|buyback)",
        r"(?:repurchase|buyback).*?\$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million)?\s+(?:remaining|available)",
        r"\$ ?([0-9]+(?:\.[0-9]+)?)\s*(billion|million)?\s+remains?\s+available\s+under\s+(?:the\s+)?(?:share|stock)\s+repurchase",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue
        raw_value = float(match.group(1))
        scale = (match.group(2) or "").lower()
        multiplier = 1.0
        if scale == "billion":
            multiplier = 1_000_000_000
        elif scale == "million":
            multiplier = 1_000_000
        value = raw_value * multiplier
        start = max(match.start() - 120, 0)
        end = min(match.end() + 120, len(normalized))
        return {"amount": value, "evidence": normalized[start:end].strip()}
    return None


def analyze_company(ticker: str, discount_rate: float) -> dict:
    return analyze_company_with_terminal_growth(ticker, discount_rate, 0.03)


def analyze_company_with_terminal_growth(ticker: str, discount_rate: float, terminal_growth: float) -> dict:
    if discount_rate <= terminal_growth:
        raise ValueError("Discount rate must be greater than the terminal growth rate.")

    cik = resolve_cik(ticker)
    if not cik:
        raise ValueError(f"Could not resolve ticker '{ticker}' to a SEC CIK.")

    submissions = fetch_submissions(cik)
    facts = fetch_company_facts(cik)
    financials = build_financials(facts)
    periods = years_available(financials)

    if not periods:
        raise ValueError("No annual SEC financial history was found for this company.")

    market_data = get_market_data(ticker)
    market_cap = market_data.get("market_cap")
    price = market_data.get("price")
    latest_fcf = latest_metric(financials, "Free Cash Flow")

    if not market_cap and price:
        shares = latest_metric(financials, "Shares Outstanding")
        if shares:
            market_cap = price * shares

    if not market_cap:
        raise ValueError("Could not determine current market cap from available public data.")

    filing = get_latest_annual_filing(submissions)
    quarterly_filing = get_latest_quarterly_filing(submissions)
    mda_text = ""
    drivers: list[dict] = []
    segments: list[dict] = []
    special_share_counts: dict[str, float] = {}
    if filing:
        try:
            filing_html = sec_get_text(filing["url"])
            segments = extract_reported_segments_from_html(filing_html, filing.get("filing_date"))
            filing_text = filing_html_to_text(filing_html)
            if ticker.upper().replace("-", ".") in {"BRK.B", "BRK.A"}:
                special_share_counts.update(extract_berkshire_equivalent_b_shares(filing_text))
                berkshire_cash_value = extract_berkshire_balance_sheet_value(
                    filing_text,
                    r"cash and cash equivalents",
                )
                if berkshire_cash_value is None:
                    berkshire_cash_value = extract_balance_sheet_row_value_from_html(
                        filing_html,
                        filing.get("filing_date"),
                        r"cash and cash equivalents",
                    )
                berkshire_tbill_value = extract_berkshire_balance_sheet_value(
                    filing_text,
                    r"short-term investments in u\.?s\.? treasury bills",
                )
                if berkshire_tbill_value is None:
                    berkshire_tbill_value = extract_balance_sheet_row_value_from_html(
                        filing_html,
                        filing.get("filing_date"),
                        r"short-term investments in u\.?s\.? treasury bills",
                    )
                if berkshire_cash_value is not None:
                    financials["Cash & Equivalents"] = {
                        **financials.get("Cash & Equivalents", {}),
                        filing.get("filing_date"): berkshire_cash_value,
                    }
                if berkshire_tbill_value is not None:
                    financials["Short-term Investments"] = {
                        **financials.get("Short-term Investments", {}),
                        filing.get("filing_date"): berkshire_tbill_value,
                    }
            mda_text = extract_mda_section(filing_text)
            drivers = identify_key_drivers(mda_text)
        except Exception:
            drivers = []
            segments = []

    if latest_fcf is None:
        raise ValueError("Latest free cash flow could not be derived from SEC cash flow data.")

    repurchase_authorization = None
    if quarterly_filing:
        try:
            quarterly_html = sec_get_text(quarterly_filing["url"])
            if ticker.upper().replace("-", ".") in {"BRK.B", "BRK.A"}:
                quarterly_text_for_shares = filing_html_to_text(quarterly_html, drop_tables=False)
                special_share_counts.update(extract_berkshire_equivalent_b_shares(quarterly_text_for_shares))
            repurchase_authorization = extract_buyback_authorization_from_html(quarterly_html)
            if repurchase_authorization is None:
                quarterly_text = filing_html_to_text(quarterly_html, drop_tables=False)
                repurchase_authorization = extract_buyback_authorization(quarterly_text)
        except Exception:
            repurchase_authorization = None

    if special_share_counts:
        financials["Shares Outstanding"] = {
            **financials.get("Shares Outstanding", {}),
            **special_share_counts,
        }

    cash = latest_metric(financials, "Cash & Equivalents") or 0.0
    short_term_investments = latest_metric(financials, "Short-term Investments") or 0.0
    debt = latest_metric(financials, "Total Debt")
    if debt is None:
        debt = latest_metric(financials, "Long-term Debt") or 0.0
    latest_shares_outstanding = latest_metric(financials, "Shares Outstanding")

    if ticker.upper().replace("-", ".") in {"BRK.B", "BRK.A"} and price and latest_shares_outstanding:
        market_cap = price * latest_shares_outstanding

    enterprise_value = market_cap + debt - cash - short_term_investments
    reverse_dcf = reverse_dcf_analysis(
        current_fcf=latest_fcf,
        market_cap=market_cap,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
    )

    revenue = latest_metric(financials, "Revenue")
    operating_income = latest_metric(financials, "Operating Income")
    net_income = latest_metric(financials, "Net Income")
    operating_margin = operating_income / revenue if revenue and operating_income is not None else None
    fcf_margin = latest_fcf / revenue if revenue else None
    net_margin = net_income / revenue if revenue and net_income is not None else None
    roe_by_year = []
    for period in periods[-5:]:
        equity_by_year = financials.get("Shareholders Equity", {}).get(period)
        fcf_by_year = financials.get("Free Cash Flow", {}).get(period)
        roe_value = None
        if equity_by_year not in (None, 0) and fcf_by_year is not None:
            roe_value = fcf_by_year / equity_by_year
        roe_by_year.append({"period": period, "roe": roe_value})

    return {
        "ticker": ticker.upper(),
        "company_name": submissions.get("name", ticker.upper()),
        "industry": submissions.get("sicDescription") or "N/A",
        "cik": cik,
        "periods": periods[-5:],
        "historical_rows": compute_historical_summary(financials, periods[-5:]),
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "price": price,
        "cash": cash,
        "short_term_investments": short_term_investments,
        "debt": debt,
        "latest_revenue": revenue,
        "latest_operating_income": operating_income,
        "latest_net_income": net_income,
        "latest_fcf": latest_fcf,
        "latest_equity": latest_metric(financials, "Shareholders Equity"),
        "latest_shares_outstanding": latest_shares_outstanding,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "fcf_margin": fcf_margin,
        "terminal_growth": terminal_growth,
        "reverse_dcf": reverse_dcf,
        "roe_by_year": roe_by_year,
        "filing": filing,
        "segments": segments,
        "quarterly_filing": quarterly_filing,
        "repurchase_authorization": repurchase_authorization,
        "drivers": drivers,
    }


@app.template_filter("money")
def money_filter(value: Optional[float]) -> str:
    return format_large_number(value)


@app.template_filter("pct")
def pct_filter(value: Optional[float]) -> str:
    return format_percent(value)


@app.template_filter("shares")
def shares_filter(value: Optional[float]) -> str:
    return format_shares(value)


@app.template_filter("fy")
def fy_filter(period: str) -> str:
    return f"FY{period[:4]}"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    ticker = "AAPL"
    discount_rate = "0.10"
    terminal_growth = "0.03"

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        discount_rate = request.form.get("discount_rate", "0.10").strip()
        terminal_growth = request.form.get("terminal_growth", "0.03").strip()

        try:
            parsed_discount_rate = float(discount_rate)
            parsed_terminal_growth = float(terminal_growth)
            result = analyze_company_with_terminal_growth(ticker, parsed_discount_rate, parsed_terminal_growth)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        ticker=ticker,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        generated_at=datetime.now(),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
