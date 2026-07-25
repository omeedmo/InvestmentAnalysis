"""
Per-company metric templates.

The global METRIC_TAGS mapping is a one-size-fits-all view of the world, and
the coverage audit showed that fails for essentially every company: only 3 of
502 S&P 500 filers had no material line item the app ignored. Worse, the gaps
are meaning-level, not just naming-level — ORCL's negative equity makes ROE
nonsense, SPG's GAAP book value is depreciated-cost real estate, BRK's net
income is dominated by mark-to-market swings on a $271B equity portfolio.

A template lets each company be measured the way that company should be
measured, in the terminology it actually uses, while a sector template carries
what's common (the audit found gaps are mostly sector-systematic: `Deposits`
missing for 100% of banks, real estate for 96% of REITs).

Four things a template can do:

  add_tags    extend METRIC_TAGS for this filer, closing coverage gaps
  metrics     define derived metrics, including ADJUSTED ones that pair with
              the reported figure so both are shown (never silently replaced)
  annotations attach a caveat to a metric where the right adjustment is known
              but NOT computable from XBRL — say so rather than fabricate it
  suppress    hide metrics that are actively misleading for this company

Expressions are evaluated over other metric series with a whitelisted AST
walker — only names, numbers and + - * / are permitted, so a template can
never execute arbitrary code.
"""
from __future__ import annotations

import ast
import json
import os
from typing import Optional

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "company_templates")
SECTOR_DIR = os.path.join(TEMPLATE_DIR, "_sectors")

_CACHE: dict[str, Optional[dict]] = {}


# ── Safe expression evaluation ───────────────────────────────────────────────

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Constant,
)


def _check(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression element: {type(n).__name__}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ValueError("only numeric constants allowed")


def eval_expr(expr: str, values: dict[str, Optional[float]]) -> Optional[float]:
    """
    Evaluate an arithmetic expression over metric values for one period.

    Returns None if any referenced metric is missing for that period — a
    partial figure would be worse than no figure, since the reader can't tell
    which component silently dropped out.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        _check(tree)
    except Exception:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name):
            v = values.get(n.id)
            return None if v is None else float(v)
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if v is None:
                return None
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return None if b == 0 else a / b
        return None

    try:
        return ev(tree)
    except Exception:
        return None


# ── Loading & merging ────────────────────────────────────────────────────────

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_sector(name: str) -> dict:
    return _read_json(os.path.join(SECTOR_DIR, f"{name}.json")) or {}


def load_template(ticker: str) -> Optional[dict]:
    """Company template merged over its sector template (company wins)."""
    tk = (ticker or "").upper()
    if tk in _CACHE:
        return _CACHE[tk]

    company = _read_json(os.path.join(TEMPLATE_DIR, f"{tk}.json"))
    if company is None:
        _CACHE[tk] = None
        return None

    sector = load_sector(company.get("sector_template", "")) if company.get("sector_template") else {}
    merged = {
        "ticker": tk,
        "sector_template": company.get("sector_template"),
        "add_tags":    {**sector.get("add_tags", {}),    **company.get("add_tags", {})},
        # New series introduced via add_tags that are balances, not flows.
        "point_in_time": sorted(set(sector.get("point_in_time", []))
                                | set(company.get("point_in_time", []))),
        # Sector metrics first so company entries with the same key override.
        "metrics":     _dedupe_by_key(sector.get("metrics", []) + company.get("metrics", [])),
        "annotations": {**sector.get("annotations", {}), **company.get("annotations", {})},
        "suppress":    sorted(set(sector.get("suppress", [])) | set(company.get("suppress", []))),
        "caveats":     sector.get("caveats", []) + company.get("caveats", []),
        "summary":     company.get("summary", ""),
        "generated_by": company.get("generated_by", ""),
    }
    _CACHE[tk] = merged
    return merged


def _dedupe_by_key(metrics: list) -> list:
    out: dict[str, dict] = {}
    for m in metrics:
        if m.get("key"):
            out[m["key"]] = m
    return list(out.values())


def apply_add_tags(metric_tags: dict, template: Optional[dict]) -> dict:
    """METRIC_TAGS with this company's extra tags appended (originals first,
    so a company tag is only a fallback, never a silent override)."""
    if not template or not template.get("add_tags"):
        return metric_tags
    merged = dict(metric_tags)
    for metric, tags in template["add_tags"].items():
        base = list(merged.get(metric) or [])
        merged[metric] = base + [t for t in tags if t not in base]
    return merged


# ── Computing template metrics ───────────────────────────────────────────────

def compute_metrics(financials: dict, template: Optional[dict]) -> dict:
    """
    Evaluate each template metric across every period present in `financials`.

    Returns {metric_key: {period: value}} for metrics that produced at least
    one value; empty ones are dropped so the UI doesn't render a blank row.
    """
    if not template or not template.get("metrics"):
        return {}

    periods: set[str] = set()
    for series in financials.values():
        if isinstance(series, dict):
            periods.update(series.keys())

    out: dict[str, dict] = {}
    for spec in template["metrics"]:
        key, expr = spec.get("key"), spec.get("expr")
        if not key or not expr:
            continue
        series: dict[str, float] = {}
        for p in periods:
            vals = {m: (s.get(p) if isinstance(s, dict) else None)
                    for m, s in financials.items()}
            # Allow templates to build on earlier template metrics.
            for done_k, done_s in out.items():
                vals[done_k] = done_s.get(p)
            v = eval_expr(expr, vals)
            if v is not None:
                series[p] = v
        if series:
            out[key] = series
    return out


def rows_for_ui(template: Optional[dict], computed: dict) -> list:
    """Row descriptors for template metrics that actually produced data."""
    if not template:
        return []
    rows = []
    for spec in template.get("metrics", []):
        if spec.get("key") not in computed:
            continue
        rows.append({
            "key":          spec["key"],
            "label":        spec.get("label", spec["key"]),
            "t":            spec.get("type", "$"),
            "basis":        spec.get("basis", "derived"),
            "reported_ref": spec.get("reported_ref"),
            "why":          spec.get("why", ""),
            "evidence":     spec.get("evidence", ""),
            "confidence":   spec.get("confidence", ""),
        })
    return rows
