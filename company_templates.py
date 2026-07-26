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
import importlib
import json
import os
from typing import Optional

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "company_templates")
SECTOR_DIR = os.path.join(TEMPLATE_DIR, "_sectors")
PLUGIN_PKG = "company_plugins"

_CACHE: dict[str, Optional[dict]] = {}
_PLUGIN_CACHE: dict[str, object] = {}


# ── Company plugins ──────────────────────────────────────────────────────────
# Some companies need more than declarative config. Berkshire tags no
# consolidated debt concept at all, files class-A share counts that are useless
# for per-B-share math, and reports operating earnings only in an MD&A segment
# table — all of which require parsing 10-K/10-Q prose. That logic is real code,
# and pretending otherwise by inventing a regex mini-language in JSON would be
# harder to maintain, not easier.
#
# So a template may declare `"plugin": "BRK_B"`, and the matching module in
# company_plugins/ implements whatever hooks it needs. The core app calls the
# hooks generically and holds no per-company knowledge.
#
# Hooks (all optional, all mutate `financials` in place):
#   seed_from_facts(facts, financials)
#       Correct the tagged data before any filings are fetched.
#   apply_annual_filings(filings, financials, ctx)
#       Walk the 10-K list (newest first) and fill gaps from filing text.
#       ctx = {"get_text": filing -> cached text, "fy_get": ..., "min_year": ...}
#   apply_quarterly(financials, quarter_end_dates, quarter_filing_links, ctx)
#       Same for the 10-Qs. ctx = {"get_text_url": url -> text}

def load_plugin(template: Optional[dict]):
    """Import the plugin module a template declares, or None."""
    name = (template or {}).get("plugin")
    if not name:
        return None
    if name in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[name]
    try:
        mod = importlib.import_module(f"{PLUGIN_PKG}.{name}")
    except Exception:
        mod = None
    _PLUGIN_CACHE[name] = mod
    return mod


def call_hook(plugin, hook: str, *args, default=None):
    """Invoke a plugin hook if present; never let a plugin break the request."""
    if plugin is None:
        return default
    fn = getattr(plugin, hook, None)
    if not callable(fn):
        return default
    try:
        return fn(*args)
    except Exception:
        return default


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

    # Share classes are written variously as BRK.B, BRK-B or BRK/B depending on
    # where the ticker came from; template files use the dashed form.
    company = None
    for candidate in (tk, tk.replace(".", "-").replace("/", "-")):
        company = _read_json(os.path.join(TEMPLATE_DIR, f"{candidate}.json"))
        if company is not None:
            break
    if company is None:
        _CACHE[tk] = None
        return None

    # A second share class can defer to the primary one rather than duplicate it.
    seen = {tk}
    while company.get("extends"):
        parent_name = company["extends"].upper()
        if parent_name in seen:
            break
        seen.add(parent_name)
        parent = _read_json(os.path.join(TEMPLATE_DIR, f"{parent_name}.json"))
        if parent is None:
            break
        company = {**parent, **{k: v for k, v in company.items() if k != "extends"}}

    sector = load_sector(company.get("sector_template", "")) if company.get("sector_template") else {}
    _fold_structural_facts(company)
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
        # Override a STANDARD row's label with the company's own wording, for
        # concepts that have no XBRL tag at all (so the filer-label mechanism
        # via MetaLinks.json — which only covers tagged concepts — can never
        # supply it). SPG's "ffo" row otherwise stays stuck with the generic
        # "(NI + D&A - RE Gains)" formula suffix even once the underlying
        # value is Simon's own reported figure, not that formula.
        "row_labels":  {**sector.get("row_labels", {}), **company.get("row_labels", {})},
        "suppress":    sorted(set(sector.get("suppress", [])) | set(company.get("suppress", []))),
        "caveats":     sector.get("caveats", []) + company.get("caveats", []),
        "summary":     company.get("summary", ""),
        "generated_by": company.get("generated_by", ""),
        "plugin":      company.get("plugin") or sector.get("plugin"),
        "history":     company.get("history", {}),
    }
    _CACHE[tk] = merged
    return merged


def _fold_structural_facts(company: dict) -> None:
    """
    Fold `history.comparability` facts marked `"years": "all"` into the
    ordinary per-metric annotations.

    A structural fact (Berkshire's subsidiary debt is non-recourse; float is a
    funding source, not leverage) is true in every year, so pinning it to one
    year's cell would be arbitrary and repeating it in every cell would just be
    noise. It belongs on the row, same as any other standing caveat. Facts tied
    to a specific year or range are left out of this — see year_notes() below,
    which is where those are surfaced instead.
    """
    facts = ((company.get("history") or {}).get("comparability")) or []
    structural = [f for f in facts if (f.get("years") or "").strip() == "all"]
    if not structural:
        return
    annotations = dict(company.get("annotations") or {})
    for f in structural:
        note = f"{f.get('fact', '')} — {f.get('detail', '')}".strip(" —")
        for metric in f.get("affects", []):
            existing = annotations.get(metric)
            annotations[metric] = f"{existing}  {note}" if existing else note
    company["annotations"] = annotations


def _parse_fact_years(spec: str, all_years: list[int]) -> set[int]:
    """
    Parse a comparability fact's `years` field against the calendar years
    actually present in this filer's data.

      "2016"        a single year
      "2018,2019,2020"  a discrete list
      "2018-"       2018 onward — open-ended, so it always extends to
                    whatever the newest fetched year turns out to be
      "all"         every year in the data
    """
    spec = (spec or "").strip()
    if not spec:
        return set()
    if spec == "all":
        return set(all_years)
    if spec.endswith("-"):
        try:
            start = int(spec[:-1])
        except ValueError:
            return set()
        return {y for y in all_years if y >= start}
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


def year_notes(financials: dict, template: Optional[dict]) -> dict:
    """
    {metric: {year_str: [note, ...]}} from `history.comparability`.

    A company is not the same company across fifteen years — an acquisition
    gets consolidated mid-history, an accounting standard changes what a line
    MEANS, a one-off tax or impairment item lands in a single year — and the
    read on a metric should change at the year it's actually affected. This
    resolves each fact's `years` spec against the calendar years actually
    present in the data (open-ended specs like "2018-" need that to know how
    far "onward" currently reaches).

    Each fact is surfaced ONCE: on its earliest affected year, on the first
    metric named in `affects`. A fact like ASU 2016-01 runs 2018 onward and
    touches net_income, eps_diluted, roe and net_margin — flagging every one of
    those cells in every one of eight years is eight-times-four repetition of
    the same sentence, which teaches a reader to stop reading it. One cell is
    enough to make the fact discoverable; the row-level annotation (see
    _fold_structural_facts) still carries it for anyone reading that metric on
    a later year.
    """
    facts = ((template or {}).get("history") or {}).get("comparability") or []
    if not facts:
        return {}

    all_years: set[int] = set()
    for series in financials.values():
        if not isinstance(series, dict):
            continue
        for period in series:
            if len(period) >= 4 and period[:4].isdigit() and not period.startswith("Q"):
                all_years.add(int(period[:4]))
    all_years_list = sorted(all_years)

    out: dict[str, dict[str, list[str]]] = {}
    for f in facts:
        spec = (f.get("years") or "").strip()
        if spec == "all":
            continue   # structural, not dated — folded into the row annotation instead
        years = _parse_fact_years(spec, all_years_list)
        affects = f.get("affects", [])
        if not years or not affects:
            continue
        note = f"{f.get('fact', '')} — {f.get('detail', '')}".strip(" —")
        first_year = str(min(years))
        canonical_metric = affects[0]
        out.setdefault(canonical_metric, {}).setdefault(first_year, []).append(note)
    return out


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


# ── Authored history ─────────────────────────────────────────────────────────

def apply_history(financials: dict, template: Optional[dict]) -> None:
    """
    Merge a template's authored `history.series` into `financials`.

    Filed 10-Ks are immutable, so a careful year-by-year reading of them is a
    fact that can be committed rather than re-derived from filing prose on
    every request. Where a template declares history, it is AUTHORITATIVE for
    the periods it covers: it was written against the filings themselves, with
    its provenance and any restatements recorded alongside it, which is more
    than the tagged data can say for a company like Berkshire that tags no
    concept for the figure at all.

    Periods the history does not cover are left untouched, so a filing made
    after the template was written still flows through the normal path.
    """
    hist = (template or {}).get("history") or {}
    for metric, series in (hist.get("series") or {}).items():
        if not isinstance(series, dict):
            continue
        target = financials.setdefault(metric, {})
        for period, value in series.items():
            if value is not None:
                target[period] = float(value)


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
            # Optional: splice the row in directly after an existing metric
            # (e.g. right under net income in the income statement) instead of
            # relegating it to the catch-all company-specific block.
            "after":        spec.get("after"),
            "reported_ref": spec.get("reported_ref"),
            "why":          spec.get("why", ""),
            "evidence":     spec.get("evidence", ""),
            "confidence":   spec.get("confidence", ""),
        })
    return rows
