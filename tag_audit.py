"""
Where the concept vocabularies disagree.

Read-only. Nothing here changes behaviour; it answers the question that has to
be settled BEFORE the vocabularies are merged — which differences between them
are deliberate and which are drift.

The app keeps one dict of concepts; the screener keeps four named lists plus a
scatter of literals passed inline to _merge_frames; the UI keeps its own row
keys; bindings keep a third naming of the same ideas. Nothing checks them
against each other, so a concept added to one is simply absent from the others
until a filer happens to need it. Every gap found this way so far was found by
a person looking at a blank cell:

    ProfitLoss            in the screener, not the app  -> VeriSign had no net
                          income for 13 years
    SeniorNotes           in the screener, not the app  -> VeriSign had no debt
                          at all, and no EV
    pretax_income         no tags anywhere in the app, and no UI row either, so
                          the absence was invisible rather than blank

The point of this file is to find the rest of them by reading, not by waiting.

Three checks, because they fail differently:

  drift      a concept one side knows and the other does not. Sometimes right —
             the app deliberately keeps ProfitLoss OUT of its primary list,
             because extract_annual_series takes the largest absolute value and
             the consolidated figure would outrank the parent-attributable one.
             That is why the merge needs a ROLE per tag (primary / fallback /
             component) rather than one flat list.

  untagged   a metric the app names but gives no tags. pretax_income was one:
             bound filers got it from the overlay and every unbound filer
             showed an empty row.

  unrendered a metric that resolves but has no UI row, or a UI row nothing
             populates. An unrendered metric is worse than a blank one — a row
             that is not there reads as "no such measure" rather than "missing".

Usage:  python3 tag_audit.py
"""
from __future__ import annotations

import json
import re

import app
import screener


def _screener_tags() -> set:
    """Every us-gaap concept the screener names, wherever it names it."""
    out: set = set()
    for name in ("_DEBT_TOTAL_TAGS", "_DEBT_COMPONENT_TAGS", "_DPS_TAGS"):
        out |= set(getattr(screener, name, []))
    for entries in getattr(screener, "_CF_TAGS", {}).values():
        out |= {tag for ns, tag in entries if ns == "us-gaap"}
    # The inline literals. These are the ones no one can find by grepping for a
    # constant, which is most of the reason the vocabularies drifted at all.
    src = open(screener.__file__).read()
    for block in re.findall(r"_merge_frames\(\s*\[(.*?)\]", src, re.S):
        out |= set(re.findall(r'"([A-Za-z][A-Za-z0-9]+)"', block))
    return out


def _ui_row_keys() -> set:
    html = open("templates/index.html").read()
    return set(re.findall(r"key:'([a-z0-9_]+)'", html))


def _binding_metrics() -> set:
    try:
        g = json.load(open("bindings/global.json"))
    except Exception:
        return set()
    return {m.get("metric") for m in g.get("metrics", []) if m.get("metric")}


def main() -> None:
    app_tags = {t for tags in app.METRIC_TAGS.values() for t in (tags or [])}
    scr_tags = _screener_tags()
    ui_keys = _ui_row_keys()
    metrics = set(app.METRIC_TAGS)

    print(f"app concepts {len(app_tags)} across {len(metrics)} metrics | "
          f"screener concepts {len(scr_tags)} | UI rows {len(ui_keys)}\n")

    only_scr = sorted(scr_tags - app_tags)
    print(f"── in the screener, absent from the app ({len(only_scr)}) "
          f"— each is a metric that can resolve on one surface and not the other")
    for t in only_scr:
        print(f"     {t}")

    untagged = sorted(k for k, v in app.METRIC_TAGS.items() if not v)
    print(f"\n── app metrics with NO tags ({len(untagged)}) "
          f"— never read from companyfacts; bound filers only")
    for k in untagged:
        print(f"     {k}{'   (has a UI row)' if k in ui_keys else '   (no UI row either)'}")

    known = metrics | _binding_metrics()
    unrendered = sorted(m for m in metrics if m not in ui_keys)
    print(f"\n── metrics with no UI row ({len(unrendered)}) "
          f"— resolve but are never shown")
    for m in unrendered:
        print(f"     {m}")

    orphan = sorted(k for k in ui_keys if k not in known)
    print(f"\n── UI rows nothing names ({len(orphan)}) "
          f"— populated by a derivation, a template, or not at all")
    for k in orphan:
        print(f"     {k}")


if __name__ == "__main__":
    main()
