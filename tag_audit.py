"""
Does every surface still share one vocabulary?

Read-only. Nothing here changes behaviour.

This used to be a report on how far the concept lists had drifted apart. They
are merged now — vocabulary.py owns every us-gaap / ifrs-full / dei element
name, and app.py, screener.py, the annual pass, the quarterly pass and the UI
all read from it. So this is no longer a survey; it is the check that keeps it
that way, and it should print CLEAN.

What it can catch, and what each failure looked like the last time it happened
for real:

  literal      a concept name written directly into a consumer instead of
               declared in the vocabulary. This is the drift itself, and it is
               invisible until a filer needs the tag: ProfitLoss sat in the
               screener and not the app, so VeriSign showed no net income for
               thirteen years while the screen ranked it on earnings; SeniorNotes
               likewise, so the same company showed no debt and no enterprise
               value at all.

  unrendered   a metric that resolves but has no UI row. Worse than a blank
               cell — a row that is not there reads as "no such measure" rather
               than "missing", which is how pretax_income stayed invisible.
               A metric may opt out with row=False, but only with a note saying
               what renders it instead.

  contradiction  the UI renders a row the vocabulary declared internal, or the
               vocabulary promises a row the template does not have.

  untagged     a metric the vocabulary names but gives no concepts, on either
               surface. pretax_income was one.

  template     a company template reaching past a vocabulary decision rather
               than extending it. Adding an element name nothing else has heard
               of is the whole point of a template — a coverage gap real for one
               filer is noise for the next. Adding one the vocabulary declares
               screen-only, or declares as a component or gap-fill, is not:
               ORCL's template appended DebtLongtermAndShorttermCombinedAmount
               to long_term_debt, which already contains the current portion the
               analyze page adds separately. It never won for Oracle, so it was
               dead weight rather than a live double count — but nothing said
               so, and nothing would have when it started winning.
               Also checked: a template naming a metric the vocabulary does not
               have (it would arrive with no kind and no row), and a template
               declaring its own point_in_time list (kind is the vocabulary's,
               so the annual and quarterly columns cannot disagree).

  asymmetry    a concept one surface reads and the other does not. Not a
               failure — the `on` map exists to record exactly this, and there
               are good reasons for most of them (the screen resolves a frames
               slice across ~3,700 filers and cannot afford the analyze page's
               wider lists). Printed so the list stays short and reviewed
               rather than growing quietly.

Usage:  python3 tag_audit.py      (exit 0 = clean)
"""
from __future__ import annotations

import glob
import json
import re
import sys

import vocabulary as V

# The two surfaces that MEASURE with concepts. coverage_audit.py reads the
# vocabulary too, but the concept names left in it are a suppression list — the
# roll-up tags it must NOT report as blind spots — which is a different kind of
# list and correctly stays there.
CONSUMERS = ("app.py", "screener.py")

# Not migrated, and reported rather than checked. intrinsic_value.py is a
# standalone script — nothing imports it, and it keeps a sixth concept list of
# its own, keyed by display label rather than metric name. Folding it in is not
# a rename: its "Total Debt" mixes current and noncurrent concepts into one
# largest-wins list, and it puts bare LongTermDebt AHEAD of
# LongTermDebtNoncurrent, which is the partial-reading order this screen was
# just moved off. Merging it means settling what that row means first.
UNMIGRATED = ("intrinsic_value.py",)

# Three or more CamelCase words: what a us-gaap element name looks like and
# what a report caption ("STATEMENTSOFOPERATIONS") does not.
CONCEPT_SHAPED = re.compile(r'"((?:[A-Z][a-z0-9]+){3,})"')

# Strings that are concept-shaped but are not concepts.
NOT_CONCEPTS = {
    "ValueAnchor",
}


def _ui_row_keys() -> set:
    html = open("templates/index.html").read()
    return set(re.findall(r"key:'([a-z0-9_]+)'", html))


def _binding_metrics() -> set:
    try:
        g = json.load(open("bindings/global.json"))
    except Exception:
        return set()
    return {m.get("metric") for m in g.get("metrics", []) if m.get("metric")}


def _all_concepts() -> set:
    return {c.tag for m in V.VOCAB.values() for c in m.concepts}


def main() -> int:
    failures = 0
    concepts = _all_concepts()
    ui_keys  = _ui_row_keys()
    metrics  = set(V.VOCAB)

    n_analyze = sum(1 for m in V.VOCAB.values() for c in m.concepts if c.role(V.ANALYZE))
    n_screen  = sum(1 for m in V.VOCAB.values() for c in m.concepts if c.role(V.SCREEN))
    print(f"{len(concepts)} concepts across {len(metrics)} metrics "
          f"({n_analyze} read by the analyze page, {n_screen} by the screen) | "
          f"{len(ui_keys)} UI rows\n")

    # ── literals ────────────────────────────────────────────────────────────
    stray = []
    for path in CONSUMERS:
        src = open(path).read()
        for tag in sorted(set(CONCEPT_SHAPED.findall(src))):
            if tag in NOT_CONCEPTS:
                continue
            if tag in concepts:
                stray.append((path, tag, "declared in the vocabulary, but written out here too"))
            elif re.match(r"^(?:[A-Z][a-z0-9]+){4,}$", tag):
                stray.append((path, tag, "looks like a concept and is in no vocabulary metric"))
    print(f"── concept names written into a consumer ({len(stray)})")
    for path, tag, why in stray:
        print(f"     {path}: {tag}  — {why}")
    failures += len(stray)

    # ── untagged ────────────────────────────────────────────────────────────
    untagged = sorted(k for k, m in V.VOCAB.items()
                      if not m.concepts and not m.by_template)
    print(f"\n── metrics with no concepts ({len(untagged)}) "
          f"— excludes the {len(V.by_template())} whose concepts a template supplies")
    for k in untagged:
        print(f"     {k}")
    failures += len(untagged)

    unreadable = sorted(k for k in metrics - V.by_template()
                        if not V.tags(k, V.SERIES, V.ANALYZE)
                        and not V.tags(k, V.SERIES, V.SCREEN)
                        and not V.ns_tags(k, V.SERIES, V.SCREEN))
    print(f"\n── metrics no surface can resolve ({len(unreadable)})")
    for k in unreadable:
        print(f"     {k}")
    failures += len(unreadable)

    # ── rows ────────────────────────────────────────────────────────────────
    promised = sorted(k for k in V.rendered() if k not in ui_keys)
    print(f"\n── metrics promised a UI row that the template has not got ({len(promised)})")
    for k in promised:
        print(f"     {k}")
    failures += len(promised)

    contradicted = sorted(k for k in V.internal() if k in ui_keys)
    print(f"\n── metrics declared internal that the template renders anyway ({len(contradicted)})")
    for k in contradicted:
        print(f"     {k}")
    failures += len(contradicted)

    # ── company templates ───────────────────────────────────────────────────
    # A template extends the vocabulary for one filer. An element name nothing
    # else has heard of is exactly what belongs in one; reaching past a
    # decision the vocabulary made about the analyze surface is not.
    tmpl_files = sorted(glob.glob("company_templates/*.json")
                        + glob.glob("company_templates/_sectors/*.json"))
    bad_tmpl, unknown_metric, stale_pit, supplied = [], [], [], set()
    for path in tmpl_files:
        try:
            d = json.load(open(path))
        except Exception as e:
            bad_tmpl.append((path, "", f"unreadable: {e}"))
            continue
        if "point_in_time" in d:
            stale_pit.append(path)
        for metric, added in (d.get("add_tags") or {}).items():
            if metric not in metrics:
                unknown_metric.append((path, metric))
                continue
            supplied.add(metric)
            for tag, why in V.template_conflicts(metric, added or []):
                bad_tmpl.append((path, metric, f"{tag} — {why}"))

    print(f"\n── template concepts the vocabulary refuses ({len(bad_tmpl)}) "
          f"— apply_add_tags drops these at request time")
    for path, metric, why in bad_tmpl:
        print(f"     {path}: {metric}: {why}")
    failures += len(bad_tmpl)

    print(f"\n── template add_tags naming a metric the vocabulary does not have "
          f"({len(unknown_metric)}) — no kind, no row, nothing checks it")
    for path, metric in unknown_metric:
        print(f"     {path}: {metric}")
    failures += len(unknown_metric)

    print(f"\n── templates still declaring point_in_time ({len(stale_pit)}) "
          f"— kind is the vocabulary's to say, so both columns agree")
    for path in stale_pit:
        print(f"     {path}")
    failures += len(stale_pit)

    orphan_tmpl = sorted(V.by_template() - supplied)
    print(f"\n── metrics declared by_template that no template supplies "
          f"({len(orphan_tmpl)}) — nothing can ever resolve them")
    for k in orphan_tmpl:
        print(f"     {k}")
    failures += len(orphan_tmpl)

    unnoted = sorted(k for k, m in V.VOCAB.items() if not m.row and not m.note)
    print(f"\n── metrics with no UI row and no note saying why ({len(unnoted)})")
    for k in unnoted:
        print(f"     {k}")
    failures += len(unnoted)

    # ── informational ───────────────────────────────────────────────────────
    known = metrics | _binding_metrics()
    orphan = sorted(k for k in ui_keys if k not in known)
    print(f"\n── UI rows the vocabulary does not name ({len(orphan)}) "
          f"— derived rows; not a failure, but each is a row no concept backs")
    for k in orphan:
        print(f"     {k}")

    asym = []
    for name, m in sorted(V.VOCAB.items()):
        for c in m.concepts:
            a, s = c.role(V.ANALYZE), c.role(V.SCREEN)
            if c.ns != V.GAAP or (a and s):
                continue    # ifrs/dei asymmetry is structural, not a choice
            asym.append(f"{name:<28} {c.tag:<62} "
                        f"{'analyze only (' + a + ')' if a else 'screen only (' + s + ')'}")
    print(f"\n── concepts one surface reads and the other does not ({len(asym)}) "
          f"— declared, not drift; review when it grows")
    for line in asym:
        print(f"     {line}")

    # Two metrics reading one concept only risks a double count where the SAME
    # surface reads both — CommercialPaper sits under current_debt and under
    # long_term_debt, but the analyze page reads only the first and the screen
    # only the second, so neither ever adds it to itself.
    shared = []
    for surface in (V.ANALYZE, V.SCREEN):
        dupes: dict[str, set] = {}
        for name, m in V.VOCAB.items():
            for c in m.concepts:
                if c.role(surface):
                    dupes.setdefault(c.tag, set()).add(name)
        shared += [(surface, t, ms) for t, ms in dupes.items() if len(ms) > 1]
    shared.sort()
    print(f"\n── concepts one surface reads under two metrics ({len(shared)}) "
          f"— legal, but each is a place two rows can double count")
    for surface, tag, ms in shared:
        print(f"     {surface:<8} {tag:<62} {', '.join(sorted(ms))}")

    for path in UNMIGRATED:
        try:
            src = open(path).read()
        except OSError:
            continue
        own = sorted(set(CONCEPT_SHAPED.findall(src)) - NOT_CONCEPTS)
        outside = [t for t in own if t not in concepts]
        print(f"\n── {path}: not migrated ({len(own)} concepts, {len(outside)} of them "
              f"in no shared metric) — see UNMIGRATED in this file for why")
        for t in outside:
            print(f"     {t}")

    print()
    if failures:
        print(f"FAIL — {failures} finding(s) above need fixing.")
    else:
        print("CLEAN — every consumer reads the shared vocabulary.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
