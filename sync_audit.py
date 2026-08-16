"""
Do the annual and quarterly passes apply the same rules?

Read-only. Nothing here changes behaviour.

The two passes read one vocabulary through two loops about 2,400 lines apart
in app.py, and nothing ties them together. Three times in one session a role
was wired into the annual loop and not the quarterly one, and each time the
symptom was the same: a correct annual column directly above a wrong quarterly
one, which is worse than both being wrong, because the correct year lends the
quarter credibility.

  SeniorNotes gap-fill   VeriSign's annual debt resolved, its quarters stayed
                         blank, and quarterly total debt and net cash with them.
  debt components        Collegium's year read $809m and its quarters carried
                         only the $55m current portion, so quarterly net cash
                         read POSITIVE $74m for a company a billion in debt.

Two checks, because they fail at different times.

  wiring   a role dict the annual pass consumes that the quarterly pass does
           not. Source-level, so it fails the moment the second loop is
           forgotten rather than when a filer that needs it turns up. This is
           the check that would have caught all three.

  live     a metric declared quarterly that resolves annually for a filer and
           produces nothing at all for any quarter. Weaker -- a filer can
           legitimately report a line annually and not quarterly, which is why
           the roster is small, named, and each entry says what it exercises.
           Treated as a report, not a failure.

Usage:  python3 sync_audit.py      (exit 0 = wiring in sync)
"""
from __future__ import annotations

import re
import sys

import vocabulary as V

# The role dictionaries app.py builds from the vocabulary. Each must be applied
# in both passes; add a row here whenever a new role is introduced.
ROLE_DICTS = ("GAP_FILL_TAGS", "COMPONENT_TAGS")

# Filers chosen for what they exercise, not for coverage. Each needs a path
# that the plain series list does not reach.
ROSTER = {
    "COLL": "debt components summed — tags no total debt concept at all",
    "VRSN": "senior-notes gap-fill — its only borrowing, no total tagged",
    "HCC":  "held-to-maturity short-term investments",
    "BDN":  "control: resolves from the series list, should be unaffected",
}


def _classify(src: str, name: str) -> dict:
    """{'annual': n, 'quarterly': n} for one role dict's application sites.

    Classified by what each loop BODY calls, not by where it sits in the file.
    Splitting on position was tried and is wrong: both application loops live
    below the quarterly extractor's own definition, so a positional split put
    the annual ones in the quarterly half and reported every role as missing
    from a pass it was actually in.
    """
    counts = {"annual": 0, "quarterly": 0}
    for m in re.finditer(rf"for _metric, _\w+ in {name}\.items\(\):", src):
        body = src[m.end():m.end() + 900]
        if "extract_post_annual_quarters" in body:
            counts["quarterly"] += 1
        elif "extract_point_in_time_series" in body or "extract_annual_series" in body:
            counts["annual"] += 1
    return counts


def check_wiring() -> list:
    src = open("app.py").read()
    bad = []
    for name in ROLE_DICTS:
        c = _classify(src, name)
        if c["annual"] and not c["quarterly"]:
            bad.append((name, "applied in the annual pass, never in the quarterly one"))
        elif c["quarterly"] and not c["annual"]:
            bad.append((name, "applied in the quarterly pass, never in the annual one"))
        elif not c["annual"] and not c["quarterly"]:
            bad.append((name, "declared but applied in neither pass"))
    return bad


def check_live(client) -> list:
    out = []
    roles = {}
    for name, fn in (("gap-fill", V.gap_fill_tags), ("component", V.component_tags)):
        for metric in fn(V.ANALYZE):
            roles.setdefault(metric, []).append(name)

    for ticker, why in ROSTER.items():
        d = client.get(f"/api/analyze?ticker={ticker}").get_json()
        if not d or d.get("error"):
            out.append((ticker, "-", "no data"))
            continue
        fin = d.get("financials") or {}
        if not (d.get("quarter_dates") or {}):
            continue                       # nothing to compare against
        for metric, kinds in roles.items():
            m = V.VOCAB.get(metric)
            if not m or not m.quarterly:
                continue
            series = fin.get(metric) or {}
            annual = [v for k, v in series.items()
                      if not str(k).startswith("Q") and v is not None]
            qtr = [v for k, v in series.items()
                   if str(k).startswith("Q") and v is not None]
            if annual and not qtr:
                out.append((ticker, metric,
                            f"{len(annual)} annual values, no quarter "
                            f"({'/'.join(kinds)}) — {why}"))
    return out


def main() -> int:
    print(f"{len(ROLE_DICTS)} role dicts | {len(ROSTER)} filers on the roster\n")

    bad = check_wiring()
    print(f"── roles applied in one pass and not the other ({len(bad)})")
    for name, why in bad:
        print(f"     {name}: {why}")

    import app
    live = check_live(app.app.test_client())
    print(f"\n── resolves annually, nothing quarterly ({len(live)})")
    for ticker, metric, why in live:
        print(f"     {ticker:6} {metric:24} {why}")
    if live:
        print("     (report, not a failure — a filer may legitimately report a"
              " line only annually)")

    print(f"\n{'CLEAN — both passes apply the same roles' if not bad else 'PASSES OUT OF SYNC'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
