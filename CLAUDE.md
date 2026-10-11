# InvestmentAnalysis

## Git workflow

**Other people push to `main`. Always start from the current remote tip, never
from whatever the container cloned.** A cloud session's checkout can be hours
or days stale, and a branch whose PR was already merged must be restarted from
the current tip rather than extended.

At the start of any work, and again immediately before pushing:

```bash
git fetch origin main
git log --oneline -8 origin/main      # what landed since last time
```

Work lands on `main` directly, fast-forward, no merge commit:

```bash
git checkout -B <work-branch> origin/main    # develop here
# ... commit ...
git fetch origin main && git rebase origin/main
git checkout main && git merge --ff-only <work-branch>
git push origin main
```

Rebase the work onto the refetched tip; do not merge `main` into the work
branch. History stays linear and each change stays reviewable on its own.

If a push is rejected as non-fast-forward, someone pushed while you worked:
refetch, rebase, **re-run the verification that justified the change**, then
push. Never force-push `main`.

A textual rebase succeeding is not evidence the change is still correct. The
dangerous collision in this repo is semantic: a parser or vocabulary change
plus someone else's new binding merges cleanly and silently moves numbers. So
after rebasing onto commits you have not seen, look at what they added and
re-verify against it.

Note: this environment's git proxy allows pushes but refuses ref deletions
(403). Deleting a remote branch has to be done from GitHub.

## Verifying a change before it lands

This app's output is numbers a reader will trust, so a change ships with
evidence, not a plausible diff.

- Snapshot the affected metrics before and after, and diff them. Flask's test
  client against `/api/analyze?ticker=X` and `/api/scorecard?ticker=X` is
  enough; keep the scripts in the scratchpad.
- Include control filers that should **not** move, and report how many cells
  stayed put.
- When touching `statement_store.py`, `scorecard.py` or `vocabulary.py`, check
  every filer with an authored binding (`bindings/companies/*.json`). A parser
  change can silently re-point a binding at a different row — that is how
  Lumen's intangibles lost $3.9B in a change whose diff looked unrelated.
- Reconcile any new figure against the company's own printed subtotal.
- Run the read-only audits: `tag_audit.py`, `sync_audit.py`, `derived_audit.py`
  (each prints CLEAN or names what broke). They are the cheapest shared gate
  there is, and there is no CI yet.
- Prefer a blank cell to a number you cannot source. A segment column standing
  in for a consolidated total is the failure mode this codebase works hardest
  to avoid.

Record in the commit body which filers were verified and which did not move.
With several people working on the same rows, the commit log is the only shared
memory of what was checked.

`statement_store.py` caches parses, not filings: bump `_PARSE_VERSION` with any
change to `parse_report` that alters the rows it emits, or the cache keeps
serving the old reading.

## Setup

```bash
pip install -q --ignore-installed blinker -r requirements.txt
```

`--ignore-installed blinker` is required: the Debian-packaged blinker has no
RECORD file and a plain install aborts on it.
