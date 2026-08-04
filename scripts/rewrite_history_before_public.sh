#!/usr/bin/env bash
# Remove MIMIC-CXR subject and study identifiers from the whole of git history, not just the tip.
#
# Why this exists. De-identifying the working tree does not de-identify the repository. Until this
# runs, `git show <pre-migration>:experiments/mimic_cxr_text/results/solo_records.jsonl` yields real
# MIMIC ids, 633 distinct subject/study pairs across the nine result files. The exposure is worse
# than a plain leak: `case_index` is a deterministic rank over the sorted union of those same 633
# ids, so anyone with history access can rebuild the map and re-link every committed row. Flipping
# a repository public is a one-way door; assume every object is cloned and cached the moment it is.
#
# Run this on a fresh clone, with every collaborator's work merged, immediately before the
# visibility flip. It rewrites every commit that ever touched the ten paths and therefore changes
# every SHA from the first such commit onward. Coordinate: open PRs must be merged or rebased after.
#
# Requires git-filter-repo (`pip install git-filter-repo`); git filter-branch is not a substitute.
set -euo pipefail

WORKDIR=${1:-}
if [[ -z "$WORKDIR" ]]; then
  echo "usage: $0 <path-for-fresh-clone>" >&2
  exit 2
fi
if [[ -e "$WORKDIR" ]]; then
  echo "refusing to reuse an existing path: $WORKDIR" >&2
  exit 2
fi
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo not found: pip install git-filter-repo" >&2
  exit 2
fi

REMOTE=https://github.com/criticaldata/benchmaxxing.git
LANE=experiments/mimic_cxr_text/results
STASH=$(mktemp -d)
trap 'rm -rf "$STASH"' EXIT

# 1. Fresh clone, with a work tree.
#
#    NOT `--mirror`. A mirror is bare, so step 3 cannot check out or commit the de-identified rows
#    back; both `git checkout` and `git commit` exit 128 with "this operation must be run in a work
#    tree", and under `set -euo pipefail` that aborts the run before the step 5 scan and step 6
#    pytest ever execute. A mirror's other property, fetching every ref rather than just the default
#    branch, is what reaches the tenth file on the unmerged #390 branch, so that is restored
#    explicitly below rather than inherited from `--mirror`.
#
#    Never rewrite a clone you work in: reflogs and stashes keep the old objects alive.
git clone --no-local "$REMOTE" "$WORKDIR"
cd "$WORKDIR"

# Fetch every remote branch as a local branch, so filter-repo rewrites all of them. This replaces
# what `--mirror` gave us, without the bare-repo problem. Tags come along with the clone.
git fetch --no-tags origin '+refs/heads/*:refs/heads/*' --update-head-ok

# 2. Stash the de-identified rows BEFORE the rewrite.
#
#    These are restored in step 4. The old script re-fetched them from the remote instead, which
#    silently undid the rewrite: `git fetch upstream main` writes `refs/remotes/upstream/main`
#    pointing at the un-rewritten history, so the raw blobs come straight back. The step 5 scan
#    walks `--all`, which includes remote-tracking refs, so it would fail on ids the script itself
#    reintroduced; and `git push --force --mirror` pushes `refs/remotes/*` too, carrying those blobs
#    back to the remote. A tarball has no ancestry and no refs, so it cannot resurrect anything.
tar -cf "$STASH/deid-rows.tar" "$LANE"

# 3. Drop every historical version of the ten result files. Note the tenth,
#    referee_deployable_falsifiability_check.json: it is not in `main` at all. It lives only on the
#    unmerged branch `docs/referee-tautology-correction` (commit d5ac221, #390) and carries 5 real
#    ids, which is why step 1 materialises every branch and step 5 walks all refs.
#    Blob-level redaction is not enough: the ids are the row keys, so a redacted row is not the row
#    the paper computed from, and a partial scrub leaves the index map reconstructible from whatever
#    survives. The tip versions come back in step 4 from the stash, which is the only version worth
#    keeping.
git filter-repo --force \
  --path "$LANE/blind_metric.jsonl" \
  --path "$LANE/break_it_a_per_case.jsonl" \
  --path "$LANE/break_it_d_per_case.jsonl" \
  --path "$LANE/cascade_results.json" \
  --path "$LANE/deliberation_framing.jsonl" \
  --path "$LANE/push_c_per_case.jsonl" \
  --path "$LANE/referee_deployable.jsonl" \
  --path "$LANE/referee_judge.jsonl" \
  --path "$LANE/solo_records.jsonl" \
  --path "$LANE/referee_deployable_falsifiability_check.json" \
  --invert-paths

# 4. Restore only the de-identified versions, from the stash, as one commit with no ancestry to the
#    raw blobs. filter-repo removes `origin` itself, so there is nothing to re-fetch from even by
#    accident; the guard below is belt-and-braces for anyone editing this script later.
git checkout main
tar -xf "$STASH/deid-rows.tar"
git add "$LANE"
git commit -m "chore: restore the de-identified MIMIC-CXR text rows after the history rewrite"

# Any remote or remote-tracking ref at this point points at un-rewritten history and would both
# fail the scan and get pushed by `--mirror`. Remove them before scanning, so what step 5 covers is
# exactly what step 7 pushes.
for r in $(git remote); do git remote remove "$r"; done
git for-each-ref --format='%(refname)' refs/remotes/ \
  | while read -r ref; do git update-ref -d "$ref"; done
# Old objects stay reachable through the reflog and filter-repo's replace refs until expired.
git for-each-ref --format='%(refname)' refs/replace/ \
  | while read -r ref; do git update-ref -d "$ref"; done
git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now --aggressive --quiet

# 5. Verify against the rewritten history, which is the check that actually matters. Every blob
#    reachable from every ref, not just the tree at the tip.
#
#    `grep -a` is load-bearing. The concatenated blob stream contains the caches, so plain grep
#    decides it is binary and prints "binary file matches" to stdout instead of the matches, which
#    a `grep -q`/pipeline reads as clean. That false pass is exactly the failure mode this whole
#    script exists to prevent, so treat the stream as text explicitly.
echo "== scanning every blob in rewritten history for MIMIC identifiers =="
git rev-list --objects --all \
  | awk '{print $1}' \
  | git cat-file --batch-check='%(objectname) %(objecttype)' \
  | awk '$2=="blob" {print $1}' \
  | git cat-file --batch \
  | grep -a -oE 'mimic-cxr-text-[0-9]{8}-[0-9]{8}' \
  | sort -u > "$STASH/ids-in-history.txt" || true

# The synthetic fixtures in the adapter and index tests are id-shaped but not real cases, each
# verified absent from the 633-id pre-migration union. They are expected to survive. Read the set
# from the guard test rather than restating it here, so the two cannot drift apart.
python3 - "$STASH/ids-in-history.txt" "$STASH/ids-real.txt" <<'PY'
import pathlib, re, sys

guard = pathlib.Path("tests/test_no_credentialed_identifiers.py").read_text()
block = re.search(r"SYNTHETIC_FIXTURE_IDS\s*=\s*\{(.*?)\}", guard, re.S)
if not block:
    sys.exit("could not read SYNTHETIC_FIXTURE_IDS from the guard test")
allowed = set(re.findall(r'"([^"]+)"', block.group(1)))
if not allowed:
    sys.exit("SYNTHETIC_FIXTURE_IDS parsed empty; refusing to treat every id as allowed")

found = [ln for ln in pathlib.Path(sys.argv[1]).read_text().splitlines() if ln]
real = [i for i in found if i not in allowed]
pathlib.Path(sys.argv[2]).write_text("".join(f"{i}\n" for i in real))
PY

if [[ -s "$STASH/ids-real.txt" ]]; then
  echo "FAIL: $(wc -l < "$STASH/ids-real.txt") real MIMIC identifier(s) still in history:" >&2
  head -5 "$STASH/ids-real.txt" >&2
  echo "Do not flip visibility." >&2
  exit 1
fi
echo "OK: no real MIMIC identifiers in any blob reachable from any ref."

# 6. The tree-level guard must also pass on the rewritten tree.
python3 -m pytest tests/test_no_credentialed_identifiers.py -q

cat <<'EOF'

Rewritten and verified. Remaining manual steps, in order, none of them skippable:

  1. git push --force --mirror <remote>   (rewrites every ref on the remote; run it from this
     clone, whose remotes were deliberately removed above, so pass the URL explicitly)
  2. Ask GitHub Support to garbage-collect unreachable objects. Force-pushing does NOT delete
     them: an old SHA stays fetchable at /commit/<sha> until GitHub GCs, and forks keep their
     own copies. This is a support ticket, not a self-service button.
  3. Delete every fork, and confirm no fork network member retains the pre-rewrite objects.
  4. Invalidate any cached mirror: CI caches, Zenodo or archive snapshots, local clones held by
     collaborators. Tell collaborators to re-clone rather than pull.
  5. Only then flip visibility to public.

EOF
