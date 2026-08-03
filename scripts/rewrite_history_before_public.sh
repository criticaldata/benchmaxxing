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
# Run this on a fresh mirror clone, with every collaborator's work merged, immediately before the
# visibility flip. It rewrites every commit that ever touched the nine paths and therefore changes
# every SHA from the first such commit onward. Coordinate: open PRs must be merged or rebased after.
#
# Requires git-filter-repo (`pip install git-filter-repo`); git filter-branch is not a substitute.
set -euo pipefail

MIRROR=${1:-}
if [[ -z "$MIRROR" ]]; then
  echo "usage: $0 <path-for-fresh-mirror-clone>" >&2
  exit 2
fi
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo not found: pip install git-filter-repo" >&2
  exit 2
fi

REMOTE=https://github.com/criticaldata/benchmaxxing.git
LANE=experiments/mimic_cxr_text/results

# 1. Fresh mirror. Never rewrite a working clone: reflogs and stashes keep the old objects alive.
git clone --mirror "$REMOTE" "$MIRROR"
cd "$MIRROR"

# 2. Drop every historical version of the ten result files. Note the tenth,
#    referee_deployable_falsifiability_check.json: it is not in `main` at all. It lives only on the
#    unmerged branch `docs/referee-tautology-correction` (commit d5ac221, #390) and carries 5 real
#    ids. A mirror clone fetches every ref, so scoping the rewrite to what `main` tracks would have
#    left it behind. This is why the verification in step 4 walks all refs rather than the tip tree.
#    Blob-level redaction is not enough:
#    the ids are the row keys, so a redacted row is not the row the paper computed from, and a
#    partial scrub leaves the index map reconstructible from whatever survives. The tip versions
#    come back in step 3 from the de-identified commit, which is the only version worth keeping.
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

# 3. Restore only the de-identified versions, as one commit with no ancestry to the raw blobs.
git remote add upstream "$REMOTE"
git fetch upstream main
git checkout main
git checkout upstream/main -- "$LANE"
git commit -m "chore: restore the de-identified MIMIC-CXR text rows after the history rewrite"

# 4. Verify against the rewritten history, which is the check that actually matters. Every blob
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
  | sort -u > /tmp/mimic-ids-in-history.txt || true

# The synthetic fixtures in the adapter and index tests are id-shaped but not real cases, each
# verified absent from the 633-id pre-migration union. They are expected to survive.
grep -a -vxE 'mimic-cxr-text-(10000000-50000000|10000001-50000001|10000032-50000001|10009999-50000009)' \
  /tmp/mimic-ids-in-history.txt > /tmp/mimic-ids-real.txt || true

if [[ -s /tmp/mimic-ids-real.txt ]]; then
  echo "FAIL: $(wc -l < /tmp/mimic-ids-real.txt) real MIMIC identifier(s) still in history:" >&2
  head -5 /tmp/mimic-ids-real.txt >&2
  echo "Do not flip visibility." >&2
  exit 1
fi
echo "OK: no real MIMIC identifiers in any blob reachable from any ref."

# 5. The tree-level guard must also pass on the rewritten tree.
python3 -m pytest tests/test_no_credentialed_identifiers.py -q

cat <<'EOF'

Rewritten and verified. Remaining manual steps, in order, none of them skippable:

  1. git push --force --mirror   (rewrites every ref on the remote)
  2. Ask GitHub Support to garbage-collect unreachable objects. Force-pushing does NOT delete
     them: an old SHA stays fetchable at /commit/<sha> until GitHub GCs, and forks keep their
     own copies. This is a support ticket, not a self-service button.
  3. Delete every fork, and confirm no fork network member retains the pre-rewrite objects.
  4. Invalidate any cached mirror: CI caches, Zenodo or archive snapshots, local clones held by
     collaborators. Tell collaborators to re-clone rather than pull.
  5. Only then flip visibility to public.

EOF
