"""Repo-wide screen for reported metrics that cannot fail by construction (#374).

#374 collected 17 confirmed instances of the same defect: a number that reads like evidence but
whose predicate could not have come out otherwise. Predicates that reduce to the label they are
scored against, controls that are unsatisfiable, recalls forced to 1.0, comparators that are
structurally zero, hardcoded significance verdicts. Every one of them reached the paper because
nothing looked for them.

This module is what looks for them. It reads the committed artifacts, not the prose about them, so
a comment cannot satisfy it. Three screens:

`constant_columns`
    Every committed per-case file under `experiments/*/results/**` (`.jsonl`, and the de-identified
    `deid/*.csv` exports). For every binary column, i.e. one whose values all live in
    {0, 1, true, false}, flag it when it is constant across every row. A scored column that never
    varies is either definitional or a saturated arm, and in both cases the rate computed from it
    is not evidence until someone says which.

`hardcoded_verdicts`
    Python under `experiments/` that embeds a significance verdict as a literal string in the same
    f-string or `.format` call that interpolates a p-value. The verdict is then fixed at authoring
    time while looking like it was read off the test.

`rounded_pvalues`
    Committed summary JSON carrying a p-value field of exactly 0.0 or exactly 1.0. An exact test
    cannot return 0, so 0.0 is a rounded or underflowed asymptotic p printed as if exact. Exactly
    1.0 is usually an empty contingency table. Two benign cases are properties of the method rather
    than of any one artifact, so they are computed away instead of allowlisted:

    - A two-sided exact binomial on b successes out of b+c returns exactly 1.0 whenever b+c > 0 and
      |b-c| <= 1, because then no outcome is less likely than the observed one. When a flagged
      field sits beside its own discordant pair and that pair explains the 1.0, it is dropped.
    - Holm and Benjamini-Hochberg adjusted p-values are clamped to min(1, ...) by definition, so
      exactly 1.0 in an adjusted field carries no information about the table. `p_raw` beside it is
      still screened, which is where the real signal is.

WHAT THIS CANNOT CATCH, stated plainly so nobody reads a green suite as an all-clear:

- **Near-constant predicates.** A column that is true on 998 of 1000 rows is not constant and will
  not be flagged, even though a 99.8% rate is just as definitional in practice as a 100% one.
- **Algebraic reduction between two varying columns.** The original #374 instance, `deployable` in
  `referee_deployable.py`, is character-for-character equal to `adopted` on all 40 rows, and both
  vary. Column-wise constancy cannot see that. Neither can it see a predicate that reduces to the
  label through an intermediate variable.
- **Uncommitted artifacts.** The screen reads what is in git. The eight SUPPORT2 summary p-values
  named in #374 are not in the tree, so nothing here reports them.
- **Non-Python report generators**, and verdicts assembled by concatenating variables rather than
  by literal text.

Duplicate-column equality and a near-constant threshold are the obvious next screens; they need a
per-arm judgement about what saturation is legitimate, which is why they are not here yet.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXEMPTIONS_PATH = REPO_ROOT / "tests" / "degeneracy_exemptions.json"

# A per-case file is one where every record carries a case identifier. This is what separates a
# scored cohort from the two other things that live under results/: the call caches, whose records
# are {k, model, resp}, and the per-turn transcripts, where only the header row names a case. Both
# would otherwise contribute meaningless constant columns.
CASE_KEYS = ("case_id", "case_index", "case", "case_idx", "id")

# Values that make a column binary. CSV has no types, so the string spellings count too.
_TRUE = {"1", "true", "True", "TRUE"}
_FALSE = {"0", "false", "False", "FALSE"}

# Any of these in a literal string is a significance verdict.
VERDICT_PHRASES = (
    "NOT significant",
    "not significant",
    "significant at",
    "no effect",
    "is significant",
)

# What makes a field, or an interpolated expression, a p-value.
_P_FIELD = re.compile(r"^(p|p_?val(ue)?|.*_p|p_.*|.*_p_?val(ue)?)$", re.IGNORECASE)
_P_EXPR = re.compile(r"\bp_?val(ue)?\b|\bp_(raw|adj|bh|holm)\b|\b\w*_p\b|\bpvalue\b", re.IGNORECASE)

# Sibling field names holding the two discordant counts of a paired test.
_DISCORDANT_PAIRS = (("gain", "lose"), ("discordant_gain", "discordant_lose"), ("b", "c"))

# Multiplicity-adjusted p-values are clamped to 1.0 by definition, so 1.0 there is uninformative
# rather than degenerate. 0.0 in an adjusted field is still screened.
_ADJUSTED_P_FIELD = re.compile(
    r"^p_(bh|holm|adj|adjusted|bonf|bonferroni|fdr)$|^(bh|holm|adj|bonferroni|fdr)_p(val(ue)?)?$",
    re.IGNORECASE,
)

MIN_ROWS = 8  # below this, constancy is not informative enough to be worth a reviewer's time


@dataclass(frozen=True)
class Finding:
    """One suspected cannot-fail metric."""

    kind: str  # constant_column | hardcoded_verdict | rounded_pvalue | duplicate_column | forced_direction
    path: str  # repo-relative
    locus: str  # column name, function:phrase, or JSON path. Stable under line moves.
    detail: str  # human-readable evidence

    @property
    def key(self) -> str:
        """The identity used by the exemption file. Deliberately excludes `detail`."""
        return f"{self.kind}|{self.path}|{self.locus}"

    def __str__(self) -> str:
        return f"{self.key}  ({self.detail})"


# ----------------------------------------------------------------------------- file discovery


def _tracked(root: Path, pattern: str) -> list[Path]:
    """Committed files matching a git pathspec, or a glob when `root` is not a repo (tests)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", pattern],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [root / p for p in out.split("\0") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(root.glob(pattern.replace("**.", "**/*.")))


def per_case_files(root: Path) -> list[Path]:
    """Committed per-case artifacts: every results jsonl, plus the de-identified csv exports.

    Only `deid/` csv is included. The other committed csv under results/ are cohort manifests, where
    a constant label column is the point rather than a defect.
    """
    files = list(_tracked(root, "experiments/*/results/**.jsonl"))
    files += [p for p in _tracked(root, "experiments/*/results/**.csv") if "deid" in p.parts]
    return sorted(set(files))


# ----------------------------------------------------------------------------- screen 1


def _binary(values: list) -> list[bool] | None:
    """Normalise a column to bools, or None if it is not binary."""
    out = []
    for v in values:
        if isinstance(v, bool):
            out.append(v)
        elif isinstance(v, int):
            if v not in (0, 1):
                return None
            out.append(bool(v))
        elif isinstance(v, float):
            if v not in (0.0, 1.0):
                return None
            out.append(bool(v))
        elif isinstance(v, str):
            if v in _TRUE:
                out.append(True)
            elif v in _FALSE:
                out.append(False)
            else:
                return None
        else:
            return None
    return out


def _read_rows(path: Path) -> list[dict] | None:
    """Rows of a per-case file, or None when the file is not one."""
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
    else:
        rows = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if not isinstance(rec, dict):
                    return None
                rows.append(rec)
    if not rows:
        return None
    common = set(rows[0]).intersection(*(set(r) for r in rows[1:])) if len(rows) > 1 else set(rows[0])
    if not common.intersection(CASE_KEYS):
        return None  # a cache or a transcript, not a scored cohort
    return rows


def constant_columns(root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in per_case_files(root):
        rows = _read_rows(path)
        if rows is None or len(rows) < MIN_ROWS:
            continue
        common = sorted(set(rows[0]).intersection(*(set(r) for r in rows[1:])))
        for col in common:
            if col in CASE_KEYS:
                continue
            vals = _binary([r[col] for r in rows])
            if vals is None or len(set(vals)) != 1:
                continue
            findings.append(
                Finding(
                    "constant_column",
                    str(path.relative_to(root)),
                    col,
                    f"binary column constant at {vals[0]} across all {len(rows)} rows",
                )
            )
    return findings


# ----------------------------------------------------------------------------- screen 2


def _enclosing(tree: ast.AST) -> dict[int, str]:
    """Map every node id to the name of the function it sits in, for a line-stable locus."""
    owner: dict[int, str] = {}

    def walk(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            here = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else name
            owner[id(child)] = here
            walk(child, here)

    owner[id(tree)] = "<module>"
    walk(tree, "<module>")
    return owner


def _verdict_in(text: str) -> str | None:
    for phrase in VERDICT_PHRASES:
        if phrase in text:
            return phrase
    return None


def hardcoded_verdicts(root: Path = REPO_ROOT) -> list[Finding]:
    """A verdict literal in the same interpolation that reports a p-value.

    Only interpolating expressions count, so a docstring or comment stating a past finding is not
    flagged: it interpolates nothing and therefore claims nothing about a value computed here.
    """
    findings: list[Finding] = []
    for path in _tracked(root, "experiments/**.py"):
        src = path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        owner = _enclosing(tree)
        for node in ast.walk(tree):
            literal, exprs = "", []
            if isinstance(node, ast.JoinedStr):
                # Implicit concatenation of adjacent literals collapses into one JoinedStr, so a
                # verdict in one fragment and the p-value in the next are correctly seen together.
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        literal += part.value
                    elif isinstance(part, ast.FormattedValue):
                        exprs.append(ast.get_source_segment(src, part.value) or "")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)
            ):
                literal = node.func.value.value
                exprs = [ast.get_source_segment(src, a) or "" for a in node.args]
                exprs += [ast.get_source_segment(src, k.value) or "" for k in node.keywords]
            elif (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mod)
                and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
            ):
                literal = node.left.value
                exprs = [ast.get_source_segment(src, node.right) or ""]
            else:
                continue

            phrase = _verdict_in(literal)
            if not phrase:
                continue
            # The interpolation must actually carry a p-value: either a named p expression, or a
            # bare `p=` / `p =` in the literal immediately preceding a placeholder.
            carries_p = any(_P_EXPR.search(e) for e in exprs) or bool(
                exprs and re.search(r"\bp\s*=\s*(\{|%)", literal)
            )
            if not carries_p:
                continue
            findings.append(
                Finding(
                    "hardcoded_verdict",
                    str(path.relative_to(root)),
                    f"{owner.get(id(node), '<module>')}:{phrase}",
                    f"line {node.lineno}: verdict {phrase!r} is a literal in the same "
                    f"interpolation that reports {', '.join(e for e in exprs if _P_EXPR.search(e)) or 'a p-value'}",
                )
            )
    return findings


# ----------------------------------------------------------------------------- screen 3


def _exact_binomial_p_is_one(b: int, c: int) -> bool:
    """Does a two-sided exact binomial on (b, c) legitimately return exactly 1.0?

    It does when the table is non-empty and the split is as even as it can be, since then no
    outcome is less likely than the observed one and the two-sided p sums to the whole mass.
    """
    return (b + c) > 0 and abs(b - c) <= 1


def _sibling_discordant(container) -> tuple[int, int] | None:
    if not isinstance(container, dict):
        return None
    for lo, hi in _DISCORDANT_PAIRS:
        if isinstance(container.get(lo), int) and isinstance(container.get(hi), int):
            return container[lo], container[hi]
    return None


def rounded_pvalues(root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in _tracked(root, "experiments/*/results/**.json"):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(root))

        def visit(node, trail: list[str], parent) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    visit(v, trail + [str(k)], node)
                return
            if isinstance(node, list):
                for i, v in enumerate(node):
                    visit(v, trail + [str(i)], node)
                return
            if not trail or isinstance(node, bool) or not isinstance(node, (int, float)):
                return
            if not _P_FIELD.match(trail[-1]) or node not in (0.0, 1.0):
                return
            pair = _sibling_discordant(parent)
            if node == 1.0:
                if _ADJUSTED_P_FIELD.match(trail[-1]):
                    return  # a multiplicity correction clamps to 1.0; nothing to learn from it
                if pair and _exact_binomial_p_is_one(*pair):
                    return  # a real exact 1.0 on a non-empty near-even table
                why = (
                    f"empty discordant table gain={pair[0]} lose={pair[1]}"
                    if pair
                    else "no discordant counts alongside it to explain the 1.0"
                )
                detail = f"p is exactly 1.0 with {why}"
            else:
                detail = "p is exactly 0.0, which no exact test returns; rounded or underflowed"
            findings.append(Finding("rounded_pvalue", rel, ".".join(trail), detail))

        visit(doc, [], None)
    return findings


# ----------------------------------------------------------------------------- screen 4


def duplicate_columns(root: Path = REPO_ROOT) -> list[Finding]:
    """Pairs of VARYING binary columns that are identical, or exact complements, on every row.

    This is the screen the constant-column pass cannot be: the defects that actually cost this
    project its headline referee numbers (#357, #366, #368, #390) all involved two columns that
    each vary, and are nonetheless the same predicate under two names. `deployable == adopted`
    is invisible to a constancy test precisely because adoption is a real, varying quantity.

    Complement is included because a comparator built as the negation of its own label is the
    same defect wearing a minus sign.

    Reported once per pair, on the lexicographically first column, so the key is stable.
    """
    findings: list[Finding] = []
    for path in per_case_files(root):
        rows = _read_rows(path)
        if rows is None or len(rows) < MIN_ROWS:
            continue
        common = sorted(set(rows[0]).intersection(*(set(r) for r in rows[1:])))
        cols: dict[str, list[bool]] = {}
        for col in common:
            if col in CASE_KEYS:
                continue
            vals = _binary([r[col] for r in rows])
            if vals is None or len(set(vals)) == 1:
                continue  # constant columns are screen 1's business, not ours
            cols[col] = vals
        names = sorted(cols)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                va, vb = cols[a], cols[b]
                if va == vb:
                    rel = "identical to"
                elif all(x != y for x, y in zip(va, vb)):
                    rel = "the exact complement of"
                else:
                    continue
                findings.append(
                    Finding(
                        "duplicate_column",
                        str(path.relative_to(root)),
                        f"{a} vs {b}",
                        f"{a} is {rel} {b} on all {len(rows)} rows, so scoring one against the "
                        f"other cannot fail",
                    )
                )
    return findings


# ----------------------------------------------------------------------------- screen 5


_RATE_FIELD = re.compile(r"(rate|adoption|adopt|contagion|precision|recall|fpr|accuracy)", re.I)


def _saturated_rate_beside(node: dict, parent) -> str | None:
    """A rate field pinned at exactly 0.0 or 1.0, in this container or the one above it."""
    for scope, where in ((node, "here"), (parent if isinstance(parent, dict) else {}, "alongside")):
        for k, v in scope.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if _RATE_FIELD.search(str(k)) and float(v) in (0.0, 1.0):
                return f"{k}={v} ({where})"
    return None


def forced_direction(root: Path = REPO_ROOT) -> list[Finding]:
    """A paired test whose discordant table permits only one direction.

    When `lose` is 0 and `gain` is not, McNemar can only ever point one way, so its p-value is a
    statement about the sample size rather than about the effect. #391 is exactly this: an
    isolated comparator that is 0 by construction forces `lose` to 0, and the resulting
    `p<1e-6` reads as evidence when at an adoption rate of 10/115 it would still return 0.002.

    The machinery is already here: `_sibling_discordant` is used by screen 3 to EXCULPATE a p of
    1.0 on a near-even table. This uses the same reading to FLAG the opposite case.
    """
    findings: list[Finding] = []
    for path in _tracked(root, "experiments/*/results/**.json"):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(root))

        def visit(node, trail: list[str], parent=None) -> None:
            if isinstance(node, list):
                for i, v in enumerate(node):
                    visit(v, trail + [str(i)], parent)
                return
            if not isinstance(node, dict):
                return
            pair = _sibling_discordant(node)
            if pair is not None:
                gain, lose = pair
                if (gain == 0) != (lose == 0):  # exactly one side empty
                    # An empirically one-sided table is a strong effect, not a defect: a real
                    # 71-to-0 split is the finding. What makes the direction STRUCTURAL is a
                    # saturated comparator beside it, a rate pinned at exactly 0 or 1, because
                    # then the zero side could not have been anything else. Require that, or the
                    # screen condemns its own best results.
                    sat = _saturated_rate_beside(node, parent)
                    if sat is not None:
                        zero, nonzero = ("lose", "gain") if lose == 0 else ("gain", "lose")
                        findings.append(
                            Finding(
                                "forced_direction",
                                rel,
                                ".".join(trail) or "<root>",
                                f"{zero}=0 with {nonzero}={max(gain, lose)} beside {sat}, so the "
                                f"zero side is pinned by a saturated comparator rather than "
                                f"observed, and the p measures n rather than the effect",
                            )
                        )
            for k, v in node.items():
                visit(v, trail + [str(k)], node)

        visit(doc, [], None)
    return findings


# ----------------------------------------------------------------------------- driver


def scan(root: Path = REPO_ROOT) -> list[Finding]:
    return sorted(
        constant_columns(root)
        + hardcoded_verdicts(root)
        + rounded_pvalues(root)
        + duplicate_columns(root)
        + forced_direction(root),
        key=lambda f: f.key,
    )


def load_exemptions(path: Path = EXEMPTIONS_PATH) -> tuple[dict[str, str], dict[str, str]]:
    """(allowlist, preexisting). Both map a finding key to a written reason."""
    if not path.exists():
        return {}, {}
    doc = json.loads(path.read_text())
    return dict(doc.get("allowlist", {})), dict(doc.get("preexisting", {}))


def unexplained(root: Path = REPO_ROOT, exemptions: Path = EXEMPTIONS_PATH) -> list[Finding]:
    allow, pre = load_exemptions(exemptions)
    return [f for f in scan(root) if f.key not in allow and f.key not in pre]


def main() -> int:
    import time

    t0 = time.perf_counter()
    found = scan()
    allow, pre = load_exemptions()
    elapsed = time.perf_counter() - t0
    for f in found:
        tag = "ALLOWED" if f.key in allow else ("pre-existing" if f.key in pre else "NEW")
        print(f"[{tag:12}] {f}")
    new = [f for f in found if f.key not in allow and f.key not in pre]
    print(
        f"\n{len(found)} findings, {len(new)} unexplained, "
        f"{len(allow)} allowlisted, {len(pre)} pre-existing, in {elapsed:.2f}s"
    )
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
