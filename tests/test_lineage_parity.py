"""Parity of a second lineage with the sibling lineages, as a test rather than an audit.

The owner's rule for a lineage PR is apples to apples: every text-lane result file any other
lineage committed must exist for this lineage too, at the same path, so the cross-lineage
comparisons are over the same arms. Imaging lanes are excluded for a text-only model. One naming
divergence is declared rather than hidden: the Qwen lineage keeps its 600-case MIMIC solo as
results/<model>/solo_results_600.json while Gemini and gpt-oss keep it under results_n600/<model>/.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = "nvidia_nemotron-3-super-120b-a12b"
TEXT_ONLY = {LINEAGE}
IMAGING_LANES = {"imaging", "imaging_chexpert", "chexpert", "mimic_cxr_image"}
# sibling path -> the path this lane uses for the same artifact
ALIASES = {
    "results/{slug}/solo_results_600.json": "results_n600/{slug}/solo_results.json",
}
PAT = re.compile(r"experiments/([a-z_0-9]+)/(results[^/]*)/([^/]+)/([^/]+\.json)$")


def _result_files():
    out = {}
    for p in ROOT.glob("experiments/*/results*/*/*.json"):
        m = PAT.match(p.relative_to(ROOT).as_posix())
        if not m or "cache" in m.group(4):
            continue
        lane, res, slug, name = m.groups()
        out.setdefault(slug, set()).add((lane, res, name))
    return out


def _siblings(files):
    return {s for s in files if s != LINEAGE and s not in ("n100",)}


@pytest.mark.parametrize("lineage", sorted(TEXT_ONLY))
def test_the_lineage_has_every_text_lane_file_a_sibling_has(lineage):
    files = _result_files()
    mine = files.get(lineage, set())
    missing = []
    for sib in _siblings(files):
        for lane, res, name in files[sib]:
            if lane in IMAGING_LANES:
                continue
            want = (lane, res, name)
            alias = ALIASES.get(f"{res}/{{slug}}/{name}")
            if alias:
                ares, aname = alias.format(slug=lineage).split("/")[0], alias.split("/")[-1]
                want = (lane, ares, aname)
            if want not in mine:
                missing.append(f"{lane}/{res}/{sib}/{name}")
    assert not missing, "sibling result files with no counterpart for " + lineage + ":\n  " + "\n  ".join(sorted(missing))
