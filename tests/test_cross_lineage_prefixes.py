"""cross_lineage_report.py counts as Gemini only the caches no other lineage wrote."""
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "medqa"))

import cross_lineage_report as clr  # noqa: E402

def test_no_second_lineage_cache_is_counted_as_gemini():
    model_slugs = {os.path.basename(d) for d in glob.glob(str(ROOT / "experiments/*/results/*")) if os.path.isdir(d)
                   and any(t in os.path.basename(d) for t in ("gpt-oss", "nemotron", "Qwen", "llama", "deepseek"))}
    for f in glob.glob(str(ROOT / clr.GEMINI_DIR / "*cache*.jsonl")):
        name = os.path.basename(f)
        if any(name.startswith(s + "_") for s in model_slugs):
            assert name.startswith(clr.OTHER_LINEAGE_PREFIXES), f"{name} would be counted as a Gemini cache"
