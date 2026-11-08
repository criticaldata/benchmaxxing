"""Where do a reasoning model's empty answers fall: on bare prompts or on seeded ones?

A completion that spends its whole budget in the hidden channel is scored as no answer, which can
only lower an adoption rate. If those rows sit mostly in seeded conditions (the model reasons longer
when a cue is present) the committed adoption rates are biased downward in exactly the conditions
the experiments measure. The call caches store prompt hashes, not prompts, so the split is recovered
by replaying every MedQA runner keyless, in process, with the shared Cache instrumented: each
completion is classified from its prompt (a bare MCQ prompt starts at ``Question:`` and carries no
board; anything else is seeded) and tallied per runner and condition. No API call is made; a cache
miss aborts the runner and is reported, never filled.

Writes ``empty_content_by_condition.json`` under the model's results directory; output of each
runner goes to a scratch directory so no committed file moves.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNNERS = [
    "text_cue_types", "dose_response", "authority_ladder", "test_awareness", "committee_size_sweep",
    "plausible_distractor", "attributed_tier", "deliberation_framing", "rationale_validity",
    "seed_confidence", "paraphrase_robustness", "contamination_cascade", "leader_as_auditor",
    "hierarchy_dominance", "hierarchy_temp", "majority_pressure", "orchestrator_failure",
    "seed_timing", "true_peer_control", "unanimity_break",
]
# clean_a, push_c, break_it take the hard-case arguments and scale_c a different n; they are left out
# rather than replayed with the wrong flags.
_BARE = re.compile(r"\AQuestion: [^\n]*(?:\n(?!\n)[^\n]*)*\n\nOptions:\n(?:[A-Z]\. [^\n]*\n)+\nAnswer with only the single letter")


def condition(prompt: str) -> str:
    return "bare" if _BARE.match(prompt) else "seeded"


def main():
    ap = argparse.ArgumentParser(description="Empty answers by prompt condition, from a keyless replay.")
    ap.add_argument("--manifest", required=True)
    _lane.add_model_arg(ap)
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--runners", default=",".join(RUNNERS))
    args = ap.parse_args()
    model = args.model
    out_dir, _ = _lane.scoped(model, args.out, "experiments/medqa/results/call_cache.jsonl", None)

    tally = defaultdict(lambda: {"bare": [0, 0], "seeded": [0, 0]})
    current = {"runner": None}
    orig = _lane.Cache.complete

    def instrumented(self, prompt, model=None):
        resp = orig(self, prompt, model)
        c = tally[current["runner"]][condition(prompt)]
        c[0] += 1
        c[1] += int((resp or "") == "")
        return resp

    _lane.Cache.complete = instrumented
    _lane.key_for = lambda m: None  # a miss must abort, never call out
    status = {}
    scratch = Path(tempfile.mkdtemp(prefix="empty_by_cond_"))
    for name in [r for r in args.runners.split(",") if r]:
        path = HERE / f"{name}.py"
        if not path.exists():
            status[name] = "no such runner"
            continue
        current["runner"] = name
        spec = importlib.util.spec_from_file_location(f"runner_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [str(path), "--manifest", args.manifest, "--model", model, "--out", str(scratch / name)]
        (scratch / name).mkdir(parents=True, exist_ok=True)
        try:
            spec.loader.exec_module(mod)
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main()
            status[name] = "replayed"
        except SystemExit as e:
            status[name] = f"aborted: {str(e)[:80]}"
        except Exception as e:  # noqa: BLE001
            status[name] = f"error: {type(e).__name__}: {str(e)[:80]}"

    per_runner = {}
    tot = {"bare": [0, 0], "seeded": [0, 0]}
    for name, c in tally.items():
        per_runner[name] = {k: {"calls": v[0], "empty": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None}
                            for k, v in c.items()}
        for k in tot:
            tot[k][0] += c[k][0]; tot[k][1] += c[k][1]
    overall = {k: {"calls": v[0], "empty": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None} for k, v in tot.items()}
    # Fisher exact on the 2x2 (bare vs seeded x empty vs not)
    from benchmaxxing.stats import fisher_exact  # noqa: E402
    try:
        p = fisher_exact([[tot["bare"][1], tot["bare"][0] - tot["bare"][1]],
                          [tot["seeded"][1], tot["seeded"][0] - tot["seeded"][1]]]).pvalue
    except Exception:  # noqa: BLE001
        from scipy.stats import fisher_exact as fe
        p = fe([[tot["bare"][1], tot["bare"][0] - tot["bare"][1]],
                [tot["seeded"][1], tot["seeded"][0] - tot["seeded"][1]]])[1]
    report = {
        "model": model, "new_api_calls_this_run": 0,
        "definition": "A completion is bare when its prompt is the plain MCQ with no board or preamble; "
                      "everything else is seeded. Empty means the stored completion is the empty string.",
        "overall": overall, "fisher_bare_vs_seeded": round(float(p), 6),
        "per_runner": per_runner, "runner_status": status,
        "read": (f"Empty answers on bare prompts {overall['bare']['empty']}/{overall['bare']['calls']} "
                 f"({overall['bare']['rate']}) against seeded prompts {overall['seeded']['empty']}/"
                 f"{overall['seeded']['calls']} ({overall['seeded']['rate']}), Fisher p={round(float(p), 4)}. "
                 "A seeded rate above the bare one means the committed adoption rates are biased downward "
                 "in the seeded conditions by the no-answer rows."),
    }
    (out_dir / "empty_content_by_condition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_runner"}, indent=2))


if __name__ == "__main__":
    main()
