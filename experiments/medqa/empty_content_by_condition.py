"""Where do a reasoning model's empty answers fall: on bare prompts or on seeded ones?

A completion that spends its whole budget in the hidden channel is scored as no answer, which can
only lower an adoption rate. If those rows sit mostly in seeded conditions (the model reasons longer
when a cue is present) the committed adoption rates are biased downward in exactly the conditions
the experiments measure. The call caches store prompt hashes, not prompts, so the split is recovered
by replaying the MedQA runners in ``RUNNERS`` keyless, in process, with their caches instrumented:
each completion is classified from its prompt and tallied per runner and condition. Both cache
classes are instrumented, the shared ``_lane.Cache`` and the ``_Cache`` the committee runners define
for themselves, so a runner cannot replay without being counted. No API call is made; a cache miss
aborts the runner and is reported, never filled.

Definition of bare. A prompt is bare when it is exactly the plain multiple-choice rendering of a
manifest case, ``Question: <stem>``, a blank line, ``Options:``, one ``<letter>. <option>`` entry per
option, a blank line, and the instruction ``Answer with only the single letter of the best option.``,
with nothing before the stem and nothing between the options and the instruction. The stem may run
over several paragraphs and an option over several lines, so the test is equality with the rendered
case, not a pattern on the text. Every other prompt is seeded: a preamble or a board is present, the
question is reworded, or the instruction differs (this includes the open-ended recall prompt of
``contamination_cascade``, which has no options).

A count is one completion requested by a runner, so a prompt a runner asks several times is counted
each time. The runners call the cache from worker threads; the tally is updated under a lock.

Writes ``empty_content_by_condition.json`` under the model's results directory; output of each
runner goes to a scratch directory so no committed file moves.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
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
# Runners with a call cache that are not in RUNNERS, and why; written to the output as ``not_replayed``.
NOT_REPLAYED = {
    "clean_a": "needs --solo-records (the hard-case arguments); not replayed with the wrong flags",
    "push_c": "needs --solo-records (the hard-case arguments); not replayed with the wrong flags",
    "break_it": "needs --solo-records (the hard-case arguments); not replayed with the wrong flags",
    "scale_c": "runs at its own n; not in the default list, can be added with --runners",
    "live_peer_organic": "not in the default list, can be added with --runners",
    "pre_emptive_referee": "not in the default list, can be added with --runners",
    "super_additivity": "not in the default list, can be added with --runners",
    "temperature_sensitivity": "sampled completions keyed on temperature and draw, through its own cache",
    "reasoning_effort_sensitivity": "completions keyed on reasoning effort, through its own cache",
    "deliberation_channel": "its own store, keyed on condition; not instrumented",
    "reproduce": "the base cascade, through its own backend cache; not instrumented",
}
_TAIL = "Answer with only the single letter of the best option."
# The committed outputs were computed with each runner at its default --n. Four runner defaults were
# later aligned with the MedQA lane's committed sizes; their earlier defaults are pinned here so this
# replay keeps measuring the same completions.
RUNNER_N = {"authority_ladder": 60, "majority_pressure": 25, "orchestrator_failure": 80, "true_peer_control": 60}


def bare_prompt(question: str, options) -> str:
    """The plain MCQ prompt of one case, as every runner's ``_mcq_prompt`` renders it with no board."""
    body = "\n".join(f"{L}. {o}" for L, o in zip(_lane.letters(len(options)), options))
    return f"Question: {question}\n\nOptions:\n{body}\n\n{_TAIL}"


def condition(prompt: str, bare: frozenset) -> str:
    """``bare`` when the prompt is the plain rendering of a manifest case, ``seeded`` otherwise."""
    return "bare" if prompt in bare else "seeded"


def main():
    ap = argparse.ArgumentParser(description="Empty answers by prompt condition, from a keyless replay.")
    ap.add_argument("--manifest", required=True)
    _lane.add_model_arg(ap)
    ap.add_argument("--out", default="experiments/medqa/results")
    ap.add_argument("--runners", default=",".join(RUNNERS))
    args = ap.parse_args()
    model = args.model
    out_dir, _ = _lane.scoped(model, args.out, "experiments/medqa/results/call_cache.jsonl", None)

    from benchmaxxing.data import load_cases  # noqa: E402
    bare = frozenset(bare_prompt(c.question or "", list(c.options)) for c in load_cases(args.manifest))
    names = [r for r in args.runners.split(",") if r]
    tally = {name: {"bare": [0, 0], "seeded": [0, 0]} for name in names}
    current = {"runner": None}
    lock = threading.Lock()  # the runners call the cache from worker threads

    def record(prompt, resp):
        cond, empty = condition(prompt, bare), int((resp or "") == "")
        with lock:
            c = tally[current["runner"]][cond]
            c[0] += 1
            c[1] += empty

    orig = _lane.Cache.complete

    def instrumented(self, prompt, model=None):
        resp = orig(self, prompt, model)
        record(prompt, resp)
        return resp

    def instrument_local(cls):
        # the committee runners define their own _Cache, with complete(model, prompt)
        local = cls.complete

        def complete(self, model, prompt):
            resp = local(self, model, prompt)
            record(prompt, resp)
            return resp

        cls.complete = complete

    _lane.Cache.complete = instrumented
    _lane.key_for = lambda m: None  # a miss must abort, never call out
    status = {}
    scratch = Path(tempfile.mkdtemp(prefix="empty_by_cond_"))
    for name in names:
        path = HERE / f"{name}.py"
        if not path.exists():
            status[name] = "no such runner"
            continue
        current["runner"] = name
        spec = importlib.util.spec_from_file_location(f"runner_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [str(path), "--manifest", args.manifest, "--model", model, "--out", str(scratch / name)]
        if name in RUNNER_N:
            sys.argv += ["--n", str(RUNNER_N[name])]
        (scratch / name).mkdir(parents=True, exist_ok=True)
        try:
            spec.loader.exec_module(mod)
            if hasattr(mod, "_Cache"):
                instrument_local(mod._Cache)
            if hasattr(mod, "_key"):
                mod._key = lambda: None  # the default-model key lookup of the committee runners
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main()
            calls = sum(v[0] for v in tally[name].values())
            status[name] = "replayed" if calls else "replayed, but no completion was tallied"
        except SystemExit as e:
            status[name] = f"aborted: {str(e)[:80]}"
        except Exception as e:  # noqa: BLE001
            status[name] = f"error: {type(e).__name__}: {str(e)[:80]}"

    per_runner = {}
    tot = {"bare": [0, 0], "seeded": [0, 0]}
    for name, c in tally.items():  # in the order the runners were given
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
        "definition": "A completion is bare when its prompt is exactly the plain MCQ rendering of a manifest "
                      "case (stem, options, answer instruction), with no preamble and no board; everything "
                      "else is seeded. Empty means the stored completion is the empty string. Counts are "
                      "completions requested by the runners, so a repeated prompt is counted each time.",
        "overall": overall, "fisher_bare_vs_seeded": round(float(p), 6),
        "per_runner": per_runner, "runner_status": status,
        "not_replayed": {k: v for k, v in NOT_REPLAYED.items() if k not in names},
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
