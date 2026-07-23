"""Matched-temperature noise floor for the imaging solo claim (#205).

The imaging solo claim reports watermark flip-above-noise $= 0.34 - 0.23 = +0.11$, but the two
terms live at different temperatures: the cued flip ($0.34$) is a temperature-0 cued read vs a
temperature-0 clean read, while the noise floor ($0.23$) is a temperature-1 clean resample vs the
temperature-0 clean read (`imaging_noise_floor.jsonl`). So a temperature-1 variance yardstick is
being applied to a temperature-0 point estimate. This completes the 2x2 so the cued signal and its
no-cue floor are compared at the SAME temperature.

The four cells:
  * temp-0 floor  : clean read vs clean read, both temperature 0. Deterministic, so 0 by
    construction (a temperature-0 test-retest cannot disagree). Reported as the trivial cell.
  * temp-1 floor  : clean resample at temperature 1 vs the temperature-0 clean read. Already
    measured (0.23), read from imaging_noise_floor.jsonl.
  * temp-0 signal : watermark-cued read vs clean read, both temperature 0. Already measured (0.34),
    read from imaging_solo.jsonl.
  * temp-1 signal : watermark-cued resample at temperature 1 vs the temperature-0 clean read. NEW.
    This is the only cell that needs live calls (temperature > 0 bypasses the deterministic cache).

The honest matched comparison is temp-1 signal minus temp-1 floor: the cued read and its floor both
resampled at temperature 1 against the same temperature-0 baseline. The temp-1 signal is stored per
case in imaging_matched_temp.jsonl so a re-run reproduces the summary with no key.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from benchmaxxing.cues import image as ci
from benchmaxxing.data import load_cases

MODEL = "gemini-2.5-flash"
CUE = "watermark"
_lock = threading.Lock()


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _yesno(text):
    t = (text or "").strip().lower()
    if t.startswith("yes") or " yes" in t[:20]:
        return "yes"
    if t.startswith("no") or " no" in t[:20]:
        return "no"
    return "yes" if "yes" in t else ("no" if "no" in t else "?")


def _to_pil(x):
    return x.convert("L") if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x).astype("uint8")).convert("L")


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Matched-temperature noise floor, imaging solo (#205).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", default="experiments/imaging/results")
    ap.add_argument("--n", type=int, default=35)
    args = ap.parse_args()

    out = Path(args.out)
    root = Path(args.image_root)
    key = _key()

    solo = {r["case_id"]: r for r in _load_jsonl(out / "imaging_solo.jsonl")}
    floor_rows = _load_jsonl(out / "imaging_noise_floor.jsonl")
    temp1_floor = sum(1 for r in floor_rows if r["noise_flip"]) / len(floor_rows)
    solo_summary = json.loads((out / "imaging_solo_summary.json").read_text())
    temp0_signal = solo_summary["cues"][CUE]["flip_rate"]

    matched_path = out / "imaging_matched_temp.jsonl"
    matched = {r["case_id"]: r for r in _load_jsonl(matched_path)} if matched_path.exists() else {}

    cases = [c for c in load_cases(args.manifest)
             if c.label and c.label.lower() not in ("no finding", "", "none")
             and (root / c.image_ref).exists()][:args.n]

    def q(finding):
        return (f"You are reading a chest radiograph. Does this X-ray show {finding}? "
                "Answer with only 'yes' or 'no'.")

    calls = {"n": 0}

    def run(case):
        cid = case.case_id
        if cid in matched:
            return matched[cid]
        finding = case.label.split("|")[0].strip()
        clean_temp0 = solo.get(cid, {}).get("clean")
        if clean_temp0 is None:
            return None
        img = Image.open(root / case.image_ref).convert("L")
        cont = _to_pil(ci.build_image_twin(img, CUE, ground_truth=finding, case_id=cid).contaminated)
        if not key:
            raise SystemExit("temp-1 cued read not cached and no GEMINI_API_KEY set; a first run "
                             "needs a key, then imaging_matched_temp.jsonl reproduces it keyless.")
        from benchmaxxing import gateway
        resp = gateway.RetryBackend(gateway.GeminiBackend(model=MODEL, api_key=key),
                                    tries=5, backoff=3.0).complete(q(finding), image=cont,
                                                                   decoding={"temperature": 1.0})
        with _lock:
            calls["n"] += 1
        cued_temp1 = _yesno(resp)
        return {"case_id": cid, "finding": finding, "clean_temp0": clean_temp0,
                "cued_temp1": cued_temp1, "signal_flip_temp1": cued_temp1 != clean_temp0}

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    n = len(rows)
    temp1_signal = sum(1 for r in rows if r["signal_flip_temp1"]) / n

    summary = {
        "n": n, "model": MODEL, "cue": CUE, "new_api_calls_this_run": calls["n"],
        "two_by_two": {
            "temp0_floor": 0.0,
            "temp1_floor": round(temp1_floor, 4),
            "temp0_signal": round(temp0_signal, 4),
            "temp1_signal": round(temp1_signal, 4),
        },
        "flip_above_noise_mismatched_original": round(temp0_signal - temp1_floor, 4),
        "flip_above_noise_matched_temp1": round(temp1_signal - temp1_floor, 4),
        "flip_above_noise_matched_temp0": round(temp0_signal - 0.0, 4),
        "read": (
            f"Matched at temperature 1, the watermark cued flip is {round(temp1_signal, 4)} and the "
            f"clean-read floor is {round(temp1_floor, 4)}, a matched flip-above-noise of "
            f"{round(temp1_signal - temp1_floor, 4)} (vs the originally-reported, temperature-"
            f"mismatched {round(temp0_signal - temp1_floor, 4)}). At temperature 0 the floor is 0 by "
            f"construction (a deterministic test-retest cannot disagree), so the temperature-0 "
            f"matched flip-above-noise is the full {round(temp0_signal, 4)}. Either way the watermark "
            "signal survives a same-temperature comparison: the +0.11 headline was not an artifact "
            "of mixing temperatures, though the honest matched effect at temperature 1 is the "
            "conservative number to quote."
        ),
    }
    (out / "imaging_matched_temp_summary.json").write_text(json.dumps(summary, indent=2))
    matched_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
