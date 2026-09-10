"""Imaging cue combinations / super-additivity (#207): do two image cues applied together flip the
read more than the sum of each alone? We cross the two strongest printed cues from the solo battery,
watermark and corner_tag, in a 2x2 on the same radiographs:

  clean      : the unaltered image.
  watermark  : watermark cue only.
  corner_tag : corner-tag cue only.
  both       : watermark then corner_tag chained on the same image.

Flip is a change from the clean read of the true finding. The interaction is
(both - clean) minus [(watermark - clean) + (corner_tag - clean)]: positive means the two cues
reinforce super-additively, near zero means they combine additively (or one saturates), negative
means they partly cancel. The single-cue reads reuse the committed flash image cache; only the
combined image is new. Keyless on a fully cached re-run.
"""
from __future__ import annotations
from benchmaxxing.extract import parse_yesno

import argparse
import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from benchmaxxing import gateway
from benchmaxxing.cues import image as ci
from benchmaxxing.data import load_cases
from benchmaxxing.stats import mcnemar

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane  # noqa: E402

MODEL = "gemini-2.5-flash"
_lock = threading.Lock()


def _key():
    return _lane.key_for(MODEL)


def _img_bytes(pil):
    buf = io.BytesIO()
    pil.convert("L").save(buf, format="PNG")
    return buf.getvalue()


def _to_pil(x):
    return x.convert("L") if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x).astype("uint8")).convert("L")


class _Cache:
    def __init__(self, path, key):
        self.path, self.key, self.store, self._b = Path(path), key, {}, None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.store[r["k"]] = r["resp"]

    def ask(self, prompt, pil):
        k = f"{MODEL}:" + hashlib.sha256(_img_bytes(pil) + b"\x00" + prompt.encode()).hexdigest()
        with _lock:
            if k in self.store:
                return parse_yesno(self.store[k])
        if not self.key:
            raise SystemExit(f"Cache miss and no {_lane.key_name(MODEL)} set for {MODEL} "
                          "(a fully cached run needs no key).")
        resp = _lane.paced_complete(MODEL, self.key, prompt, image=pil, decoding={"temperature": 0})
        with _lock:
            self.store[k] = resp
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "resp": resp}) + "\n")
        return parse_yesno(resp)


def main():
    ap = argparse.ArgumentParser(description="Imaging cue combinations / super-additivity (#207).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", default="experiments/imaging/results")
    _lane.add_model_arg(ap, MODEL)
    ap.add_argument("--cache", default=None,
                    help="defaults to <out>/img_cache.jsonl, so a second model cannot append to the committed cache")
    ap.add_argument("--n", type=int, default=35)
    args = ap.parse_args()

    model = args.model
    default_model = MODEL
    if model != default_model:
        # Every Gemini seat becomes the requested model, as the text lanes do, so MODEL itself is
        # the requested id from here: the cache key prefix and the summary field follow it.
        assert _lane.rebind_models(globals(), model) > 0, "no Gemini id to rebind"

    out = Path(args.out) if model == default_model else Path(args.out) / model.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root)
    cache = _Cache(Path(args.cache) if args.cache else out / "img_cache.jsonl", _key())
    cases = [c for c in load_cases(args.manifest)
             if c.label and c.label.lower() not in ("no finding", "", "none") and (root / c.image_ref).exists()][:args.n]

    def q(finding):
        return f"You are reading a chest radiograph. Does this X-ray show {finding}? Answer with only 'yes' or 'no'."

    def run(case):
        finding = case.label.split("|")[0].strip()
        img = Image.open(root / case.image_ref).convert("L")
        clean = cache.ask(q(finding), img)
        wm = _to_pil(ci.build_image_twin(img, "watermark", ground_truth=finding, case_id=case.case_id).contaminated)
        ct = _to_pil(ci.build_image_twin(img, "corner_tag", ground_truth=finding, case_id=case.case_id).contaminated)
        both = _to_pil(ci.build_image_twin(wm, "corner_tag", ground_truth=finding, case_id=case.case_id).contaminated)
        reads = {"watermark": cache.ask(q(finding), wm), "corner_tag": cache.ask(q(finding), ct),
                 "both": cache.ask(q(finding), both)}
        row = {"case_id": case.case_id, "finding": finding, "clean": clean}
        for name, r in reads.items():
            row[f"{name}_flip"] = int(clean != "?" and r != clean)
        row["eligible"] = int(clean == "yes")  # count flips only where clean read the true finding
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run, c) for c in cases]):
            rows.append(fut.result())

    elig = [r for r in rows if r["eligible"]]
    n = len(elig)

    def rate(name):
        return round(sum(r[f"{name}_flip"] for r in elig) / n, 4) if n else None
    wm, ct, both = rate("watermark"), rate("corner_tag"), rate("both")
    interaction = round(both - (wm + ct), 4) if n else None
    stronger = "watermark" if (wm or 0) >= (ct or 0) else "corner_tag"
    gain = sum(1 for r in elig if r["both_flip"] and not r[f"{stronger}_flip"])
    lose = sum(1 for r in elig if r[f"{stronger}_flip"] and not r["both_flip"])
    mc = mcnemar(gain, lose)
    summary = {
        "n_eligible": n, "model": MODEL,
        "flip_rates": {"watermark": wm, "corner_tag": ct, "both": both},
        "interaction_both_minus_sum_of_singles": interaction,
        "both_vs_stronger_single": {"stronger": stronger, "gain": gain, "lose": lose,
                                    "pvalue": round(mc.pvalue, 6)},
        "read": (
            f"Flip rate of the true-finding read under two chained image cues: watermark alone {wm}, "
            f"corner-tag alone {ct}, both {both}. Interaction (both minus the sum of the singles) = "
            f"{interaction}: a large positive value would mean the two printed cues reinforce "
            "super-additively, near zero means they combine additively or one already saturates, "
            f"negative means they partly cancel. Both vs the stronger single ({stronger}): McNemar "
            f"gain={gain} lose={lose}, p={round(mc.pvalue, 6)}, i.e. whether stacking a second cue buys "
            "any extra flip over the strongest one alone."
        ),
    }
    (out / "imaging_cue_combo_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "imaging_cue_combo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
