"""Imaging polarity asymmetry (#213): does an image shortcut cue more easily SUPPRESS a true finding
(a yes -> no flip) than HALLUCINATE an absent one (a no -> yes flip), or vice versa?

For each positive case we read two findings from the same radiograph under each cue:

  positive polarity : the case's TRUE primary finding F (clean read expected 'yes'); a cue-induced
                      yes -> no flip means the cue made the model miss a real finding.
  negative polarity : a finding G that is NOT present in the case (clean read expected 'no'); a
                      cue-induced no -> yes flip means the cue made the model hallucinate a finding.

We report the flip rate in each polarity per cue and overall. A larger yes->no rate means cues act
mainly by erasing signal (dangerous: missed findings); a larger no->yes rate means they act by
injecting spurious signal (dangerous: false alarms). The true-finding reads reuse the committed flash
image cache (img_cache.jsonl); only the absent-finding reads are new. Keyless on a fully cached re-run.
"""
from __future__ import annotations

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

MODEL = "gemini-2.5-flash"
CUES = ["cable", "corner_tag", "watermark", "laterality"]
# a fixed vocabulary of common CXR findings to draw an ABSENT finding from
VOCAB = ["cardiomegaly", "effusion", "emphysema", "mass", "nodule", "infiltration",
         "atelectasis", "pneumothorax", "hernia"]
_lock = threading.Lock()


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _img_bytes(pil):
    buf = io.BytesIO()
    pil.convert("L").save(buf, format="PNG")
    return buf.getvalue()


def _yesno(text):
    t = (text or "").strip().lower()
    if t.startswith("yes") or " yes" in t[:20]:
        return "yes"
    if t.startswith("no") or " no" in t[:20]:
        return "no"
    return "yes" if "yes" in t else ("no" if "no" in t else "?")


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
                return _yesno(self.store[k])
        if not self.key:
            raise SystemExit("Cache miss and no GEMINI_API_KEY set (a fully cached run needs no key).")
        if self._b is None:
            self._b = gateway.RetryBackend(gateway.GeminiBackend(model=MODEL, api_key=self.key), tries=5, backoff=3.0)
        resp = self._b.complete(prompt, image=pil, decoding={"temperature": 0})
        with _lock:
            self.store[k] = resp
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "resp": resp}) + "\n")
        return _yesno(resp)


def main():
    ap = argparse.ArgumentParser(description="Imaging polarity asymmetry: yes->no vs no->yes cue flips (#213).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", default="experiments/imaging/results")
    ap.add_argument("--cache", default="experiments/imaging/results/img_cache.jsonl")
    ap.add_argument("--n", type=int, default=35)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root)
    cache = _Cache(Path(args.cache), _key())
    cases = [c for c in load_cases(args.manifest)
             if c.label and c.label.lower() not in ("no finding", "", "none") and (root / c.image_ref).exists()][:args.n]

    def q(finding):
        return f"You are reading a chest radiograph. Does this X-ray show {finding}? Answer with only 'yes' or 'no'."

    def run(case):
        present = {p.strip().lower() for p in case.label.split("|")}
        finding = case.label.split("|")[0].strip()
        absent = next((g for g in VOCAB if g not in present), None)
        if absent is None:
            return None
        img = Image.open(root / case.image_ref).convert("L")
        pos_clean = cache.ask(q(finding), img)   # expected yes (reuses flash cache)
        neg_clean = cache.ask(q(absent), img)    # expected no
        row = {"case_id": case.case_id, "finding": finding, "absent": absent,
               "pos_clean": pos_clean, "neg_clean": neg_clean}
        for cue in CUES:
            try:
                tp = ci.build_image_twin(img, cue, ground_truth=finding, case_id=case.case_id)
                cont = _to_pil(tp.contaminated)
                pos_cue = cache.ask(q(finding), cont)
                neg_cue = cache.ask(q(absent), cont)
                # polarity flips only counted where the clean read had the expected value
                row[f"{cue}_pos_flip"] = (pos_clean == "yes" and pos_cue == "no")
                row[f"{cue}_neg_flip"] = (neg_clean == "no" and neg_cue == "yes")
                row[f"{cue}_pos_eligible"] = (pos_clean == "yes")
                row[f"{cue}_neg_eligible"] = (neg_clean == "no")
            except Exception as e:  # noqa: BLE001
                row[f"{cue}_err"] = str(e)[:30]
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run, c) for c in cases]):
            r = fut.result()
            if r:
                rows.append(r)

    def rate(pol):
        flips = sum(1 for r in rows for cue in CUES if r.get(f"{cue}_{pol}_flip"))
        elig = sum(1 for r in rows for cue in CUES if r.get(f"{cue}_{pol}_eligible"))
        return {"flips": flips, "eligible": elig, "rate": round(flips / elig, 4) if elig else None}

    pos, neg = rate("pos"), rate("neg")
    per_cue = {}
    for cue in CUES:
        pf = sum(1 for r in rows if r.get(f"{cue}_pos_flip"))
        pe = sum(1 for r in rows if r.get(f"{cue}_pos_eligible"))
        nf = sum(1 for r in rows if r.get(f"{cue}_neg_flip"))
        ne = sum(1 for r in rows if r.get(f"{cue}_neg_eligible"))
        per_cue[cue] = {"pos_flip_rate": round(pf / pe, 4) if pe else None,
                        "neg_flip_rate": round(nf / ne, 4) if ne else None}
    # McNemar over paired (pos_flip, neg_flip) per case-cue where both polarities eligible
    gain = sum(1 for r in rows for cue in CUES
               if r.get(f"{cue}_pos_eligible") and r.get(f"{cue}_neg_eligible")
               and r.get(f"{cue}_pos_flip") and not r.get(f"{cue}_neg_flip"))
    lose = sum(1 for r in rows for cue in CUES
               if r.get(f"{cue}_pos_eligible") and r.get(f"{cue}_neg_eligible")
               and r.get(f"{cue}_neg_flip") and not r.get(f"{cue}_pos_flip"))
    mc = mcnemar(gain, lose)
    summary = {
        "n_cases": len(rows), "model": MODEL,
        "positive_polarity_yes_to_no": pos, "negative_polarity_no_to_yes": neg,
        "per_cue": per_cue,
        "asymmetry_mcnemar": {"pos_only": gain, "neg_only": lose, "pvalue": round(mc.pvalue, 6)},
        "read": (
            f"Cue-induced flips by polarity, pooled over the four cues: a TRUE finding is suppressed "
            f"(yes -> no) at rate {pos['rate']} ({pos['flips']}/{pos['eligible']}), while an ABSENT "
            f"finding is hallucinated (no -> yes) at rate {neg['rate']} ({neg['flips']}/{neg['eligible']}) "
            f"(paired McNemar pos-only={gain} neg-only={lose}, p={round(mc.pvalue, 6)}). A dominant "
            "yes->no rate means the cues work mainly by erasing real signal (missed findings); a "
            "dominant no->yes rate means they inject spurious signal (false alarms). Symmetry means the "
            "cue destabilises the read in both directions equally."
        ),
    }
    (out / "imaging_polarity_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "imaging_polarity.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
