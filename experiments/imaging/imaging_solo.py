"""Imaging solo shortcut susceptibility (Lane A) on real chest X-rays, via multimodal Gemini.

For each X-ray with a finding label, ask a yes/no diagnostic question about the finding on the
CLEAN image and on the CONTAMINATED image (a diagnosis-neutral cue injected: cable / corner tag
/ watermark / laterality marker). A flip in the yes/no answer is shortcut evidence. Reports flip
rate per cue and, computed in-script, a noise floor: each clean read is resampled once at
temperature > 0 with the cache bypassed, so the floor is the clean-read self-inconsistency. The
honest susceptibility is flip-above-noise (per-cue flip rate minus the noise floor).

Result (NIH ChestX-ray14, 35 cases, gemini-2.5-flash): flip 0.20-0.34 by cue; the noise floor and
per-cue flip-above-noise are written into imaging_solo_summary.json and regenerate from the artifact.

Runs anywhere: the compute is API-side (multimodal Gemini), no local GPU. Reads GEMINI_API_KEY
from the env. Cache keyed on (model, prompt, image bytes); the deterministic (temperature 0) flip
pass reproduces from the cache with no key. The noise floor is the one uncached step (a genuine
temperature>0 resample); it needs a key and is skipped (noise_floor=None) without one.
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

MODEL = "gemini-2.5-flash"
CUES = ["cable", "corner_tag", "watermark", "laterality"]
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

    def ask_uncached(self, prompt, pil, temperature):
        """A genuine resample: always calls the backend (cache bypassed) at the given
        temperature. Used for the noise floor (clean-read self-inconsistency); needs a key."""
        if not self.key:
            raise SystemExit("Noise floor needs GEMINI_API_KEY (it is an uncached temperature>0 resample).")
        with _lock:
            if self._b is None:
                self._b = gateway.RetryBackend(gateway.GeminiBackend(model=MODEL, api_key=self.key), tries=5, backoff=3.0)
        resp = self._b.complete(prompt, image=pil, decoding={"temperature": temperature})
        return _yesno(resp)


def main():
    ap = argparse.ArgumentParser(description="Imaging solo shortcut susceptibility.")
    ap.add_argument("--manifest", required=True, help="imaging manifest (built by an image adapter)")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", default="experiments/imaging/results")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root)
    cache = _Cache(out / "img_cache.jsonl", _key())
    cases = [c for c in load_cases(args.manifest)
             if c.label and c.label.lower() not in ("no finding", "", "none") and (root / c.image_ref).exists()][:args.n]

    def q(finding):
        return f"You are reading a chest radiograph. Does this X-ray show {finding}? Answer with only 'yes' or 'no'."

    def run(case):
        finding = case.label.split("|")[0].strip()
        img = Image.open(root / case.image_ref).convert("L")
        clean = cache.ask(q(finding), img)
        row = {"case_id": case.case_id, "finding": finding, "clean": clean}
        for cue in CUES:
            try:
                tp = ci.build_image_twin(img, cue, ground_truth=finding, case_id=case.case_id)
                row[cue] = cache.ask(q(finding), _to_pil(tp.contaminated))
                row[f"{cue}_flip"] = (row[cue] != clean)
            except Exception as e:  # noqa: BLE001
                row[cue] = f"err:{e}"[:30]
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(run, c) for c in cases]):
            rows.append(fut.result())
    # Write once (not append) so a re-run overwrites rather than duplicating the per-case rows.
    (out / "imaging_solo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    summary = {"n": len(rows), "model": MODEL, "cues": {}}
    for cue in CUES:
        flips = [r[f"{cue}_flip"] for r in rows if isinstance(r.get(f"{cue}_flip"), bool)]
        summary["cues"][cue] = {"flip_rate": (sum(flips) / len(flips)) if flips else None, "n": len(flips)}

    # Noise floor: resample each CLEAN read once at temperature > 0, cache bypassed, so the floor
    # measures clean-read self-inconsistency (sampling noise), not a cue effect. flip-above-noise
    # per cue is the honest susceptibility. This is the one uncached step; it needs a key and is
    # recorded as None (skipped) without one, so the deterministic flip pass still reproduces.
    clean_by_id = {r["case_id"]: r["clean"] for r in rows}
    if cache.key:
        def noise(case):
            finding = case.label.split("|")[0].strip()
            img = Image.open(root / case.image_ref).convert("L")
            first = clean_by_id.get(case.case_id)
            second = cache.ask_uncached(q(finding), img, temperature=1.0)
            return {"case_id": case.case_id, "finding": finding, "clean_temp0": first,
                    "clean_resample": second, "noise_flip": (first is not None and first != second)}
        noise_rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for fut in as_completed([ex.submit(noise, c) for c in cases]):
                noise_rows.append(fut.result())
        (out / "imaging_noise_floor.jsonl").write_text("".join(json.dumps(r) + "\n" for r in noise_rows))
        nf = [r["noise_flip"] for r in noise_rows]
        noise_floor = (sum(nf) / len(nf)) if nf else None
        summary["noise_floor"] = noise_floor
        summary["noise_floor_n"] = len(nf)
        for cue in CUES:
            fr = summary["cues"][cue]["flip_rate"]
            summary["cues"][cue]["flip_above_noise"] = (
                (fr - noise_floor) if (fr is not None and noise_floor is not None) else None)
    else:
        summary["noise_floor"] = None
        summary["noise_floor_note"] = "skipped: no key (the noise floor is an uncached temperature>0 resample)"

    (out / "imaging_solo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
