"""Recover the committed blind-metric justifications from the committed caches.

The committed arms kept only flags per case; the justification text sits in the call caches,
keyed by a hash of the prompt (and, for films, of the re-encoded image bytes). This rebuilds each
case's blind prompt exactly as the committed runner built it, recomputes the key with the repo's
own code, and writes the recovered text beside the committed flags, so the blinded re-read (Run 3)
can include the original note-following justifications and not only the fresh ones.

    text     key = sha256(f"{model}\\x00{prompt}")                     (blind_metric.py)
    imaging  key = f"{model}:" + sha256(png_bytes(image) + b"\\x00" + prompt)  (imaging_blind_metric.py)

The imaging key depends on the PNG encoder that produced the committed cache; a different Pillow or
zlib build re-encodes the same pixels to different bytes and every key misses. The script reports
hits and misses per cohort rather than guessing; a miss means "run this where the cache was made".

Usage
    PYTHONPATH=. python experiments/blind_metric/recover_justifications.py --lane text \
        --manifest <medqa manifest> --rows experiments/blind_metric/results/blind_metric.jsonl \
        --cache experiments/blind_metric/results/call_cache.jsonl --out <dir>
    PYTHONPATH=. python experiments/blind_metric/recover_justifications.py --lane imaging \
        --manifest experiments/imaging/results/nih_manifest.csv --image-root <films> \
        --rows experiments/imaging/results/imaging_blind_metric.jsonl \
        --cache experiments/imaging/results/img_cache.jsonl --model gemini-2.5-flash --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched_conditions import NAMING, _img_bytes, imaging_prompts, text_prompts  # noqa: E402

from benchmaxxing.data import load_cases  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", choices=("text", "imaging"), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--rows", required=True, help="the committed per-case jsonl of the arm")
    ap.add_argument("--cache", required=True, help="the committed call cache of the arm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    model = args.model or ("gemini-2.5-flash-lite" if args.lane == "text" else "gemini-2.5-flash")
    cache = {}
    for line in Path(args.cache).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cache[r["k"]] = r["resp"]
    rows = {r["case_id"]: r for r in (json.loads(l) for l in Path(args.rows).read_text().splitlines()) if r}
    cases = [c for c in load_cases(args.manifest) if c.case_id in rows]

    out, hit, miss = [], 0, 0
    for c in cases:
        r = rows[c.case_id]
        if args.lane == "text":
            _, _, _, P = text_prompts(c)
            prompt = P["note"](r["decoy_letter"])
            k = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
        else:
            from PIL import Image
            finding = c.label.split("|")[0].strip()
            _, P = imaging_prompts(finding)
            prompt = P["note"](r["decoy"])
            pil = Image.open(Path(args.image_root) / c.image_ref).convert("L")
            k = f"{model}:" + hashlib.sha256(_img_bytes(pil) + b"\x00" + prompt.encode()).hexdigest()
        txt = cache.get(k)
        if txt is None:
            miss += 1
            continue
        hit += 1
        out.append({"case_id": c.case_id, "patient_id": c.patient_id, "blind_is_decoy": r["blind_is_decoy"],
                    "named_keyword_screen": bool(NAMING.search(txt)), "text": txt})
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "recovered_justifications.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    summary = {"lane": args.lane, "model": model, "cases": len(cases), "recovered": hit, "missed": miss,
               "drifters_recovered": sum(1 for x in out if x["blind_is_decoy"]),
               "note": None if miss == 0 else ("keys did not reproduce here; for films the key depends on the "
                                               "PNG encoder of the machine that wrote the cache")}
    (od / "recovered_justifications_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
