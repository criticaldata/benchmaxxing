import csv, json
from pathlib import Path
BASE=Path(".")
Q="Qwen_Qwen2.5-VL-72B-Instruct"
def rows_jsonl(p):
    out=[]
    for line in p.read_text().splitlines():
        if line.strip():
            try: x=json.loads(line)
            except Exception: continue
            if isinstance(x,dict): out.append(x)
    return out
def ids_from(p):
    return list(dict.fromkeys(str(r["case_id"]) for r in rows_jsonl(p) if r.get("case_id") is not None))
def make_one(lane,name,manifest,reference,meta,note=None):
    source=BASE/manifest
    rows=list(csv.DictReader(source.open()))
    if not rows: raise RuntimeError(f"empty manifest {source}")
    idcol="case_id"
    lookup={str(r[idcol]):r for r in rows}
    ref=BASE/reference
    ids=ids_from(ref)
    missing=[x for x in ids if x not in lookup]
    if missing: raise RuntimeError(f"{lane}/{name} missing {len(missing)} from {source}: {missing[:4]}")
    chosen=[lookup[x] for x in ids]
    outdir=BASE/"data"/"q4_case_cohorts"/lane
    outdir.mkdir(parents=True,exist_ok=True)
    p=outdir/(name+".csv")
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(chosen)
    (outdir/(name+".ids.txt")).write_text("\n".join(ids)+"\n")
    entry={"n":len(ids),"case_ids":ids,"manifest":str(p),"source_manifest":str(source),"reference_rows":str(ref)}
    if note: entry["note"]=note
    meta["case_sets"][lane+"/"+name]=entry
    print(lane+"/"+name,"n",len(ids),"source",manifest)
meta={"model_comparator":"Qwen/Qwen2.5-VL-72B-Instruct","case_sets":{}}
nih=Path(f"experiments/imaging/results/{Q}")
for name,file in {
"imaging_solo":"imaging_solo.jsonl","imaging_solo_lite":"imaging_solo_lite.jsonl",
"imaging_noise_floor":"imaging_noise_floor.jsonl","imaging_noise_floor_lite":"imaging_noise_floor_lite.jsonl",
"imaging_cascade":"imaging_cascade.jsonl","imaging_cascade_cable":"imaging_cascade_cable.jsonl",
"imaging_cascade_corner_tag":"imaging_cascade_corner_tag.jsonl","imaging_cascade_laterality":"imaging_cascade_laterality.jsonl",
"imaging_cascade_scale":"imaging_cascade_scale.jsonl","imaging_cue_combo":"imaging_cue_combo.jsonl",
"imaging_cue_families":"imaging_cue_families.jsonl","imaging_cue_families_noise":"imaging_cue_families_noise.jsonl",
"imaging_judge_referee":"imaging_judge_referee.jsonl","imaging_judge_referee_film":"imaging_judge_referee_film.jsonl",
"imaging_majority_pressure":"imaging_majority_pressure.jsonl","imaging_matched_temp":"imaging_matched_temp.jsonl",
"imaging_multi_round":"imaging_multi_round.jsonl","imaging_peer_size_curve":"imaging_peer_size_curve.jsonl",
"imaging_polarity":"imaging_polarity.jsonl","imaging_referee":"imaging_referee.jsonl",
"imaging_strength_cascade":"imaging_strength_cascade.jsonl","imaging_system_flag":"imaging_system_flag.jsonl",
}.items():
    make_one("imaging",name,"experiments/imaging/results/nih_manifest_scaled.csv",str(nih/file),meta)
r=Path("experiments/imaging_chexpert/results")
q35=r/Q
for name,file in {"imaging_solo":"imaging_solo.jsonl","imaging_cascade":"imaging_cascade.jsonl","imaging_matched_temp":"imaging_matched_temp.jsonl"}.items():
    make_one("chexpert",name,"experiments/imaging_chexpert/results/nih_match_35.csv",str(q35/file),meta)
# Qwen's old stand-alone noise file used a different, unavailable 35-image set.
# Recompute noise inline with imaging_solo on the common 35 matched cases above.
make_one("chexpert","imaging_noise_floor","experiments/imaging_chexpert/results/nih_match_35.csv",
         str(q35/"imaging_solo.jsonl"),meta,
         "Recompute as part of imaging_solo on the common Qwen solo 35; the legacy noise-floor artifact uses an unavailable separate cohort.")
make_one("chexpert","device_absent_cascade","experiments/imaging_chexpert/results/device_absent/cascade_150.csv",
         str(r/"device_absent"/Q/"imaging_cascade_none.jsonl"),meta)
make_one("chexpert","natural_cues_cascade","experiments/imaging_chexpert/results/natural_cues/cascade_150.csv",
         str(r/"natural_cues"/Q/"imaging_cascade_none.jsonl"),meta)
make_one("chexpert","natural_independent_cascade","experiments/imaging_chexpert/results/natural_independent/cascade_150.csv",
         str(r/"natural_independent"/Q/"imaging_cascade_none.jsonl"),meta)
make_one("chexpert","system_flag","experiments/imaging_chexpert/results/natural_cues/solo_600.csv",
         str(r/"system_flag"/Q/"imaging_system_flag.jsonl"),meta)
(BASE/"experiments/model_metadata").mkdir(parents=True,exist_ok=True)
(BASE/"experiments/model_metadata/q4_image_case_sets.json").write_text(json.dumps(meta,indent=2)+"\n")
