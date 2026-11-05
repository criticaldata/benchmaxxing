import csv, json
from pathlib import Path

BASE=Path("/data/PROJECTS/kennedy/model-tests/benchmaxxing-main")
MODEL="openai_gpt-oss-120b"
specs=[
    ("medqa","data/medqa/staged/test/medqa_manifest.csv","experiments/medqa/results",[
        "reproduce_solo","reproduce_cascade","attributed_tier","authority_ladder","break_it","clean_a",
        "push_c","scale_c","committee_size_sweep","contamination_cascade","deliberation_channel",
        "deliberation_framing","dose_response","hierarchy_dominance","hierarchy_temp","leader_as_auditor",
        "live_peer_organic","majority_pressure","orchestrator_failure","paraphrase_robustness",
        "plausible_distractor","pre_emptive_referee","rationale_validity","seed_confidence","seed_timing",
        "super_additivity","temperature_sensitivity","test_awareness","text_cue_types",
        "true_peer_control","unanimity_break"]),
    ("medmcqa","data/medmcqa_dev.csv","experiments/medmcqa/results",[
        "reproduce_solo","reproduce_cascade","attributed_tier","authority_ladder","break_it","clean_a",
        "scale_c","committee_size_sweep","deliberation_framing","dose_response","hierarchy_dominance",
        "leader_as_auditor","majority_pressure","orchestrator_failure","plausible_distractor",
        "rationale_validity","referee_deployable","referee_judge","seed_confidence","super_additivity",
        "temperature_sensitivity","test_awareness","text_cue_types","true_peer_control","unanimity_break"])
]
def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def uniq(values):
    out=[]
    seen=set()
    for v in values:
        if v is None: continue
        v=str(v)
        if v not in seen: seen.add(v); out.append(v)
    return out
def arm_ids(arm, d, manifest_rows, idcol):
    if arm=="reproduce_solo":
        return uniq(r["case_id"] for r in read_jsonl(d/"solo_records.jsonl"))
    if arm=="reproduce_cascade":
        obj=json.loads((d/"cascade_results.json").read_text())
        return uniq(r["case_id"] for r in obj["per_case"])
    if arm=="clean_a":
        hard=set()
        # clean_a/break_it/push_c use the shared Gemini solo-records cohort.
        for r in read_jsonl(d.parent/"solo_records.jsonl"):
            if r.get("clean_correct") is False: hard.add(str(r["case_id"]))
        raw=json.loads((d/"clean_a_summary.json").read_text())
        n=raw.get("openai/gpt-oss-120b",{}).get("n")
        ids=[str(r[idcol]) for r in manifest_rows if str(r[idcol]) in hard][:n]
        if len(ids)!=n: raise RuntimeError(f"{arm}: expected n={n}, got {len(ids)}")
        return ids
    if arm=="break_it":
        f=d/"break_it_A_per_case.jsonl"
    elif arm=="scale_c":
        f=d/"scale_c_per_case.jsonl"
    elif arm=="push_c":
        f=d/"push_c_per_case.jsonl"
    else:
        f=d/(arm+".jsonl")
    if not f.exists(): raise FileNotFoundError(f"missing reference rows: {f}")
    ids=[]
    for r in read_jsonl(f):
        cid=r.get("case_id",r.get("id"))
        if cid is not None:
            ids.append(str(cid).split("::",1)[0])
    return uniq(ids)

metadata={"model_comparator":"openai/gpt-oss-120b","case_sets":{}}
for lane,manifest,root,arms in specs:
    rows=list(csv.DictReader((BASE/manifest).open()))
    idcol=next((k for k in rows[0] if k in ("case_id","id","question_id")),None)
    if idcol is None: raise RuntimeError(f"no case ID in {manifest}")
    available={str(r[idcol]) for r in rows}
    ref=BASE/root/MODEL
    outdir=BASE/"data"/"q4_case_cohorts"/lane
    outdir.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        ids=arm_ids(arm,ref,rows,idcol)
        missing=[x for x in ids if x not in available]
        if missing: raise RuntimeError(f"{lane}/{arm}: missing IDs {missing[:5]}")
        if len(ids)!=len(set(ids)): raise RuntimeError(f"{lane}/{arm}: duplicate IDs")
        ordered={x:i for i,x in enumerate(ids)}
        selected=[r for r in rows if str(r[idcol]) in ordered]
        selected.sort(key=lambda r: ordered[str(r[idcol])])
        if len(selected)!=len(ids): raise RuntimeError(f"{lane}/{arm}: row count mismatch")
        p=outdir/(arm+".csv")
        with p.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(selected)
        metadata["case_sets"][f"{lane}/{arm}"]={"n":len(ids),"case_ids":ids,"manifest":str(p.relative_to(BASE))}
        print(f"{lane}/{arm}: n={len(ids)}")
(BASE/"experiments/model_metadata").mkdir(parents=True,exist_ok=True)
(BASE/"experiments/model_metadata/q4_text_case_sets.json").write_text(json.dumps(metadata,indent=2)+"\n")
