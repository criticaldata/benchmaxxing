#!/usr/bin/env python3
import csv, json
from pathlib import Path

ROOT = Path('/data/PROJECTS/kennedy/model-tests/benchmaxxing-main')
MEDQA = ROOT / 'data/medqa/staged/test/medqa_manifest.csv'
SUPPORT = ROOT / 'experiments/support2/results/support2_manifest.csv'
SLUGS = {
    'gptoss': 'openai_gpt-oss-120b',
    'qwen': 'Qwen_Qwen2.5-VL-72B-Instruct',
}

def json_rows(path):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row=json.loads(line)
        except Exception:
            continue
        if isinstance(row,dict) and row.get('case_id'):
            rows.append(row)
    return rows

def ids_from(path, collapse_clean=False):
    out=[]
    seen=set()
    for r in json_rows(path):
        cid=str(r['case_id'])
        if collapse_clean and cid.endswith('::clean'):
            cid=cid[:-7]
        if cid not in seen:
            seen.add(cid); out.append(cid)
    return out

manifest_meta={}
def build(name, src, rows_path, expected, out_subdir, collapse_clean=False):
    ids=ids_from(rows_path, collapse_clean)
    if len(ids) != expected:
        raise RuntimeError(f'{name}: expected {expected} unique IDs, got {len(ids)} from {rows_path}')
    with src.open(newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames
        all_rows=list(reader)
    by={r['case_id']:r for r in all_rows}
    missing=[x for x in ids if x not in by]
    if missing:
        raise RuntimeError(f'{name}: {len(missing)} IDs missing from {src}: {missing[:4]}')
    # Keep the source manifest order so load_cases sees the same deterministic order.
    selected=[r for r in all_rows if r['case_id'] in set(ids)]
    if len(selected)!=expected:
        raise RuntimeError(f'{name}: filtered row count {len(selected)} != {expected}')
    outdir=ROOT/'data/q4_case_cohorts'/out_subdir
    outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/f'{name}.csv'
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(selected)
    (outdir/f'{name}.ids.txt').write_text(''.join(r['case_id']+'\n' for r in selected))
    manifest_meta[name]={'n':expected,'case_ids':[r['case_id'] for r in selected],
                         'manifest':str(out.relative_to(ROOT)),
                         'source_manifest':str(src.relative_to(ROOT)),
                         'reference_rows':str(rows_path.relative_to(ROOT))}

ref=ROOT/'experiments/referee/results'/SLUGS['gptoss']
for arm in ('referee_deployable','referee_judge','referee_requery_design','referee_self_inconsistency','referee_threshold'):
    build('referee_'+arm,MEDQA,ref/f'{arm}.jsonl',40,'referee',collapse_clean=(arm=='referee_deployable'))

cascade=ROOT/'experiments/cascade/results'
build('multi_round_n40',MEDQA,cascade/SLUGS['gptoss']/'multi_round.jsonl',40,'cascade')
build('multi_round_n120',MEDQA,cascade/SLUGS['qwen']/'multi_round_n120.jsonl',120,'cascade')

cont=ROOT/'experiments/contamination/results'/SLUGS['gptoss']/'contamination_audit.jsonl'
build('contamination_audit',MEDQA,cont,100,'contamination')

modeldep=ROOT/'experiments/model_dependence/results'/SLUGS['gptoss']/'cascade_C_flash.jsonl'
build('cascade_C_flash',MEDQA,modeldep,22,'model_dependence')

support_ref=ROOT/'experiments/support2/results'/SLUGS['gptoss']/'support2_solo.jsonl'
build('support2',SUPPORT,support_ref,120,'support2')

meta={'comparator_for_text':SLUGS['gptoss'],'comparator_for_multi_round_n120':SLUGS['qwen'],
      'case_sets':manifest_meta,
      'notes':['Case manifests are exact ID filters against the named comparator rows.',
               'Manifests preserve original source order for deterministic runner iteration.',
               'referee_deployable comparator includes planted and ::clean rows; IDs are deduplicated to the 40 cases.']}
(ROOT/'experiments/model_metadata/q4_additional_case_sets.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:v['n'] for k,v in manifest_meta.items()},indent=2))
