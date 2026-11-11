import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'runtime/ct2'),str(ROOT)]
from translator.selective_translator import SelectiveTranslator
from sacrebleu.metrics import BLEU,CHRF
out=ROOT/'experiments/results/selective_translation';cases=json.loads((out/'acceptance_cases.json').read_text(encoding='utf8'))
code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['translator/selective_translator.py','translator/qwen_expert.py','translator/fact_checks.py']}
x=SelectiveTranslator();rows=[]
for case in cases:
 row=dict(case,outputs={},reports={})
 for mode in ['baseline','selective']:
  x.mode=mode;report=x.translate_detailed(case['source'])
  assert not any(s['selection'].get('error') for s in report['segments']),report
  row['outputs'][mode]=report['translation'];row['reports'][mode]=report
 rows.append(row);print(json.dumps({'source':row['source'],'outputs':row['outputs']},ensure_ascii=False),flush=True)
metrics={}
for split in ['technical','facts','general']:
 part=[r for r in rows if r['split']==split];refs=[[r['reference'] for r in part]]
 metrics[split]={mode:{'BLEU':BLEU().corpus_score([r['outputs'][mode] for r in part],refs).score,'chrFpp':CHRF(word_order=2).corpus_score([r['outputs'][mode] for r in part],refs).score,'missing_domain_terms':sum(len(x._missing_terms(r['source'],r['outputs'][mode])) for r in part)} for mode in ['baseline','selective']}
summary={'cases':len(rows),'case_sha256':hashlib.sha256((out/'acceptance_cases.json').read_bytes()).hexdigest(),'code_hashes':code_hashes,'metrics':metrics,'changed_cases':sum(r['outputs']['baseline']!=r['outputs']['selective'] for r in rows),'general_outputs_unchanged':all(r['outputs']['baseline']==r['outputs']['selective'] for r in rows if r['split']=='general'),'limits':'Small pre-fixed AI-authored acceptance set, no independent human evaluation or significance test. Not a publication-level result.'}
(out/'acceptance_outputs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');(out/'acceptance_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)
