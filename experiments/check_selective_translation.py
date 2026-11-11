import sys,json,time,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'runtime/ct2'),str(ROOT)]
from translator.selective_translator import SelectiveTranslator
from translator.qwen_expert import QwenExpert
from sacrebleu.metrics import BLEU,CHRF
import psutil
out=ROOT/'experiments/results/selective_translation'
cases=json.loads((out/'frozen_cases.json').read_text(encoding='utf8'))
x=SelectiveTranslator();rss=psutil.Process().memory_info().rss
start=time.perf_counter();x.expert=QwenExpert(glossary=x.domain_terms);cold=time.perf_counter()-start
x.expert.translate(['这是一个测试。']);rows=[]
for case in cases:
 row=dict(case,outputs={},reports={})
 for mode in ['baseline','selective','quality']:
  x.mode=mode;report=x.translate_detailed(case['source'])
  assert not any(s['selection'].get('error') for s in report['segments']),report
  row['outputs'][mode]=report['translation'];row['reports'][mode]=report
 rows.append(row)
 print(json.dumps({'source':row['source'],'outputs':row['outputs']},ensure_ascii=False),flush=True)
metrics={}
for split in ['technical','general']:
 part=[r for r in rows if r['split']==split];refs=[[r['reference'] for r in part]]
 metrics[split]={m:{'BLEU':BLEU().corpus_score([r['outputs'][m] for r in part],refs).score,'chrFpp':CHRF(word_order=2).corpus_score([r['outputs'][m] for r in part],refs).score,'missing_domain_terms':sum(len(x._missing_terms(r['source'],r['outputs'][m])) for r in part)} for m in ['baseline','selective','quality']}
# Three warm rotated runs over four frozen inputs; compare system variants, not equal-weight inference engines.
sample=[cases[i]['source'] for i in [0,2,8,10]];times={m:[] for m in ['baseline','selective','quality']}
for rep in range(3):
 modes=list(times);modes=modes[rep:]+modes[:rep]
 for m in modes:
  x.mode=m;start=time.perf_counter()
  for source in sample:x.translate(source)
  times[m].append(time.perf_counter()-start)
summary={'cases':16,'case_sha256':hashlib.sha256((out/'frozen_cases.json').read_bytes()).hexdigest(),'metrics':metrics,'timing_four_inputs':{m:{'runs':a,'median_s':statistics.median(a)} for m,a in times.items()},'expert_load_s':cold,'rss_growth_mb':(psutil.Process().memory_info().rss-rss)/1024**2,'expert_routed':sum(any(s['selection']['reason'].startswith('expert_selected') for s in r['reports']['selective']['segments']) for r in rows),'limitations':['AI-authored references frozen before first NLLB evaluation; Qwen is a subsequent development experiment on the same set; small functional set, not an independent human benchmark','Different pretrained models and beam settings; do not attribute gains to MarianMT inference optimization','No statistical significance or human blind assessment']}
(out/'outputs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2),flush=True)
