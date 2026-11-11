import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'runtime/ct2'),str(ROOT)]
from translator.bidirectional_translator import BidirectionalTranslator
from translator.fact_checks import check
x=BidirectionalTranslator(mode='auto');rows=[]
positive=['价格从30元降到13元。','这本书售价39.9元。','运费为15元。','订单金额为125.50元。','退款金额为36元。','票价为68元。','单价为8.5元。','付款金额为120元。']
controls=['价格从30美元降到13美元。','价格为30日元。','价格为30加元。','价格从30元降到13美元。','请不要关闭电源。','会议从上午9点持续到11点。','数据大小为30GB。','价格从人民币30元降到人民币13元。']
for source in positive+controls:
 x.mode='baseline';base=x.translate(source)
 x.mode='auto';start=time.perf_counter();d=x.translate_detailed(source)
 errors=[seg['selection'].get('error') for seg in d['segments'] if seg['selection'].get('error')]
 assert not errors,errors
 row={'source':source,'baseline':base,'auto':d['translation'],'seconds':time.perf_counter()-start,'report':d,'expected_currency_scope':source in positive}
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='report'},ensure_ascii=False),flush=True)
 if source in controls:assert d['translation']==base,(source,base,d['translation'])
# Selection must remain optional and the baseline must remain reproducible.
x.currency_rescue=False;x.mode='auto';disabled=x.translate(positive[0]);x.mode='baseline';assert disabled==x.translate(positive[0])
summary={'positive_cases':len(positive),'candidate_recovery_selected':sum('currency_mismatch' in check(z['source'],z['baseline'])['violations'] and not check(z['source'],z['auto'])['violations'] and any(s['selection']['reason']=='currency_candidate_recovery' for s in z['report']['segments']) for z in rows[:len(positive)]),'currency_mismatch_removed':sum('currency_mismatch' in check(z['source'],z['baseline'])['violations'] and not check(z['source'],z['auto'])['violations'] for z in rows[:len(positive)]),'controls_unchanged':len(controls),'limitations':'Narrow engineering checks, AI-inspected outputs, not independent human evaluation or held-out research benchmark.'}
out=ROOT/'experiments/results/bidirectional_small';out.mkdir(parents=True,exist_ok=True);(out/'currency_recovery.json').write_text(json.dumps({'summary':summary,'outputs':rows},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary),flush=True)
