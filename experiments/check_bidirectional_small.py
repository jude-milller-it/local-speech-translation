import sys,json,time,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'runtime/ct2'),str(ROOT)]
from translator.bidirectional_translator import BidirectionalTranslator
from translator.fact_checks import check
import psutil
CASES=['价格从30元降到13元。','价格从人民币30元降到人民币13元。','温度从30摄氏度降到13摄氏度。','准确率从90%提高到95%。','会议定于2026年11月7日举行。','请不要关闭电源。','未经许可不得删除文件。','只有检查完成后才能重新启动设备。','该方法并不总是有效。','请使用Python保存结果。','卷积神经网络包含池化层和全连接层。','订单金额为125.50美元。','会议从上午9点持续到11点。','这是一个测试。','请打开窗户。','如果设备过热，请立即停止运行。']

def main():
 t=BidirectionalTranslator(); rows=[];rss0=psutil.Process().memory_info().rss
 start=time.perf_counter();t._load_reverse();cold=time.perf_counter()-start
 t.mode='bidirectional';warm=t.translate_detailed('这是一个测试。')
 f=t.engine.score_batch([t.tokens(t.tokenizer,'这是一个测试。')],[t.tokens(t.tokenizer,'This is a test.',True)])[0]
 assert f.tokens.count('</s>')==1, f.tokens
 assert not check('准确率从90%提高到95%。','Accuracy increased from 90 to 95 percent.')['violations']
 assert not check('会议定于2026年11月7日举行。','The meeting is on 7 November 2026.')['violations']
 assert not check('价格从30元降到13元。','The price fell from thirty yuan to thirteen yuan.')['violations']
 assert 'from_to_order_mismatch' in check(CASES[0],'The price rose from 13 yuan to 30 yuan.')['violations']
 assert 'percentage_mismatch' in check('准确率为95%。','Accuracy is 59 percent.')['violations']
 for source in CASES:
  row={'source':source,'modes':{}}
  for mode in ['baseline','mbr','bidirectional','auto']:
   t.mode=mode;d=t.translate_detailed(source)
   assert not any(s['selection'].get('error') for s in d['segments']),d
   row['modes'][mode]=d
  rows.append(row)
 # Rotated, warm paired timing; same four short inputs, three runs.
 times={m:[] for m in ['baseline','mbr','bidirectional','auto']}
 sample=[CASES[i] for i in [0,3,5,13]]
 for rep in range(3):
  modes=list(times);modes=modes[rep:]+modes[:rep]
  for mode in modes:
   t.mode=mode;start=time.perf_counter()
   for source in sample:t.translate(source)
   times[mode].append(time.perf_counter()-start)
 summary={'cases':len(rows),'reverse_cold_load_s':cold,'rss_growth_mb':(psutil.Process().memory_info().rss-rss0)/1024**2,'timing_four_inputs':{m:{'runs':ts,'median_s':statistics.median(ts)} for m,ts in times.items()},'device':'CPU, four CT2 threads; no CUDA allocation by either translator','gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip(),'changed_vs_baseline':{m:sum(x['modes'][m]['translation']!=x['modes']['baseline']['translation'] for x in rows) for m in times}}
 out=ROOT/'experiments/results/bidirectional_small';out.mkdir(parents=True,exist_ok=True)
 (out/'outputs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 print(json.dumps(summary,indent=2),flush=True)
 for row in rows:
  print(json.dumps({'source':row['source'],'outputs':{m:d['translation'] for m,d in row['modes'].items()}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
