"""Actual-model checks for GUI-compatible reports and the long-input regression."""
import json
from pathlib import Path
from translator.ct2_translator import CT2Translator
from experiments.corpus import CASES
ROOT=Path(__file__).resolve().parents[1]
engine=CT2Translator()
assert engine.precision=='float32' and engine.num_beams==6 and engine.max_decoding_length==512
results=[]
for source in ['他说：“请保存实验结果。”',next(c['text'] for c in CASES if c['id']=='t10_no_punct_long')]:
    report=engine.translate_detailed(source)
    assert report['backend']=='CTranslate2' and report['precision']=='float32'
    assert all(k in segment for segment in report['segments'] for k in ['warnings','retry_attempted','retry_accepted','source','translation'])
    assert ''.join(s['source'] for s in report['segments'])==engine._preprocess(source)
    if source.startswith('他说'):
        assert len(report['segments'])==1
    else:
        assert 'forty' in report['translation'].lower() or '42' in report['translation']
    results.append(dict(source=source,report=report))
out=ROOT/'experiments/results/ct2_interface';out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: FP32/6-beam/512 defaults, GUI report keys, source preservation, quoted sentence and long-input tail.')
