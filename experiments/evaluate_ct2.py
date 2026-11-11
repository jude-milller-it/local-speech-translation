import argparse,copy,csv,hashlib,json,platform,statistics,time
from pathlib import Path
import torch,transformers,ctranslate2
from sacrebleu.metrics import BLEU,CHRF
from translator.reliable_translator import ReliableTranslator
from translator.ct2_translator import CT2Translator
ROOT=Path(__file__).resolve().parents[1]


class TorchReference(ReliableTranslator):
    def __init__(self,beams=6,budget=512):
        super().__init__(device='cpu',batch_size=4)
        self.config=copy.deepcopy(self.model.generation_config)
        self.config.max_length=None;self.config.max_new_tokens=budget
        self.config.num_beams=beams;self.config.length_penalty=1.0
        self.config.early_stopping=False;self.config.do_sample=False
    def _generate(self,chunks):
        output=[]
        with torch.inference_mode():
            for start in range(0,len(chunks),self.batch_size):
                batch=chunks[start:start+self.batch_size]
                inputs=self.tokenizer(batch,return_tensors='pt',padding=True,truncation=False)
                generated=self.model.generate(**inputs,generation_config=self.config)
                output.extend(self.tokenizer.batch_decode(generated,skip_special_tokens=True))
        return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--beams',type=int,default=6);p.add_argument('--budget',type=int,default=512)
    p.add_argument('--out',type=Path,default=ROOT/'experiments/results/ct2_default_evaluation');args=p.parse_args()
    if args.repeats<1:p.error('repeats must be positive')
    transformers.utils.logging.set_verbosity_error();torch.set_num_threads(4);torch.manual_seed(0)
    engines={'torch_fp32':TorchReference(args.beams,args.budget),'ct2_fp32':CT2Translator(precision='float32',num_beams=args.beams,max_decoding_length=args.budget),
             'ct2_int8':CT2Translator(precision='int8',num_beams=args.beams,max_decoding_length=args.budget)}
    cases=json.loads((ROOT/'experiments/results/constrained_evaluation/cases.json').read_text(encoding='utf-8'))
    out=args.out;out.mkdir(exist_ok=True)
    chunks=[];ranges=[]
    for case in cases:
        parts=engines['torch_fp32']._split_long_text(case['source'])
        for engine in engines.values():assert engine._split_long_text(case['source'])==parts
        ranges.append((len(chunks),len(chunks)+len(parts)));chunks.extend(parts)
    quality={}
    for name,engine in engines.items():
        engine._generate(['你好。'])
        start=time.perf_counter();outputs=engine._generate(chunks)
        quality[name]=dict(outputs=[' '.join(outputs[a:b]) for a,b in ranges],wall_s=time.perf_counter()-start)
        print('quality',name,round(quality[name]['wall_s'],3),flush=True)
    metrics={}
    for split in dict.fromkeys(c['split'] for c in cases):
        metrics[split]={};indices=[i for i,c in enumerate(cases) if c['split']==split]
        for name,q in quality.items():
            bleu=BLEU();chrf=CHRF(word_order=2)
            hypotheses=[q['outputs'][i] for i in indices];refs=[[cases[i]['reference'] for i in indices]]
            metrics[split][name]=dict(BLEU=bleu.corpus_score(hypotheses,refs).score,
                chrF2pp=chrf.corpus_score(hypotheses,refs).score,
                BLEU_signature=str(bleu.get_signature()),chrF_signature=str(chrf.get_signature()),cases=len(indices),
                exact_same_as_torch=sum(q['outputs'][i]==quality['torch_fp32']['outputs'][i] for i in indices))
    benchmark={}
    selected=chunks[:12]
    for batch_size in [1,4]:
        times={name:[] for name in engines};recorded={}
        for rep in range(args.repeats):
            order=list(engines);order=order[rep%3:]+order[:rep%3]
            for name in order:
                engine=engines[name];engine.batch_size=batch_size
                start=time.perf_counter();result=engine._generate(selected);duration=time.perf_counter()-start
                if name in recorded and result!=recorded[name]:raise RuntimeError('Repeated decoding changed')
                recorded[name]=result;times[name].append(duration)
        benchmark[str(batch_size)]={name:dict(seconds=ts,median_s=statistics.median(ts),sentences=len(selected)) for name,ts in times.items()}
        print('benchmark',batch_size,{k:round(v['median_s'],3) for k,v in benchmark[str(batch_size)].items()},flush=True)
    rows=[dict(**case,outputs={name:q['outputs'][i] for name,q in quality.items()}) for i,case in enumerate(cases)]
    (out/'outputs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    conversion={precision:json.loads((ROOT/f'translator/ct2_zh_en_{precision}/conversion.json').read_text(encoding='utf-8')) for precision in ['float32','int8']}
    metadata=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                  ctranslate2=ctranslate2.__version__,device='cpu',threads=4,beam_size=args.beams,length_penalty=1.0,
                  max_new_tokens=args.budget,repeats=args.repeats,batch_sizes=[1,4],
                  cpu_supported_compute_types=sorted(ctranslate2.get_supported_compute_types('cpu')),
                  note='Same weights, same source segmentation and source tokenization, CPU-only paired benchmark. 24 assistant-authored technical references and 50 fixed FLORES-200 reference sentences. Quality measured once; timing subset first 12 segments, three repeats. Not human accuracy.',
                  hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'translator/ct2_translator.py',ROOT/'translator/reliable_translator.py',ROOT/'experiments/evaluate_ct2.py']})
    summary=dict(metadata=metadata,quality=metrics,benchmark=benchmark,conversion=conversion,
                 quality_wall_seconds={k:v['wall_s'] for k,v in quality.items()})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','split','source','reference','torch_fp32','ct2_fp32','ct2_int8']);writer.writeheader()
        for row in rows:writer.writerow({k:row[k] for k in ['id','split','source','reference']}|row['outputs'])
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
