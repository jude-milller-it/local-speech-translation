import argparse,json
from pathlib import Path
from .selective_translator import SelectiveTranslator


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--precision',choices=['int8','float32'],default='float32')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--mode',choices=['baseline','mbr','bidirectional','auto','selective'],default='selective')
    args=p.parse_args()
    translator=SelectiveTranslator(precision=args.precision,threads=args.threads,mode=args.mode)
    report=translator.translate_detailed(args.input.read_text(encoding='utf-8-sig'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(report['translation'],encoding='utf-8')
    args.output.with_suffix('.audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report['translation'])


if __name__=='__main__':main()
