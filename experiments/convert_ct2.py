import hashlib,json
from pathlib import Path
import ctranslate2,torch
from ctranslate2.converters import TransformersConverter
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
source=ROOT/'translator/opus-mt-zh-en'
for precision in ['float32','int8']:
    target=ROOT/'translator'/('ct2_zh_en_'+precision)
    if not ctranslate2.contains_model(str(target)):
        TransformersConverter(str(source)).convert(str(target),quantization=precision)
    manifest=dict(source=str(source),source_model_sha256=hashlib.sha256((source/'pytorch_model.bin').read_bytes()).hexdigest(),
                  engine='CTranslate2',version=ctranslate2.__version__,quantization=precision,
                  model_bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and p.name!='conversion.json'),
                  source_model='https://huggingface.co/Helsinki-NLP/opus-mt-zh-en',
                  source_license='CC BY 4.0',algorithm_source='https://github.com/OpenNMT/CTranslate2')
    (target/'conversion.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)
