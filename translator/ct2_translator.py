"""Official CTranslate2 Marian backend. No lexical constraints or text rewriting."""
from pathlib import Path
import json
import ctranslate2
from transformers import MarianTokenizer
from .reliable_translator import ReliableTranslator


class CT2Translator(ReliableTranslator):
    def __init__(self,precision='float32',device='cpu',threads=4,batch_size=4,num_beams=6,max_decoding_length=512):
        if precision not in {'int8','float32'}:raise ValueError('Unknown precision')
        root=Path(__file__).resolve().parent
        self.tokenizer=MarianTokenizer.from_pretrained(str(root/'opus-mt-zh-en'),local_files_only=True)
        self.engine=ctranslate2.Translator(str(root/('ct2_zh_en_'+precision)),device=device,
                 compute_type='int8' if precision=='int8' else 'float32',
                 intra_threads=threads,inter_threads=1)
        self.batch_size=batch_size
        self.num_beams=num_beams
        self.max_decoding_length=max_decoding_length
        self.max_token_length=512
        self.chunk_token_budget=128
        self.glossary=json.loads((root/'glossary.json').read_text(encoding='utf-8'))
        self.last_report=None
        self.precision=precision

    def _generate(self,chunks):
        if not chunks:return []
        # Transformers-converted models require explicit source special tokens.
        ids=[self.tokenizer.encode(text,add_special_tokens=True) for text in chunks]
        if any(len(x)>512 for x in ids):raise ValueError('No silent input truncation allowed')
        tokens=[self.tokenizer.convert_ids_to_tokens(x) for x in ids]
        result=self.engine.translate_batch(tokens,beam_size=self.num_beams,length_penalty=1.0,
                    patience=1,max_decoding_length=self.max_decoding_length,max_input_length=0,
                    max_batch_size=self.batch_size,batch_type='examples',return_scores=False)
        return [self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(x.hypotheses[0]),skip_special_tokens=True)
                for x in result]

    def translate_detailed(self,text,retry=False):
        if retry:raise ValueError('Automatic retry is not enabled for the CT2 backend')
        report=super().translate_detailed(text,retry=False)
        report.update(backend='CTranslate2',precision=self.precision,num_beams=self.num_beams)
        return report
