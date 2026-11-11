"""Local MarianMT improvements; compatible translate(str) interface.

Audits are conservative review hints, never a guarantee of semantic correctness.
"""
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
import json
import torch
from .translator_core import ChineseToEnglishTranslator as LegacyTranslator


def numbers(text):
    # Exclude embedded identifiers; ignore formatting commas, preserve signs.
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    return Counter(str(Decimal(m.group()).normalize()) for m in
                   re.finditer(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?!\w|\.\d)',
                               re.sub(r'[\u4e00-\u9fff]', ' ', text)))


class ReliableTranslator(LegacyTranslator):
    def __init__(self, model_path=None, batch_size=4, device='cpu', glossary_path=None):
        super().__init__(str(model_path or Path(__file__).parent / 'opus-mt-zh-en'))
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        glossary_path = glossary_path or Path(__file__).parent / 'glossary.json'
        self.glossary = json.loads(Path(glossary_path).read_text(encoding='utf-8'))
        self.last_report = None

    def _split_long_text(self, text):
        # Split at sentence boundaries, including a sentence after a digit.
        # Keep closing quotation/bracket marks attached to the sentence. Sending
        # a standalone closing quote to Marian can generate unrelated boilerplate.
        parts = re.split(r'''([。！？!?]|\.(?!\d))([”’"'）)】》]*)''', text)
        sentences = [parts[i] + (parts[i+1] if i+1 < len(parts) else '') +
                     (parts[i+2] if i+2 < len(parts) else '')
                     for i in range(0, len(parts), 3)]
        chunks = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(self.tokenizer.encode(sentence)) <= self.chunk_token_budget:
                chunks.append(sentence)
                continue
            # Long sentences prefer clause boundaries; hard split only a clause
            # with no usable punctuation. Source concatenation remains unchanged.
            clauses = re.split(r'(?<=[，,；;：:])', sentence)
            current = ''
            for clause in clauses:
                if not clause:
                    continue
                if current and len(self.tokenizer.encode(current + clause)) > self.chunk_token_budget:
                    chunks.append(current)
                    current = ''
                if len(self.tokenizer.encode(clause)) > self.chunk_token_budget:
                    if current:
                        chunks.append(current)
                        current = ''
                    chunks.extend(self._split_char_safe(clause))
                else:
                    current += clause
            if current:
                chunks.append(current)
        return chunks

    def _split_char_safe(self, text):
        # Cut original characters rather than decode partial subword sequences.
        result, start = [], 0
        while start < len(text):
            lo, hi = start + 1, len(text)
            if len(self.tokenizer.encode(text[start:lo])) > self.chunk_token_budget:
                raise ValueError('A character exceeds the token budget')
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(self.tokenizer.encode(text[start:mid])) <= self.chunk_token_budget:
                    lo = mid
                else:
                    hi = mid - 1
            end = lo
            # Avoid splitting ordinary decimal numbers or ASCII identifiers.
            if end < len(text) and re.match(r'[\w.]', text[end], re.ASCII):
                back = end
                while back > start and re.match(r'[\w.]', text[back - 1], re.ASCII):
                    back -= 1
                if back > start:
                    end = back
            result.append(text[start:end])
            start = end
        return result

    def _generate(self, chunks):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start:start + self.batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True,
                                        truncation=False).to(self.device)
                if inputs['input_ids'].shape[1] > self.max_token_length:
                    raise ValueError('Input exceeds model budget; no silent truncation allowed')
                generated = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs

    def audit(self, source, target):
        missing = list((numbers(source) - numbers(target)).elements())
        warnings = []
        if missing:
            warnings.append('疑似数字遗漏或格式变化：' + ', '.join(missing))
        for term, alternatives in self.glossary.items():
            if term in source and not any(re.search(r'(?<!\w)' + re.escape(x) + r'(?!\w)', target, re.I)
                                          for x in alternatives):
                warnings.append('术语待核对：' + term)
        if not target.strip():
            warnings.append('译文为空')
        return warnings

    def translate_detailed(self, text, retry=False):
        text = self._preprocess(text)
        chunks = self._split_long_text(text)
        translated = self._generate(chunks)
        segments = []
        for source, initial in zip(chunks, translated):
            chosen, attempted, accepted = initial, False, False
            # A narrowly scoped recovery, without inserting missing facts into output.
            if retry and numbers(source) - numbers(initial):
                pieces = [x for x in re.split(r'(?<=[，,；;：:])', source) if x]
                if len(pieces) > 1:
                    attempted = True
                    candidate = ' '.join(self._generate(pieces))
                    if sum((numbers(source) - numbers(candidate)).values()) < sum((numbers(source) - numbers(initial)).values()):
                        chosen, accepted = candidate, True
            segments.append(dict(source=source, initial=initial, translation=chosen,
                                 retry_attempted=attempted, retry_accepted=accepted,
                                 warnings=self.audit(source, chosen)))
        report = dict(translation=' '.join(x['translation'] for x in segments), segments=segments,
                      limitations='Rule checks do not verify negation, factuality or overall translation quality.')
        self.last_report = report
        return report

    def translate(self, text):
        return self.translate_detailed(text)['translation']
