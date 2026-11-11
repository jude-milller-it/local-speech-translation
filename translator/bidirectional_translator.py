"""Inference-only direct/channel reranking, inspired by Yee et al. EMNLP 2019.
Not a full noisy-channel reproduction: no separate target-language-model term.
"""
import json,math,re,time
from pathlib import Path
from sacrebleu.metrics import CHRF
import ctranslate2
from transformers import MarianTokenizer
from .ct2_translator import CT2Translator
from .fact_checks import check

class BidirectionalTranslator(CT2Translator):
    MODES={'baseline','mbr','bidirectional','auto'}
    def __init__(self,precision='float32',mode='auto',currency_rescue=True,**kwargs):
        super().__init__(precision=precision,**kwargs)
        if mode not in self.MODES:raise ValueError('Unknown translation mode')
        self.mode=mode;self.reverse_engine=None;self.reverse_tokenizer=None
        self.root=Path(__file__).resolve().parent
        self.last_choices=[]
        self.metric=CHRF(word_order=0)
        self.reverse_weight=0.5;self.switch_margin=0.15;self.forward_tolerance=0.7
        self.currency_rescue=currency_rescue
        self.fact_repair_forward_tolerance=1.5
        self.hard_terms={}
        path=self.root/'required_terms.json'
        if path.exists():self.hard_terms=json.loads(path.read_text(encoding='utf8'))

    def _load_reverse(self):
        if self.reverse_engine is None:
            tokenizer=MarianTokenizer.from_pretrained(str(self.root/'opus-mt-en-zh'),local_files_only=True)
            engine=ctranslate2.Translator(str(self.root/'ct2_en_zh_float32'),device='cpu',compute_type='float32',intra_threads=4,inter_threads=1)
            self.reverse_tokenizer=tokenizer;self.reverse_engine=engine

    @staticmethod
    def tokens(tokenizer,text,target=False):
        ids=tokenizer(text_target=text,add_special_tokens=True)['input_ids'] if target else tokenizer.encode(text,add_special_tokens=True)
        if len(ids)>512:raise ValueError('Scoring input exceeds 512 tokens; no truncation')
        # CT2 score_batch appends EOS itself. Supplying it here would score EOS twice.
        if target and ids and ids[-1] == tokenizer.eos_token_id:
            ids=ids[:-1]
        return tokenizer.convert_ids_to_tokens(ids)

    @staticmethod
    def mean_score(result):
        # score_batch returns per-target-token log probabilities, including EOS.
        if not result.log_probs:raise ValueError('Empty conditional score')
        if not all(math.isfinite(x) for x in result.log_probs):raise ValueError('Nonfinite conditional score')
        return sum(result.log_probs)/len(result.log_probs)

    def _risk(self,source):
        return bool(re.search(r'\d|不|未|无|禁止|不得|[A-Za-z]',source) or any(x in source for x in list(self.glossary)+list(self.hard_terms)))

    @staticmethod
    def currency_hint(source):
        # Scope is literal yuan in a monetary context. Never normalize other
        # currencies, quotations, mixed-currency statements, or bare numbers.
        if not re.search(r'价格|售价|金额|运费|费用|付款|支付|单价|票价',source):return None
        if re.search(r'美元|日元|加元|澳元|港元|欧元|新元|韩元|[“”"「」]',source):return None
        hint=re.sub(r'(?<![\d.])(\d+(?:\.\d+)?)元',r'人民币\1元',source)
        if '人民币' in source or hint==source:return None
        return hint

    def _generate(self,chunks):
        if not chunks:self.last_choices=[];return []
        if self.mode=='baseline':
            out=super()._generate(chunks)
            self.last_choices=[dict(reason='baseline',candidates=[]) for _ in chunks]
            return out
        # Safe fast path: no n-best generation or reverse loading for ordinary text.
        active=[i for i,s in enumerate(chunks) if self.mode!='auto' or self._risk(s)]
        passive=[i for i in range(len(chunks)) if i not in active]
        output=[None]*len(chunks);choices=[None]*len(chunks)
        for i,text in zip(passive,super()._generate([chunks[i] for i in passive])):
            output[i]=text;choices[i]=dict(reason='fast_path',candidates=[])
        if not active:self.last_choices=choices;return output
        src=[self.tokens(self.tokenizer,chunks[i]) for i in active]
        results=self.engine.translate_batch(src,beam_size=6,num_hypotheses=6,length_penalty=1.0,patience=1,max_decoding_length=512,max_input_length=0,max_batch_size=self.batch_size,return_scores=True)
        groups=[]
        for result in results:
            candidates=[]
            for h in result.hypotheses:
                text=self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(h),skip_special_tokens=True)
                if text not in candidates:candidates.append(text)
            groups.append(candidates)
        if self.mode=='mbr':
            for i,candidates in zip(active,groups):
                utilities=[sum(self.metric.sentence_score(h,[r]).score for r in candidates)/len(candidates) for h in candidates]
                chosen=max(range(len(candidates)),key=utilities.__getitem__)
                output[i]=candidates[chosen];choices[i]=dict(reason='chrf_mbr',selected=chosen,candidates=candidates,utilities=utilities)
            self.last_choices=choices;return output
        if self.mode=='auto':
            kept_active=[];kept_groups=[]
            for i,cs in zip(active,groups):
                facts=check(chunks[i],cs[0],self.hard_terms)
                if facts['violations'] or facts['warnings'] or self.audit(chunks[i],cs[0]):
                    kept_active.append(i);kept_groups.append(cs)
                else:
                    output[i]=cs[0];choices[i]=dict(reason='checks_passed_skip_reverse',candidates=cs)
            active,groups=kept_active,kept_groups
            if not active:self.last_choices=choices;return output
        rescued={};rescue_errors={}
        if self.currency_rescue:
            requests=[]
            for i,cs in zip(active,groups):
                if all('currency_mismatch' in check(chunks[i],c,self.hard_terms)['violations'] for c in cs):
                    hint=self.currency_hint(chunks[i])
                    if hint:requests.append((i,hint))
            if requests:
                try:
                    extra=self.engine.translate_batch([self.tokens(self.tokenizer,hint) for i,hint in requests],beam_size=6,num_hypotheses=2,length_penalty=1.0,patience=1,max_decoding_length=512,max_input_length=0,max_batch_size=self.batch_size)
                    for (i,hint),result in zip(requests,extra):
                        cs=groups[active.index(i)]
                        for h in result.hypotheses:
                            text=self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(h),skip_special_tokens=True)
                            if text not in cs:
                                cs.append(text);rescued[(i,text)]=hint
                except Exception as error:
                    rescue_errors={i:str(error) for i,hint in requests}
        try:
            self._load_reverse()
            flat=[(i,c) for i,cs in zip(active,groups) for c in cs]
            direct_src=[self.tokens(self.tokenizer,chunks[i]) for i,c in flat]
            direct_tgt=[self.tokens(self.tokenizer,c,target=True) for i,c in flat]
            # en-zh is multilingual on its target side; request simplified Mandarin.
            reverse_src=[self.tokens(self.reverse_tokenizer,'>>cmn_Hans<< '+c) for i,c in flat]
            reverse_tgt=[self.tokens(self.reverse_tokenizer,chunks[i],target=True) for i,c in flat]
            forward=self.engine.score_batch(direct_src,direct_tgt,max_batch_size=4,max_input_length=0)
            reverse=self.reverse_engine.score_batch(reverse_src,reverse_tgt,max_batch_size=4,max_input_length=0)
            scored={i:[] for i in active}
            for (i,c),f,b in zip(flat,forward,reverse):
                fs=self.mean_score(f);bs=self.mean_score(b)
                scored[i].append(dict(text=c,forward=fs,reverse=bs,combined=fs+self.reverse_weight*bs,generation_source=rescued.get((i,c),chunks[i]),rescued=(i,c) in rescued,**check(chunks[i],c,self.hard_terms)))
            for i in active:
                cs=scored[i];base=cs[0];valid=[j for j,c in enumerate(cs) if not c['violations']]
                selected=0;reason='keep_original'
                if valid:
                    best=max(valid,key=lambda j:cs[j]['combined'])
                    if base['violations']:
                        # Do not replace a factual error with an extremely unlikely sentence.
                        if cs[best].get('rescued'):
                            if cs[best]['forward']>=base['forward']-self.fact_repair_forward_tolerance and cs[best]['reverse']>=base['reverse']+0.1:
                                selected=best;reason='currency_candidate_recovery'
                        elif cs[best]['forward']>=base['forward']-self.forward_tolerance:
                            selected=best;reason='fact_filter'
                    elif cs[best]['combined']>=base['combined']+self.switch_margin and cs[best]['forward']>=base['forward']-self.forward_tolerance:
                        selected=best;reason='bidirectional_margin'
                else:reason='no_valid_candidate_keep_and_warn'
                output[i]=cs[selected]['text'];choices[i]=dict(reason=reason,selected=selected,candidates=cs,rescue_error=rescue_errors.get(i))
        except Exception as error:
            # Optional scoring must not make the existing translator unavailable.
            for i,cs in zip(active,groups):
                output[i]=cs[0];choices[i]=dict(reason='scoring_unavailable_keep_original',error=str(error),candidates=cs)
        self.last_choices=choices
        return output

    def translate_detailed(self,text,retry=False):
        if retry:raise ValueError('Automatic rewriting is disabled')
        start=time.perf_counter()
        report=super().translate_detailed(text,retry=False)
        for segment,choice in zip(report['segments'],self.last_choices):
            segment['selection']=choice
            candidates=choice.get('candidates',[])
            if candidates:
                segment['initial']=candidates[0]['text'] if isinstance(candidates[0],dict) else candidates[0]
            facts=check(segment['source'],segment['translation'],self.hard_terms)
            segment['fact_checks']=facts
            labels={'empty_translation':'译文为空','percentage_needs_review':'请核对百分比或小数表达是否等价','number_mismatch':'数字可能遗漏或发生变化','date_mismatch':'日期可能发生变化','time_mismatch':'时间可能发生变化','percentage_mismatch':'百分比可能发生变化','from_to_order_mismatch':'起止数字顺序可能颠倒','direction_mismatch':'增加或减少的方向可能翻反','currency_mismatch':'货币单位可能发生变化','negation_needs_review':'请核对否定或禁止含义是否保留'}
            for code in facts['violations']+facts['warnings']:
                if code.startswith('required_term:'):message='必需术语未匹配：'+code.split(':',1)[1]
                elif code.startswith('name_needs_review:'):message='名称可能未保留：'+code.split(':',1)[1]
                else:message=labels.get(code,code)
                segment['warnings'].append(message)
            if choice.get('rescue_error'):segment['warnings'].append('货币候选补充失败，已使用原候选。')
            if choice.get('error'):segment['warnings'].append('反向评分不可用，已保留原译文：'+choice['error'])
        report.update(mode=self.mode,seconds=time.perf_counter()-start,limitations='Conditional likelihood and narrow fact rules cannot establish semantic correctness. No separate language-model term; not a full noisy-channel reproduction.')
        self.last_report=report
        return report
