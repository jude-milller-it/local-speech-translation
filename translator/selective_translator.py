"""Bounded, auditable pretrained-model fallback; no training or invented gains."""
import re
from .bidirectional_translator import BidirectionalTranslator
from .fact_checks import check
from .qwen_expert import QwenExpert

class SelectiveTranslator(BidirectionalTranslator):
    MODES=BidirectionalTranslator.MODES|{'selective','quality'}
    def __init__(self,mode='baseline',max_expert_segments=4,**kwargs):
        if mode not in self.MODES:raise ValueError('Unknown translation mode')
        super().__init__(mode='baseline',**kwargs)
        self.mode=mode;self.expert=None;self.max_expert_segments=max_expert_segments
        self.context_domain_terms=['人工智能','神经网络','机器学习']
        if max_expert_segments<1:raise ValueError('Expert segment budget must be positive')
        self.domain_terms=dict(self.glossary)
        self.domain_terms.update({'特征图':['feature map','feature maps'],'回归问题':['regression problem','regression task'],'分类问题':['classification problem','classification task'],'置信度':['confidence']})
    def _split_long_text(self,text):
        parts=super()._split_long_text(text)
        if self.mode not in {'selective','quality'}:return parts
        # Keep adjacent ASR fragments in the same bounded context window.
        packed=[];current=''
        for part in parts:
            if current and len(self.tokenizer.encode(current+part))>self.chunk_token_budget:
                packed.append(current);current=''
            current+=part
        if current:packed.append(current)
        return packed

    def _missing_terms(self,source,target):
        return [term for term,aliases in self.domain_terms.items() if term in source and not any(re.search(r'(?<!\w)'+re.escape(alias)+r'(?:s)?(?!\w)',target,re.I) for alias in aliases)]
    def _generate(self,chunks):
        if self.mode not in {'selective','quality'}:return super()._generate(chunks)
        # Reuse original inference directly; never invoke a stack of rerankers.
        from .ct2_translator import CT2Translator
        initial=CT2Translator._generate(self,chunks)
        output=list(initial);choices=[dict(reason='fast_checks_passed',candidates=[text]) for text in initial]
        requests=[];contextual=set()
        for i,(source,text) in enumerate(zip(chunks,initial)):
            facts=check(source,text,self.hard_terms);missing=self._missing_terms(source,text)
            context_risk=len(re.findall(r'[。！？]',source))>1 and any(term in source for term in getattr(self,'context_domain_terms',[]))
            if context_risk:contextual.add(i)
            high_confidence_warning=any(w.startswith('name_needs_review:') for w in facts['warnings']) or ('negation_needs_review' in facts['warnings'] and bool(re.search(r'请不要|禁止|不得|不能|不允许',source)))
            if self.mode=='quality' or facts['violations'] or high_confidence_warning or missing or context_risk:
                requests.append((i,len(facts['violations']),len(missing)))
        if self.mode=='selective':
            requests.sort(key=lambda item:(item[1],item[2]),reverse=True)
            for i,_,_ in requests[self.max_expert_segments:]:choices[i]['reason']='expert_budget_exhausted_keep_original'
            requests=requests[:self.max_expert_segments]
        if requests:
            try:
                if self.expert is None:self.expert=QwenExpert(self.root,self.domain_terms)
                generated=self.expert.translate([chunks[i] for i,_,_ in requests])
                if len(generated)!=len(requests):raise ValueError('Expert output count mismatch')
                for (i,_,_),candidate in zip(requests,generated):
                    old=check(chunks[i],initial[i],self.hard_terms);new=check(chunks[i],candidate,self.hard_terms)
                    old_missing=self._missing_terms(chunks[i],initial[i]);new_missing=self._missing_terms(chunks[i],candidate)
                    safe=bool(candidate.strip()) and len(new['violations'])<=len(old['violations']) and len(new_missing)<=len(old_missing)
                    # Do not trade a clean numeric fact for a different known error.
                    safe=safe and set(new['violations']).issubset(set(old['violations']))
                    if self.mode=='selective':
                        improves_known=(len(new['violations'])<len(old['violations']) or len(new_missing)<len(old_missing) or len(new['warnings'])<len(old['warnings']))
                        safe=safe and (improves_known or i in contextual)
                    if safe:output[i]=candidate
                    choices[i]=dict(reason=('expert_selected_context' if i in contextual else 'expert_selected') if safe else 'expert_rejected_by_checks',selected=int(safe),candidates=[initial[i],candidate],missing_terms_before=old_missing,missing_terms_after=new_missing,expert_fact_checks=new)
            except Exception as error:
                for i,_,_ in requests:
                    output[i]=initial[i]
                    choices[i]=dict(reason='expert_unavailable_keep_original',error=str(error),candidates=[initial[i]])
        self.last_choices=choices
        return output
    def translate_detailed(self,text,retry=False):
        report=super().translate_detailed(text,retry=retry)
        for segment in report['segments']:
            choice=segment['selection']
            if choice['reason']=='expert_budget_exhausted_keep_original':segment['warnings'].append('本段未执行增强翻译，请核对。')
            if choice['reason']=='expert_unavailable_keep_original':
                segment['warnings']=[x for x in segment['warnings'] if not x.startswith('反向评分不可用')]
                segment['warnings'].append('增强翻译不可用，已保留快速译文：'+choice['error'])
        if self.mode in {'selective','quality'}:
            report.update(backend='CTranslate2 MarianMT with optional Qwen fallback',expert_model='Qwen/Qwen2.5-0.5B-Instruct',expert_precision='int8',expert_device='cpu')
        report['limitations']='Pretrained-model fallback with narrow checks; not a guarantee of semantic correctness. Model substitution is reported explicitly; no training or original-model accuracy claim.'
        return report
