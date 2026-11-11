import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'runtime/ct2'),str(ROOT)]
from unittest.mock import patch
from translator.selective_translator import SelectiveTranslator
from translator.ct2_translator import CT2Translator
class Expert:
 def __init__(self,output):self.output=output;self.calls=[]
 def translate(self,texts):self.calls.append(texts);return [self.output for _ in texts]
class SelectiveTests(unittest.TestCase):
 def make(self,output,mode='selective'):
  x=SelectiveTranslator.__new__(SelectiveTranslator);x.mode=mode;x.hard_terms={};x.domain_terms={'池化层':['pooling layer']};x.max_expert_segments=1;x.expert=Expert(output);return x
 def test_fast_path_does_not_call_expert(self):
  x=self.make('ignored')
  with patch.object(CT2Translator,'_generate',return_value=['Open the window.']):self.assertEqual(x._generate(['打开窗户。']),['Open the window.'])
  self.assertFalse(x.expert.calls)
 def test_terminology_fallback(self):
  x=self.make('Use a pooling layer.')
  with patch.object(CT2Translator,'_generate',return_value=['Use a pool layer.']):self.assertEqual(x._generate(['使用池化层。']),['Use a pooling layer.'])
  self.assertEqual(x.last_choices[0]['reason'],'expert_selected')
 def test_expert_cannot_change_number(self):
  x=self.make('Use 13 pooling layers.')
  with patch.object(CT2Translator,'_generate',return_value=['Use 30 pool layers.']):self.assertEqual(x._generate(['使用30个池化层。']),['Use 30 pool layers.'])
  self.assertEqual(x.last_choices[0]['reason'],'expert_rejected_by_checks')
 def test_budget_is_enforced(self):
  x=self.make('Use a pooling layer.')
  with patch.object(CT2Translator,'_generate',return_value=['Use a pool layer.']*2):x._generate(['使用池化层。']*2)
  self.assertEqual(len(x.expert.calls[0]),1);self.assertEqual(x.last_choices[1]['reason'],'expert_budget_exhausted_keep_original')
 def test_optional_model_failure_keeps_original(self):
  x=self.make('ignored')
  x.expert.translate=lambda texts:(_ for _ in ()).throw(RuntimeError('unavailable'))
  with patch.object(CT2Translator,'_generate',return_value=['Use a pool layer.']):self.assertEqual(x._generate(['使用池化层。']),['Use a pool layer.'])
  self.assertEqual(x.last_choices[0]['reason'],'expert_unavailable_keep_original')
 def test_context_packing_preserves_text(self):
  x=self.make('unused')
  class Tokenizer:
   def encode(self,text):return [0]+list(text)+[1]
  x.tokenizer=Tokenizer();x.chunk_token_budget=24
  text='人工智能是一门学科。它研究智能。'
  parts=x._split_long_text(text)
  self.assertEqual(''.join(parts),text);self.assertEqual(len(parts),1)
  text=text*4;parts=x._split_long_text(text)
  self.assertEqual(''.join(parts),text)
  self.assertTrue(all(len(x.tokenizer.encode(p))<=24 for p in parts))
 def test_partial_failure_rolls_back_all_outputs(self):
  x=self.make('unused');x.max_expert_segments=2
  x.expert.translate=lambda texts:['Use a pooling layer.',None]
  with patch.object(CT2Translator,'_generate',return_value=['Use a pool layer.']*2):
   self.assertEqual(x._generate(['使用池化层。']*2),['Use a pool layer.']*2)
  self.assertTrue(all(c['reason']=='expert_unavailable_keep_original' for c in x.last_choices))
if __name__=='__main__':unittest.main()

