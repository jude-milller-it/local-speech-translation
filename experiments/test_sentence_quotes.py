import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from translator.reliable_translator import ReliableTranslator


class CharTokenizer:
    def encode(self,text):return [0]+list(text)+[1]


class QuoteRegressions(unittest.TestCase):
    def setUp(self):
        self.t=ReliableTranslator.__new__(ReliableTranslator)
        self.t.tokenizer=CharTokenizer();self.t.chunk_token_budget=128
    def test_chinese_quote_stays_with_sentence(self):
        source='他说：“请保存结果。”'
        self.assertEqual(self.t._split_long_text(source),[source])
    def test_ascii_quote_stays_with_sentence(self):
        source='他说："请保存结果。"然后关闭窗口。'
        self.assertEqual(self.t._split_long_text(source),['他说："请保存结果。"','然后关闭窗口。'])
    def test_nested_closers(self):
        source='他说：“请保存结果！”）然后关闭窗口。'
        chunks=self.t._split_long_text(source)
        self.assertEqual(chunks[0],'他说：“请保存结果！”）')
        self.assertEqual(''.join(chunks),source)
    def test_decimals_preserved(self):
        source='数值是39.9。第二阶段128个样本.最后阶段256个。'
        chunks=self.t._split_long_text(source)
        self.assertEqual(''.join(chunks),source)
        self.assertIn('39.9',chunks[0])
        self.assertEqual(len(chunks),3)
    def test_long_quoted_text_source_preserved(self):
        source='他说：“'+('这是用于检查源文保留的长句，'*20)+'最终答案42。”'
        chunks=self.t._split_long_text(source)
        self.assertEqual(''.join(chunks),source)
        self.assertFalse(any(x in {'”','"','）'} for x in chunks))


if __name__=='__main__':unittest.main()
