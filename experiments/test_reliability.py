"""Focused regression tests: source preservation, decimals, audits and retry choice."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translator.reliable_translator import ReliableTranslator, numbers


class CharacterTokenizer:
    def encode(self, text):
        return [0] + list(text) + [1]


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReliableTranslator.__new__(ReliableTranslator)
        self.engine.tokenizer = CharacterTokenizer()
        self.engine.chunk_token_budget = 24
        self.engine.glossary = {'池化层': ['pooling layer']}

    def test_source_preserved_and_budgets(self):
        for text in ['中文，' * 40 + '价格39.9元。', '没有标点的长句' * 50,
                     '价格39.9元.第二阶段128个样本。', '值为0.001，数量50。']:
            chunks = self.engine._split_long_text(text)
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(self.engine.tokenizer.encode(x)) <= 24 for x in chunks))

    def test_decimal_and_identifier_not_broken(self):
        text = '中' * 18 + '39.9元' + '中' * 20
        chunks = self.engine._split_long_text(text)
        self.assertTrue(any('39.9' in x for x in chunks))

    def test_numbers_sentence_period_and_multiplicity(self):
        self.assertEqual(numbers('学习率0.001，训练50轮。'), numbers('Learning rate 0.001, train 50.'))
        self.assertEqual(sum((numbers('50和50') - numbers('50.')).values()), 1)
        self.assertEqual(numbers('1,000'), numbers('1000'))
        self.assertNotEqual(numbers('-5'), numbers('5'))

    def test_term_word_boundary(self):
        self.assertEqual(self.engine.audit('池化层', 'a pooling layer'), [])
        self.assertTrue(self.engine.audit('池化层', 'a pooling layercake'))

    def test_retry_accepts_only_improved_numeric_check(self):
        self.engine.chunk_token_budget = 128
        self.engine._generate = lambda chunks: ['there are samples'] if len(chunks) == 1 else ['17 samples', '29 samples']
        report = self.engine.translate_detailed('第一组17个，第二组29个。', retry=True)
        self.assertTrue(report['segments'][0]['retry_accepted'])
        self.assertEqual(report['segments'][0]['warnings'], [])
        self.engine._generate = lambda chunks: ['no numbers'] * len(chunks)
        report = self.engine.translate_detailed('第一组17个，第二组29个。', retry=True)
        self.assertTrue(report['segments'][0]['retry_attempted'])
        self.assertFalse(report['segments'][0]['retry_accepted'])
        self.assertEqual(report['translation'], 'no numbers')


if __name__ == '__main__':
    unittest.main()
