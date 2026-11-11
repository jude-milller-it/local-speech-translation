import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from translator.fact_checks import check
class Facts(unittest.TestCase):
 def test_reversed_order(self):
  self.assertIn('from_to_order_mismatch',check('价格从30元降到13元。','The price fell from 13 yuan to 30 yuan.')['violations'])
 def test_spelled_numbers(self):
  self.assertFalse(check('价格从30元降到13元。','The price fell from thirty yuan to thirteen yuan.')['violations'])
 def test_shared_percent(self):
  self.assertFalse(check('准确率从90%提高到95%。','Accuracy increased from 90 to 95 percent.')['violations'])
 def test_date(self):
  self.assertFalse(check('会议是2026年11月7日。','The meeting is on November 7, 2026.')['violations'])
  self.assertIn('date_mismatch',check('会议是2026年11月7日。','The meeting is on 8 November 2026.')['violations'])
 def test_time_format(self):
  self.assertFalse(check('会议下午3点开始。','The meeting starts at 15:00.')['violations'])
  self.assertFalse(check('会议15点开始。','The meeting starts at 3 p.m.')['violations'])
  self.assertIn('time_mismatch',check('会议下午3点开始。','The meeting starts at 3 a.m.')['violations'])
 def test_unknown_number_format_abstains(self):
  self.assertFalse(check('人数是100人。','There are one hundred people.')['violations'])
 def test_negation_advisory(self):
  x=check('请不要关闭电源。','Turn off the power.')
  self.assertFalse(x['violations']);self.assertIn('negation_needs_review',x['warnings'])
 def test_term_opt_in(self):
  x=check('请使用池化层。','Use the pool layer.',{'池化层':['pooling layer']})
  self.assertIn('required_term:池化层',x['violations'])
 def test_name_advisory(self):
  self.assertIn('name_needs_review:Python',check('请使用Python。','Please use a tool.')['warnings'])
 def test_half_hour(self):
  self.assertFalse(check('会议下午3点半开始。','The meeting starts at 3:30 p.m.')['violations'])
 def test_mixed_chinese_numerals_abstain(self):
  self.assertFalse(check('请下载3个文件，然后选择其中一个。','Download 3 files and select one.')['violations'])
 def test_independent_direction_clause(self):
  self.assertFalse(check('价格从30元降到13元，销量增加。','The price fell from 30 yuan to 13 yuan, and sales increased.')['violations'])
 def test_non_cny_yuan_abstain(self):
  self.assertFalse(check('价格为30加元。','The price is 30 Canadian dollars.')['violations'])
 def test_shared_time_suffix(self):
  self.assertFalse(check('会议从上午9点持续到11点。','The meeting lasted from 9 to 11 a.m.')['violations'])
  self.assertIn('time_mismatch',check('会议从上午9点持续到11点。','The meeting lasted until 11 a.m.')['violations'])
 def test_attached_metric_units(self):
  self.assertFalse(check('数据大小为30GB。','The data size is 30GB.')['violations'])
  self.assertFalse(check('重量为30公斤。','The weight is 30kg.')['violations'])
 def test_currency_prefix_order(self):
  self.assertIn('from_to_order_mismatch',check('价格从人民币30元降到人民币13元。','The price fell from RMB 13 to RMB 30.')['violations'])
  self.assertFalse(check('价格从人民币30元降到人民币13元。','The price fell from RMB 30 to RMB 13.')['violations'])
 def test_empty_output_rejected(self):
  self.assertIn('empty_translation',check('请打开窗户。','')['violations'])
 def test_chinese_hour_and_noon(self):
  self.assertIn('time_mismatch',check('请在明天上午九点前提交报告。','Please submit the report by noon tomorrow.')['violations'])
  self.assertFalse(check('会议下午两点开始。','The meeting starts at 2 p.m.')['violations'])
 def test_unspecified_meridiem_abstains(self):
  self.assertFalse(check('会议3点开始。','The meeting starts at 3 p.m.')['violations'])
  self.assertFalse(check('会议下午3点到5点举行。','The meeting runs from 3 to 5 p.m.')['violations'])
 def test_evening_marker(self):
  self.assertFalse(check('会议傍晚六点开始。','The meeting starts at 6 p.m.')['violations'])
  self.assertIn('time_mismatch',check('会议傍晚六点开始。','The meeting starts at 6 a.m.')['violations'])
if __name__=='__main__':unittest.main()

