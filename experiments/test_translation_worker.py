import os,sys,time,unittest
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen';ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
from ui import TranslationPage,MainWindow
from translator.ui_worker import start_translation
APP=QApplication.instance() or QApplication([])
class Fake:
 def translate_detailed(self,text):
  time.sleep(.08)
  if text=='bad':raise ValueError('expected test error')
  return {'translation':'English '+text,'segments':[{'warnings':[]}]}
class WorkerTests(unittest.TestCase):
 def wait(self,page):
  start=time.perf_counter()
  while getattr(page,'_translation_worker',None) is not None:
   APP.processEvents();time.sleep(.002)
   self.assertLess(time.perf_counter()-start,5)
  APP.processEvents()
 def test_heartbeat_and_reentry(self):
  p=TranslationPage();ticks=[];timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(5)
  self.assertTrue(start_translation(p,Fake(),['hello']))
  self.assertFalse(start_translation(p,Fake(),['duplicate']))
  self.assertFalse(p.translate_btn.isEnabled())
  self.wait(p);timer.stop()
  self.assertGreater(len(ticks),3);self.assertEqual(p.dst_text.toPlainText(),'English hello');self.assertTrue(p.translate_btn.isEnabled())
 def test_error_continues_and_unlocks(self):
  p=TranslationPage();start_translation(p,Fake(),['bad','ok'],file_mode=True);self.wait(p)
  self.assertIn('expected test error',p.dst_text.toPlainText());self.assertIn('English ok',p.dst_text.toPlainText());self.assertTrue(p.mode_combo.isEnabled())
  start_translation(p,Fake(),['again']);self.wait(p);self.assertEqual(p.dst_text.toPlainText(),'English again')
 def test_close_while_running(self):
  window=MainWindow();start_translation(window.trans_page,Fake(),['hello']);event=QCloseEvent();window.closeEvent(event);self.assertFalse(event.isAccepted());self.wait(window.trans_page)
if __name__=='__main__':unittest.main()
