"""Small integration smoke through actual GUI translation/TTS slots and ASR worker."""
import os,sys,runpy,time,json
from pathlib import Path
import argparse
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('audio', type=Path, help='Path to a Chinese speech WAV file')
args = parser.parse_args()
audio_path = args.audio.resolve()
if not audio_path.is_file():
    parser.error('Audio file does not exist')
sys.argv[1:] = []
old_exec=QApplication.exec_
def check(app):
    g=sys._getframe(1).f_globals;window=g['window'];ap=window.asr_page;tp=window.trans_page;sp=window.tts_page
    import soundfile as sf
    clip,sr=sf.read(audio_path);out=ROOT/'experiments/results/bidirectional_small';out.mkdir(parents=True,exist_ok=True)
    source=out/'speech_input.wav';sf.write(source,clip,sr)
    times={};ap.current_audio_path=str(source)
    assert g['step_vad'](ap)
    assert g['step_preprocess'](ap)
    assert g['step_mfcc'](ap)
    start=time.perf_counter();assert g['step_asr'](ap)
    while ap._asr_worker is not None:
        app.processEvents();time.sleep(0.02)
        if time.perf_counter()-start>150:raise TimeoutError('ASR worker timeout')
    times['asr_s']=time.perf_counter()-start
    chinese=ap.result_text.toPlainText();assert chinese.strip(),ap.log_text.toPlainText()
    tp.src_text.setPlainText(chinese);tp.mode_combo.setCurrentIndex(tp.mode_combo.findData('selective'))
    start=time.perf_counter();tp.translate_btn.click()
    while getattr(tp,'_translation_worker',None) is not None:
        app.processEvents();time.sleep(0.01)
        if time.perf_counter()-start>120:raise TimeoutError('Translation worker timeout')
    times['translation_s']=time.perf_counter()-start
    english=tp.dst_text.toPlainText();assert english.strip(),tp.log_text.toPlainText()
    speech_translation_report=g['translator'].last_report
    before=set(Path(g['TTS_OUT_PATH']).glob('*.wav'));sp.input_text_edit.setPlainText(english)
    start=time.perf_counter();sp.single_synth_btn.click();times['tts_s']=time.perf_counter()-start
    files=set(Path(g['TTS_OUT_PATH']).glob('*.wav'))-before
    assert files,sp.log_text.toPlainText()
    wav=next(iter(files));assert sf.info(wav).frames>0
    import torch
    result={'input_seconds':len(clip)/sr,'chinese':chinese,'english':english,'times':times,'tts_file':wav.relative_to(ROOT).as_posix(),'tts_seconds':sf.info(wav).duration,'translation_device':'cpu','torch_cuda_peak_mb':torch.cuda.max_memory_allocated()/1024**2,'ui_log':tp.log_text.toPlainText(),'translation_report':speech_translation_report}
    (out/'speech_pipeline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('SPEECH_PIPELINE_OK',json.dumps(result,ensure_ascii=False),flush=True)
    QTimer.singleShot(200,app.quit);return old_exec()
QApplication.exec_=check
runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
