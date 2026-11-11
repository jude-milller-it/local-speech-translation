"""Translation-only background execution; GUI updates stay on the main thread."""
from PyQt5.QtCore import QThread,pyqtSignal

class TranslationWorker(QThread):
    item=pyqtSignal(int,str,object)
    def __init__(self,translator,texts,mode,parent=None):
        super().__init__(parent);self.translator=translator;self.texts=list(texts);self.mode=mode
    def run(self):
        self.translator.mode=self.mode
        for index,text in enumerate(self.texts,1):
            try:payload={'report':self.translator.translate_detailed(text)}
            except Exception as error:payload={'error':str(error)}
            self.item.emit(index,text,payload)

def start_translation(page,translator,texts,file_mode=False):
    # Retain the worker until finished, not merely until the final result signal.
    if getattr(page,'_translation_worker',None) is not None:
        page.log_text.append('翻译正在进行，请等待完成。');return False
    worker=TranslationWorker(translator,texts,page.mode_combo.currentData(),page)
    page._translation_worker=worker
    page.log_text.clear();page.dst_text.clear()
    page.log_text.append('正在翻译，请稍候；首次增强翻译需要加载本地模型。')
    for control in [page.translate_btn,page.file_btn,page.mode_combo]:control.setEnabled(False)
    page.src_text.setReadOnly(True)
    results=[]
    def on_item(index,source,payload):
        if 'error' in payload:
            translation='翻译失败：'+payload['error'];page.log_text.append(f'[第{index}条错误] '+payload['error'])
        else:
            report=payload['report'];translation=report['translation']
            for segment in report.get('segments',[]):
                for warning in segment.get('warnings',[]):page.log_text.append(f'[第{index}条待核对] '+warning)
        results.append(f'原文：{source}\n译文：{translation}\n' if file_mode else translation)
        page.dst_text.setPlainText('\n'.join(results))
    def on_finished():
        for control in [page.translate_btn,page.file_btn,page.mode_combo]:control.setEnabled(True)
        page.src_text.setReadOnly(False)
        page._translation_worker=None
        page.log_text.append('翻译任务结束。')
        worker.deleteLater()
    worker.item.connect(on_item)
    worker.finished.connect(on_finished)
    worker.start()
    return True
