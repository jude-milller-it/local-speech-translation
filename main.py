import os
import re
from datetime import datetime
import sys
import librosa
import numpy as np
import soundfile as sf
import whisper
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog

# Use the bundled working eSpeak build before importing Coqui TTS.
_ESPEAK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime", "espeak151", "bin")
if os.path.isfile(os.path.join(_ESPEAK_DIR, "espeak-ng.exe")):
    os.environ["PATH"] = _ESPEAK_DIR + os.pathsep + os.environ.get("PATH", "")
    os.environ["ESPEAK_DATA_PATH"] = os.path.join(_ESPEAK_DIR, "espeak-ng-data")

from ui import MainWindow
from audio_tts.tts_core import text_to_speech
from TTS.api import TTS
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime', 'ct2'))
from translator.selective_translator import SelectiveTranslator as ChineseToEnglishTranslator
from voice_to_word.chinese_asr_functions import voice_activity_detection, preprocess_audio, extract_mfcc_features, \
    speech_recognition_whisper

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# The bundled TTS config also contains project-relative resource paths.
os.chdir(PROJECT_DIR)

def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)

TTS_MODEL_PATH = project_path("audio_tts", "tts_models", "model_file.pth")  # tts模型文件路径
TTS_CONFIG_PATH = project_path("audio_tts", "tts_models", "config.json")  # tts配置信息路径
TTS_OUT_PATH = project_path("outputs", "tts_out_audio")  # tts生成的语音保存的文件夹路径
TRANSLATER_MODEL_PATH = project_path("translator", "opus-mt-zh-en")  # 翻译模型的配置信息路径
WHISPER_MODEL_PATH = project_path("voice_to_word", "large-v3.pt")  # whisper模型文件路径
VOICED_AUDIO = project_path("outputs", "voiced_audio")  # 端点检测截取后的保存路径
MFCC_FEATURES = project_path("outputs", "mfcc_features")

os.makedirs(TTS_OUT_PATH, exist_ok=True)
os.makedirs(VOICED_AUDIO, exist_ok=True)
os.makedirs(MFCC_FEATURES, exist_ok=True)

tts = TTS(
    model_path=TTS_MODEL_PATH,
    config_path=TTS_CONFIG_PATH,
    gpu=False,
    progress_bar=False
)
translator = ChineseToEnglishTranslator(precision='float32', mode='selective')
whisper_model = whisper.load_model(WHISPER_MODEL_PATH)


def t1_load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    return audio, sr


def t1_detection(audio, sr):
    voiced_audio, start_idx, end_idx = voice_activity_detection(audio, sr)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"voiced_audio{timestamp}.wav"
    voiced_audio_path = os.path.join(VOICED_AUDIO, filename)
    sf.write(voiced_audio_path, voiced_audio, sr)
    return voiced_audio_path


def t1_preprocess_audio(voiced_audio, sr):
    frames = preprocess_audio(voiced_audio, sr)
    return frames.shape[0] if frames is not None else 0


def t1_mfcc_extract(voiced_audio, sr):
    mfcc_features = extract_mfcc_features(voiced_audio, sr)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"mfcc_features{timestamp}.npy"
    mfcc_features_path = os.path.join(MFCC_FEATURES, filename)
    np.save(mfcc_features_path, mfcc_features)
    return mfcc_features_path


def t1_speech_recognition(re_audio_path):
    recognition_result = speech_recognition_whisper(
        whisper_model, re_audio_path)
    return recognition_result


# ---------- UI 槽函数 ----------
def select_audio_file(asr_page):
    """选择WAV文件"""
    file_path, _ = QFileDialog.getOpenFileName(
        asr_page, "选择音频文件", "", "WAV Files (*.wav);;All Files (*)"
    )
    if file_path:
        asr_page.current_audio_path = file_path
        asr_page.file_label.setText(file_path)
        asr_page.log_text.clear()
        asr_page.result_text.clear()
        asr_page.voiced_audio_path = None
        asr_page.mfcc_features_path = None
        asr_page.asr_result_text = ""
        asr_page.log_text.append(f"[信息] 已选择文件: {file_path}")
    else:
        asr_page.current_audio_path = None
        asr_page.file_label.setText("未选择文件")


def step_vad(asr_page):
    """语音端点检测"""
    if not asr_page.current_audio_path:
        asr_page.log_text.append("[错误] 请先选择WAV文件")
        return False
    try:
        asr_page.log_text.append("[步骤1] 加载音频...")
        audio, sr = t1_load_audio(asr_page.current_audio_path)
        asr_page.log_text.append(
            "        采样率: {} Hz, 时长: {:.2f}s".format(sr, len(audio) / sr)
        )
        asr_page.log_text.append("[步骤1] 进行语音端点检测...")
        voiced_path = t1_detection(audio, sr)
        asr_page.voiced_audio_path = voiced_path
        # 重新读取以获取时长信息
        voiced_audio, voiced_sr = librosa.load(voiced_path, sr=None)
        asr_page.log_text.append(f"        有效语音已保存至: {voiced_path}")
        asr_page.log_text.append(
            "        有效语音时长: {:.2f}s".format(len(voiced_audio) / voiced_sr)
        )
        asr_page.log_text.append("[步骤1] 端点检测完成。")
        return True
    except Exception as e:
        asr_page.log_text.append(f"[错误] 端点检测失败: {str(e)}")
        return False


def step_preprocess(asr_page):
    """语音预处理"""
    if not asr_page.voiced_audio_path:
        asr_page.log_text.append("[错误] 请先执行语音端点检测，获得有效语音片段")
        return False
    try:
        asr_page.log_text.append("[步骤2] 加载有效语音...")
        voiced_audio, sr = librosa.load(asr_page.voiced_audio_path, sr=None)
        asr_page.log_text.append("[步骤2] 进行预处理（分帧、预加重、加窗）...")
        n_frames = t1_preprocess_audio(voiced_audio, sr)
        asr_page.log_text.append(f"        预处理完成，帧数: {n_frames}")
        asr_page.log_text.append("        参数：预加重系数0.97，帧长2048，帧移512")
        asr_page.log_text.append("[步骤2] 预处理完成。")
        return True
    except Exception as e:
        asr_page.log_text.append(f"[错误] 预处理失败: {str(e)}")
        return False


def step_mfcc(asr_page):
    """提取MFCC特征"""
    if not asr_page.voiced_audio_path:
        asr_page.log_text.append("[错误] 请先执行语音端点检测，获得有效语音片段")
        return False
    try:
        asr_page.log_text.append("[步骤3] 加载有效语音...")
        voiced_audio, sr = librosa.load(asr_page.voiced_audio_path, sr=None)
        asr_page.log_text.append("[步骤3] 提取MFCC特征...")
        mfcc_path = t1_mfcc_extract(voiced_audio, sr)
        asr_page.mfcc_features_path = mfcc_path
        # 读取特征矩阵以显示形状
        mfcc = np.load(mfcc_path)
        asr_page.log_text.append(f"        MFCC特征已保存至: {mfcc_path}")
        asr_page.log_text.append(f"        特征矩阵形状: {mfcc.shape} (帧数 × 特征维度)")
        asr_page.log_text.append("[步骤3] MFCC提取完成。")
        return True
    except Exception as e:
        asr_page.log_text.append(f"[错误] MFCC提取失败: {str(e)}")
        return False


class ASRWorker(QThread):
    # success, text, lang, error_msg
    finished = pyqtSignal(bool, str, str, str)
    log_signal = pyqtSignal(str)

    def __init__(self, voiced_path):
        super().__init__()
        self.voiced_path = voiced_path

    def run(self):
        try:
            self.log_signal.emit("[步骤4] 使用Whisper进行语音识别...")
            result = t1_speech_recognition(self.voiced_path)
            text = result.get('text', '')
            lang = result.get('language', '')
            self.log_signal.emit(f"        识别语言: {lang}")
            self.log_signal.emit(f"        识别文本: {text}")
            self.log_signal.emit("[步骤4] 语音识别完成。")
            self.finished.emit(True, text, lang, "")
        except Exception as e:
            self.log_signal.emit(f"[错误] 语音识别失败: {str(e)}")
            self.finished.emit(False, "", "", str(e))


def step_asr(asr_page):
    """语音识别"""
    if not asr_page.voiced_audio_path:
        asr_page.log_text.append("[错误] 请先执行语音端点检测，获得有效语音片段")
        return False

    # 如果已有正在运行的线程，先不处理
    if hasattr(
            asr_page, '_asr_worker') and asr_page._asr_worker is not None and asr_page._asr_worker.isRunning():
        asr_page.log_text.append("[提示] 语音识别已在执行中，请稍后")
        return False

    # 禁用按钮，避免重复点击
    asr_page.asr_btn.setEnabled(False)

    def on_finished(success, text, lang, error):
        if success:
            asr_page.asr_result_text = text
            asr_page.result_text.setPlainText(text)
        else:
            asr_page.log_text.append(f"[错误] 识别失败: {error}")
        asr_page.asr_btn.setEnabled(True)
        # 清理线程引用
        asr_page._asr_worker = None
        asr_page.log_text.append("========== 中文语音识别全流程处理完成 ==========")

    worker = ASRWorker(asr_page.voiced_audio_path)
    worker.finished.connect(on_finished)
    worker.log_signal.connect(asr_page.log_text.append)
    worker.start()
    asr_page._asr_worker = worker

    return True


def t1_full_pipeline(asr_page):
    """一键全流程：端点检测 → 预处理 → MFCC → 识别"""
    if not asr_page.current_audio_path:
        asr_page.log_text.append("[错误] 请先选择WAV文件")
        return
    asr_page.log_text.append("========== 开始全流程处理 ==========")
    if not step_vad(asr_page):
        return
    if not step_preprocess(asr_page):
        return
    if not step_mfcc(asr_page):
        return
    if not step_asr(asr_page):
        return


def translate_single(trans_page):
    from translator.ui_worker import start_translation
    src = trans_page.src_text.toPlainText().strip()
    if not src:
        trans_page.log_text.append("[提示] 请输入中文文本")
        return
    start_translation(trans_page, translator, [src])


def translate_file(trans_page):
    from translator.ui_worker import start_translation
    if getattr(trans_page, '_translation_worker', None) is not None:
        trans_page.log_text.append("翻译正在进行，请等待完成。")
        return
    file_path, _ = QFileDialog.getOpenFileName(trans_page, "选择文本文件", "", "TXT Files (*.txt);;All Files (*)")
    if not file_path:
        return
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            lines = [line.strip() for line in file if line.strip()]
        if not lines:
            trans_page.log_text.append("[错误] 文件为空或无有效文本行")
            return
    except Exception as error:
        trans_page.log_text.append(f"[错误] 读取文件失败：{error}")
        return
    start_translation(trans_page, translator, lines, file_mode=True)


# TTS 文件选择与合成
def select_tts_file(tts_page):
    """选择英文txt文件，仅保存路径"""
    from PyQt5.QtWidgets import QFileDialog
    file_path, _ = QFileDialog.getOpenFileName(
        tts_page, "选择英文文本文件", "", "TXT Files (*.txt);;All Files (*)"
    )
    if file_path:
        tts_page.current_file_path = file_path
        tts_page.file_label.setText(file_path)
        tts_page.log_text.clear()
        tts_page.log_text.append(f"[信息] 已选择文件: {file_path}")
    else:
        tts_page.current_file_path = None
        tts_page.file_label.setText("未选择文件")


def synthesize_tts(tts_page):
    """根据已选择的文件、语速、音色，逐行合成语音"""
    if not tts_page.current_file_path:
        tts_page.log_text.append("[错误] 请先选择英文TXT文件")
        return

    file_path = tts_page.current_file_path
    tts_page.log_text.append(f"[信息] 使用文件: {file_path}")

    # 读取所有非空行
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            tts_page.log_text.append("[错误] 文件为空或无有效文本行")
            return
        tts_page.log_text.append(f"共读取 {len(lines)} 行英文文本")
    except Exception as e:
        tts_page.log_text.append(f"[错误] 读取文件失败: {str(e)}")
        return

    # 获取合成参数
    speed_percent = tts_page.speed_slider.value()
    speed = speed_percent / 100.0  # 转换为小数
    voice_index = tts_page.voice_combo.currentIndex()  # 0~3
    voice_code = voice_index + 1  # 映射为1~4（对应p231/p258/p260/p273）

    # 输出目录使用全局变量 TTS_OUT_PATH
    output_dir = TTS_OUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    tts_page.log_text.append("开始逐行合成...")
    QApplication.processEvents()

    success_count = 0
    for i, line in enumerate(lines, start=1):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_filename = "{}_line{}_voice{}_speed{}.wav".format(
            base_name, i, voice_code, speed_percent)
        out_path = os.path.join(output_dir, out_filename)

        tts_page.log_text.append(f"正在合成第 {i}/{len(lines)} 行：{line[:50]}...")
        QApplication.processEvents()

        try:
            from audio_tts.tts_core import text_to_speech
            # 注意 text_to_speech 参数顺序: (tts, text, voice_code, speed,
            # output_path)
            text_to_speech(tts, line, voice_code, speed, out_path)
            tts_page.log_text.append(f"  已保存: {out_path}")
            success_count += 1
        except Exception as e:
            tts_page.log_text.append(f"  [错误] 合成失败: {str(e)}")

        QApplication.processEvents()

    tts_page.log_text.append(
        f"\n合成完成！成功 {success_count}/{len(lines)} 个文件，保存在 {output_dir} 目录")
    tts_page.waveform_label.setText(
        f"【合成完成】共生成 {success_count} 个音频文件，请查看目录 {output_dir}，波形/频谱图差异在报告中展示。")


def synthesize_single_text(tts_page):
    """根据文本框内容合成单个音频文件"""
    text = tts_page.input_text_edit.toPlainText().strip()
    if not text:
        tts_page.log_text.append("[错误] 请输入英文文本")
        return

    text = text.replace('\n', ' ')
    # 获取参数
    speed_percent = tts_page.speed_slider.value()
    speed = speed_percent / 100.0
    voice_index = tts_page.voice_combo.currentIndex()
    voice_code = voice_index + 1   # 1~4
    # 生成输出文件名（时间戳 + 文本前20个字符作为标识）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_prefix = text[:20].replace(' ', '_') if text else "manual"
    text_prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text_prefix)
    filename = f"manual_{timestamp}_{text_prefix}.wav"
    output_path = os.path.join(TTS_OUT_PATH, filename)
    tts_page.log_text.append(f"正在合成文本：{text[:80]}...")
    QApplication.processEvents()
    try:
        from audio_tts.tts_core import text_to_speech
        text_to_speech(tts, text, voice_code, speed, output_path)
        tts_page.log_text.append(f"合成成功！已保存至：{output_path}")
        tts_page.waveform_label.setText(f"【单次合成完成】音频已保存至：{output_path}")
    except Exception as e:
        tts_page.log_text.append(f"[错误] 合成失败：{str(e)}")
        tts_page.waveform_label.setText("【合成失败】请查看日志")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置全局字体：使用微软雅黑，11pt
    from PyQt5.QtGui import QFont

    font = QFont("Microsoft YaHei", 11)
    app.setFont(font)
    window = MainWindow()

    asr_page = window.asr_page
    trans_page = window.trans_page
    tts_page = window.tts_page

    asr_page.select_btn.clicked.connect(lambda: select_audio_file(asr_page))
    asr_page.vad_btn.clicked.connect(lambda: step_vad(asr_page))
    asr_page.preproc_btn.clicked.connect(lambda: step_preprocess(asr_page))
    asr_page.mfcc_btn.clicked.connect(lambda: step_mfcc(asr_page))
    asr_page.asr_btn.clicked.connect(lambda: step_asr(asr_page))
    asr_page.full_btn.clicked.connect(lambda: t1_full_pipeline(asr_page))

    trans_page.translate_btn.clicked.connect(
        lambda: translate_single(trans_page))
    trans_page.file_btn.clicked.connect(lambda: translate_file(trans_page))

    tts_page.select_file_btn.clicked.connect(lambda: select_tts_file(tts_page))
    tts_page.synthesize_btn.clicked.connect(lambda: synthesize_tts(tts_page))
    tts_page.single_synth_btn.clicked.connect(
        lambda: synthesize_single_text(tts_page))

    window.show()
    sys.exit(app.exec_())
