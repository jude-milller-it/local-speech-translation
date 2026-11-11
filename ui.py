import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QTabWidget, QStatusBar, QFileDialog,
    QGroupBox, QSlider, QComboBox, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ASRPage(QWidget):
    """中文语音识别页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        # 边距和间距设置
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # ---------- 文件选择行 ----------
        file_layout = QHBoxLayout()
        self.select_btn = QPushButton("选择WAV文件")
        self.select_btn.setMinimumHeight(38)
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        self.file_label.setMinimumHeight(38)
        file_layout.addWidget(self.select_btn)
        file_layout.addWidget(self.file_label, 1)
        layout.addLayout(file_layout)

        # 五个功能按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.vad_btn = QPushButton("1. 语音端点检测")
        self.preproc_btn = QPushButton("2. 语音预处理")
        self.mfcc_btn = QPushButton("3. 提取MFCC特征")
        self.asr_btn = QPushButton("4. 语音识别")
        self.full_btn = QPushButton("5. 一键全流程处理")
        for btn in [self.vad_btn, self.preproc_btn,
                    self.mfcc_btn, self.asr_btn, self.full_btn]:
            btn.setMinimumHeight(38)
        btn_layout.addWidget(self.vad_btn)
        btn_layout.addWidget(self.preproc_btn)
        btn_layout.addWidget(self.mfcc_btn)
        btn_layout.addWidget(self.asr_btn)
        btn_layout.addWidget(self.full_btn)
        layout.addLayout(btn_layout)

        # 识别结果展示
        result_label = QLabel("识别结果：")
        result_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(result_label)
        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("语音识别结果显示在此处...")
        self.result_text.setMinimumHeight(150)
        layout.addWidget(self.result_text)

        # 处理日志展示
        log_label = QLabel("处理日志：")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("端点检测、预处理、MFCC提取等步骤的详细信息...")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        layout.addWidget(self.log_text)

        # 保存当前选中的文件路径以及中间步骤产生的文件路径
        self.current_audio_path = None   # 用户选择的原始wav
        self.voiced_audio_path = None    # 端点检测后保存的路径
        self.mfcc_features_path = None   # MFCC特征保存路径
        self.asr_result_text = ""        # 识别结果文本
        self._asr_worker = None


class TranslationPage(QWidget):
    """中英翻译页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 中文输入区
        src_label = QLabel("中文原文：")
        src_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(src_label)
        self.src_text = QTextEdit()
        self.src_text.setPlaceholderText("请输入要翻译的中文文本...")
        self.src_text.setMinimumHeight(150)
        layout.addWidget(self.src_text)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.translate_btn = QPushButton("翻译")
        self.file_btn = QPushButton("选择TXT文件逐行翻译")
        for btn in [self.translate_btn, self.file_btn]:
            btn.setMinimumHeight(38)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("按需增强（推荐）", "selective")
        self.mode_combo.addItem("快速翻译", "baseline")
        self.mode_combo.addItem("候选共识", "mbr")
        self.mode_combo.addItem("双向校验", "bidirectional")
        self.mode_combo.addItem("按需双向校验", "auto")
        self.mode_combo.setToolTip("增强模式按需加载本地模型，首次使用稍慢；快速模式可关闭增强。")
        btn_layout.addWidget(self.mode_combo)
        btn_layout.addWidget(self.translate_btn)
        btn_layout.addWidget(self.file_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 英文翻译结果展示区
        dst_label = QLabel("英文翻译结果：")
        dst_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(dst_label)
        self.dst_text = QTextEdit()
        self.dst_text.setPlaceholderText("翻译结果将显示在此处...")
        self.dst_text.setReadOnly(True)
        self.dst_text.setMinimumHeight(150)
        layout.addWidget(self.dst_text)

        # 处理日志（显示逐行翻译进度）
        log_label = QLabel("处理日志：")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("翻译进度信息...")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        layout.addWidget(self.log_text)

        # 存储文件翻译时的取消标志（可选）
        self._worker = None


class TTSPage(QWidget):
    """英文语音合成页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 手动输入文本
        manual_group = QGroupBox("手动输入文本")
        manual_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setSpacing(12)

        self.input_text_edit = QTextEdit()
        self.input_text_edit.setPlaceholderText(
            "在此输入英文文本，例如：Hello world, this is a test.")
        self.input_text_edit.setMinimumHeight(120)
        manual_layout.addWidget(self.input_text_edit)

        # 合成当前文本按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.single_synth_btn = QPushButton("合成当前文本")
        self.single_synth_btn.setMinimumHeight(38)
        self.single_synth_btn.setMinimumWidth(150)
        btn_layout.addWidget(self.single_synth_btn)
        manual_layout.addLayout(btn_layout)

        main_layout.addWidget(manual_group)

        # 文件批量合成
        file_group = QGroupBox("批量合成（从TXT文件）")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(12)

        # 文件选择行
        select_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("选择英文TXT文件")
        self.select_file_btn.setMinimumHeight(38)
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        select_layout.addWidget(self.select_file_btn)
        select_layout.addWidget(self.file_label, 1)
        file_layout.addLayout(select_layout)

        # 批量合成按钮
        self.synthesize_btn = QPushButton("开始逐行合成 (使用选定的TXT文件)")
        self.synthesize_btn.setMinimumHeight(45)
        file_layout.addWidget(self.synthesize_btn)

        main_layout.addWidget(file_group)

        # 合成参数设置
        param_group = QGroupBox("合成参数设置")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(12)

        # 语速控制
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("语速："))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.setMinimumHeight(30)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("100%")
        self.speed_label.setMinimumWidth(50)
        speed_layout.addWidget(self.speed_label)
        param_layout.addLayout(speed_layout)

        # 音色选择
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("合成音色："))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems([
            "成年男声1-p231",
            "成年男声2-p258",
            "青年女声1-p260",
            "青年女声2-p273"
        ])
        self.voice_combo.setMinimumHeight(35)
        voice_layout.addWidget(self.voice_combo)
        voice_layout.addStretch()
        param_layout.addLayout(voice_layout)

        main_layout.addWidget(param_group)

        # 合成日志
        log_label = QLabel("合成日志：")
        log_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("显示合成进度和结果信息...")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        main_layout.addWidget(self.log_text)

        self.waveform_label = QLabel("【合成完成后，可在输出目录查看音频文件，波形/频谱图差异将在报告中展示】")
        self.waveform_label.setAlignment(Qt.AlignCenter)
        self.waveform_label.setStyleSheet(
            "border: 1px solid #ccc; background: #f8f8f8; padding: 12px; font-size: 11pt;")
        self.waveform_label.setMinimumHeight(70)
        main_layout.addWidget(self.waveform_label)

        # 内部变量
        self.current_file_path = None

        # 信号连接
        self.speed_slider.valueChanged.connect(self._update_speed_label)

    def _update_speed_label(self, value):
        self.speed_label.setText(f"{value}%")


class MainWindow(QMainWindow):
    """主窗口，包含三个子页面"""

    def __init__(self):
        super().__init__()
        # 原标题写作「基于FunASR和LLM」，但实际使用的是 Whisper + MarianMT + VITS，
        # 既无 FunASR 也无 LLM，故改为与实现一致的名称（如需还原课程报告标题可改回）。
        self.setWindowTitle("中文→英文语音翻译系统（Whisper + MarianMT + VITS）")
        self.resize(1300, 850)
        self.setMinimumSize(1000, 700)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabBar::tab { font-size: 11pt; min-width: 120px; min-height: 35px; }")
        self.setCentralWidget(self.tab_widget)

        self.asr_page = ASRPage()
        self.trans_page = TranslationPage()
        self.tts_page = TTSPage()

        self.tab_widget.addTab(self.asr_page, "中文语音识别")
        self.tab_widget.addTab(self.trans_page, "中文→英文翻译")
        self.tab_widget.addTab(self.tts_page, "英文语音合成")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        self.status_bar.setStyleSheet("font-size: 10pt;")

    def closeEvent(self, event):
        """重写关闭事件，弹出确认对话框"""
        if getattr(self.trans_page, '_translation_worker', None) is not None:
            self.status_bar.showMessage("翻译尚未结束，请完成后退出。")
            event.ignore()
            return
        reply = QMessageBox.question(
            self,
            "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 11)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
