import numpy as np
import librosa
import whisper
from scipy.signal import lfilter
from opencc import OpenCC
import warnings

# 过滤警告
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 繁简转换器（全局单例）
cc = OpenCC('t2s')

def voice_activity_detection(audio, sr, frame_length=2048, hop_length=512,
                             energy_threshold=0.02, zcr_threshold=0.02,
                             buffer_frames=8):
    """语音端点检测，返回有效语音片段及起止索引"""
    energy = np.array([
        np.sum(np.square(audio[i:i + frame_length]))
        for i in range(0, len(audio) - frame_length, hop_length)
    ])
    zcr = np.array([
        np.sum(np.abs(np.diff(np.sign(audio[i:i + frame_length])))) / 2
        for i in range(0, len(audio) - frame_length, hop_length)
    ])
    energy = energy / (np.max(energy) + 1e-10)
    zcr = zcr / (np.max(zcr) + 1e-10)
    voiced_frames = (energy > energy_threshold) & (zcr > zcr_threshold)
    from scipy.ndimage import binary_erosion, binary_dilation
    structure = np.ones(3)
    voiced_frames = binary_dilation(binary_erosion(voiced_frames, structure), structure)
    voiced_indices = np.where(voiced_frames)[0]
    if len(voiced_indices) == 0:
        return audio, 0, len(audio)
    start_frame = max(0, voiced_indices[0] - buffer_frames)
    end_frame = min(len(energy) - 1, voiced_indices[-1] + buffer_frames)
    start_idx = start_frame * hop_length
    end_idx = end_frame * hop_length + frame_length
    start_idx = max(0, start_idx)
    end_idx = min(len(audio), end_idx)
    return audio[start_idx:end_idx], start_idx, end_idx

def preprocess_audio(audio, sr, frame_length=2048, hop_length=512, pre_emphasis_coeff=0.97):
    """语音前处理：预加重、分帧、加汉明窗，返回帧矩阵"""
    pre_emphasized_audio = lfilter([1, -pre_emphasis_coeff], [1], audio)
    n_frames = 1 + int((len(pre_emphasized_audio) - frame_length) / hop_length)
    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i] = pre_emphasized_audio[start:end]
    hamming_window = np.hamming(frame_length)
    frames = frames * hamming_window
    return frames

def extract_mfcc_features(audio, sr, n_mfcc=13, frame_length=2048, hop_length=512):
    """提取MFCC特征及一阶、二阶差分，返回特征矩阵 (帧数, 39)"""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=frame_length, hop_length=hop_length, window='hamming'
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0).T
    return mfcc_features

def speech_recognition_whisper(model,audio_path):
    """Whisper语音识别，使用指定本地模型文件，返回带句号的简体中文结果字典"""
    # 加载本地指定路径的Whisper模型
    # model = whisper.load_model(model_path)
    result = model.transcribe(audio_path, language="zh")
    segments = result['segments']
    processed_texts = []
    for seg in segments:
        text = seg['text'].strip()
        if not text:
            continue
        text = cc.convert(text)
        if text[-1] not in '。！？；':
            text += '。'
        processed_texts.append(text)
    result['text'] = ''.join(processed_texts)
    return result