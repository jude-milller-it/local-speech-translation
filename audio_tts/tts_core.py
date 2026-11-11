import os
import numpy as np
import soundfile as sf
from TTS.api import TTS

# ====================== 固定配置 ======================
MODEL_DIR = "./tts_models"

# 音色代号 1~4
VOICE_MAP = {
    1: "p231",
    2: "p258",
    3: "p260",
    4: "p273",
}


# # # 全局加载模型
# tts = TTS(
#     model_path=os.path.join(MODEL_DIR, "model_file.pth"),
#     config_path=os.path.join(MODEL_DIR, "config.json"),
#     gpu=False,
#     progress_bar=False
# )





def text_to_speech(tts, text, voice_code, speed=1.0, output_path=None):
    """
    输出路径作为参数，手动传入
    :param text: 文本
    :param voice_code: 1-4
    :param speed: 语速
    :param output_path: 输出完整路径（例如 output_audio/line1_voice1.wav）
    :return: wav 数据
    """
    speaker_id = VOICE_MAP[voice_code]
    wav = tts.tts(text=text, speaker=speaker_id, speed=speed)
    wav = np.array(wav)

    # 如果传入路径，就保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, 22050)

    return wav

