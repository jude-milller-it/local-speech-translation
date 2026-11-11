from transformers import MarianMTModel, MarianTokenizer
import re
import os
from typing import List

class ChineseToEnglishTranslator:
    def __init__(self, model_path="./opus-mt-zh-en"):
        """初始化翻译器，加载本地预训练模型"""
        self.model_path = model_path
        self.max_token_length = 512  # 模型支持的最大输入 token 数
        # 单个句子超过该 token 数时才触发硬切（正常句子远小于此值）。
        # 取值依据：opus-mt-zh-en 单段超过约 200 token 会静默丢尾，故取 128 保守。
        self.chunk_token_budget = 128
        # 提前检查所有必需文件是否存在
        required_files = [
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "source.spm",
            "target.spm",
            "tokenizer_config.json",
            "vocab.json"
        ]
        for file in required_files:
            full_path = os.path.join(model_path, file)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"模型文件缺失：{full_path}\n请检查opus-mt-zh-en文件夹是否包含所有7个必需文件")
        self.tokenizer = MarianTokenizer.from_pretrained(
            model_path,
            local_files_only=True,  # 强制使用本地文件，不联网下载
            source_spm=os.path.join(model_path, "source.spm"),
            target_spm=os.path.join(model_path, "target.spm")
        )
        self.model = MarianMTModel.from_pretrained(
            model_path,
            local_files_only=True
        )

    def _preprocess(self, text: str) -> str:
        """文本预处理：过滤无效字符，统一格式"""
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("输入文本不能为空")
        # 去除多余空格、换行符
        text = re.sub(r'\s+', ' ', text).strip()
        # 简单中文检测（可选，加分项）
        if not re.search(r'[\u4e00-\u9fff]', text):
            raise ValueError("请输入中文文本")
        return text

    # 句末标点（保留标点）；英文句点排除小数点：39.9 这种不被切开
    _SENT_SPLIT_RE = re.compile(r'([。！？!?]|(?<!\d)\.(?!\d))')

    def _hard_split(self, text: str) -> List[str]:
        """对不含标点的超长单句按 token 数硬切，保证每段不超过模型上限。"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        limit = self.chunk_token_budget
        chunks = []
        for i in range(0, len(ids), limit):
            sub = self.tokenizer.decode(ids[i:i + limit], skip_special_tokens=True)
            if sub.strip():
                chunks.append(sub)
        return chunks or [text]

    def _split_long_text(self, text: str) -> List[str]:
        """按句切分并逐句翻译；单句超过上限时再按 token 硬切。

        实测发现 opus-mt-zh-en 会静默丢弃长段落的尾部内容，因此放弃「把多句
        打包成一个大段」的策略，改为以句子为单位切分，从根上避免丢尾。
        """
        # 1) 按句末标点切分（保留标点；小数点不切）
        parts = self._SENT_SPLIT_RE.split(text)
        sentences = []
        for i in range(0, len(parts), 2):
            s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            if s.strip():
                sentences.append(s)

        # 2) 逐句成段；单句超长（如无标点长文本）才硬切
        chunks = []
        for s in sentences:
            if len(self.tokenizer.encode(s)) > self.chunk_token_budget:
                chunks.extend(self._hard_split(s))
            else:
                chunks.append(s)
        return chunks

    def translate(self, text: str) -> str:
        """对外暴露的核心翻译接口。

        出错时抛出异常，由调用方决定如何处理；不再把错误伪装成一段译文。
        """
        processed_text = self._preprocess(text)
        text_chunks = self._split_long_text(processed_text)

        # 逐段翻译并合并
        translated_chunks = []
        for chunk in text_chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_token_length
            )
            # max_new_tokens：只限制「新生成」的长度，而不是 input+output 总长。
            # 否则长输入会被截断到几乎为空（原实现用 max_length=512 就是这个 bug）。
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_token_length)
            translated_chunk = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_chunks.append(translated_chunk)

        return ' '.join(translated_chunks)


# 本地测试代码
if __name__ == "__main__":
    print("正在加载翻译模型...")
    translator = ChineseToEnglishTranslator()
    print("模型加载成功！")

    test_text = "人工智能是一门旨在使计算机系统能够模拟、延伸和扩展人类智能的技术科学，它涉及多个学科领域。"
    print(f"\n输入：{test_text}")
    print(f"输出：{translator.translate(test_text)}")