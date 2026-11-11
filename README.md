# Selective Enhancement for Local Speech Translation

A local desktop app that turns Chinese recordings into English text and speech. The translation study asks **when a second model can fix an error, and how much extra CPU time that correction costs**.

## From failure analysis to design

For “价格从30元降到13元” (“The price fell from 30 yuan to 13 yuan”), all six original MarianMT candidates used dollar signs. Reranking could only choose among those candidates. This motivated a second model that could generate a different translation, with checks tied to the original Chinese.

- **Fast first pass.** MarianMT runs on CTranslate2. Adjacent transcript fragments are grouped within a length limit so the translator can use more of the surrounding text.
- **Selective second pass.** Glossary gaps, detectable fact discrepancies, or certain technical passages trigger local Qwen2.5-0.5B. The system caps the number of segments sent for enhancement and checks the new translation against the source. It keeps the first draft if the candidate fails the checks or the second model fails to run.

For “卷积层的步长设为2”, this changed “The length of the curved layer is set at 2” to “The stride of the convolutional layer is set to 2.” Both models use pretrained weights and run on CPU.

## Evaluation

| Question | Comparison and result |
|---|---|
| How much does the inference backend matter? | With the same MarianMT weights, decoding settings, and 12 input segments, CPU inference fell from **4.92 s to 2.41 s**. |
| What does the second pass cost? | On four short inputs, fast mode took **0.54 s**, selective enhancement **1.23 s**, and attempting enhancement on every input **1.57 s**. Selective enhancement took about **22% less time** than the last mode on this workload. |
| How did terminology change? | On six technical sentences in a separate 12-case set, unmatched glossary terms fell from **8 to 2**. The four general sentences kept their original translations. |

Times are medians of three warm runs. The acceptance cases were fixed before evaluating the Qwen fallback. The glossary counts measure the combined effect of the enhanced pipeline on a small custom set. Broader translation quality would require a larger evaluation with human judgments.

The desktop app runs translation in the background. **40 regression checks passed**, and a **25.4-second recording** completed the Whisper → translation → VITS workflow.

## Contribution

Both team members contributed to the system design and its iterations. I implemented the Whisper recognition and VITS synthesis stages, led integration and end-to-end testing, and developed the translation workflow with a focus on speed, terminology, and context.

## Run

Prepare the MarianMT model using the [translation setup guide](docs/translation-setup.md). For selective enhancement, run `python experiments/setup_translation_fallback.py` to fetch the pinned Qwen model and tokenizer. For the desktop app, install the GUI and speech dependencies listed in `requirements.txt` and make FFmpeg and eSpeak NG available on PATH. Place the Whisper checkpoint at `voice_to_word/large-v3.pt` and the matching Coqui VITS checkpoint and configuration at `audio_tts/tts_models/model_file.pth` and `audio_tts/tts_models/config.json`. Run `python main.py` from the project root. Generated files go to `outputs/`.

## Built with

[Whisper](https://github.com/openai/whisper) · [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) · [CTranslate2](https://github.com/OpenNMT/CTranslate2) · [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [Coqui TTS / VITS](https://github.com/idiap/coqui-ai-TTS) · PyQt5
