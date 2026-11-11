# Translation Backend: Setup and Reproduction

This guide covers the translation tools only. Speech recognition and synthesis dependencies are needed only for the desktop application. For the project overview and measured results, see [README.md](../README.md).

## 1. Prepare the environment

Use a Python environment with PyTorch, Transformers, SentencePiece, NumPy, and Sacremoses. The tested versions are recorded in [benchmark metadata](../experiments/results/ct2_final/summary.json). The root `requirements.txt` is a legacy environment snapshot, not a minimal or independently validated installation recipe.

Install the additional backend and evaluation packages into the directory used by the launcher:

```bash
python -m pip install --target runtime/ct2 -r tools/requirements_ct2.txt
```

If reusing bundled packages, their Python version and operating system must match your environment. Otherwise, install compatible packages rather than reusing native binaries from another platform.

## 2. Prepare the model files

The tools expect these directories:

```text
translator/
  opus-mt-zh-en/          # Original model and tokenizer files
  ct2_zh_en_float32/      # Converted model used by default
  ct2_zh_en_int8/         # Required for the three-backend evaluation
```

Obtain the original checkpoint from [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en). Keep its tokenizer and configuration files together. The conversion script expects the original weights as `pytorch_model.bin`; tokenizer files alone are insufficient for conversion or the PyTorch comparison.

Convert the checkpoint from the project root:

```bash
python tools/ct2_run.py convert
```

This creates both FP32 and INT8 models and writes a `conversion.json` manifest in each directory. The manifest records the source weight hash, converter version, precision, and model size. Existing converted models are skipped; use a separate checkout when rebuilding from different source weights.

A copy containing the FP32 runtime model and tokenizer can translate without the original PyTorch weights. Reproducing the full comparison requires the original weights and both converted models. Large checkpoints are not supplied by the README and may be absent from a source-only checkout.

## 3. Translate a UTF-8 text file

```bash
python tools/ct2_run.py translate examples/demo_ct2.txt --output english.txt
```

The command writes the translation and an adjacent `english.audit.json` report. To select the comparison precision explicitly:

```bash
python tools/ct2_run.py translate examples/demo_ct2.txt --output english_int8.txt --precision int8
```

Run `python tools/ct2_run.py translate --help` for available options.

## 4. Reproduce the evaluation

The evaluator reads the fixed inputs and reference translations from `experiments/results/constrained_evaluation/cases.json`. That directory name is historical; this evaluator uses ordinary decoding.

Create the output directory first, since the evaluator does not create missing parent directories:

```bash
python -c "from pathlib import Path; Path('experiments/results/reproduced').mkdir(parents=True, exist_ok=True)"
python tools/ct2_run.py evaluate --out experiments/results/reproduced
```

Outputs include:

- `summary.json`: environment, decoding settings, automatic scores, and timing measurements.
- `outputs.json`: source text, reference translation, and each backend's translation for every case.
- `results.csv`: the same translations in a spreadsheet-friendly format.

The original recorded run is in `experiments/results/ct2_final/`. Use a new directory for your results. Runtime varies with hardware and environment; keep decoding settings fixed when comparing results. Available overrides are listed by `python tools/ct2_run.py evaluate --help`.

## 5. Run focused regression checks

```bash
python experiments/test_sentence_quotes.py
python experiments/test_reliability.py
```

These check text handling and related safeguards. They do not replace translation-quality evaluation or a full desktop application test.

## Troubleshooting

**CTranslate2 cannot be imported:** install its matching wheel into `runtime/ct2` using the same Python interpreter used to launch the tools.

**Translation works but conversion or evaluation fails:** check for the original `pytorch_model.bin` and the INT8 comparison model. A working FP32 deployment does not imply that all benchmarking assets are present.

**The desktop application fails in a speech module:** first run the text-only translation command above to separate translation setup from Whisper, Coqui TTS, and eSpeak dependencies.

## Upstream references

[CTranslate2 conversion guide](https://opennmt.net/CTranslate2/guides/transformers.html) · [Source model](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) · [Third-party notices](../third_party_notices/)

## Optional source-aware selection

The desktop translation page offers **Fast**, **Candidate consensus**, **Bidirectional**, and **On-demand bidirectional** modes. The current default is selective enhancement, described below; fast mode and the earlier reranking modes remain available.

```bash
python tools/ct2_run.py translate examples/demo_ct2.txt --output english.txt --mode bidirectional
python tools/ct2_run.py translate examples/demo_ct2.txt --output english.txt --mode auto
```

Bidirectional mode uses the local `opus-mt-en-zh` tokenizer and `ct2_en_zh_float32` model. Both translators use CPU inference; the reverse model loads on demand. It combines mean token log probabilities in both directions, with reverse weight 0.5, and switches only above a conservative margin. These fixed values are provisional, not tuned research results. This adapts the channel-scoring idea from [Yee et al. (2019)](https://aclanthology.org/D19-1571/); it omits the separate language-model component of their full method.

On-demand mode skips reverse scoring when the initial output passes the supported checks. Unsupported facts are not automatically corrected. If every candidate fails a check, the original output is retained with a warning. Missing scoring assets also fall back to the original output and emit a warning.

`translator/required_terms.json` accepts explicitly mandatory terminology, e.g. `{"池化层": ["pooling layer"]}`. It is empty by default to avoid treating legitimate paraphrases as errors. General glossary and proper-name checks remain advisory.

Audit files contain candidate scores, checks, selection reasons, and elapsed time. Run `python experiments/check_bidirectional_small.py` for the 16-input functional comparison and short paired timing check. This is not a semantic-accuracy benchmark.

### Bounded currency recovery and GUI execution

When every original candidate incorrectly changes yuan to dollars, bidirectional modes may generate at most two extra candidates from an explicitly normalized monetary input. This applies only to supported monetary contexts (price, fare, payment, amount, or shipping); quoted text and other or mixed currencies are excluded. The original source is retained for both probability scoring and audit. Acceptance requires passing fact checks, improved reverse likelihood, and a bounded forward-likelihood decrease. This is a narrow engineering heuristic, not a paper reproduction or general semantic repair. Disable it with `currency_rescue=False` when constructing `BidirectionalTranslator`.

The scope treats bare Chinese 元 as yuan in those monetary contexts. If another local currency is intended, name it explicitly; the system cannot infer unstated geographical context. General terminology hints and wider candidate pools were tested but not deployed because they did not reliably fix the observed errors.

Translation runs in a background worker in the desktop app. Translation controls remain locked until completion; closing the window during translation is prevented. ASR and TTS algorithms remain unchanged. CPU use and overall speech-model memory pressure can still affect responsiveness.

## Current default: selective local enhancement

The app and text CLI now default to `selective`. This combines the original fast MarianMT backend with a local Qwen2.5-0.5B-Instruct INT8 fallback. It does **not** fine-tune or improve the original MarianMT weights. The NLLB experiment is not used by the default path.

Prepare the optional model using the existing Python environment:

```bash
python experiments/setup_translation_fallback.py
python tools/ct2_run.py translate examples/demo_ct2.txt --output english.txt --mode selective
```

The setup script pins repository revisions and verifies the downloaded model hash. It uses a third-party CTranslate2 conversion of the official Qwen model and records both sources. Weights are approximately 496 MB; tokenizer assets are separate. Nothing is downloaded automatically during translation.

Use `--mode baseline` or the GUI's fast option to disable enhancement. If fallback assets are missing, the system keeps the original translation and reports the failure. The `quality` mode is an internal comparison mode, not the recommended GUI path.

Selective mode checks the fast output for supported fact errors and glossary gaps. It calls the fallback only for triggered segments, or supported technical paragraphs that need neighboring context. At most four segments per request use the fallback by default. Replacements must not introduce additional recognized fact errors or lose recognized domain terms. Ordinary triggered replacements must improve a check; context-driven replacements are recorded separately. These are conservative heuristics, not semantic guarantees.

The CPU-only fallback uses deterministic decoding and a 256-token generation cap; reaching that cap is treated as failure, not silently accepted truncation. Short adjacent source fragments are packed without changing their characters, within the original 128-token source budget. Audit JSON records the original text, initial translation, replacement candidate, selection reason and warnings.

Run `python experiments/check_selective_translation.py` for the small development comparison and timings, and `python experiments/check_selective_acceptance.py` for the pre-fixed acceptance inputs. These are small AI-authored functional sets, not independent human assessments. Their automatic metrics must not be presented as universal accuracy or publication-level evidence.
