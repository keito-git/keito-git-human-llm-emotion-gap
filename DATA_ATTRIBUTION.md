# Data and Model Attribution

The analysis code in this repository is released under the MIT License (see
`LICENSE`). The datasets and pretrained models used in our experiments are
**not** redistributed here and remain subject to their original licenses and
terms of use. Please consult the upstream sources before redistributing or
using them in derivative work.

---

## Datasets

### GoEmotions

- **Source**: <https://github.com/google-research/google-research/tree/master/goemotions>
- **License**: Apache License 2.0
- **Citation**:
  > Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi,
  > S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* In Proceedings
  > of the 58th Annual Meeting of the Association for Computational Linguistics
  > (ACL 2020).

We use the **full release** with individual annotator labels (three CSV
shards). The download script is `code/data_preparation/download_goemotions.py`.

### EmoBank

- **Source**: <https://github.com/JULIELab/EmoBank>
- **License**: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0)
- **Citation**:
  > Buechel, S., & Hahn, U. (2017). *EmoBank: Studying the Impact of Annotation
  > Perspective and Representation Format on Dimensional Emotion Analysis.* In
  > Proceedings of the 15th Conference of the European Chapter of the
  > Association for Computational Linguistics (EACL 2017).

We use the per-rater Reader-perspective VAD ratings. The download script is
`code/data_preparation/download_emobank.py`.

---

## Models

### Proprietary APIs

- **OpenAI GPT-5.4-mini** — used via the OpenAI API
  (<https://platform.openai.com/>). Subject to OpenAI's usage policies.
- **Anthropic Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) — used via the
  Anthropic API (<https://www.anthropic.com/>). Subject to Anthropic's usage
  policies.

### Open-weight Models (used through vLLM)

- **Meta Llama 3.1 8B Instruct**
  - Hugging Face: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
  - License: Llama 3.1 Community License Agreement
- **Alibaba Qwen3-8B**
  - Hugging Face: <https://huggingface.co/Qwen/Qwen3-8B>
  - License: Apache License 2.0

### Pretrained Baseline (fine-tuned RoBERTa)

- **SamLowe/roberta-base-go_emotions** (used as the supervised baseline)
  - Hugging Face: <https://huggingface.co/SamLowe/roberta-base-go_emotions>
  - License: MIT (per the model card on Hugging Face at the time of writing —
    please re-check before redistribution)

---

## Lexicons

- **NRC Emotion Lexicon** (used in the lexical-transparency analysis)
  - Source: <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>
  - License: free for non-commercial research use; commercial users must
    contact the author. See the upstream license for details.

---

## How to cite this repository

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{inoshita2026emotion,
  title     = {LLMs Capture Emotion Labels, Not Emotion Uncertainty:
               Distributional Analysis and Calibration of Human--LLM
               Judgment Gaps},
  author    = {Inoshita, Keito},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods
               in Natural Language Processing (EMNLP)},
  year      = {2026},
  note      = {Under review}
}
```
