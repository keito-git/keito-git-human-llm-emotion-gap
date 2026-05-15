# LLMs Capture Emotion Labels, Not Emotion Uncertainty: Distributional Analysis and Calibration of Human-LLM Judgment Gaps

Source code accompanying the paper

> **LLMs Capture Emotion Labels, Not Emotion Uncertainty: Distributional
> Analysis and Calibration of Human–LLM Judgment Gaps**
> Keito Inoshita, Xiaokang Zhou, Akira Kawai, Katsutoshi Yada.

We compare four LLMs (GPT-5.4-mini, Claude Haiku 4.5, Llama 3.1 8B, Qwen3-8B)
against human annotators on **GoEmotions** (28 categorical emotions) and
**EmoBank** (continuous V/A/D), and analyse where the gap between humans and
models is concentrated. The headline finding: LLMs reproduce the *modal*
emotion label well but systematically underestimate the *uncertainty* humans
express, even when sampled at high temperature. We also evaluate three
post-hoc calibration methods that close part of this gap.

This repository ships **code only**. Datasets, model outputs, intermediate
artefacts, and the paper PDF are *not* redistributed here. The data
preparation scripts will pull GoEmotions and EmoBank from their upstream
sources on first run; LLM outputs are produced by the inference scripts
against your own API/GPU access.

---

## Repository layout

```
.
├── README.md                  # This file
├── LICENSE                    # MIT (code only)
├── DATA_ATTRIBUTION.md        # Dataset / model licences and citations
├── .env.example               # Template for API keys
├── .gitignore
├── requirements.txt
└── code/
    ├── config/
    │   └── experiment_config.yaml   # Edit paths before first run
    ├── data_preparation/      # Download + core-set construction
    ├── llm_inference/         # API + vLLM inference scripts
    ├── analysis/              # Comparison, calibration, lexical, EmoBank
    ├── utils/                 # Shared I/O and metrics helpers
    ├── run_emobank_all.sh     # Convenience pipeline for EmoBank
    └── run_all_remaining.sh   # Convenience pipeline (Ollama variant)
```

The scripts assume the following project layout at runtime:

```
<PROJECT_ROOT>/
├── code/                      # this directory
└── data/
    ├── raw/                   # populated by data_preparation/download_*.py
    └── processed/             # populated by the pipeline (intermediates,
                               # core set, LLM outputs, analysis JSON, …)
```

`data/` is created automatically — you only need to clone this repository
and run the pipeline.

---

## Quick start

### 1. Install Python dependencies

Python ≥ 3.10 is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the **core** dependencies. A few scripts have
optional extras (commented at the bottom of the file):

| Optional extra | Needed for |
|---|---|
| `torch`, `transformers` | `analysis/finetuned_baseline.py` (RoBERTa baseline) |
| `vllm` | `llm_inference/run_vllm_inference.py` (open-weight models) |
| `sentence-transformers` | `analysis/lexical_transparency.py` (embedding mode) |

Install whichever you need on top of the base set.

### 2. Configure paths

Open `code/config/experiment_config.yaml` and replace the placeholder
`/PATH/TO/PROJECT_ROOT` with the absolute path to your local clone (the
parent of `code/`). A one-liner from the project root:

```bash
sed -i.bak "s|/PATH/TO/PROJECT_ROOT|$(pwd)|g" code/config/experiment_config.yaml
```

### 3. Provide API keys

```bash
cp .env.example .env
# then edit .env and fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
```

The inference scripts read `.env` automatically. Multiple OpenAI keys
(`OPENAI_API_KEY_1`, `OPENAI_API_KEY_2`, …) are rotated for parallel
requests.

---

## Reproducing the paper

Expect roughly 640,000 LLM calls in total
(2 datasets × 4 models × 4 temperatures × 10 samples × 2,000 texts). On the
API side this took us a few hours per provider; the open-weight models were
run on a single H100 with vLLM at roughly 70 q/s.

All commands below are run from the `code/` directory.

### Step 0 — download raw datasets

```bash
python data_preparation/download_goemotions.py --config config/experiment_config.yaml
python data_preparation/download_emobank.py    --config config/experiment_config.yaml
```

### Step 1 — build human distributions and the 2,000-text core set

```bash
python data_preparation/prepare_annotator_dist.py --config config/experiment_config.yaml
python data_preparation/sample_core_set.py        --config config/experiment_config.yaml
python data_preparation/prepare_emobank.py        --config config/experiment_config.yaml
```

### Step 2 — GoEmotions LLM inference

```bash
# API models
python llm_inference/run_api_inference.py  --config config/experiment_config.yaml --model gpt-5.4-mini             --workers 10
python llm_inference/run_api_inference.py  --config config/experiment_config.yaml --model claude-haiku-4-5-20251001 --workers 8

# Open-weight models (run on a GPU host with vLLM)
python llm_inference/run_vllm_inference.py --config config/experiment_config.yaml --model Qwen/Qwen3-8B
python llm_inference/run_vllm_inference.py --config config/experiment_config.yaml --model meta-llama/Llama-3.1-8B-Instruct
python llm_inference/convert_vllm_results.py --input <csv> --model-name qwen3_8b
python llm_inference/convert_vllm_results.py --input <csv> --model-name llama3_1_8b
```

All inference scripts checkpoint every 500 queries; interrupting and
re-running picks up where the previous attempt stopped.

### Step 3 — EmoBank LLM inference

```bash
bash run_emobank_all.sh
```

### Step 4 — analysis (regenerates every table and figure)

```bash
# Human-side basic statistics (Section 3.1)
python analysis/human_distribution_analysis.py --config config/experiment_config.yaml

# Tables 1, 2, 3, 8, 11 + per-category statistics (Figures 2/4/5)
python analysis/llm_human_comparison.py --config config/experiment_config.yaml

# Table 9 (bootstrap 95% CI), Table 12 (effect sizes), Figures 6/7
python analysis/enhanced_analysis.py --config config/experiment_config.yaml

# Table 5 (calibration JSD)
python analysis/distributional_calibration.py --config config/experiment_config.yaml

# Section 4.4 Wilcoxon tests + lexical sensitivity (n>=50 filter)
# This script takes the processed-data directory directly:
python analysis/reviewer_revisions_turn2.py --data_dir ../data/processed/

# Table 4 (lexical transparency)
python analysis/lexical_transparency.py --config config/experiment_config.yaml

# Tables 6, 10 (EmoBank)
python analysis/emobank_analysis.py        --config config/experiment_config.yaml
python analysis/emobank_bootstrap_ci.py    --config config/experiment_config.yaml

# Fine-tuned RoBERTa baseline (Section 4.2) — needs torch + transformers
python analysis/finetuned_baseline.py --config config/experiment_config.yaml
```

---

## Hardware and runtime notes

- **API runs**: a laptop is enough. Parallel workers are throttled to keep
  within rate limits.
- **vLLM runs**: a single 80 GB GPU (H100) was sufficient for both
  Qwen3-8B and Llama 3.1 8B at fp16. We use `CUDA_VISIBLE_DEVICES=1` by
  default; override via the env var if needed.
- **Bootstrap analyses** (`enhanced_analysis.py`, `emobank_bootstrap_ci.py`)
  use 1,000 iterations and run in a few minutes on a modern CPU.
- **Random seed** is fixed at 42 (`analysis.random_seed` in the YAML config).

---

## Citation

```bibtex
@inproceedings{inoshita2026emotion,
  title     = {LLMs Capture Emotion Labels, Not Emotion Uncertainty:
               Distributional Analysis and Calibration of Human--LLM
               Judgment Gaps},
  author    = {Inoshita, Keito and Zhou, Xiaokang and Kawai, Akira and Yada, Katsutoshi},
  booktitle = {arXiv},
  year      = {2026},
}
```

If you use the upstream datasets or models, please also cite them — see
`DATA_ATTRIBUTION.md` for the relevant citations and licences.

---


## Licence

- **Code** in this repository: MIT (see `LICENSE`).
- **Datasets and models** referenced by this code are *not* redistributed
  here and remain governed by their original licences. See
  `DATA_ATTRIBUTION.md` for details.
