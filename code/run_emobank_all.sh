#!/bin/bash
# Run all EmoBank experiments
# Usage: bash run_emobank_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== EmoBank Experiment Pipeline ==="
echo "Start time: $(date)"

# Step 1: Data preparation (if not already done)
echo ""
echo "--- Step 1: Data Preparation ---"
if [ ! -f "../data/processed/emobank/emobank_core_set.parquet" ]; then
    python data_preparation/download_emobank.py
    python data_preparation/prepare_emobank.py
else
    echo "  Core set already exists, skipping."
fi

# Step 2: API inference
echo ""
echo "--- Step 2: API Inference ---"

# GPT-5.4-mini (parallel with 10 workers)
if [ ! -f "../data/processed/emobank/llm_results/emobank_gpt_5_4_mini_results.parquet" ]; then
    echo "  Running GPT-5.4-mini..."
    python llm_inference/run_emobank_api_inference.py --model gpt-5.4-mini --workers 10
else
    echo "  GPT-5.4-mini results exist, skipping."
fi

# Claude Haiku (parallel with 8 workers)
if [ ! -f "../data/processed/emobank/llm_results/emobank_claude_haiku_4_5_20251001_results.parquet" ]; then
    echo "  Running Claude Haiku..."
    python llm_inference/run_emobank_api_inference.py --model claude-haiku-4-5-20251001 --workers 8
else
    echo "  Claude Haiku results exist, skipping."
fi

# Step 3: OSS model inference (on GPU server - manual step)
echo ""
echo "--- Step 3: OSS Model Inference (GPU Server) ---"
echo "  SSH to GPU server and run:"
echo "    CUDA_VISIBLE_DEVICES=1 python3 run_emobank_vllm_inference.py --model Qwen/Qwen3-8B --output-name qwen3-8b"
echo "    CUDA_VISIBLE_DEVICES=1 python3 run_emobank_vllm_inference.py --model meta-llama/Llama-3.1-8B-Instruct --output-name llama3.1-8b"
echo "  Then copy results back and convert:"
echo "    python llm_inference/convert_emobank_vllm_results.py --input <csv> --model-name qwen3_8b"
echo "    python llm_inference/convert_emobank_vllm_results.py --input <csv> --model-name llama3_1_8b"

# Step 4: Analysis (run after all LLM results are available)
echo ""
echo "--- Step 4: Analysis ---"
python analysis/emobank_analysis.py

echo ""
echo "=== Pipeline complete ==="
echo "End time: $(date)"
