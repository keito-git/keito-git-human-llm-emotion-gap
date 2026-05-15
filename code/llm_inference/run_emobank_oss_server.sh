#!/bin/bash
# Run EmoBank OSS model inference on GPU server
# Execute this on the GPU server after GoEmotions inference completes
# Usage: bash run_emobank_oss_server.sh

WORK_DIR="/home/d9999993/emobank_experiment"
cd "$WORK_DIR"

echo "=== EmoBank OSS Model Inference ==="
echo "Start time: $(date)"

# Check if core set exists
if [ ! -f "emobank_core_set_with_text.csv" ]; then
    echo "ERROR: emobank_core_set_with_text.csv not found in $WORK_DIR"
    exit 1
fi

# Run Qwen3-8B
echo ""
echo "--- Qwen3-8B ---"
CUDA_VISIBLE_DEVICES=1 python3 run_emobank_vllm_inference.py \
    --model Qwen/Qwen3-8B \
    --output-name qwen3-8b \
    --work-dir "$WORK_DIR" \
    --max-model-len 4096

echo ""
echo "--- Llama 3.1 8B ---"
CUDA_VISIBLE_DEVICES=1 python3 run_emobank_vllm_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-name llama3.1-8b \
    --work-dir "$WORK_DIR" \
    --max-model-len 4096

echo ""
echo "=== All OSS inference complete ==="
echo "End time: $(date)"
echo "Results:"
ls -la "$WORK_DIR"/emobank_*_results.csv
