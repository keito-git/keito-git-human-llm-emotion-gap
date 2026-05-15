#!/bin/bash
# Run all remaining experiments: local LLM inference -> analysis update -> paper finalization
# This script monitors Ollama downloads, runs inference, and triggers analysis.

set -e
OLLAMA=${OLLAMA:-ollama}
PYTHON=${PYTHON:-python3}
# Resolve project code directory relative to this script
PROJ="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=${LOG_DIR:-/tmp}

cd "$PROJ"

echo "=== Waiting for Ollama model downloads ==="

# Wait for Llama 3.1 8B
echo "[$(date)] Waiting for llama3.1:8b..."
while true; do
    if $OLLAMA list 2>/dev/null | grep -q "llama3.1:8b"; then
        echo "[$(date)] llama3.1:8b ready!"
        break
    fi
    sleep 30
done

# Start Llama 3.1 inference immediately (it's smaller, can run while Qwen downloads)
echo "[$(date)] Starting Llama 3.1 8B inference..."
PYTHONUNBUFFERED=1 $PYTHON llm_inference/run_ollama_inference.py --model llama3.1:8b --workers 4 > $LOG_DIR/llama31_inference.log 2>&1
echo "[$(date)] Llama 3.1 8B inference complete!"

# Wait for Qwen3 32B
echo "[$(date)] Waiting for qwen3:32b..."
while true; do
    if $OLLAMA list 2>/dev/null | grep -q "qwen3:32b"; then
        echo "[$(date)] qwen3:32b ready!"
        break
    fi
    sleep 30
done

# Start Qwen3 inference (use fewer workers since it's a large model)
echo "[$(date)] Starting Qwen3 32B inference..."
PYTHONUNBUFFERED=1 $PYTHON llm_inference/run_ollama_inference.py --model qwen3:32b --workers 1 > $LOG_DIR/qwen3_inference.log 2>&1
echo "[$(date)] Qwen3 32B inference complete!"

echo ""
echo "=== All 4 model inferences complete ==="
echo "[$(date)] Results:"
ls -lh "$PROJ/../data/processed/llm_results/"

echo ""
echo "=== ALL DONE ==="
echo "[$(date)] All local LLM experiments finished. Ready for final analysis update."
