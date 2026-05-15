"""
vLLM inference for EmoBank VAD prediction with OSS models (Qwen3-8B, Llama 3.1 8B).

Uses chat template with system + user prompt (aligned with API models).
GPU 1 only (CUDA_VISIBLE_DEVICES=1).

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_emobank_vllm_inference.py --model Qwen/Qwen3-8B --output-name qwen3-8b
    CUDA_VISIBLE_DEVICES=1 python run_emobank_vllm_inference.py --model meta-llama/Llama-3.1-8B-Instruct --output-name llama3.1-8b
"""

import argparse
import json
import os
import re
import time
import csv
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from vllm import LLM, SamplingParams

# Unified VAD prompt (identical to API version)
SYSTEM_PROMPT = """You are an emotion annotation assistant. Your task is to rate the emotional content of a given text on three continuous dimensions.

Dimensions (rate each on a scale from 1.0 to 5.0):
- Valence (V): How positive or negative the emotion is. 1=very negative, 3=neutral, 5=very positive.
- Arousal (A): How calm or excited the emotion is. 1=very calm, 3=neutral, 5=very excited.
- Dominance (D): How submissive or dominant the emotion is. 1=very submissive/weak, 3=neutral, 5=very dominant/strong.

Rules:
- Rate from the READER's perspective (how the text makes you feel).
- Use decimal values (e.g., 2.4, 3.8) for precise ratings.
- Return ONLY a JSON object with keys "V", "A", "D" and numeric values.
- Example: {"V": 3.2, "A": 2.8, "D": 3.0}
- Example: {"V": 1.5, "A": 4.2, "D": 2.0}"""


def build_user_prompt(text: str) -> str:
    return f'Rate the emotional content of this text on V (Valence), A (Arousal), and D (Dominance) scales (1.0-5.0).\n\nText: "{text}"'


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]


def parse_vad_response(response_text: str) -> dict[str, float] | None:
    """Parse VAD values from LLM response."""
    response_text = response_text.strip()

    # Handle Qwen3 thinking tags
    think_end = response_text.find("</think>")
    if think_end != -1:
        response_text = response_text[think_end + len("</think>"):].strip()

    # Try JSON
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(response_text[start:end + 1])
            v = float(obj.get("V", obj.get("v", -1)))
            a = float(obj.get("A", obj.get("a", -1)))
            d = float(obj.get("D", obj.get("d", -1)))
            if all(0.5 <= x <= 5.5 for x in [v, a, d]):
                return {
                    "V": max(1.0, min(5.0, v)),
                    "A": max(1.0, min(5.0, a)),
                    "D": max(1.0, min(5.0, d)),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback regex
    v_match = re.search(r'[Vv](?:alence)?["\s:=]+(\d+\.?\d*)', response_text)
    a_match = re.search(r'[Aa](?:rousal)?["\s:=]+(\d+\.?\d*)', response_text)
    d_match = re.search(r'[Dd](?:ominance)?["\s:=]+(\d+\.?\d*)', response_text)

    if v_match and a_match and d_match:
        v = float(v_match.group(1))
        a = float(a_match.group(1))
        d = float(d_match.group(1))
        if all(0.5 <= x <= 5.5 for x in [v, a, d]):
            return {
                "V": max(1.0, min(5.0, v)),
                "A": max(1.0, min(5.0, a)),
                "D": max(1.0, min(5.0, d)),
            }

    return None


def main():
    parser = argparse.ArgumentParser(description="vLLM EmoBank VAD inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--work-dir", type=str,
                        default="/home/d9999993/emobank_experiment")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True)

    # Load core set
    core_set_path = work_dir / "emobank_core_set_with_text.csv"
    if not core_set_path.exists():
        print(f"ERROR: {core_set_path} not found")
        print("Please copy emobank_core_set_with_text.csv to the work directory first.")
        return

    texts = []
    with open(core_set_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("text"):
                texts.append({"id": row["id"], "text": row["text"]})

    print(f"Loaded {len(texts)} texts")

    # Initialize vLLM
    print(f"Loading {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    temperatures = [0.0, 0.3, 0.7, 1.0]
    samples_per_temp = 10
    seed = 42

    total = len(texts) * len(temperatures) * samples_per_temp
    print(f"Total queries: {total}")

    # Build all tasks
    all_tasks = []
    for temp in temperatures:
        for sample_idx in range(samples_per_temp):
            for t in texts:
                messages = build_messages(t["text"])
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_tasks.append({
                    "id": t["id"],
                    "text": t["text"],
                    "temperature": temp,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                })

    # Checkpoint
    safe_name = args.output_name.replace(":", "_").replace(".", "_").replace("-", "_")
    checkpoint_path = work_dir / f"emobank_{safe_name}_checkpoint.csv"
    output_path = work_dir / f"emobank_{safe_name}_results.csv"

    results = []
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            csv_reader = csv.DictReader(f)
            results = list(csv_reader)
        completed_keys = set()
        for r in results:
            completed_keys.add((r["id"], str(r["temperature"]), str(r["sample_idx"])))
        all_tasks = [t for t in all_tasks
                     if (t["id"], str(t["temperature"]), str(t["sample_idx"])) not in completed_keys]
        print(f"Resumed: {len(results)} done, {len(all_tasks)} remaining")

    if not all_tasks:
        print("Nothing to do!")
        return

    # Process in batches
    batch_size = 2000
    start_time = time.time()
    parse_errors = 0

    for batch_start in range(0, len(all_tasks), batch_size):
        batch = all_tasks[batch_start:batch_start + batch_size]
        prompts = [t["prompt"] for t in batch]

        temp = batch[0]["temperature"]
        s_idx = batch[0]["sample_idx"]

        if temp == 0.0:
            sampling_params = SamplingParams(
                temperature=0.01,
                max_tokens=500,
                seed=seed,
            )
        else:
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=500,
                seed=seed + s_idx,
            )

        outputs = llm.generate(prompts, sampling_params)

        for task, output in zip(batch, outputs):
            response = output.outputs[0].text
            vad = parse_vad_response(response)

            if vad is None:
                parse_errors += 1
                results.append({
                    "id": task["id"],
                    "model": args.output_name,
                    "temperature": task["temperature"],
                    "sample_idx": task["sample_idx"],
                    "raw_response": response[:1000],
                    "V_pred": "",
                    "A_pred": "",
                    "D_pred": "",
                    "parse_error": True,
                    "api_error": False,
                })
            else:
                results.append({
                    "id": task["id"],
                    "model": args.output_name,
                    "temperature": task["temperature"],
                    "sample_idx": task["sample_idx"],
                    "raw_response": response[:1000],
                    "V_pred": vad["V"],
                    "A_pred": vad["A"],
                    "D_pred": vad["D"],
                    "parse_error": False,
                    "api_error": False,
                })

        done = len(results)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate / 60 if rate > 0 else 0
        print(f"  Progress: {done}/{total} ({done/total*100:.1f}%), "
              f"rate: {rate:.1f} q/s, ETA: {remaining:.0f} min, "
              f"parse_errors: {parse_errors}")

        # Checkpoint every 10000
        if done % 10000 < batch_size:
            fieldnames = list(results[0].keys())
            with open(checkpoint_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"  [checkpoint saved: {done} results]")

    # Save final
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"\n=== Complete ===")
    print(f"Saved: {output_path}")
    print(f"Total: {len(results)}, Parse errors: {parse_errors}")
    print(f"Time: {elapsed/60:.1f} min, Rate: {len(all_tasks)/elapsed:.1f} q/s")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Summary
    summary = {
        "dataset": "emobank",
        "task": "VAD_prediction",
        "model": args.model,
        "output_name": args.output_name,
        "prompt_format": "system_prompt + user_prompt (chat_template)",
        "total_queries": total,
        "results": len(results),
        "parse_errors": parse_errors,
        "temperatures": temperatures,
        "samples_per_temperature": samples_per_temp,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    summary_path = work_dir / f"emobank_{safe_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
