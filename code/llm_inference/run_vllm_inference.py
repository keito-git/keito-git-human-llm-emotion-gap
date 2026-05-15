"""
vLLM inference for OSS models (Qwen3-8B, Llama 3.1 8B) on GPU server.
Uses chat template with system + user prompt (aligned with API models).

GPU 1 only (CUDA_VISIBLE_DEVICES=1).

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_vllm_inference.py --model Qwen/Qwen3-8B --output-name qwen3-8b
    CUDA_VISIBLE_DEVICES=1 python run_vllm_inference.py --model meta-llama/Llama-3.1-8B-Instruct --output-name llama3.1-8b
"""

import argparse
import json
import os
import time
import csv
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from vllm import LLM, SamplingParams

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

SYSTEM_PROMPT = """You are an emotion annotation assistant. Your task is to identify the emotions expressed in a given text.

Available emotion labels (select ALL that apply):
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

Rules:
- Select one or more emotions from the list above.
- If no specific emotion is expressed, select "neutral".
- Return ONLY a JSON array of selected emotion labels, nothing else.
- Example: ["admiration", "joy"]
- Example: ["neutral"]"""


def build_user_prompt(text: str) -> str:
    return f'What emotions are expressed in this text?\n\nText: "{text}"'


def build_messages(text: str) -> list[dict]:
    """Build chat messages in system + user format (identical to API models)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]


def parse_response(response_text: str) -> list[str]:
    response_text = response_text.strip()
    # Handle Qwen3 thinking tags: skip <think>...</think> block
    think_end = response_text.find("</think>")
    if think_end != -1:
        response_text = response_text[think_end + len("</think>"):].strip()
    start = response_text.find("[")
    end = response_text.rfind("]")
    if start != -1 and end != -1:
        try:
            labels = json.loads(response_text[start:end + 1])
            valid = [l.lower().strip() for l in labels if l.lower().strip() in EMOTION_LABELS]
            return valid if valid else ["neutral"]
        except json.JSONDecodeError:
            pass
    return ["neutral"]


def labels_to_distribution(labels: list[str]) -> dict[str, float]:
    dist = {emo: 0.0 for emo in EMOTION_LABELS}
    for label in labels:
        if label in dist:
            dist[label] = 1.0
    return dist


def main():
    parser = argparse.ArgumentParser(description="vLLM inference with chat template")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--output-name", type=str, required=True,
                        help="Output name for results (e.g. qwen3-8b)")
    parser.add_argument("--work-dir", type=str,
                        default="/home/d9999993/qwen3_experiment",
                        help="Working directory")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True)

    # Load core set
    core_set_path = work_dir / "core_set.csv"
    if not core_set_path.exists():
        print(f"ERROR: {core_set_path} not found")
        return

    texts = []
    with open(core_set_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
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

    # Build all tasks with chat-templated prompts
    all_tasks = []
    for temp in temperatures:
        for sample_idx in range(samples_per_temp):
            for t in texts:
                messages = build_messages(t["text"])
                # Apply chat template to get the formatted prompt string
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

    # Check for checkpoint
    safe_name = args.output_name.replace(":", "_").replace(".", "_").replace("-", "_")
    checkpoint_path = work_dir / f"{safe_name}_v2_checkpoint.csv"
    output_path = work_dir / f"{safe_name}_v2_results.csv"

    results = []
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            reader = csv.DictReader(f)
            results = list(reader)
        completed_keys = set()
        for r in results:
            completed_keys.add((r["id"], str(r["temperature"]), str(r["sample_idx"])))
        all_tasks = [t for t in all_tasks
                     if (t["id"], str(t["temperature"]), str(t["sample_idx"])) not in completed_keys]
        print(f"Resumed from checkpoint: {len(results)} done, {len(all_tasks)} remaining")

    if not all_tasks:
        print("Nothing to do!")
        return

    # Process in batches grouped by (temperature, sample_idx) for consistent sampling params
    batch_size = 2000
    start_time = time.time()

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
            labels = parse_response(response)
            dist = labels_to_distribution(labels)

            results.append({
                "id": task["id"],
                "model": args.output_name,
                "temperature": task["temperature"],
                "sample_idx": task["sample_idx"],
                "raw_response": response[:1000],
                "parsed_labels": json.dumps(labels),
                "error": False,
                **{f"{emo}_pred": v for emo, v in dist.items()},
            })

        done = len(results)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate / 60 if rate > 0 else 0
        print(f"  Progress: {done}/{total} ({done/total*100:.1f}%), "
              f"rate: {rate:.1f} q/s, ETA: {remaining:.0f} min")

        # Checkpoint every 10000
        if done % 10000 < batch_size:
            fieldnames = list(results[0].keys())
            with open(checkpoint_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"  [checkpoint saved: {done} results]")

    # Save final results
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"\n=== Complete ===")
    print(f"Saved: {output_path}")
    print(f"Total: {len(results)}, Time: {elapsed/60:.1f} min")
    print(f"Rate: {len(all_tasks)/elapsed:.1f} q/s")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Save summary
    summary = {
        "model": args.model,
        "output_name": args.output_name,
        "prompt_format": "system_prompt + user_prompt (chat_template)",
        "total_queries": total,
        "results": len(results),
        "temperatures": temperatures,
        "samples_per_temperature": samples_per_temp,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    summary_path = work_dir / f"{safe_name}_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
