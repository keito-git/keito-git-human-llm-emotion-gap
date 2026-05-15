"""
Phase 2: Run local LLM inference via Ollama API (parallel version).

Usage:
    python run_ollama_inference.py --config ../config/experiment_config.yaml --model qwen3:32b --workers 2
    python run_ollama_inference.py --config ../config/experiment_config.yaml --model llama3.1:8b --workers 4
"""

import argparse
import json
import os
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def parse_response(response_text: str) -> list[str]:
    response_text = response_text.strip()
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
    dist = {e: 0.0 for e in EMOTION_LABELS}
    for label in labels:
        if label in dist:
            dist[label] = 1.0
    return dist


def query_ollama(model: str, text: str, temperature: float, seed: int,
                 base_url: str = "http://localhost:11434") -> str:
    """Query Ollama API with system + user prompt (aligned with API models)."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": 500,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["message"]["content"]


def query_with_retry(fn, max_retries=5, base_delay=3.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


def save_checkpoint(results: list[dict], checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(checkpoint_path, index=False)


def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        df = pd.read_parquet(checkpoint_path)
        print(f"  [RESUME] Loaded checkpoint: {len(df)} results")
        return df
    return None


def run_inference(model_name: str, core_set, metadata, output_dir: str,
                  num_workers: int, temperatures: list, samples_per_temp: int, seed: int):
    texts = core_set.merge(metadata[["id", "text"]], on="id", how="left")

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace(":", "_").replace(".", "_")
    checkpoint_path = os.path.join(output_dir, f"{safe_name}_checkpoint.parquet")
    final_path = os.path.join(output_dir, f"{safe_name}_results.parquet")

    checkpoint_df = load_checkpoint(checkpoint_path)
    completed_keys = set()
    results = []
    if checkpoint_df is not None:
        results = checkpoint_df.to_dict("records")
        completed_keys = set(
            zip(checkpoint_df["id"], checkpoint_df["temperature"], checkpoint_df["sample_idx"])
        )

    tasks = []
    for temp in temperatures:
        for sample_idx in range(samples_per_temp):
            for _, row in texts.iterrows():
                if (row["id"], temp, sample_idx) not in completed_keys:
                    tasks.append({
                        "id": row["id"],
                        "text": row["text"],
                        "temperature": temp,
                        "sample_idx": sample_idx,
                    })

    total_queries = len(texts) * len(temperatures) * samples_per_temp
    print(f"\n=== Running {model_name} (parallel: {num_workers} workers) ===")
    print(f"  Texts: {len(texts)}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Samples per temperature: {samples_per_temp}")
    print(f"  Total queries: {total_queries}")
    print(f"  Already completed: {len(completed_keys)}")
    print(f"  Remaining: {len(tasks)}")

    if not tasks:
        print("  Nothing to do!")
        return

    lock = threading.Lock()
    errors = 0
    completed = len(completed_keys)
    last_checkpoint_count = 0

    def worker(task_idx_and_task):
        nonlocal errors, completed, last_checkpoint_count
        task_idx, task = task_idx_and_task

        try:
            response = query_with_retry(
                lambda: query_ollama(model_name, task["text"], task["temperature"],
                                     seed + task["sample_idx"])
            )
            labels = parse_response(response)
            dist = labels_to_distribution(labels)
            result = {
                "id": task["id"],
                "model": model_name,
                "temperature": task["temperature"],
                "sample_idx": task["sample_idx"],
                "raw_response": response,
                "parsed_labels": json.dumps(labels),
                "error": False,
                **{f"{emo}_pred": v for emo, v in dist.items()},
            }
        except Exception as exc:
            dist = labels_to_distribution(["neutral"])
            result = {
                "id": task["id"],
                "model": model_name,
                "temperature": task["temperature"],
                "sample_idx": task["sample_idx"],
                "raw_response": f"ERROR: {exc}",
                "parsed_labels": json.dumps(["neutral"]),
                "error": True,
                **{f"{emo}_pred": v for emo, v in dist.items()},
            }

        with lock:
            results.append(result)
            if result["error"]:
                errors += 1
            completed += 1
            new_since = completed - len(completed_keys)
            if new_since - last_checkpoint_count >= 200:
                save_checkpoint(results, checkpoint_path)
                last_checkpoint_count = new_since
                elapsed = time.time() - start_time
                rate = new_since / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - new_since) / rate / 60 if rate > 0 else 0
                print(f"    Progress: {completed}/{total_queries} "
                      f"({completed/total_queries*100:.1f}%), "
                      f"errors: {errors}, rate: {rate:.1f} q/s, "
                      f"ETA: {remaining:.0f} min [checkpoint]")

        return result

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, (i, t)): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"    Worker exception: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_parquet(final_path, index=False)
    elapsed = time.time() - start_time

    print(f"\n  Saved: {final_path}")
    print(f"  Total: {len(results_df)}, Errors: {errors}")
    print(f"  Time: {elapsed/60:.1f} min, Rate: {len(tasks)/elapsed:.1f} q/s")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    summary = {
        "model": model_name, "total_queries": total_queries,
        "errors": errors, "elapsed_minutes": round(elapsed/60, 1),
    }
    with open(os.path.join(output_dir, f"{safe_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment_config.yaml"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]
    output_dir = os.path.join(processed_dir, "llm_results")

    core_set = pd.read_parquet(os.path.join(processed_dir, "core_set.parquet"))
    metadata = pd.read_parquet(os.path.join(processed_dir, "goemotions_metadata.parquet"))

    temperatures = [0.0, 0.3, 0.7, 1.0]
    samples_per_temp = 10
    seed = 42

    print(f"Core set: {len(core_set)} texts")
    run_inference(args.model, core_set, metadata, output_dir, args.workers,
                  temperatures, samples_per_temp, seed)
    print("\n=== Inference complete ===")


if __name__ == "__main__":
    main()
