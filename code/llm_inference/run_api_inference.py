"""
Phase 2: Run LLM inference on GoEmotions core set via API (parallel version).

Features:
    - Parallel requests using ThreadPoolExecutor
    - Multiple OpenAI API keys for load distribution
    - Checkpoint every 500 queries (auto-resume on restart)
    - Retry with exponential backoff on errors

Usage:
    python run_api_inference.py --config ../config/experiment_config.yaml --model gpt-5.4-mini --workers 10
    python run_api_inference.py --config ../config/experiment_config.yaml --model claude-haiku-4-5-20251001 --workers 8
"""

import argparse
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_api_keys():
    """Load API keys for OpenAI / Anthropic.

    Lookup order (first hit wins, but later sources fill in missing keys):
      1. Environment variables already exported in the shell
      2. `<repo_root>/.env` (preferred for public release)
      3. `~/.claude/api_keys.env` (legacy local-dev location)

    Recognised keys:
      OPENAI_API_KEY, OPENAI_API_KEY_1, OPENAI_API_KEY_2, ...
      ANTHROPIC_API_KEY
    """
    keys = {}
    candidate_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
        os.path.expanduser("~/.claude/api_keys.env"),
    ]
    for env_path in candidate_paths:
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Don't overwrite already-loaded keys
                        if key not in keys:
                            keys[key] = value
                            os.environ.setdefault(key, value)
    # Pull any matching keys from the live environment as well
    for k, v in os.environ.items():
        if (k.startswith("OPENAI_API_KEY") or k.startswith("ANTHROPIC_API_KEY")) and k not in keys:
            keys[k] = v
    return keys


def get_openai_keys(all_keys: dict) -> list[str]:
    """Extract all OpenAI API keys for rotation."""
    oai_keys = []
    for k, v in all_keys.items():
        if k.startswith("OPENAI_API_KEY_") and v.startswith("sk-"):
            oai_keys.append(v)
    if not oai_keys and "OPENAI_API_KEY" in all_keys:
        oai_keys.append(all_keys["OPENAI_API_KEY"])
    return oai_keys


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


def query_with_retry(query_fn, max_retries: int = 5, base_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            return query_fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


def process_single_query(task: dict, provider: str, model_name: str, client) -> dict:
    """Process a single query and return result dict."""
    text_id = task["id"]
    text = task["text"]
    temp = task["temperature"]
    sample_idx = task["sample_idx"]
    seed = task["seed"]

    try:
        if provider == "openai":
            def _call():
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(text)},
                    ],
                    temperature=temp,
                    seed=seed + sample_idx,
                    max_completion_tokens=200,
                )
                return resp.choices[0].message.content
            response = query_with_retry(_call)
        else:
            def _call():
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_user_prompt(text)}],
                    temperature=temp,
                )
                return resp.content[0].text
            response = query_with_retry(_call)

        labels = parse_response(response)
        dist = labels_to_distribution(labels)
        return {
            "id": text_id,
            "model": model_name,
            "temperature": temp,
            "sample_idx": sample_idx,
            "raw_response": response,
            "parsed_labels": json.dumps(labels),
            "error": False,
            **{f"{emo}_pred": v for emo, v in dist.items()},
        }

    except Exception as exc:
        dist = labels_to_distribution(["neutral"])
        return {
            "id": text_id,
            "model": model_name,
            "temperature": temp,
            "sample_idx": sample_idx,
            "raw_response": f"ERROR: {exc}",
            "parsed_labels": json.dumps(["neutral"]),
            "error": True,
            **{f"{emo}_pred": v for emo, v in dist.items()},
        }


def load_checkpoint(checkpoint_path: str) -> pd.DataFrame | None:
    if os.path.exists(checkpoint_path):
        df = pd.read_parquet(checkpoint_path)
        print(f"  [RESUME] Loaded checkpoint: {len(df)} results")
        return df
    return None


def save_checkpoint(results: list[dict], checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(checkpoint_path, index=False)


def run_inference(
    config: dict,
    model_name: str,
    core_set: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: str,
    num_workers: int,
    all_keys: dict,
):
    # Find model config
    model_config = None
    provider = None
    for m in config["models"]["api"]:
        if m["name"] == model_name:
            model_config = m
            provider = m["provider"]
            break
    if model_config is None:
        raise ValueError(f"Model {model_name} not found in config")

    temperatures = model_config["temperature_settings"]
    samples_per_temp = model_config["samples_per_temperature"]
    seed = model_config["seed"]

    # Merge text
    texts = core_set.merge(metadata[["id", "text"]], on="id", how="left")

    # Checkpoint setup
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace(".", "_")
    checkpoint_path = os.path.join(output_dir, f"{safe_name}_checkpoint.parquet")
    final_path = os.path.join(output_dir, f"{safe_name}_results.parquet")

    # Load checkpoint
    checkpoint_df = load_checkpoint(checkpoint_path)
    completed_keys = set()
    results = []
    if checkpoint_df is not None:
        results = checkpoint_df.to_dict("records")
        completed_keys = set(
            zip(checkpoint_df["id"], checkpoint_df["temperature"], checkpoint_df["sample_idx"])
        )

    # Build task list (skip completed)
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
                        "seed": seed,
                    })

    total_queries = len(texts) * len(temperatures) * samples_per_temp
    print(f"\n=== Running {model_name} (parallel: {num_workers} workers) ===")
    print(f"  Provider: {provider}")
    print(f"  Texts: {len(texts)}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Samples per temperature: {samples_per_temp}")
    print(f"  Total queries: {total_queries}")
    print(f"  Already completed: {len(completed_keys)}")
    print(f"  Remaining: {len(tasks)}")

    if not tasks:
        print("  Nothing to do!")
        return

    # Create client pool
    if provider == "openai":
        from openai import OpenAI
        oai_keys = get_openai_keys(all_keys)
        num_keys = len(oai_keys)
        print(f"  OpenAI keys available: {num_keys}")
        clients = [OpenAI(api_key=oai_keys[i % num_keys]) for i in range(num_workers)]
    else:
        import anthropic
        clients = [anthropic.Anthropic() for _ in range(num_workers)]

    # Thread-safe results collection
    lock = threading.Lock()
    errors = 0
    completed = len(completed_keys)
    checkpoint_interval = 500
    last_checkpoint = 0

    def worker(task_idx_and_task):
        nonlocal errors, completed, last_checkpoint
        task_idx, task = task_idx_and_task
        client = clients[task_idx % num_workers]
        result = process_single_query(task, provider, model_name, client)

        with lock:
            results.append(result)
            if result["error"]:
                errors += 1
            completed += 1

            # Checkpoint
            new_since_checkpoint = completed - len(completed_keys) - last_checkpoint
            if new_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, checkpoint_path)
                last_checkpoint = completed - len(completed_keys)
                print(f"    Progress: {completed}/{total_queries} "
                      f"({completed/total_queries*100:.1f}%), "
                      f"errors: {errors} [checkpoint saved]")

        return result

    # Run parallel
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(worker, (i, task)): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"    Worker exception: {e}")

            # Print progress periodically
            if completed % 2000 == 0 and completed > 0:
                elapsed = time.time() - start_time
                rate = (completed - len(completed_keys)) / elapsed if elapsed > 0 else 0
                remaining = (len(tasks) - (completed - len(completed_keys))) / rate if rate > 0 else 0
                print(f"    Progress: {completed}/{total_queries} "
                      f"({completed/total_queries*100:.1f}%), "
                      f"rate: {rate:.1f} q/s, "
                      f"ETA: {remaining/60:.0f} min")

    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(final_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n  Saved final: {final_path}")
    print(f"  Total results: {len(results_df)}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Rate: {len(tasks)/elapsed:.1f} queries/sec")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Save summary
    summary = {
        "model": model_name,
        "provider": provider,
        "total_queries": total_queries,
        "successful": total_queries - errors,
        "errors": errors,
        "temperatures": temperatures,
        "samples_per_temperature": samples_per_temp,
        "elapsed_minutes": round(elapsed / 60, 1),
        "queries_per_second": round(len(tasks) / elapsed, 1),
    }
    summary_path = os.path.join(output_dir, f"{safe_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run LLM API inference (parallel)")
    parser.add_argument("--config", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment_config.yaml"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers")
    args = parser.parse_args()

    all_keys = load_api_keys()
    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]
    output_dir = os.path.join(processed_dir, "llm_results")

    core_set = pd.read_parquet(os.path.join(processed_dir, "core_set.parquet"))
    metadata = pd.read_parquet(os.path.join(processed_dir, "goemotions_metadata.parquet"))

    print(f"Core set: {len(core_set)} texts")

    run_inference(config, args.model, core_set, metadata, output_dir, args.workers, all_keys)

    print("\n=== Inference complete ===")


if __name__ == "__main__":
    main()
