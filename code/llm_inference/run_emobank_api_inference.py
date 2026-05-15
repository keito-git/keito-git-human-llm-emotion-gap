"""
Phase 2b: Run LLM inference on EmoBank core set via API (VAD prediction).

Unlike GoEmotions (categorical), EmoBank requires continuous VAD predictions.
Each model predicts Valence, Arousal, and Dominance on a 1-5 scale.

Features:
    - Parallel requests using ThreadPoolExecutor
    - Multiple OpenAI API keys for load distribution
    - Checkpoint every 500 queries (auto-resume on restart)
    - Retry with exponential backoff on errors

Usage:
    python run_emobank_api_inference.py --model gpt-5.4-mini --workers 10
    python run_emobank_api_inference.py --model claude-haiku-4-5-20251001 --workers 8
"""

import argparse
import json
import os
import re
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

    Lookup order:
      1. Live environment variables
      2. `<repo_root>/.env` (preferred for public release)
      3. `~/.claude/api_keys.env` (legacy local-dev location)
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
                        if key not in keys:
                            keys[key] = value
                            os.environ.setdefault(key, value)
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


# Unified VAD prompt (same across all models)
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


def parse_vad_response(response_text: str) -> dict[str, float] | None:
    """Parse VAD values from LLM response.

    Returns dict with V, A, D keys or None if parsing fails.
    """
    response_text = response_text.strip()

    # Try to find JSON object
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(response_text[start:end + 1])
            v = float(obj.get("V", obj.get("v", -1)))
            a = float(obj.get("A", obj.get("a", -1)))
            d = float(obj.get("D", obj.get("d", -1)))

            # Validate range (allow slight overflow, clamp)
            if all(0.5 <= x <= 5.5 for x in [v, a, d]):
                return {
                    "V": max(1.0, min(5.0, v)),
                    "A": max(1.0, min(5.0, a)),
                    "D": max(1.0, min(5.0, d)),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: try regex for numbers
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
    """Process a single VAD query and return result dict."""
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

        vad = parse_vad_response(response)
        if vad is None:
            return {
                "id": text_id,
                "model": model_name,
                "temperature": temp,
                "sample_idx": sample_idx,
                "raw_response": response,
                "V_pred": np.nan,
                "A_pred": np.nan,
                "D_pred": np.nan,
                "parse_error": True,
                "api_error": False,
            }

        return {
            "id": text_id,
            "model": model_name,
            "temperature": temp,
            "sample_idx": sample_idx,
            "raw_response": response,
            "V_pred": vad["V"],
            "A_pred": vad["A"],
            "D_pred": vad["D"],
            "parse_error": False,
            "api_error": False,
        }

    except Exception as exc:
        return {
            "id": text_id,
            "model": model_name,
            "temperature": temp,
            "sample_idx": sample_idx,
            "raw_response": f"ERROR: {exc}",
            "V_pred": np.nan,
            "A_pred": np.nan,
            "D_pred": np.nan,
            "parse_error": False,
            "api_error": True,
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
    core_set_with_text: pd.DataFrame,
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

    # Checkpoint setup
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace(".", "_").replace("-", "_")
    checkpoint_path = os.path.join(output_dir, f"emobank_{safe_name}_checkpoint.parquet")
    final_path = os.path.join(output_dir, f"emobank_{safe_name}_results.parquet")

    # Load checkpoint
    checkpoint_df = load_checkpoint(checkpoint_path)
    completed_keys = set()
    results = []
    if checkpoint_df is not None:
        results = checkpoint_df.to_dict("records")
        completed_keys = set(
            zip(checkpoint_df["id"], checkpoint_df["temperature"], checkpoint_df["sample_idx"])
        )

    # Build task list
    tasks = []
    for temp in temperatures:
        for sample_idx in range(samples_per_temp):
            for _, row in core_set_with_text.iterrows():
                if (row["id"], temp, sample_idx) not in completed_keys:
                    tasks.append({
                        "id": row["id"],
                        "text": row["text"],
                        "temperature": temp,
                        "sample_idx": sample_idx,
                        "seed": seed,
                    })

    total_queries = len(core_set_with_text) * len(temperatures) * samples_per_temp
    print(f"\n=== Running EmoBank VAD inference: {model_name} (workers: {num_workers}) ===")
    print(f"  Provider: {provider}")
    print(f"  Texts: {len(core_set_with_text)}")
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

    # Thread-safe results
    lock = threading.Lock()
    errors = 0
    parse_errors = 0
    completed = len(completed_keys)
    checkpoint_interval = 500
    last_checkpoint = 0

    def worker(task_idx_and_task):
        nonlocal errors, parse_errors, completed, last_checkpoint
        task_idx, task = task_idx_and_task
        client = clients[task_idx % num_workers]
        result = process_single_query(task, provider, model_name, client)

        with lock:
            results.append(result)
            if result["api_error"]:
                errors += 1
            if result["parse_error"]:
                parse_errors += 1
            completed += 1

            new_since_checkpoint = completed - len(completed_keys) - last_checkpoint
            if new_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, checkpoint_path)
                last_checkpoint = completed - len(completed_keys)
                print(f"    Progress: {completed}/{total_queries} "
                      f"({completed/total_queries*100:.1f}%), "
                      f"api_errors: {errors}, parse_errors: {parse_errors} [checkpoint]")

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

    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(final_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n  Saved: {final_path}")
    print(f"  Total results: {len(results_df)}")
    print(f"  API errors: {errors}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Rate: {len(tasks)/elapsed:.1f} queries/sec")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Save summary
    valid_results = results_df[~results_df["api_error"] & ~results_df["parse_error"]]
    summary = {
        "dataset": "emobank",
        "task": "VAD_prediction",
        "model": model_name,
        "provider": provider,
        "total_queries": total_queries,
        "successful": len(valid_results),
        "api_errors": errors,
        "parse_errors": parse_errors,
        "temperatures": temperatures,
        "samples_per_temperature": samples_per_temp,
        "elapsed_minutes": round(elapsed / 60, 1),
        "queries_per_second": round(len(tasks) / elapsed, 1) if elapsed > 0 else 0,
        "V_pred_mean": float(valid_results["V_pred"].mean()) if len(valid_results) > 0 else None,
        "A_pred_mean": float(valid_results["A_pred"].mean()) if len(valid_results) > 0 else None,
        "D_pred_mean": float(valid_results["D_pred"].mean()) if len(valid_results) > 0 else None,
    }
    summary_path = os.path.join(output_dir, f"emobank_{safe_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run EmoBank VAD inference via API")
    parser.add_argument("--config", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment_config.yaml"))
    parser.add_argument("--model", type=str, required=True,
                        choices=["gpt-5.4-mini", "claude-haiku-4-5-20251001"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    all_keys = load_api_keys()
    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]
    output_dir = os.path.join(processed_dir, "emobank", "llm_results")

    # Load core set with text
    core_set_path = os.path.join(processed_dir, "emobank", "emobank_core_set.parquet")
    metadata_path = os.path.join(processed_dir, "emobank", "emobank_metadata.parquet")

    core_set = pd.read_parquet(core_set_path)
    metadata = pd.read_parquet(metadata_path)
    core_set_with_text = core_set[["id"]].merge(metadata[["id", "text"]], on="id", how="left")

    # Drop rows without text
    core_set_with_text = core_set_with_text.dropna(subset=["text"])
    print(f"Core set: {len(core_set_with_text)} texts")

    run_inference(config, args.model, core_set_with_text, output_dir, args.workers, all_keys)
    print("\n=== EmoBank inference complete ===")


if __name__ == "__main__":
    main()
