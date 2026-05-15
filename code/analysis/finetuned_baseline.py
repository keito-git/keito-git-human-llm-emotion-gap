"""
W5: Fine-tuned baseline evaluation on GoEmotions core set.

Uses a HuggingFace GoEmotions fine-tuned model (SamLowe/roberta-base-go_emotions)
to get emotion predictions on our 2,000-text core set, then compares distributional
metrics with zero-shot LLM results.

Usage (on GPU server):
    CUDA_VISIBLE_DEVICES=1 python3 finetuned_baseline.py \
        --input_csv /path/to/core_set_with_text.csv \
        --output_dir /path/to/output/

Outputs:
    - finetuned_predictions.csv
    - finetuned_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import spearmanr, entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon


# ============================================================
# Constants
# ============================================================

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# SamLowe model uses the exact same 28 GoEmotions labels
MODEL_NAME = "SamLowe/roberta-base-go_emotions"


def normalize_distribution(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize to sum to 1."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.maximum(dist, 0)
    total = dist.sum()
    if total == 0:
        return np.ones_like(dist) / len(dist)
    return dist / total


def shannon_entropy(dist: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy."""
    dist = normalize_distribution(dist)
    return float(scipy_entropy(dist, base=base))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD (squared, [0, 1])."""
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return float(jensenshannon(p, q, base=2) ** 2)


def run_finetuned_inference(
    texts: list[str],
    text_ids: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run fine-tuned model inference and return probability distributions."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Get label mapping from model config
    id2label = model.config.id2label
    model_labels = [id2label[i] for i in range(len(id2label))]
    print(f"Model labels ({len(model_labels)}): {model_labels[:5]}...")

    # Map model label order to our standard order
    label_to_idx = {label: i for i, label in enumerate(model_labels)}

    all_probs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # Softmax over logits -> probability distribution
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches}")

    all_probs = np.vstack(all_probs)  # (N, num_model_labels)

    # Re-order to match our standard emotion label order
    reordered_probs = np.zeros((len(texts), len(EMOTION_LABELS)))
    for i, label in enumerate(EMOTION_LABELS):
        if label in label_to_idx:
            reordered_probs[:, i] = all_probs[:, label_to_idx[label]]
        else:
            print(f"  WARNING: label '{label}' not found in model output")

    # Build DataFrame
    records = []
    for i in range(len(texts)):
        record = {"id": text_ids[i], "model": "RoBERTa-GoEm (fine-tuned)"}
        for j, label in enumerate(EMOTION_LABELS):
            record[f"{label}_dist"] = float(reordered_probs[i, j])
        dist = normalize_distribution(reordered_probs[i])
        record["entropy"] = shannon_entropy(dist)
        records.append(record)

    return pd.DataFrame(records)


def compute_comparison_metrics(
    human_df: pd.DataFrame,
    finetuned_df: pd.DataFrame,
) -> dict:
    """Compute JSD, entropy correlation, per-category metrics."""
    dist_cols = [f"{e}_dist" for e in EMOTION_LABELS]

    merged = finetuned_df.merge(
        human_df[["id"] + dist_cols + ["agreement_level", "entropy"]],
        on="id",
        suffixes=("_ft", "_human"),
    )

    # Per-text JSD
    jsds = []
    for _, row in merged.iterrows():
        h_dist = np.array([row[f"{e}_dist_human"] for e in EMOTION_LABELS])
        f_dist = np.array([row[f"{e}_dist_ft"] for e in EMOTION_LABELS])
        jsds.append(jensen_shannon_divergence(h_dist, f_dist))

    merged["jsd"] = jsds

    # Overall JSD
    jsd_mean = float(np.mean(jsds))
    jsd_std = float(np.std(jsds))
    jsd_median = float(np.median(jsds))

    # Entropy correlation
    rho, p_val = spearmanr(merged["entropy_human"], merged["entropy_ft"])

    # Per-agreement-level JSD
    agreement_stats = {}
    for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
        subset = merged[merged["agreement_level"] == level]
        if len(subset) > 0:
            agreement_stats[level] = {
                "n": len(subset),
                "jsd_mean": float(subset["jsd"].mean()),
                "jsd_std": float(subset["jsd"].std()),
            }

    # Per-category analysis
    per_category = {}
    for emotion in EMOTION_LABELS:
        h_rates = merged[f"{emotion}_dist_human"].values
        f_rates = merged[f"{emotion}_dist_ft"].values
        delta = float(np.mean(f_rates) - np.mean(h_rates))
        rho_cat, p_cat = spearmanr(h_rates, f_rates)
        n_nonzero = int(np.sum(h_rates > 0))
        per_category[emotion] = {
            "delta": delta,
            "spearman_rho": float(rho_cat) if not np.isnan(rho_cat) else 0.0,
            "spearman_p": float(p_cat) if not np.isnan(p_cat) else 1.0,
            "human_mean_rate": float(np.mean(h_rates)),
            "ft_mean_rate": float(np.mean(f_rates)),
            "n_nonzero_human": n_nonzero,
        }

    return {
        "model": "RoBERTa-GoEm (fine-tuned)",
        "model_hf_name": MODEL_NAME,
        "n_texts": len(merged),
        "overall": {
            "jsd_mean": jsd_mean,
            "jsd_std": jsd_std,
            "jsd_median": jsd_median,
        },
        "entropy_correlation": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
        },
        "agreement_level": agreement_stats,
        "per_category": per_category,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Core set CSV with text and id columns")
    parser.add_argument("--human_parquet", required=True, help="Human distribution parquet (core_set.parquet)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    text_df = pd.read_csv(args.input_csv)
    human_df = pd.read_parquet(args.human_parquet)

    # Merge text with IDs
    if "text" not in text_df.columns:
        raise ValueError("input_csv must have 'text' column")

    texts = text_df["text"].tolist()
    text_ids = text_df["id"].tolist()

    print(f"Running inference on {len(texts)} texts...")
    ft_df = run_finetuned_inference(
        texts, text_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save predictions
    pred_path = os.path.join(args.output_dir, "finetuned_predictions.csv")
    ft_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    # Compute metrics
    print("Computing comparison metrics...")
    results = compute_comparison_metrics(human_df, ft_df)

    # Save results
    results_path = os.path.join(args.output_dir, "finetuned_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Print summary
    print("\n=== Fine-tuned Baseline Results ===")
    print(f"  JSD (mean): {results['overall']['jsd_mean']:.3f}")
    print(f"  JSD (std):  {results['overall']['jsd_std']:.3f}")
    print(f"  Entropy rho: {results['entropy_correlation']['spearman_rho']:.3f}")
    for level, stats in results["agreement_level"].items():
        print(f"  JSD [{level}]: {stats['jsd_mean']:.3f} (n={stats['n']})")


if __name__ == "__main__":
    main()
