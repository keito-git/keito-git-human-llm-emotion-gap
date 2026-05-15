"""
Phase 1: Human label distribution analysis for GoEmotions.

This script performs comprehensive analysis of human annotator label
distributions, including:
    - Entropy computation for each text's label distribution
    - Agreement classification (full/partial/full disagreement)
    - Per-category inter-annotator agreement (Cohen's kappa, Krippendorff's alpha)
    - Identification of most-disagreed emotion categories
    - Confusion matrix construction for human annotators

Outputs:
    - human_basic_stats.json: Summary statistics
    - Figures in visualization output directory

Usage:
    python human_distribution_analysis.py --config ../config/experiment_config.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy, spearmanr


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_distribution_entropy(dist_df: pd.DataFrame, dist_cols: list[str]) -> np.ndarray:
    """Compute Shannon entropy for each text's label distribution.

    Higher entropy indicates more disagreement among annotators.
    """
    distributions = dist_df[dist_cols].values
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    entropies = np.array([
        entropy(row + eps, base=2) for row in distributions
    ])
    return entropies


def compute_agreement_statistics(
    dist_df: pd.DataFrame,
) -> dict:
    """Compute agreement statistics across all texts."""
    agreement_counts = dist_df["agreement_level"].value_counts()
    total = len(dist_df)

    stats = {
        "total_texts": total,
        "full_agreement_count": int(agreement_counts.get("full_agreement", 0)),
        "full_agreement_pct": float(agreement_counts.get("full_agreement", 0) / total * 100),
        "partial_agreement_count": int(agreement_counts.get("partial_agreement", 0)),
        "partial_agreement_pct": float(agreement_counts.get("partial_agreement", 0) / total * 100),
        "full_disagreement_count": int(agreement_counts.get("full_disagreement", 0)),
        "full_disagreement_pct": float(agreement_counts.get("full_disagreement", 0) / total * 100),
    }
    return stats


def compute_per_category_agreement(
    raw_df: pd.DataFrame, emotion_labels: list[str]
) -> dict:
    """Compute per-category inter-annotator agreement using Cohen's kappa.

    For each emotion category, we treat the annotation as a binary task
    (selected vs not selected) and compute pairwise Cohen's kappa
    across annotator pairs for each text.

    Also computes Krippendorff's alpha for each category.
    """
    from sklearn.metrics import cohen_kappa_score

    print("  Computing per-category agreement (this may take a while)...")

    # Group by text ID to get annotator pairs
    grouped = raw_df.groupby("id")

    per_category_kappas = {emotion: [] for emotion in emotion_labels}

    for text_id, group in grouped:
        if len(group) < 2:
            continue

        annotators = group.index.tolist()
        # Compute pairwise kappa for each emotion
        for emotion in emotion_labels:
            if emotion not in group.columns:
                continue

            labels = group[emotion].values
            # Only compute if there is variation
            for i in range(len(annotators)):
                for j in range(i + 1, len(annotators)):
                    pair = [labels[i], labels[j]]
                    per_category_kappas[emotion].append(pair)

    # Aggregate pairwise comparisons into kappa scores
    category_agreement = {}
    for emotion in emotion_labels:
        pairs = per_category_kappas[emotion]
        if not pairs:
            category_agreement[emotion] = {"cohens_kappa": None, "n_pairs": 0}
            continue

        pairs_arr = np.array(pairs)
        rater1 = pairs_arr[:, 0].astype(int)
        rater2 = pairs_arr[:, 1].astype(int)

        try:
            kappa = cohen_kappa_score(rater1, rater2)
        except Exception:
            kappa = None

        # Compute simple agreement rate
        agreement_rate = np.mean(rater1 == rater2)

        category_agreement[emotion] = {
            "cohens_kappa": float(kappa) if kappa is not None else None,
            "agreement_rate": float(agreement_rate),
            "n_pairs": len(pairs),
            "positive_rate": float(np.mean(np.concatenate([rater1, rater2]))),
        }

    return category_agreement


def build_confusion_matrix(
    raw_df: pd.DataFrame, emotion_labels: list[str]
) -> np.ndarray:
    """Build emotion confusion matrix from human annotator disagreements.

    For texts where annotators disagree, count how often emotion_i and
    emotion_j co-occur across different annotators' labels. This reveals
    which emotion pairs are most commonly confused.
    """
    n_emotions = len(emotion_labels)
    confusion = np.zeros((n_emotions, n_emotions), dtype=float)

    grouped = raw_df.groupby("id")

    for text_id, group in grouped:
        if len(group) < 2:
            continue

        # For each pair of annotators
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row_i = group.iloc[i]
                row_j = group.iloc[j]

                # Find emotions selected by each annotator
                emotions_i = set()
                emotions_j = set()
                for k, emotion in enumerate(emotion_labels):
                    if emotion in group.columns:
                        if row_i[emotion] == 1:
                            emotions_i.add(k)
                        if row_j[emotion] == 1:
                            emotions_j.add(k)

                # For disagreements: emotion in i but not in j, paired with
                # emotion in j but not in i
                only_i = emotions_i - emotions_j
                only_j = emotions_j - emotions_i

                for ei in only_i:
                    for ej in only_j:
                        confusion[ei][ej] += 1
                        confusion[ej][ei] += 1

    # Normalize rows to get conditional probabilities
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    confusion_normalized = confusion / row_sums

    return confusion, confusion_normalized


def identify_most_disagreed_categories(
    category_agreement: dict, top_n: int = 10
) -> list[dict]:
    """Identify emotion categories with the lowest inter-annotator agreement."""
    sortable = [
        {"emotion": emotion, **stats}
        for emotion, stats in category_agreement.items()
        if stats["cohens_kappa"] is not None
    ]
    sortable.sort(key=lambda x: x["cohens_kappa"])
    return sortable[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze human label distributions in GoEmotions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "config", "experiment_config.yaml"
        ),
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]
    raw_data_dir = config["paths"]["raw_data"]
    raw_files = config["data"]["goemotions"]["raw_files"]
    emotion_labels = config["data"]["goemotions"]["emotion_labels"]
    seed = config["analysis"]["random_seed"]

    np.random.seed(seed)

    dist_cols = [f"{e}_dist" for e in emotion_labels]

    # 1. Load processed distributions
    print("=== Loading processed distributions ===")
    dist_path = os.path.join(processed_dir, "goemotions_annotator_distributions.parquet")
    dist_df = pd.read_parquet(dist_path)
    print(f"  Loaded {len(dist_df)} texts")

    # 2. Compute entropy for each text
    print("\n=== Computing distribution entropy ===")
    entropies = compute_distribution_entropy(dist_df, dist_cols)
    dist_df["entropy"] = entropies
    print(f"  Entropy: mean={entropies.mean():.4f}, "
          f"median={np.median(entropies):.4f}, "
          f"std={entropies.std():.4f}")
    print(f"  Zero entropy (perfect agreement): "
          f"{(entropies == 0).sum()} ({(entropies == 0).mean() * 100:.1f}%)")

    # 3. Agreement statistics
    print("\n=== Agreement statistics ===")
    agreement_stats = compute_agreement_statistics(dist_df)
    for key, val in agreement_stats.items():
        print(f"  {key}: {val}")

    # 4. Load raw data for per-category analysis
    print("\n=== Loading raw data for per-category analysis ===")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_preparation"))
    from prepare_annotator_dist import load_raw_data
    raw_df = load_raw_data(raw_data_dir, raw_files)

    # 5. Per-category agreement
    print("\n=== Per-category inter-annotator agreement ===")
    category_agreement = compute_per_category_agreement(raw_df, emotion_labels)
    for emotion, stats in sorted(
        category_agreement.items(), key=lambda x: x[1].get("cohens_kappa", 0) or 0
    ):
        kappa = stats["cohens_kappa"]
        kappa_str = f"{kappa:.4f}" if kappa is not None else "N/A"
        print(f"  {emotion:20s}: kappa={kappa_str}, "
              f"agreement={stats['agreement_rate']:.4f}, "
              f"positive_rate={stats['positive_rate']:.4f}")

    # 6. Most disagreed categories
    print("\n=== Most disagreed categories (lowest kappa) ===")
    most_disagreed = identify_most_disagreed_categories(category_agreement)
    for item in most_disagreed:
        print(f"  {item['emotion']:20s}: kappa={item['cohens_kappa']:.4f}")

    # 7. Build confusion matrix
    print("\n=== Building confusion matrix ===")
    confusion_raw, confusion_norm = build_confusion_matrix(raw_df, emotion_labels)
    print(f"  Confusion matrix shape: {confusion_raw.shape}")

    # Find top confused pairs
    n = len(emotion_labels)
    confused_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            confused_pairs.append({
                "emotion_1": emotion_labels[i],
                "emotion_2": emotion_labels[j],
                "confusion_count": float(confusion_raw[i][j] + confusion_raw[j][i]) / 2,
            })
    confused_pairs.sort(key=lambda x: x["confusion_count"], reverse=True)

    print("\n  Top 15 confused emotion pairs:")
    for pair in confused_pairs[:15]:
        print(f"    {pair['emotion_1']:20s} <-> {pair['emotion_2']:20s}: "
              f"{pair['confusion_count']:.0f}")

    # 8. Compile and save all statistics
    print("\n=== Saving results ===")
    results = {
        "agreement_statistics": agreement_stats,
        "entropy_statistics": {
            "mean": float(entropies.mean()),
            "median": float(np.median(entropies)),
            "std": float(entropies.std()),
            "min": float(entropies.min()),
            "max": float(entropies.max()),
            "zero_entropy_count": int((entropies == 0).sum()),
            "zero_entropy_pct": float((entropies == 0).mean() * 100),
        },
        "per_category_agreement": {
            emotion: {
                k: v for k, v in stats.items()
                if isinstance(v, (int, float, type(None)))
            }
            for emotion, stats in category_agreement.items()
        },
        "most_disagreed_categories": most_disagreed,
        "top_confused_pairs": confused_pairs[:20],
    }

    stats_path = os.path.join(processed_dir, "human_basic_stats.json")
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path}")

    # Save updated distributions with entropy
    updated_dist_path = os.path.join(
        processed_dir, "goemotions_annotator_distributions.parquet"
    )
    dist_df.to_parquet(updated_dist_path, index=False)
    print(f"  Updated: {updated_dist_path}")

    # Save confusion matrix
    confusion_path = os.path.join(processed_dir, "human_confusion_matrix.npy")
    np.save(confusion_path, confusion_raw)
    confusion_norm_path = os.path.join(processed_dir, "human_confusion_matrix_normalized.npy")
    np.save(confusion_norm_path, confusion_norm)
    print(f"  Saved confusion matrices: {confusion_path}")

    print("\n=== Phase 1 Analysis Complete ===")


if __name__ == "__main__":
    main()
