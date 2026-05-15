"""
Prepare individual annotator label distributions from GoEmotions raw data.

This script reads the raw GoEmotions CSV files (which contain per-annotator
labels) and constructs label distributions for each text, preserving
individual annotator information instead of majority-vote aggregation.

Outputs:
    - goemotions_annotator_distributions.parquet
    - goemotions_metadata.parquet

Usage:
    python prepare_annotator_dist.py --config ../config/experiment_config.yaml
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_data_dir: str, raw_files: list[str] = None) -> pd.DataFrame:
    """Load raw GoEmotions CSV file (single merged file from HuggingFace).

    The raw CSV file has columns:
        text, id, author, subreddit, link_id, parent_id, created_utc,
        rater_id, example_very_unclear,
        admiration, amusement, anger, ..., neutral
    """
    goemotions_dir = os.path.join(raw_data_dir, "goemotions")
    filepath = os.path.join(goemotions_dir, "goemotions_raw.csv")
    print(f"  Loading {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Total rows (individual annotations): {len(df)}")
    print(f"  Unique texts: {df['id'].nunique()}")
    print(f"  Unique raters: {df['rater_id'].nunique()}")
    return df


def load_emotion_labels(raw_data_dir: str) -> list[str]:
    """Load emotion label names from emotions.txt."""
    emotions_path = os.path.join(raw_data_dir, "goemotions", "emotions.txt")
    with open(emotions_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(labels)} emotion labels")
    return labels


def build_annotator_distributions(
    df: pd.DataFrame, emotion_labels: list[str]
) -> pd.DataFrame:
    """Build per-text label distributions from individual annotator labels.

    For each text (identified by 'id'), compute the proportion of
    annotators who selected each emotion label. This creates a
    soft label distribution over 28 categories.

    Returns a DataFrame with columns:
        id, num_annotators, <emotion_1>_dist, ..., <emotion_28>_dist
    """
    print("\n=== Building annotator distributions ===")

    # Group by text ID
    grouped = df.groupby("id")

    records = []
    for text_id, group in grouped:
        num_annotators = len(group)

        # For each emotion, compute the fraction of annotators who selected it
        dist = {}
        for emotion in emotion_labels:
            if emotion in group.columns:
                dist[f"{emotion}_dist"] = group[emotion].mean()
            else:
                dist[f"{emotion}_dist"] = 0.0

        dist["id"] = text_id
        dist["num_annotators"] = num_annotators
        records.append(dist)

    dist_df = pd.DataFrame(records)

    # Reorder columns
    id_cols = ["id", "num_annotators"]
    dist_cols = [f"{e}_dist" for e in emotion_labels]
    dist_df = dist_df[id_cols + dist_cols]

    print(f"  Distributions built for {len(dist_df)} texts")
    print(f"  Annotators per text: min={dist_df['num_annotators'].min()}, "
          f"max={dist_df['num_annotators'].max()}, "
          f"mean={dist_df['num_annotators'].mean():.2f}")

    return dist_df


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique text metadata (one row per text ID).

    Takes the first occurrence of each text ID to get the text content
    and associated metadata.
    """
    print("\n=== Building metadata ===")

    metadata_cols = ["id", "text", "author", "subreddit", "created_utc"]
    available_cols = [c for c in metadata_cols if c in df.columns]

    metadata = df.drop_duplicates(subset=["id"], keep="first")[available_cols].copy()
    metadata = metadata.reset_index(drop=True)

    print(f"  Metadata entries: {len(metadata)}")
    print(f"  Columns: {list(metadata.columns)}")

    return metadata


def classify_agreement(dist_df: pd.DataFrame, emotion_labels: list[str]) -> pd.Series:
    """Classify each text's agreement level among annotators.

    Categories:
        - 'full_agreement': all annotators chose exactly the same label(s)
        - 'partial_agreement': some overlap in chosen labels
        - 'full_disagreement': no overlap in chosen labels

    This is determined by checking if each distribution value is
    either 0.0 or 1.0 (full agreement), a mix (partial), or
    if no single emotion exceeds 1/num_annotators (full disagreement).
    """
    dist_cols = [f"{e}_dist" for e in emotion_labels]

    def classify_row(row):
        dists = row[dist_cols].values
        # Full agreement: all values are 0 or 1
        if all(v == 0.0 or v == 1.0 for v in dists):
            return "full_agreement"
        # Full disagreement: no emotion was selected by more than one annotator
        # (assuming 3 annotators, max distribution for any label is 1/3)
        num_ann = row["num_annotators"]
        if num_ann > 0 and all(v <= 1.0 / num_ann for v in dists):
            return "full_disagreement"
        return "partial_agreement"

    return dist_df.apply(classify_row, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GoEmotions annotator distributions"
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
    raw_data_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    raw_files = config["data"]["goemotions"]["raw_files"]

    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load emotion labels
    print("=== Loading emotion labels ===")
    emotion_labels = load_emotion_labels(raw_data_dir)

    # 2. Load raw data
    print("\n=== Loading raw data ===")
    df = load_raw_data(raw_data_dir, raw_files)

    # 3. Build annotator distributions
    dist_df = build_annotator_distributions(df, emotion_labels)

    # 4. Classify agreement levels
    print("\n=== Classifying agreement levels ===")
    dist_df["agreement_level"] = classify_agreement(dist_df, emotion_labels)
    agreement_counts = dist_df["agreement_level"].value_counts()
    print(f"  Full agreement: {agreement_counts.get('full_agreement', 0)} "
          f"({agreement_counts.get('full_agreement', 0) / len(dist_df) * 100:.1f}%)")
    print(f"  Partial agreement: {agreement_counts.get('partial_agreement', 0)} "
          f"({agreement_counts.get('partial_agreement', 0) / len(dist_df) * 100:.1f}%)")
    print(f"  Full disagreement: {agreement_counts.get('full_disagreement', 0)} "
          f"({agreement_counts.get('full_disagreement', 0) / len(dist_df) * 100:.1f}%)")

    # 5. Build metadata
    metadata = build_metadata(df)

    # 6. Save outputs
    print("\n=== Saving outputs ===")
    dist_path = os.path.join(processed_dir, "goemotions_annotator_distributions.parquet")
    meta_path = os.path.join(processed_dir, "goemotions_metadata.parquet")

    dist_df.to_parquet(dist_path, index=False)
    print(f"  Saved distributions: {dist_path}")

    metadata.to_parquet(meta_path, index=False)
    print(f"  Saved metadata: {meta_path}")

    # 7. Print summary statistics
    print("\n=== Summary ===")
    print(f"  Total texts: {len(dist_df)}")
    print(f"  Distribution shape: {dist_df.shape}")
    print(f"  Metadata shape: {metadata.shape}")

    # Distribution statistics for each emotion
    dist_cols = [f"{e}_dist" for e in emotion_labels]
    print("\n=== Per-emotion statistics (mean distribution value) ===")
    for col in dist_cols:
        emotion_name = col.replace("_dist", "")
        mean_val = dist_df[col].mean()
        nonzero_frac = (dist_df[col] > 0).mean()
        print(f"  {emotion_name:20s}: mean={mean_val:.4f}, "
              f"nonzero={nonzero_frac:.4f} ({nonzero_frac * 100:.1f}%)")


if __name__ == "__main__":
    main()
