"""
Download GoEmotions raw data via HuggingFace datasets.

This downloads the 'raw' configuration which contains individual annotator
labels (rater_id), not the aggregated version.

Usage:
    python download_goemotions.py --config ../config/experiment_config.yaml
"""

import argparse
import os
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Download GoEmotions raw data via HuggingFace"
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
    goemotions_dir = os.path.join(raw_data_dir, "goemotions")
    os.makedirs(goemotions_dir, exist_ok=True)

    print("=== Downloading GoEmotions (raw) via HuggingFace ===")
    from datasets import load_dataset

    ds = load_dataset(
        "google-research-datasets/go_emotions", "raw", trust_remote_code=True
    )
    df = ds["train"].to_pandas()

    print(f"\n=== Dataset statistics ===")
    print(f"  Total rows (individual annotations): {len(df)}")
    print(f"  Unique texts: {df['id'].nunique()}")
    print(f"  Unique raters: {df['rater_id'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    # Save as CSV for compatibility with downstream scripts
    save_path = os.path.join(goemotions_dir, "goemotions_raw.csv")
    df.to_csv(save_path, index=False)
    print(f"\n  Saved: {save_path}")
    print(f"  Size: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB")

    # Save emotions list
    emotion_labels = config["data"]["goemotions"]["emotion_labels"]
    emotions_path = os.path.join(goemotions_dir, "emotions.txt")
    with open(emotions_path, "w") as f:
        for label in emotion_labels:
            f.write(label + "\n")
    print(f"  Saved emotion labels: {emotions_path}")

    print("\n=== Download complete ===")


if __name__ == "__main__":
    main()
