"""
Download and prepare EmoBank dataset for VAD (Valence-Arousal-Dominance) analysis.

EmoBank provides continuous VAD annotations from multiple annotators.
Source: https://github.com/JULIELab/EmoBank

Outputs:
    - data/raw/emobank/emobank.csv (aggregated VAD values)
    - data/raw/emobank/emobank_raw.csv (individual annotator VAD values)

Usage:
    python download_emobank.py --config ../config/experiment_config.yaml
"""

import argparse
import os
import ssl
import urllib.request
from pathlib import Path

import yaml

# Create unverified SSL context for environments with certificate issues
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


EMOBANK_BASE_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/"

FILES = {
    "emobank.csv": f"{EMOBANK_BASE_URL}emobank.csv",
    "raw.csv": f"{EMOBANK_BASE_URL}raw.csv",
    "reader.csv": f"{EMOBANK_BASE_URL}reader.csv",
    "writer.csv": f"{EMOBANK_BASE_URL}writer.csv",
    "individual_reader_ratings.csv": f"{EMOBANK_BASE_URL}individual_reader_ratings.csv",
    "individual_writer_ratings.csv": f"{EMOBANK_BASE_URL}individual_writer_ratings.csv",
    "meta.tsv": f"{EMOBANK_BASE_URL}meta.tsv",
}


def download_file(url: str, dest: str):
    """Download a file with progress indication."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=_ssl_context) as response:
        with open(dest, "wb") as out_file:
            out_file.write(response.read())
    size_kb = os.path.getsize(dest) / 1024
    print(f"  Done ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Download EmoBank dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "config", "experiment_config.yaml"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_data_dir = config["paths"]["raw_data"]

    emobank_dir = os.path.join(raw_data_dir, "emobank")
    os.makedirs(emobank_dir, exist_ok=True)

    print("=== Downloading EmoBank dataset ===")
    for filename, url in FILES.items():
        dest = os.path.join(emobank_dir, filename)
        download_file(url, dest)

    print("\n=== Download complete ===")
    print(f"  Files saved to: {emobank_dir}")

    # Quick validation
    import pandas as pd

    agg_path = os.path.join(emobank_dir, "emobank.csv")
    reader_indiv_path = os.path.join(emobank_dir, "individual_reader_ratings.csv")

    agg_df = pd.read_csv(agg_path)
    print(f"\n=== EmoBank aggregated data ===")
    print(f"  Shape: {agg_df.shape}")
    print(f"  Columns: {list(agg_df.columns)}")
    print(f"  Splits: {agg_df['split'].value_counts().to_dict()}")
    print(f"  V range: [{agg_df['V'].min():.2f}, {agg_df['V'].max():.2f}]")
    print(f"  A range: [{agg_df['A'].min():.2f}, {agg_df['A'].max():.2f}]")
    print(f"  D range: [{agg_df['D'].min():.2f}, {agg_df['D'].max():.2f}]")

    if os.path.exists(reader_indiv_path):
        reader_df = pd.read_csv(reader_indiv_path)
        print(f"\n=== EmoBank individual reader ratings ===")
        print(f"  Shape: {reader_df.shape}")
        print(f"  Columns: {list(reader_df.columns)}")
        if "id" in reader_df.columns:
            print(f"  Unique texts: {reader_df['id'].nunique()}")


if __name__ == "__main__":
    main()
