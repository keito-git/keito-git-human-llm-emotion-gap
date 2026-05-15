"""
Prepare EmoBank dataset for VAD analysis.

Builds:
    1. Per-text annotator VAD statistics (mean, std, individual ratings)
    2. Core set for LLM inference (stratified by VAD variance)
    3. Metadata with text content

EmoBank VAD scale: 1-5 (1=low, 5=high)
    V (Valence): negative to positive
    A (Arousal): calm to excited
    D (Dominance): submissive to dominant

Outputs:
    - emobank_annotator_stats.parquet
    - emobank_metadata.parquet
    - emobank_core_set.parquet

Usage:
    python prepare_emobank.py --config ../config/experiment_config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_emobank_data(raw_data_dir: str) -> dict:
    """Load all EmoBank data files."""
    emobank_dir = os.path.join(raw_data_dir, "emobank")

    # Aggregated data (has split info and text)
    agg_df = pd.read_csv(os.path.join(emobank_dir, "emobank.csv"))
    print(f"  Aggregated data: {len(agg_df)} texts")

    # Individual reader ratings (per-annotator VAD)
    indiv_reader = pd.read_csv(
        os.path.join(emobank_dir, "individual_reader_ratings.csv")
    )
    print(f"  Individual reader ratings: {len(indiv_reader)} rows")
    print(f"  Unique texts in reader ratings: {indiv_reader['id'].nunique()}")

    # Reader aggregated (includes std and N)
    reader_agg = pd.read_csv(os.path.join(emobank_dir, "reader.csv"))
    print(f"  Reader aggregated: {len(reader_agg)} texts")

    # Raw text data (may have more texts than emobank.csv)
    raw_text = pd.read_csv(os.path.join(emobank_dir, "raw.csv"))
    print(f"  Raw text data: {len(raw_text)} texts")

    return {
        "aggregated": agg_df,
        "individual_reader": indiv_reader,
        "reader_aggregated": reader_agg,
        "raw_text": raw_text,
    }


def build_annotator_stats(individual_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-text VAD statistics from individual annotator ratings.

    For each text, compute:
        - Mean, std, min, max for V, A, D
        - Number of annotators
        - Annotator agreement (inverse of mean std across VAD)
        - Individual annotator values stored as lists
    """
    print("\n=== Building annotator statistics ===")

    records = []
    for text_id, group in individual_df.groupby("id"):
        n_annotators = len(group)

        record = {
            "id": text_id,
            "num_annotators": n_annotators,
        }

        for dim in ["V", "A", "D"]:
            values = group[dim].values.astype(float)
            record[f"{dim}_mean"] = np.mean(values)
            record[f"{dim}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            record[f"{dim}_min"] = np.min(values)
            record[f"{dim}_max"] = np.max(values)
            record[f"{dim}_range"] = np.max(values) - np.min(values)
            # Store individual values as comma-separated string
            record[f"{dim}_individual"] = ",".join(str(v) for v in values)

        # Overall disagreement: mean std across VAD dimensions
        record["mean_vad_std"] = np.mean([
            record["V_std"], record["A_std"], record["D_std"]
        ])

        records.append(record)

    stats_df = pd.DataFrame(records)
    print(f"  Stats built for {len(stats_df)} texts")
    print(f"  Annotators per text: min={stats_df['num_annotators'].min()}, "
          f"max={stats_df['num_annotators'].max()}, "
          f"mean={stats_df['num_annotators'].mean():.2f}")
    print(f"  V mean range: [{stats_df['V_mean'].min():.2f}, {stats_df['V_mean'].max():.2f}]")
    print(f"  A mean range: [{stats_df['A_mean'].min():.2f}, {stats_df['A_mean'].max():.2f}]")
    print(f"  D mean range: [{stats_df['D_mean'].min():.2f}, {stats_df['D_mean'].max():.2f}]")

    return stats_df


def classify_vad_agreement(stats_df: pd.DataFrame) -> pd.Series:
    """Classify texts by annotator agreement level based on VAD std.

    Categories:
        - high_agreement: mean VAD std < 0.3 (annotators closely agree)
        - moderate_agreement: 0.3 <= mean VAD std < 0.7
        - low_agreement: mean VAD std >= 0.7 (substantial disagreement)
    """
    def classify(row):
        std = row["mean_vad_std"]
        if std < 0.3:
            return "high_agreement"
        elif std < 0.7:
            return "moderate_agreement"
        else:
            return "low_agreement"

    return stats_df.apply(classify, axis=1)


def sample_core_set(
    stats_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    core_set_size: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample a balanced core set for LLM inference.

    Stratified sampling to ensure:
        1. Balanced across agreement levels
        2. Balanced across VAD value ranges
        3. Only texts with >= 3 annotators (for meaningful distributions)

    Uses the emobank.csv split info to prefer test/dev texts.
    """
    print(f"\n=== Sampling core set (target: {core_set_size}) ===")
    rng = np.random.RandomState(seed)

    # Only texts with 3+ annotators
    eligible = stats_df[stats_df["num_annotators"] >= 3].copy()
    print(f"  Eligible texts (>= 3 annotators): {len(eligible)}")

    # Merge split info from aggregated data
    split_info = agg_df[["id", "split"]].drop_duplicates()
    eligible = eligible.merge(split_info, on="id", how="left")

    # Agreement classification
    eligible["agreement_level"] = classify_vad_agreement(eligible)
    print(f"  Agreement distribution:")
    for level, count in eligible["agreement_level"].value_counts().items():
        print(f"    {level}: {count} ({count / len(eligible) * 100:.1f}%)")

    # Valence bins for stratification (low/mid/high)
    eligible["V_bin"] = pd.cut(
        eligible["V_mean"], bins=[0, 2.5, 3.5, 6], labels=["low_V", "mid_V", "high_V"]
    )

    # Stratified sampling: agreement_level x V_bin
    strata = eligible.groupby(["agreement_level", "V_bin"], observed=True)
    n_strata = strata.ngroups
    per_stratum = max(1, core_set_size // n_strata)

    sampled = []
    for (agreement, v_bin), group in strata:
        n_sample = min(per_stratum, len(group))
        sampled.append(group.sample(n=n_sample, random_state=rng))

    core_set = pd.concat(sampled, ignore_index=True)

    # If under target, sample more from remaining
    if len(core_set) < core_set_size:
        remaining_ids = set(eligible["id"]) - set(core_set["id"])
        remaining = eligible[eligible["id"].isin(remaining_ids)]
        extra_n = min(core_set_size - len(core_set), len(remaining))
        if extra_n > 0:
            extra = remaining.sample(n=extra_n, random_state=rng)
            core_set = pd.concat([core_set, extra], ignore_index=True)

    # Trim to exact size if over
    if len(core_set) > core_set_size:
        core_set = core_set.sample(n=core_set_size, random_state=rng).reset_index(drop=True)

    print(f"  Core set size: {len(core_set)}")
    print(f"  Agreement distribution in core set:")
    for level, count in core_set["agreement_level"].value_counts().items():
        print(f"    {level}: {count} ({count / len(core_set) * 100:.1f}%)")

    return core_set


def main():
    parser = argparse.ArgumentParser(description="Prepare EmoBank data")
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
    processed_dir = config["paths"]["processed_data"]
    seed = config["analysis"]["random_seed"]

    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load data
    print("=== Loading EmoBank data ===")
    data = load_emobank_data(raw_data_dir)

    # 2. Build annotator statistics
    stats_df = build_annotator_stats(data["individual_reader"])

    # 3. Build metadata (text content)
    # Merge text from raw_text and aggregated
    raw_text = data["raw_text"][["id", "text"]].drop_duplicates(subset=["id"])
    agg_text = data["aggregated"][["id", "text"]].drop_duplicates(subset=["id"])
    # Use aggregated as primary (has split info), fill from raw
    text_lookup = pd.concat([agg_text, raw_text]).drop_duplicates(subset=["id"], keep="first")

    metadata = stats_df[["id"]].merge(text_lookup, on="id", how="left")
    print(f"\n=== Metadata ===")
    print(f"  Texts with content: {metadata['text'].notna().sum()}/{len(metadata)}")

    # 4. Sample core set
    core_set = sample_core_set(stats_df, data["aggregated"], core_set_size=2000, seed=seed)

    # 5. Save outputs
    print("\n=== Saving outputs ===")
    emobank_processed = os.path.join(processed_dir, "emobank")
    os.makedirs(emobank_processed, exist_ok=True)

    stats_path = os.path.join(emobank_processed, "emobank_annotator_stats.parquet")
    stats_df.to_parquet(stats_path, index=False)
    print(f"  Saved annotator stats: {stats_path}")

    meta_path = os.path.join(emobank_processed, "emobank_metadata.parquet")
    metadata.to_parquet(meta_path, index=False)
    print(f"  Saved metadata: {meta_path}")

    core_path = os.path.join(emobank_processed, "emobank_core_set.parquet")
    core_set.to_parquet(core_path, index=False)
    print(f"  Saved core set: {core_path}")

    # Also save core set texts as CSV for GPU server transfer
    core_with_text = core_set[["id"]].merge(metadata, on="id", how="left")
    core_csv_path = os.path.join(emobank_processed, "emobank_core_set_with_text.csv")
    core_with_text.to_csv(core_csv_path, index=False)
    print(f"  Saved core set CSV (for server): {core_csv_path}")

    # 6. Summary
    print(f"\n=== Summary ===")
    print(f"  Total texts with annotator stats: {len(stats_df)}")
    print(f"  Core set size: {len(core_set)}")
    print(f"  VAD scale: 1-5")
    print(f"  Annotation perspective: reader")


if __name__ == "__main__":
    main()
