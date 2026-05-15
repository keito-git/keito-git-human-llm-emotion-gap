"""
Sample core set and extended set from GoEmotions annotator distributions.

Core set (2,000 samples): stratified by agreement level
    - Full agreement: 500
    - Partial agreement: 1,000
    - Full disagreement: 500

Extended set (10,000 samples): stratified random sample for robustness checks.

Usage:
    python sample_core_set.py --config ../config/experiment_config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stratified_sample(
    dist_df: pd.DataFrame,
    n_full_agree: int,
    n_partial: int,
    n_full_disagree: int,
    seed: int,
    emotion_labels: list[str],
) -> pd.DataFrame:
    """Sample texts stratified by agreement level.

    Ensures the emotion category distribution in the sample
    reflects the overall dataset distribution.
    """
    rng = np.random.RandomState(seed)

    full_agree = dist_df[dist_df["agreement_level"] == "full_agreement"]
    partial = dist_df[dist_df["agreement_level"] == "partial_agreement"]
    full_disagree = dist_df[dist_df["agreement_level"] == "full_disagreement"]

    print(f"  Available full_agreement: {len(full_agree)}")
    print(f"  Available partial_agreement: {len(partial)}")
    print(f"  Available full_disagreement: {len(full_disagree)}")

    # Sample from each stratum
    n_full_agree = min(n_full_agree, len(full_agree))
    n_partial = min(n_partial, len(partial))
    n_full_disagree = min(n_full_disagree, len(full_disagree))

    sampled_full = full_agree.sample(n=n_full_agree, random_state=rng)
    sampled_partial = partial.sample(n=n_partial, random_state=rng)
    sampled_disagree = full_disagree.sample(n=n_full_disagree, random_state=rng)

    core_set = pd.concat([sampled_full, sampled_partial, sampled_disagree])
    core_set = core_set.sample(frac=1, random_state=rng).reset_index(drop=True)

    print(f"\n  Core set size: {len(core_set)}")
    print(f"  Composition:")
    print(f"    full_agreement: {n_full_agree}")
    print(f"    partial_agreement: {n_partial}")
    print(f"    full_disagreement: {n_full_disagree}")

    # Verify emotion distribution similarity
    dist_cols = [f"{e}_dist" for e in emotion_labels]
    print("\n  Emotion distribution comparison (mean, overall -> core set):")
    for col in dist_cols[:5]:  # show top 5 for brevity
        emotion = col.replace("_dist", "")
        overall_mean = dist_df[col].mean()
        core_mean = core_set[col].mean()
        print(f"    {emotion:20s}: {overall_mean:.4f} -> {core_mean:.4f}")
    print("    ...")

    return core_set


def sample_extended_set(
    dist_df: pd.DataFrame,
    core_set_ids: set,
    n_extended: int,
    seed: int,
) -> pd.DataFrame:
    """Sample extended set excluding core set texts."""
    rng = np.random.RandomState(seed + 1)  # different seed from core set

    available = dist_df[~dist_df["id"].isin(core_set_ids)]
    n_extended = min(n_extended, len(available))

    extended_set = available.sample(n=n_extended, random_state=rng).reset_index(drop=True)
    print(f"\n  Extended set size: {len(extended_set)}")

    return extended_set


def main():
    parser = argparse.ArgumentParser(
        description="Sample core and extended sets from GoEmotions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "config", "experiment_config.yaml"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]
    seed = config["analysis"]["random_seed"]
    emotion_labels = config["data"]["goemotions"]["emotion_labels"]
    sampling_config = config["sampling"]

    # Load distributions
    dist_path = os.path.join(processed_dir, "goemotions_annotator_distributions.parquet")
    dist_df = pd.read_parquet(dist_path)
    print(f"Loaded {len(dist_df)} texts\n")

    # Sample core set
    print("=== Sampling core set ===")
    core_set = stratified_sample(
        dist_df,
        n_full_agree=sampling_config["core_set_composition"]["full_agreement"],
        n_partial=sampling_config["core_set_composition"]["partial_agreement"],
        n_full_disagree=sampling_config["core_set_composition"]["full_disagreement"],
        seed=seed,
        emotion_labels=emotion_labels,
    )

    # Sample extended set
    print("\n=== Sampling extended set ===")
    core_ids = set(core_set["id"].values)
    extended_set = sample_extended_set(
        dist_df, core_ids, sampling_config["extended_set_size"], seed
    )

    # Save
    print("\n=== Saving ===")
    core_path = os.path.join(processed_dir, "core_set.parquet")
    extended_path = os.path.join(processed_dir, "extended_set.parquet")

    core_set.to_parquet(core_path, index=False)
    print(f"  Core set saved: {core_path}")

    extended_set.to_parquet(extended_path, index=False)
    print(f"  Extended set saved: {extended_path}")


if __name__ == "__main__":
    main()
