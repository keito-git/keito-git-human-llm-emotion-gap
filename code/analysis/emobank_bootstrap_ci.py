"""
EmoBank Bootstrap CI and Effect Size Analysis.

Computes:
    1. Bootstrap 95% CIs for MAE and correlation per model per dimension
    2. Cohen's d and Cliff's delta for pairwise model comparisons
    3. Summary statistics for paper integration

Usage:
    python emobank_bootstrap_ci.py --config ../config/experiment_config.yaml
"""

import argparse
import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(42)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_dir: str):
    """Load human and LLM data."""
    emobank_dir = os.path.join(processed_dir, "emobank")
    stats = pd.read_parquet(os.path.join(emobank_dir, "emobank_annotator_stats.parquet"))
    core_set = pd.read_parquet(os.path.join(emobank_dir, "emobank_core_set.parquet"))
    core_stats = stats[stats["id"].isin(core_set["id"])].copy()

    results_dir = os.path.join(emobank_dir, "llm_results")
    model_results = {}
    for f in os.listdir(results_dir):
        if f.startswith("emobank_") and f.endswith("_results.parquet"):
            model_name = f.replace("emobank_", "").replace("_results.parquet", "")
            df = pd.read_parquet(os.path.join(results_dir, f))
            mask = df["V_pred"].notna() & df["A_pred"].notna() & df["D_pred"].notna()
            if "api_error" in df.columns:
                mask &= ~df["api_error"]
            if "parse_error" in df.columns:
                mask &= ~df["parse_error"]
            model_results[model_name] = df[mask].copy()

    return core_stats, core_set, model_results


def compute_per_text_errors(core_stats, model_results):
    """Compute per-text MAE for each model and dimension."""
    per_text = {}
    for model_name, df in model_results.items():
        llm_means = df.groupby("id").agg({
            "V_pred": "mean", "A_pred": "mean", "D_pred": "mean"
        }).reset_index()
        merged = core_stats[["id", "V_mean", "A_mean", "D_mean"]].merge(
            llm_means, on="id", how="inner"
        )
        errors = {}
        for dim in ["V", "A", "D"]:
            errors[f"{dim}_ae"] = np.abs(merged[f"{dim}_mean"] - merged[f"{dim}_pred"]).values
            errors[f"{dim}_human"] = merged[f"{dim}_mean"].values
            errors[f"{dim}_llm"] = merged[f"{dim}_pred"].values
        errors["ids"] = merged["id"].values
        per_text[model_name] = errors
    return per_text


def bootstrap_ci(values, n_iter=1000, ci=0.95):
    """Compute bootstrap CI for the mean."""
    means = []
    n = len(values)
    for _ in range(n_iter):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lower), float(upper)


def bootstrap_correlation_ci(x, y, n_iter=1000, ci=0.95):
    """Bootstrap CI for Pearson correlation."""
    corrs = []
    n = len(x)
    for _ in range(n_iter):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = scipy_stats.pearsonr(x[idx], y[idx])
        corrs.append(r)
    lower = np.percentile(corrs, (1 - ci) / 2 * 100)
    upper = np.percentile(corrs, (1 + ci) / 2 * 100)
    r_obs, _ = scipy_stats.pearsonr(x, y)
    return float(r_obs), float(lower), float(upper)


def cohens_d(a, b):
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def cliffs_delta(a, b):
    """Cliff's delta (non-parametric effect size)."""
    n_a, n_b = len(a), len(b)
    # Efficient computation
    more = 0
    less = 0
    for x in a:
        more += np.sum(x > b)
        less += np.sum(x < b)
    return float((more - less) / (n_a * n_b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment_config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]

    print("=== Loading data ===")
    core_stats, core_set, model_results = load_data(processed_dir)
    per_text = compute_per_text_errors(core_stats, model_results)

    results = {}

    # 1. Bootstrap CIs for MAE per model per dimension
    print("\n=== Bootstrap CIs for MAE ===")
    bootstrap_results = {}
    for model_name, errors in per_text.items():
        model_boot = {}
        for dim in ["V", "A", "D"]:
            mean, lower, upper = bootstrap_ci(errors[f"{dim}_ae"])
            model_boot[f"{dim}_mae"] = mean
            model_boot[f"{dim}_mae_ci_lower"] = lower
            model_boot[f"{dim}_mae_ci_upper"] = upper
            print(f"  {model_name} {dim} MAE: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

        # Overall MAE
        overall = np.concatenate([errors[f"{dim}_ae"] for dim in ["V", "A", "D"]])
        mean, lower, upper = bootstrap_ci(overall)
        model_boot["overall_mae"] = mean
        model_boot["overall_mae_ci_lower"] = lower
        model_boot["overall_mae_ci_upper"] = upper
        print(f"  {model_name} Overall MAE: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

        bootstrap_results[model_name] = model_boot

    results["bootstrap_mae"] = bootstrap_results

    # 2. Bootstrap CIs for correlation per model per dimension
    print("\n=== Bootstrap CIs for Correlation ===")
    corr_results = {}
    for model_name, errors in per_text.items():
        model_corr = {}
        for dim in ["V", "A", "D"]:
            r, lower, upper = bootstrap_correlation_ci(
                errors[f"{dim}_human"], errors[f"{dim}_llm"]
            )
            model_corr[f"{dim}_pearson_r"] = r
            model_corr[f"{dim}_pearson_ci_lower"] = lower
            model_corr[f"{dim}_pearson_ci_upper"] = upper
            print(f"  {model_name} {dim} r: {r:.4f} [{lower:.4f}, {upper:.4f}]")

        corr_results[model_name] = model_corr

    results["bootstrap_correlation"] = corr_results

    # 3. Pairwise effect sizes (overall MAE)
    print("\n=== Pairwise Effect Sizes (Overall MAE) ===")
    effect_sizes = {}
    model_names = list(per_text.keys())
    for m1, m2 in combinations(model_names, 2):
        overall_1 = np.concatenate([per_text[m1][f"{dim}_ae"] for dim in ["V", "A", "D"]])
        overall_2 = np.concatenate([per_text[m2][f"{dim}_ae"] for dim in ["V", "A", "D"]])
        d = cohens_d(overall_1, overall_2)
        delta = cliffs_delta(overall_1, overall_2)
        _, mw_p = scipy_stats.mannwhitneyu(overall_1, overall_2, alternative="two-sided")
        key = f"{m1}_vs_{m2}"
        effect_sizes[key] = {
            "cohens_d": d,
            "cliffs_delta": delta,
            "mw_pvalue": float(mw_p),
        }
        print(f"  {m1} vs {m2}: d={d:.3f}, delta={delta:.3f}, p={mw_p:.2e}")

    # API vs OSS
    api_models = [m for m in model_names if "gpt" in m or "claude" in m]
    oss_models = [m for m in model_names if "llama" in m or "qwen" in m]
    if api_models and oss_models:
        api_errors = np.concatenate([
            np.concatenate([per_text[m][f"{dim}_ae"] for dim in ["V", "A", "D"]])
            for m in api_models
        ])
        oss_errors = np.concatenate([
            np.concatenate([per_text[m][f"{dim}_ae"] for dim in ["V", "A", "D"]])
            for m in oss_models
        ])
        d = cohens_d(api_errors, oss_errors)
        delta = cliffs_delta(
            # Use subsample for efficiency
            np.random.choice(api_errors, size=min(5000, len(api_errors)), replace=False),
            np.random.choice(oss_errors, size=min(5000, len(oss_errors)), replace=False)
        )
        _, mw_p = scipy_stats.mannwhitneyu(api_errors, oss_errors, alternative="two-sided")
        effect_sizes["api_vs_oss"] = {
            "cohens_d": d,
            "cliffs_delta": delta,
            "mw_pvalue": float(mw_p),
        }
        print(f"  API vs OSS: d={d:.3f}, delta={delta:.3f}, p={mw_p:.2e}")

    results["effect_sizes"] = effect_sizes

    # Save
    output_path = os.path.join(processed_dir, "emobank", "emobank_bootstrap_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n=== Results saved: {output_path} ===")


if __name__ == "__main__":
    main()
