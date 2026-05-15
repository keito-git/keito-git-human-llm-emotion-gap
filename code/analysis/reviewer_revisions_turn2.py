"""
Reviewer Turn 2 revisions:
  1. Paired Wilcoxon signed-rank test for isotonic calibration significance
  2. Sensitivity analysis for lexical transparency correlation (exclude n < 50)

Usage:
    python3 reviewer_revisions_turn2.py \
        --data_dir /path/to/data/processed/

Outputs:
    - calibration_wilcoxon_results.json
    - lexical_sensitivity_analysis.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon, spearmanr, norm
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression


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

RANDOM_SEED = 42


def normalize_distribution(dist, eps=1e-12):
    """Normalize array to sum to 1."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.maximum(dist, 0)
    total = dist.sum()
    if total == 0:
        return np.ones_like(dist) / len(dist)
    return dist / total


def compute_jsd(p, q):
    """JSD (squared, [0, 1])."""
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return float(jensenshannon(p, q, base=2) ** 2)


# ============================================================
# Part 1: Wilcoxon signed-rank test for isotonic calibration
# ============================================================

def run_wilcoxon_test(data_dir):
    """Run paired Wilcoxon signed-rank test comparing uncalibrated vs isotonic per-text JSD."""
    print("=" * 60)
    print("Part 1: Wilcoxon signed-rank test for isotonic calibration")
    print("=" * 60)

    dist_cols = [f"{e}_dist" for e in EMOTION_LABELS]

    human_df = pd.read_parquet(os.path.join(data_dir, "core_set.parquet"))
    llm_df = pd.read_parquet(os.path.join(data_dir, "llm_distributions.parquet"))

    if "temperature" in llm_df.columns:
        llm_agg = llm_df[llm_df["temperature"] == "all"]
    else:
        llm_agg = llm_df

    results = {}

    for model in llm_agg["model"].unique():
        print(f"\n--- {model} ---")
        model_df = llm_agg[llm_agg["model"] == model]
        merged = model_df.merge(human_df[["id"]], on="id")
        merged_human = human_df[human_df["id"].isin(merged["id"])].sort_values("id")
        merged_llm = merged.sort_values("id")

        h_dists = merged_human[dist_cols].values
        l_dists = merged_llm[dist_cols].values.copy()
        for i in range(len(l_dists)):
            l_dists[i] = normalize_distribution(l_dists[i])

        # 5-fold CV with per-text JSD collection
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        all_uncal_jsds = []
        all_iso_jsds = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(h_dists)):
            h_train, h_test = h_dists[train_idx], h_dists[test_idx]
            l_train, l_test = l_dists[train_idx], l_dists[test_idx]

            # Fit isotonic regression on train fold
            n_cats = h_train.shape[1]
            iso_models = []
            for k in range(n_cats):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                ir.fit(l_train[:, k], h_train[:, k])
                iso_models.append(ir)

            # Apply isotonic to test fold
            cal_test = np.zeros_like(l_test)
            for k, ir_model in enumerate(iso_models):
                cal_test[:, k] = ir_model.predict(l_test[:, k])
            cal_test = np.maximum(cal_test, 0)
            row_sums = cal_test.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            cal_test = cal_test / row_sums

            # Per-text JSD (paired: uncalibrated vs isotonic for same text)
            for i in range(len(h_test)):
                uncal_jsd = compute_jsd(h_test[i], l_test[i])
                cal_jsd = compute_jsd(h_test[i], cal_test[i])
                all_uncal_jsds.append(uncal_jsd)
                all_iso_jsds.append(cal_jsd)

        all_uncal_jsds = np.array(all_uncal_jsds)
        all_iso_jsds = np.array(all_iso_jsds)
        diffs = all_uncal_jsds - all_iso_jsds  # positive = improvement

        # Paired Wilcoxon signed-rank test
        stat, p_value = wilcoxon(all_uncal_jsds, all_iso_jsds, alternative="two-sided")

        # Effect size: r = Z / sqrt(N)
        n_pairs = len(all_uncal_jsds)
        # Approximate Z from W statistic
        mean_w = n_pairs * (n_pairs + 1) / 4
        std_w = np.sqrt(n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24)
        z_val = (stat - mean_w) / std_w
        effect_size_r = abs(z_val) / np.sqrt(n_pairs)

        median_improvement = float(np.median(diffs))
        mean_improvement = float(np.mean(diffs))
        pct_improved = float((diffs > 0).sum() / len(diffs) * 100)

        print(f"  N pairs: {n_pairs}")
        print(f"  Uncalibrated JSD mean: {all_uncal_jsds.mean():.4f}")
        print(f"  Isotonic JSD mean: {all_iso_jsds.mean():.4f}")
        print(f"  Mean improvement: {mean_improvement:.4f}")
        print(f"  Median improvement: {median_improvement:.4f}")
        print(f"  % texts improved: {pct_improved:.1f}%")
        print(f"  Wilcoxon W stat: {stat:.1f}")
        print(f"  p-value: {p_value:.2e}")
        print(f"  Effect size r: {effect_size_r:.3f}")

        results[model] = {
            "n_pairs": int(n_pairs),
            "uncal_jsd_mean": float(all_uncal_jsds.mean()),
            "iso_jsd_mean": float(all_iso_jsds.mean()),
            "mean_improvement": float(mean_improvement),
            "median_improvement": float(median_improvement),
            "pct_improved": pct_improved,
            "wilcoxon_stat": float(stat),
            "p_value": float(p_value),
            "z_value": float(z_val),
            "effect_size_r": float(effect_size_r),
        }

    # Save
    out_path = os.path.join(data_dir, "calibration_wilcoxon_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPart 1 results saved to: {out_path}")

    return results


# ============================================================
# Part 2: Sensitivity analysis for lexical transparency
# ============================================================

def run_sensitivity_analysis(data_dir):
    """Run sensitivity analysis excluding low-frequency categories (n < 50)."""
    print("\n" + "=" * 60)
    print("Part 2: Sensitivity analysis for lexical transparency (n >= 50)")
    print("=" * 60)

    with open(os.path.join(data_dir, "lexical_transparency_scores.json")) as f:
        transparency_scores = json.load(f)

    cat_df = pd.read_parquet(os.path.join(data_dir, "per_category_divergence.parquet"))
    cat_all = cat_df[cat_df["temperature"] == "all"]

    # Identify categories with n < 50
    small_categories = [
        e for e in EMOTION_LABELS
        if transparency_scores[e]["n_positive_texts"] < 50
    ]
    print(f"Categories with n < 50: {small_categories}")
    for e in small_categories:
        print(f"  {e}: n={transparency_scores[e]['n_positive_texts']}")

    # Build per-model rho dict
    per_cat_rho = {}
    for model in cat_all["model"].unique():
        model_data = cat_all[cat_all["model"] == model]
        per_cat_rho[model] = {}
        for _, row in model_data.iterrows():
            per_cat_rho[model][row["emotion"]] = row["spearman_r"]

    # Full analysis (all 28 categories)
    combined_scores_full = []
    avg_rhos_full = []
    for e in EMOTION_LABELS:
        cs = transparency_scores[e].get("combined_score")
        if cs is None or np.isnan(cs):
            continue
        model_rhos = []
        for m in per_cat_rho:
            if e in per_cat_rho[m]:
                model_rhos.append(per_cat_rho[m][e])
        if not model_rhos:
            continue
        combined_scores_full.append(cs)
        avg_rhos_full.append(np.mean(model_rhos))

    r_full, p_full = spearmanr(combined_scores_full, avg_rhos_full)
    print(f"\nFull analysis (n={len(combined_scores_full)}): r_s = {r_full:.3f}, p = {p_full:.4f}")

    # Sensitivity: exclude n < 50
    combined_scores_filt = []
    avg_rhos_filt = []
    excluded_set = set(small_categories)
    for e in EMOTION_LABELS:
        if e in excluded_set:
            continue
        cs = transparency_scores[e].get("combined_score")
        if cs is None or np.isnan(cs):
            continue
        model_rhos = []
        for m in per_cat_rho:
            if e in per_cat_rho[m]:
                model_rhos.append(per_cat_rho[m][e])
        if not model_rhos:
            continue
        combined_scores_filt.append(cs)
        avg_rhos_filt.append(np.mean(model_rhos))

    r_filt, p_filt = spearmanr(combined_scores_filt, avg_rhos_filt)
    print(f"Filtered (n={len(combined_scores_filt)}, excluding n<50): r_s = {r_filt:.3f}, p = {p_filt:.4f}")

    results = {
        "full": {
            "n_categories": len(combined_scores_full),
            "spearman_r": float(r_full),
            "p_value": float(p_full),
        },
        "filtered_n_ge_50": {
            "n_categories": len(combined_scores_filt),
            "excluded_categories": small_categories,
            "excluded_n": {e: transparency_scores[e]["n_positive_texts"] for e in small_categories},
            "spearman_r": float(r_filt),
            "p_value": float(p_filt),
        },
    }

    out_path = os.path.join(data_dir, "lexical_sensitivity_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPart 2 results saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to data/processed/ directory",
    )
    args = parser.parse_args()

    wilcoxon_results = run_wilcoxon_test(args.data_dir)
    sensitivity_results = run_sensitivity_analysis(args.data_dir)

    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print("\nWilcoxon signed-rank test (isotonic vs uncalibrated):")
    for model, r in wilcoxon_results.items():
        print(
            f"  {model}: W={r['wilcoxon_stat']:.0f}, "
            f"p={r['p_value']:.2e}, r={r['effect_size_r']:.3f}, "
            f"improved={r['pct_improved']:.1f}%"
        )

    sf = sensitivity_results["full"]
    sn = sensitivity_results["filtered_n_ge_50"]
    print(
        f"\nLexical transparency sensitivity:"
        f"\n  Full (n={sf['n_categories']}): r_s={sf['spearman_r']:.3f}, p={sf['p_value']:.4f}"
        f"\n  Filtered n>=50 (n={sn['n_categories']}): r_s={sn['spearman_r']:.3f}, p={sn['p_value']:.4f}"
    )


if __name__ == "__main__":
    main()
