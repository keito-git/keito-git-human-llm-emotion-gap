"""
Phase 3+: Enhanced analysis for EMNLP-level quality.

This script adds:
    Phase 2 - Statistical Reliability:
        1. Bootstrap confidence intervals (1000 iterations) for all key metrics
        2. Effect sizes (Cohen's d, Cliff's delta)
        3. Dunn's post-hoc test after Kruskal-Wallis
    Phase 3 - Advanced Analysis:
        4. Uncertainty pattern clustering (K-Means on LLM distribution features)
        5. Confusion matrix comparison (human vs. LLM confusion patterns)
        6. High-disagreement text linguistic feature analysis

Usage:
    python enhanced_analysis.py --config ../config/experiment_config.yaml

Outputs:
    - data/processed/enhanced_results.json
    - data/processed/bootstrap_ci.json
    - data/processed/clustering_results.parquet
    - paper/en/figures/bootstrap_*.pdf
    - paper/en/figures/confusion_matrix_*.pdf
    - paper/en/figures/clustering_*.pdf
    - paper/en/figures/linguistic_*.pdf
"""

import argparse
import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import (
    spearmanr, pearsonr, mannwhitneyu, kruskal, rankdata,
    norm as scipy_norm,
)
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "utils"))

from metrics import (
    normalize_distribution,
    shannon_entropy,
    jensen_shannon_divergence,
)

# ============================================================
# Configuration
# ============================================================

FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "legend.fontsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.titlesize": 24,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_DISPLAY_NAMES = {
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "llama3.1:8b": "Llama 3.1 8B",
    "llama3.1-8b": "Llama 3.1 8B",
    "qwen3-8b": "Qwen3-8B",
    "roberta-ft": "RoBERTa-FT",
    "finetuned": "RoBERTa-FT",
}

MODEL_COLORS = {
    "gpt-5.4-mini": "#1f77b4",
    "claude-haiku-4-5-20251001": "#ff7f0e",
    "llama3.1:8b": "#2ca02c",
    "llama3.1-8b": "#2ca02c",
    "qwen3-8b": "#d62728",
}

API_MODELS = {"gpt-5.4-mini", "claude-haiku-4-5-20251001"}


def get_display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Phase 2: Bootstrap Confidence Intervals
# ============================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations.
        statistic_func: Function that takes an array and returns a scalar.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level (e.g. 0.95).
        seed: Random seed.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, se.
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    point_estimate = statistic_func(data)

    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic_func(sample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    se = float(boot_stats.std())

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }


def compute_bootstrap_cis(
    metrics_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CIs for JSD, KLD, Wasserstein per model (temp=all)."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    results = {}

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        results[model] = {}

        for metric in ["jsd", "kld", "wasserstein"]:
            values = m_df[metric].dropna().values
            ci = bootstrap_ci(
                values,
                statistic_func=np.mean,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
            results[model][metric] = ci

        # Bootstrap CI for entropy correlation (Spearman rho)
        h_ent = m_df["human_entropy"].values
        l_ent = m_df["llm_entropy"].values
        rng = np.random.RandomState(seed)
        n = len(h_ent)
        rho_point = float(spearmanr(h_ent, l_ent).statistic)
        boot_rhos = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            boot_rhos[i] = spearmanr(h_ent[idx], l_ent[idx]).statistic
        alpha = 1.0 - confidence_level
        results[model]["entropy_spearman_rho"] = {
            "point_estimate": rho_point,
            "ci_lower": float(np.percentile(boot_rhos, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_rhos, 100 * (1 - alpha / 2))),
            "se": float(boot_rhos.std()),
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
        }

        # Per agreement level CIs
        results[model]["by_agreement"] = {}
        for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
            sub = m_df[m_df["agreement_level"] == level]
            if len(sub) > 0:
                ci = bootstrap_ci(
                    sub["jsd"].values,
                    statistic_func=np.mean,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                )
                results[model]["by_agreement"][level] = ci

    return results


# ============================================================
# Phase 2: Effect Sizes
# ============================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta (non-parametric effect size)."""
    n1, n2 = len(group1), len(group2)
    count = 0
    for x in group1:
        for y in group2:
            if x > y:
                count += 1
            elif x < y:
                count -= 1
    return float(count / (n1 * n2))


def cliffs_delta_fast(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta efficiently using broadcasting."""
    diff = group1[:, None] - group2[None, :]
    return float((np.sign(diff).sum()) / diff.size)


def compute_effect_sizes(metrics_df: pd.DataFrame) -> dict:
    """Compute effect sizes for API vs OSS and between agreement levels."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    df["model_type"] = df["model"].apply(lambda m: "API" if m in API_MODELS else "OSS")

    results = {}

    # API vs OSS overall
    api_jsd = df[df["model_type"] == "API"]["jsd"].values
    oss_jsd = df[df["model_type"] == "OSS"]["jsd"].values
    results["api_vs_oss"] = {
        "cohens_d": cohens_d(api_jsd, oss_jsd),
        "cliffs_delta": cliffs_delta_fast(api_jsd, oss_jsd),
        "api_n": len(api_jsd),
        "oss_n": len(oss_jsd),
    }

    # Per-model pairwise effect sizes
    models = sorted(df["model"].unique())
    results["pairwise"] = {}
    for m1, m2 in combinations(models, 2):
        g1 = df[df["model"] == m1]["jsd"].values
        g2 = df[df["model"] == m2]["jsd"].values
        u_stat, u_p = mannwhitneyu(g1, g2, alternative="two-sided")
        results["pairwise"][f"{m1}_vs_{m2}"] = {
            "cohens_d": cohens_d(g1, g2),
            "cliffs_delta": cliffs_delta_fast(g1, g2),
            "mannwhitney_U": float(u_stat),
            "mannwhitney_p": float(u_p),
        }

    # Effect sizes across agreement levels (per model)
    results["agreement_effect_sizes"] = {}
    for model in models:
        m_df = df[df["model"] == model]
        levels = ["full_agreement", "partial_agreement", "full_disagreement"]
        results["agreement_effect_sizes"][model] = {}
        for l1, l2 in combinations(levels, 2):
            g1 = m_df[m_df["agreement_level"] == l1]["jsd"].values
            g2 = m_df[m_df["agreement_level"] == l2]["jsd"].values
            if len(g1) > 0 and len(g2) > 0:
                results["agreement_effect_sizes"][model][f"{l1}_vs_{l2}"] = {
                    "cohens_d": cohens_d(g1, g2),
                    "cliffs_delta": cliffs_delta_fast(g1, g2),
                }

    return results


# ============================================================
# Phase 2: Dunn's Post-hoc Test (manual implementation)
# ============================================================

def dunns_test(groups: list[np.ndarray], p_adjust: str = "bonferroni") -> dict:
    """Perform Dunn's post-hoc test after Kruskal-Wallis.

    Manual implementation to avoid scikit-posthocs dependency.

    Args:
        groups: List of arrays, one per group.
        p_adjust: Correction method ('bonferroni' or 'holm').

    Returns:
        Dict with pairwise comparisons.
    """
    # Combine all data and compute overall ranks
    all_data = np.concatenate(groups)
    ranks = rankdata(all_data)

    # Split ranks back into groups
    group_ranks = []
    offset = 0
    for g in groups:
        group_ranks.append(ranks[offset:offset + len(g)])
        offset += len(g)

    N = len(all_data)
    k = len(groups)

    # Compute mean rank per group
    mean_ranks = [r.mean() for r in group_ranks]
    n_per_group = [len(g) for g in groups]

    # Tie correction factor
    # C = 1 - sum(t^3 - t) / (N^3 - N), where t = number of ties
    _, counts = np.unique(ranks, return_counts=True)
    tie_correction = 1.0 - np.sum(counts**3 - counts) / (N**3 - N)
    if tie_correction == 0:
        tie_correction = 1.0

    results = {}
    n_comparisons = k * (k - 1) // 2
    raw_p_values = []
    comparison_keys = []

    for i, j in combinations(range(k), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        se = np.sqrt(
            (N * (N + 1) / 12 - np.sum(counts**3 - counts) / (12 * (N - 1)))
            * (1.0 / n_per_group[i] + 1.0 / n_per_group[j])
        )
        if se == 0:
            z = 0.0
            p = 1.0
        else:
            z = diff / se
            p = 2.0 * (1.0 - scipy_norm.cdf(abs(z)))

        raw_p_values.append(p)
        comparison_keys.append((i, j))

    # Adjust p-values
    if p_adjust == "bonferroni":
        adjusted_p = [min(p * n_comparisons, 1.0) for p in raw_p_values]
    elif p_adjust == "holm":
        # Holm-Bonferroni
        sorted_indices = np.argsort(raw_p_values)
        adjusted_p = [0.0] * len(raw_p_values)
        for rank_idx, orig_idx in enumerate(sorted_indices):
            adjusted_p[orig_idx] = min(
                raw_p_values[orig_idx] * (n_comparisons - rank_idx), 1.0
            )
        # Enforce monotonicity
        for rank_idx in range(1, len(sorted_indices)):
            curr = sorted_indices[rank_idx]
            prev = sorted_indices[rank_idx - 1]
            adjusted_p[curr] = max(adjusted_p[curr], adjusted_p[prev])
    else:
        adjusted_p = raw_p_values

    for idx, (i, j) in enumerate(comparison_keys):
        results[f"group{i}_vs_group{j}"] = {
            "mean_rank_diff": abs(mean_ranks[i] - mean_ranks[j]),
            "z_statistic": float(diff / se) if se > 0 else 0.0,
            "p_raw": float(raw_p_values[idx]),
            "p_adjusted": float(adjusted_p[idx]),
        }

    return results


def compute_posthoc_tests(metrics_df: pd.DataFrame) -> dict:
    """Run Dunn's test for agreement level comparisons per model."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    results = {}

    levels = ["full_agreement", "partial_agreement", "full_disagreement"]
    level_labels = {0: "full_agreement", 1: "partial_agreement", 2: "full_disagreement"}

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        groups = []
        for level in levels:
            sub = m_df[m_df["agreement_level"] == level]["jsd"].values
            groups.append(sub)

        # First run Kruskal-Wallis
        if all(len(g) > 0 for g in groups):
            h_stat, kw_p = kruskal(*groups)
            dunn_results = dunns_test(groups, p_adjust="bonferroni")

            # Rename keys with actual level names
            renamed = {}
            for key, val in dunn_results.items():
                i = int(key.split("_vs_")[0].replace("group", ""))
                j = int(key.split("_vs_")[1].replace("group", ""))
                new_key = f"{level_labels[i]}_vs_{level_labels[j]}"
                renamed[new_key] = val

            results[model] = {
                "kruskal_wallis_H": float(h_stat),
                "kruskal_wallis_p": float(kw_p),
                "dunns_test": renamed,
            }

    return results


# ============================================================
# Phase 3: Uncertainty Pattern Clustering
# ============================================================

def compute_uncertainty_features(
    metrics_df: pd.DataFrame,
    llm_dist_df: pd.DataFrame,
    emotion_labels: list[str],
) -> pd.DataFrame:
    """Compute features for uncertainty pattern clustering.

    For each text (temp=all), compute:
        - JSD per model
        - LLM entropy per model
        - Human entropy
        - Top-1 LLM probability per model (concentration)
        - Number of emotions with >0.1 probability (diversity)
    """
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    dist_df = llm_dist_df[llm_dist_df["temperature"] == "all"].copy()
    dist_cols = [f"{e}_dist" for e in emotion_labels]

    # Pivot to wide format: one row per text, columns per model
    records = {}
    for _, row in df.iterrows():
        text_id = row["id"]
        model = row["model"]
        if text_id not in records:
            records[text_id] = {
                "id": text_id,
                "human_entropy": row["human_entropy"],
                "agreement_level": row["agreement_level"],
            }
        records[text_id][f"jsd_{model}"] = row["jsd"]
        records[text_id][f"llm_entropy_{model}"] = row["llm_entropy"]

    # Add distribution concentration features
    for _, row in dist_df.iterrows():
        text_id = row["id"]
        model = row["model"]
        if text_id in records:
            dist_vals = np.array([row[c] for c in dist_cols])
            dist_norm = normalize_distribution(dist_vals)
            records[text_id][f"top1_prob_{model}"] = float(dist_norm.max())
            records[text_id][f"n_emotions_above_01_{model}"] = int((dist_norm > 0.1).sum())

    feature_df = pd.DataFrame(list(records.values()))
    return feature_df


def cluster_uncertainty_patterns(
    feature_df: pd.DataFrame,
    n_clusters: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """K-Means clustering on uncertainty features.

    Returns:
        - DataFrame with cluster assignments
        - Dict with cluster statistics
    """
    # Select numeric features for clustering
    feature_cols = [c for c in feature_df.columns
                    if c.startswith(("jsd_", "llm_entropy_", "top1_prob_", "n_emotions_"))
                    or c == "human_entropy"]

    X = feature_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    result_df = feature_df.copy()
    result_df["cluster"] = labels
    result_df["pca_1"] = X_pca[:, 0]
    result_df["pca_2"] = X_pca[:, 1]

    # Cluster statistics
    cluster_stats = {}
    for c in range(n_clusters):
        c_df = result_df[result_df["cluster"] == c]
        stats = {
            "n_texts": len(c_df),
            "human_entropy_mean": float(c_df["human_entropy"].mean()),
            "human_entropy_std": float(c_df["human_entropy"].std()),
            "agreement_distribution": c_df["agreement_level"].value_counts().to_dict(),
        }
        for col in feature_cols:
            if col != "human_entropy":
                stats[f"{col}_mean"] = float(c_df[col].mean())
        cluster_stats[f"cluster_{c}"] = stats

    cluster_stats["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    cluster_stats["feature_columns"] = feature_cols

    return result_df, cluster_stats


def plot_clustering(
    cluster_df: pd.DataFrame,
    fig_dir: str,
    n_clusters: int = 4,
) -> None:
    """Visualize clustering results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: PCA scatter colored by cluster
    palette = sns.color_palette("Set2", n_clusters)
    for c in range(n_clusters):
        sub = cluster_df[cluster_df["cluster"] == c]
        axes[0].scatter(
            sub["pca_1"], sub["pca_2"],
            c=[palette[c]], s=15, alpha=0.6,
            label=f"Cluster {c} (n={len(sub)})",
            edgecolors="none",
        )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=2,
        fontsize=14,
    )

    # Panel 2: PCA scatter colored by agreement level
    ag_palette = {
        "full_agreement": "#66c2a5",
        "partial_agreement": "#fc8d62",
        "full_disagreement": "#8da0cb",
    }
    for level, color in ag_palette.items():
        sub = cluster_df[cluster_df["agreement_level"] == level]
        axes[1].scatter(
            sub["pca_1"], sub["pca_2"],
            c=[color], s=15, alpha=0.6,
            label=level.replace("_", " ").title(),
            edgecolors="none",
        )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=3,
        fontsize=14,
    )

    # Panel 3: Cluster composition by agreement level
    cluster_labels = sorted(cluster_df["cluster"].unique())
    agreement_levels = ["full_agreement", "partial_agreement", "full_disagreement"]
    bottom = np.zeros(len(cluster_labels))
    for level in agreement_levels:
        counts = []
        for c in cluster_labels:
            c_df = cluster_df[cluster_df["cluster"] == c]
            counts.append((c_df["agreement_level"] == level).sum())
        counts = np.array(counts)
        axes[2].bar(
            cluster_labels, counts, bottom=bottom,
            label=level.replace("_", " ").title(),
            color=ag_palette[level],
        )
        bottom += counts
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Count")
    axes[2].legend(
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=3,
        fontsize=14,
    )

    plt.tight_layout()
    path = os.path.join(fig_dir, f"clustering_uncertainty.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Phase 3: Confusion Matrix Comparison
# ============================================================

def build_confusion_matrix(
    df: pd.DataFrame,
    dist_cols: list[str],
    n_emotions: int,
) -> np.ndarray:
    """Build a co-occurrence/confusion matrix from distribution data.

    For each text, find the top-1 and top-2 predicted emotions.
    Matrix[i][j] counts how often emotion i co-occurs with emotion j as top choices.
    """
    matrix = np.zeros((n_emotions, n_emotions))
    for _, row in df.iterrows():
        dist = np.array([row[c] for c in dist_cols])
        dist_norm = normalize_distribution(dist)
        sorted_idx = np.argsort(dist_norm)[::-1]

        # Top-1 is always counted on diagonal
        top1 = sorted_idx[0]
        matrix[top1, top1] += 1.0

        # If second choice has non-trivial probability, count co-occurrence
        if dist_norm[sorted_idx[1]] > 0.05:
            top2 = sorted_idx[1]
            matrix[top1, top2] += 0.5
            matrix[top2, top1] += 0.5

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def compare_confusion_matrices(
    human_matrix: np.ndarray,
    llm_matrices: dict[str, np.ndarray],
) -> dict:
    """Compare human confusion matrix to each LLM's confusion matrix.

    Uses Frobenius norm of difference and correlation.
    """
    results = {}
    # Flatten upper triangle for correlation
    n = human_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=0)
    h_flat = human_matrix[triu_idx]

    for model, llm_matrix in llm_matrices.items():
        l_flat = llm_matrix[triu_idx]
        frobenius = float(np.linalg.norm(human_matrix - llm_matrix, ord="fro"))
        rho, p = spearmanr(h_flat, l_flat)
        pearson_r, pearson_p = pearsonr(h_flat, l_flat)
        results[model] = {
            "frobenius_norm": frobenius,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        }

    # Pairwise LLM comparison
    model_names = sorted(llm_matrices.keys())
    results["llm_pairwise"] = {}
    for m1, m2 in combinations(model_names, 2):
        l1 = llm_matrices[m1][triu_idx]
        l2 = llm_matrices[m2][triu_idx]
        rho, p = spearmanr(l1, l2)
        frob = float(np.linalg.norm(llm_matrices[m1] - llm_matrices[m2], ord="fro"))
        results["llm_pairwise"][f"{m1}_vs_{m2}"] = {
            "frobenius_norm": frob,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        }

    return results


def plot_confusion_comparison(
    human_matrix: np.ndarray,
    llm_matrices: dict[str, np.ndarray],
    emotion_labels: list[str],
    fig_dir: str,
) -> None:
    """Side-by-side confusion matrices: human + each LLM."""
    short_labels = [e[:5] for e in emotion_labels]
    n_panels = 1 + len(llm_matrices)
    models = sorted(llm_matrices.keys())

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6.5))
    if n_panels == 1:
        axes = [axes]

    # Human
    sns.heatmap(
        human_matrix, ax=axes[0], cmap="YlOrRd",
        xticklabels=short_labels, yticklabels=short_labels,
        square=True, linewidths=0.1,
        cbar_kws={"shrink": 0.6, "label": "Normalized co-occurrence"},
        vmin=0, vmax=0.5,
    )
    axes[0].set_title("Human Annotators")
    axes[0].tick_params(axis="both", labelsize=10)

    for i, model in enumerate(models):
        sns.heatmap(
            llm_matrices[model], ax=axes[i + 1], cmap="YlOrRd",
            xticklabels=short_labels, yticklabels=short_labels,
            square=True, linewidths=0.1,
            cbar_kws={"shrink": 0.6, "label": "Normalized co-occurrence"},
            vmin=0, vmax=0.5,
        )
        axes[i + 1].set_title(get_display_name(model))
        axes[i + 1].tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"confusion_matrix_comparison.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # Difference heatmaps (LLM - Human)
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6.5))
    if len(models) == 1:
        axes = [axes]

    for i, model in enumerate(models):
        diff = llm_matrices[model] - human_matrix
        sns.heatmap(
            diff, ax=axes[i], cmap="RdBu_r", center=0,
            xticklabels=short_labels, yticklabels=short_labels,
            square=True, linewidths=0.1,
            cbar_kws={"shrink": 0.6, "label": "Difference"},
            vmin=-0.3, vmax=0.3,
        )
        axes[i].set_title(f"{get_display_name(model)} - Human")
        axes[i].tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"confusion_matrix_diff.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Phase 3: Linguistic Feature Analysis of High-Disagreement Texts
# ============================================================

def compute_linguistic_features(
    metrics_df: pd.DataFrame,
    core_set: pd.DataFrame,
) -> pd.DataFrame:
    """Compute linguistic features for high/low disagreement texts.

    Features: text length, word count, avg word length, punctuation ratio,
              question mark presence, exclamation mark presence.
    """
    df = metrics_df[metrics_df["temperature"] == "all"].copy()

    # Get per-text mean JSD across all models
    text_jsd = df.groupby("id")["jsd"].agg(["mean", "std"]).reset_index()
    text_jsd.columns = ["id", "mean_jsd_across_models", "std_jsd_across_models"]

    # Merge with core set to get text content
    # Try to get text from core_set; if not available, load from metadata
    if "text" in core_set.columns:
        text_source = core_set[["id", "text", "agreement_level"]]
    else:
        # Load metadata which contains text content
        metadata_path = os.path.join(
            str(Path(os.path.dirname(__file__)).parent.parent / "data" / "processed"),
            "goemotions_metadata.parquet",
        )
        if os.path.exists(metadata_path):
            metadata = pd.read_parquet(metadata_path)
            text_source = core_set[["id", "agreement_level"]].merge(
                metadata[["id", "text"]], on="id", how="left"
            )
        else:
            # Fallback: use empty text (linguistic analysis will be limited)
            text_source = core_set[["id", "agreement_level"]].copy()
            text_source["text"] = ""

    merged = text_jsd.merge(text_source, on="id", how="left")

    # Compute features
    features = []
    for _, row in merged.iterrows():
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        words = text.split()

        features.append({
            "id": row["id"],
            "text": text,
            "agreement_level": row["agreement_level"],
            "mean_jsd_across_models": row["mean_jsd_across_models"],
            "std_jsd_across_models": row["std_jsd_across_models"],
            "char_length": len(text),
            "word_count": len(words),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "has_question_mark": 1 if "?" in text else 0,
            "has_exclamation": 1 if "!" in text else 0,
            "n_punctuation": sum(1 for c in text if c in ".,;:!?-\"'()[]"),
            "punct_ratio": sum(1 for c in text if c in ".,;:!?-\"'()[]") / max(len(text), 1),
            "n_uppercase_words": sum(1 for w in words if w.isupper() and len(w) > 1),
            "unique_word_ratio": len(set(w.lower() for w in words)) / max(len(words), 1),
        })

    return pd.DataFrame(features)


def plot_linguistic_analysis(
    ling_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Visualize linguistic features vs. model disagreement."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    features = [
        ("char_length", "Character Length"),
        ("word_count", "Word Count"),
        ("avg_word_length", "Avg Word Length"),
        ("has_question_mark", "Has Question Mark"),
        ("unique_word_ratio", "Unique Word Ratio"),
        ("punct_ratio", "Punctuation Ratio"),
    ]

    for ax, (feat, label) in zip(axes.flat, features):
        if feat in ["has_question_mark", "has_exclamation"]:
            # Binary: boxplot of JSD by feature value
            sns.boxplot(
                data=ling_df, x=feat, y="mean_jsd_across_models",
                ax=ax, palette="Set2", fliersize=1,
            )
            ax.set_xlabel(label)
            ax.set_ylabel("Mean JSD")
        else:
            # Continuous: scatter
            ax.scatter(
                ling_df[feat], ling_df["mean_jsd_across_models"],
                alpha=0.3, s=10, c="#333333", edgecolors="none",
            )
            # Add correlation
            rho, p = spearmanr(ling_df[feat], ling_df["mean_jsd_across_models"])
            ax.set_xlabel(label)
            ax.set_ylabel("Mean JSD")
            ax.set_title(f"$\\rho$ = {rho:.3f}, p = {p:.2e}")

    plt.tight_layout()
    path = os.path.join(fig_dir, f"linguistic_features.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # Feature comparison by agreement level
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    order = ["full_agreement", "partial_agreement", "full_disagreement"]
    labels = ["Full Agr.", "Partial", "Full Dis."]

    for ax, (feat, label) in zip(axes, [
        ("word_count", "Word Count"),
        ("unique_word_ratio", "Unique Word Ratio"),
        ("avg_word_length", "Avg Word Length"),
    ]):
        sns.boxplot(
            data=ling_df, x="agreement_level", y=feat,
            order=order, palette="Set2", ax=ax,
            fliersize=1, linewidth=0.8,
        )
        ax.set_xticklabels(labels)
        ax.set_xlabel("Agreement Level")
        ax.set_ylabel(label)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"linguistic_by_agreement.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Phase 2: Bootstrap CI Visualization
# ============================================================

def plot_bootstrap_cis(
    bootstrap_results: dict,
    fig_dir: str,
) -> None:
    """Forest plot of bootstrap CIs for JSD per model."""
    models = sorted(bootstrap_results.keys())
    display_names = [get_display_name(m) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics = [("jsd", "JSD"), ("kld", "KL Divergence"), ("wasserstein", "Wasserstein")]

    for ax, (metric, label) in zip(axes, metrics):
        y_pos = range(len(models))
        means = []
        ci_lowers = []
        ci_uppers = []

        for model in models:
            r = bootstrap_results[model][metric]
            means.append(r["point_estimate"])
            ci_lowers.append(r["ci_lower"])
            ci_uppers.append(r["ci_upper"])

        means = np.array(means)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        errors = np.array([means - ci_lowers, ci_uppers - means])

        colors = [MODEL_COLORS.get(m, "#333333") for m in models]
        ax.errorbar(
            means, y_pos, xerr=errors,
            fmt="o", capsize=4, capthick=1.5,
            markersize=7, linewidth=1.5,
            color="black",
        )
        for i, (m, c) in enumerate(zip(models, colors)):
            ax.plot(means[i], i, "o", color=c, markersize=8, zorder=5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(display_names)
        ax.set_xlabel(f"Mean {label}")
        ax.set_title(f"{label}\n(95% Bootstrap CI)", fontsize=13)
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(fig_dir, f"bootstrap_ci_forest.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Agreement-level bootstrap CIs
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    levels = ["full_agreement", "partial_agreement", "full_disagreement"]
    level_labels = ["Full Agr.", "Partial", "Full Dis."]
    level_colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

    for ax, model in zip(axes, models):
        if "by_agreement" not in bootstrap_results[model]:
            continue
        y_pos = range(len(levels))
        means = []
        ci_lowers = []
        ci_uppers = []
        for level in levels:
            r = bootstrap_results[model]["by_agreement"].get(level, {})
            means.append(r.get("point_estimate", 0))
            ci_lowers.append(r.get("ci_lower", 0))
            ci_uppers.append(r.get("ci_upper", 0))

        means = np.array(means)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        errors = np.array([means - ci_lowers, ci_uppers - means])

        for i, c in enumerate(level_colors):
            ax.errorbar(
                means[i], i, xerr=[[errors[0][i]], [errors[1][i]]],
                fmt="o", capsize=4, color=c, markersize=8, linewidth=1.5,
            )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(level_labels)
        ax.set_xlabel("Mean JSD")
        ax.set_title(get_display_name(model))
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(fig_dir, f"bootstrap_ci_by_agreement.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Phase 2: Effect Size Visualization
# ============================================================

def plot_effect_sizes(
    effect_sizes: dict,
    fig_dir: str,
) -> None:
    """Heatmap of pairwise Cohen's d and Cliff's delta."""
    pairwise = effect_sizes["pairwise"]
    models = set()
    for key in pairwise:
        parts = key.split("_vs_")
        models.add(parts[0])
        models.add(parts[1])
    models = sorted(models)
    n = len(models)

    # Build matrices
    d_matrix = np.zeros((n, n))
    delta_matrix = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in pairwise else f"{m2}_vs_{m1}"
            if key in pairwise:
                d = pairwise[key]["cohens_d"]
                delta = pairwise[key]["cliffs_delta"]
                # Flip sign if order is reversed
                if key == f"{m2}_vs_{m1}":
                    d = -d
                    delta = -delta
                d_matrix[i, j] = d
                delta_matrix[i, j] = delta

    display_names = [get_display_name(m) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        d_matrix, ax=axes[0], cmap="RdBu_r", center=0,
        xticklabels=display_names, yticklabels=display_names,
        annot=True, fmt=".3f", square=True,
        linewidths=0.5, vmin=-1, vmax=1,
        annot_kws={"size": 10},
    )
    axes[0].set_title("Cohen's d (row - column)", fontsize=14)

    sns.heatmap(
        delta_matrix, ax=axes[1], cmap="RdBu_r", center=0,
        xticklabels=display_names, yticklabels=display_names,
        annot=True, fmt=".3f", square=True,
        linewidths=0.5, vmin=-1, vmax=1,
        annot_kws={"size": 10},
    )
    axes[1].set_title("Cliff's $\\delta$ (row - column)", fontsize=14)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"effect_sizes_heatmap.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Phase 3: Model Agreement Analysis
# ============================================================

def compute_model_agreement(
    metrics_df: pd.DataFrame,
) -> dict:
    """Compute inter-model agreement: for each text, do models agree on divergence?"""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())

    # Pivot: rows=text, columns=model, values=jsd
    pivot = df.pivot_table(index="id", columns="model", values="jsd")

    # Pairwise model correlation
    results = {"model_jsd_correlation": {}}
    for m1, m2 in combinations(models, 2):
        rho, p = spearmanr(pivot[m1].values, pivot[m2].values)
        results["model_jsd_correlation"][f"{m1}_vs_{m2}"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        }

    # For each text, compute std of JSD across models (inter-model disagreement)
    pivot["jsd_std"] = pivot[models].std(axis=1)
    pivot["jsd_mean"] = pivot[models].mean(axis=1)

    results["inter_model_jsd_std"] = {
        "mean": float(pivot["jsd_std"].mean()),
        "std": float(pivot["jsd_std"].std()),
        "median": float(pivot["jsd_std"].median()),
    }

    # Top 10 texts with highest inter-model disagreement
    top_disagreement = pivot.nlargest(10, "jsd_std")
    results["top_inter_model_disagreement_texts"] = top_disagreement.index.tolist()

    return results


def plot_model_agreement(
    metrics_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Pairwise scatter of JSD values between models."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())
    pivot = df.pivot_table(index="id", columns="model", values="jsd")

    n = len(models)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(5 * (n - 1), 5 * (n - 1)))

    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i][j] if n > 2 else axes
            if j > i:
                ax.set_visible(False)
                continue
            m1 = models[j]
            m2 = models[i + 1]
            rho, _ = spearmanr(pivot[m1].values, pivot[m2].values)
            ax.scatter(
                pivot[m1], pivot[m2],
                alpha=0.2, s=8, c="#333333", edgecolors="none",
            )
            max_val = max(pivot[m1].max(), pivot[m2].max())
            ax.plot([0, max_val], [0, max_val], "r--", alpha=0.3)
            ax.set_xlabel(get_display_name(m1))
            ax.set_ylabel(get_display_name(m2))
            ax.set_title(f"$\\rho$={rho:.3f}")

    plt.tight_layout()
    path = os.path.join(fig_dir, f"model_jsd_agreement.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced analysis for EMNLP-level quality")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR.parent / "config" / "experiment_config.yaml"),
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]
    root_dir = config["paths"]["root"]
    emotion_labels = config["data"]["goemotions"]["emotion_labels"]
    fig_dir = os.path.join(root_dir, "paper", "en", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    seed = config["analysis"]["random_seed"]
    n_bootstrap = config["analysis"]["bootstrap_n_iterations"]
    confidence_level = config["analysis"]["confidence_level"]
    np.random.seed(seed)

    dist_cols = [f"{e}_dist" for e in emotion_labels]

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("=" * 60)
    print("Enhanced Analysis: Phase 2 & 3")
    print("=" * 60)

    print("\n--- Loading data ---")
    metrics_df = pd.read_parquet(os.path.join(processed_dir, "per_text_metrics.parquet"))
    print(f"  Per-text metrics: {len(metrics_df)} rows")

    llm_dist_df = pd.read_parquet(os.path.join(processed_dir, "llm_distributions.parquet"))
    print(f"  LLM distributions: {len(llm_dist_df)} rows")

    human_df = pd.read_parquet(os.path.join(processed_dir, "goemotions_annotator_distributions.parquet"))
    core_set = pd.read_parquet(os.path.join(processed_dir, "core_set.parquet"))
    human_core = human_df[human_df["id"].isin(set(core_set["id"].unique()))].copy()
    print(f"  Human core set: {len(human_core)} texts")
    print(f"  Models: {sorted(metrics_df['model'].unique())}")

    # ========================================================
    # Phase 2: Statistical Reliability
    # ========================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Statistical Reliability Enhancement")
    print("=" * 60)

    # --- 2.1 Bootstrap CIs ---
    print(f"\n--- Bootstrap CIs (n={n_bootstrap}, alpha={1-confidence_level}) ---")
    bootstrap_results = compute_bootstrap_cis(
        metrics_df,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    for model, res in bootstrap_results.items():
        jsd_ci = res["jsd"]
        print(f"  {model}: JSD = {jsd_ci['point_estimate']:.4f} "
              f"[{jsd_ci['ci_lower']:.4f}, {jsd_ci['ci_upper']:.4f}]")

    # Save bootstrap results
    bootstrap_path = os.path.join(processed_dir, "bootstrap_ci.json")
    with open(bootstrap_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {bootstrap_path}")

    # --- 2.2 Effect Sizes ---
    print("\n--- Effect Sizes ---")
    effect_sizes = compute_effect_sizes(metrics_df)
    print(f"  API vs OSS Cohen's d: {effect_sizes['api_vs_oss']['cohens_d']:.4f}")
    print(f"  API vs OSS Cliff's delta: {effect_sizes['api_vs_oss']['cliffs_delta']:.4f}")
    for key, val in effect_sizes["pairwise"].items():
        print(f"  {key}: d={val['cohens_d']:.4f}, delta={val['cliffs_delta']:.4f}, "
              f"MW-p={val['mannwhitney_p']:.2e}")

    # --- 2.3 Dunn's Post-hoc Tests ---
    print("\n--- Dunn's Post-hoc Tests ---")
    posthoc_results = compute_posthoc_tests(metrics_df)
    for model, res in posthoc_results.items():
        print(f"  {model}: KW H={res['kruskal_wallis_H']:.2f}, p={res['kruskal_wallis_p']:.2e}")
        for comp, vals in res["dunns_test"].items():
            sig = "***" if vals["p_adjusted"] < 0.001 else "**" if vals["p_adjusted"] < 0.01 else "*" if vals["p_adjusted"] < 0.05 else "ns"
            print(f"    {comp}: p_adj={vals['p_adjusted']:.4f} {sig}")

    # --- Phase 2 Visualizations ---
    print("\n--- Phase 2 Visualizations ---")
    plot_bootstrap_cis(bootstrap_results, fig_dir)
    plot_effect_sizes(effect_sizes, fig_dir)

    # ========================================================
    # Phase 3: Advanced Analysis
    # ========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Advanced Analysis")
    print("=" * 60)

    # --- 3.1 Uncertainty Clustering ---
    print("\n--- Uncertainty Pattern Clustering ---")
    feature_df = compute_uncertainty_features(metrics_df, llm_dist_df, emotion_labels)
    cluster_df, cluster_stats = cluster_uncertainty_patterns(feature_df, n_clusters=4, seed=seed)
    for c_name, c_stat in cluster_stats.items():
        if c_name.startswith("cluster_"):
            print(f"  {c_name}: n={c_stat['n_texts']}, "
                  f"human_entropy={c_stat['human_entropy_mean']:.3f}")

    cluster_path = os.path.join(processed_dir, "clustering_results.parquet")
    cluster_df.to_parquet(cluster_path, index=False)
    print(f"  Saved: {cluster_path}")

    plot_clustering(cluster_df, fig_dir)

    # --- 3.2 Confusion Matrix Comparison ---
    print("\n--- Confusion Matrix Comparison ---")
    # Build human confusion matrix from distributions
    human_confusion = build_confusion_matrix(human_core, dist_cols, len(emotion_labels))
    print(f"  Human confusion matrix: {human_confusion.shape}")

    # Build LLM confusion matrices
    llm_confusion = {}
    for model in sorted(llm_dist_df["model"].unique()):
        llm_all = llm_dist_df[
            (llm_dist_df["model"] == model) & (llm_dist_df["temperature"] == "all")
        ]
        # Filter to core set texts
        llm_core = llm_all[llm_all["id"].isin(set(core_set["id"].unique()))]
        llm_confusion[model] = build_confusion_matrix(llm_core, dist_cols, len(emotion_labels))
        print(f"  {model} confusion matrix: {llm_confusion[model].shape}")

    confusion_comparison = compare_confusion_matrices(human_confusion, llm_confusion)
    for model, stats in confusion_comparison.items():
        if model != "llm_pairwise":
            print(f"  {model} vs Human: Frobenius={stats['frobenius_norm']:.4f}, "
                  f"rho={stats['spearman_rho']:.4f}")

    plot_confusion_comparison(human_confusion, llm_confusion, emotion_labels, fig_dir)

    # --- 3.3 Linguistic Feature Analysis ---
    print("\n--- Linguistic Feature Analysis ---")
    ling_df = compute_linguistic_features(metrics_df, core_set)
    print(f"  Linguistic features computed for {len(ling_df)} texts")

    # Correlations
    for feat in ["char_length", "word_count", "avg_word_length", "unique_word_ratio", "punct_ratio"]:
        rho, p = spearmanr(ling_df[feat], ling_df["mean_jsd_across_models"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {feat:25s}: rho={rho:+.4f}, p={p:.2e} {sig}")

    # Question mark effect
    q_yes = ling_df[ling_df["has_question_mark"] == 1]["mean_jsd_across_models"]
    q_no = ling_df[ling_df["has_question_mark"] == 0]["mean_jsd_across_models"]
    if len(q_yes) > 0 and len(q_no) > 0:
        u, p = mannwhitneyu(q_yes, q_no, alternative="two-sided")
        print(f"  Question mark effect: has_?={q_yes.mean():.4f}, no_?={q_no.mean():.4f}, MW p={p:.2e}")

    plot_linguistic_analysis(ling_df, fig_dir)

    # --- 3.4 Inter-Model Agreement ---
    print("\n--- Inter-Model Agreement Analysis ---")
    model_agreement = compute_model_agreement(metrics_df)
    print(f"  Inter-model JSD std: mean={model_agreement['inter_model_jsd_std']['mean']:.4f}")
    for key, val in model_agreement["model_jsd_correlation"].items():
        print(f"  {key}: rho={val['spearman_rho']:.4f}")

    plot_model_agreement(metrics_df, fig_dir)

    # ========================================================
    # Save all results
    # ========================================================
    print("\n--- Saving enhanced results ---")
    enhanced_results = {
        "bootstrap_ci": bootstrap_results,
        "effect_sizes": effect_sizes,
        "posthoc_tests": posthoc_results,
        "cluster_stats": cluster_stats,
        "confusion_comparison": confusion_comparison,
        "model_agreement": model_agreement,
        "linguistic_correlations": {},
    }

    for feat in ["char_length", "word_count", "avg_word_length", "unique_word_ratio", "punct_ratio"]:
        rho, p = spearmanr(ling_df[feat], ling_df["mean_jsd_across_models"])
        enhanced_results["linguistic_correlations"][feat] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        }

    results_path = os.path.join(processed_dir, "enhanced_results.json")
    with open(results_path, "w") as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 60)
    print("Enhanced Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
