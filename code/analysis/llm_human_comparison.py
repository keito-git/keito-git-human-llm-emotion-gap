"""
Phase 3: Comprehensive comparison of human and LLM emotion distributions.

This script performs:
    1. LLM distribution construction (aggregate across samples per text/temperature)
    2. Human vs. LLM distributional comparison (JSD, KLD, Wasserstein)
    3. Agreement-level stratified analysis (full/partial/full disagreement)
    4. Per-emotion category divergence analysis
    5. Human uncertainty vs. LLM uncertainty correlation
    6. Visualization: heatmaps, distribution comparison plots, scatter plots

Usage:
    python llm_human_comparison.py --config ../config/experiment_config.yaml

Outputs:
    - data/processed/llm_human_comparison_results.json
    - data/processed/per_text_metrics.parquet
    - paper/en/figures/*.pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal
from scipy.spatial.distance import jensenshannon

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "utils"))

from metrics import (
    normalize_distribution,
    shannon_entropy,
    kl_divergence,
    jensen_shannon_divergence,
    wasserstein_dist,
    batch_jsd,
)


# ============================================================
# Configuration & Constants
# ============================================================

FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"

# Model display name mapping for consistent figure labels
MODEL_DISPLAY_NAMES = {
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "llama3.1:8b": "Llama 3.1 8B",
    "llama3.1-8b": "Llama 3.1 8B",
    "qwen3-8b": "Qwen3-8B",
    "roberta-ft": "RoBERTa-FT",
    "finetuned": "RoBERTa-FT",
}


def get_display_name(model: str) -> str:
    """Return paper-consistent display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model, model)


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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 1. Build LLM Emotion Distributions
# ============================================================

def build_llm_distributions(
    llm_df: pd.DataFrame,
    emotion_labels: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Aggregate LLM predictions into emotion distributions per text.

    For each text, across all samples at all temperatures, compute:
      - Per-temperature distribution: fraction of samples selecting each emotion
      - Overall distribution: fraction across all temperatures/samples

    Returns DataFrame with columns:
        id, model, temperature, {emotion}_dist for each emotion, entropy
    """
    pred_cols = [f"{e}_pred" for e in emotion_labels]

    records = []

    # Per-temperature distributions
    for temp, temp_group in llm_df.groupby("temperature"):
        for text_id, text_group in temp_group.groupby("id"):
            preds = text_group[pred_cols].values  # (n_samples, n_emotions)
            dist = preds.mean(axis=0)  # fraction of samples selecting each emotion
            dist_norm = normalize_distribution(dist)
            ent = shannon_entropy(dist_norm)

            record = {
                "id": text_id,
                "model": model_name,
                "temperature": str(temp),
                "n_samples": len(text_group),
                "entropy": ent,
            }
            for i, emotion in enumerate(emotion_labels):
                record[f"{emotion}_dist"] = dist[i]
            records.append(record)

    # Overall distribution (all temperatures combined)
    for text_id, text_group in llm_df.groupby("id"):
        preds = text_group[pred_cols].values
        dist = preds.mean(axis=0)
        dist_norm = normalize_distribution(dist)
        ent = shannon_entropy(dist_norm)

        record = {
            "id": text_id,
            "model": model_name,
            "temperature": "all",
            "n_samples": len(text_group),
            "entropy": ent,
        }
        for i, emotion in enumerate(emotion_labels):
            record[f"{emotion}_dist"] = dist[i]
        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 2. Compute Per-Text Comparison Metrics
# ============================================================

def compute_per_text_metrics(
    human_df: pd.DataFrame,
    llm_dist_df: pd.DataFrame,
    emotion_labels: list[str],
) -> pd.DataFrame:
    """Compute JSD, KLD, Wasserstein between human and LLM distributions per text.

    Returns DataFrame with one row per (text, model, temperature).
    """
    dist_cols_human = [f"{e}_dist" for e in emotion_labels]
    dist_cols_llm = [f"{e}_dist" for e in emotion_labels]

    # Merge human distributions
    merged = llm_dist_df.merge(
        human_df[["id"] + dist_cols_human + ["agreement_level", "entropy"]],
        on="id",
        suffixes=("_llm", "_human"),
    )

    results = []
    for idx, row in merged.iterrows():
        h_dist = np.array([row[f"{e}_dist_human"] for e in emotion_labels])
        l_dist = np.array([row[f"{e}_dist_llm"] for e in emotion_labels])

        jsd = jensen_shannon_divergence(h_dist, l_dist)
        kld = kl_divergence(h_dist, l_dist)
        wass = wasserstein_dist(h_dist, l_dist)

        results.append({
            "id": row["id"],
            "model": row["model"],
            "temperature": row["temperature"],
            "jsd": jsd,
            "kld": kld,
            "wasserstein": wass,
            "human_entropy": row["entropy_human"],
            "llm_entropy": row["entropy_llm"],
            "agreement_level": row["agreement_level"],
            "n_samples": row["n_samples"],
        })

    return pd.DataFrame(results)


# ============================================================
# 3. Aggregate Statistics
# ============================================================

def compute_aggregate_stats(metrics_df: pd.DataFrame) -> dict:
    """Compute summary statistics for each model/temperature combination."""
    stats = {}

    for (model, temp), group in metrics_df.groupby(["model", "temperature"]):
        key = f"{model}_temp{temp}"
        stats[key] = {
            "model": model,
            "temperature": str(temp),
            "n_texts": len(group),
            "jsd": {
                "mean": float(group["jsd"].mean()),
                "std": float(group["jsd"].std()),
                "median": float(group["jsd"].median()),
                "q25": float(group["jsd"].quantile(0.25)),
                "q75": float(group["jsd"].quantile(0.75)),
            },
            "kld": {
                "mean": float(group["kld"].mean()),
                "std": float(group["kld"].std()),
                "median": float(group["kld"].median()),
            },
            "wasserstein": {
                "mean": float(group["wasserstein"].mean()),
                "std": float(group["wasserstein"].std()),
                "median": float(group["wasserstein"].median()),
            },
            "entropy_correlation": {
                "spearman_r": float(spearmanr(group["human_entropy"], group["llm_entropy"]).statistic),
                "spearman_p": float(spearmanr(group["human_entropy"], group["llm_entropy"]).pvalue),
                "pearson_r": float(pearsonr(group["human_entropy"], group["llm_entropy"]).statistic),
                "pearson_p": float(pearsonr(group["human_entropy"], group["llm_entropy"]).pvalue),
            },
        }

    return stats


def compute_agreement_level_stats(metrics_df: pd.DataFrame) -> dict:
    """Compute metrics stratified by agreement level."""
    stats = {}

    for (model, temp), model_group in metrics_df.groupby(["model", "temperature"]):
        key = f"{model}_temp{temp}"
        stats[key] = {}

        for level, level_group in model_group.groupby("agreement_level"):
            stats[key][level] = {
                "n_texts": len(level_group),
                "jsd_mean": float(level_group["jsd"].mean()),
                "jsd_std": float(level_group["jsd"].std()),
                "jsd_median": float(level_group["jsd"].median()),
                "kld_mean": float(level_group["kld"].mean()),
                "wasserstein_mean": float(level_group["wasserstein"].mean()),
                "human_entropy_mean": float(level_group["human_entropy"].mean()),
                "llm_entropy_mean": float(level_group["llm_entropy"].mean()),
            }

        # Kruskal-Wallis test across agreement levels
        groups = [
            g["jsd"].values
            for _, g in model_group.groupby("agreement_level")
        ]
        if len(groups) >= 2:
            h_stat, p_val = kruskal(*groups)
            stats[key]["kruskal_wallis"] = {
                "H": float(h_stat),
                "p": float(p_val),
            }

    return stats


# ============================================================
# 4. Per-Category Divergence Analysis
# ============================================================

def compute_per_category_divergence(
    human_df: pd.DataFrame,
    llm_dist_df: pd.DataFrame,
    emotion_labels: list[str],
) -> pd.DataFrame:
    """For each emotion category, compute mean human rate vs. LLM rate and divergence."""
    results = []

    for model in llm_dist_df["model"].unique():
        for temp in llm_dist_df["temperature"].unique():
            llm_sub = llm_dist_df[
                (llm_dist_df["model"] == model) & (llm_dist_df["temperature"] == temp)
            ]

            merged = llm_sub.merge(human_df[["id"] + [f"{e}_dist" for e in emotion_labels]],
                                   on="id", suffixes=("_llm", "_human"))

            for emotion in emotion_labels:
                h_rates = merged[f"{emotion}_dist_human"].values
                l_rates = merged[f"{emotion}_dist_llm"].values

                # Mean rate difference
                h_mean = h_rates.mean()
                l_mean = l_rates.mean()

                # Per-text absolute difference
                abs_diff = np.abs(h_rates - l_rates).mean()

                # Correlation of per-text rates
                if h_rates.std() > 0 and l_rates.std() > 0:
                    corr, p_val = spearmanr(h_rates, l_rates)
                else:
                    corr, p_val = 0.0, 1.0

                results.append({
                    "model": model,
                    "temperature": str(temp),
                    "emotion": emotion,
                    "human_mean_rate": float(h_mean),
                    "llm_mean_rate": float(l_mean),
                    "rate_diff": float(l_mean - h_mean),
                    "abs_rate_diff": float(abs_diff),
                    "spearman_r": float(corr),
                    "spearman_p": float(p_val),
                })

    return pd.DataFrame(results)


# ============================================================
# 5. Visualization Functions
# ============================================================

def plot_jsd_by_agreement(
    metrics_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Box plot of JSD by agreement level, faceted by model."""
    # Filter to 'all' temperature aggregation
    df = metrics_df[metrics_df["temperature"] == "all"].copy()

    order = ["full_agreement", "partial_agreement", "full_disagreement"]
    labels = ["Full\nAgreement", "Partial\nAgreement", "Full\nDisagreement"]

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", 3)

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model]
        sns.boxplot(
            data=model_df,
            x="agreement_level",
            y="jsd",
            order=order,
            palette=palette,
            ax=ax,
            fliersize=2,
            linewidth=0.8,
        )
        ax.set_xticklabels(labels)
        ax.set_xlabel("Human Agreement Level")
        ax.set_ylabel("JSD (Human vs. LLM)")
        ax.set_title(get_display_name(model))

    plt.tight_layout()
    path = os.path.join(fig_dir, f"jsd_by_agreement.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_entropy_correlation(
    metrics_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Scatter plot of human entropy vs LLM entropy."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    palette = {"full_agreement": "#66c2a5", "partial_agreement": "#fc8d62", "full_disagreement": "#8da0cb"}

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model]
        for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
            sub = model_df[model_df["agreement_level"] == level]
            ax.scatter(
                sub["human_entropy"],
                sub["llm_entropy"],
                alpha=0.4,
                s=12,
                color=palette[level],
                label=level.replace("_", " ").title(),
                edgecolors="none",
            )

        # Add diagonal
        max_val = max(model_df["human_entropy"].max(), model_df["llm_entropy"].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=0.8)

        rho, p = spearmanr(model_df["human_entropy"], model_df["llm_entropy"])
        ax.set_xlabel("Human Entropy (bits)")
        ax.set_ylabel("LLM Entropy (bits)")
        ax.set_title(get_display_name(model))
        # Panel annotation: statistical summary (kept as in-plot text)
        ax.annotate(
            f"$\\rho$ = {rho:.3f}, p < {max(p, 1e-300):.1e}",
            xy=(0.05, 0.93), xycoords="axes fraction",
            fontsize=16, ha="left", va="top",
        )
        ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"entropy_correlation.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_category_heatmap(
    cat_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Heatmap of per-category rate difference (LLM - Human) for each model."""
    df = cat_df[cat_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 10))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model].sort_values("emotion")
        pivot = model_df.set_index("emotion")["rate_diff"]
        pivot_sorted = pivot.sort_values()

        colors = ["#d73027" if v < 0 else "#4575b4" for v in pivot_sorted.values]
        ax.barh(pivot_sorted.index, pivot_sorted.values, color=colors, height=0.7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Rate Difference (LLM - Human)")
        ax.set_title(get_display_name(model))

    plt.tight_layout()
    path = os.path.join(fig_dir, f"category_rate_diff.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_category_correlation_heatmap(
    cat_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Heatmap of per-category Spearman correlation between human and LLM rates."""
    df = cat_df[cat_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 10))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model].sort_values("emotion")
        pivot = model_df.set_index("emotion")["spearman_r"]
        pivot_sorted = pivot.sort_values(ascending=True)

        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=-0.2, vmax=1.0)
        colors = [cmap(norm(v)) for v in pivot_sorted.values]

        ax.barh(pivot_sorted.index, pivot_sorted.values, color=colors, height=0.7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Spearman $\\rho$ (Human rate vs. LLM rate)")
        ax.set_title(get_display_name(model))
        ax.set_xlim(-0.3, 1.0)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"category_spearman.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_temperature_effect(
    metrics_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Line plot of JSD across temperatures for each model."""
    df = metrics_df[metrics_df["temperature"].astype(str) != "all"].copy()
    df["temperature"] = df["temperature"].astype(float)

    models = sorted(df["model"].unique())
    palette = sns.color_palette("tab10", len(models))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        summary = model_df.groupby("temperature")["jsd"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["temperature"],
            summary["mean"],
            yerr=summary["std"],
            label=get_display_name(model),
            color=palette[i],
            marker="o",
            capsize=3,
            linewidth=1.5,
        )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean JSD (Human vs. LLM)")
    ax.legend(
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=2,
        fontsize=14,
    )
    ax.set_xticks([0.0, 0.3, 0.7, 1.0])

    plt.tight_layout()
    path = os.path.join(fig_dir, f"temperature_effect.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_overall_distribution_comparison(
    human_df: pd.DataFrame,
    llm_dist_dfs: dict,
    emotion_labels: list[str],
    fig_dir: str,
) -> None:
    """Bar chart comparing overall emotion frequency: human vs each LLM."""
    dist_cols = [f"{e}_dist" for e in emotion_labels]

    # Human overall marginal distribution
    h_marginal = human_df[dist_cols].mean().values

    models = sorted(llm_dist_dfs.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(emotion_labels))
    width = 0.8 / (n_models + 1)

    # Human bars
    ax.bar(x - width * n_models / 2, h_marginal, width, label="Human", color="#333333", alpha=0.8)

    palette = sns.color_palette("Set2", n_models)
    for i, model in enumerate(models):
        llm_df = llm_dist_dfs[model]
        llm_all = llm_df[llm_df["temperature"] == "all"]
        l_marginal = llm_all[dist_cols].mean().values
        offset = width * (i + 1) - width * n_models / 2
        ax.bar(x + offset, l_marginal, width, label=get_display_name(model), color=palette[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(emotion_labels, rotation=60, ha="right", fontsize=16)
    ax.set_ylabel("Mean Rate")
    ax.legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(fig_dir, f"overall_distribution_comparison.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_jsd_distribution(
    metrics_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Histogram of JSD values across texts for each model (all temperatures)."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Set2", len(models))

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        ax.hist(
            model_df["jsd"],
            bins=50,
            alpha=0.5,
            label=f"{get_display_name(model)} (mean={model_df['jsd'].mean():.3f})",
            color=palette[i],
            density=True,
        )

    ax.set_xlabel("Jensen-Shannon Divergence")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(fig_dir, f"jsd_distribution.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_api_vs_oss_comparison(
    metrics_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Compare API-based vs OSS models across key metrics."""
    api_models = {"gpt-5.4-mini", "claude-haiku-4-5-20251001"}
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    df["model_type"] = df["model"].apply(lambda m: "API" if m in api_models else "OSS")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: JSD by model type
    sns.boxplot(data=df, x="model_type", y="jsd", ax=axes[0],
                palette={"API": "#4575b4", "OSS": "#d73027"},
                fliersize=1, linewidth=0.8)
    axes[0].set_xlabel("Model Type")
    axes[0].set_ylabel("JSD (Human vs. LLM)")

    # Panel 2: JSD by model type and agreement level
    order = ["full_agreement", "partial_agreement", "full_disagreement"]
    labels_map = {"full_agreement": "Full Agr.", "partial_agreement": "Partial", "full_disagreement": "Full Dis."}
    df["agreement_short"] = df["agreement_level"].map(labels_map)
    sns.boxplot(data=df, x="agreement_short", y="jsd", hue="model_type",
                order=["Full Agr.", "Partial", "Full Dis."],
                palette={"API": "#4575b4", "OSS": "#d73027"},
                ax=axes[1], fliersize=1, linewidth=0.8)
    axes[1].set_xlabel("Agreement Level")
    axes[1].set_ylabel("JSD")
    # Move legend outside (bottom) for Panel 2
    axes[1].legend(
        title="Type",
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=2,
        fontsize=14,
    )

    # Panel 3: Entropy correlation by model
    models = sorted(df["model"].unique())
    palette_model = sns.color_palette("Set2", len(models))
    for i, model in enumerate(models):
        m_df = df[df["model"] == model]
        rho = spearmanr(m_df["human_entropy"], m_df["llm_entropy"]).statistic
        axes[2].bar(i, rho, color=palette_model[i], label=get_display_name(model))
    axes[2].set_ylabel("Spearman $\\rho$")
    axes[2].set_xticks(range(len(models)))
    axes[2].set_xticklabels([get_display_name(m) for m in models], rotation=30, ha="right")
    # Move legend outside (bottom) for Panel 3
    axes[2].legend(
        bbox_to_anchor=(0.5, -0.35),
        loc="upper center",
        ncol=2,
        fontsize=14,
    )

    plt.tight_layout()
    path = os.path.join(fig_dir, f"api_vs_oss_comparison.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_ranking_summary(
    metrics_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    fig_dir: str,
) -> None:
    """Radar-style summary of model performance across metrics."""
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    models = sorted(df["model"].unique())

    summary_data = []
    for model in models:
        m_df = df[df["model"] == model]
        rho = spearmanr(m_df["human_entropy"], m_df["llm_entropy"]).statistic

        cat_all = cat_df[(cat_df["model"] == model) & (cat_df["temperature"] == "all")]
        mean_abs_diff = cat_all["abs_rate_diff"].mean()
        mean_spearman = cat_all["spearman_r"].mean()

        summary_data.append({
            "model": model,
            "jsd_mean": m_df["jsd"].mean(),
            "wasserstein_mean": m_df["wasserstein"].mean(),
            "entropy_rho": rho,
            "mean_cat_corr": mean_spearman,
            "mean_abs_rate_diff": mean_abs_diff,
        })

    summary_df = pd.DataFrame(summary_data)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    palette = sns.color_palette("Set2", len(models))
    short_names = [get_display_name(m) for m in models]

    metrics_to_plot = [
        ("jsd_mean", "Mean JSD (lower=better)", True),
        ("entropy_rho", "Entropy Corr. $\\rho$ (higher=better)", False),
        ("mean_cat_corr", "Mean Category Corr. (higher=better)", False),
        ("mean_abs_rate_diff", "Mean |Rate Diff| (lower=better)", True),
    ]

    for ax, (col, title, lower_better) in zip(axes, metrics_to_plot):
        vals = summary_df[col].values
        # Draw all bars with no edge color (edgecolor='none') to avoid unintended outlines
        bars = ax.bar(range(len(models)), vals, color=palette, edgecolor="none")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.set_title(title, fontsize=14)
        # Highlight best with black edge only (no fill change)
        best_idx = vals.argmin() if lower_better else vals.argmax()
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"model_ranking_summary.{FIGURE_FORMAT}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def compute_api_vs_oss_stats(metrics_df: pd.DataFrame) -> dict:
    """Compute statistical comparison between API and OSS models."""
    api_models = {"gpt-5.4-mini", "claude-haiku-4-5-20251001"}
    df = metrics_df[metrics_df["temperature"] == "all"].copy()
    df["model_type"] = df["model"].apply(lambda m: "API" if m in api_models else "OSS")

    api_jsd = df[df["model_type"] == "API"]["jsd"].values
    oss_jsd = df[df["model_type"] == "OSS"]["jsd"].values

    u_stat, u_p = mannwhitneyu(api_jsd, oss_jsd, alternative="two-sided")

    stats = {
        "api_jsd_mean": float(api_jsd.mean()),
        "api_jsd_std": float(api_jsd.std()),
        "oss_jsd_mean": float(oss_jsd.mean()),
        "oss_jsd_std": float(oss_jsd.std()),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(u_p),
    }

    # Per agreement level
    for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
        api_sub = df[(df["model_type"] == "API") & (df["agreement_level"] == level)]["jsd"].values
        oss_sub = df[(df["model_type"] == "OSS") & (df["agreement_level"] == level)]["jsd"].values
        if len(api_sub) > 0 and len(oss_sub) > 0:
            u, p = mannwhitneyu(api_sub, oss_sub, alternative="two-sided")
            stats[f"{level}_api_mean"] = float(api_sub.mean())
            stats[f"{level}_oss_mean"] = float(oss_sub.mean())
            stats[f"{level}_mannwhitney_p"] = float(p)

    return stats


def plot_confusion_pattern_comparison(
    human_df: pd.DataFrame,
    llm_dist_dfs: dict,
    emotion_labels: list[str],
    fig_dir: str,
) -> None:
    """Heatmap of co-occurrence patterns for human vs each LLM."""
    dist_cols = [f"{e}_dist" for e in emotion_labels]
    n = len(emotion_labels)

    # Compute correlation matrix of emotion rates for human
    h_matrix = human_df[dist_cols].values
    h_corr = np.corrcoef(h_matrix.T)

    models = sorted(llm_dist_dfs.keys())
    n_panels = 1 + len(models)

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))

    # Short labels
    short_labels = [e[:4] for e in emotion_labels]

    mask = np.triu(np.ones((n, n), dtype=bool), k=0)

    sns.heatmap(
        h_corr, mask=mask, ax=axes[0], cmap="RdBu_r", center=0,
        vmin=-0.5, vmax=0.5,
        xticklabels=short_labels, yticklabels=short_labels,
        square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.6},
    )
    axes[0].set_title("Human")
    axes[0].tick_params(axis="both", labelsize=10)

    for i, model in enumerate(models):
        llm_df = llm_dist_dfs[model]
        llm_all = llm_df[llm_df["temperature"] == "all"]
        l_matrix = llm_all[dist_cols].values
        l_corr = np.corrcoef(l_matrix.T)

        sns.heatmap(
            l_corr, mask=mask, ax=axes[i + 1], cmap="RdBu_r", center=0,
            vmin=-0.5, vmax=0.5,
            xticklabels=short_labels, yticklabels=short_labels,
            square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.6},
        )
        axes[i + 1].set_title(get_display_name(model))
        axes[i + 1].tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"emotion_correlation_heatmaps.{FIGURE_FORMAT}")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: LLM-Human comparison analysis")
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
    np.random.seed(seed)

    dist_cols = [f"{e}_dist" for e in emotion_labels]

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("=" * 60)
    print("Phase 3: LLM-Human Comparison Analysis")
    print("=" * 60)

    print("\n--- Loading human distributions ---")
    human_df = pd.read_parquet(os.path.join(processed_dir, "goemotions_annotator_distributions.parquet"))
    print(f"  Human texts: {len(human_df)}")

    # Load core set to filter human_df to only texts used in LLM inference
    core_set = pd.read_parquet(os.path.join(processed_dir, "core_set.parquet"))
    core_ids = set(core_set["id"].unique())
    human_core = human_df[human_df["id"].isin(core_ids)].copy()
    print(f"  Core set texts (human): {len(human_core)}")

    # Load LLM results
    llm_results_dir = os.path.join(processed_dir, "llm_results")
    llm_files = [f for f in os.listdir(llm_results_dir) if f.endswith("_results.parquet")]

    llm_raw = {}
    for fname in sorted(llm_files):
        path = os.path.join(llm_results_dir, fname)
        df = pd.read_parquet(path)
        model_name = df["model"].iloc[0]
        llm_raw[model_name] = df
        print(f"  Loaded {model_name}: {len(df)} rows")

    # --------------------------------------------------------
    # Step 1: Build LLM distributions
    # --------------------------------------------------------
    print("\n--- Building LLM distributions ---")
    llm_dist_dfs = {}
    for model_name, raw_df in llm_raw.items():
        llm_dist = build_llm_distributions(raw_df, emotion_labels, model_name)
        llm_dist_dfs[model_name] = llm_dist
        print(f"  {model_name}: {len(llm_dist)} distribution records")

    # Save LLM distributions
    all_llm_dists = pd.concat(llm_dist_dfs.values(), ignore_index=True)
    # Ensure temperature is string type to avoid mixed-type parquet issues
    all_llm_dists["temperature"] = all_llm_dists["temperature"].astype(str)
    llm_dist_path = os.path.join(processed_dir, "llm_distributions.parquet")
    all_llm_dists.to_parquet(llm_dist_path, index=False)
    print(f"  Saved: {llm_dist_path}")

    # --------------------------------------------------------
    # Step 2: Per-text comparison metrics
    # --------------------------------------------------------
    print("\n--- Computing per-text comparison metrics ---")
    all_metrics = []
    for model_name, llm_dist in llm_dist_dfs.items():
        metrics = compute_per_text_metrics(human_core, llm_dist, emotion_labels)
        all_metrics.append(metrics)
        print(f"  {model_name}: {len(metrics)} comparisons")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df["temperature"] = metrics_df["temperature"].astype(str)
    metrics_path = os.path.join(processed_dir, "per_text_metrics.parquet")
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    # --------------------------------------------------------
    # Step 3: Aggregate statistics
    # --------------------------------------------------------
    print("\n--- Computing aggregate statistics ---")
    agg_stats = compute_aggregate_stats(metrics_df)
    for key, val in agg_stats.items():
        print(f"  {key}: JSD mean={val['jsd']['mean']:.4f} +/- {val['jsd']['std']:.4f}, "
              f"entropy corr rho={val['entropy_correlation']['spearman_r']:.3f}")

    # --------------------------------------------------------
    # Step 4: Agreement-level stratified analysis
    # --------------------------------------------------------
    print("\n--- Agreement-level stratified analysis ---")
    agreement_stats = compute_agreement_level_stats(metrics_df)
    for key, val in agreement_stats.items():
        print(f"  {key}:")
        for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
            if level in val:
                s = val[level]
                print(f"    {level}: JSD={s['jsd_mean']:.4f}, n={s['n_texts']}")

    # --------------------------------------------------------
    # Step 5: Per-category divergence
    # --------------------------------------------------------
    print("\n--- Per-category divergence analysis ---")
    cat_div = compute_per_category_divergence(human_core, all_llm_dists, emotion_labels)
    cat_div_path = os.path.join(processed_dir, "per_category_divergence.parquet")
    cat_div.to_parquet(cat_div_path, index=False)
    print(f"  Saved: {cat_div_path}")

    # Show top over/under-predicted categories for each model (all temps)
    for model in cat_div["model"].unique():
        m_df = cat_div[(cat_div["model"] == model) & (cat_div["temperature"] == "all")]
        top_over = m_df.nlargest(5, "rate_diff")
        top_under = m_df.nsmallest(5, "rate_diff")
        print(f"\n  {model} - Top over-predicted:")
        for _, row in top_over.iterrows():
            print(f"    {row['emotion']:20s}: diff={row['rate_diff']:+.4f}, rho={row['spearman_r']:.3f}")
        print(f"  {model} - Top under-predicted:")
        for _, row in top_under.iterrows():
            print(f"    {row['emotion']:20s}: diff={row['rate_diff']:+.4f}, rho={row['spearman_r']:.3f}")

    # --------------------------------------------------------
    # Step 6: Visualizations
    # --------------------------------------------------------
    print("\n--- Generating visualizations ---")

    plot_jsd_by_agreement(metrics_df, fig_dir)
    plot_entropy_correlation(metrics_df, fig_dir)
    plot_per_category_heatmap(cat_div, fig_dir)
    plot_category_correlation_heatmap(cat_div, fig_dir)
    plot_temperature_effect(metrics_df, fig_dir)
    plot_overall_distribution_comparison(human_core, llm_dist_dfs, emotion_labels, fig_dir)
    plot_jsd_distribution(metrics_df, fig_dir)
    plot_confusion_pattern_comparison(human_core, llm_dist_dfs, emotion_labels, fig_dir)
    plot_api_vs_oss_comparison(metrics_df, fig_dir)
    plot_model_ranking_summary(metrics_df, cat_div, fig_dir)

    # --------------------------------------------------------
    # Compile and save results
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Step 7: API vs OSS comparison
    # --------------------------------------------------------
    print("\n--- API vs OSS comparison ---")
    api_oss_stats = compute_api_vs_oss_stats(metrics_df)
    print(f"  API JSD mean: {api_oss_stats['api_jsd_mean']:.4f} +/- {api_oss_stats['api_jsd_std']:.4f}")
    print(f"  OSS JSD mean: {api_oss_stats['oss_jsd_mean']:.4f} +/- {api_oss_stats['oss_jsd_std']:.4f}")
    print(f"  Mann-Whitney p: {api_oss_stats['mannwhitney_p']:.2e}")

    print("\n--- Saving final results ---")
    results = {
        "aggregate_stats": agg_stats,
        "agreement_level_stats": agreement_stats,
        "api_vs_oss_stats": api_oss_stats,
        "per_category_summary": {},
    }

    # Per-category summary for 'all' temperature
    for model in cat_div["model"].unique():
        m_df = cat_div[(cat_div["model"] == model) & (cat_div["temperature"] == "all")]
        results["per_category_summary"][model] = {
            row["emotion"]: {
                "human_mean_rate": row["human_mean_rate"],
                "llm_mean_rate": row["llm_mean_rate"],
                "rate_diff": row["rate_diff"],
                "spearman_r": row["spearman_r"],
            }
            for _, row in m_df.iterrows()
        }

    results_path = os.path.join(processed_dir, "llm_human_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 60)
    print("Phase 3 Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
