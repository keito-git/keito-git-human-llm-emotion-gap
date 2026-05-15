"""
EmoBank VAD Analysis: Compare human annotator distributions with LLM predictions.

Analyses:
    1. Human VAD distribution statistics
    2. LLM VAD prediction statistics per model
    3. Correlation analysis (Pearson, Spearman) between human mean and LLM predictions
    4. MAE (Mean Absolute Error) per dimension per model
    5. Distribution comparison (KDE overlap, KS test)
    6. Agreement-level analysis (do LLMs better match high-agreement texts?)
    7. Cross-dataset consistency with GoEmotions findings

Outputs:
    - emobank_analysis_results.json
    - Visualization figures

Usage:
    python emobank_analysis.py --config ../config/experiment_config.yaml
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_human_data(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load EmoBank human annotator stats and metadata."""
    emobank_dir = os.path.join(processed_dir, "emobank")
    stats = pd.read_parquet(os.path.join(emobank_dir, "emobank_annotator_stats.parquet"))
    core_set = pd.read_parquet(os.path.join(emobank_dir, "emobank_core_set.parquet"))
    return stats, core_set


def load_llm_results(processed_dir: str) -> dict[str, pd.DataFrame]:
    """Load all available EmoBank LLM results."""
    results_dir = os.path.join(processed_dir, "emobank", "llm_results")
    if not os.path.exists(results_dir):
        print(f"  No LLM results directory found: {results_dir}")
        return {}

    model_results = {}
    for f in os.listdir(results_dir):
        if f.startswith("emobank_") and f.endswith("_results.parquet"):
            model_name = f.replace("emobank_", "").replace("_results.parquet", "")
            path = os.path.join(results_dir, f)
            df = pd.read_parquet(path)
            # Filter valid results (handle missing error columns for OSS models)
            mask = pd.Series(True, index=df.index)
            if "api_error" in df.columns:
                mask &= ~df["api_error"]
            if "parse_error" in df.columns:
                mask &= ~df["parse_error"]
            # Also filter out rows where VAD predictions are NaN
            mask &= df["V_pred"].notna() & df["A_pred"].notna() & df["D_pred"].notna()
            valid = df[mask].copy()
            model_results[model_name] = valid
            print(f"  Loaded {model_name}: {len(valid)}/{len(df)} valid results")

    return model_results


def analyze_human_vad(stats: pd.DataFrame, core_set: pd.DataFrame) -> dict:
    """Analyze human VAD distribution statistics."""
    print("\n=== Human VAD Analysis ===")

    # Filter to core set
    core_stats = stats[stats["id"].isin(core_set["id"])].copy()

    results = {
        "n_texts": len(core_stats),
        "n_annotators_mean": float(core_stats["num_annotators"].mean()),
        "n_annotators_min": int(core_stats["num_annotators"].min()),
        "n_annotators_max": int(core_stats["num_annotators"].max()),
    }

    for dim in ["V", "A", "D"]:
        results[f"human_{dim}_mean"] = float(core_stats[f"{dim}_mean"].mean())
        results[f"human_{dim}_std_of_means"] = float(core_stats[f"{dim}_mean"].std())
        results[f"human_{dim}_mean_within_std"] = float(core_stats[f"{dim}_std"].mean())
        results[f"human_{dim}_median"] = float(core_stats[f"{dim}_mean"].median())

    print(f"  Core set texts: {results['n_texts']}")
    for dim in ["V", "A", "D"]:
        print(f"  {dim}: mean={results[f'human_{dim}_mean']:.3f}, "
              f"between-text std={results[f'human_{dim}_std_of_means']:.3f}, "
              f"within-text std={results[f'human_{dim}_mean_within_std']:.3f}")

    return results


def analyze_llm_vad(model_results: dict[str, pd.DataFrame]) -> dict:
    """Analyze LLM VAD prediction statistics."""
    print("\n=== LLM VAD Analysis ===")
    results = {}

    for model_name, df in model_results.items():
        model_stats = {"n_predictions": len(df)}

        for dim in ["V", "A", "D"]:
            col = f"{dim}_pred"
            model_stats[f"{dim}_mean"] = float(df[col].mean())
            model_stats[f"{dim}_std"] = float(df[col].std())
            model_stats[f"{dim}_median"] = float(df[col].median())
            model_stats[f"{dim}_min"] = float(df[col].min())
            model_stats[f"{dim}_max"] = float(df[col].max())

        # Per-temperature stats
        temp_stats = {}
        for temp in sorted(df["temperature"].unique()):
            temp_df = df[df["temperature"] == temp]
            temp_stats[str(temp)] = {
                "n": len(temp_df),
            }
            for dim in ["V", "A", "D"]:
                # Variance across samples for each text at this temperature
                text_var = temp_df.groupby("id")[f"{dim}_pred"].var().mean()
                temp_stats[str(temp)][f"{dim}_within_text_var"] = float(text_var) if not np.isnan(text_var) else 0.0

        model_stats["per_temperature"] = temp_stats
        results[model_name] = model_stats

        print(f"\n  {model_name}:")
        for dim in ["V", "A", "D"]:
            print(f"    {dim}: mean={model_stats[f'{dim}_mean']:.3f}, "
                  f"std={model_stats[f'{dim}_std']:.3f}")

    return results


def compute_correlations(
    core_stats: pd.DataFrame,
    model_results: dict[str, pd.DataFrame],
) -> dict:
    """Compute Pearson and Spearman correlations between human means and LLM predictions."""
    print("\n=== Correlation Analysis ===")
    results = {}

    for model_name, df in model_results.items():
        # Average LLM predictions across all samples for each text
        llm_means = df.groupby("id").agg({
            "V_pred": "mean",
            "A_pred": "mean",
            "D_pred": "mean",
        }).reset_index()

        # Merge with human stats
        merged = core_stats[["id", "V_mean", "A_mean", "D_mean"]].merge(
            llm_means, on="id", how="inner"
        )

        model_corr = {"n_texts": len(merged)}
        for dim in ["V", "A", "D"]:
            human_col = f"{dim}_mean"
            llm_col = f"{dim}_pred"

            pearson_r, pearson_p = scipy_stats.pearsonr(merged[human_col], merged[llm_col])
            spearman_r, spearman_p = scipy_stats.spearmanr(merged[human_col], merged[llm_col])

            model_corr[f"{dim}_pearson_r"] = float(pearson_r)
            model_corr[f"{dim}_pearson_p"] = float(pearson_p)
            model_corr[f"{dim}_spearman_r"] = float(spearman_r)
            model_corr[f"{dim}_spearman_p"] = float(spearman_p)

        results[model_name] = model_corr

        print(f"\n  {model_name} (n={model_corr['n_texts']}):")
        for dim in ["V", "A", "D"]:
            print(f"    {dim}: Pearson r={model_corr[f'{dim}_pearson_r']:.4f} "
                  f"(p={model_corr[f'{dim}_pearson_p']:.2e}), "
                  f"Spearman r={model_corr[f'{dim}_spearman_r']:.4f} "
                  f"(p={model_corr[f'{dim}_spearman_p']:.2e})")

    return results


def compute_mae(
    core_stats: pd.DataFrame,
    model_results: dict[str, pd.DataFrame],
) -> dict:
    """Compute MAE between human means and LLM predictions."""
    print("\n=== MAE Analysis ===")
    results = {}

    for model_name, df in model_results.items():
        llm_means = df.groupby("id").agg({
            "V_pred": "mean",
            "A_pred": "mean",
            "D_pred": "mean",
        }).reset_index()

        merged = core_stats[["id", "V_mean", "A_mean", "D_mean"]].merge(
            llm_means, on="id", how="inner"
        )

        model_mae = {"n_texts": len(merged)}
        for dim in ["V", "A", "D"]:
            mae = np.abs(merged[f"{dim}_mean"] - merged[f"{dim}_pred"]).mean()
            rmse = np.sqrt(((merged[f"{dim}_mean"] - merged[f"{dim}_pred"]) ** 2).mean())
            model_mae[f"{dim}_mae"] = float(mae)
            model_mae[f"{dim}_rmse"] = float(rmse)

        # Overall MAE across all dimensions
        model_mae["overall_mae"] = float(np.mean([
            model_mae["V_mae"], model_mae["A_mae"], model_mae["D_mae"]
        ]))

        results[model_name] = model_mae

        print(f"\n  {model_name}:")
        for dim in ["V", "A", "D"]:
            print(f"    {dim}: MAE={model_mae[f'{dim}_mae']:.4f}, "
                  f"RMSE={model_mae[f'{dim}_rmse']:.4f}")
        print(f"    Overall MAE: {model_mae['overall_mae']:.4f}")

    return results


def compute_distribution_tests(
    core_stats: pd.DataFrame,
    model_results: dict[str, pd.DataFrame],
) -> dict:
    """KS test and distribution statistics comparing human and LLM VAD distributions."""
    print("\n=== Distribution Comparison (KS Test) ===")
    results = {}

    for model_name, df in model_results.items():
        llm_means = df.groupby("id").agg({
            "V_pred": "mean",
            "A_pred": "mean",
            "D_pred": "mean",
        }).reset_index()

        merged = core_stats[["id", "V_mean", "A_mean", "D_mean"]].merge(
            llm_means, on="id", how="inner"
        )

        model_ks = {}
        for dim in ["V", "A", "D"]:
            ks_stat, ks_p = scipy_stats.ks_2samp(
                merged[f"{dim}_mean"].values,
                merged[f"{dim}_pred"].values,
            )
            model_ks[f"{dim}_ks_statistic"] = float(ks_stat)
            model_ks[f"{dim}_ks_pvalue"] = float(ks_p)

        results[model_name] = model_ks

        print(f"\n  {model_name}:")
        for dim in ["V", "A", "D"]:
            sig = "***" if model_ks[f"{dim}_ks_pvalue"] < 0.001 else \
                  "**" if model_ks[f"{dim}_ks_pvalue"] < 0.01 else \
                  "*" if model_ks[f"{dim}_ks_pvalue"] < 0.05 else "n.s."
            print(f"    {dim}: KS={model_ks[f'{dim}_ks_statistic']:.4f}, "
                  f"p={model_ks[f'{dim}_ks_pvalue']:.2e} {sig}")

    return results


def analyze_by_agreement(
    core_stats: pd.DataFrame,
    model_results: dict[str, pd.DataFrame],
    core_set: pd.DataFrame,
) -> dict:
    """Analyze LLM accuracy by human agreement level."""
    print("\n=== Analysis by Agreement Level ===")
    results = {}

    # Add agreement level to core_stats
    merged_core = core_stats[core_stats["id"].isin(core_set["id"])].copy()
    agreement_col = core_set[["id", "agreement_level"]].drop_duplicates()
    merged_core = merged_core.merge(agreement_col, on="id", how="left")

    for model_name, df in model_results.items():
        llm_means = df.groupby("id").agg({
            "V_pred": "mean",
            "A_pred": "mean",
            "D_pred": "mean",
        }).reset_index()

        merged = merged_core[["id", "V_mean", "A_mean", "D_mean", "agreement_level"]].merge(
            llm_means, on="id", how="inner"
        )

        model_agreement = {}
        for level in ["high_agreement", "moderate_agreement", "low_agreement"]:
            subset = merged[merged["agreement_level"] == level]
            if len(subset) == 0:
                continue

            level_results = {"n_texts": len(subset)}
            for dim in ["V", "A", "D"]:
                mae = np.abs(subset[f"{dim}_mean"] - subset[f"{dim}_pred"]).mean()
                pearson_r, _ = scipy_stats.pearsonr(
                    subset[f"{dim}_mean"], subset[f"{dim}_pred"]
                ) if len(subset) > 2 else (np.nan, np.nan)
                level_results[f"{dim}_mae"] = float(mae)
                level_results[f"{dim}_pearson_r"] = float(pearson_r) if not np.isnan(pearson_r) else None

            model_agreement[level] = level_results

        results[model_name] = model_agreement

        print(f"\n  {model_name}:")
        for level in ["high_agreement", "moderate_agreement", "low_agreement"]:
            if level in model_agreement:
                r = model_agreement[level]
                v_mae = r.get("V_mae", "N/A")
                print(f"    {level} (n={r['n_texts']}): "
                      f"V_MAE={v_mae:.4f}, "
                      f"V_r={r.get('V_pearson_r', 'N/A')}")

    return results


def analyze_temperature_effect(
    core_stats: pd.DataFrame,
    model_results: dict[str, pd.DataFrame],
) -> dict:
    """Analyze how temperature affects VAD prediction variance."""
    print("\n=== Temperature Effect on VAD Variance ===")
    results = {}

    for model_name, df in model_results.items():
        temp_results = {}
        for temp in sorted(df["temperature"].unique()):
            temp_df = df[df["temperature"] == temp]

            # Per-text variance at this temperature
            text_vars = temp_df.groupby("id").agg({
                "V_pred": "var",
                "A_pred": "var",
                "D_pred": "var",
            }).mean()

            # MAE at this temperature
            temp_means = temp_df.groupby("id").agg({
                "V_pred": "mean",
                "A_pred": "mean",
                "D_pred": "mean",
            }).reset_index()

            merged = core_stats[["id", "V_mean", "A_mean", "D_mean"]].merge(
                temp_means, on="id", how="inner"
            )

            temp_result = {"temperature": float(temp), "n_texts": len(merged)}
            for dim in ["V", "A", "D"]:
                mae = np.abs(merged[f"{dim}_mean"] - merged[f"{dim}_pred"]).mean()
                temp_result[f"{dim}_mae"] = float(mae)
                temp_result[f"{dim}_within_text_var"] = float(text_vars[f"{dim}_pred"])

            temp_results[str(temp)] = temp_result

        results[model_name] = temp_results

        print(f"\n  {model_name}:")
        for temp_key, tr in temp_results.items():
            print(f"    T={temp_key}: V_MAE={tr['V_mae']:.4f}, "
                  f"V_var={tr['V_within_text_var']:.4f}")

    return results


MODEL_DISPLAY_NAMES = {
    # Standard keys (used in GoEmotions pipeline)
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "llama3.1:8b": "Llama 3.1 8B",
    "llama3.1-8b": "Llama 3.1 8B",
    "qwen3-8b": "Qwen3-8B",
    "roberta-ft": "RoBERTa-FT",
    "finetuned": "RoBERTa-FT",
    # EmoBank filename-derived keys (underscores instead of hyphens/dots)
    "gpt_5_4_mini": "GPT-5.4-mini",
    "claude_haiku_4_5_20251001": "Claude Haiku 4.5",
    "llama3_1-8b": "Llama 3.1 8B",
    "llama3_1_8b": "Llama 3.1 8B",
}


def get_display_name(model: str) -> str:
    """Return paper-consistent display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model, model)


def generate_visualizations(
    core_stats: pd.DataFrame,
    model_results: dict[str, pd.DataFrame],
    core_set: pd.DataFrame,
    output_dir: str,
):
    """Generate analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "figure.titlesize": 24,
    })

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Filter core_stats to core set
    cs = core_stats[core_stats["id"].isin(core_set["id"])].copy()

    # --- Figure 1: VAD distribution comparison (human vs LLMs) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for i, dim in enumerate(["V", "A", "D"]):
        ax = axes[i]
        # Human
        ax.hist(cs[f"{dim}_mean"].values, bins=30, alpha=0.5, density=True,
                label="Human", color="gray")
        # LLMs
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for j, (model_name, df) in enumerate(model_results.items()):
            llm_means = df.groupby("id")[f"{dim}_pred"].mean()
            ax.hist(llm_means.values, bins=30, alpha=0.3, density=True,
                    label=get_display_name(model_name), color=colors[j % len(colors)])
        ax.set_xlabel(f"{dim} (1-5)")
        ax.set_ylabel("Density")
        dim_names = {"V": "Valence", "A": "Arousal", "D": "Dominance"}
        ax.set_title(f"{dim_names[dim]} Distribution")
        ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "emobank_vad_distributions.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # --- Figure 2: Scatter plots (human vs LLM for each dimension) ---
    n_models = len(model_results)
    if n_models > 0:
        fig, axes = plt.subplots(n_models, 3, figsize=(16, 6 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        for i, (model_name, df) in enumerate(model_results.items()):
            llm_means = df.groupby("id").agg({
                "V_pred": "mean", "A_pred": "mean", "D_pred": "mean"
            }).reset_index()
            merged = cs[["id", "V_mean", "A_mean", "D_mean"]].merge(
                llm_means, on="id", how="inner"
            )

            for j, dim in enumerate(["V", "A", "D"]):
                ax = axes[i, j]
                ax.scatter(merged[f"{dim}_mean"], merged[f"{dim}_pred"],
                          alpha=0.2, s=10)
                ax.plot([1, 5], [1, 5], "r--", alpha=0.5, label="y=x")
                r, _ = scipy_stats.pearsonr(merged[f"{dim}_mean"], merged[f"{dim}_pred"])
                ax.set_xlabel(f"Human {dim}")
                ax.set_ylabel(f"LLM {dim}")
                ax.set_title(get_display_name(model_name))
                # Panel annotation: statistical summary (kept as in-plot text)
                ax.annotate(
                    f"r = {r:.3f}",
                    xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=16, ha="left", va="top",
                )
                ax.set_xlim(0.8, 5.2)
                ax.set_ylim(0.8, 5.2)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "emobank_scatter_human_vs_llm.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # --- Figure 3: MAE by agreement level ---
    if n_models > 0:
        agreement_col = core_set[["id", "agreement_level"]].drop_duplicates()
        cs_agree = cs.merge(agreement_col, on="id", how="left")

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        levels = ["high_agreement", "moderate_agreement", "low_agreement"]
        level_labels = ["High", "Moderate", "Low"]

        for j, dim in enumerate(["V", "A", "D"]):
            ax = axes[j]
            x_pos = np.arange(len(levels))
            width = 0.8 / max(n_models, 1)

            for k, (model_name, df) in enumerate(model_results.items()):
                llm_means = df.groupby("id").agg({
                    f"{dim}_pred": "mean"
                }).reset_index()
                merged = cs_agree[["id", f"{dim}_mean", "agreement_level"]].merge(
                    llm_means, on="id", how="inner"
                )

                maes = []
                for level in levels:
                    subset = merged[merged["agreement_level"] == level]
                    if len(subset) > 0:
                        mae = np.abs(subset[f"{dim}_mean"] - subset[f"{dim}_pred"]).mean()
                        maes.append(mae)
                    else:
                        maes.append(0)

                ax.bar(x_pos + k * width, maes, width, label=get_display_name(model_name), alpha=0.8)

            ax.set_xticks(x_pos + width * (n_models - 1) / 2)
            ax.set_xticklabels(level_labels)
            ax.set_ylabel("MAE")
            dim_names = {"V": "Valence", "A": "Arousal", "D": "Dominance"}
            ax.set_title(f"{dim_names[dim]} MAE by Agreement")
            ax.legend(
                bbox_to_anchor=(0.5, -0.20),
                loc="upper center",
                ncol=2,
                fontsize=14,
            )

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "emobank_mae_by_agreement.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    # --- Figure 4: Temperature effect on within-text variance ---
    if n_models > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for j, dim in enumerate(["V", "A", "D"]):
            ax = axes[j]
            for model_name, df in model_results.items():
                temps = sorted(df["temperature"].unique())
                variances = []
                for temp in temps:
                    temp_df = df[df["temperature"] == temp]
                    text_var = temp_df.groupby("id")[f"{dim}_pred"].var().mean()
                    variances.append(text_var)
                ax.plot(temps, variances, marker="o", label=get_display_name(model_name))

            ax.set_xlabel("Temperature")
            ax.set_ylabel("Mean within-text variance")
            dim_names = {"V": "Valence", "A": "Arousal", "D": "Dominance"}
            ax.set_title(f"{dim_names[dim]}")
            ax.legend(loc="center right", fontsize=12)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "emobank_temperature_variance.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="EmoBank VAD analysis")
    parser.add_argument("--config", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment_config.yaml"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_data"]

    # Load data
    print("=== Loading data ===")
    stats, core_set = load_human_data(processed_dir)
    model_results = load_llm_results(processed_dir)

    # Filter stats to core set
    core_stats = stats[stats["id"].isin(core_set["id"])].copy()

    # Run analyses
    all_results = {}
    all_results["human_stats"] = analyze_human_vad(stats, core_set)

    if model_results:
        all_results["llm_stats"] = analyze_llm_vad(model_results)
        all_results["correlations"] = compute_correlations(core_stats, model_results)
        all_results["mae"] = compute_mae(core_stats, model_results)
        all_results["ks_tests"] = compute_distribution_tests(core_stats, model_results)
        all_results["by_agreement"] = analyze_by_agreement(core_stats, model_results, core_set)
        all_results["temperature_effect"] = analyze_temperature_effect(core_stats, model_results)

        if not args.no_figures:
            print("\n=== Generating Figures ===")
            root_dir = config["paths"]["root"]
            output_dir = os.path.join(root_dir, "paper", "en")
            os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
            generate_visualizations(core_stats, model_results, core_set, output_dir)
    else:
        print("\n  No LLM results available yet. Run inference first.")

    # Save results
    output_path = os.path.join(processed_dir, "emobank", "emobank_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n=== Results saved: {output_path} ===")


if __name__ == "__main__":
    main()
