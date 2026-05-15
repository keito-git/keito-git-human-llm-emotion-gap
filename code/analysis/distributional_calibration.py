"""
W1: Distributional Calibration Methods.

Proposes and evaluates three post-hoc calibration methods to align LLM emotion
distributions closer to human annotator distributions:
    1. Temperature Scaling: learn a scalar temperature to calibrate softmax outputs
    2. Label-wise Bias Correction: subtract per-category bias estimated on dev set
    3. Isotonic Regression: non-parametric calibration per category

Uses 5-fold cross-validation on the 2,000-text core set to evaluate each method.

Usage:
    python3 distributional_calibration.py \
        --human_parquet /path/to/core_set.parquet \
        --llm_parquet /path/to/llm_distributions.parquet \
        --output_dir /path/to/output/

Outputs:
    - calibration_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon
from scipy.optimize import minimize_scalar
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


def normalize_distribution(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize to sum to 1."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.maximum(dist, 0)
    total = dist.sum()
    if total == 0:
        return np.ones_like(dist) / len(dist)
    return dist / total


def shannon_entropy(dist: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy."""
    dist = normalize_distribution(dist)
    return float(scipy_entropy(dist, base=base))


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD (squared, [0, 1])."""
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return float(jensenshannon(p, q, base=2) ** 2)


# ============================================================
# Calibration Method 1: Temperature Scaling
# ============================================================

def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to logits and return softmax distribution."""
    if T <= 0:
        T = 0.01
    scaled = logits / T
    # Numerical stability
    scaled = scaled - scaled.max()
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum()


def find_optimal_temperature(
    human_dists: np.ndarray,
    llm_dists: np.ndarray,
) -> float:
    """Find optimal temperature T that minimizes mean JSD on training data."""
    # Convert LLM distributions to "logits" via log
    llm_logits = np.log(llm_dists + 1e-12)

    def objective(T):
        jsds = []
        for i in range(len(human_dists)):
            calibrated = temperature_scale(llm_logits[i], T)
            jsds.append(compute_jsd(human_dists[i], calibrated))
        return np.mean(jsds)

    result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
    return result.x


def apply_temperature_scaling(
    llm_dists: np.ndarray,
    T: float,
) -> np.ndarray:
    """Apply temperature scaling to all distributions."""
    llm_logits = np.log(llm_dists + 1e-12)
    calibrated = np.zeros_like(llm_dists)
    for i in range(len(llm_dists)):
        calibrated[i] = temperature_scale(llm_logits[i], T)
    return calibrated


# ============================================================
# Calibration Method 2: Label-wise Bias Correction
# ============================================================

def estimate_bias(
    human_dists: np.ndarray,
    llm_dists: np.ndarray,
) -> np.ndarray:
    """Estimate per-category bias (LLM mean - Human mean)."""
    return llm_dists.mean(axis=0) - human_dists.mean(axis=0)


def apply_bias_correction(
    llm_dists: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Subtract bias and re-normalize."""
    corrected = llm_dists - bias[np.newaxis, :]
    corrected = np.maximum(corrected, 0)
    # Re-normalize
    row_sums = corrected.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    return corrected / row_sums


# ============================================================
# Calibration Method 3: Isotonic Regression
# ============================================================

def fit_isotonic(
    human_dists: np.ndarray,
    llm_dists: np.ndarray,
) -> list:
    """Fit isotonic regression per emotion category."""
    n_categories = human_dists.shape[1]
    models = []
    for k in range(n_categories):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(llm_dists[:, k], human_dists[:, k])
        models.append(ir)
    return models


def apply_isotonic(
    llm_dists: np.ndarray,
    models: list,
) -> np.ndarray:
    """Apply isotonic regression and re-normalize."""
    calibrated = np.zeros_like(llm_dists)
    for k, ir in enumerate(models):
        calibrated[:, k] = ir.predict(llm_dists[:, k])
    calibrated = np.maximum(calibrated, 0)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    return calibrated / row_sums


# ============================================================
# Evaluation
# ============================================================

def evaluate_distributions(
    human_dists: np.ndarray,
    pred_dists: np.ndarray,
) -> dict:
    """Compute JSD and entropy correlation metrics."""
    jsds = np.array([compute_jsd(human_dists[i], pred_dists[i]) for i in range(len(human_dists))])
    human_entropies = np.array([shannon_entropy(human_dists[i]) for i in range(len(human_dists))])
    pred_entropies = np.array([shannon_entropy(pred_dists[i]) for i in range(len(pred_dists))])

    rho, p_val = spearmanr(human_entropies, pred_entropies)

    return {
        "jsd_mean": float(jsds.mean()),
        "jsd_std": float(jsds.std()),
        "jsd_median": float(np.median(jsds)),
        "entropy_spearman_rho": float(rho),
        "entropy_spearman_p": float(p_val),
    }


def cross_validate_calibration(
    human_dists: np.ndarray,
    llm_dists: np.ndarray,
    text_ids: np.ndarray,
    agreement_levels: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """5-fold cross-validation of all calibration methods."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    methods = {
        "uncalibrated": {"jsds": [], "entropy_rhos": []},
        "temperature_scaling": {"jsds": [], "entropy_rhos": [], "temps": []},
        "bias_correction": {"jsds": [], "entropy_rhos": []},
        "isotonic_regression": {"jsds": [], "entropy_rhos": []},
    }

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(human_dists)):
        print(f"  Fold {fold_idx + 1}/{n_folds}...")

        h_train, h_test = human_dists[train_idx], human_dists[test_idx]
        l_train, l_test = llm_dists[train_idx], llm_dists[test_idx]
        agr_test = agreement_levels[test_idx]

        fold_data = {"fold": fold_idx}

        # Uncalibrated
        uncal_metrics = evaluate_distributions(h_test, l_test)
        fold_data["uncalibrated"] = uncal_metrics

        # Temperature scaling
        opt_T = find_optimal_temperature(h_train, l_train)
        cal_ts = apply_temperature_scaling(l_test, opt_T)
        ts_metrics = evaluate_distributions(h_test, cal_ts)
        ts_metrics["optimal_temperature"] = opt_T
        fold_data["temperature_scaling"] = ts_metrics

        # Bias correction
        bias = estimate_bias(h_train, l_train)
        cal_bc = apply_bias_correction(l_test, bias)
        bc_metrics = evaluate_distributions(h_test, cal_bc)
        fold_data["bias_correction"] = bc_metrics

        # Isotonic regression
        iso_models = fit_isotonic(h_train, l_train)
        cal_iso = apply_isotonic(l_test, iso_models)
        iso_metrics = evaluate_distributions(h_test, cal_iso)
        fold_data["isotonic_regression"] = iso_metrics

        # Per-agreement-level analysis for best method
        for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
            mask = agr_test == level
            if mask.sum() > 0:
                for method_name, cal_dists in [
                    ("uncalibrated", l_test),
                    ("temperature_scaling", cal_ts),
                    ("bias_correction", cal_bc),
                    ("isotonic_regression", cal_iso),
                ]:
                    key = f"{method_name}_{level}"
                    sub_jsds = [compute_jsd(h_test[i], cal_dists[i]) for i in range(len(h_test)) if mask[i]]
                    fold_data[key] = {
                        "jsd_mean": float(np.mean(sub_jsds)),
                        "n": int(mask.sum()),
                    }

        fold_results.append(fold_data)

    # Aggregate across folds
    summary = {}
    for method in ["uncalibrated", "temperature_scaling", "bias_correction", "isotonic_regression"]:
        jsds = [f[method]["jsd_mean"] for f in fold_results]
        rhos = [f[method]["entropy_spearman_rho"] for f in fold_results]
        summary[method] = {
            "jsd_mean": float(np.mean(jsds)),
            "jsd_std_across_folds": float(np.std(jsds)),
            "entropy_rho_mean": float(np.mean(rhos)),
            "entropy_rho_std": float(np.std(rhos)),
        }
        if method == "temperature_scaling":
            temps = [f[method]["optimal_temperature"] for f in fold_results]
            summary[method]["mean_optimal_temperature"] = float(np.mean(temps))

    # Per-agreement improvement
    for level in ["full_agreement", "partial_agreement", "full_disagreement"]:
        for method in ["temperature_scaling", "bias_correction", "isotonic_regression"]:
            uncal_key = f"uncalibrated_{level}"
            cal_key = f"{method}_{level}"
            uncal_jsds = [f.get(uncal_key, {}).get("jsd_mean", float("nan")) for f in fold_results]
            cal_jsds = [f.get(cal_key, {}).get("jsd_mean", float("nan")) for f in fold_results]
            valid = [(u, c) for u, c in zip(uncal_jsds, cal_jsds) if not np.isnan(u) and not np.isnan(c)]
            if valid:
                u_mean = np.mean([v[0] for v in valid])
                c_mean = np.mean([v[1] for v in valid])
                summary[f"{method}_{level}"] = {
                    "uncalibrated_jsd": float(u_mean),
                    "calibrated_jsd": float(c_mean),
                    "improvement": float(u_mean - c_mean),
                    "improvement_pct": float((u_mean - c_mean) / u_mean * 100) if u_mean > 0 else 0,
                }

    return {"fold_results": fold_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_parquet", required=True, help="Human distribution parquet")
    parser.add_argument("--llm_parquet", required=True, help="LLM distributions parquet")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    human_df = pd.read_parquet(args.human_parquet)
    llm_df = pd.read_parquet(args.llm_parquet)

    dist_cols = [f"{e}_dist" for e in EMOTION_LABELS]

    # Human distributions
    human_dists = human_df[dist_cols].values
    text_ids = human_df["id"].values
    agreement_levels = human_df["agreement_level"].values

    # Process each model
    all_results = {}

    # Filter to aggregate temperature only
    if "temperature" in llm_df.columns:
        llm_agg = llm_df[llm_df["temperature"] == "all"]
    else:
        llm_agg = llm_df

    for model in llm_agg["model"].unique():
        print(f"\n=== Calibrating: {model} ===")
        model_df = llm_agg[llm_agg["model"] == model]

        # Align with human data by ID
        merged = model_df.merge(human_df[["id"]], on="id")
        if len(merged) == 0:
            print(f"  WARNING: No matching IDs for {model}")
            continue

        # Get aligned distributions
        merged_human = human_df[human_df["id"].isin(merged["id"])].sort_values("id")
        merged_llm = merged.sort_values("id")

        h_dists = merged_human[dist_cols].values
        l_dists = merged_llm[dist_cols].values
        ids = merged_human["id"].values
        agr = merged_human["agreement_level"].values

        # Normalize LLM distributions
        for i in range(len(l_dists)):
            l_dists[i] = normalize_distribution(l_dists[i])

        print(f"  N texts: {len(h_dists)}")

        # Cross-validate
        results = cross_validate_calibration(h_dists, l_dists, ids, agr, args.n_folds)
        all_results[model] = results

        # Print summary
        print(f"\n  Results for {model}:")
        for method, stats in results["summary"].items():
            if "jsd_mean" in stats:
                print(f"    {method}: JSD={stats['jsd_mean']:.3f}, rho={stats.get('entropy_rho_mean', 'N/A')}")

    # Save results
    results_path = os.path.join(args.output_dir, "calibration_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
