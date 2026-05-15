"""
Convert vLLM CSV results to parquet format compatible with analysis pipeline.

Ensures column names match API inference output:
    id, model, temperature, sample_idx, raw_response, parsed_labels, error, {emotion}_pred

Usage:
    python convert_vllm_results.py --input results.csv --output results.parquet --model-name qwen3-8b
"""

import argparse
import json
import pandas as pd

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name to set (e.g. qwen3-8b, llama3.1:8b)")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Rows: {len(df)}")

    # Standardize model name
    df["model"] = args.model_name

    # Ensure error column exists
    if "error" not in df.columns:
        df["error"] = False

    # Ensure all pred columns are float
    pred_cols = [f"{emo}_pred" for emo in EMOTION_LABELS]
    for col in pred_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            print(f"  WARNING: missing column {col}, filling with 0.0")
            df[col] = 0.0

    # Ensure temperature is float
    df["temperature"] = df["temperature"].astype(float)
    df["sample_idx"] = df["sample_idx"].astype(int)

    # Select and order columns to match API output
    output_cols = [
        "id", "model", "temperature", "sample_idx",
        "raw_response", "parsed_labels", "error",
    ] + pred_cols

    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]

    print(f"Saving {args.output}...")
    df.to_parquet(args.output, index=False)
    print(f"  Done. {len(df)} rows saved.")

    # Quick validation
    print(f"\n  Models: {df['model'].unique()}")
    print(f"  Temperatures: {sorted(df['temperature'].unique())}")
    print(f"  Texts: {df['id'].nunique()}")
    print(f"  Samples per text (approx): {len(df) / df['id'].nunique():.0f}")

    # Check for errors
    n_errors = df["error"].sum() if "error" in df.columns else 0
    print(f"  Errors: {n_errors}")

    # Check for neutral-only responses
    neutral_only = (df["neutral_pred"] == 1.0) & (df[pred_cols].drop(columns=["neutral_pred"]).sum(axis=1) == 0.0)
    print(f"  Neutral-only responses: {neutral_only.sum()} ({neutral_only.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
