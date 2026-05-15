"""
Convert EmoBank vLLM results (CSV from GPU server) to parquet format
compatible with analysis scripts.

Usage:
    python convert_emobank_vllm_results.py --input <csv_path> --model-name <name>
"""

import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert EmoBank vLLM CSV to parquet")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (e.g., qwen3_8b, llama3_1_8b)")
    # Default output dir is resolved relative to this script:
    #   <repo>/code/llm_inference/  ->  <repo>/data/processed/emobank/llm_results
    _default_output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                     "emobank", "llm_results")
    )
    parser.add_argument("--output-dir", type=str, default=_default_output_dir)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Convert numeric columns
    for col in ["V_pred", "A_pred", "D_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert boolean columns
    for col in ["parse_error", "api_error"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)

    valid = df[~df["parse_error"] & ~df["api_error"]]
    print(f"  Valid rows: {len(valid)}")
    print(f"  Parse errors: {df['parse_error'].sum()}")

    # Save as parquet
    safe_name = args.model_name.replace(".", "_").replace("-", "_")
    output_path = os.path.join(args.output_dir, f"emobank_{safe_name}_results.parquet")
    df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Quick stats
    for dim in ["V", "A", "D"]:
        col = f"{dim}_pred"
        print(f"  {dim}: mean={valid[col].mean():.3f}, std={valid[col].std():.3f}")


if __name__ == "__main__":
    main()
