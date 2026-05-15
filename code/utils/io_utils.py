"""
I/O utility functions for the human_vs_ai_emotion project.

Provides consistent file loading/saving with logging and error handling.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load experiment configuration from YAML file.

    If no path is given, looks for the default config location.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "experiment_config.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data as JSON with consistent formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    print(f"  [SAVED] {path}")


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame as Parquet file."""
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)
    print(f"  [SAVED] {path} ({len(df)} rows)")


def load_parquet(path: str) -> pd.DataFrame:
    """Load DataFrame from a Parquet file."""
    df = pd.read_parquet(path)
    print(f"  [LOADED] {path} ({len(df)} rows)")
    return df


def save_numpy(arr: np.ndarray, path: str) -> None:
    """Save NumPy array to .npy file."""
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)
    print(f"  [SAVED] {path} (shape={arr.shape})")


def load_numpy(path: str) -> np.ndarray:
    """Load NumPy array from .npy file."""
    arr = np.load(path)
    print(f"  [LOADED] {path} (shape={arr.shape})")
    return arr
