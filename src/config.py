"""Centralised configuration for the SAE probing project."""

import os
from pathlib import Path

import numpy as np
import torch


def _setup_dirs() -> tuple[Path, Path]:
    """Mount Google Drive (Colab only) and return (cache_dir, results_dir)."""
    in_colab = os.path.exists("/content")
    if in_colab:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive", force_remount=False)
            base = Path("/content/drive/MyDrive/sae_probing")
            print(f"Google Drive mounted — using {base}")
        except Exception:
            base = Path("/content/sae_probing")
            print("Drive unavailable — using local Colab storage.")
    else:
        base = Path(__file__).parent.parent

    cache_dir   = base / "cache"
    results_dir = base / "result"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, results_dir


_cache_dir, _results_dir = _setup_dirs()

CONFIG = {
    # Model
    "model_name":   "google/gemma-2-2b",
    "target_layer": 20,
    "batch_size":   8,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    # SAE
    "sae_release":  "gemma-scope-2b-pt-res-canonical",
    "sae_id":       "layer_20/width_16k/canonical",
    # Probing
    "k_values":     [16, 128],
    "C_values":     list(np.logspace(5, -5, 10)),
    # Data
    "data_url":     (
        "https://raw.githubusercontent.com/sudikshyapant/sae_probing"
        "/main/data/154_athlete_sport_football.csv"
    ),
    "test_size":    0.2,
    "val_size":     0.2,
    "random_state": 42,
    # I/O
    "cache_dir":    _cache_dir,
    "results_dir":  _results_dir,
}
