"""Centralised configuration for the SAE probing project."""

import os
from pathlib import Path

import numpy as np
import torch

_base = Path("/content") if os.path.exists("/content") else Path(__file__).parent.parent

# Google Drive paths (used when Drive is mounted in Colab)
_gdrive_root = Path("/content/drive/MyDrive/sae_probing")
_gdrive_mounted = Path("/content/drive/MyDrive").exists()

_cache_dir   = _gdrive_root / "cache"   if _gdrive_mounted else _base / "cache"
_results_dir = _gdrive_root / "result"  if _gdrive_mounted else _base / "result"

if _gdrive_mounted:
    print(f"Google Drive detected — using Drive dirs: {_gdrive_root}")

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
    "C_values":     list(np.logspace(5, -5, 10)),   # paper Appendix C.3
    # Data
    "data_url":     (
        "https://raw.githubusercontent.com/sudikshyapant/sae_probing"
        "/main/data/154_athlete_sport_football.csv"
    ),
    "test_size":    0.4,
    "val_size":     0.2,
    "random_state": 42,
    # I/O
    "cache_dir":    _cache_dir,
    "results_dir":  _results_dir,
    # Google Drive
    "gdrive_mounted": _gdrive_mounted,
}

Path(CONFIG["cache_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)
