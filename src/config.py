"""Centralised configuration for the SAE probing project."""

import os
from pathlib import Path

import numpy as np
import torch

_base = Path("/content") if os.path.exists("/content") else Path(__file__).parent.parent

CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────
    "model_name":   "google/gemma-2-2b",
    "target_layer": 20,
    "batch_size":   8,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    # ── SAE ────────────────────────────────────────────────────────────────
    "sae_release":  "gemma-scope-2b-pt-res-canonical",
    "sae_id":       "layer_20/width_16k/canonical",
    # ── Probing ────────────────────────────────────────────────────────────
    "k_values":     [16, 128],
    "C_values":     list(np.logspace(5, -5, 10)),   # paper Appendix C.3
    # ── Data ───────────────────────────────────────────────────────────────
    "data_url":     (
        "https://raw.githubusercontent.com/sudikshyapant/sae_probing"
        "/main/data/154_athlete_sport_football.csv"
    ),
    "test_size":    0.4,
    "val_size":     0.2,
    "random_state": 42,
    # ── I/O ────────────────────────────────────────────────────────────────
    "cache_dir":    _base / "cache",
    "results_dir":  _base / "result",
}

Path(CONFIG["cache_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)
