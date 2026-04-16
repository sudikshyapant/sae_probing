"""Dataset loading, train/val/test splitting, and regime data helpers."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Loading & splitting ───────────────────────────────────────────────────────

def load_dataset(data_url: str) -> pd.DataFrame:
    """Download (or read) the CSV dataset and return a DataFrame."""
    return pd.read_csv(data_url)


def make_splits(df: pd.DataFrame, test_size: float = 0.4,
                val_size: float = 0.2, random_state: int = 42):
    """Stratified train / val / test split.

    Returns
    -------
    X_train, X_val, X_test : list[str]   prompt strings
    y_train, y_val, y_test : np.ndarray  integer labels
    """
    prompts = df["prompt"].tolist()
    labels  = df["target"].tolist()

    X_pool, X_test, y_pool, y_test = train_test_split(
        prompts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        test_size=val_size,
        stratify=y_pool,
        random_state=random_state,
    )

    return (
        X_train, X_val, X_test,
        np.array(y_train), np.array(y_val), np.array(y_test),
    )


# ── Regime helpers ────────────────────────────────────────────────────────────

def subsample_stratified(X_act: np.ndarray, Z: np.ndarray, y: np.ndarray,
                         n: int, rng: np.random.RandomState | None = None):
    """Subsample *n* examples with approximate class balance.

    Parameters
    ----------
    X_act : (N, d_model)  dense activations
    Z     : (N, d_sae)    SAE latents
    y     : (N,)          integer labels
    n     : target number of examples (capped at N)

    Returns
    -------
    X_act[idx], Z[idx], y[idx]
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = min(int(n), len(y))
    classes = np.unique(y)

    if n < len(classes):
        idx = rng.choice(len(y), n, replace=False)
    else:
        parts = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            n_cls = max(1, round(n * len(cls_idx) / len(y)))
            n_cls = min(n_cls, len(cls_idx))
            parts.append(rng.choice(cls_idx, n_cls, replace=False))
        idx = np.concatenate(parts)
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        elif len(idx) < n:
            pool  = np.setdiff1d(np.arange(len(y)), idx)
            extra = rng.choice(pool, min(n - len(idx), len(pool)), replace=False)
            idx   = np.concatenate([idx, extra])

    rng.shuffle(idx)
    return X_act[idx], Z[idx], y[idx]


def make_imbalanced(X_act: np.ndarray, Z: np.ndarray, y: np.ndarray,
                    ratio: float, rng: np.random.RandomState | None = None):
    """Return the largest subset with positive-class fraction = *ratio* = n1/n.

    Both training and test sets should be passed through this function
    independently with the same ratio (Section 3.2 of the paper).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    # largest n such that both classes can be represented
    n_avail = int(min(len(idx_pos) / ratio, len(idx_neg) / (1 - ratio)))
    n1 = max(1, int(round(n_avail * ratio)))
    n0 = max(1, n_avail - n1)
    n1 = min(n1, len(idx_pos))
    n0 = min(n0, len(idx_neg))
    idx_p = rng.choice(idx_pos, n1, replace=False)
    idx_n = rng.choice(idx_neg, n0, replace=False)
    idx   = np.concatenate([idx_p, idx_n])
    rng.shuffle(idx)
    return X_act[idx], Z[idx], y[idx]


def corrupt_labels(y: np.ndarray, fraction: float,
                   rng: np.random.RandomState | None = None) -> np.ndarray:
    """Randomly flip *fraction* of binary labels (in-place copy)."""
    if rng is None:
        rng = np.random.RandomState(42)
    y_noisy = y.copy()
    n_flip  = int(len(y) * fraction)
    if n_flip > 0:
        flip_idx          = rng.choice(len(y), n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy
