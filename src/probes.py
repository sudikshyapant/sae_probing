"""All probing methods: SAE probe + five baselines (p1–p5).

Each function takes (X_tr, y_tr, X_va, y_va, X_te, y_te, ...) and returns
a dict with keys ``val_auc``, ``test_auc``, ``best_hp``.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm


# SAE probe

def select_top_k_latents(Z_train: np.ndarray, y_train: np.ndarray,
                          k: int) -> np.ndarray:
    """Return indices of the k SAE latents with the largest mean-activation
    difference between classes (Equation 1 of the paper)."""
    mean1 = Z_train[y_train == 1].mean(axis=0)
    mean0 = Z_train[y_train == 0].mean(axis=0)
    delta = np.abs(mean1 - mean0)
    return np.argsort(delta)[-k:]


def train_sae_probe(Z_tr, y_tr, Z_va, y_va, Z_te, y_te,
                    C_values: list) -> dict:
    """SAE logistic-regression probe with L1 regularisation (paper Appendix C.3).

    C is chosen by maximum val AUC over the supplied grid.
    """
    best_C, best_val_auc, best_clf = None, -1.0, None
    for C in tqdm(C_values, desc="SAE probe C search", leave=False):
        clf = LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                 max_iter=10_000, random_state=42)
        clf.fit(Z_tr, y_tr)
        val_auc = roc_auc_score(y_va, clf.predict_proba(Z_va)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_C, best_clf = val_auc, C, clf

    test_auc = roc_auc_score(y_te, best_clf.predict_proba(Z_te)[:, 1])
    return {"best_hp": {"C": best_C}, "val_auc": best_val_auc,
            "test_auc": test_auc, "model": best_clf}


# Baseline probes

def probe_p1_logreg(X_tr, y_tr, X_va, y_va, X_te, y_te,
                    C_values: list) -> dict:
    """p1 — Logistic Regression (L2) on StandardScaled activations."""
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_te_s  = scaler.transform(X_te)

    best_C, best_val_auc, best_clf = None, -1.0, None
    for C in tqdm(C_values, desc="p1 C search", leave=False):
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                 max_iter=1_000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        val_auc = roc_auc_score(y_va, clf.predict_proba(X_va_s)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_C, best_clf = val_auc, C, clf

    test_auc = roc_auc_score(y_te, best_clf.predict_proba(X_te_s)[:, 1])
    return {"best_hp": {"C": best_C}, "val_auc": best_val_auc, "test_auc": test_auc}


def probe_p2_pca(X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    """p2 — PCA dimensionality reduction + unregularised LogReg."""
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_te_s  = scaler.transform(X_te)

    max_n   = min(100, X_tr_s.shape[0], X_tr_s.shape[1])
    n_range = np.unique(np.logspace(0, np.log10(max(max_n, 1)), 10).astype(int))

    best_n, best_val_auc, best_clf, best_pca = None, -1.0, None, None
    for n in tqdm(n_range, desc="p2 n_components search", leave=False):
        pca = PCA(n_components=int(n)).fit(X_tr_s)
        clf = LogisticRegression(C=1e12, penalty="l2", solver="lbfgs",
                                 max_iter=2_000, random_state=42)
        clf.fit(pca.transform(X_tr_s), y_tr)
        val_auc = roc_auc_score(y_va, clf.predict_proba(pca.transform(X_va_s))[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_n, best_clf, best_pca = val_auc, n, clf, pca

    test_auc = roc_auc_score(y_te, best_clf.predict_proba(best_pca.transform(X_te_s))[:, 1])
    return {"best_hp": {"n_components": int(best_n)}, "val_auc": best_val_auc,
            "test_auc": test_auc}


def probe_p3_knn(X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    """p3 — K-Nearest Neighbours on StandardScaled activations."""
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_te_s  = scaler.transform(X_te)

    max_k   = min(100, X_tr_s.shape[0] - 1)
    k_range = np.unique(np.logspace(0, np.log10(max(max_k, 1)), 10).astype(int))

    best_k, best_val_auc, best_clf = None, -1.0, None
    for k in tqdm(k_range, desc="p3 k search", leave=False):
        clf = KNeighborsClassifier(n_neighbors=int(k))
        clf.fit(X_tr_s, y_tr)
        val_auc = roc_auc_score(y_va, clf.predict_proba(X_va_s)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_k, best_clf = val_auc, k, clf

    test_auc = roc_auc_score(y_te, best_clf.predict_proba(X_te_s)[:, 1])
    return {"best_hp": {"k": int(best_k)}, "val_auc": best_val_auc, "test_auc": test_auc}


def probe_p4_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te,
                     n_iter: int = 10, random_state: int = 42) -> dict | None:
    """p4 — XGBoost with 10-sample random HP search (no feature scaling)."""
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        print("xgboost not installed — skipping p4.  Run: pip install xgboost")
        return None

    rng = np.random.RandomState(random_state)
    best_val_auc, best_clf, best_hp = -1.0, None, None

    for _ in tqdm(range(n_iter), desc="p4 XGBoost search", leave=False):
        hp = {
            "n_estimators":     int(rng.randint(50, 251)),
            "max_depth":        int(rng.randint(2, 6)),
            "learning_rate":    float(np.exp(rng.uniform(np.log(0.001), np.log(0.1)))),
            "subsample":        float(rng.uniform(0.7, 1.0)),
            "colsample_bytree": float(rng.uniform(0.7, 1.0)),
            "reg_alpha":        float(np.exp(rng.uniform(np.log(0.001), np.log(10)))),
            "reg_lambda":       float(np.exp(rng.uniform(np.log(0.001), np.log(10)))),
            "min_child_weight": int(rng.randint(1, 10)),
        }
        clf = XGBClassifier(**hp, eval_metric="logloss",
                            random_state=42, verbosity=0)
        clf.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_clf, best_hp = val_auc, clf, hp

    test_auc = roc_auc_score(y_te, best_clf.predict_proba(X_te)[:, 1])
    return {"best_hp": best_hp, "val_auc": best_val_auc, "test_auc": test_auc}


def probe_p5_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te,
                 n_iter: int = 10, random_state: int = 42) -> dict:
    """p5 — MLP (ReLU + Adam) with random HP search on StandardScaled activations."""
    rng    = np.random.RandomState(random_state)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    best_val_auc, best_clf, best_hp = -1.0, None, None
    for _ in tqdm(range(n_iter), desc="p5 MLP search", leave=False):
        depth  = int(rng.choice([1, 2, 3]))
        width  = int(rng.choice([16, 32, 64]))
        lr     = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-2))))
        alpha  = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))))
        hidden = tuple([width] * depth)

        clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                            solver="adam", learning_rate_init=lr,
                            alpha=alpha, max_iter=500, random_state=42)
        clf.fit(X_tr_s, y_tr)
        val_auc = roc_auc_score(y_va, clf.predict_proba(X_va_s)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc, best_clf = val_auc, clf
            best_hp = {"hidden": hidden, "lr": lr, "alpha": alpha}

    test_auc = roc_auc_score(y_te, best_clf.predict_proba(X_te_s)[:, 1])
    return {"best_hp": best_hp, "val_auc": best_val_auc, "test_auc": test_auc}
