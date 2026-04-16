"""Quiver-of-Arrows evaluation logic (paper Section 2.4 & 3.2)."""

import numpy as np

from probes import (probe_p1_logreg, probe_p2_pca, probe_p3_knn,
                    select_top_k_latents, train_sae_probe)


def best_in_quiver(results: dict) -> tuple[str, dict]:
    """Return the (method_name, result_dict) with the highest val AUC."""
    valid = {k: v for k, v in results.items() if v is not None}
    best  = max(valid, key=lambda m: valid[m]["val_auc"])
    return best, valid[best]


def run_quiver(X_tr, y_tr, Z_tr,
               X_va, y_va, Z_va,
               X_te, y_te, Z_te,
               C_vals: list,
               include_sae: bool = True,
               sae_k: int = 128) -> tuple[float, float]:
    """Run {LogReg, PCA, KNN} ± SAE probe and apply the quiver-of-arrows rule.

    Both quivers pick the method with the highest **val** AUC and report its
    **test** AUC.

    Parameters
    ----------
    include_sae : if True, the SAE quiver adds the SAE probe to the toolkit.
    sae_k       : number of SAE latents for the probe.

    Returns
    -------
    (test_auc_no_sae_quiver, test_auc_sae_quiver)
    If ``include_sae=False``, both values are equal (same quiver).
    """
    if len(np.unique(y_tr)) < 2:
        return np.nan, np.nan

    methods: dict = {}

    # Baseline probes ---------------------------------------------------------
    for name, fn in [
        ("logreg", lambda: probe_p1_logreg(X_tr, y_tr, X_va, y_va, X_te, y_te, C_vals)),
        ("pca",    lambda: probe_p2_pca   (X_tr, y_tr, X_va, y_va, X_te, y_te)),
        ("knn",    lambda: probe_p3_knn   (X_tr, y_tr, X_va, y_va, X_te, y_te)),
    ]:
        try:
            methods[name] = fn()
        except Exception:
            pass

    if not methods:
        return np.nan, np.nan

    _, best_base_res = best_in_quiver(methods)
    no_sae_auc = best_base_res["test_auc"]

    if not include_sae:
        return no_sae_auc, no_sae_auc

    # SAE probe ---------------------------------------------------------------
    try:
        k_actual = min(sae_k, Z_tr.shape[1])
        feat_idx = select_top_k_latents(Z_tr, y_tr, k_actual)
        methods[f"sae_k{sae_k}"] = train_sae_probe(
            Z_tr[:, feat_idx], y_tr,
            Z_va[:, feat_idx], y_va,
            Z_te[:, feat_idx], y_te,
            C_vals,
        )
    except Exception:
        pass

    _, best_all_res = best_in_quiver(methods)
    sae_auc = best_all_res["test_auc"]

    return no_sae_auc, sae_auc
