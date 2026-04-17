"""Quiver-of-Arrows evaluation logic (paper Section 2.4 & 3.2)."""

import numpy as np

from probes import (probe_p1_logreg, probe_p2_pca, probe_p3_knn,
                    select_top_k_latents, train_sae_probe)


def best_in_quiver(results: dict) -> tuple[str, dict]:
    """Return the (method_name, result_dict) with the highest val AUC."""
    valid = {k: v for k, v in results.items() if v is not None}
    best  = max(valid, key=lambda m: valid[m]["val_auc"])
    return best, valid[best]


def run_all_probes(
    X_tr, y_tr, Z_tr,
    X_va, y_va, Z_va,
    X_te, y_te, Z_te,
    C_vals: list,
    sae_k: int = 128,
) -> dict:
    """Train every probe in the toolkit and return all results.

    Runs {LogReg L2, PCA+LogReg, KNN} plus the SAE probe (k=``sae_k``).

    Returns
    -------
    dict  {method_name: result_dict}
          where each result_dict has keys ``val_auc``, ``test_auc``, ``best_hp``.
          Returns an empty dict when ``y_tr`` has fewer than 2 classes.
    """
    if len(np.unique(y_tr)) < 2:
        return {}

    results: dict = {}

    for name, fn in [
        ("logreg", lambda: probe_p1_logreg(X_tr, y_tr, X_va, y_va, X_te, y_te, C_vals)),
        ("pca",    lambda: probe_p2_pca   (X_tr, y_tr, X_va, y_va, X_te, y_te)),
        ("knn",    lambda: probe_p3_knn   (X_tr, y_tr, X_va, y_va, X_te, y_te)),
    ]:
        try:
            results[name] = fn()
        except Exception:
            pass

    try:
        k_actual = min(sae_k, Z_tr.shape[1])
        feat_idx = select_top_k_latents(Z_tr, y_tr, k_actual)
        results[f"sae_k{sae_k}"] = train_sae_probe(
            Z_tr[:, feat_idx], y_tr,
            Z_va[:, feat_idx], y_va,
            Z_te[:, feat_idx], y_te,
            C_vals,
        )
    except Exception:
        pass

    return results


def quiver_table(probe_results: dict) -> dict:
    """Build the SAE-quiver vs non-SAE-quiver summary from ``run_all_probes`` output.

    Splits probes into *baseline* (logreg, pca, knn) and *full* (all including SAE)
    groups, picks the best in each by val AUC, and returns a summary dict.

    Returns
    -------
    {
        "all"    : probe_results,        # pass-through for convenience
        "no_sae" : {"method": str, "val_auc": float, "test_auc": float},
        "sae"    : {"method": str, "val_auc": float, "test_auc": float},
    }
    Returns an empty dict when ``probe_results`` is empty.
    """
    if not probe_results:
        return {}

    base = {k: v for k, v in probe_results.items() if not k.startswith("sae_")}

    def _pick(pool: dict) -> dict:
        if not pool:
            return {}
        name, res = best_in_quiver(pool)
        return {"method": name, "val_auc": res["val_auc"], "test_auc": res["test_auc"]}

    return {
        "all":    probe_results,
        "no_sae": _pick(base),
        "sae":    _pick(probe_results),
    }


def run_quiver(
    X_tr, y_tr, Z_tr,
    X_va, y_va, Z_va,
    X_te, y_te, Z_te,
    C_vals: list,
    include_sae: bool = True,
    sae_k: int = 128,
) -> tuple[float, float]:
    """Run the quiver-of-arrows rule and return (no_sae_test_auc, sae_test_auc).

    Thin wrapper around ``run_all_probes`` + ``quiver_table`` for use in
    regime loops where only the two scalar AUCs are needed.

    Parameters
    ----------
    include_sae : if False, both returned values equal the non-SAE quiver AUC.
    sae_k       : number of SAE latents for the probe.
    """
    probes = run_all_probes(
        X_tr, y_tr, Z_tr, X_va, y_va, Z_va, X_te, y_te, Z_te,
        C_vals, sae_k=sae_k,
    )
    if not probes:
        return np.nan, np.nan

    base = {k: v for k, v in probes.items() if not k.startswith("sae_")}
    if not base:
        return np.nan, np.nan

    _, base_res = best_in_quiver(base)
    no_sae_auc  = base_res["test_auc"]

    if not include_sae:
        return no_sae_auc, no_sae_auc

    _, all_res = best_in_quiver(probes)
    return no_sae_auc, all_res["test_auc"]
