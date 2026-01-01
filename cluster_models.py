# Functions for k-means training and evaluation, later shift this to a separate .py file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

def _to_numpy(X):
    """Accepts np.ndarray or pd.DataFrame; returns np.ndarray (no copy if possible)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)

def train_kmeans(
    X,
    k: int,
    *,
    batch_size: int = 8192,
    n_init: str | int = "auto",
    max_iter: int = 200,
    random_state: int = 42,
    reassignment_ratio: float = 0.01,
):
    """
    Train a MiniBatchKMeans model for a given k.
    Returns fitted model and labels on X.
    """
    Xn = _to_numpy(X)

    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        reassignment_ratio=reassignment_ratio,
        verbose=0,
    )
    labels = model.fit_predict(Xn)
    return model, labels

def evaluate_k(
    X,
    labels,
    *,
    silhouette_sample_size: int = 20_000,
    random_state: int = 42,
):
    """
    Compute 3 clustering quality metrics on X given labels:
      - Silhouette (higher is better)
      - Calinski–Harabasz (higher is better)
      - Davies–Bouldin (lower is better)

    For speed, silhouette is computed on a random subset if X is large.
    """
    Xn = _to_numpy(X)
    n = Xn.shape[0]

    # Silhouette can be expensive; subsample for the metric only.
    if n > silhouette_sample_size:
        sil = silhouette_score(
            Xn, labels,
            metric="euclidean",
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
    else:
        sil = silhouette_score(Xn, labels, metric="euclidean")

    ch = calinski_harabasz_score(Xn, labels)
    db = davies_bouldin_score(Xn, labels)

    return sil, ch, db

def search_optimal_k(
    X_sub,
    k_values,
    *,
    batch_size: int = 8192,
    n_init: str | int = "auto",
    max_iter: int = 200,
    random_state: int = 19,
    silhouette_sample_size: int = 20_000,
    reassignment_ratio: float = 0.01,
):
    """
    Train k-means for each k in k_values on X_sub and compute silhouette, CH, DB.
    Returns:
      - results_df: pd.DataFrame with columns [k, silhouette, calinski_harabasz, davies_bouldin]
      - models: dict[k] -> fitted model (optional handy for later)
    """
    Xn = _to_numpy(X_sub)

    rows = []
    models = {}

    for k in k_values:
        model, labels = train_kmeans(
            Xn, k,
            batch_size=batch_size,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            reassignment_ratio=reassignment_ratio,
        )
        sil, ch, db = evaluate_k(
            Xn, labels,
            silhouette_sample_size=silhouette_sample_size,
            random_state=random_state,
        )
        rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
        models[k] = model
        print(f"k={k:>3} | silhouette={sil:.4f} | CH={ch:.1f} | DB={db:.4f}")

    results_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return results_df, models

def plot_metrics_vs_k(results_df: pd.DataFrame):
    """
    Plot all 3 metrics vs k on one figure (with two y-axes, because scales differ a lot).
    - Left axis: Silhouette and Davies–Bouldin
    - Right axis: Calinski–Harabasz
    """
    k = results_df["k"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Left axis
    ax1.plot(k, results_df["silhouette"].to_numpy(), marker="o", label="Silhouette (higher better)")
    ax1.plot(k, results_df["davies_bouldin"].to_numpy(), marker="o", label="Davies–Bouldin (lower better)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette / Davies–Bouldin")
    ax1.grid(True, alpha=0.3)

    # Right axis
    ax2 = ax1.twinx()
    ax2.plot(k, results_df["calinski_harabasz"].to_numpy(), marker="o", label="Calinski–Harabasz (higher better)")
    ax2.set_ylabel("Calinski–Harabasz")

    # One combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title("Clustering metrics vs k (choose k by elbow/plateau)")
    plt.tight_layout()
    plt.show()
# Functions for GMM training and evaluation (similar API to k-means)
# Later you can move this into a separate .py file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

def train_gmm(
    X,
    k: int,
    *,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int = 19,
):
    """
    Train a GaussianMixture model for a given k (n_components).
    Returns fitted model and hard labels on X.
    """
    Xn = _to_numpy(X)

    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        init_params="kmeans",  # good default
    )
    labels = model.fit_predict(Xn)  # hard assignment
    return model, labels

def evaluate_gmm_k(
    X,
    labels,
    model=None,
    *,
    silhouette_sample_size: int = 20_000,
    random_state: int = 42,
):
    """
    Compute clustering quality metrics on X given labels:
      - Silhouette (higher is better)
      - Calinski–Harabasz (higher is better)
      - Davies–Bouldin (lower is better)

    Additionally (optional, if model is provided):
      - AIC (lower is better)
      - BIC (lower is better)
      - Avg log-likelihood per sample (higher is better)

    For speed, silhouette is computed on a random subset if X is large.
    """
    Xn = _to_numpy(X)
    n = Xn.shape[0]

    # Silhouette can be expensive; subsample for the metric only.
    if n > silhouette_sample_size:
        sil = silhouette_score(
            Xn, labels,
            metric="euclidean",
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
    else:
        sil = silhouette_score(Xn, labels, metric="euclidean")

    ch = calinski_harabasz_score(Xn, labels)
    db = davies_bouldin_score(Xn, labels)

    out = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}

    if model is not None:
        out["aic"] = model.aic(Xn)
        out["bic"] = model.bic(Xn)
        out["avg_loglik"] = float(model.score(Xn))  # mean log-likelihood per sample

    return out

def search_optimal_k_gmm(
    X_sub,
    k_values,
    *,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int = 19,
    silhouette_sample_size: int = 20_000,
):
    """
    Train GMM for each k in k_values on X_sub and compute silhouette, CH, DB.
    Also computes BIC/AIC/log-likelihood (model-based diagnostics) which are
    especially relevant for GMM.

    Returns:
      - results_df: pd.DataFrame with columns:
          [k, silhouette, calinski_harabasz, davies_bouldin, bic, aic, avg_loglik]
      - models: dict[k] -> fitted model
    """
    Xn = _to_numpy(X_sub)

    rows = []
    models = {}

    for k in k_values:
        model, labels = train_gmm(
            Xn, k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

        metrics = evaluate_gmm_k(
            Xn, labels, model=model,
            silhouette_sample_size=silhouette_sample_size,
            random_state=random_state,
        )

        row = {"k": k, **metrics}
        rows.append(row)
        models[k] = model

        print(
            f"k={k:>3} | silhouette={metrics['silhouette']:.4f} | "
            f"CH={metrics['calinski_harabasz']:.1f} | DB={metrics['davies_bouldin']:.4f} | "
            f"BIC={metrics['bic']:.1f} | AIC={metrics['aic']:.1f}"
        )

    results_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return results_df, models

def plot_metrics_vs_k_gmm(results_df: pd.DataFrame, show_bic_aic: bool = True):
    """
    Plot clustering metrics vs k for GMM.
    - Left axis: Silhouette and Davies–Bouldin
    - Right axis: Calinski–Harabasz
    Optionally overlays BIC/AIC in a separate figure (recommended for GMM).
    """
    k = results_df["k"].to_numpy()

    # --- Figure 1: Silhouette/DB + CH (same style as your k-means plot)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(k, results_df["silhouette"].to_numpy(), marker="o", label="Silhouette (higher better)")
    ax1.plot(k, results_df["davies_bouldin"].to_numpy(), marker="o", label="Davies–Bouldin (lower better)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette / Davies–Bouldin")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(k, results_df["calinski_harabasz"].to_numpy(), marker="o", label="Calinski–Harabasz (higher better)")
    ax2.set_ylabel("Calinski–Harabasz")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title("GMM clustering metrics vs k (choose k by consensus)")
    plt.tight_layout()
    plt.show()

    # --- Figure 2: BIC/AIC (model selection diagnostics for GMM)
    if show_bic_aic and ("bic" in results_df.columns) and ("aic" in results_df.columns):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(k, results_df["bic"].to_numpy(), marker="o", label="BIC (lower better)")
        ax.plot(k, results_df["aic"].to_numpy(), marker="o", label="AIC (lower better)")
        ax.set_xlabel("k")
        ax.set_ylabel("Criterion value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.title("GMM model selection: BIC/AIC vs k")
        plt.tight_layout()
        plt.show()
# ============================================================
# Problem 2: Feature importance / feature selection utilities
# ============================================================

def _get_feature_names(X):
    """Return feature names if X is a DataFrame, else x0, x1, ..."""
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return [f"x{j}" for j in range(_to_numpy(X).shape[1])]


def kmeans_feature_importance_anova(
    X,
    k: int,
    *,
    batch_size: int = 8192,
    n_init: str | int = "auto",
    max_iter: int = 200,
    random_state: int = 19,
    reassignment_ratio: float = 0.01,
):
    """
    K-means feature importance via one-way ANOVA F-statistic (feature-wise).
    Steps:
      1) Fit K-means and get hard labels.
      2) For each feature j compute:
           SS_B,j = sum_k n_k (mu_kj - mu_j)^2
           SS_W,j = sum_k sum_{i in Ck} (x_ij - mu_kj)^2
           MS_B,j = SS_B,j / (K-1)
           MS_W,j = SS_W,j / (n-K)
           F_j    = MS_B,j / MS_W,j
    Returns:
      importance_df (sorted by F desc), labels, model
    """
    Xn = _to_numpy(X)
    feat_names = _get_feature_names(X)

    model, labels = train_kmeans(
        Xn, k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        reassignment_ratio=reassignment_ratio,
    )
    labels = np.asarray(labels)
    n, d = Xn.shape

    # global mean per feature
    mu = Xn.mean(axis=0)

    F_vals = np.zeros(d, dtype=float)

    # Precompute cluster indices
    clusters = np.unique(labels)
    # degrees of freedom
    df_between = max(k - 1, 1)
    df_within = max(n - k, 1)

    for j in range(d):
        ss_b = 0.0
        ss_w = 0.0
        for ck in clusters:
            idx = (labels == ck)
            nk = int(idx.sum())
            if nk == 0:
                continue
            xk = Xn[idx, j]
            mu_k = xk.mean()
            ss_b += nk * (mu_k - mu[j]) ** 2
            ss_w += np.sum((xk - mu_k) ** 2)

        ms_b = ss_b / df_between
        ms_w = ss_w / df_within if ss_w > 0 else np.nan
        F_vals[j] = ms_b / ms_w if (ms_w is not None and ms_w > 0) else np.nan

    importance_df = (
        pd.DataFrame({"feature": feat_names, "anova_F": F_vals})
        .sort_values("anova_F", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df, labels, model


def kmeans_feature_importance_silhouette_drop(
    X,
    k: int,
    *,
    n_repeats: int = 5,
    silhouette_sample_size: int = 20000,
    batch_size: int = 8192,
    n_init: str | int = "auto",
    max_iter: int = 200,
    random_state: int = 19,
    reassignment_ratio: float = 0.01,
):
    """
    K-means feature importance via permutation sensitivity:
      - Fit K-means once, keep labels fixed.
      - Compute baseline silhouette on X with labels.
      - For each feature j: permute column j (n_repeats times),
        recompute silhouette on permuted X with same labels.
      - Importance = average silhouette drop.

    Returns:
      importance_df (sorted by silhouette_drop desc), baseline_silhouette, labels, model
    """
    Xn = _to_numpy(X)
    feat_names = _get_feature_names(X)

    model, labels = train_kmeans(
        Xn, k,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        reassignment_ratio=reassignment_ratio,
    )

    # baseline (reuse your existing metric helper)
    base_sil, _, _ = evaluate_k(
        Xn, labels,
        silhouette_sample_size=silhouette_sample_size,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    n, d = Xn.shape
    drops = np.zeros(d, dtype=float)

    for j in range(d):
        drop_reps = []
        for _ in range(n_repeats):
            Xp = Xn.copy()
            perm_idx = rng.permutation(n)
            Xp[:, j] = Xp[perm_idx, j]
            sil_p, _, _ = evaluate_k(
                Xp, labels,
                silhouette_sample_size=silhouette_sample_size,
                random_state=random_state,
            )
            drop_reps.append(base_sil - sil_p)
        drops[j] = float(np.mean(drop_reps))

    importance_df = (
        pd.DataFrame({"feature": feat_names, "silhouette_drop": drops})
        .sort_values("silhouette_drop", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df, float(base_sil), labels, model


def gmm_feature_importance_bic_drop(
    X,
    k: int,
    *,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int = 19,
    n_repeats: int = 5,
    sample_size: int = 20_000,
    fit_on_subset: bool = True,
):
    """
    GMM feature importance via permutation sensitivity using BIC, evaluated on a subset.

    Strategy:
      - Subsample up to `sample_size` rows for speed.
      - Fit GMM either on the subset (fit_on_subset=True) or on full data (False).
      - Compute baseline BIC on the subset.
      - For each feature j: permute column j within the subset, recompute BIC using the
        same fitted model, and record average BIC increase.

    Returns:
      importance_df (sorted by bic_increase desc), baseline_bic, labels_subset, model, subset_idx
    """
    Xn_full = _to_numpy(X)
    feat_names = _get_feature_names(X)

    rng = np.random.default_rng(random_state)
    n_full = Xn_full.shape[0]

    # Choose subset indices
    if sample_size is not None and n_full > sample_size:
        subset_idx = rng.choice(n_full, size=sample_size, replace=False)
        Xn = Xn_full[subset_idx].copy()
    else:
        subset_idx = None
        Xn = Xn_full.copy()

    # Fit model
    if fit_on_subset:
        X_fit = Xn
    else:
        X_fit = Xn_full

    model, _labels_fit = train_gmm(
        X_fit, k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    # Labels for the subset (for optional downstream plots; not required for BIC)
    labels_subset = model.predict(Xn)

    # Baseline BIC on subset
    base_bic = float(model.bic(Xn))

    # Permutation BIC increases on subset
    n_sub, d = Xn.shape
    inc = np.zeros(d, dtype=float)

    for j in range(d):
        inc_reps = []
        for _ in range(n_repeats):
            Xp = Xn.copy()
            perm_idx = rng.permutation(n_sub)
            Xp[:, j] = Xp[perm_idx, j]
            bic_p = float(model.bic(Xp))
            inc_reps.append(bic_p - base_bic)
        inc[j] = float(np.mean(inc_reps))

    importance_df = (
        pd.DataFrame({"feature": feat_names, "bic_increase": inc})
        .sort_values("bic_increase", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df, base_bic, labels_subset, model, subset_idx

