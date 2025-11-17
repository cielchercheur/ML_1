#!/usr/bin/env python3
"""
Tutorial 1 – Unsupervised Machine Learning (HS25)

python 3.12

What this script does
---------------------
Solves the Tutorial I tasks and writes all results into a `results/` folder:
    1) Preparing the data: load, sanity-check, sort; visualize original vs interpolated spectra for main-sequence (V) stars.
    2) Dimensionality reduction (PCA): normalize (per spectrum) and standardize across samples (per wavelength),
       compute PCA, scree curve, and 2D/3D projections colored by spectral type and luminosity.
    3) Clustering: K-Means model selection (elbow, silhouette, CH, DB indices), DBSCAN grid search including
       silhouette vs. number of clusters; final 2D PCA plots colored by cluster labels; external scores (ARI, V, NMI)
       vs. true spectral type and luminosity.
    4) Outlier detection: robust Z-score (median/MAD), Isolation Forest, and Local Outlier Factor.
    5) Save lists and example plots.

Run:
$ python tut_1.py --data tut1_dataset.parquet

"""


# Standard library imports
import os                              # for filesystem paths (saving results)
import json                            # to save numeric summaries as .json files
import math                            # for ceiling in subplot grids
import argparse                        # to expose --part and --data options
from dataclasses import dataclass      # small, typed configuration container


# Numerical/plotting stack
import numpy as np                     # vectorized numerical workhorse
import pandas as pd                    # convenient table (DataFrame) utilities
import matplotlib                      # we save figures to files (no GUI required)
matplotlib.use("Agg")                  # headless backend — avoids display/GUI needs
import matplotlib.pyplot as plt        # plotting (Tutorial shows lots of figures)
from matplotlib import patches as mpatches  # for custom legend handles


# Scikit-learn: preprocessing + dimensionality reduction + clustering + metrics
from sklearn.preprocessing import StandardScaler                    # W1: standardize for PCA
from sklearn.decomposition import PCA                               # PCA
from sklearn.cluster import KMeans, DBSCAN                          # clustering
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor  # k-dist & LOF
from sklearn.ensemble import IsolationForest                        # outliers
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)

# Configuration
@dataclass
class Config:
    # Path to the parquet dataset
    data_path: str = "tut1_dataset.parquet"

    # Random seed for reproducibility (K-Means init, IF sampling, PCA whitening, etc.)
    seed: int = 42

    # Number of principal components used for clustering/outliers (retain rich variance)
    n_pcs_for_models: int = 50

    # Range of k for K-Means model selection
    kmin: int = 2
    kmax: int = 20

    # DBSCAN min_samples grid (typical neighborhood sizes).
    dbscan_min_samples_grid: tuple = (5, 10, 15, 20, 30)

    # Percentiles for eps grid boundaries (derived from k-distance curve).
    eps_lo_pct: float = 60.0
    eps_hi_pct: float = 95.0
    eps_num: int = 15

    # Outlier selection quantile (top x% are considered outliers for plots/lists)
    outlier_top_quantile: float = 0.99

    # Output directory for all results (figures and JSON summaries)
    outdir: str = "results"


# Small utilities
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):            # check existence to avoid errors on mkdir
        os.makedirs(path, exist_ok=True)    # safe even if created in parallel


def savefig(path: str) -> None:
    plt.tight_layout()                          # compact whitespace around axes
    plt.savefig(path, dpi=300)            # high-res PNG
    plt.close()                                 # free figure memory


# Data loading & sanity checks
def load_dataset(cfg: Config) -> pd.DataFrame:

    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"Dataset not found at: {cfg.data_path}\n"
            "Make sure the path is correct, or pass --data /path/to/tut1_dataset.parquet"
        )
    df = pd.read_parquet(cfg.data_path)
    return df


# PART 1
def sort_by_spectral_type(df: pd.DataFrame) -> pd.DataFrame:
    # Sort the DataFrame by spectral type in astrophysical order (O0...M9)

    spectral_classes = [f"{letter}{num}" for letter in "OBAFGKM" for num in range(10)]

    sort_dict = {s: i for i, s in enumerate(spectral_classes)}

    return df.sort_values(by=["spectral_type"], key=lambda x: x.map(sort_dict))


def assert_common_grid(df: pd.DataFrame) -> np.ndarray:
    # Grab the first wavelength grid; convert to numpy for vectorized comparisons
    w0 = np.asarray(df["common_wavelengths"].iloc[0])

    # Verify every row's grid equals the first one (exact equality because interpolation step is fixed).
    for w in df["common_wavelengths"].values:
        if not np.array_equal(np.asarray(w), w0):
            raise AssertionError("Inconsistent 'common_wavelengths' across rows.")
    return w0


def pick_representative_per_sptype_V(df: pd.DataFrame) -> pd.DataFrame:
    # Choose the median-by-L2 row per group to avoiding obvious outliers
    df_V = df[df["luminosity"] == "V"].copy()

    rows = []
    for sptype, g in df_V.groupby("spectral_type"):
        X = np.vstack(g["interpolated_fluxes"].values).astype(np.float32)
        c = X.mean(axis=0)
        idx = np.argmin(((X - c) ** 2).sum(axis=1))
        rows.append(g.iloc[idx])

    return pd.DataFrame(rows).reset_index(drop=True)


def plot_original_vs_interpolated_grid(df_rep: pd.DataFrame, outpath: str) -> None:
    # Determine subplot layout: 5 columns is a good balance for legends/axes density
    n = len(df_rep)

    ncols = 5
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(ncols * 3.2, nrows * 2.4))

    for i, (_, row) in enumerate(df_rep.iterrows()):
        ax = fig.add_subplot(nrows, ncols, i + 1)

        wl_orig = np.asarray(row["wavelength"])
        fl_orig = np.asarray(row["flux"])
        wl_common = np.asarray(row["common_wavelengths"])
        fl_interp = np.asarray(row["interpolated_fluxes"])

        # Plot: original as thin line, interpolated as slightly thicker line
        ax.plot(wl_orig, fl_orig, lw=0.8, alpha=0.7, label="original")
        ax.plot(wl_common, fl_interp, lw=1.0, alpha=0.9, label="interpolated")

        # Title shows spectral type (e.g., "G2 V")
        ax.set_title(f"{row['spectral_type']} V", fontsize=9)

        # Use a compact x-label tick formatting to avoid clutter
        if i // ncols == nrows - 1:
            ax.set_xlabel("Wavelength (Å)")
        if i % ncols == 0:
            ax.set_ylabel("Flux (arb. units)")

        # Small legend per panel keeps figure self-contained.
        ax.legend(fontsize=7, loc="best")

    savefig(outpath)


# PART 2
def minmax_per_row(X: np.ndarray) -> np.ndarray:
    # Normalize each spectrum to [0, 1]
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)

    # Add small epsilon to avoid division by zero if a spectrum is flat.
    eps = 1e-12
    X01 = (X - mins) / (maxs - mins + eps)

    return X01


def standardize_columns(X: np.ndarray) -> np.ndarray:
    # Standardize features (columns) across samples to mean=0 and std=1 before PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    return X_std


def fit_pca(X_std: np.ndarray, seed: int, nmax: int = 200) -> PCA:
    # Fit PCA with whitening (decorrelates PCs; helps K-Means) and keep up to `nmax` components
    ncomp = min(nmax, X_std.shape[1])
    pca = PCA(n_components=ncomp, whiten=True, random_state=seed)
    pca.fit(X_std)

    return pca


def plot_pca_scree(pca: PCA, outpath: str, evr_hline: float = 0.95) -> None:
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o", lw=1.0)
    plt.axhline(evr_hline, ls="--", alpha=0.5)  # helps decide if PC3 is needed (Tutorial §2.2.3)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA scree curve")

    savefig(outpath)


def numericalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Convert 'spectral_type' and 'luminosity' to integer codes for color mapping in scatter plots
    df = df.copy()
    df["spectral_type_code"] = pd.factorize(df["spectral_type"])[0]
    df["luminosity_code"] = pd.factorize(df["luminosity"])[0]
    return df


def legend_handles_from_codes(ax_scatter, df: pd.DataFrame, col_label: str, col_code: str):
    #Build legend handles reflecting the colormap encoding used in the scatter (Tutorial §2.2 hint).
    cmap = ax_scatter.cmap
    norm = ax_scatter.norm
    unique_classes = df[col_label].unique()

    handles = [
        mpatches.Patch(color=cmap(norm(df[col_code][df[col_label] == cls].iloc[0])), label=cls)
        for cls in unique_classes
    ]

    return handles


def plot_pca_scatter(projected: np.ndarray, df_codes: pd.DataFrame, outpath_prefix: str) -> None:
    # Make two PCA(2) scatter plots: colored by spectral type and by luminosity class

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(projected[:, 0], projected[:, 1], c=df_codes["spectral_type_code"], s=6, cmap="coolwarm")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA(2) colored by spectral type")

    handles = legend_handles_from_codes(sc, df_codes, "spectral_type", "spectral_type_code")
    ax.legend(handles=handles, title="Spectral Class", ncol=6, fontsize=7)

    savefig(f"{outpath_prefix}_by_sptype.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(projected[:, 0], projected[:, 1], c=df_codes["luminosity_code"], s=6, cmap="Accent")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA(2) colored by luminosity class")

    handles = legend_handles_from_codes(sc, df_codes, "luminosity", "luminosity_code")
    ax.legend(handles=handles, title="Luminosity Class", ncol=5, fontsize=7)

    savefig(f"{outpath_prefix}_by_luminosity.png")


# PART 3
def kmeans_model_selection(X: np.ndarray, kmin: int, kmax: int, seed: int) -> pd.DataFrame:
    # Sweep `k` for K-Means on features `X` and collect common internal metrics:
    #    - inertia (SSE), silhouette, Calinski–Harabasz (CH), Davies–Bouldin (DB)

    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init="auto", max_iter=500, random_state=seed)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        rows.append((k, km.inertia_, s, ch, db))

    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin"])


def plot_kmeans_selection(df_metrics: pd.DataFrame, outpath: str) -> None:
    # Plot K-Means selection curves: inertia (elbow), silhouette, CH, and DB vs k
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.plot(df_metrics["k"], df_metrics["inertia"], marker="o")
    ax.set_title("K-Means: inertia (elbow)")
    ax.set_xlabel("k"); ax.set_ylabel("SSE (inertia)")

    ax = axes[0, 1]
    ax.plot(df_metrics["k"], df_metrics["silhouette"], marker="o")
    ax.set_title("K-Means: silhouette")
    ax.set_xlabel("k"); ax.set_ylabel("silhouette score")

    ax = axes[1, 0]
    ax.plot(df_metrics["k"], df_metrics["calinski_harabasz"], marker="o")
    ax.set_title("K-Means: Calinski–Harabasz")
    ax.set_xlabel("k"); ax.set_ylabel("CH index")

    ax = axes[1, 1]
    ax.plot(df_metrics["k"], df_metrics["davies_bouldin"], marker="o")
    ax.set_title("K-Means: Davies–Bouldin (lower is better)")
    ax.set_xlabel("k"); ax.set_ylabel("DB index")

    savefig(outpath)


def choose_best_k(df_metrics: pd.DataFrame) -> int:
    # Choice of k: maximize the silhouette score
    idx = df_metrics["silhouette"].idxmax()
    return int(df_metrics.loc[idx, "k"])


def compute_kdistance_curve(X: np.ndarray, k: int = 10) -> np.ndarray:
    # Compute the k-distance curve used to eyeball "eps"
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    kth = np.sort(dists[:, -1])                     # distance to the k-th neighbor

    return kth


def dbscan_grid_silhouette_vs_clusters(X: np.ndarray,
                                       min_samples_grid=(5, 10, 15, 20, 30),
                                       eps_grid=None) -> pd.DataFrame:

    rows = []

    for ms in min_samples_grid:
        for eps in eps_grid:
            db = DBSCAN(eps=float(eps), min_samples=int(ms), n_jobs=-1).fit(X)
            labels = db.labels_
            # Count clusters (ignore label -1 for noise)
            n_cl = len(set(labels)) - (1 if -1 in labels else 0)

            # Compute silhouette only if there are at least 2 clusters and enough non-noise points
            mask = labels != -1

            if n_cl >= 2 and mask.sum() > 10:
                sil = silhouette_score(X[mask], labels[mask])
            else:
                sil = np.nan
            rows.append((int(ms), float(eps), int(n_cl), float(sil)))

    return pd.DataFrame(rows, columns=["min_samples", "eps", "n_clusters", "silhouette"])


def plot_dbscan_sil_vs_ncl(df_grid: pd.DataFrame, outpath: str) -> None:
    # Group by min_samples to produce one line per ms
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ms, g in df_grid.groupby("min_samples"):
        # Some (n_clusters) may repeat for different eps; aggregate by taking max silhouette per n_clusters:
        gg = g.groupby("n_clusters", as_index=False)["silhouette"].max().sort_values("n_clusters")
        ax.plot(gg["n_clusters"], gg["silhouette"], marker="o", label=f"min_samples={ms}")

    ax.set_xlabel("#clusters (excl. noise)")
    ax.set_ylabel("silhouette (max over eps)")
    ax.set_title("DBSCAN: silhouette vs number of clusters")
    ax.legend()

    savefig(outpath)


def choose_best_dbscan(df_grid: pd.DataFrame) -> tuple:
    #Pick the (eps, min_samples) combination with the highest silhouette score
    df_valid = df_grid.dropna(subset=["silhouette"])

    if df_valid.empty:
        idx = df_grid["n_clusters"].idxmax()
    else:
        idx = df_valid["silhouette"].idxmax()
    row = df_grid.loc[idx]

    return float(row["eps"]), int(row["min_samples"]), int(row["n_clusters"]), float(row["silhouette"])


def scatter_by_labels(X_2d: np.ndarray, labels: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)

    savefig(outpath)


def external_scores(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    # Compute ARI, V-measure, and NMI
    return {
        "ARI": float(adjusted_rand_score(true_labels, pred_labels)),
        "V_measure": float(v_measure_score(true_labels, pred_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, pred_labels))
    }


# PART 4
def robust_zscore_scores(X: np.ndarray) -> np.ndarray:
    """
    Robust per-feature z-score using median/MAD across samples; aggregate to one score per spectrum.

    We rescale by 0.6745 so that MAD matches std for normal data (consistency factor).
    Final spectrum-score = 95th percentile of |Z| across wavelengths – robust aggregation.
    """

    med = np.median(X, axis=0, keepdims=True)
    mad = np.median(np.abs(X - med), axis=0, keepdims=True) + 1e-12
    Zr = 0.6745 * (X - med) / mad

    # Aggregate absolute z magnitudes within each spectrum:
    spec_score = np.percentile(np.abs(Zr), 95, axis=1)
    return spec_score


def isolation_forest_scores(X: np.ndarray, seed: int) -> np.ndarray:
    # Isolation Forest anomaly scores (the higher, the more anomalous)
    iso = IsolationForest(
        n_estimators=400, max_samples="auto", contamination="auto", random_state=seed, n_jobs=-1
    )
    iso.fit(X)

    # score_samples: higher means less anomalous in sklearn; invert for consistency
    scores = -iso.score_samples(X)

    return scores


def lof_scores(X: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    # Local Outlier Factor raw scores (the higher, the more anomalous)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto", novelty=False, n_jobs=-1)
    labels = lof.fit_predict(X)
    raw = -lof.negative_outlier_factor_

    return raw


def top_quantile_indices(scores: np.ndarray, q: float) -> np.ndarray:
    # Return indices with score >= q-quantile (top 1%)
    thr = np.quantile(scores, q)

    return np.where(scores >= thr)[0]


def plot_outlier_spectra(wl: np.ndarray,
                         X_norm: np.ndarray,
                         indices: np.ndarray,
                         title: str, outpath: str,
                         ncols: int = 4) -> None:

    n = len(indices)

    if n == 0:
        # Create a tiny placeholder figure to indicate none found.
        plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, "No outliers selected at this threshold.", ha="center", va="center")
        plt.axis("off")
        savefig(outpath)

        return

    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(ncols * 3.0, nrows * 2.0))

    for j, idx in enumerate(indices):
        ax = fig.add_subplot(nrows, ncols, j + 1)
        ax.plot(wl, X_norm[idx], lw=0.9)
        ax.set_title(f"idx={idx}", fontsize=8)
        if j // ncols == nrows - 1:
            ax.set_xlabel("Wavelength (Å)")
        if j % ncols == 0:
            ax.set_ylabel("Norm. flux")
    fig.suptitle(title, fontsize=10)

    savefig(outpath)


def solve_part1(cfg: Config, df: pd.DataFrame) -> None:
    ensure_dir(cfg.outdir)
    df_sorted = sort_by_spectral_type(df)
    df_rep = pick_representative_per_sptype_V(df_sorted)

    plot_original_vs_interpolated_grid(df_rep, os.path.join(cfg.outdir, "part1_vstars_original_vs_interpolated.png"))


def solve_part2(cfg: Config, df: pd.DataFrame) -> dict:
    ensure_dir(cfg.outdir)
    X = np.vstack(df["interpolated_fluxes"].values).astype(np.float32)
    X01 = minmax_per_row(X)
    X_std = standardize_columns(X01)
    pca = fit_pca(X_std, seed=cfg.seed, nmax=200)
    evr = pca.explained_variance_ratio_
    cum_evr = float(np.cumsum(evr)[1])  # cumulative for first 2 PCs
    cum_evr3 = float(np.cumsum(evr)[2]) # cumulative for first 3 PCs

    plot_pca_scree(pca, os.path.join(cfg.outdir, "part2_pca_scree.png"))

    X_pca_full = pca.transform(X_std)
    X_pca2 = X_pca_full[:, :2]
    X_pca3 = X_pca_full[:, :3]
    df_codes = numericalize_labels(df)

    plot_pca_scatter(X_pca2, df_codes, os.path.join(cfg.outdir, "part2_pca2_scatter"))

    summary = {
        "explained_variance_pc1_pc2": cum_evr,
        "explained_variance_pc1_pc2_pc3": cum_evr3,
        "n_components_available": int(X_pca_full.shape[1]),
    }

    with open(os.path.join(cfg.outdir, "part2_pca_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "wl": np.asarray(df["common_wavelengths"].iloc[0]),
        "X01": X01,               # normalized spectra (for outlier plots)
        "X_std": X_std,           # standardized spectra (for PCA fit reference)
        "pca": pca,               # fitted PCA object
        "X_pca_full": X_pca_full, # all PCs (whitened)
        "X_pca2": X_pca2,         # 2D for plotting
        "df_codes": df_codes      # DataFrame with numeric label codes
    }


def solve_part3(cfg: Config, payload: dict, df: pd.DataFrame) -> None:
    ensure_dir(cfg.outdir)
    X_pca = payload["X_pca_full"][:, :cfg.n_pcs_for_models]
    X_pca2 = payload["X_pca2"]
    df_codes = payload["df_codes"]
    km_metrics = kmeans_model_selection(X_pca, cfg.kmin, cfg.kmax, cfg.seed)

    plot_kmeans_selection(km_metrics, os.path.join(cfg.outdir, "part3_kmeans_model_selection.png"))

    best_k = choose_best_k(km_metrics)

    km_best = KMeans(n_clusters=best_k, n_init="auto", max_iter=500, random_state=cfg.seed)
    km_labels = km_best.fit_predict(X_pca)
    scatter_by_labels(X_pca2, km_labels, f"K-Means (k={best_k}) on PCA(2)", os.path.join(cfg.outdir, "part3_kmeans_scatter.png"))

    kdist = compute_kdistance_curve(X_pca, k=10)

    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(kdist)), kdist, lw=1.0)
    plt.xlabel("sorted samples"); plt.ylabel("10-NN distance")
    plt.title("DBSCAN: k-distance curve (k=10)")

    savefig(os.path.join(cfg.outdir, "part3_dbscan_kdist_curve.png"))

    eps_lo = np.percentile(kdist, cfg.eps_lo_pct)
    eps_hi = np.percentile(kdist, cfg.eps_hi_pct)
    eps_grid = np.linspace(eps_lo, eps_hi, cfg.eps_num)

    db_grid = dbscan_grid_silhouette_vs_clusters(X_pca,
                                                 min_samples_grid=cfg.dbscan_min_samples_grid,
                                                 eps_grid=eps_grid)

    db_grid_path = os.path.join(cfg.outdir, "part3_dbscan_grid.csv")
    db_grid.to_csv(db_grid_path, index=False)

    plot_dbscan_sil_vs_ncl(db_grid, os.path.join(cfg.outdir, "part3_dbscan_silhouette_vs_nclusters.png"))

    eps_best, ms_best, ncl_best, sil_best = choose_best_dbscan(db_grid)

    db = DBSCAN(eps=eps_best, min_samples=ms_best, n_jobs=-1).fit(X_pca)
    db_labels = db.labels_
    scatter_by_labels(X_pca2, db_labels, f"DBSCAN (eps={eps_best:.4f},"
                                         f"min_samples={ms_best}) on PCA(2)",
                      os.path.join(cfg.outdir, "part3_dbscan_scatter.png"))

    sptype_true = df_codes["spectral_type_code"].to_numpy()
    lum_true = df_codes["luminosity_code"].to_numpy()

    scores = {
        "kmeans_vs_sptype": external_scores(sptype_true, km_labels),
        "kmeans_vs_luminosity": external_scores(lum_true, km_labels),
        "dbscan_vs_sptype": external_scores(sptype_true, db_labels),
        "dbscan_vs_luminosity": external_scores(lum_true, db_labels),
        "dbscan_best_params": {
            "eps": eps_best, "min_samples": ms_best, "n_clusters": ncl_best, "silhouette": sil_best
        },
        "kmeans_best_k": int(best_k)
    }

    with open(os.path.join(cfg.outdir, "part3_clustering_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)


def solve_part4(cfg: Config, payload: dict) -> None:
    ensure_dir(cfg.outdir)

    wl = payload["wl"]                                          # wavelength grid (common)
    X01 = payload["X01"]                                        # per-spectrum normalized spectra (for plotting)
    X_pca = payload["X_pca_full"][:, :cfg.n_pcs_for_models]     # compact features for IF/LOF

    z_scores = robust_zscore_scores(X01)
    z_top_idx = top_quantile_indices(z_scores, cfg.outlier_top_quantile)

    plot_outlier_spectra(wl, X01, z_top_idx, "Robust z-score: top 1% spectra", os.path.join(cfg.outdir, "part4_outliers_robust_zscore.png"))

    if_scores = isolation_forest_scores(X_pca, cfg.seed)
    if_top_idx = top_quantile_indices(if_scores, cfg.outlier_top_quantile)

    plot_outlier_spectra(wl, X01, if_top_idx, "Isolation Forest: top 1% spectra", os.path.join(cfg.outdir, "part4_outliers_isolation_forest.png"))

    lof_raw = lof_scores(X_pca, n_neighbors=30)
    lof_top_idx = top_quantile_indices(lof_raw, cfg.outlier_top_quantile)

    plot_outlier_spectra(wl, X01, lof_top_idx, "LOF (k=30): top 1% spectra", os.path.join(cfg.outdir, "part4_outliers_lof.png"))

    with open(os.path.join(cfg.outdir, "part4_outlier_indices.json"), "w") as f:
        json.dump({
            "robust_zscore_top1pct": z_top_idx.tolist(),
            "isolation_forest_top1pct": if_top_idx.tolist(),
            "lof_top1pct": lof_top_idx.tolist()
        }, f, indent=2)


def main():

    parser = argparse.ArgumentParser(description="HS25 Tutorial 1 – Unsupervised ML")

    parser.add_argument("--data", type=str, default="tut1_dataset.parquet", help="Path to tut1_dataset.parquet")
    parser.add_argument("--part", type=str, default="all")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save figures and JSON outputs")

    args = parser.parse_args()

    cfg = Config(data_path=args.data, outdir=args.outdir)

    # Load the dataset
    df = load_dataset(cfg)

    # Part 1
    print("[Part 1] Preparing data and plotting original vs interpolated spectra ...")

    solve_part1(cfg, df)

    print("  -> results/part1_vstars_original_vs_interpolated.png")

    # Part 2
    print("[Part 2] Running normalization, standardization, PCA and projections ...")

    payload = solve_part2(cfg, df)

    print("  -> results/part2_pca_scree.png")
    print("  -> results/part2_pca2_scatter_by_sptype.png, _by_luminosity.png")

    # Part 3
    print("[Part 3] Clustering (K-Means & DBSCAN), model selection, and external scoring ...")

    solve_part3(cfg, payload, df)

    print("  -> results/part3_kmeans_model_selection.png")
    print("  -> results/part3_kmeans_scatter.png")
    print("  -> results/part3_dbscan_kdist_curve.png")
    print("  -> results/part3_dbscan_silhouette_vs_nclusters.png")
    print("  -> results/part3_dbscan_scatter.png")
    print("  -> results/part3_clustering_scores.json")

    # Part 4
    print("[Part 4] Outlier detection (robust Z, Isolation Forest, LOF) ...")

    solve_part4(cfg, payload)

    print("  -> results/part4_outliers_robust_zscore.png")
    print("  -> results/part4_outliers_isolation_forest.png")
    print("  -> results/part4_outliers_lof.png")
    print("  -> results/part4_outlier_indices.json")

    print("Done.")


if __name__ == "__main__":
    main()
