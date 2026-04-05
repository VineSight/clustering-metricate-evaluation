"""
Clustering metric implementations.

This module contains all 34 metric functions extracted from the research notebook.
Each metric function follows the signature:
    metric_name(X, labels, **precomputed) -> float

The precomputed kwargs allow sharing expensive computations across metrics.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix

# =============================================================================
# Shared Precomputation Functions
# =============================================================================


def compute_scatter_matrices(X: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute within-cluster (W), between-cluster (B), total (T) scatter matrices,
    and per-cluster within-scatter matrices.

    Returns:
        W: Within-cluster scatter matrix (d×d)
        B: Between-cluster scatter matrix (d×d)
        T: Total scatter matrix (d×d)
        W_per_cluster: Dict mapping label -> within-scatter matrix
        global_mean: Global centroid (d,)
    """
    n, d = X.shape
    global_mean = X.mean(axis=0)
    T = (X - global_mean).T @ (X - global_mean)

    unique_labels = np.unique(labels)
    W = np.zeros((d, d))
    B = np.zeros((d, d))
    W_per_cluster = {}

    for label in unique_labels:
        mask = labels == label
        Xc = X[mask]
        nj = Xc.shape[0]
        cj = Xc.mean(axis=0)
        diff = Xc - cj
        Wj = diff.T @ diff
        W += Wj
        W_per_cluster[label] = Wj
        diff_bc = (cj - global_mean).reshape(-1, 1)
        B += nj * (diff_bc @ diff_bc.T)

    return W, B, T, W_per_cluster, global_mean


def compute_cluster_stats(X: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute centroids, sizes, per-cluster WGSS, total WGSS, BGSS, TSS.

    Returns:
        centroids: Dict mapping label -> centroid vector
        sizes: Dict mapping label -> cluster size
        wgss_per_cluster: Dict mapping label -> within-cluster sum of squares
        total_wgss: Total within-group sum of squares
        total_bgss: Total between-group sum of squares
        total_tss: Total sum of squares
        global_mean: Global centroid
    """
    unique_labels = np.unique(labels)
    global_mean = X.mean(axis=0)
    centroids = {}
    sizes = {}
    wgss_per_cluster = {}
    total_wgss = 0.0
    total_bgss = 0.0
    total_tss = np.sum((X - global_mean) ** 2)

    for label in unique_labels:
        mask = labels == label
        Xc = X[mask]
        cj = Xc.mean(axis=0)
        centroids[label] = cj
        sizes[label] = len(Xc)
        wgss_j = np.sum((Xc - cj) ** 2)
        wgss_per_cluster[label] = wgss_j
        total_wgss += wgss_j
        total_bgss += len(Xc) * np.sum((cj - global_mean) ** 2)

    return centroids, sizes, wgss_per_cluster, total_wgss, total_bgss, total_tss, global_mean


def compute_concordance_pairs(dist_matrix: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute concordant (S+) and discordant (S-) pairs for Gamma/Tau/G-plus.
    S+ = within < between, S- = within > between.

    Returns:
        S_plus: Number of concordant pairs
        S_minus: Number of discordant pairs
        N_w: Number of within-cluster pairs
        N_b: Number of between-cluster pairs
    """
    n = len(labels)
    triu_i, triu_j = np.triu_indices(n, k=1)
    all_dists = dist_matrix[triu_i, triu_j]
    same_cluster = labels[triu_i] == labels[triu_j]

    within_dists = all_dists[same_cluster]
    between_dists = np.sort(all_dists[~same_cluster])

    S_plus = 0
    S_minus = 0
    for w in within_dists:
        idx_right = np.searchsorted(between_dists, w, side="right")
        S_plus += len(between_dists) - idx_right
        idx_left = np.searchsorted(between_dists, w, side="left")
        S_minus += idx_left

    return S_plus, S_minus, len(within_dists), len(between_dists)


def precompute_all(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Precompute all shared values for metric calculation.

    Returns:
        Dictionary with all precomputed values that can be passed to metric functions.
    """
    centroids, sizes, wgss_pc, wgss, bgss, tss, gmean = compute_cluster_stats(X, labels)
    W, B, T, W_pc, _ = compute_scatter_matrices(X, labels)
    dm = pairwise_distances(X)
    Sp, Sm, Nw, Nb = compute_concordance_pairs(dm, labels)

    return {
        "centroids": centroids,
        "sizes": sizes,
        "wgss_per_cluster": wgss_pc,
        "total_wgss": wgss,
        "total_bgss": bgss,
        "total_tss": tss,
        "global_mean": gmean,
        "W": W,
        "B": B,
        "T": T,
        "W_per_cluster": W_pc,
        "dist_matrix": dm,
        "S_plus": Sp,
        "S_minus": Sm,
        "N_w": Nw,
        "N_b": Nb,
    }


# =============================================================================
# Original Internal Metrics (6)
# =============================================================================


def silhouette(X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    """Silhouette Score: mean intra-cluster distance vs nearest-cluster distance."""
    return silhouette_score(X, labels)


def davies_bouldin(X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    """Davies-Bouldin Index: average similarity ratio of clusters."""
    return davies_bouldin_score(X, labels)


def calinski_harabasz(X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    """Calinski-Harabasz Index: between-cluster to within-cluster dispersion ratio."""
    return calinski_harabasz_score(X, labels)


def dunn_index(
    X: np.ndarray, labels: np.ndarray, dist_matrix: np.ndarray = None, **kwargs
) -> float:
    """Dunn Index: ratio of min inter-cluster distance to max intra-cluster diameter."""
    if dist_matrix is None:
        dist_matrix = pairwise_distances(X)

    unique_labels = np.unique(labels)

    # Min inter-cluster distance (between any two points in different clusters)
    min_inter = np.inf
    for i, li in enumerate(unique_labels):
        for lj in unique_labels[i + 1 :]:
            mask_i = labels == li
            mask_j = labels == lj
            inter_dists = dist_matrix[np.ix_(mask_i, mask_j)]
            min_inter = min(min_inter, inter_dists.min())

    # Max intra-cluster diameter
    max_intra = 0.0
    for li in unique_labels:
        mask = labels == li
        if mask.sum() > 1:
            intra_dists = dist_matrix[np.ix_(mask, mask)]
            max_intra = max(max_intra, intra_dists.max())

    return min_inter / max_intra if max_intra > 0 else 0.0


def sse(X: np.ndarray, labels: np.ndarray, total_wgss: float = None, **kwargs) -> float:
    """Sum of Squared Errors (within-cluster sum of squares)."""
    if total_wgss is not None:
        return total_wgss

    wgss = 0.0
    for label in np.unique(labels):
        mask = labels == label
        Xc = X[mask]
        centroid = Xc.mean(axis=0)
        wgss += np.sum((Xc - centroid) ** 2)
    return wgss


def new_correlation_index(X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    """NCI: Pearson correlation between point-to-centroid and centroid-to-global distances."""
    unique_labels = np.unique(labels)
    global_mean = X.mean(axis=0)

    point_to_centroid = np.zeros(len(X))
    centroid_to_global = np.zeros(len(X))

    for li in unique_labels:
        mask = labels == li
        centroid = X[mask].mean(axis=0)
        dists = np.linalg.norm(X[mask] - centroid, axis=1)
        point_to_centroid[mask] = dists
        centroid_to_global[mask] = np.linalg.norm(centroid - global_mean)

    # Guard against constant arrays
    if np.std(point_to_centroid) == 0 or np.std(centroid_to_global) == 0:
        return 0.0

    corr, _ = pearsonr(point_to_centroid, centroid_to_global)
    return corr


# =============================================================================
# Tier 1: Centroid-based CVIs — O(n·d), no pairwise distances (6)
# =============================================================================


def ball_hall(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: dict = None,
    sizes: dict = None,
    wgss_per_cluster: dict = None,
    **kwargs,
) -> float:
    """Ball-Hall: mean of per-cluster mean dispersion. Lower = tighter."""
    K = len(centroids)
    return sum(wgss_per_cluster[l] / sizes[l] for l in centroids) / K


def ratkowsky_lance(X: np.ndarray, labels: np.ndarray, centroids: dict = None, **kwargs) -> float:
    """Ratkowsky-Lance: sqrt(mean per-feature BGSS/TSS / K). Higher = better."""
    unique = np.unique(labels)
    K = len(unique)
    d = X.shape[1]
    global_mean = X.mean(axis=0)
    bgss_feat = np.zeros(d)
    tss_feat = np.sum((X - global_mean) ** 2, axis=0)
    for l in unique:
        mask = labels == l
        nj = mask.sum()
        bgss_feat += nj * (centroids[l] - global_mean) ** 2
    ratios = bgss_feat / np.maximum(tss_feat, 1e-10)
    return np.sqrt(np.mean(ratios) / K)


def ray_turi(
    X: np.ndarray, labels: np.ndarray, total_wgss: float = None, centroids: dict = None, **kwargs
) -> float:
    """Ray-Turi: (WGSS/N) / min_sq_center_dist. Lower = better."""
    N = len(X)
    centers = np.array(list(centroids.values()))
    min_sq = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sq = np.sum((centers[i] - centers[j]) ** 2)
            min_sq = min(min_sq, sq)
    return (total_wgss / N) / max(min_sq, 1e-10)


def rmsstd_index(
    X: np.ndarray, labels: np.ndarray, total_wgss: float = None, sizes: dict = None, **kwargs
) -> float:
    """RMSSTD: root mean square std dev. Lower = tighter."""
    d = X.shape[1]
    dof = sum(n - 1 for n in sizes.values())
    return np.sqrt(total_wgss / max(dof * d, 1)) if dof > 0 else np.inf


def r_squared(
    X: np.ndarray, labels: np.ndarray, total_bgss: float = None, total_tss: float = None, **kwargs
) -> float:
    """R-squared: BGSS/TSS. Higher = better separation."""
    return total_bgss / max(total_tss, 1e-10)


def wemmert_gancarski(
    X: np.ndarray, labels: np.ndarray, centroids: dict = None, sizes: dict = None, **kwargs
) -> float:
    """Wemmert-Gancarski: centroid membership quality. Higher = better."""
    N = len(X)
    unique = np.unique(labels)
    result = 0.0
    for l in unique:
        mask = labels == l
        Xc = X[mask]
        cj = centroids[l]
        nj = sizes[l]
        dist_own = np.linalg.norm(Xc - cj, axis=1)
        other_centers = np.array([centroids[ol] for ol in unique if ol != l])
        if len(other_centers) == 0:
            continue
        dist_others = np.min(
            np.linalg.norm(Xc[:, None, :] - other_centers[None, :, :], axis=2), axis=1
        )
        ratios = dist_own / np.maximum(dist_others, 1e-10)
        result += nj * np.mean(np.maximum(0, 1 - ratios))
    return result / N


# =============================================================================
# Tier 2: Scatter-matrix or pairwise-distance CVIs (14)
# =============================================================================


def cs_index(
    X: np.ndarray,
    labels: np.ndarray,
    dist_matrix: np.ndarray = None,
    centroids: dict = None,
    **kwargs,
) -> float:
    """CS Index: max intra-nearest / min centroid dist. Lower = better."""
    unique = np.unique(labels)
    numerator = 0.0
    for l in unique:
        indices = np.where(labels == l)[0]
        if len(indices) < 2:
            continue
        sub = dist_matrix[np.ix_(indices, indices)].copy()
        np.fill_diagonal(sub, np.inf)
        max_nearest = sub.min(axis=1).max()
        numerator += max_nearest / len(indices)
    centers = np.array([centroids[l] for l in unique])
    min_cd = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            min_cd = min(min_cd, np.linalg.norm(centers[i] - centers[j]))
    return numerator / max(min_cd, 1e-10)


def cop_index(
    X: np.ndarray,
    labels: np.ndarray,
    dist_matrix: np.ndarray = None,
    centroids: dict = None,
    **kwargs,
) -> float:
    """COP: mean(centroid_dist / max_intra_dist). Lower = better."""
    N = len(X)
    unique = np.unique(labels)
    result = 0.0
    for l in unique:
        indices = np.where(labels == l)[0]
        cj = centroids[l]
        for idx in indices:
            d_cent = np.linalg.norm(X[idx] - cj)
            max_intra = dist_matrix[idx, indices].max()
            result += d_cent / max(max_intra, 1e-10)
    return result / N


def s_dbw_index(X: np.ndarray, labels: np.ndarray, centroids: dict = None, **kwargs) -> float:
    """S_Dbw: scatter + inter-cluster density. Lower = better."""
    unique = np.unique(labels)
    K = len(unique)
    sigma_all_norm = np.linalg.norm(np.std(X, axis=0))
    cluster_sigmas = {}
    scat = 0.0
    for l in unique:
        mask = labels == l
        sigma_j = np.std(X[mask], axis=0)
        cluster_sigmas[l] = sigma_j
        scat += np.linalg.norm(sigma_j) / max(sigma_all_norm, 1e-10)
    scat /= K
    stdev_avg = np.mean([np.linalg.norm(cluster_sigmas[l]) for l in unique])
    dens_bw = 0.0
    n_pairs = 0
    for i, li in enumerate(unique):
        for lj in unique[i + 1 :]:
            midpt = (centroids[li] + centroids[lj]) / 2.0
            mask_i, mask_j = labels == li, labels == lj
            combined = np.vstack([X[mask_i], X[mask_j]])
            n_mid = np.sum(np.linalg.norm(combined - midpt, axis=1) <= stdev_avg)
            n_ci = np.sum(np.linalg.norm(X[mask_i] - centroids[li], axis=1) <= stdev_avg)
            n_cj = np.sum(np.linalg.norm(X[mask_j] - centroids[lj], axis=1) <= stdev_avg)
            dens_bw += n_mid / max(max(n_ci, n_cj), 1)
            n_pairs += 1
    dens_bw /= max(n_pairs, 1)
    return scat + dens_bw


def det_ratio_index(
    X: np.ndarray, labels: np.ndarray, T: np.ndarray = None, W: np.ndarray = None, **kwargs
) -> float:
    """Det Ratio: det(T)/det(W). Higher = better."""
    det_W = np.linalg.det(W)
    return np.linalg.det(T) / max(abs(det_W), 1e-10)


def gamma_index(S_plus: int = None, S_minus: int = None, **kwargs) -> float:
    """Gamma: (S+ - S-) / (S+ + S-). Higher = better."""
    total = S_plus + S_minus
    return (S_plus - S_minus) / max(total, 1) if total > 0 else 0.0


def generalized_dunn_index(
    X: np.ndarray,
    labels: np.ndarray,
    dist_matrix: np.ndarray = None,
    centroids: dict = None,
    **kwargs,
) -> float:
    """Generalized Dunn: centroid-based inter / max diameter. Higher = better."""
    unique = np.unique(labels)
    max_diam = 0.0
    for l in unique:
        indices = np.where(labels == l)[0]
        if len(indices) > 1:
            max_diam = max(max_diam, dist_matrix[np.ix_(indices, indices)].max())
    centers = np.array([centroids[l] for l in unique])
    min_inter = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            min_inter = min(min_inter, np.linalg.norm(centers[i] - centers[j]))
    return min_inter / max(max_diam, 1e-10)


def g_plus_index(S_minus: int = None, N_w: int = None, N_b: int = None, **kwargs) -> float:
    """G-plus: 2*S- / total_pairs². Lower = better."""
    N_pairs = N_w + N_b
    denom = N_pairs * (N_pairs - 1)
    return 2 * S_minus / max(denom, 1)


def i_index_pbm(X: np.ndarray, labels: np.ndarray, centroids: dict = None, **kwargs) -> float:
    """I-index (PBM): (1/K × E_T/E_W × D_B)². Higher = better."""
    unique = np.unique(labels)
    K = len(unique)
    global_mean = X.mean(axis=0)
    E_T = np.sum(np.linalg.norm(X - global_mean, axis=1))
    E_W = sum(np.sum(np.linalg.norm(X[labels == l] - centroids[l], axis=1)) for l in unique)
    centers = np.array([centroids[l] for l in unique])
    D_B = max(
        np.linalg.norm(centers[i] - centers[j])
        for i in range(len(centers))
        for j in range(i + 1, len(centers))
    )
    return ((1.0 / K) * (E_T / max(E_W, 1e-10)) * D_B) ** 2


def log_det_ratio_index(
    X: np.ndarray, labels: np.ndarray, T: np.ndarray = None, W: np.ndarray = None, **kwargs
) -> float:
    """Log_Det_Ratio: N*log(det(T)/det(W)). Higher = better."""
    det_W = np.linalg.det(W)
    ratio = np.linalg.det(T) / max(abs(det_W), 1e-10)
    return len(X) * np.log(max(ratio, 1e-10))


def mcclain_rao(
    X: np.ndarray, labels: np.ndarray, dist_matrix: np.ndarray = None, **kwargs
) -> float:
    """McClain-Rao: mean_within / mean_between. Lower = better."""
    n = len(labels)
    triu_i, triu_j = np.triu_indices(n, k=1)
    same = labels[triu_i] == labels[triu_j]
    dists = dist_matrix[triu_i, triu_j]
    sw = dists[same].sum()
    nw = same.sum()
    sb = dists[~same].sum()
    nb = (~same).sum()
    return (sw / max(nw, 1)) / max(sb / max(nb, 1), 1e-10)


def point_biserial_index(
    X: np.ndarray, labels: np.ndarray, dist_matrix: np.ndarray = None, **kwargs
) -> float:
    """Point-Biserial: corr(distance, same_cluster). Higher = better (negated)."""
    n = len(labels)
    triu_i, triu_j = np.triu_indices(n, k=1)
    dists = dist_matrix[triu_i, triu_j]
    same = (labels[triu_i] == labels[triu_j]).astype(float)
    corr, _ = pearsonr(dists, same)
    return -corr  # Negate: higher means within-cluster pairs have smaller distances


def sd_validity_index(X: np.ndarray, labels: np.ndarray, centroids: dict = None, **kwargs) -> float:
    """SD validity: scatter + distance measure. Lower = better."""
    unique = np.unique(labels)
    K = len(unique)
    sigma_all_norm = np.linalg.norm(np.var(X, axis=0))
    scat = sum(np.linalg.norm(np.var(X[labels == l], axis=0)) for l in unique) / (
        K * max(sigma_all_norm, 1e-10)
    )
    centers = np.array([centroids[l] for l in unique])
    D_max, D_min = 0.0, np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(centers[i] - centers[j])
            D_max = max(D_max, d)
            D_min = min(D_min, d)
    dis = 0.0
    for i in range(K):
        s = sum(
            1.0 / max(np.linalg.norm(centers[i] - centers[j]), 1e-10) for j in range(K) if j != i
        )
        dis += s
    dis *= D_max / max(D_min, 1e-10)
    return scat * scat + dis


def tau_index(
    S_plus: int = None, S_minus: int = None, N_w: int = None, N_b: int = None, **kwargs
) -> float:
    """Tau: normalized concordance. Higher = better."""
    t_pairs = N_w + N_b
    denom = np.sqrt(max((t_pairs - N_w) * (t_pairs - N_b), 1))
    return (S_plus - S_minus) / denom if denom > 0 else 0.0


def trace_wib_index(
    X: np.ndarray, labels: np.ndarray, W: np.ndarray = None, B: np.ndarray = None, **kwargs
) -> float:
    """Trace(W⁻¹B): separation relative to compactness. Higher = better."""
    try:
        return np.trace(np.linalg.inv(W) @ B)
    except np.linalg.LinAlgError:
        return 0.0


def ksq_detw_index(X: np.ndarray, labels: np.ndarray, W: np.ndarray = None, **kwargs) -> float:
    """Ksq_DetW: K²×det(W). Lower = tighter clusters."""
    K = len(np.unique(labels))
    return K**2 * np.linalg.det(W)


# =============================================================================
# Tier 3: Per-cluster covariance or specialized logic (5)
# =============================================================================


def banfield_raftery(
    X: np.ndarray, labels: np.ndarray, W_per_cluster: dict = None, sizes: dict = None, **kwargs
) -> float:
    """Banfield-Raftery: sum nj*log(trace(Wj)/nj). Lower = better."""
    result = 0.0
    for l in W_per_cluster:
        nj = sizes[l]
        trace_wj = np.trace(W_per_cluster[l])
        if nj > 0 and trace_wj > 0:
            result += nj * np.log(trace_wj / nj)
    return result


def negentropy_increment(
    X: np.ndarray, labels: np.ndarray, W_per_cluster: dict = None, sizes: dict = None, **kwargs
) -> float:
    """Negentropy: cluster entropy relative to Gaussian. Lower = better."""
    N, d = X.shape
    result = 0.0
    for l in W_per_cluster:
        nj = sizes[l]
        if nj < d + 1:
            continue
        cov_j = W_per_cluster[l] / nj
        det_cov = np.linalg.det(cov_j)
        if det_cov > 0:
            h_j = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * np.log(det_cov)
            result += (nj / N) * h_j
    return result


def niva_index(
    X: np.ndarray, labels: np.ndarray, dist_matrix: np.ndarray = None, **kwargs
) -> float:
    """NIVA: nearest intra/inter distance ratio. Lower = better."""
    unique = np.unique(labels)
    compactness = 0.0
    separability = 0.0
    for l in unique:
        indices = np.where(labels == l)[0]
        other_indices = np.where(labels != l)[0]
        if len(indices) < 2 or len(other_indices) == 0:
            continue
        sub = dist_matrix[np.ix_(indices, indices)].copy()
        np.fill_diagonal(sub, np.inf)
        compactness += sub.min(axis=1).mean()
        inter = dist_matrix[np.ix_(indices, other_indices)]
        separability += inter.min(axis=1).mean()
    return compactness / max(separability, 1e-10)


def score_function_index(
    X: np.ndarray, labels: np.ndarray, centroids: dict = None, sizes: dict = None, **kwargs
) -> float:
    """Score Function: 1 - 1/exp(exp(bdc-wcd)). Higher = better."""
    unique = np.unique(labels)
    global_mean = X.mean(axis=0)
    bdc = np.mean([np.linalg.norm(centroids[l] - global_mean) for l in unique])
    wcd = np.mean([np.mean(np.linalg.norm(X[labels == l] - centroids[l], axis=1)) for l in unique])
    # Clip to [-6, 6] to prevent overflow in double exponential
    # exp(exp(6.5)) ≈ exp(665) which is near float64 max
    inner = np.clip(bdc - wcd, -6, 6)
    return 1.0 - 1.0 / np.exp(np.exp(inner))


def scott_symons_index(
    X: np.ndarray, labels: np.ndarray, W_per_cluster: dict = None, sizes: dict = None, **kwargs
) -> float:
    """Scott-Symons: sum nj*log(det(Wj/nj)). Lower = better."""
    d = X.shape[1]
    result = 0.0
    for l in W_per_cluster:
        nj = sizes[l]
        if nj < d + 1:
            continue
        det_cov = np.linalg.det(W_per_cluster[l] / nj)
        if det_cov > 0:
            result += nj * np.log(det_cov)
    return result


# =============================================================================
# External Metrics (4)
# =============================================================================


def adjusted_rand(labels_true: np.ndarray, labels_pred: np.ndarray, **kwargs) -> float:
    """Adjusted Rand Index: chance-corrected Rand Index."""
    return adjusted_rand_score(labels_true, labels_pred)


def van_dongen_criterion(labels_true: np.ndarray, labels_pred: np.ndarray, **kwargs) -> float:
    """Van Dongen: 1 - (row_max_sum + col_max_sum) / (2n). Range [0,1], lower = better."""
    cm = contingency_matrix(labels_true, labels_pred)
    n = cm.sum()
    row_max_sum = cm.max(axis=1).sum()
    col_max_sum = cm.max(axis=0).sum()
    return 1.0 - (row_max_sum + col_max_sum) / (2 * n)


def variation_of_information(
    labels_true: np.ndarray, labels_pred: np.ndarray, normalized: bool = True, **kwargs
) -> float:
    """VI = H(U|V) + H(V|U). Lower = better. Optionally normalized to [0,1]."""
    n = len(labels_true)
    cm = contingency_matrix(labels_true, labels_pred)
    joint = cm / n  # joint probability

    p_true = joint.sum(axis=1)  # marginal for true labels
    p_pred = joint.sum(axis=0)  # marginal for predicted labels

    vi = 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if joint[i, j] > 0:
                vi -= joint[i, j] * np.log2(joint[i, j] / p_pred[j])
                vi -= joint[i, j] * np.log2(joint[i, j] / p_true[i])

    if normalized and n > 1:
        vi /= np.log2(n)
    return vi


def omega_indicator(labels_true: np.ndarray, labels_pred: np.ndarray, **kwargs) -> float:
    """Omega: mean([1-ARI], VD, VI_normalized). Lower = better."""
    ari = adjusted_rand_score(labels_true, labels_pred)
    rand_dist = 1.0 - ari
    vd = van_dongen_criterion(labels_true, labels_pred)
    vi_norm = variation_of_information(labels_true, labels_pred, normalized=True)
    return np.mean([rand_dist, vd, vi_norm])


# =============================================================================
# Metric Registry
# =============================================================================

# Map metric names to functions
METRIC_FUNCTIONS = {
    # Original (6)
    "Silhouette": silhouette,
    "Davies-Bouldin": davies_bouldin,
    "Calinski-Harabasz": calinski_harabasz,
    "Dunn Index": dunn_index,
    "SSE": sse,
    "NCI": new_correlation_index,
    # Tier 1 (6)
    "Ball-Hall": ball_hall,
    "Ratkowsky-Lance": ratkowsky_lance,
    "Ray-Turi": ray_turi,
    "RMSSTD": rmsstd_index,
    "R-squared": r_squared,
    "Wemmert-Gancarski": wemmert_gancarski,
    # Tier 2 (14)
    "CS index": cs_index,
    "COP": cop_index,
    "S_Dbw": s_dbw_index,
    "Det Ratio": det_ratio_index,
    "Gamma": gamma_index,
    "Generalized Dunn": generalized_dunn_index,
    "G-plus": g_plus_index,
    "I-index (PBM)": i_index_pbm,
    "Log_Det_Ratio": log_det_ratio_index,
    "McClain-Rao": mcclain_rao,
    "Point-Biserial": point_biserial_index,
    "SD validity": sd_validity_index,
    "Tau": tau_index,
    "Trace_WiB": trace_wib_index,
    "Ksq_DetW": ksq_detw_index,
    # Tier 3 (5)
    "Banfield-Raftery": banfield_raftery,
    "Negentropy": negentropy_increment,
    "NIVA": niva_index,
    "Score Function": score_function_index,
    "Scott-Symons": scott_symons_index,
}

# External metrics (require ground truth)
EXTERNAL_METRIC_FUNCTIONS = {
    "Adjusted Rand Index": adjusted_rand,
    "Van Dongen": van_dongen_criterion,
    "Variation of Information": variation_of_information,
    "Omega": omega_indicator,
}
