# Metrics Reference

Metricate computes 36 clustering quality metrics organized into 5 tiers based on their computational requirements and use cases.

---

## Overview

| Tier | Count | Description |
|------|-------|-------------|
| Original | 6 | Classic, widely-used metrics |
| Tier 1 | 6 | Centroid-based, O(n·d) complexity |
| Tier 2 | 14 | Scatter-matrix or pairwise-distance based |
| Tier 3 | 5 | Per-cluster covariance or specialized |
| External | 4 | Require ground truth labels |

---

## Internal Original (6 Metrics)

These are the most commonly used clustering validation indices.

### Silhouette

| Property | Value |
|----------|-------|
| Range | [-1, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n²) |

Average silhouette coefficient across all samples. Measures how similar points are to their own cluster compared to other clusters.

**Interpretation:**
- `1.0`: Perfect clustering
- `0.0`: Overlapping clusters
- `-1.0`: Points assigned to wrong clusters

**Strengths:** Depends only on actual partition, not the algorithm. Useful for comparing different algorithms. Suitable for compact clusters and multiple densities.

**Weaknesses:** Tied to specific distance measures—can't compare results using different distances. Only applicable to spherical clusters.

---

### Davies-Bouldin

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n) |

Average similarity ratio of each cluster with its most similar cluster. Lower values indicate better separation.

**Strengths:** Hardly affected by cluster overlap. Demonstrates good clustering partition quality.

**Weaknesses:** Makes strong assumptions not valid in many real situations. Too simple for arbitrarily shaped clusters with dispersed density. Only applicable to spherical clusters.

---

### Calinski-Harabasz

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n) |

Also known as the Variance Ratio Criterion. Ratio of between-cluster dispersion to within-cluster dispersion.

**Strengths:** Assesses clustering quality regardless of the choice of distance measure.

**Weaknesses:** Affected by data size and level of data overlap. Data-dependent—behavior may change with different data structures. Only applicable to spherical clusters.

---

### Dunn Index

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n²) |

Ratio of minimum inter-cluster distance to maximum intra-cluster diameter. Higher values indicate compact, well-separated clusters.

**Strengths:** Capable of finding correct clustering structure for arbitrarily shaped clusters with high density.

**Weaknesses:** Makes strong assumptions not valid in many real situations. Computationally expensive and sensitive to noise. Only applicable to spherical clusters.

---

### SSE (Sum of Squared Errors)

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n) |

Within-cluster sum of squares. Measures total compactness of clusters.

**Strengths:** Provides clear numerical compactness value. Versatile—can be used with k-means, hierarchical clustering, and others.

**Weaknesses:** Highly sensitive to outliers. Computationally expensive for very large datasets or high-dimensional data.

---

### NCI (New Correlation Index)

| Property | Value |
|----------|-------|
| Range | [-1, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n) |

Correlation between point-to-centroid distances and centroid-to-global-mean distances.

---

## Tier 1 CVIs (6 Metrics)

Centroid-based metrics with O(n·d) complexity.

### Ball-Hall

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d) |

Mean of per-cluster mean dispersion.

**Strengths:** No absolute threshold used for similarity criterion. Independent of pattern presentation sequence. Capable of finding correct structure for arbitrarily shaped clusters with high density.

**Weaknesses:** Weighted metrics can make clustering interpretation difficult for data analysis.

---

### Ratkowsky-Lance

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n·d) |

Per-feature BGSS/TSS ratio (between-group sum of squares / total sum of squares).

**Strengths:** Superior performance in validating clusters in binary datasets.

**Weaknesses:** Weakness in correct absolute cluster profile identification.

---

### Ray-Turi

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d) |

Mean squared distance divided by minimum centroid distance squared.

**Strengths:** Superior performance in cluster validation for dynamic connectivity data.

**Weaknesses:** Exhibits sensitivity problems.

---

### RMSSTD

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d) |

Root mean square standard deviation of clusters.

**Strengths:** Valid for rectangular data.

**Weaknesses:** Only valid with average, centroid, and ward linkage methods. Can only validate well-separated hyper-sphere-shaped clusters.

---

### R-squared

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n·d) |

Between-group sum of squares / Total sum of squares. Measures proportion of variance explained by clustering.

---

### Wemmert-Gancarski

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n·k·d) |

Centroid membership quality index.

**Strengths:** Performance stability across all distance measures for both synthetic and real datasets.

**Weaknesses:** Performance sensitive to noise.

---

## Tier 2 CVIs (14 Metrics)

Scatter-matrix or pairwise-distance based metrics.

### CS Index

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n²) |

Compact-Separated index: ratio of maximum intra-nearest to minimum centroid distance.

**Strengths:** Efficient in handling clusters with different dimensions, densities, or sizes. Produces good quality solutions.

**Weaknesses:** Computationally intensive and expensive.

---

### COP

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n²) |

Mean of centroid distance divided by maximum intra-cluster distance.

**Strengths:** Not affected by the number of clusters. Hardly affected by cluster overlap.

**Weaknesses:** Only applicable to spherical clusters.

---

### S_Dbw

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n²) |

Scatter plus inter-cluster density measure.

**Strengths:** Works well for compact and well-separated clusters. Robust to noise.

**Weaknesses:** Cannot work with non-convex clusters or clusters with curved geometries. High computational cost.

---

### Det Ratio

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n·d²) |

det(Total scatter matrix) / det(Within scatter matrix).

**Strengths:** One of the best validity criteria for arbitrarily shaped closed contour clusters. Capable of finding correct structure for arbitrarily shaped high-density clusters.

**Weaknesses:** Highly sensitive to the size and shape of clusters. Does not explicitly account for cluster overlap.

---

### Gamma

| Property | Value |
|----------|-------|
| Range | [-1, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n²) |
| Auto-skip | Yes (>50k rows) |

Concordant-discordant pair ratio. Measures consistency between distance ordering and cluster assignment.

**Strengths:** Suitable for datasets with compactness properties and datasets with multiple densities.

**Weaknesses:** Data-dependent—behavior varies per data structure. Computationally expensive. Inefficient with overlapping clusters. Difficulties with arbitrarily shaped clusters.

---

### Generalized Dunn

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n²) |

Centroid-based inter-cluster distance / maximum diameter.

**Strengths:** Good for validating hyper-spherical/cloud and shell-type clusters.

**Weaknesses:** Computationally intensive and expensive.

---

### G-plus

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↓ Lower is better |
| Complexity | O(n²) |
| Auto-skip | Yes (>50k rows) |

Normalized discordant pairs.

**Strengths:** Capable of finding correct clustering structure for arbitrarily shaped clusters with high density.

**Weaknesses:** Computationally expensive. Inefficient with overlapping clusters. Difficulties with arbitrarily shaped clusters.

---

### I-index (PBM)

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n·d) |

Composite index: (1/K × E_T/E_W × D_B)².

**Strengths:** More consistent and reliable in indicating the correct number of clusters compared with Davies-Bouldin, Calinski-Harabasz, and Dunn Index.

**Weaknesses:** Requires parameter tuning.

---

### Log_Det_Ratio

| Property | Value |
|----------|-------|
| Range | (-∞, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n·d²) |

N × log(det(T)/det(W)).

**Strengths:** Capable of finding correct clustering structure for arbitrarily shaped clusters with high density.

**Weaknesses:** Assumes clusters are roughly spherical and of similar size. Focuses on compactness, potentially neglecting separation between clusters.

---

### McClain-Rao

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n²) |
| Auto-skip | Yes (>50k rows) |

Mean within-cluster distance / mean between-cluster distance.

**Strengths:** Performs relatively well in low dimensions.

**Weaknesses:** Performance degrades as dimension increases. **Identified as worst-performing CVI in comparative studies.**

---

### Point-Biserial

| Property | Value |
|----------|-------|
| Range | [-1, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n²) |
| Auto-skip | Yes (>50k rows) |

Correlation between pairwise distances and cluster membership.

**Strengths:** Capable of finding correct clustering structure for arbitrarily shaped clusters with high density.

**Weaknesses:** Sensitivity to varying numbers of clusters or dimensions in datasets.

---

### SD Validity

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d) |

Scatter plus centroid distance measure.

**Strengths:** Finds optimal partition independent of the clustering algorithm.

**Weaknesses:** Sensitive to the geometry of cluster centres and number of clusters.

---

### Tau

| Property | Value |
|----------|-------|
| Range | [-1, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n²) |
| Auto-skip | Yes (>50k rows) |

Normalized concordance index (Kendall's Tau).

**Strengths:** Capable of finding correct clustering structure for arbitrarily shaped clusters with high density.

**Weaknesses:** High computational cost.

---

### Trace_WiB

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↑ Higher is better |
| Complexity | O(n·d²) |

trace(W⁻¹B): separation relative to compactness.

**Strengths:** Normalized, which helps comparing clustering results across different datasets or methods.

**Weaknesses:** May be influenced by initial conditions or the clustering algorithm used, leading to variability.

---

### Ksq_DetW

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d²) |

K² × det(W), where K is number of clusters.

**Strengths:** Capable of finding correct clustering structure for arbitrarily shaped clusters with high density.

**Weaknesses:** Does not allow for direct comparison between clustering algorithms.

---

## Tier 3 CVIs (5 Metrics)

Per-cluster covariance or specialized metrics.

### Banfield-Raftery

| Property | Value |
|----------|-------|
| Range | (-∞, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d²) |

Sum of n_j × log(trace(W_j)/n_j).

**Strengths:** Incorporates a penalty for model parameters, helping prevent overfitting. Uses likelihood function to evaluate model fit. Can be used with large datasets and complex models.

**Weaknesses:** Extremely compute-intensive for large datasets. Effectiveness relies on correctness of underlying models.

---

### Negentropy

| Property | Value |
|----------|-------|
| Range | (-∞, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d²) |

Cluster entropy relative to Gaussian distribution.

**Strengths:** Calculation simplicity. Satisfactory performance on clusters with heterogeneous orientation, densities, and scales. Assesses correct number of clusters more reliably than Davies-Bouldin, Dunn, and PBM.

**Weaknesses:** Poor performance with datasets with low number of data points.

---

### NIVA

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n²) |
| Auto-skip | Yes (>50k rows) |

Nearest intra/inter distance ratio.

**Strengths:** Takes advantage of cluster density, size, and shape.

**Weaknesses:** Can place too much emphasis on within-cluster variance, potentially neglecting overall structure or topology of the data.

---

### Score Function

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n·d) |

1 - 1/exp(exp(bdc-wcd)).

**Strengths:** Good for validating hyper-spheroidal clusters as well as multidimensional and noisy datasets. Can handle single cluster case and sub-cluster hierarchies.

**Weaknesses:** Restricted to datasets containing hyper-spheroidal clusters.

---

### Scott-Symons

| Property | Value |
|----------|-------|
| Range | (-∞, ∞) |
| Direction | ↓ Lower is better |
| Complexity | O(n·d²) |

Sum of n_j × log(det(W_j/n_j)).

**Strengths:** Suitable for clusters of different shapes, sizes, and orientations.

**Weaknesses:** Cannot be properly calculated where clusters are not well represented. Not robust to noise.

---

## External Metrics (4 Metrics)

These metrics require ground truth labels for comparison.

### Adjusted Rand Index

| Property | Value |
|----------|-------|
| Range | [-1, 1] |
| Direction | ↑ Higher is better |
| Complexity | O(n) |

Chance-corrected Rand Index. Measures similarity between predicted and true labels.

---

### Van Dongen

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↓ Lower is better |
| Complexity | O(n) |

1 - (row_max_sum + col_max_sum) / 2n.

---

### Variation of Information

| Property | Value |
|----------|-------|
| Range | [0, log(n)] |
| Direction | ↓ Lower is better |
| Complexity | O(n) |

H(U|V) + H(V|U) - information lost and gained between clusterings.

---

### Omega

| Property | Value |
|----------|-------|
| Range | [0, 1] |
| Direction | ↓ Lower is better |
| Complexity | O(n) |

Composite: mean([1-ARI], VD, VI_normalized).

---

## Metric Strengths & Weaknesses

Based on research from [Oyelade et al. (2025)](https://doi.org/10.1016/j.heliyon.2025.e41953), here's a comprehensive comparison of clustering validity indices.

### Quick Reference Table

| Metric | Best For | Limitations |
|--------|----------|-------------|
| **Silhouette** | Compact clusters, multiple densities | Only spherical clusters, tied to distance measure |
| **Davies-Bouldin** | Good partition detection, handles overlap | Assumes spherical clusters, too simple for complex shapes |
| **Calinski-Harabasz** | Algorithm-agnostic quality assessment | Affected by data size/overlap, only spherical clusters |
| **Dunn Index** | High-density arbitrarily shaped clusters | Expensive, sensitive to noise, assumes spherical |
| **SSE** | Clear compactness measure, versatile | Sensitive to outliers, expensive for large/high-dim data |
| **Ball-Hall** | No absolute threshold needed, arbitrary shapes | Weighted metrics complicate interpretation |
| **Ratkowsky-Lance** | Binary datasets | Poor absolute cluster profile identification |
| **Ray-Turi** | Dynamic connectivity data | Sensitivity problems |
| **RMSSTD** | Rectangular data, well-separated clusters | Only works with average/centroid/ward methods |
| **Wemmert-Gancarski** | Stable across distance measures | Sensitive to noise |
| **CS Index** | Different dimensions/densities/sizes | Computationally intensive |
| **COP** | Unaffected by cluster count or overlap | Only spherical clusters |
| **S_Dbw** | Compact, well-separated clusters, robust to noise | Can't handle non-convex/curved clusters, expensive |
| **Det Ratio** | Arbitrarily shaped closed contour clusters | Sensitive to cluster size/shape, ignores overlap |
| **Gamma** | Compact clusters, multiple densities | Data-dependent, expensive, struggles with overlap |
| **Generalized Dunn** | Hyper-spherical and shell-type clusters | Computationally intensive |
| **G-plus** | Arbitrarily shaped high-density clusters | Expensive, struggles with overlap and arbitrary shapes |
| **I-index (PBM)** | More reliable than DB, CH, Dunn for cluster count | Requires parameter tuning |
| **Log_Det_Ratio** | Arbitrarily shaped high-density clusters | Assumes spherical similar-sized clusters |
| **McClain-Rao** | Low-dimensional data | Degrades with dimension increase, worst-performing CVI |
| **Point-Biserial** | Arbitrarily shaped high-density clusters | Sensitive to cluster count and dimensions |
| **Tau** | Arbitrarily shaped high-density clusters | High computational cost |
| **Trace_WiB** | Normalized cross-dataset comparison | Sensitive to algorithm/initialization |
| **Banfield-Raftery** | Large datasets, prevents overfitting | Compute-intensive, model-dependent |
| **Negentropy** | Heterogeneous orientation/density/scales | Poor with small datasets |
| **NIVA** | Handles density, size, and shape | Over-emphasizes within-cluster variance |
| **Score Function** | Hyper-spheroidal, multidimensional, noisy data | Only hyper-spheroidal clusters |
| **Scott-Symons** | Different shapes, sizes, orientations | Fails with poorly represented clusters, not robust to noise |

### Detailed Analysis

#### Metrics Good for Arbitrary Shapes

These metrics can handle non-spherical cluster structures:

- **Det Ratio** - Best for arbitrarily shaped closed contour clusters
- **Gamma** - Works with compact clusters at multiple densities  
- **G-plus** - Handles arbitrarily shaped high-density clusters
- **Tau** - Good for arbitrarily shaped high-density clusters
- **Point-Biserial** - Works with arbitrarily shaped high-density clusters

#### Metrics Limited to Spherical Clusters

These metrics assume roughly spherical/hyper-spherical clusters:

- **Silhouette** - Only applicable to spherical clusters
- **Davies-Bouldin** - Assumes spherical clusters
- **Calinski-Harabasz** - Only applicable to spherical clusters
- **Dunn Index** - Only applicable to spherical clusters
- **COP** - Only applicable to spherical clusters
- **Score Function** - Restricted to hyper-spheroidal clusters

#### Most Computationally Expensive

These metrics have O(n²) complexity and should be excluded for large datasets:

| Metric | Issue |
|--------|-------|
| **Gamma** | Computationally prohibitive for most real applications |
| **Tau** | High computational cost |
| **G-plus** | Computationally expensive |
| **Generalized Dunn** | Computationally intensive |
| **McClain-Rao** | Additionally: worst-performing CVI overall |
| **NIVA** | Computationally intensive |

#### Most Robust Metrics

Based on research findings:

- **I-index (PBM)** - More consistent and reliable than DB, CH, and Dunn for finding correct cluster count
- **Negentropy** - Better at assessing correct cluster count than DB, Dunn, and PBM
- **S_Dbw** - Robust to noise
- **COP** - Not affected by number of clusters, hardly affected by overlap

#### Metrics with Known Issues

| Metric | Known Problems |
|--------|----------------|
| **McClain-Rao** | Worst performing CVI in comparative studies |
| **Calinski-Harabasz** | Data-dependent behavior changes with data structure |
| **Dunn Index** | Sensitive to noise, makes invalid assumptions |
| **Negentropy** | Poor with small datasets |
| **Scott-Symons** | Not robust to noise |

---

## Excluded Metrics (Redundant)

Metricate excludes 5 metrics that are mathematically equivalent or redundant with included metrics:

| Excluded Metric | Reason |
|-----------------|--------|
| **Trace_W** | Redundant with other scatter-based metrics |
| **Baker-Hubert Gamma** | Equivalent to Gamma index |
| **Sym-index** | Only works with internally symmetric datasets, algorithm-dependent |
| **Rand Index** | Superseded by Adjusted Rand Index (chance-corrected) |
| **Log_SS_Ratio** | Sensitive to outliers, redundant with other compactness measures |

### Baker-Hubert Gamma vs Gamma

These are the same metric with different names. The Baker-Hubert Gamma index measures concordant-discordant pair ratios and is computationally prohibitive (O(n²)) for most real applications.

### Rand Index vs Adjusted Rand Index

The Adjusted Rand Index is always preferred because it corrects for chance agreement. The raw Rand Index can show high similarity even between random clusterings.

---

## Metric Correlations

Many metrics are mathematically related and tend to agree:

**High Correlation Groups:**

1. **Compactness-focused**: SSE, Ball-Hall, RMSSTD
2. **Separation-focused**: Dunn Index, Generalized Dunn
3. **Ratio-based**: Calinski-Harabasz, R-squared, Det Ratio
4. **Pair-counting**: Gamma, Tau, G-plus, Point-Biserial

When training metric weights, consider that:
- Correlated metrics cause implicit overweighting with equal weights
- Lasso regularization will arbitrarily zero one of correlated metrics
- Ridge regularization distributes weight among correlated metrics (more stable)

---

## Large Dataset Handling

For datasets exceeding 50,000 rows, the following O(n²) metrics are automatically skipped:

- Gamma
- Tau
- Point-Biserial
- G-plus
- McClain-Rao
- NIVA

### Override Auto-Skip

```python
# Force all metrics
result = metricate.evaluate("large_data.csv", force_all=True)
```

```bash
# CLI
metricate evaluate large_data.csv --force-all
```

> **Warning**: Computing O(n²) metrics on large datasets may take several minutes and require significant memory.

---

## Excluding Metrics

Common reasons to exclude metrics:

1. **Optimization bias**: You chose K by optimizing for Tau? Don't use Tau for evaluation.
2. **Runtime**: Skip expensive metrics for faster results
3. **Redundancy**: Some metrics are highly correlated

```python
# Exclude specific metrics
result = metricate.evaluate(
    "clustering.csv",
    exclude=["Gamma", "Tau", "S_Dbw"]
)
```

```bash
# CLI
metricate evaluate clustering.csv --exclude Gamma,Tau,S_Dbw
```

---

## References

The strengths and weaknesses information in this document is based on:

> **Oyelade, O. N., et al. (2025)**. "A comprehensive survey of cluster validity indices for automatic clustering algorithms." *Heliyon*, 11(2), e41953. [https://doi.org/10.1016/j.heliyon.2025.e41953](https://doi.org/10.1016/j.heliyon.2025.e41953)

This paper provides an extensive review of 43 cluster validity indices used in metaheuristic-based automatic clustering algorithms, including their mathematical formulations, strengths, and limitations.
