# Degradation Types

Metricate provides 19 degradation types for systematically corrupting clusterings. These are used to:

1. **Test metric robustness** - See how metrics respond to different types of degradation
2. **Generate training data** - Create labeled positive/negative examples for learning weights
3. **Benchmark algorithms** - Compare how different clustering methods handle noise

---

## Overview

| Category | Types | Description |
|----------|-------|-------------|
| Label Manipulation | 3 | Swap cluster assignments |
| Cluster Structure | 6 | Merge or split clusters |
| Point Manipulation | 4 | Modify or remove individual points |
| Cluster Removal | 3 | Remove entire clusters |
| Embedding Manipulation | 2 | Perturb embedding vectors |

---

## Label Manipulation

These degradations change cluster assignments without modifying embeddings.

### `label_swap_random`

Randomly swap cluster labels for a fraction of points.

**Effect**: Points get assigned to random clusters, breaking cluster coherence everywhere.

**Best for testing**: Overall metric sensitivity to label noise.

---

### `label_swap_neighboring`

Swap labels only between nearby clusters (clusters with close centroids).

**Effect**: Creates realistic boundary confusion, as if the clustering algorithm couldn't decide on borders.

**Best for testing**: Metrics that focus on cluster separation.

---

### `label_swap_distant`

Swap labels only between distant clusters (clusters with far centroids).

**Effect**: Creates obvious misassignments that should be easy to detect.

**Best for testing**: Metrics that should catch gross errors.

---

## Cluster Structure

These degradations change the number or composition of clusters.

### `merge_random`

Randomly select two clusters and merge them into one.

**Effect**: Reduces cluster count, may combine unrelated points.

**Best for testing**: Metrics sensitive to under-clustering.

---

### `merge_nearest`

Merge the two closest clusters (by centroid distance).

**Effect**: Simulates failing to distinguish similar groups.

**Best for testing**: Metrics that reward fine-grained separation.

---

### `merge_farthest`

Merge the two most distant clusters.

**Effect**: Creates an obviously heterogeneous cluster.

**Best for testing**: Compactness metrics.

---

### `split_random`

Randomly split a cluster into two using k-means.

**Effect**: Increases cluster count, may create artificial boundaries.

**Best for testing**: Metrics sensitive to over-clustering.

---

### `split_largest`

Split the largest cluster into two.

**Effect**: Specifically targets imbalanced clusterings.

**Best for testing**: Metrics that penalize cluster size imbalance.

---

### `split_loosest`

Split the cluster with highest internal variance.

**Effect**: Targets the most spread-out cluster.

**Best for testing**: Compactness-focused metrics.

---

## Point Manipulation

These degradations modify or remove individual points.

### `noise_injection`

Add Gaussian noise to embedding vectors.

**Effect**: Points drift from their cluster centers, blurring boundaries.

**Best for testing**: Metrics based on point-to-centroid distances.

---

### `random_removal`

Randomly remove a fraction of points.

**Effect**: General data loss, maintains cluster proportions.

**Best for testing**: Baseline metric stability.

---

### `core_removal`

Remove points closest to cluster centroids.

**Effect**: Hollows out clusters, removing their most representative members.

**Best for testing**: Metrics that rely on centroid quality.

---

### `boundary_reassignment`

Move boundary points (far from centroid) to nearby clusters.

**Effect**: Creates realistic cluster overlap.

**Best for testing**: Separation and overlap metrics.

---

## Cluster Removal

These degradations remove entire clusters from the data.

### `remove_smallest_clusters`

Remove the N smallest clusters entirely.

**Effect**: Simplifies the clustering, loses minority groups.

**Best for testing**: Metrics that reward having all groups represented.

---

### `remove_largest_clusters`

Remove the N largest clusters entirely.

**Effect**: Dramatic data loss, may remove dominant patterns.

**Best for testing**: Robustness to major structural changes.

---

### `remove_tightest_clusters`

Remove the N most compact clusters (lowest internal variance).

**Effect**: Removes the "easy" clusters, keeps difficult ones.

**Best for testing**: How metrics handle removal of clear structure.

---

## Embedding Manipulation

These degradations modify the embedding space itself.

### `embedding_perturbation`

Add random noise to all embedding dimensions.

**Effect**: Global degradation of embedding quality.

**Best for testing**: Overall metric sensitivity to embedding noise.

---

### `centroid_displacement`

Shift cluster centroids while keeping relative point positions.

**Effect**: Clusters move in space but maintain internal structure.

**Best for testing**: Metrics based on inter-cluster distances.

---

## Usage

### Generate All Degradations

```python
import metricate

result = metricate.degrade(
    "clustering.csv",
    output_dir="./degraded/",
    levels=["5pct", "10pct", "25pct", "50pct"]
)

print(result.summary())
# Output:
# Degradation Generation Complete
# ========================================
# Output directory: ./degraded/
# Total degradations: 76
# Manifest: ./degraded/manifest.json
```

### Generate Specific Types

```python
result = metricate.degrade(
    "clustering.csv",
    output_dir="./output/",
    types=["label_swap_random", "noise_injection", "merge_nearest"],
    levels=["10pct", "25pct"]
)
```

### List All Available Types

```python
import metricate

types = metricate.list_degradations()
for category, degradations in types.items():
    print(f"\n{category}:")
    for d in degradations:
        print(f"  - {d}")
```

Output:

```
Label Manipulation:
  - label_swap_random
  - label_swap_neighboring
  - label_swap_distant

Cluster Structure:
  - merge_random
  - merge_nearest
  - merge_farthest
  - split_random
  - split_largest
  - split_loosest

Point Manipulation:
  - noise_injection
  - random_removal
  - core_removal
  - boundary_reassignment

Cluster Removal:
  - remove_smallest_clusters
  - remove_largest_clusters
  - remove_tightest_clusters

Embedding Manipulation:
  - embedding_perturbation
  - centroid_displacement
```

---

## Degradation Levels

Each degradation type supports multiple intensity levels:

| Level | Fraction | Description |
|-------|----------|-------------|
| `5pct` | 5% | Minor degradation |
| `10pct` | 10% | Light degradation |
| `25pct` | 25% | Moderate degradation |
| `50pct` | 50% | Severe degradation |

The fraction meaning varies by degradation type:
- **Label swaps**: Fraction of points with swapped labels
- **Point removal**: Fraction of points removed
- **Noise injection**: Scale of noise relative to cluster spread
- **Cluster merges/splits**: Applied to N% of clusters

---

## Visualizations

When `visualize=True`, Metricate generates interactive HTML visualizations:

```python
result = metricate.degrade(
    "clustering.csv",
    output_dir="./output/",
    visualize=True
)

print(f"Index page: {result.index_html_path}")
```

**Generated files:**
- `index.html` - Dashboard linking to all visualizations
- `<degradation_type>.html` - Interactive Plotly scatter plot per type

**Viewing:**
```bash
# Open directly
open ./output/index.html

# Or serve with Python
cd ./output && python -m http.server 8000
```

---

## Default Exclusions

Some degradation types are excluded by default because they're redundant or problematic:

- `split_loosest` - Similar results to `split_random`
- `split_random` - Can be unpredictable
- `random_removal` - Less informative than targeted removal
- `remove_largest_clusters` - Too destructive
- `remove_smallest_clusters` - May remove noise clusters
- `merge_nearest` - Results similar to `merge_random`

To include them:

```python
result = metricate.degrade(
    "clustering.csv",
    output_dir="./output/",
    types=["random_removal", "merge_nearest"]  # Explicitly include
)
```
