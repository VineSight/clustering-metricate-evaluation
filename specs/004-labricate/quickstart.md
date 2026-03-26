# Quickstart: Labricate

**Date**: March 18, 2026  
**Feature**: 004-labricate

---

## Installation

Labricate is included with Metricate:

```bash
pip install metricate
```

Or from source:

```bash
pip install -e .
```

---

## 5-Minute Tutorial

### 1. Prepare Your Data

You need:
- **Embeddings**: Pre-computed vectors (before dimensionality reduction)
- **Config**: Pipeline configuration (JSON or dict)

```python
import numpy as np

# Example: Load your embeddings
embeddings = np.load("my_embeddings.npy")  # Shape: (n_samples, n_dims)
print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: (10000, 384)
```

### 2. Define Base Configuration

```python
config = {
    "random_seed": 42,
    "umap": {
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine"
    },
    "clustering_algorithm": "hdbscan",
    "hdbscan": {
        "min_cluster_size": 10,
        "min_samples": 5
    }
}
```

### 3. Run Single-Parameter Experiment

```python
from metricate.labricate import Experiment

# Create experiment
exp = Experiment(embeddings, config)

# Test different min_cluster_size values
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50]
)

# View summary
print(result.summary)
```

Output:
```
ExperimentSummary(
    total_runs=6,
    completed_runs=6,
    failed_runs=0,
    total_duration_seconds=45.2
)
```

### 4. Analyze Results

```python
# Convert to DataFrame
df = result.to_dataframe()
print(df[["hdbscan.min_cluster_size", "n_clusters", "Silhouette", "Davies-Bouldin"]])
```

Output:
```
   hdbscan.min_cluster_size  n_clusters  Silhouette  Davies-Bouldin
0                         5          42      0.312           1.823
1                        10          28      0.398           1.456
2                        15          19      0.425           1.312
3                        20          14      0.441           1.289
4                        30           9      0.467           1.201
5                        50           5      0.423           1.345
```

### 5. Find Best Configuration

```python
# Get run with best Silhouette score
best = result.get_best_run("Silhouette")
print(f"Best Silhouette: {best.metrics['Silhouette']:.3f}")
print(f"At min_cluster_size: {best.param_values['hdbscan.min_cluster_size']}")
```

---

## CLI Quick Start

```bash
# Run experiment from command line
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values "5,10,15,20,30,50"
```

---

## Common Workflows

### Testing UMAP Parameters

```python
# Test n_neighbors
result = exp.run(
    param="umap.n_neighbors",
    values=[5, 10, 15, 20, 30, 50, 100]
)

# Test n_components (output dimensions)
result = exp.run(
    param="umap.n_components",
    values=[2, 3, 5, 10, 15, 20]
)
```

### Grid Search (Multiple Parameters)

```python
result = exp.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20],
        "hdbscan.min_cluster_size": [10, 20, 30]
    },
    n_workers=4  # Parallel execution
)
# Runs 3 × 3 = 9 combinations
```

### Using K-Means Instead of HDBSCAN

```python
config = {
    "umap": {"n_neighbors": 15, "n_components": 5},
    "clustering_algorithm": "kmeans",  # Switch to K-Means
    "kmeans": {"n_clusters": 10}
}

exp = Experiment(embeddings, config)
result = exp.run(
    param="kmeans.n_clusters",
    values=[5, 10, 15, 20, 25, 30]
)
```

### Filtering Metrics (Faster Runs)

```python
# Only calculate fast metrics
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    include_metrics=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
)
```

### Resume Interrupted Experiment

```python
# If experiment crashes, resume from checkpoint
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 25, 30],
    resume=True  # Resume from checkpoint
)
```

---

## Output Files

After running an experiment, find results in:

```
experiments/my_experiment_20260318_143022/
├── config.json           # Experiment configuration
├── results.json          # Full results (metrics, configs)
├── results.csv           # Tabular results
├── runs/
│   ├── run_000.csv       # Clustering output for run 0
│   ├── run_001.csv       # Clustering output for run 1
│   └── ...
└── visualizations/
    ├── Silhouette_vs_param.png
    ├── Davies-Bouldin_vs_param.png
    └── ...
```

---

## Tips

1. **Start simple**: Test one parameter at a time before grid search
2. **Use parallel execution**: Set `n_workers=4` for faster experiments
3. **Filter metrics**: Use `include_metrics` for faster iteration
4. **Save intermediate outputs**: Clustering CSVs in `runs/` can be reanalyzed
5. **Set random seed**: Default seed=42 ensures reproducibility
