# Data Model: Labricate

**Date**: March 22, 2026 (Updated)  
**Feature**: 004-labricate

---

## Entities

### 1. PipelineConfig

Configuration for a single pipeline execution (BERTopic-based).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| random_seed | int \| None | No | Random seed for reproducibility (default: 42) |
| umap | UMAPConfig | Yes | UMAP dimensionality reduction settings |
| clustering_algorithm | str | Yes | "hdbscan" or "kmeans" |
| hdbscan | HDBSCANConfig | No | HDBSCAN settings (required if algorithm="hdbscan") |
| kmeans | KMeansConfig | No | K-Means settings (required if algorithm="kmeans") |
| enable_topic_representation | bool | No | Enable BERTopic c-TF-IDF (default: false for speed) |
| calculate_probabilities | bool | No | Calculate topic probabilities (default: false) |

### 2. UMAPConfig

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| n_neighbors | int | No | 15 | Number of neighbors for local structure |
| n_components | int | No | 5 | Target dimensionality |
| min_dist | float | No | 0.0 | Minimum distance between points |
| metric | str | No | "cosine" | Distance metric |
| repulsion_strength | float | No | 1.0 | Repulsion strength parameter |
| low_memory | bool | No | false | Enable low-memory mode for large datasets |

### 3. HDBSCANConfig

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| min_cluster_size | int | No | 10 | Minimum cluster size |
| min_samples | int | No | 10 | Minimum samples for core points |
| cluster_selection_method | str | No | "eom" | "eom" or "leaf" |
| metric | str | No | "euclidean" | Distance metric for clustering |

### 4. KMeansConfig

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| n_clusters | int | Yes | 10 | Number of clusters |

### 5. ExperimentConfig

Configuration for an experiment session.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| name | str | No | auto-generated | Experiment name |
| base_config | PipelineConfig | Yes | - | Base pipeline configuration |
| param | str | Yes* | - | Dot-notation parameter path (single-param mode) |
| values | list | Yes* | - | Values to test (single-param mode) |
| params | dict[str, list] | Yes* | - | Parameter paths to value lists (grid mode) |
| output_dir | str | No | "./experiments" | Output directory |
| output_format | str | No | "json" | "json", "csv", or "both" |
| n_workers | int | No | 1 | Number of parallel workers |
| error_handling | str | No | "continue" | "continue" or "fail_fast" |
| include_metrics | list[str] | No | None | Metrics to include (None = all) |
| exclude_metrics | list[str] | No | None | Metrics to exclude |
| ground_truth | np.ndarray | No | None | Ground truth labels for supervised metrics |

*Either (param + values) OR params is required, not both.

### 6. PipelineResult

Output from a single pipeline run.

| Field | Type | Description |
|-------|------|-------------|
| run_id | int | Sequential run identifier |
| config | PipelineConfig | Configuration used for this run |
| labels | np.ndarray | Cluster labels (1D, length n_samples) |
| reduced_embeddings | np.ndarray | Reduced embeddings (2D, n_samples × n_components) |
| n_clusters | int | Number of clusters found (excluding noise) |
| n_noise | int | Number of noise points (label=-1) |
| timing | TimingInfo | Execution timing breakdown |
| status | str | "completed", "failed", or "skipped" |
| error | str \| None | Error message if failed |

### 7. TimingInfo

| Field | Type | Description |
|-------|------|-------------|
| bertopic_seconds | float | BERTopic fit_transform time (UMAP + clustering) |
| evaluation_seconds | float | Metricate evaluation time |
| total_seconds | float | Total run time |

### 8. MetricResult

Single metric evaluation (from Metricate).

| Field | Type | Description |
|-------|------|-------------|
| name | str | Metric name |
| value | float | Calculated value |
| range | tuple[float, float] | Valid range |
| direction | str | "higher" or "lower" is better |

### 9. RunResult

Complete result for a single experiment run.

| Field | Type | Description |
|-------|------|-------------|
| run_id | int | Sequential run identifier |
| param_values | dict | Parameter path → value mapping for this run |
| pipeline_result | PipelineResult | Pipeline execution result |
| metrics | list[MetricResult] | Metricate evaluation results |

### 10. ExperimentResult

Complete experiment output.

| Field | Type | Description |
|-------|------|-------------|
| experiment_id | str | Unique identifier (timestamp-based) |
| experiment_name | str | User-provided or auto-generated name |
| config | ExperimentConfig | Experiment configuration |
| runs | list[RunResult] | All run results |
| summary | ExperimentSummary | Aggregated statistics |
| output_path | str | Path to saved results |

### 11. ExperimentSummary

| Field | Type | Description |
|-------|------|-------------|
| total_runs | int | Total number of runs planned |
| completed_runs | int | Successfully completed runs |
| failed_runs | int | Failed runs |
| skipped_runs | int | Skipped runs (resumed) |
| total_duration_seconds | float | Total experiment time |

### 12. Checkpoint

Resume state for interrupted experiments.

| Field | Type | Description |
|-------|------|-------------|
| experiment_id | str | Experiment identifier |
| config | ExperimentConfig | Original experiment config |
| config_hash | str | Hash of original config for mismatch detection |
| total_runs | int | Total runs planned |
| completed_run_ids | list[int] | IDs of completed runs |
| failed_run_ids | list[int] | IDs of failed runs |
| current_run_id | int | Next run to execute |
| partial_results | dict[int, RunResult] | Results for completed runs |
| created_at | str | ISO timestamp |
| updated_at | str | ISO timestamp |

---

## State Transitions

### Experiment States

```
INITIALIZED → RUNNING → COMPLETED
                ↓
             PAUSED (checkpoint saved)
                ↓
             RESUMED → RUNNING → COMPLETED
```

### Run States

```
PENDING → RUNNING → COMPLETED
              ↓
           FAILED (error logged)
              ↓
           SKIPPED (if resumed)
```

---

## Validation Rules

### PipelineConfig
- `clustering_algorithm` must be "hdbscan" or "kmeans"
- If "hdbscan", `hdbscan` config must be present
- If "kmeans", `kmeans.n_clusters` must be > 0

### ExperimentConfig
- Either `param` + `values` (single-param) OR `params` (grid) required
- `param` must be valid dot-notation path in `base_config`
- All paths in `params` must be valid dot-notation paths
- `n_workers` must be >= 1
- `error_handling` must be "continue" or "fail_fast"
- `output_format` must be "json", "csv", or "both"

### Parameter Paths
- Must match pattern: `<section>.<key>` or `<key>` (top-level)
- Valid sections: "umap", "hdbscan", "kmeans"
- Key must exist in corresponding section

---

## File Outputs

### Directory Structure

```
experiments/
└── {experiment_name}_{timestamp}/
    ├── config.json           # ExperimentConfig
    ├── checkpoint.json       # Checkpoint (during run)
    ├── results.json          # ExperimentResult (if json)
    ├── results.csv           # Flattened results (if csv)
    ├── runs/
    │   ├── run_000.csv       # Clustering output (labels + embeddings)
    │   ├── run_001.csv
    │   └── ...
    └── visualizations/
        ├── silhouette_vs_param.png
        ├── davies_bouldin_vs_param.png
        └── ...
```

### CSV Results Schema

| Column | Type | Description |
|--------|------|-------------|
| run_id | int | Run identifier |
| {param_path} | varies | Value of varying parameter |
| n_clusters | int | Clusters found |
| n_noise | int | Noise points |
| Silhouette | float | Metric value |
| Davies-Bouldin | float | Metric value |
| ... | float | (all 34 metrics) |
| status | str | Run status |
| duration_seconds | float | Total run time |
