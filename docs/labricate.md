# Labricate: Hyperparameter Experimentation

Labricate is Metricate's hyperparameter experimentation framework for finding optimal clustering configurations. It runs experiments with varying parameter values and evaluates each configuration using Metricate's comprehensive metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Python API](#python-api)
5. [Configuration Reference](#configuration-reference)
6. [Examples](#examples)
7. [CLI Reference](#cli-reference)
8. [Output Files](#output-files)
9. [Visualization](#visualization)
10. [Parallel Execution](#parallel-execution)
11. [Checkpoint & Resume](#checkpoint--resume)
12. [Custom Pipelines](#custom-pipelines)
13. [Tips & Best Practices](#tips--best-practices)

---

## Overview

### Why Labricate?

Clustering quality depends heavily on hyperparameters like:
- **UMAP**: `n_neighbors`, `n_components`, `min_dist`, `metric`
- **HDBSCAN**: `min_cluster_size`, `min_samples`, `cluster_selection_method`
- **K-Means**: `n_clusters`

Finding the right combination requires systematic experimentation. Labricate automates this by:

1. **Running experiments** with different parameter values
2. **Evaluating each run** with Metricate's 34 metrics
3. **Comparing results** to find optimal configurations
4. **Visualizing trends** across parameter sweeps
5. **Saving outputs** for reproducibility

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Experiment                            │
├─────────────────────────────────────────────────────────────┤
│  Embeddings + Config + Parameter Values                     │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  For each parameter combination:                     │   │
│  │    1. Modify config with new param value             │   │
│  │    2. Run pipeline (BERTopic or custom)              │   │
│  │    3. Evaluate with Metricate                        │   │
│  │    4. Store results                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ExperimentResult (runs, metrics, summary)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

Labricate is included with Metricate:

```bash
pip install metricate
```

Or from source:

```bash
git clone https://github.com/VineSight/clustering-metricate-evaluation.git
cd clustering-metricate-evaluation
pip install -e .
```

### Dependencies

Labricate automatically installs:
- `bertopic` for the default clustering pipeline
- `umap-learn` for dimensionality reduction
- `hdbscan` for density-based clustering
- `matplotlib` for visualization

---

## Quick Start

### 1. Prepare Your Data

```python
import numpy as np

# Load your pre-computed embeddings
embeddings = np.load("embeddings.npy")  # Shape: (n_samples, n_dims)
print(f"Shape: {embeddings.shape}")
# Output: Shape: (10000, 384)
```

### 2. Define Configuration

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

### 3. Run Experiment

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

# Find best configuration
best = result.get_best_run("Silhouette")
print(f"Best: min_cluster_size={best.param_values['hdbscan.min_cluster_size']}")
```

---

## Python API

### Importing

```python
from metricate.labricate import (
    # Core classes
    Experiment,
    BERTopicPipeline,
    
    # Result types
    ExperimentResult,
    RunResult,
    PipelineResult,
    ExperimentSummary,
    
    # Config helpers
    load_config,
    validate_config,
    
    # Utilities
    load_embeddings,
)
```

Or via metricate:

```python
from metricate import labricate

exp = labricate.Experiment(embeddings, config)
```

### Experiment Class

#### Constructor

```python
class Experiment:
    def __init__(
        self,
        embeddings: np.ndarray | pd.DataFrame | str,  # Array, DataFrame, or CSV path
        config: dict | str,                            # Config dict or JSON path
        name: str | None = None,                       # Experiment name (auto-generated if None)
        output_dir: str = "./experiments",             # Output directory
        output_format: Literal["json", "csv", "both"] = "json",
        pipeline: Callable | None = None,              # Custom pipeline (default: BERTopicPipeline)
    ) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | array, DataFrame, or str | Input embeddings in any supported format |
| `config` | dict or str | Base pipeline configuration |
| `name` | str or None | Experiment name (default: auto-generated timestamp) |
| `output_dir` | str | Directory for results (default: `./experiments`) |
| `output_format` | str | `"json"`, `"csv"`, or `"both"` |
| `pipeline` | Callable or None | Custom pipeline function (default: BERTopicPipeline) |

**Example:**

```python
# From numpy array
exp = Experiment(embeddings, config)

# From CSV file
exp = Experiment("embeddings.csv", "config.json")

# From pandas DataFrame
exp = Experiment(df, config, name="my_experiment")

# With custom output settings
exp = Experiment(
    embeddings,
    config,
    name="cluster_sweep",
    output_dir="./results",
    output_format="both"
)
```

#### run() Method

Run a single-parameter experiment.

```python
def run(
    self,
    param: str,                                        # Dot-notation parameter path
    values: list,                                      # Values to test
    n_workers: int = 1,                                # Parallel workers
    error_handling: Literal["continue", "fail_fast"] = "continue",
    include_metrics: list[str] | None = None,          # Specific metrics to calculate
    exclude_metrics: list[str] | None = None,          # Metrics to skip
    ground_truth: np.ndarray | None = None,            # For supervised metrics
    resume: bool = False,                              # Resume from checkpoint
    force: bool = False,                               # Force fresh start
    verbose: bool = True,                              # Print progress
) -> ExperimentResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `param` | str | Dot-notation path (e.g., `"hdbscan.min_cluster_size"`) |
| `values` | list | List of values to test |
| `n_workers` | int | Number of parallel workers (1 = sequential) |
| `error_handling` | str | `"continue"` or `"fail_fast"` |
| `include_metrics` | list or None | Only calculate these metrics |
| `exclude_metrics` | list or None | Skip these metrics |
| `ground_truth` | array or None | Ground truth labels for supervised metrics |
| `resume` | bool | Resume from checkpoint if exists |
| `force` | bool | Force fresh start if config mismatch |
| `verbose` | bool | Print progress information |

**Example:**

```python
# Basic usage
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 25, 30]
)

# Parallel execution with metric filtering
result = exp.run(
    param="umap.n_neighbors",
    values=[5, 10, 15, 20, 30, 50],
    n_workers=4,
    include_metrics=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
)

# With error handling
result = exp.run(
    param="hdbscan.min_samples",
    values=[1, 3, 5, 10],
    error_handling="continue"  # Continue even if some runs fail
)
```

#### run_grid() Method

Run a grid search over multiple parameters.

```python
def run_grid(
    self,
    params: dict[str, list],                           # Param paths to value lists
    n_workers: int = 1,
    error_handling: Literal["continue", "fail_fast"] = "continue",
    include_metrics: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
    ground_truth: np.ndarray | None = None,
    resume: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> ExperimentResult
```

**Example:**

```python
# 3x3 grid = 9 combinations
result = exp.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20],
        "hdbscan.min_cluster_size": [10, 20, 30]
    },
    n_workers=4
)

# 4x4x3 grid = 48 combinations
result = exp.run_grid(
    params={
        "umap.n_neighbors": [5, 10, 15, 20],
        "umap.n_components": [3, 5, 10, 15],
        "hdbscan.min_cluster_size": [10, 20, 30]
    },
    n_workers=8,
    exclude_metrics=["Gamma", "Tau", "G-plus"]  # Skip expensive metrics
)
```

#### validate_param() Method

Validate a parameter path against the config.

```python
def validate_param(self, param: str) -> bool
```

**Example:**

```python
try:
    exp.validate_param("hdbscan.min_cluster_size")  # Valid
    exp.validate_param("hdbscan.invalid_param")     # Raises ValueError
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

### ExperimentResult Class

Returned by `run()` and `run_grid()`.

```python
@dataclass
class ExperimentResult:
    experiment_id: str
    experiment_name: str
    config: dict
    runs: list[RunResult]
    summary: ExperimentSummary
    output_path: str | None
```

#### to_dataframe() Method

Convert results to a pandas DataFrame.

```python
df = result.to_dataframe()
print(df.columns.tolist())
# ['run_id', 'hdbscan.min_cluster_size', 'n_clusters', 'n_noise', 
#  'status', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', ...]
```

#### get_best_run() Method

Find the run with the best metric value.

```python
def get_best_run(
    self,
    metric: str,
    direction: str | None = None  # "higher" or "lower" (auto-detected)
) -> RunResult
```

**Example:**

```python
# Auto-detect direction from metric metadata
best_silhouette = result.get_best_run("Silhouette")  # higher is better
best_db = result.get_best_run("Davies-Bouldin")      # lower is better

# Override direction
best = result.get_best_run("custom_metric", direction="lower")
```

### RunResult Class

Result for a single experiment run.

```python
@dataclass
class RunResult:
    run_id: int
    param_values: dict[str, Any]
    pipeline_result: PipelineResult
    metrics: list[MetricResult]
```

**Example:**

```python
for run in result.runs:
    if run.pipeline_result.status == "completed":
        print(f"Run {run.run_id}: {run.param_values}")
        print(f"  Clusters: {run.pipeline_result.n_clusters}")
        print(f"  Noise: {run.pipeline_result.n_noise}")
        for metric in run.metrics:
            print(f"  {metric.name}: {metric.value:.4f}")
```

### PipelineResult Class

Output from pipeline execution.

```python
@dataclass
class PipelineResult:
    run_id: int
    config: dict
    labels: np.ndarray
    reduced_embeddings: np.ndarray
    n_clusters: int
    n_noise: int
    timing: TimingInfo
    status: str  # "completed", "failed", "skipped"
    error: str | None
```

### ExperimentSummary Class

Aggregate statistics for the experiment.

```python
@dataclass
class ExperimentSummary:
    total_runs: int
    completed_runs: int
    failed_runs: int
    skipped_runs: int
    total_duration_seconds: float
```

**Example:**

```python
print(result.summary)
# ExperimentSummary(
#     total_runs=6,
#     completed_runs=6,
#     failed_runs=0,
#     total_duration_seconds=45.2
# )
```

### Utility Functions

#### load_embeddings()

Load embeddings from various formats.

```python
from metricate.labricate import load_embeddings

# From numpy file
embeddings = load_embeddings("embeddings.npy")

# From CSV file
embeddings = load_embeddings("data.csv")

# From CSV with specific columns
embeddings = load_embeddings("data.csv", embedding_cols=["dim_0", "dim_1", "dim_2"])

# From DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
embeddings = load_embeddings(df)

# From numpy array (passthrough)
arr = np.random.randn(1000, 50)
embeddings = load_embeddings(arr)
```

#### load_config()

Load configuration from JSON file.

```python
from metricate.labricate import load_config

config = load_config("config.json")
```

#### validate_config()

Validate configuration structure.

```python
from metricate.labricate import validate_config

errors = validate_config(config)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Config is valid!")
```

---

## Configuration Reference

### Full Configuration Schema

```python
config = {
    # Global settings
    "random_seed": 42,                    # For reproducibility
    
    # UMAP dimensionality reduction
    "umap": {
        "n_neighbors": 15,                # Local neighborhood size
        "n_components": 5,                # Output dimensions
        "min_dist": 0.0,                  # Minimum distance between points
        "metric": "cosine",               # Distance metric
        "repulsion_strength": 1.0,        # Repulsion strength
        "low_memory": False,              # Use low memory algorithm
    },
    
    # Clustering algorithm choice
    "clustering_algorithm": "hdbscan",    # "hdbscan" or "kmeans"
    
    # HDBSCAN settings (when clustering_algorithm="hdbscan")
    "hdbscan": {
        "min_cluster_size": 10,           # Minimum cluster size
        "min_samples": 10,                # Core point threshold
        "cluster_selection_method": "eom",# "eom" or "leaf"
        "metric": "euclidean",            # Distance metric
    },
    
    # K-Means settings (when clustering_algorithm="kmeans")
    "kmeans": {
        "n_clusters": 10,                 # Number of clusters
    },
    
    # BERTopic-specific settings
    "enable_topic_representation": False, # Skip c-TF-IDF for speed
    "calculate_probabilities": False,     # Skip probability calculation
}
```

### Parameter Paths

Use dot notation to reference nested parameters:

| Path | Description |
|------|-------------|
| `random_seed` | Global random seed |
| `umap.n_neighbors` | UMAP neighborhood size |
| `umap.n_components` | UMAP output dimensions |
| `umap.min_dist` | UMAP minimum distance |
| `umap.metric` | UMAP distance metric |
| `clustering_algorithm` | Algorithm choice |
| `hdbscan.min_cluster_size` | HDBSCAN minimum cluster size |
| `hdbscan.min_samples` | HDBSCAN core point threshold |
| `hdbscan.cluster_selection_method` | HDBSCAN selection method |
| `kmeans.n_clusters` | K-Means cluster count |

### Config as JSON File

```json
{
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

---

## Examples

### Example 1: UMAP n_neighbors Sweep

```python
from metricate.labricate import Experiment
import numpy as np

# Load embeddings
embeddings = np.load("my_embeddings.npy")

# Base config
config = {
    "random_seed": 42,
    "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine"},
    "clustering_algorithm": "hdbscan",
    "hdbscan": {"min_cluster_size": 10, "min_samples": 5}
}

# Create experiment
exp = Experiment(embeddings, config, name="umap_neighbors_sweep")

# Test n_neighbors values
result = exp.run(
    param="umap.n_neighbors",
    values=[5, 10, 15, 20, 30, 50, 75, 100]
)

# Analyze
df = result.to_dataframe()
print(df[["umap.n_neighbors", "n_clusters", "Silhouette", "Davies-Bouldin"]])

# Find best
best = result.get_best_run("Silhouette")
print(f"\nBest n_neighbors: {best.param_values['umap.n_neighbors']}")
```

### Example 2: UMAP n_components Sweep

```python
# Test output dimensions
result = exp.run(
    param="umap.n_components",
    values=[2, 3, 5, 10, 15, 20, 30]
)

df = result.to_dataframe()
print(df[["umap.n_components", "n_clusters", "Silhouette"]])
```

### Example 3: HDBSCAN min_cluster_size Sweep

```python
# Most important HDBSCAN parameter
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50, 100, 200]
)

df = result.to_dataframe()
print("Effect of min_cluster_size:")
print(df[["hdbscan.min_cluster_size", "n_clusters", "n_noise", "Silhouette"]])
```

### Example 4: HDBSCAN min_samples Sweep

```python
# Test core point threshold
result = exp.run(
    param="hdbscan.min_samples",
    values=[1, 3, 5, 10, 15, 20]
)

df = result.to_dataframe()
print(df[["hdbscan.min_samples", "n_clusters", "n_noise", "Silhouette"]])
```

### Example 5: K-Means n_clusters Sweep

```python
# Use K-Means instead of HDBSCAN
config = {
    "random_seed": 42,
    "umap": {"n_neighbors": 15, "n_components": 5},
    "clustering_algorithm": "kmeans",
    "kmeans": {"n_clusters": 10}
}

exp = Experiment(embeddings, config, name="kmeans_sweep")

result = exp.run(
    param="kmeans.n_clusters",
    values=[5, 10, 15, 20, 25, 30, 40, 50]
)

df = result.to_dataframe()
print(df[["kmeans.n_clusters", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]])

# Find optimal k using elbow method
best_db = result.get_best_run("Davies-Bouldin", direction="lower")
print(f"Optimal k (Davies-Bouldin): {best_db.param_values['kmeans.n_clusters']}")
```

### Example 6: Grid Search (2 Parameters)

```python
# Sweep UMAP and HDBSCAN together
result = exp.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20],
        "hdbscan.min_cluster_size": [10, 20, 30]
    },
    n_workers=4
)

print(f"Total combinations: {result.summary.total_runs}")

df = result.to_dataframe()
print(df[["umap.n_neighbors", "hdbscan.min_cluster_size", "n_clusters", "Silhouette"]])

# Find best combination
best = result.get_best_run("Silhouette")
print(f"\nBest combination:")
print(f"  n_neighbors: {best.param_values['umap.n_neighbors']}")
print(f"  min_cluster_size: {best.param_values['hdbscan.min_cluster_size']}")
```

### Example 7: Grid Search (3 Parameters)

```python
# Full grid search
result = exp.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20],
        "umap.n_components": [5, 10],
        "hdbscan.min_cluster_size": [10, 20, 30]
    },
    n_workers=8,
    exclude_metrics=["Gamma", "Tau", "G-plus"]  # Skip expensive metrics
)

print(f"Total combinations: {result.summary.total_runs}")  # 3 × 2 × 3 = 18
```

### Example 8: Filtering Metrics for Speed

```python
# Only calculate fast metrics (much faster)
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50],
    include_metrics=[
        "Silhouette",
        "Davies-Bouldin", 
        "Calinski-Harabasz",
        "Dunn Index"
    ]
)

# Or exclude slow metrics
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50],
    exclude_metrics=["Gamma", "Tau", "G-plus", "Point-Biserial"]
)
```

### Example 9: Resume Interrupted Experiment

```python
# If experiment crashes mid-way, resume from checkpoint
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 25, 30, 40, 50],  # Same values
    resume=True  # Will skip already-completed runs
)

# Force fresh start even if checkpoint exists
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    resume=True,
    force=True  # Override config mismatch
)
```

### Example 10: Load Embeddings from CSV

```python
# CSV with embeddings
# cluster_id,dim_0,dim_1,dim_2,...,dim_49
# 0,0.123,0.456,...
# 1,0.234,0.567,...

exp = Experiment(
    embeddings="my_data.csv",
    config="config.json",
    name="csv_experiment"
)

result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[10, 20, 30]
)
```

### Example 11: Compare Multiple Metrics

```python
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50, 100]
)

df = result.to_dataframe()

# Find best for each metric
metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Dunn Index"]
for metric in metrics:
    direction = "lower" if metric == "Davies-Bouldin" else "higher"
    best = result.get_best_run(metric, direction=direction)
    print(f"Best {metric}: min_cluster_size={best.param_values['hdbscan.min_cluster_size']}")
```

### Example 12: Export Results

```python
# Results automatically saved based on output_format
exp = Experiment(
    embeddings,
    config,
    name="export_example",
    output_dir="./results",
    output_format="both"  # JSON + CSV
)

result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[10, 20, 30]
)

# Also export DataFrame manually
df = result.to_dataframe()
df.to_csv("my_results.csv", index=False)
df.to_excel("my_results.xlsx", index=False)
```

---

## CLI Reference

### experiment Command

Run experiments from the command line.

```bash
metricate labricate experiment [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--embeddings PATH` | Path to embeddings (CSV or NPY) |
| `--config PATH` | Path to config JSON |
| `--param TEXT` | Parameter path for single-param experiment |
| `--values TEXT` | Comma-separated values (e.g., "5,10,15,20") |
| `--grid TEXT` | Grid search params (e.g., "param1=1,2;param2=a,b") |
| `--output-dir PATH` | Output directory (default: ./experiments) |
| `--output-format TEXT` | json, csv, or both |
| `--workers INT` | Number of parallel workers |
| `--include-metrics TEXT` | Comma-separated metrics to include |
| `--exclude-metrics TEXT` | Comma-separated metrics to exclude |
| `--verbose / --no-verbose` | Print progress |

**Examples:**

```bash
# Single parameter experiment
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values "5,10,15,20,30,50"

# With parallel execution
metricate labricate experiment \
    --embeddings embeddings.npy \
    --config config.json \
    --param "umap.n_neighbors" \
    --values "5,10,15,20,30,50" \
    --workers 4

# Grid search
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --grid "hdbscan.min_cluster_size=5,10,15;umap.n_neighbors=10,15,20"

# With metric filtering
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values "10,20,30" \
    --exclude-metrics "Gamma,Tau,G-plus"

# Output to both formats
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values "10,20,30" \
    --output-format both \
    --output-dir ./my_results
```

### validate Command

Validate config and embeddings.

```bash
metricate labricate validate [OPTIONS]
```

**Examples:**

```bash
# Validate config
metricate labricate validate --config config.json

# Validate embeddings
metricate labricate validate --embeddings embeddings.csv

# Validate both
metricate labricate validate --config config.json --embeddings embeddings.csv
```

### resume Command

Resume an interrupted experiment.

```bash
metricate labricate resume EXPERIMENT_DIR [OPTIONS]
```

**Examples:**

```bash
# Resume from checkpoint
metricate labricate resume ./experiments/my_experiment_20260318_143022/

# Force resume with config override
metricate labricate resume ./experiments/my_experiment_20260318_143022/ --force
```

---

## Output Files

After running an experiment, results are saved to:

```
experiments/my_experiment_20260318_143022/
├── config.json           # Experiment configuration
├── results.json          # Full results with metrics
├── results.csv           # Tabular results (if output_format includes csv)
├── checkpoint.json       # For resuming interrupted experiments
├── runs/
│   ├── run_001.csv       # Clustering output for run 1
│   ├── run_002.csv       # Clustering output for run 2
│   └── ...
└── visualizations/
    ├── Silhouette_vs_min_cluster_size.png
    ├── Davies-Bouldin_vs_min_cluster_size.png
    └── ...
```

### results.json Structure

```json
{
  "experiment_id": "my_experiment_20260318_143022",
  "experiment_name": "my_experiment",
  "config": {
    "base_config": { ... },
    "param": "hdbscan.min_cluster_size",
    "values": [5, 10, 15, 20]
  },
  "runs": [
    {
      "run_id": 1,
      "param_values": {"hdbscan.min_cluster_size": 5},
      "n_clusters": 42,
      "n_noise": 150,
      "status": "completed",
      "metrics": {
        "Silhouette": 0.312,
        "Davies-Bouldin": 1.823,
        ...
      }
    },
    ...
  ],
  "summary": {
    "total_runs": 4,
    "completed_runs": 4,
    "failed_runs": 0,
    "total_duration_seconds": 45.2
  }
}
```

### results.csv Structure

```csv
run_id,hdbscan.min_cluster_size,n_clusters,n_noise,status,Silhouette,Davies-Bouldin,Calinski-Harabasz,...
1,5,42,150,completed,0.312,1.823,1523.45,...
2,10,28,89,completed,0.398,1.456,1876.23,...
3,15,19,52,completed,0.425,1.312,2102.89,...
4,20,14,31,completed,0.441,1.289,2234.56,...
```

---

## Visualization

### Line Charts (Single Parameter)

```python
from metricate.labricate.output.visualization import plot_metric_vs_param

# Create line chart
fig = plot_metric_vs_param(
    result,
    metric="Silhouette",
    param="hdbscan.min_cluster_size",
    output_path="silhouette_trend.png"
)

# Multiple metrics
for metric in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]:
    plot_metric_vs_param(
        result,
        metric=metric,
        param="hdbscan.min_cluster_size",
        output_path=f"{metric}_vs_min_cluster_size.png"
    )
```

### Heatmaps (Grid Search)

```python
from metricate.labricate.output.visualization import plot_heatmap

# Grid search result
result = exp.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20],
        "hdbscan.min_cluster_size": [10, 20, 30]
    }
)

# Create heatmap
fig = plot_heatmap(
    result,
    metric="Silhouette",
    param_x="hdbscan.min_cluster_size",
    param_y="umap.n_neighbors",
    output_path="silhouette_heatmap.png"
)
```

### Using matplotlib Directly

```python
import matplotlib.pyplot as plt

df = result.to_dataframe()

# Custom plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Silhouette vs parameter
axes[0, 0].plot(df["hdbscan.min_cluster_size"], df["Silhouette"], "b-o")
axes[0, 0].set_xlabel("min_cluster_size")
axes[0, 0].set_ylabel("Silhouette")
axes[0, 0].set_title("Silhouette Score")

# Plot 2: Davies-Bouldin vs parameter
axes[0, 1].plot(df["hdbscan.min_cluster_size"], df["Davies-Bouldin"], "r-o")
axes[0, 1].set_xlabel("min_cluster_size")
axes[0, 1].set_ylabel("Davies-Bouldin")
axes[0, 1].set_title("Davies-Bouldin Index")

# Plot 3: Number of clusters
axes[1, 0].bar(df["hdbscan.min_cluster_size"], df["n_clusters"])
axes[1, 0].set_xlabel("min_cluster_size")
axes[1, 0].set_ylabel("Number of Clusters")
axes[1, 0].set_title("Cluster Count")

# Plot 4: Noise points
axes[1, 1].bar(df["hdbscan.min_cluster_size"], df["n_noise"])
axes[1, 1].set_xlabel("min_cluster_size")
axes[1, 1].set_ylabel("Noise Points")
axes[1, 1].set_title("Noise Count")

plt.tight_layout()
plt.savefig("analysis.png")
```

---

## Parallel Execution

Labricate supports parallel execution for faster experiments.

### Basic Usage

```python
# Run with 4 workers
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50],
    n_workers=4
)
```

### Worker Count

Workers are automatically capped at CPU count:

```python
import os
print(f"CPU count: {os.cpu_count()}")

# Requesting more workers than CPUs will be capped
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    n_workers=100  # Will be capped to CPU count
)
```

### Error Handling

```python
# Continue on failure (default)
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    n_workers=4,
    error_handling="continue"  # Failed runs recorded, others continue
)

# Fail fast
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    n_workers=4,
    error_handling="fail_fast"  # Stop on first failure
)

# Check for failures
if result.summary.failed_runs > 0:
    print(f"Warning: {result.summary.failed_runs} runs failed")
    for run in result.runs:
        if run.pipeline_result.status == "failed":
            print(f"  Run {run.run_id}: {run.pipeline_result.error}")
```

---

## Checkpoint & Resume

Long experiments can be resumed if interrupted.

### How Checkpoints Work

1. After each completed run, state is saved to `checkpoint.json`
2. Checkpoint includes completed run IDs and config hash
3. On resume, completed runs are skipped

### Resuming an Experiment

```python
# First run (interrupted)
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50],  # Runs 1-3 complete, then crash
)

# Resume (continues from run 4)
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 30, 50],  # Same values!
    resume=True
)
```

### Config Mismatch Handling

```python
# If you change values after starting, use force=True
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 25, 30],  # Added 25
    resume=True,
    force=True  # Override config mismatch, restart fresh
)
```

### CLI Resume

```bash
metricate labricate resume ./experiments/my_experiment_20260318_143022/

# Force restart
metricate labricate resume ./experiments/my_experiment_20260318_143022/ --force
```

---

## Custom Pipelines

Create your own clustering pipeline for non-BERTopic workflows.

### Pipeline Protocol

A pipeline must:
1. Accept `(embeddings, config)` as arguments
2. Return `(labels, reduced_embeddings)` as tuple

```python
def my_pipeline(
    embeddings: np.ndarray,  # Shape: (n_samples, n_dims)
    config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        labels: 1D array of cluster assignments (length n_samples)
        reduced_embeddings: 2D array (n_samples, reduced_dims)
    """
    # Your clustering logic here
    return labels, reduced_embeddings
```

### Example: PCA + Agglomerative Clustering

```python
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def pca_agglomerative_pipeline(embeddings: np.ndarray, config: dict):
    # Normalize
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # PCA reduction
    pca = PCA(n_components=config["pca"]["n_components"])
    reduced = pca.fit_transform(normalized)
    
    # Agglomerative clustering
    clusterer = AgglomerativeClustering(
        n_clusters=config["agg"]["n_clusters"],
        linkage=config["agg"].get("linkage", "ward")
    )
    labels = clusterer.fit_predict(reduced)
    
    return labels, reduced

# Usage
config = {
    "pca": {"n_components": 10},
    "agg": {"n_clusters": 15, "linkage": "ward"}
}

exp = Experiment(
    embeddings,
    config,
    pipeline=pca_agglomerative_pipeline
)

result = exp.run(
    param="agg.n_clusters",
    values=[5, 10, 15, 20, 25, 30]
)
```

### Example: UMAP + Spectral Clustering

```python
from umap import UMAP
from sklearn.cluster import SpectralClustering

def umap_spectral_pipeline(embeddings: np.ndarray, config: dict):
    # UMAP reduction
    umap_model = UMAP(
        n_neighbors=config["umap"]["n_neighbors"],
        n_components=config["umap"]["n_components"],
        min_dist=config["umap"]["min_dist"],
        random_state=config.get("random_seed", 42)
    )
    reduced = umap_model.fit_transform(embeddings)
    
    # Spectral clustering
    spectral = SpectralClustering(
        n_clusters=config["spectral"]["n_clusters"],
        affinity=config["spectral"].get("affinity", "rbf"),
        random_state=config.get("random_seed", 42)
    )
    labels = spectral.fit_predict(reduced)
    
    return labels, reduced

config = {
    "random_seed": 42,
    "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0},
    "spectral": {"n_clusters": 10, "affinity": "rbf"}
}

exp = Experiment(embeddings, config, pipeline=umap_spectral_pipeline)
result = exp.run(param="spectral.n_clusters", values=[5, 10, 15, 20])
```

### Example: Custom Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from hdbscan import HDBSCAN

def preprocessed_bertopic_pipeline(embeddings: np.ndarray, config: dict):
    # Custom preprocessing
    if config.get("preprocessing", {}).get("normalize", False):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    if config.get("preprocessing", {}).get("standardize", False):
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    # UMAP
    umap_model = UMAP(
        n_neighbors=config["umap"]["n_neighbors"],
        n_components=config["umap"]["n_components"],
        random_state=config.get("random_seed", 42)
    )
    reduced = umap_model.fit_transform(embeddings)
    
    # HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=config["hdbscan"]["min_cluster_size"],
        min_samples=config["hdbscan"]["min_samples"]
    )
    labels = hdbscan_model.fit_predict(reduced)
    
    return labels, reduced

config = {
    "random_seed": 42,
    "preprocessing": {"normalize": True, "standardize": False},
    "umap": {"n_neighbors": 15, "n_components": 5},
    "hdbscan": {"min_cluster_size": 10, "min_samples": 5}
}

exp = Experiment(embeddings, config, pipeline=preprocessed_bertopic_pipeline)
```

---

## Tips & Best Practices

### 1. Start Simple

Test one parameter at a time before grid search:

```python
# First: find good UMAP settings
result1 = exp.run(param="umap.n_neighbors", values=[5, 10, 15, 20, 30])
best_neighbors = result1.get_best_run("Silhouette").param_values["umap.n_neighbors"]

# Update config
config["umap"]["n_neighbors"] = best_neighbors

# Then: find good HDBSCAN settings
result2 = exp.run(param="hdbscan.min_cluster_size", values=[5, 10, 15, 20, 30])
```

### 2. Use Parallel Execution

```python
# Always use multiple workers for large experiments
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=list(range(5, 101, 5)),  # 20 values
    n_workers=8
)
```

### 3. Filter Metrics for Iteration

```python
# During exploration, skip expensive metrics
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    include_metrics=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
)

# Final run with all metrics
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[best_value - 5, best_value, best_value + 5]
    # No metric filtering - get all metrics
)
```

### 4. Set Random Seed

Always use a random seed for reproducibility:

```python
config = {
    "random_seed": 42,  # Important!
    ...
}
```

### 5. Save Intermediate Results

Clustering CSVs in `runs/` can be reanalyzed:

```python
import pandas as pd

# Load a specific run's clustering
run_df = pd.read_csv("experiments/my_exp/runs/run_003.csv")
print(f"Clusters: {run_df['cluster_id'].nunique()}")
```

### 6. Check for Failures

```python
if result.summary.failed_runs > 0:
    print(f"⚠️ {result.summary.failed_runs} runs failed")
    for run in result.runs:
        if run.pipeline_result.status == "failed":
            print(f"  Run {run.run_id}: {run.pipeline_result.error}")
```

### 7. Use Meaningful Names

```python
exp = Experiment(
    embeddings,
    config,
    name="product_embeddings_hdbscan_sweep_v2"  # Descriptive name
)
```

### 8. Consider the Bias-Variance Tradeoff

- **Low min_cluster_size**: Many small clusters (high variance)
- **High min_cluster_size**: Few large clusters (high bias)

```python
# Test wide range first
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 20, 50, 100, 200]
)

# Then narrow down
best_approx = result.get_best_run("Silhouette").param_values["hdbscan.min_cluster_size"]
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=list(range(best_approx - 10, best_approx + 15, 5))
)
```

### 9. Monitor Noise Points

High noise count may indicate:
- Too high `min_cluster_size`
- Poor embedding quality
- Natural outliers in data

```python
df = result.to_dataframe()
print(df[["hdbscan.min_cluster_size", "n_clusters", "n_noise", "Silhouette"]])

# Filter runs with reasonable noise
df_filtered = df[df["n_noise"] < df["n_noise"].median()]
```

### 10. Use Ground Truth When Available

```python
# If you have ground truth labels
ground_truth = np.load("true_labels.npy")

result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20],
    ground_truth=ground_truth  # Enables supervised metrics
)

# Check supervised metrics
df = result.to_dataframe()
print(df[["hdbscan.min_cluster_size", "Adjusted Rand Index", "Normalized MI"]])
```

---

## See Also

- [Getting Started](getting-started.md) - Metricate basics
- [Metrics Reference](metrics-reference.md) - All 34 metrics explained
- [CLI Reference](cli-reference.md) - Full command-line documentation
- [API Reference](api-reference.md) - Complete Python API
