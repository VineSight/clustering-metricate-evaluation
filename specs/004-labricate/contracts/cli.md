# CLI Contract: Labricate

**Date**: March 18, 2026  
**Feature**: 004-labricate

---

## Command Structure

```
metricate labricate <subcommand> [options]
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `experiment` | Run hyperparameter experiment |
| `resume` | Resume interrupted experiment |
| `validate` | Validate config file |

---

## metricate labricate experiment

Run a hyperparameter experiment.

### Usage

```bash
metricate labricate experiment \
    --embeddings <path> \
    --config <path> \
    --param <dot.path> \
    --values <v1,v2,v3,...> \
    [options]
```

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--embeddings`, `-e` | path | Path to embeddings file (CSV with dim_* columns) |
| `--config`, `-c` | path | Path to JSON config file |
| `--param`, `-p` | string | Dot-notation parameter path to vary |
| `--values`, `-v` | string | Comma-separated values to test |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir`, `-o` | path | ./experiments | Output directory |
| `--output-format`, `-f` | choice | json | Output format: json, csv, both |
| `--workers`, `-w` | int | 1 | Number of parallel workers |
| `--error-handling` | choice | continue | Error mode: continue, fail_fast |
| `--include-metrics` | string | (all) | Comma-separated metrics to include |
| `--exclude-metrics` | string | (none) | Comma-separated metrics to exclude |
| `--resume` | flag | false | Resume from checkpoint if exists |
| `--force` | flag | false | Force start fresh when config mismatch detected |
| `--ground-truth` | path | (none) | Path to ground truth labels CSV for supervised metrics |
| `--name`, `-n` | string | (auto) | Experiment name |
| `--quiet`, `-q` | flag | false | Suppress progress output |
| `--verbose` | flag | false | Show detailed timing logs |

### Examples

```bash
# Basic single-parameter experiment
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    -p "hdbscan.min_cluster_size" \
    -v "5,10,15,20,25,30"

# With parallelism and custom output
metricate labricate experiment \
    --embeddings data/embeddings.csv \
    --config configs/base.json \
    --param "umap.n_neighbors" \
    --values "5,10,15,20,30,50" \
    --workers 4 \
    --output-dir ./results \
    --output-format both \
    --name "neighbors_sweep"

# Resume interrupted experiment
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    -p "hdbscan.min_cluster_size" \
    -v "5,10,15,20,25,30" \
    --resume

# With metric filtering
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    -p "kmeans.n_clusters" \
    -v "5,10,15,20,25" \
    --include-metrics "Silhouette,Davies-Bouldin,Calinski-Harabasz"
```

### Output

```
Labricate Experiment: neighbors_sweep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Config: configs/base.json
Parameter: umap.n_neighbors
Values: [5, 10, 15, 20, 30, 50]
Workers: 4

Running experiments...
Run 1/6: umap.n_neighbors=5   [████████████████████████████████] 100%
  UMAP: 2.3s | Clustering: 0.4s | Evaluation: 5.1s | Total: 7.8s
Run 2/6: umap.n_neighbors=10  [████████████████████████████████] 100%
  UMAP: 2.1s | Clustering: 0.3s | Evaluation: 4.9s | Total: 7.3s
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment Complete

Total runs: 6
Completed: 6 | Failed: 0 | Skipped: 0
Duration: 45.2s

Results saved to: ./results/neighbors_sweep_20260318_143022/
  - config.json
  - results.json
  - results.csv
  - runs/ (6 files)
  - visualizations/ (34 charts)
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments |
| 2 | Config validation error |
| 3 | Experiment failed (all runs failed) |
| 4 | Partial failure (some runs failed, results saved) |

---

## metricate labricate experiment --grid

Run a grid search experiment (multiple parameters).

### Usage

```bash
metricate labricate experiment \
    --embeddings <path> \
    --config <path> \
    --grid \
    --params "<path1>=<v1,v2,...>" \
    --params "<path2>=<v1,v2,...>" \
    [options]
```

### Example

```bash
metricate labricate experiment \
    -e embeddings.csv \
    -c config.json \
    --grid \
    --params "umap.n_neighbors=10,15,20" \
    --params "hdbscan.min_cluster_size=5,10,15" \
    --workers 4
```

This runs 3 × 3 = 9 combinations.

---

## metricate labricate resume

Resume an interrupted experiment from its checkpoint.

### Usage

```bash
metricate labricate resume <experiment_dir>
```

### Example

```bash
metricate labricate resume ./experiments/neighbors_sweep_20260318_143022/
```

### Output

```
Found checkpoint: 4/6 runs completed
Resume experiment? [y/N]: y

Resuming from run 5...
```

### Config Mismatch Handling

```
metricate labricate resume ./experiments/my_experiment/

Error: Config mismatch detected!
Checkpoint was created with different configuration.
Use --force to start fresh, discarding previous results.
Run 5/6: umap.n_neighbors=30  [████████████████████████████████] 100%
Run 6/6: umap.n_neighbors=50  [████████████████████████████████] 100%

Experiment Complete
```

---

## metricate labricate validate

Validate a configuration file.

### Usage

```bash
metricate labricate validate <config_path> [--param <dot.path>]
```

### Examples

```bash
# Validate config structure
metricate labricate validate config.json

# Validate specific parameter path
metricate labricate validate config.json --param "hdbscan.min_cluster_size"
```

### Output

```
✓ Config file is valid
✓ Parameter path 'hdbscan.min_cluster_size' is valid (type: int, current: 10)
```

Or on error:

```
✗ Invalid parameter path: 'hdbscan.invalid_param'
  Available paths in 'hdbscan' section:
    - hdbscan.min_cluster_size (int)
    - hdbscan.min_samples (int)
    - hdbscan.cluster_selection_method (str)
```

---

## Config File Format

Expected JSON structure:

```json
{
  "random_seed": 42,
  "umap": {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "repulsion_strength": 1.0
  },
  "clustering_algorithm": "hdbscan",
  "hdbscan": {
    "min_cluster_size": 10,
    "min_samples": 10,
    "cluster_selection_method": "eom"
  },
  "kmeans": {
    "n_clusters": 10
  }
}
```

---

## Embeddings File Format

CSV with embedding columns (auto-detected):

```csv
dim_0,dim_1,dim_2,dim_3,dim_4
0.123,0.456,0.789,0.012,0.345
0.234,0.567,0.890,0.123,0.456
...
```

Supported column patterns:
- `dim_*`, `dim0`, `dim1`, ...
- `embedding_*`, `embedding_0`, ...
- `x_*`, `x0`, `x1`, ...
- Any numeric columns (if no pattern matches)
