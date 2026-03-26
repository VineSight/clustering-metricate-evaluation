# CLI Reference

Metricate provides a comprehensive command-line interface for all major operations.

---

## Installation

The CLI is automatically available after installing metricate:

```bash
pip install metricate
```

Verify installation:

```bash
metricate --version
```

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `evaluate` | Evaluate clustering metrics for a CSV file |
| `compare` | Compare two clusterings and determine winner |
| `train` | Train metric weights from labeled data |
| `degrade` | Generate degraded versions of a clustering |
| `list` | List available metrics or degradation types |
| `labricate experiment` | Run hyperparameter experiments |
| `labricate validate` | Validate config and embeddings |
| `labricate resume` | Resume interrupted experiment |
| `web` | Start the browser-based UI |

---

## `metricate evaluate`

Evaluate clustering metrics for a CSV file.

### Usage

```bash
metricate evaluate CSV_PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CSV_PATH` | Path to the CSV file containing clustering data |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--label-col` | `-l` | auto | Column name for cluster labels |
| `--embedding-cols` | `-e` | auto | Comma-separated embedding column names |
| `--exclude` | `-x` | none | Comma-separated metric names to exclude |
| `--force-all` | `-f` | false | Force O(n²) metrics on large datasets |
| `--format` | `-F` | table | Output format: `table`, `json`, `csv`, `markdown` |
| `--output` | `-o` | stdout | Output file path |
| `--weights` | `-w` | none | Path to weights JSON for compound score |

### Examples

```bash
# Basic evaluation
metricate evaluate clustering.csv

# JSON output for scripting
metricate evaluate clustering.csv --format json

# Save to file
metricate evaluate clustering.csv --format csv -o results.csv

# Exclude expensive metrics
metricate evaluate clustering.csv --exclude Gamma,Tau,G-plus

# With compound score
metricate evaluate clustering.csv --weights weights.json

# Force all metrics on large data
metricate evaluate large_data.csv --force-all

# Specify columns explicitly
metricate evaluate data.csv -l cluster_id -e "dim_0,dim_1,dim_2"
```

---

## `metricate compare`

Compare two clusterings and determine the winner.

### Usage

```bash
metricate compare CSV_PATH_A CSV_PATH_B [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CSV_PATH_A` | Path to first CSV file |
| `CSV_PATH_B` | Path to second CSV file |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--label-col` | `-l` | auto | Column name for cluster labels |
| `--embedding-cols` | `-e` | auto | Comma-separated embedding column names |
| `--exclude` | `-x` | none | Comma-separated metric names to exclude |
| `--force-all` | `-f` | false | Force O(n²) metrics on large datasets |
| `--format` | `-F` | table | Output format: `table`, `json`, `csv`, `markdown` |
| `--output` | `-o` | stdout | Output file path |
| `--name-a` | | "A" | Label for first clustering |
| `--name-b` | | "B" | Label for second clustering |
| `--weights` | `-w` | none | Path to weights JSON for weighted winner |

### Examples

```bash
# Basic comparison
metricate compare v1.csv v2.csv

# With custom names
metricate compare old.csv new.csv --name-a "Old" --name-b "New"

# JSON output
metricate compare a.csv b.csv --format json -o comparison.json

# With weighted winner determination
metricate compare v1.csv v2.csv --weights weights.json
```

---

## `metricate train`

Train metric weights from a labeled training dataset.

### Usage

```bash
metricate train CSV_PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CSV_PATH` | Path to training CSV with normalized metrics and quality_score |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | none | Output file path for weights JSON |
| `--regularization` | `-r` | ridge | Type: `ridge` or `lasso` |
| `--alpha` | `-a` | 1.0 | Regularization strength |
| `--auto-alpha` | | false | Auto-select optimal alpha via CV |
| `--cv-splits` | | 5 | Number of cross-validation folds |
| `--skip-cv` | | false | Skip cross-validation |
| `--skip-sanity-check` | | false | Skip sanity check |
| `--top-n` | | 10 | Number of top features to display |

### Examples

```bash
# Basic training
metricate train training_data.csv -o weights.json

# Auto-tuned regularization
metricate train training_data.csv -o weights.json --auto-alpha

# Lasso for feature selection
metricate train training_data.csv -o weights.json -r lasso --auto-alpha

# Full options
metricate train training_data.csv \
  -o weights.json \
  --regularization ridge \
  --alpha 1.0 \
  --cv-splits 10 \
  --top-n 15

# Quick training (no CV, no sanity check)
metricate train training_data.csv -o weights.json --skip-cv --skip-sanity-check
```

---

## `metricate degrade`

Generate degraded versions of a clustering dataset.

### Usage

```bash
metricate degrade CSV_PATH OUTPUT_DIR [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CSV_PATH` | Path to input clustering CSV |
| `OUTPUT_DIR` | Directory to write degraded datasets |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--label-col` | `-l` | auto | Column name for cluster labels |
| `--embedding-cols` | `-e` | auto | Comma-separated embedding column names |
| `--types` | `-t` | all | Comma-separated degradation types |
| `--levels` | | 5pct,10pct,25pct,50pct | Comma-separated intensity levels |
| `--no-visualize` | | false | Skip HTML visualization generation |
| `--seed` | | 42 | Random seed for reproducibility |

### Examples

```bash
# Generate all degradations
metricate degrade clustering.csv ./output/

# Specific types and levels
metricate degrade clustering.csv ./output/ \
  --types label_swap_random,noise_injection \
  --levels 10pct,25pct

# Skip visualizations (faster)
metricate degrade clustering.csv ./output/ --no-visualize

# With specific seed
metricate degrade clustering.csv ./output/ --seed 123
```

---

## `metricate list`

List available metrics or degradation types.

### Usage

```bash
metricate list [metrics|degradations]
```

### Examples

```bash
# List all 36 metrics
metricate list metrics

# List all 19 degradation types
metricate list degradations
```

---

## `metricate labricate`

Hyperparameter experimentation commands.

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `experiment` | Run single-param or grid search experiments |
| `validate` | Validate configuration and embeddings |
| `resume` | Resume an interrupted experiment |

---

### `metricate labricate experiment`

Run hyperparameter experiments.

#### Usage

```bash
metricate labricate experiment [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--embeddings` | `-e` | required | Path to embeddings (CSV or NPY) |
| `--config` | `-c` | required | Path to config JSON |
| `--param` | `-p` | none | Parameter path for single-param experiment |
| `--values` | `-v` | none | Comma-separated values (e.g., "5,10,15") |
| `--grid` | `-g` | none | Grid search params (e.g., "param1=1,2;param2=a,b") |
| `--output-dir` | `-o` | ./experiments | Output directory |
| `--output-format` | `-f` | json | json, csv, or both |
| `--workers` | `-w` | 1 | Number of parallel workers |
| `--include-metrics` | | none | Comma-separated metrics to include |
| `--exclude-metrics` | | none | Comma-separated metrics to exclude |
| `--verbose` | | true | Print progress |

#### Examples

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

# Grid search (multiple parameters)
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --grid "hdbscan.min_cluster_size=5,10,15;umap.n_neighbors=10,15,20" \
    --workers 4

# Filter metrics for speed
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values "10,20,30" \
    --exclude-metrics "Gamma,Tau,G-plus"

# Output both JSON and CSV
metricate labricate experiment \
    --embeddings embeddings.csv \
    --config config.json \
    --param "hdbscan.min_cluster_size" \
    --values "10,20,30" \
    --output-format both \
    --output-dir ./my_results
```

---

### `metricate labricate validate`

Validate configuration and embeddings before running experiments.

#### Usage

```bash
metricate labricate validate [OPTIONS]
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to config JSON to validate |
| `--embeddings` | `-e` | Path to embeddings file to validate |

#### Examples

```bash
# Validate config only
metricate labricate validate --config config.json

# Validate embeddings only
metricate labricate validate --embeddings embeddings.csv

# Validate both
metricate labricate validate --config config.json --embeddings embeddings.csv
```

---

### `metricate labricate resume`

Resume an interrupted experiment from checkpoint.

#### Usage

```bash
metricate labricate resume EXPERIMENT_DIR [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `EXPERIMENT_DIR` | Path to experiment directory with checkpoint |

#### Options

| Option | Description |
|--------|-------------|
| `--force` | Force restart even if config mismatches |

#### Examples

```bash
# Resume from checkpoint
metricate labricate resume ./experiments/my_experiment_20260318_143022/

# Force restart (ignore config mismatch)
metricate labricate resume ./experiments/my_experiment_20260318_143022/ --force
```

---

## `metricate web`

Start the browser-based UI.

### Usage

```bash
metricate web [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--port` | `-p` | 5000 | Port to run the server on |
| `--host` | `-h` | 127.0.0.1 | Host to bind to |
| `--debug` | | false | Run in debug mode |

### Examples

```bash
# Start on default port
metricate web

# Custom port
metricate web --port 8080

# Allow external connections
metricate web --host 0.0.0.0

# Debug mode
metricate web --debug
```

---

## Global Options

Available for all commands:

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help for a command |

```bash
# Show version
metricate --version

# Show help
metricate --help
metricate evaluate --help
metricate compare --help
```

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error (file not found, invalid data, etc.) |

---

## Piping and Scripting

### JSON Output for Scripts

```bash
# Parse with jq
metricate evaluate clustering.csv --format json | jq '.metrics.Silhouette'

# Save and process
metricate compare v1.csv v2.csv --format json > result.json
```

### CSV for Spreadsheets

```bash
# Open in Excel/Google Sheets
metricate evaluate clustering.csv --format csv -o results.csv

# Concatenate results
for f in *.csv; do
  metricate evaluate "$f" --format csv >> all_results.csv
done
```

### Batch Processing

```bash
#!/bin/bash
# Evaluate all clusterings in a directory

for csv in ./clusterings/*.csv; do
  echo "Processing: $csv"
  metricate evaluate "$csv" --format json -o "${csv%.csv}_metrics.json"
done
```

### Compare with Automated Selection

```bash
#!/bin/bash
# Compare and get winner

result=$(metricate compare v1.csv v2.csv --format json)
winner=$(echo "$result" | jq -r '.winner')

echo "Winner: $winner"

if [ "$winner" = "A" ]; then
  cp v1.csv best.csv
else
  cp v2.csv best.csv
fi
```
