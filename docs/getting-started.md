# Getting Started

This guide will help you get up and running with Metricate in minutes.

---

## Installation

### From PyPI (Recommended)

```bash
pip install metricate
```

### From Source

```bash
git clone https://github.com/VineSight/clustering-metricate-evaluation.git
cd clustering-metricate-evaluation
pip install -e .
```

### Optional Dependencies

When installing from source, use:

```bash
# Install with web UI support (Flask)
pip install -e ".[web]"

# Install with dev tools (pytest, black, ruff)
pip install -e ".[dev]"

# Install with everything
pip install -e ".[web,dev]"
```

---

## Your First Evaluation

### Prepare Your Data

Metricate expects a CSV file with:
- **Label column**: Cluster assignments (integers, -1 for noise)
- **Embedding columns**: Numeric vectors representing each data point

Example CSV:

```csv
cluster_id,dim_0,dim_1,dim_2,dim_3
0,0.123,0.456,0.789,0.012
0,0.234,0.567,0.890,0.123
1,0.345,0.678,0.901,0.234
1,0.456,0.789,0.012,0.345
```

### Evaluate with Python

```python
import metricate

# Load and evaluate
result = metricate.evaluate("clustering.csv")

# Print formatted table
print(result.to_table())
```

Output:

```
Clustering Evaluation: clustering.csv
=====================================
Points: 10,000 | Clusters: 17 | Dimensions: 50

┌─────────────────────┬──────────┬────────────┬───────────┐
│ Metric              │ Value    │ Range      │ Direction │
├─────────────────────┼──────────┼────────────┼───────────┤
│ Silhouette          │ 0.4521   │ [-1, 1]    │ ↑ Higher  │
│ Davies-Bouldin      │ 1.2341   │ [0, ∞)     │ ↓ Lower   │
│ Calinski-Harabasz   │ 1523.45  │ [0, ∞)     │ ↑ Higher  │
│ ...                 │ ...      │ ...        │ ...       │
└─────────────────────┴──────────┴────────────┴───────────┘
```

### Evaluate with CLI

```bash
# Table output
metricate evaluate clustering.csv

# JSON output
metricate evaluate clustering.csv --format json

# Save to file
metricate evaluate clustering.csv --format csv -o results.csv
```

---

## Comparing Clusterings

### Python API

```python
import metricate

comparison = metricate.compare(
    "baseline.csv",
    "improved.csv",
    name_a="Baseline",
    name_b="Improved"
)

# See the winner
print(f"Overall winner: {comparison.winner}")
print(f"Wins: {comparison.wins}")  # {'Baseline': 12, 'Improved': 18, 'Tie': 4}

# Print comparison table
print(comparison.to_table())
```

### CLI

```bash
metricate compare baseline.csv improved.csv

# With custom names
metricate compare v1.csv v2.csv --name-a "Version 1" --name-b "Version 2"
```

---

## Working with Results

### Export Formats

```python
result = metricate.evaluate("clustering.csv")

# Get as pandas DataFrame
df = result.to_dataframe()

# Export to JSON
json_str = result.to_json()

# Export to CSV
csv_str = result.to_csv()
```

### Filter Metrics

```python
# Exclude expensive O(n²) metrics
result = metricate.evaluate(
    "clustering.csv",
    exclude=["Gamma", "Tau", "G-plus"]
)

# Force computation of all metrics on large datasets
result = metricate.evaluate(
    "large_data.csv",
    force_all=True
)
```

---

## Next Steps

- [API Reference](api-reference.md) - Full Python API documentation
- [Metrics Reference](metrics-reference.md) - Details on all 36 metrics
- [Training Weights](training-weights.md) - Learn custom scoring models
- [CLI Reference](cli-reference.md) - Complete command-line options
