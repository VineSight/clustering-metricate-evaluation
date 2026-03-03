# Metricate

A comprehensive clustering evaluation toolkit that calculates 34 quality metrics, compares clusterings, and generates degraded datasets for testing metric robustness.

## Installation

```bash
pip install metricate
```

Or install from source:

```bash
git clone https://github.com/VineSight/clustering-metricate-evaluation.git
cd clustering-metricate-evaluation
pip install -e .
```

### Requirements

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- plotly ≥ 5.0
- click ≥ 8.0

## Quick Start

### Evaluate a Single Clustering

```python
import metricate

# Evaluate and get all 36 metrics
result = metricate.evaluate("clustering.csv")

# Print formatted table
print(result.to_table())

# Output:
# Clustering Evaluation: clustering.csv
# =====================================
# Points: 10,000 | Clusters: 17 | Dimensions: 50
#
# ┌─────────────────────┬──────────┬────────────┬───────────┐
# │ Metric              │ Value    │ Range      │ Direction │
# ├─────────────────────┼──────────┼────────────┼───────────┤
# │ Silhouette          │ 0.4521   │ [-1, 1]    │ ↑ Higher  │
# │ Davies-Bouldin      │ 1.2341   │ [0, ∞)     │ ↓ Lower   │
# │ Calinski-Harabasz   │ 1523.45  │ [0, ∞)     │ ↑ Higher  │
# │ ...                 │ ...      │ ...        │ ...       │
# └─────────────────────┴──────────┴────────────┴───────────┘

# Get as DataFrame for further analysis
df = result.to_dataframe()
print(df[["metric", "value", "direction"]])

# Export to JSON or CSV
result.to_json()  # Returns JSON string
result.to_csv()   # Returns CSV string
```

### Compare Two Clusterings

```python
import metricate

# Compare two clusterings
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

# Output:
# ┌─────────────────┬──────────┬──────────┬────────┐
# │ Metric          │ Baseline │ Improved │ Winner │
# ├─────────────────┼──────────┼──────────┼────────┤
# │ Silhouette      │ 0.4521   │ 0.5234   │ Improved │
# │ Davies-Bouldin  │ 1.2341   │ 0.9876   │ Improved │
# │ ...             │ ...      │ ...      │ ...    │
# └─────────────────┴──────────┴──────────┴────────┘
```

### Generate Degraded Datasets

```python
import metricate

# Generate all 19 degradation types at 4 intensity levels (76 total)
result = metricate.degrade(
    "clustering.csv",
    output_dir="./degraded/",
    visualize=True  # Generate HTML visualizations
)

print(result.summary())
# Output:
# Degradation Generation Complete
# ========================================
# Output directory: ./degraded/
# Total degradations: 76
# Manifest: ./degraded/manifest.json
# Index HTML: ./degraded/index.html

# Generate specific degradations only
result = metricate.degrade(
    "clustering.csv",
    output_dir="./output/",
    types=["label_swap_random", "noise_injection"],
    levels=["10pct", "25pct"],
    visualize=False  # Skip visualizations for speed
)
```

### Visualize Degradations

When generating degraded datasets, you can create interactive HTML visualizations showing the effect of each degradation:

```python
import metricate

# Generate degradations with visualizations
result = metricate.degrade(
    "clustering.csv",
    output_dir="./output/",
    visualize=True  # Enable visualization generation
)

# Check generated visualization paths
print(f"Index page: {result.index_html_path}")
print(f"Visualizations: {result.visualizations}")
```

**Generated files:**
- `index.html` - Dashboard linking to all visualizations
- `<degradation_type>.html` - Interactive Plotly scatter plot for each type

**Viewing visualizations:**
```bash
# Option 1: Open directly in browser
open ./output/index.html

# Option 2: Serve with Python HTTP server
cd ./output && python -m http.server 8000
# Then visit http://localhost:8000
```

Each visualization shows a 2D projection (via PCA if needed) of the embeddings colored by cluster assignment, making it easy to see how each degradation affects the clustering structure.

### Train Metric Weights for Quality Scoring

Metricate can learn optimal weights for combining metrics into a single compound quality score. Train on your own labeled data to create a custom scoring formula.

```python
import metricate

# Train weights from labeled training data
result = metricate.train_weights(
    "training_data.csv",  # CSV with *_norm columns and quality_score
    regularization="ridge",  # or "lasso" for feature selection
    auto_alpha=True,  # Auto-tune regularization strength
)

# Check training results
print(f"CV R²: {result.cv_scores['r2_mean']:.3f}")
print(f"Non-zero metrics: {result.weights.non_zero_count}")
print(f"Sanity check: {'PASS' if result.sanity_check_passed else 'FAIL'}")

# Top 5 most important metrics
for metric, weight in result.feature_importance[:5]:
    print(f"  {metric}: {weight:+.4f}")

# Save weights for production use
result.save_weights("weights.json")
```

### Evaluate with Learned Weights

Once trained, use weights to compute a compound score for any clustering:

```python
import metricate

# Load trained weights
weights = metricate.load_weights("weights.json")

# Evaluate with compound score
result = metricate.evaluate("my_clustering.csv", weights=weights)

print(f"Compound Score: {result.compound_score:.3f}")
print(result.to_table())  # Shows all metrics + compound score
```

### Quick Score Without Weights (Not Recommended)

If you need a single number but don't have trained weights, you can use `final_score=True` to get an unweighted average of all normalized metrics:

```python
import metricate

result = metricate.evaluate("clustering.csv", final_score=True)

print(f"Final Score: {result.final_score:.3f}")
# Final Score: 0.476
```

> ⚠️ **WARNING: This approach is NOT recommended for production use!**
> 
> The unweighted final score has significant limitations:
> - **Assumes all metrics are equally important** (they are not)
> - **Correlated metrics cause implicit overweighting**
> - **Unbounded metrics use heuristic normalization**
> - **No validation against actual clustering quality**
> 
> For reliable quality scoring, train proper weights using `metricate.train_weights()` with labeled data that represents your use case.

### Compare with Weighted Scoring

Use weights to determine the winner by compound score instead of metric voting:

```python
import metricate

weights = metricate.load_weights("weights.json")

# Compare with weighted winner determination
comparison = metricate.compare(
    "clustering_v1.csv",
    "clustering_v2.csv",
    name_a="V1",
    name_b="V2",
    weights=weights
)

print(f"Metric voting winner: {comparison.winner}")
print(f"Weighted winner: {comparison.weighted_winner}")
print(f"Compound scores: V1={comparison.compound_scores['V1']:.3f}, V2={comparison.compound_scores['V2']:.3f}")
```

### Visualize Feature Importance

Generate a bar chart showing which metrics matter most:

```python
from metricate.training import train_weights, plot_feature_importance

result = train_weights("training_data.csv")

# Create interactive Plotly visualization
fig = plot_feature_importance(
    result,
    top_n=15,
    exclude_zeroed=True,  # Hide zero-weight metrics (for Lasso)
    save_path="importance.html"
)
```

### List Available Options

```python
import metricate

# List all 36 metrics
metrics = metricate.list_metrics()
print(f"{len(metrics)} metrics available")

# Get full reference with directions and ranges
df = metricate.list_metrics(include_reference=True)
print(df[["metric", "direction", "complexity"]])

# List all 19 degradation types by category
degradations = metricate.list_degradations()
for category, types in degradations.items():
    print(f"{category}: {types}")
```

## Command Line Interface

```bash
# Evaluate a clustering (table output)
metricate evaluate clustering.csv

# JSON output for scripting
metricate evaluate clustering.csv --format json

# Save to file
metricate evaluate clustering.csv --format csv -o results.csv

# Exclude expensive metrics
metricate evaluate clustering.csv --exclude Gamma,Tau,G-plus

# Evaluate with learned weights (adds compound score)
metricate evaluate clustering.csv --weights weights.json

# Compare two clusterings
metricate compare baseline.csv improved.csv
metricate compare v1.csv v2.csv --name-a "Version 1" --name-b "Version 2"

# Compare with weighted winner determination
metricate compare v1.csv v2.csv --weights weights.json

# Train metric weights from labeled data
metricate train training_data.csv -o weights.json

# Train with auto-tuned regularization
metricate train training_data.csv -o weights.json --auto-alpha

# Train with Lasso for feature selection
metricate train training_data.csv -o weights.json -r lasso --auto-alpha

# Full training options
metricate train training_data.csv \
  -o weights.json \
  --regularization ridge \
  --alpha 1.0 \
  --cv-splits 5 \
  --top-n 15

# Generate degradations
metricate degrade clustering.csv ./output/

# Specific types and levels
metricate degrade clustering.csv ./output/ \
  --types label_swap_random,noise_injection \
  --levels 10pct,25pct

# Skip visualizations (faster)
metricate degrade clustering.csv ./output/ --no-visualize

# List available options
metricate list metrics
metricate list degradations

# Start the web UI
metricate web --port 5000
```

## Web UI

Metricate includes a browser-based interface for evaluating and comparing clusterings without writing code.

### Starting the Web Server

```bash
# If installed globally or venv is activated:
metricate web

# If using a virtual environment (not activated):
.venv/bin/metricate web

# Use a custom port
metricate web --port 8080

# Run in debug mode
metricate web --debug
```

Then open `http://localhost:5000` in your browser (or the port you specified).

### Features

- **Upload CSV files** directly through the browser
- **Evaluate** a single clustering and view all 36 metrics
- **Compare** two clusterings side-by-side with winner determination
- **Formatted results** displayed in easy-to-read tables
- **Export** results as JSON or CSV

### Python API

You can also start the web server programmatically:

```python
import metricate

# Start the web server
metricate.web(port=5000, debug=False)
```

## Input Format

Your CSV must have:
- **Label column**: Cluster assignments (integer, -1 for noise points)
- **Embedding columns**: Numeric vectors representing each data point

### Label Column Detection

The label column is auto-detected by matching these patterns (case-insensitive):
- `cluster_id`, `cluster`, `new_cluster`
- `label`, `labels`, `class`
- `group`, `group_id`, `assignment`

Or specify explicitly: `metricate.evaluate("data.csv", label_col="my_cluster_col")`

### Embedding Formats

Metricate supports **two embedding formats**:

#### Format 1: Separate Numeric Columns (Recommended)

Each dimension is its own column. Auto-detected by patterns like `dim_*`, `embedding_*`, `x_*`, `pc_*`, `umap_*`, or any numeric columns.

```csv
cluster_id,dim_0,dim_1,dim_2,dim_3,dim_4
0,0.123,0.456,0.789,0.012,0.345
0,0.234,0.567,0.890,0.123,0.456
1,0.345,0.678,0.901,0.234,0.567
```

#### Format 2: String-Encoded Arrays

A single column containing the full vector as a string (common when exporting from Python/numpy). Metricate auto-expands these into separate dimensions.

```csv
cluster_id,embedding
0,"[0.123, 0.456, 0.789, 0.012, 0.345]"
0,"[0.234, 0.567, 0.890, 0.123, 0.456]"
1,"[0.345, 0.678, 0.901, 0.234, 0.567]"
```

Both formats work identically—string arrays are parsed and expanded automatically.

#### Specifying Embedding Columns

```python
# Auto-detect (default)
result = metricate.evaluate("data.csv")

# Explicit separate columns
result = metricate.evaluate("data.csv", embedding_cols=["dim_0", "dim_1", "dim_2"])

# Explicit string-array column (auto-expanded)
result = metricate.evaluate("data.csv", embedding_cols=["embedding"])

# Use reduced embeddings instead of full
result = metricate.evaluate("data.csv", embedding_cols=["reduced_embedding"])
```

### Choosing Embedding Dimensions

Your dataset may have multiple embedding options:

| Type | Typical Dims | Use Case |
|------|--------------|----------|
| **Full embeddings** | 768–1536 | Original model output (e.g., OpenAI ada-002, BERT) |
| **Reduced embeddings** | 10–50 | PCA/UMAP dimensionality reduction, faster computation |
| **Visualization coords** | 2–3 | UMAP/t-SNE for plotting only |

#### Recommendations

1. **For accurate metrics**: Use **reduced embeddings (10–50D)**
   - Avoids curse of dimensionality (distances become meaningless in very high-D)
   - Much faster computation (O(n·d) operations)
   - Preserves cluster structure if reduction was done properly

2. **Avoid full 1536D embeddings** for metrics unless necessary:
   - Silhouette, Dunn Index, and distance-based metrics degrade in high dimensions
   - 10–100x slower than reduced embeddings
   - Memory-intensive for pairwise distance matrices

3. **Never use 2D visualization coordinates** for metrics:
   - Too much information loss
   - Only suitable for visual inspection, not quantitative evaluation

#### Example: Multiple Embedding Columns

```csv
cluster_id,embedding,reduced_embedding,umap_1,umap_2
0,"[0.05, -0.01, ..., 0.03]","[0.12, 0.45, ..., 0.78]",1.23,4.56
```

```python
# Use reduced embeddings (recommended)
result = metricate.evaluate("data.csv", embedding_cols=["reduced_embedding"])

# Use full embeddings (slower, may be less meaningful)
result = metricate.evaluate("data.csv", embedding_cols=["embedding"])

# Use pre-expanded dimensions
result = metricate.evaluate("data.csv", embedding_cols=["dim_0", "dim_1", ..., "dim_9"])
```

## Training Data Format (for Metric Weights)

To train custom metric weights, you need a CSV with:
- **Normalized metric columns**: Named `<Metric>_norm` (e.g., `Silhouette_norm`)
- **Quality score column**: `quality_score` (0.0 = worst, 1.0 = best)
- **Clustering name column**: `clustering_name` (for cross-validation grouping)

### Example Training CSV

```csv
clustering_name,Silhouette_norm,Davies-Bouldin_norm,Calinski-Harabasz_norm,...,quality_score
original,0.95,0.88,0.92,...,1.0
label_swap_10pct,0.85,0.75,0.80,...,0.8
label_swap_25pct,0.70,0.60,0.65,...,0.6
noise_injection_10pct,0.90,0.82,0.88,...,0.9
```

### Generating Training Data

Use `metricate.generate_training_data()` to create labeled datasets from your clusterings:

```python
import metricate

# Generate training data from a good clustering
result = metricate.generate_training_data(
    "good_clustering.csv",
    output_dir="./training/",
    degradation_types=["label_swap_random", "noise_injection"],
    levels=["10pct", "25pct", "50pct"]
)

# Combine into single training CSV
df = result.to_dataframe()
df.to_csv("training_data.csv", index=False)
```

## 36 Clustering Metrics

### Internal Original (6)
- Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn Index, SSE, NCI

### Tier 1 CVIs (6)
- Ball-Hall, Ratkowsky-Lance, Ray-Turi, RMSSTD, R-squared, Wemmert-Gancarski

### Tier 2 CVIs (14)
- CS index, COP, S_Dbw, Det Ratio, Gamma, Generalized Dunn, G-plus, I-index (PBM), Log_Det_Ratio, McClain-Rao, Point-Biserial, SD validity, Tau, Trace_WiB, Ksq_DetW

### Tier 3 CVIs (4)
- Banfield-Raftery, Negentropy, NIVA, Score Function, Scott-Symons

### External (4)
- Adjusted Rand Index, Van Dongen, Variation of Information, Omega

## Large Dataset Handling (>50,000 rows)

For datasets exceeding 50,000 rows, the following O(n²) metrics are automatically skipped to ensure reasonable computation time:

- **Gamma**
- **Tau**
- **Point-Biserial**
- **G-plus**
- **McClain-Rao**
- **NIVA**

### Override Auto-Skip

To compute all metrics regardless of dataset size:

```python
# Python API
result = metricate.evaluate("large_data.csv", force_all=True)
```

```bash
# CLI
metricate evaluate large_data.csv --force-all
```

**Warning**: Computing O(n²) metrics on large datasets may take several minutes and require significant memory.

## Excluding Metrics
Reasons to exclude metrics:
- Your clustering picked the number of K clusters by optimising for Tau? then you shouldn't look at Tau as a metric
- Shorten runtime for the comparison\evaluation
Skip specific metrics to save computation time:

```python
result = metricate.evaluate(
    "clustering.csv",
    exclude=["Gamma", "Tau", "S_Dbw"]
)
```

```bash
metricate evaluate clustering.csv --exclude Gamma,Tau,S_Dbw
```

## 19 Degradation Types

| Category | Types |
|----------|-------|
| Point Removal | random_removal, core_removal |
| Label Modification | boundary_reassignment, label_swap_random, label_swap_neighboring, label_swap_distant |
| Embedding Noise | noise_injection, embedding_perturbation, centroid_displacement |
| Cluster Merge | merge_nearest, merge_farthest, merge_random |
| Cluster Split | split_largest, split_loosest, split_random |
| Cluster Remove | remove_smallest_clusters, remove_largest_clusters, remove_tightest_clusters |

## License

MIT
