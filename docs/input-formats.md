# Input Formats

This guide explains how to format your data for Metricate.

---

## Basic Requirements

Your CSV must have:

1. **Label column**: Cluster assignments (integers)
2. **Embedding columns**: Numeric vectors representing each data point

---

## Label Column

### Auto-Detection

Metricate auto-detects the label column by matching these patterns (case-insensitive):

- `cluster_id`, `cluster`, `new_cluster`
- `label`, `labels`, `class`
- `group`, `group_id`, `assignment`

### Manual Specification

```python
result = metricate.evaluate("data.csv", label_col="my_cluster_column")
```

```bash
metricate evaluate data.csv --label-col my_cluster_column
```

### Valid Label Values

| Value | Meaning |
|-------|---------|
| `0, 1, 2, ...` | Cluster assignments |
| `-1` | Noise point (excluded from some metrics) |

### Example

```csv
cluster_id,dim_0,dim_1,dim_2
0,0.123,0.456,0.789
0,0.234,0.567,0.890
1,0.345,0.678,0.901
-1,0.999,0.111,0.222
```

The last row is a noise point (cluster = -1).

---

## Embedding Formats

Metricate supports **two embedding formats**.

### Format 1: Separate Numeric Columns (Recommended)

Each dimension is its own column.

**Auto-detected patterns:**
- `dim_*`, `dim0`, `dim1`, ...
- `embedding_*`, `embedding_0`, ...
- `x_*`, `x0`, `x1`, ...
- `pc_*`, `pc1`, `pc2`, ...
- `umap_*`, `umap_1`, `umap_2`, ...
- Any numeric columns

```csv
cluster_id,dim_0,dim_1,dim_2,dim_3,dim_4
0,0.123,0.456,0.789,0.012,0.345
0,0.234,0.567,0.890,0.123,0.456
1,0.345,0.678,0.901,0.234,0.567
```

### Format 2: String-Encoded Arrays

A single column containing the full vector as a string. Metricate auto-expands these.

```csv
cluster_id,embedding
0,"[0.123, 0.456, 0.789, 0.012, 0.345]"
0,"[0.234, 0.567, 0.890, 0.123, 0.456]"
1,"[0.345, 0.678, 0.901, 0.234, 0.567]"
```

**Supported formats:**
- Python list: `[0.1, 0.2, 0.3]`
- Numpy-style: `[0.1 0.2 0.3]`
- Comma-separated: `0.1, 0.2, 0.3`

Both formats work identically—string arrays are parsed and expanded automatically.

---

## Specifying Embedding Columns

### Auto-Detection (Default)

```python
result = metricate.evaluate("data.csv")
```

Metricate will find embedding columns automatically.

### Explicit Columns

```python
# Separate columns
result = metricate.evaluate(
    "data.csv",
    embedding_cols=["dim_0", "dim_1", "dim_2", "dim_3", "dim_4"]
)

# String-array column (auto-expanded)
result = metricate.evaluate(
    "data.csv",
    embedding_cols=["embedding"]
)

# Use reduced embeddings
result = metricate.evaluate(
    "data.csv",
    embedding_cols=["reduced_embedding"]
)
```

### CLI

```bash
metricate evaluate data.csv --embedding-cols "dim_0,dim_1,dim_2"
```

---

## Choosing Embedding Dimensions

Your dataset may have multiple embedding options:

| Type | Typical Dims | Use Case |
|------|--------------|----------|
| **Full embeddings** | 768–1536 | Original model output (OpenAI, BERT) |
| **Reduced embeddings** | 10–50 | PCA/UMAP dimensionality reduction |
| **Visualization coords** | 2–3 | UMAP/t-SNE for plotting only |

### Recommendations

#### ✅ Use Reduced Embeddings (10–50D)

Best for accurate metrics:
- Avoids curse of dimensionality
- Much faster computation (O(n·d))
- Preserves cluster structure

```python
result = metricate.evaluate(
    "data.csv",
    embedding_cols=["reduced_embedding"]
)
```

#### ⚠️ Avoid Full 1536D Embeddings

Unless necessary:
- Silhouette, Dunn Index degrade in high dimensions
- 10–100x slower than reduced embeddings
- Memory-intensive for pairwise distances

#### ❌ Never Use 2D Visualization Coordinates

For metrics:
- Too much information loss
- Only suitable for visual inspection

### Example: Multiple Embedding Columns

```csv
cluster_id,embedding,reduced_embedding,umap_1,umap_2
0,"[0.05, -0.01, ..., 0.03]","[0.12, 0.45, ..., 0.78]",1.23,4.56
```

```python
# Use reduced embeddings (recommended)
result = metricate.evaluate("data.csv", embedding_cols=["reduced_embedding"])

# Use full embeddings (slower, may be less meaningful)
result = metricate.evaluate("data.csv", embedding_cols=["embedding"])

# Use UMAP coordinates (NOT recommended for metrics)
result = metricate.evaluate("data.csv", embedding_cols=["umap_1", "umap_2"])
```

---

## Complete Examples

### Minimal CSV

```csv
cluster,x,y
0,1.0,2.0
0,1.1,2.1
1,5.0,6.0
1,5.1,6.1
```

### Full-Featured CSV

```csv
id,text,cluster_id,embedding,reduced_embedding,umap_x,umap_y,confidence
1,"Hello world",0,"[0.1, 0.2, ..., 0.5]","[0.3, 0.4, 0.5]",1.2,3.4,0.95
2,"Goodbye world",0,"[0.11, 0.21, ..., 0.51]","[0.31, 0.41, 0.51]",1.3,3.5,0.92
3,"New topic",1,"[0.8, 0.7, ..., 0.2]","[0.7, 0.6, 0.5]",8.1,2.3,0.88
```

Usage:
```python
result = metricate.evaluate(
    "data.csv",
    label_col="cluster_id",
    embedding_cols=["reduced_embedding"]
)
```

### Separate Dimension Columns

```csv
cluster_id,dim_0,dim_1,dim_2,dim_3,dim_4,dim_5,dim_6,dim_7,dim_8,dim_9
0,0.123,0.456,0.789,0.012,0.345,0.678,0.901,0.234,0.567,0.890
0,0.234,0.567,0.890,0.123,0.456,0.789,0.012,0.345,0.678,0.901
1,0.345,0.678,0.901,0.234,0.567,0.890,0.123,0.456,0.789,0.012
```

---

## Training Data Format

For `metricate.train_weights()`, the CSV needs additional columns:

| Column | Description |
|--------|-------------|
| `clustering_name` | Identifier for grouping (for CV) |
| `*_norm` columns | Normalized metric values (0-1 scale) |
| `quality_score` | Target label (0.0 = worst, 1.0 = best) |

### Example

```csv
clustering_name,Silhouette_norm,Davies-Bouldin_norm,Calinski-Harabasz_norm,quality_score
original,0.95,0.88,0.92,1.0
label_swap_10pct,0.85,0.75,0.80,0.8
label_swap_25pct,0.70,0.60,0.65,0.6
noise_injection_10pct,0.90,0.82,0.88,0.9
```

### Generating Training Data

```python
import metricate

result = metricate.generate_training_data(
    "good_clustering.csv",
    output_dir="./training/",
)

# Automatically creates normalized metrics and quality scores
df = result.to_dataframe()
df.to_csv("training_data.csv", index=False)
```

---

## Common Issues

### "Column not found"

Check column names match exactly (case-sensitive):

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns.tolist())
```

### "No embedding columns detected"

Specify columns explicitly:

```python
result = metricate.evaluate(
    "data.csv",
    embedding_cols=["col1", "col2", "col3"]
)
```

### "Insufficient clusters"

You need at least 2 clusters:

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df["cluster_id"].nunique())  # Should be >= 2
```

### String arrays not parsing

Ensure proper format:

```python
# Good
"[0.1, 0.2, 0.3]"

# Bad (missing quotes)
[0.1, 0.2, 0.3]

# Bad (wrong brackets)
"(0.1, 0.2, 0.3)"
```

---

## File Size Considerations

| Rows | Recommendation |
|------|----------------|
| < 10,000 | All metrics run quickly |
| 10,000 - 50,000 | May take a few minutes |
| > 50,000 | O(n²) metrics auto-skipped |
| > 100,000 | Consider sampling or using `--force-all` carefully |

### Large Dataset Tips

```python
# Let Metricate auto-skip expensive metrics
result = metricate.evaluate("large_data.csv")

# Or force all (slow)
result = metricate.evaluate("large_data.csv", force_all=True)

# Or exclude specific metrics
result = metricate.evaluate(
    "large_data.csv",
    exclude=["Gamma", "Tau", "Point-Biserial", "G-plus", "McClain-Rao", "NIVA"]
)
```
