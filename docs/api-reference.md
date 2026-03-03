# API Reference

Complete reference for the Metricate Python API.

---

## Core Functions

### `metricate.evaluate()`

Evaluate a single clustering and return all metric scores.

```python
metricate.evaluate(
    csv_path: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    weights: MetricWeights | None = None,
    final_score: bool = False,
) -> EvaluationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | required | Path to CSV file containing clustering data |
| `exclude` | `list[str]` | `None` | Metric names to skip (e.g., `["Gamma", "Tau"]`) |
| `force_all` | `bool` | `False` | Compute O(n²) metrics even on large datasets |
| `label_col` | `str` | `None` | Name of cluster label column (auto-detected if None) |
| `embedding_cols` | `list[str]` | `None` | Embedding column names (auto-detected if None) |
| `weights` | `MetricWeights` | `None` | Learned weights for computing compound score |
| `final_score` | `bool` | `False` | Return unweighted average as final_score (not recommended) |

**Returns:** `EvaluationResult`

**Example:**

```python
import metricate

# Basic evaluation
result = metricate.evaluate("clustering.csv")
print(result.to_table())

# With weights for compound score
weights = metricate.load_weights("weights.json")
result = metricate.evaluate("clustering.csv", weights=weights)
print(f"Compound score: {result.compound_score:.3f}")
```

---

### `metricate.compare()`

Compare two clusterings and determine the winner.

```python
metricate.compare(
    csv_path_a: str | Path,
    csv_path_b: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    name_a: str = "A",
    name_b: str = "B",
    weights: MetricWeights | None = None,
) -> ComparisonResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path_a` | `str \| Path` | required | Path to first CSV file |
| `csv_path_b` | `str \| Path` | required | Path to second CSV file |
| `exclude` | `list[str]` | `None` | Metric names to skip |
| `force_all` | `bool` | `False` | Compute O(n²) metrics on large datasets |
| `label_col` | `str` | `None` | Cluster label column name |
| `embedding_cols` | `list[str]` | `None` | Embedding column names |
| `name_a` | `str` | `"A"` | Display name for first clustering |
| `name_b` | `str` | `"B"` | Display name for second clustering |
| `weights` | `MetricWeights` | `None` | Weights for weighted winner determination |

**Returns:** `ComparisonResult`

**Example:**

```python
import metricate

result = metricate.compare(
    "baseline.csv",
    "improved.csv",
    name_a="Baseline",
    name_b="Improved"
)

print(f"Winner: {result.winner}")
print(f"Wins: {result.wins}")
print(result.to_table())
```

---

### `metricate.degrade()`

Generate degraded versions of a clustering dataset.

```python
metricate.degrade(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    levels: list[str] | None = None,
    types: list[str] | None = None,
    visualize: bool = True,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    random_seed: int = 42,
) -> DegradationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | required | Path to input clustering CSV |
| `output_dir` | `str \| Path` | required | Directory to write degraded datasets |
| `levels` | `list[str]` | `["5pct", "10pct", "25pct", "50pct"]` | Intensity levels |
| `types` | `list[str]` | All 19 types | Degradation types to apply |
| `visualize` | `bool` | `True` | Generate HTML visualizations |
| `label_col` | `str` | `None` | Cluster label column name |
| `embedding_cols` | `list[str]` | `None` | Embedding column names |
| `random_seed` | `int` | `42` | Random seed for reproducibility |

**Returns:** `DegradationResult`

**Example:**

```python
import metricate

result = metricate.degrade(
    "clustering.csv",
    output_dir="./degraded/",
    types=["label_swap_random", "noise_injection"],
    levels=["10pct", "25pct"],
    visualize=True
)

print(result.summary())
```

---

### `metricate.train_weights()`

Train a regression model to learn optimal metric weights.

```python
metricate.train_weights(
    csv_path: str | Path,
    *,
    regularization: str = "ridge",
    alpha: float = 1.0,
    auto_alpha: bool = False,
    alphas: list[float] | None = None,
    run_cv: bool = True,
    cv_splits: int = 5,
    run_sanity_check: bool = True,
) -> TrainingResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | required | Path to training CSV with normalized metrics |
| `regularization` | `str` | `"ridge"` | Type: `"ridge"` or `"lasso"` |
| `alpha` | `float` | `1.0` | Regularization strength |
| `auto_alpha` | `bool` | `False` | Auto-tune alpha via cross-validation |
| `alphas` | `list[float]` | `None` | Candidate alpha values for auto-tuning |
| `run_cv` | `bool` | `True` | Run cross-validation |
| `cv_splits` | `int` | `5` | Number of CV folds |
| `run_sanity_check` | `bool` | `True` | Verify original > degraded scores |

**Returns:** `TrainingResult`

**Example:**

```python
import metricate

result = metricate.train_weights(
    "training_data.csv",
    regularization="ridge",
    auto_alpha=True
)

print(f"CV R²: {result.cv_scores['r2_mean']:.3f}")
print(f"Sanity check: {'PASS' if result.sanity_check_passed else 'FAIL'}")

# Save weights
result.save_weights("weights.json")
```

---

### `metricate.load_weights()`

Load learned metric weights from a JSON file.

```python
metricate.load_weights(path: str | Path) -> MetricWeights
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to JSON file containing weights |

**Returns:** `MetricWeights`

**Example:**

```python
import metricate

weights = metricate.load_weights("weights.json")
result = metricate.evaluate("clustering.csv", weights=weights)
print(f"Compound score: {result.compound_score:.3f}")
```

---

## Utility Functions

### `metricate.list_metrics()`

List all available clustering metrics.

```python
metricate.list_metrics(include_reference: bool = False) -> list[str] | pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_reference` | `bool` | `False` | Return DataFrame with full metadata |

**Returns:** List of metric names, or DataFrame with columns: `metric`, `range`, `direction`, `tier`, `complexity`

**Example:**

```python
import metricate

# Just names
metrics = metricate.list_metrics()
print(f"{len(metrics)} metrics available")

# Full reference
df = metricate.list_metrics(include_reference=True)
print(df[["metric", "direction", "complexity"]])
```

---

### `metricate.list_degradations()`

List all available degradation types organized by category.

```python
metricate.list_degradations() -> dict[str, list[str]]
```

**Returns:** Dict mapping category names to lists of degradation types

**Example:**

```python
import metricate

types = metricate.list_degradations()
for category, degradations in types.items():
    print(f"{category}: {degradations}")
```

---

## Training Data Generation

### `metricate.generate_training_data()`

Generate a training dataset from a single clustering CSV.

```python
metricate.generate_training_data(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    topic: str | None = None,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> TrainingDataResult
```

**Example:**

```python
import metricate

result = metricate.generate_training_data(
    "clustering.csv",
    "./output/",
    types=["label_swap_random", "noise_injection"],
    levels=["10pct", "25pct", "50pct"]
)

df = result.to_dataframe()
result.to_csv("training_dataset.csv")
```

---

### `metricate.generate_training_data_batch()`

Generate training data from all CSVs in a directory.

```python
metricate.generate_training_data_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    topic_mapping: dict[str, str] | None = None,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> TrainingDataResult
```

**Example:**

```python
import metricate

result = metricate.generate_training_data_batch(
    "./clusterings/",
    "./output/"
)

print(result.summary())
result.to_csv("full_training_dataset.csv")
```

---

## Result Objects

### `EvaluationResult`

Returned by `metricate.evaluate()`.

**Attributes:**
- `metrics: dict[str, float]` - Metric name to value mapping
- `warnings: list[str]` - Any warnings generated
- `compound_score: float` - Weighted score (if weights provided)
- `final_score: float` - Unweighted average (if final_score=True)

**Methods:**
- `to_table(format="simple")` - Formatted string table
- `to_dataframe()` - pandas DataFrame
- `to_json()` - JSON string
- `to_csv()` - CSV string

---

### `ComparisonResult`

Returned by `metricate.compare()`.

**Attributes:**
- `winner: str` - Name of winning clustering
- `wins: dict` - Win counts `{"A": n, "B": m, "Tie": t}`
- `metric_winners: dict` - Metric name to winner mapping
- `weighted_winner: str` - Winner by compound score (if weights provided)
- `compound_scores: dict` - Compound scores per clustering
- `warnings: list[str]` - Any warnings generated

**Methods:**
- `to_table(format="simple")` - Formatted comparison table
- `to_dataframe()` - pandas DataFrame
- `to_dict()` - Dictionary representation

---

### `DegradationResult`

Returned by `metricate.degrade()`.

**Attributes:**
- `output_dir: str` - Path to output directory
- `degradations: list[DegradationEntry]` - List of degradation entries
- `manifest_path: str` - Path to manifest.json
- `index_html_path: str` - Path to index.html (if visualize=True)
- `visualizations: list[str]` - List of visualization file paths
- `csv_files: list[str]` - List of generated CSV paths
- `warnings: list[str]` - Any warnings generated

**Methods:**
- `summary()` - Text summary of generation

---

### `MetricWeights`

Learned coefficients for compound scoring.

**Attributes:**
- `coefficients: dict[str, float]` - Metric name to weight mapping
- `bias: float` - Intercept term
- `version: str` - Schema version
- `regularization: str` - Type of regularization used
- `alpha: float` - Regularization strength
- `cv_r2: float` - Cross-validation R² score
- `non_zero_count: int` - Number of non-zero coefficients

**Methods:**
- `save(path)` - Save to JSON file
- `to_json()` - Convert to JSON string
- `to_dict()` - Convert to dictionary

---

## Exceptions

| Exception | Description |
|-----------|-------------|
| `FileNotFoundError` | CSV path does not exist |
| `InvalidCSVError` | File is not valid CSV |
| `ColumnNotFoundError` | Required columns not found |
| `InsufficientClustersError` | Fewer than 2 clusters in data |
