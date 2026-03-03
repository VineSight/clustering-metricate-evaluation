# Training Metric Weights

Metricate can learn optimal weights for combining metrics into a single compound quality score. This is the key to moving from "40 different opinions" to "one reliable score."

---

## Why Train Weights?

### The Problem

You have 36 clustering metrics. Some say A is better, some say B is better. Now what?

The naive approach: count wins. But this assumes:
- All metrics are equally important (they're not)
- Metrics are independent (many are correlated)
- Each metric is equally reliable (they have different sensitivities)

### The Solution

**Learn which metrics matter** from labeled data:

1. Start with known good clusterings
2. Degrade them systematically → known bad clusterings
3. Train a model to distinguish good from bad
4. Use learned weights for future comparisons

---

## Training Pipeline

### Step 1: Generate Training Data

```python
import metricate

# Generate degraded versions with calculated metrics
result = metricate.generate_training_data(
    "good_clustering.csv",
    output_dir="./training/",
    types=["label_swap_random", "noise_injection", "merge_nearest"],
    levels=["10pct", "25pct", "50pct"]
)

# Save as training CSV
result.to_csv("training_dataset.csv")
```

The output includes:
- Original clustering with `quality_score = 1.0`
- Degraded versions with `quality_score` proportional to degradation
- All 36 metrics (normalized as `*_norm` columns)

### Step 2: Train Weights

```python
import metricate

result = metricate.train_weights(
    "training_dataset.csv",
    regularization="ridge",  # or "lasso" for feature selection
    auto_alpha=True,          # Auto-tune regularization strength
)

print(f"CV R²: {result.cv_scores['r2_mean']:.3f}")
print(f"Non-zero metrics: {result.weights.non_zero_count}")
print(f"Sanity check: {'PASS' if result.sanity_check_passed else 'FAIL'}")

# Top 5 most important metrics
print("\nTop Features:")
for metric, weight in result.feature_importance[:5]:
    print(f"  {metric}: {weight:+.4f}")

# Save for production use
result.save_weights("weights.json")
```

### Step 3: Use Trained Weights

```python
import metricate

# Load weights
weights = metricate.load_weights("weights.json")

# Evaluate with compound score
result = metricate.evaluate("my_clustering.csv", weights=weights)
print(f"Compound Score: {result.compound_score:.3f}")

# Compare with weighted winner
comparison = metricate.compare(
    "v1.csv", "v2.csv",
    weights=weights
)
print(f"Weighted winner: {comparison.weighted_winner}")
```

---

## Training Options

### Regularization Types

**Ridge (L2) - Default**
- Keeps all metrics, shrinks weights
- Good when all metrics have some signal
- More stable with correlated features

```python
result = metricate.train_weights(
    "training.csv",
    regularization="ridge"
)
```

**Lasso (L1)**
- Zeros out unimportant metrics
- Good for feature selection
- Produces sparse, interpretable weights

```python
result = metricate.train_weights(
    "training.csv",
    regularization="lasso"
)

# Check which metrics were zeroed
print(f"Zeroed metrics: {result.zeroed_metrics}")
```

### Alpha Tuning

**Manual Alpha**
```python
result = metricate.train_weights(
    "training.csv",
    alpha=0.5  # Lower = less regularization
)
```

**Auto-Tuned Alpha**
```python
result = metricate.train_weights(
    "training.csv",
    auto_alpha=True,  # Cross-validation selection
    alphas=[0.01, 0.1, 1.0, 10.0, 100.0]  # Candidates
)

print(f"Selected alpha: {result.weights.alpha}")
```

### Cross-Validation

Leave-one-clustering-out CV by default:

```python
result = metricate.train_weights(
    "training.csv",
    run_cv=True,
    cv_splits=5
)

print(f"CV R²: {result.cv_scores['r2_mean']:.3f} ± {result.cv_scores['r2_std']:.3f}")
print(f"CV MAE: {result.cv_scores['mae_mean']:.3f}")
print(f"CV RMSE: {result.cv_scores['rmse_mean']:.3f}")
```

### Sanity Check

Verifies that original clusterings score higher than all their degraded versions:

```python
result = metricate.train_weights(
    "training.csv",
    run_sanity_check=True
)

if result.sanity_check_passed:
    print("✓ All originals rank above degraded versions")
else:
    print("✗ Sanity check failed!")
    for failure in result.sanity_failures:
        print(f"  {failure}")
```

---

## Training Data Format

The training CSV must have:

| Column | Description |
|--------|-------------|
| `clustering_name` | Identifier for grouping (for CV) |
| `*_norm` columns | Normalized metric values (0-1 scale) |
| `quality_score` | Target label (0.0 = worst, 1.0 = best) |

### Example:

```csv
clustering_name,Silhouette_norm,Davies-Bouldin_norm,...,quality_score
original,0.95,0.88,...,1.0
label_swap_10pct,0.85,0.75,...,0.8
label_swap_25pct,0.70,0.60,...,0.6
noise_injection_10pct,0.90,0.82,...,0.9
```

### Batch Generation

Process multiple clusterings at once:

```python
import metricate

result = metricate.generate_training_data_batch(
    input_dir="./good_clusterings/",
    output_dir="./training/",
    topic_mapping={
        "customers.csv": "customer_segmentation",
        "products.csv": "product_categories"
    }
)

result.to_csv("full_training_dataset.csv")
```

---

## The Compound Score Formula

The learned formula is:

$$\text{score} = \text{clip}\left(\sum_{i} w_i \cdot m_i + b, 0, 1\right)$$

Where:
- $w_i$ = learned weight for metric $i$
- $m_i$ = normalized metric value
- $b$ = bias (intercept)

### Understanding Weights

**Positive weights**: Higher metric values → higher quality score
- Makes sense for "higher is better" metrics after normalization

**Negative weights**: Higher metric values → lower quality score
- Typically for "lower is better" metrics that weren't inverted during normalization
- Or metrics that hurt more than help

**Zero weights (Lasso)**: Metric doesn't contribute
- Redundant with other metrics
- Not informative for this type of data

---

## Best Practices

### 1. Use Domain-Appropriate Training Data

The weights learned depend on your training data:
- Customer segmentation → train on customer clusterings
- Document clustering → train on document clusterings
- General purpose → train on diverse datasets

### 2. Balance Your Training Set

Include a variety of:
- Degradation types (label swaps, merges, noise, etc.)
- Degradation levels (10%, 25%, 50%)
- Multiple original clusterings

### 3. Validate Results

```python
# Always check CV scores
print(f"CV R²: {result.cv_scores['r2_mean']:.3f}")

# Always check sanity
print(f"Sanity: {'PASS' if result.sanity_check_passed else 'FAIL'}")

# Review top features for reasonableness
for metric, weight in result.feature_importance[:10]:
    print(f"  {metric}: {weight:+.4f}")
```

### 4. Start with Ridge, Move to Lasso

1. **Ridge first**: See which metrics have signal
2. **Lasso second**: If you want fewer metrics in production

---

## CLI Training

```bash
# Basic training
metricate train training_data.csv -o weights.json

# With auto-tuned alpha
metricate train training_data.csv -o weights.json --auto-alpha

# With Lasso for feature selection
metricate train training_data.csv -o weights.json -r lasso --auto-alpha

# Full options
metricate train training_data.csv \
  -o weights.json \
  --regularization ridge \
  --alpha 1.0 \
  --cv-splits 5 \
  --top-n 15
```

---

## Quick Score Without Training (Not Recommended)

If you need a single number but don't have trained weights:

```python
result = metricate.evaluate("clustering.csv", final_score=True)
print(f"Final Score: {result.final_score:.3f}")
```

> ⚠️ **WARNING: Not recommended for production!**
>
> The unweighted final score:
> - Assumes all metrics are equally important
> - Correlated metrics cause implicit overweighting  
> - Unbounded metrics use heuristic normalization
> - No validation against actual clustering quality
>
> For reliable scoring, train proper weights.
