# Quickstart: Metric Weights Learner

**Phase 1 Output** | **Date**: 2026-03-01

## Installation

No new dependencies required - scikit-learn already in pyproject.toml.

```bash
pip install -e .
```

## Usage

### 1. Train Weights from Training Dataset

```python
import metricate

# Train on existing training data
result = metricate.train_weights(
    "training_17clusters/training_data.csv",
    regularization="ridge",  # or "lasso" for feature selection
)

# Check results
print(f"CV R²: {result.cv_scores['r2']:.3f}")
print(f"Non-zero metrics: {result.weights.non_zero_count}")
print(f"Sanity check: {'PASS' if result.sanity_check_passed else 'FAIL'}")

# Top 5 most important metrics
for metric, weight in result.feature_importance[:5]:
    print(f"  {metric}: {weight:.4f}")
```

### 2. Export Weights

```python
# Save to JSON for production use
result.weights.to_json("weights.json")

# Or get as dict
weights_dict = result.weights.to_dict()
```

### 3. Use Weights in Evaluation

```python
# Load saved weights
weights = metricate.load_weights("weights.json")

# Evaluate with compound score
result = metricate.evaluate("my_clustering.csv", weights=weights)

print(f"Compound Score: {result.compound_score:.3f}")
print(result.to_table())  # Shows individual metrics + compound score
```

### 4. Compare with Weights

```python
# Compare two clusterings using compound score
comparison = metricate.compare(
    "clustering_v1.csv",
    "clustering_v2.csv",
    weights=weights
)

print(f"Winner: {comparison.winner}")
print(f"A compound score: {comparison.evaluations['A'].compound_score:.3f}")
print(f"B compound score: {comparison.evaluations['B'].compound_score:.3f}")
```

### 5. CLI Usage

```bash
# Train and save weights
metricate train training_17clusters/training_data.csv -o weights.json

# Evaluate with weights
metricate evaluate clustering.csv --weights weights.json

# Compare with weights
metricate compare v1.csv v2.csv --weights weights.json
```

## Example Output

### Training Output
```
Training on 74 samples with 34 features...
Cross-validation R²: 0.847 (± 0.032)
Cross-validation RMSE: 0.089
Cross-validation MAE: 0.071

Sanity check: PASS (1/1 base clusterings)

Top 10 metrics by importance:
  1. Silhouette_norm: 0.1523
  2. R-squared_norm: 0.1245
  3. Davies-Bouldin_norm: 0.0892
  4. Calinski-Harabasz_norm: 0.0834
  5. Wemmert-Gancarski_norm: 0.0756
  ...

Zeroed metrics (Lasso): Dunn Index_norm, G-plus_norm, Tau_norm

Weights saved to: weights.json
```

### Evaluation with Weights
```
┌────────────────────────┬──────────┬─────────┬───────────┐
│ Metric                 │ Value    │ Range   │ Direction │
├────────────────────────┼──────────┼─────────┼───────────┤
│ Silhouette             │ 0.401    │ [-1, 1] │ higher    │
│ Davies-Bouldin         │ 0.830    │ [0, ∞)  │ lower     │
│ ...                    │ ...      │ ...     │ ...       │
├────────────────────────┼──────────┼─────────┼───────────┤
│ ★ Compound Score       │ 0.892    │ [0, 1]  │ higher    │
└────────────────────────┴──────────┴─────────┴───────────┘
```

## Formula Reference

The compound score is computed as:

```
compound_score = clip(Σ(weight_i × metric_i_norm) + bias, 0, 1)
```

Where:
- `metric_i_norm` = normalized metric value (from `_norm` column or computed at runtime)
- `weight_i` = learned coefficient for metric i
- `bias` = intercept term from regression
- `clip(x, 0, 1)` = ensures output stays in [0, 1] range

If some metrics are unavailable, weights are renormalized:
```
scale = sum(all_weights) / sum(available_weights)
adjusted_score = clip(scale × Σ(w_i × m_i for available) + bias, 0, 1)
```
