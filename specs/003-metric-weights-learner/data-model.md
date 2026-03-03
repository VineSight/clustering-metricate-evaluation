# Data Model: Metric Weights Learner

**Phase 1 Output** | **Date**: 2026-03-01

## Entities

### MetricWeights

Represents learned coefficients for computing compound clustering quality score.

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `dict[str, float]` | Mapping of metric name → weight (e.g., `{"Silhouette_norm": 0.15, ...}`) |
| `bias` | `float` | Intercept term from regression |
| `version` | `str` | Schema version for forward compatibility (e.g., "1.0") |
| `regularization` | `str` | Type used: "ridge" or "lasso" |
| `alpha` | `float` | Regularization strength used |
| `created_at` | `str` | ISO 8601 timestamp |
| `training_samples` | `int` | Number of samples used for training |
| `cv_r2` | `float` | Cross-validation R² score |
| `non_zero_count` | `int` | Number of non-zero coefficients |

**Constraints**:
- All coefficient keys must end with `_norm`
- Coefficients can be positive, negative, or zero
- Version must be semver format

### TrainingResult

Output from the training process.

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `MetricWeights` | The learned weights |
| `cv_scores` | `dict[str, float]` | Cross-validation metrics: `r2`, `rmse`, `mae` |
| `feature_importance` | `list[tuple[str, float]]` | Metrics ranked by `abs(coefficient)` descending |
| `zeroed_metrics` | `list[str]` | Metrics with coefficient = 0 (for Lasso) |
| `sanity_check_passed` | `bool` | True if original > all degraded for all clusterings |
| `sanity_failures` | `list[str]` | Clustering names where sanity check failed |

### CVResult

Cross-validation fold results.

| Field | Type | Description |
|-------|------|-------------|
| `fold` | `int` | Fold index (0-based) |
| `held_out_group` | `str` | Clustering name held out |
| `train_size` | `int` | Number of training samples |
| `test_size` | `int` | Number of test samples |
| `r2` | `float` | R² score on held-out fold |
| `rmse` | `float` | Root mean squared error |
| `mae` | `float` | Mean absolute error |

## Relationships

```
TrainingResult
    └── weights: MetricWeights
    └── cv_results: list[CVResult]
    
EvaluationResult (existing, modified)
    └── compound_score: float | None  # NEW: Added when weights provided
    └── compound_score_warning: str | None  # NEW: If metrics were missing
```

## State Transitions

```
Training Dataset (CSV)
    │
    ▼ train_weights()
TrainingResult
    │
    ├── .weights.to_json() → weights.json
    │
    └── .weights → MetricWeights
                      │
                      ▼ metricate.evaluate(..., weights=)
                  EvaluationResult with compound_score
```

## Validation Rules

### MetricWeights
- `coefficients` must contain at least 1 metric
- `bias` must be finite (not NaN or inf)
- `cv_r2` should be in range [-∞, 1], warn if negative (poor fit)

### TrainingResult
- `sanity_check_passed` = True required for production use
- `cv_scores['r2']` ≥ 0.7 recommended (from success criteria)

### Compound Score Computation
- All input metrics must be from `_norm` columns (pre-normalized)
- Output clipped to [0, 1]
- Warn if < 80% of weighted metrics available
