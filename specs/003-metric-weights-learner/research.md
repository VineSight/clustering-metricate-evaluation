# Research: Metric Weights Learner

**Phase 0 Output** | **Date**: 2026-03-01

## Research Tasks

### 1. Ridge vs Lasso for Correlated Features

**Decision**: Use Ridge (L2) as default, offer Lasso (L1) as option

**Rationale**:
- With 34 metrics and only ~70 samples, features-to-samples ratio is problematic
- Many clustering metrics are highly correlated (e.g., Silhouette correlates with Davies-Bouldin inverse)
- Lasso arbitrarily picks one of correlated features, zeroing others - unstable
- Ridge distributes weight among correlated features - more stable coefficients
- Lasso still valuable for interpretability when user explicitly wants feature selection

**Alternatives considered**:
- ElasticNet (mix of L1+L2): Added complexity, not needed for this dataset size
- Plain OLS: Would overfit badly with 34 features and 70 samples

### 2. Leave-One-Clustering-Out Cross-Validation

**Decision**: Use scikit-learn's `GroupKFold` with clustering name as group key

**Rationale**:
- Training data has 1 base clustering + 73 degraded variants (74 total rows)
- All variants share underlying structure - must be held out together
- `GroupKFold` naturally handles this by grouping on base clustering name
- Alternative: custom splitter, but `GroupKFold` is battle-tested

**Implementation**:
```python
from sklearn.model_selection import GroupKFold, cross_val_score

# Extract base clustering name (strip degradation suffix)
groups = df['clustering_name'].apply(lambda x: x.split('_')[0])  
cv = GroupKFold(n_splits=min(n_unique_groups, 5))
scores = cross_val_score(model, X, y, cv=cv, groups=groups)
```

**Alternatives considered**:
- KFold: Would leak information (train/test contain variants of same clustering)
- LeaveOneGroupOut: Too few folds with single base clustering

### 3. Handling Metric Direction

**Decision**: Use existing `_norm` columns which already handle direction

**Rationale**:
- Training data already has `Silhouette_norm`, `Davies-Bouldin_norm`, etc.
- Examined normalization code in `metricate/training/normalize.py`
- The `_norm` columns apply min-max scaling AND flip direction so higher=better for all
- No additional direction flipping needed at training time

**Verification needed**: Confirm `_norm` columns flip "lower-is-better" metrics

### 4. Compound Score Formula

**Decision**: Linear combination with clip, no sigmoid

**Rationale**:
- `score = clip(Σ(weight_i × norm_metric_i) + bias, 0, 1)`
- Sigmoid unnecessary when inputs already bounded [0,1] and target is [0,1]
- Linear regression naturally produces bounded output when trained on bounded targets
- Clip handles edge cases where extrapolation exceeds bounds
- Simpler formula = easier to validate and explain

**Alternatives considered**:
- Sigmoid output: Adds complexity, compresses differences near extremes
- Softmax across metrics: Not applicable to single score output

### 5. Handling Missing Metrics at Inference

**Decision**: Renormalize weights and warn

**Rationale**:
- Large datasets auto-skip O(n²) metrics (Gamma, Tau, G-plus)
- Some metrics may fail (division by zero, numerical issues)
- Can't require all metrics - would make weighted scoring useless on large datasets
- Renormalization: `adjusted_score = Σ(w_i × m_i for available) / Σ(w_i for available) × Σ(all w_i)`
- Warning tells user which metrics were missing

**Implementation**:
```python
available_weights = {k: v for k, v in weights.items() if k in computed_metrics}
if len(available_weights) < len(weights):
    missing = set(weights.keys()) - set(available_weights.keys())
    warn(f"Missing metrics for compound score: {missing}")
    
total_original = sum(weights.values())
total_available = sum(available_weights.values())
scale_factor = total_original / total_available if total_available else 1.0

raw_score = sum(w * metrics[k] for k, w in available_weights.items())
compound_score = clip(raw_score * scale_factor + bias, 0, 1)
```

### 6. Training Data Structure Analysis

**Decision**: Use `quality_score` as target, `*_norm` columns as features

**Rationale**:
- Examined `training_17clusters/training_data.csv` structure:
  - 74 rows (1 original + 73 degraded)
  - `quality_score` column: 1.0 for original, 0.95/0.90/0.75/0.50/0.25/0.00 for degraded
  - 34 `*_norm` columns with min-max normalized metrics
- Quality score already continuous - no need to compute from degradation level
- Normalized metrics ready for training - no additional preprocessing

**Feature columns** (34 total):
```
Silhouette_norm, Davies-Bouldin_norm, Calinski-Harabasz_norm, Dunn Index_norm,
SSE_norm, NCI_norm, Ball-Hall_norm, Ratkowsky-Lance_norm, Ray-Turi_norm,
RMSSTD_norm, R-squared_norm, Wemmert-Gancarski_norm, CS index_norm, COP_norm,
S_Dbw_norm, Det Ratio_norm, Gamma_norm, Generalized Dunn_norm, G-plus_norm,
I-index (PBM)_norm, Log_Det_Ratio_norm, McClain-Rao_norm, Point-Biserial_norm,
SD validity_norm, Tau_norm, Trace_WiB_norm, Ksq_DetW_norm, Banfield-Raftery_norm,
Negentropy_norm, NIVA_norm, Score Function_norm, Scott-Symons_norm
```

### 7. Hyperparameter Tuning Strategy

**Decision**: Use `RidgeCV` with built-in alpha selection

**Rationale**:
- scikit-learn's `RidgeCV` does efficient leave-one-out CV for alpha selection
- Default alphas: `[0.1, 1.0, 10.0]` covers reasonable range
- For Lasso, use `LassoCV` with same approach
- No need for grid search - built-in CV is faster and sufficient

**Implementation**:
```python
from sklearn.linear_model import RidgeCV, LassoCV

model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
model.fit(X, y)
print(f"Best alpha: {model.alpha_}")
```

## Unresolved Items

None - all technical questions resolved.

## Dependencies to Add

None - scikit-learn already in pyproject.toml dependencies.
