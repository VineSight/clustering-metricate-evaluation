# Feature Specification: Metric Weights Learner

**Feature Branch**: `003-metric-weights-learner`  
**Created**: 2026-03-01  
**Status**: Draft  
**Input**: User description: "Logistic Regression with L1 (Lasso) regularization that uses the clusterings training dataset to learn metric weights for quality scoring"

## Overview

This feature implements a machine learning pipeline that learns optimal weights for clustering quality metrics. By training on degraded clustering datasets with known quality levels, the system produces a fixed, interpretable formula for scoring any clustering result.

The approach uses Ridge/Lasso regression with continuous quality labels (rather than binary classification) to preserve the granularity of degradation levels. The learned coefficients become the metric weights, and L1 regularization performs automatic feature selection by zeroing out uninformative metrics.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Quality Scoring Model (Priority: P1)

A researcher has generated a training dataset containing clusterings at various quality levels (original + degraded versions). They want to train a model that learns which metrics best predict clustering quality and produces a simple weighted formula for future use.

**Why this priority**: This is the core functionality - without training capability, no other features work.

**Independent Test**: Can be fully tested by providing the training dataset CSV and verifying that a model is trained with extractable coefficients.

**Acceptance Scenarios**:

1. **Given** a training dataset CSV with normalized metrics and quality scores, **When** the user initiates model training, **Then** a regression model is trained and coefficients are extracted as metric weights.

2. **Given** a training dataset with multiple degradation levels, **When** training completes, **Then** the model correctly maps quality_score values (1.0 for original, decreasing for degraded) to predicted scores.

3. **Given** a training run with L1 regularization, **When** training completes, **Then** some metric weights are set to exactly zero (feature selection).

---

### User Story 2 - Evaluate Model with Leave-One-Clustering-Out CV (Priority: P1)

A researcher wants to validate that the learned weights generalize to unseen clusterings, not just unseen degradation levels of the same clustering.

**Why this priority**: Without proper validation, the model may memorize clustering-specific patterns rather than learning universal metric importance.

**Independent Test**: Can be tested by verifying that cross-validation holds out all versions of each base clustering together.

**Acceptance Scenarios**:

1. **Given** a training dataset with multiple base clusterings and their degradations, **When** leave-one-clustering-out CV is performed, **Then** all degradation variants of a held-out clustering are excluded from training together.

2. **Given** cross-validation completes, **When** results are summarized, **Then** per-fold and aggregate metrics (R², RMSE, MAE) are reported.

---

### User Story 3 - Export Weights as Fixed Formula (Priority: P1)

A researcher has trained a model and wants to export the learned weights for use in production scoring without requiring the ML model.

**Why this priority**: The end goal is a simple, interpretable formula that can be used anywhere without dependencies.

**Independent Test**: Can be tested by exporting weights and manually computing a score using the formula.

**Acceptance Scenarios**:

1. **Given** a trained model, **When** the user exports weights, **Then** a JSON/YAML file contains coefficient for each metric plus the bias term.

2. **Given** exported weights, **When** applied to a new clustering's normalized metrics, **Then** score = sigmoid(Σ(weight_i × metric_i) + bias) produces a value in [0, 1].

3. **Given** exported weights, **When** metrics are ranked by absolute coefficient value, **Then** the most informative metrics for quality prediction are clearly identified.

---

### User Story 4 - Score New Clusterings (Priority: P2)

A user has a trained model or exported weights and wants to score a new clustering result.

**Why this priority**: This is the practical application of the trained model, but depends on training being complete first.

**Independent Test**: Can be tested by computing a quality score for a known clustering and verifying the score is reasonable.

**Acceptance Scenarios**:

1. **Given** a trained model and a clustering CSV with embeddings and labels, **When** the user requests a quality score, **Then** a score between 0 and 1 is returned.

2. **Given** an original (undegraded) clustering and all its degraded variants, **When** each is scored, **Then** the original's score is strictly higher than all degraded versions (sanity check).

3. **Given** a heavily degraded clustering, **When** scored, **Then** the score is significantly lower than the original.

---

### User Story 5 - Tune Regularization Strength (Priority: P2)

A researcher wants to experiment with different regularization strengths to balance model complexity and predictive power.

**Why this priority**: Fine-tuning is valuable but the default should work well enough for most cases.

**Independent Test**: Can be tested by training with different alpha values and comparing cross-validation scores.

**Acceptance Scenarios**:

1. **Given** a range of regularization strengths (alpha values), **When** cross-validation is run for each, **Then** performance metrics for each alpha are reported.

2. **Given** cross-validation results, **When** optimal alpha is selected, **Then** the model with the best CV score is retained.

---

### User Story 6 - Visualize Coefficient Importance (Priority: P3)

A researcher wants to understand which metrics contribute most to quality prediction through visualization.

**Why this priority**: Visualization aids interpretability but is not required for core functionality.

**Independent Test**: Can be tested by generating a coefficient bar chart and verifying all non-zero metrics are displayed.

**Acceptance Scenarios**:

1. **Given** a trained model, **When** the user requests coefficient visualization, **Then** a horizontal bar chart shows metrics ranked by absolute coefficient value.

2. **Given** the visualization, **When** L1 regularization has zeroed some metrics, **Then** those metrics are either omitted or clearly shown as zero.

---

### Edge Cases

- What happens when all metrics are highly correlated? (L1 will arbitrarily pick one; Ridge may be more stable)
- How does the system handle metrics with missing values? (Require complete data or impute)
- What happens when the training dataset has fewer samples than features? (Over-regularize or warn user)
- How does the system handle metrics that are constant across all samples? (Exclude from training)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a training dataset CSV containing normalized metric columns and a quality_score column.
- **FR-002**: System MUST support both Ridge (L2) and Lasso (L1) regularization options.
- **FR-003**: System MUST normalize input metrics using min-max scaling to [0,1] range before training.
- **FR-004**: System MUST assign quality labels as continuous values: 1.0 for original, 1 - (degradation_level / max_level) for degraded samples.
- **FR-005**: System MUST implement leave-one-clustering-out cross-validation that groups all degradation variants of each base clustering together.
- **FR-006**: System MUST extract and export learned coefficients as the metric weight vector.
- **FR-007**: System MUST clip predicted scores to the [0, 1] range.
- **FR-008**: System MUST report which metrics were zeroed out by L1 regularization.
- **FR-009**: System MUST compute and report cross-validation metrics: R², RMSE, MAE.
- **FR-010**: System MUST support hyperparameter tuning for regularization strength via cross-validation.
- **FR-011**: System MUST export weights in a portable format (JSON/YAML) including all coefficients and bias.
- **FR-012**: System MUST exclude domain embeddings from features by default to prevent overfitting on small datasets.
- **FR-013**: System MUST integrate with existing `metricate.evaluate()` via optional `weights=` parameter that returns compound score alongside individual metrics.
- **FR-014**: System MUST integrate with existing `metricate.compare()` via optional `weights=` parameter that uses compound score for winner determination.
- **FR-015**: System MUST ensure that for any base clustering, its compound score is strictly higher than all its degraded variants (sanity validation).
- **FR-016**: When computing compound score with missing metrics, system MUST renormalize remaining weights and issue a warning listing which metrics were unavailable.
- **FR-017**: System MUST flip "lower-is-better" metrics (multiply by -1) before training so all learned weights have consistent direction (positive weight = contributes to quality).

### Key Entities

- **TrainingDataset**: Contains clustering samples with normalized metrics, quality scores, clustering identifiers, and degradation metadata.
- **MetricWeights**: The learned coefficient vector mapping each metric to its importance for quality prediction, plus a bias term.
- **QualityModel**: The trained regression model that can score new clusterings and export weights.
- **CVResult**: Cross-validation results containing per-fold metrics and aggregate statistics.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model achieves R² ≥ 0.7 on leave-one-clustering-out cross-validation, demonstrating generalization to unseen clusterings.
- **SC-002**: For every base clustering in the training set, the predicted score is strictly higher than all its degraded variants (100% sanity check pass rate).
- **SC-003**: Predicted quality scores monotonically decrease as degradation level increases (within each degradation type).
- **SC-004**: L1 regularization reduces the number of non-zero coefficients by at least 30%, demonstrating effective feature selection.
- **SC-005**: The exported weight formula reproduces model predictions within ±0.01 when manually applied.
- **SC-006**: Training completes within 30 seconds on the current dataset size (~70 samples).

## Assumptions

- The training dataset contains pre-computed normalized metrics (columns ending with `_norm`).
- The `quality_score` column already contains appropriate continuous values (1.0 for originals, lower for degraded).
- The `clustering_name` column can be parsed to identify base clusterings and their degradation variants.
- Domain embeddings are excluded from the feature set to maintain domain-agnostic weights.
- Metrics with constant values across all samples will be automatically excluded.

## Clarifications

### Session 2026-03-01

- Q: How should learned weights integrate with existing metricate evaluation and comparison modules? → A: Add `weights=` parameter to existing `evaluate()` and `compare()` functions that computes compound score alongside individual metrics.
- Q: How should compound score behave when some metrics couldn't be computed? → A: Reweight using only available metrics (renormalize weights to sum to original total) and issue a warning listing which metrics were missing.
- Q: Which normalization method should be used for metrics before applying weights? → A: Min-max normalization to [0,1] range, consistent with existing `_norm` columns in training data.
- Q: How should metric direction (higher vs lower is better) be handled? → A: Flip "lower-is-better" metrics by multiplying by -1 at training time so all weights have consistent positive-means-quality interpretation.
