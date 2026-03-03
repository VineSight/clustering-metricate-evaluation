# Implementation Plan: Metric Weights Learner

**Branch**: `003-metric-weights-learner` | **Date**: 2026-03-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-metric-weights-learner/spec.md`

## Summary

Train a Ridge/Lasso regression model on degraded clustering datasets to learn optimal metric weights for quality scoring. The learned coefficients form a fixed formula `score = clip(Σ(weight_i × metric_i) + bias, 0, 1)` that integrates with existing `metricate.evaluate()` and `metricate.compare()` via a `weights=` parameter.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: scikit-learn (Ridge, Lasso, cross_val_score), pandas, numpy (already in pyproject.toml)
**Storage**: JSON files for exported weights, CSV for training data (existing)
**Testing**: pytest (existing setup in pyproject.toml)
**Target Platform**: Any platform supporting Python 3.10+
**Project Type**: Single package (metricate)
**Performance Goals**: Training completes within 30 seconds on ~70 samples
**Constraints**: Small dataset (~70 samples, 34 features) - must avoid overfitting
**Scale/Scope**: 34 metrics, 1 base clustering with 73 degraded variants

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| MAX_PACKAGES=1 | ✅ PASS | All code in `metricate` package |
| NO_ORM | ✅ PASS | Using pandas DataFrames directly |
| NO_ASYNC | ✅ PASS | Synchronous scikit-learn operations |
| PREFER_COMPOSITION | ✅ PASS | Functional approach with `train()`, `score()`, `export()` |

## Project Structure

### Documentation (this feature)

```text
specs/003-metric-weights-learner/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (weights JSON schema)
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
metricate/
├── __init__.py          # MODIFY: Add weights= param to evaluate(), compare()
├── core/
│   ├── evaluator.py     # MODIFY: Add compound score computation
│   └── reference.py     # READ: Metric direction info (existing)
├── comparison/
│   └── compare.py       # MODIFY: Add weighted winner determination
├── training/
│   ├── __init__.py      # MODIFY: Export new functions
│   ├── generator.py     # READ: Existing training data generation
│   ├── normalize.py     # READ: Existing normalization
│   ├── result.py        # READ: TrainingDataResult
│   ├── learner.py       # NEW: Core training logic (train, cross_validate, export)
│   └── weights.py       # NEW: MetricWeights dataclass, load/save, apply
└── output/
    └── report.py        # MODIFY: Add compound_score to EvaluationResult

tests/
├── unit/
│   ├── test_learner.py  # NEW: Unit tests for training
│   └── test_weights.py  # NEW: Unit tests for weight application
└── integration/
    └── test_weighted_eval.py  # NEW: Integration tests for API
```

**Structure Decision**: Single package structure maintained. New modules `learner.py` and `weights.py` added under `metricate/training/`. This keeps all weight-related logic co-located with existing training infrastructure.

## Complexity Tracking

> No constitution violations - table not needed.

## Phase 0: Research Summary

See [research.md](research.md) for full details.

**Key Decisions**:
1. Use Ridge regression (not Lasso) as primary - more stable with correlated metrics
2. Use scikit-learn's `GroupKFold` for leave-one-clustering-out CV
3. Min-max normalization to [0,1] already exists in training data (`_norm` columns)
4. Metric direction flipping handled by using `_norm` columns which are already direction-aware

## Phase 1: Design Summary

See [data-model.md](data-model.md) and [contracts/](contracts/) for full details.

**Key Entities**:
- `MetricWeights`: Dataclass holding coefficients dict, bias, and metadata
- `TrainingResult`: Training output with model, CV results, and exportable weights

**API Changes**:
- `metricate.evaluate(..., weights=)` → adds `compound_score` to result
- `metricate.compare(..., weights=)` → uses compound score for winner if provided
- `metricate.train_weights(training_csv, ...)` → returns TrainingResult
- `metricate.load_weights(path)` → returns MetricWeights
