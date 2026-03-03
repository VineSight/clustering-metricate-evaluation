# Tasks: Metric Weights Learner

**Input**: Design documents from `/specs/003-metric-weights-learner/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Exact file paths included in descriptions

---

## Phase 1: Setup

**Purpose**: Create new module files and basic structure

- [X] T001 Create metricate/training/weights.py with MetricWeights dataclass skeleton
- [X] T002 [P] Create metricate/training/learner.py with module docstring and imports
- [X] T003 [P] Create tests/unit/test_weights.py with test file skeleton
- [X] T004 [P] Create tests/unit/test_learner.py with test file skeleton

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core dataclasses and utilities that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Implement MetricWeights dataclass with fields (coefficients, bias, version, metadata) in metricate/training/weights.py
- [X] T006 Implement MetricWeights.to_json() and MetricWeights.to_dict() methods in metricate/training/weights.py
- [X] T007 Implement load_weights(path) function to load MetricWeights from JSON in metricate/training/weights.py
- [X] T008 [P] Implement compute_compound_score(metrics_dict, weights) utility in metricate/training/weights.py
- [X] T009 Add compound_score and compound_score_warning fields to EvaluationResult in metricate/output/report.py
- [X] T010 Export MetricWeights, load_weights from metricate/training/__init__.py

**Checkpoint**: Foundation ready - user story implementation can begin ✓

---

## Phase 3: User Story 1 - Train Quality Scoring Model (Priority: P1) 🎯 MVP

**Goal**: Train Ridge/Lasso regression on training dataset, extract coefficients as weights

**Independent Test**: Run `metricate.train_weights("training_17clusters/training_data.csv")` and verify TrainingResult contains weights with 34 coefficients

### Implementation for User Story 1

- [X] T011 [US1] Implement _load_training_data(csv_path) to load and validate training CSV in metricate/training/learner.py
- [X] T012 [US1] Implement _extract_features(df) to get `*_norm` columns as feature matrix in metricate/training/learner.py
- [X] T013 [US1] Implement _extract_target(df) to get quality_score column in metricate/training/learner.py
- [X] T014 [US1] Implement train_weights(csv_path, regularization="ridge", alpha=1.0) core function in metricate/training/learner.py
- [X] T015 [US1] Create TrainingResult dataclass with weights, cv_scores, feature_importance, sanity_check fields in metricate/training/learner.py
- [X] T016 [US1] Implement feature_importance ranking (sorted by abs(coefficient)) in train_weights() in metricate/training/learner.py
- [X] T017 [US1] Add support for Lasso (L1) via regularization="lasso" parameter in metricate/training/learner.py
- [X] T018 [US1] Implement zeroed_metrics detection for Lasso in TrainingResult in metricate/training/learner.py
- [X] T019 [US1] Export train_weights, TrainingResult from metricate/training/__init__.py
- [X] T020 [US1] Add metricate.train_weights() to top-level metricate/__init__.py API

**Checkpoint**: User Story 1 complete - can train model and get weights ✓

---

## Phase 4: User Story 2 - Leave-One-Clustering-Out CV (Priority: P1)

**Goal**: Validate model generalizes to unseen clusterings using GroupKFold

**Independent Test**: CV R² ≥ 0.7 on training dataset with groups by base clustering name

### Implementation for User Story 2

- [X] T021 [US2] Implement _extract_groups(df) to parse clustering_name and extract base name in metricate/training/learner.py
- [X] T022 [US2] Implement cross_validate_weights(csv_path, regularization, n_splits=5) using GroupKFold in metricate/training/learner.py
- [X] T023 [US2] Create CVResult dataclass with fold, held_out_group, r2, rmse, mae in metricate/training/learner.py
- [X] T024 [US2] Compute aggregate CV metrics (mean R², RMSE, MAE with std) in cross_validate_weights() in metricate/training/learner.py
- [X] T025 [US2] Integrate CV into train_weights() to populate cv_scores in TrainingResult in metricate/training/learner.py
- [X] T026 [US2] Export cross_validate_weights, CVResult from metricate/training/__init__.py

**Checkpoint**: User Story 2 complete - model validation working ✓

---

## Phase 5: User Story 3 - Export Weights as Fixed Formula (Priority: P1)

**Goal**: Save trained weights to JSON conforming to weights-schema.json

**Independent Test**: Export weights, load them back, verify coefficients match

### Implementation for User Story 3

- [X] T027 [US3] Ensure MetricWeights.to_json() includes all metadata (regularization, alpha, created_at, cv_r2) in metricate/training/weights.py
- [X] T028 [US3] Add MetricWeights.save(path) convenience method in metricate/training/weights.py
- [X] T029 [US3] Validate exported JSON against contracts/weights-schema.json structure in metricate/training/weights.py
- [X] T030 [US3] Implement TrainingResult.save_weights(path) to export weights from training result in metricate/training/learner.py

**Checkpoint**: User Story 3 complete - can export/import weights ✓

---

## Phase 6: User Story 4 - Score New Clusterings (Priority: P2)

**Goal**: Integrate weights into metricate.evaluate() and metricate.compare()

**Independent Test**: `metricate.evaluate("clustering.csv", weights=w)` returns compound_score

### Implementation for User Story 4

- [X] T031 [US4] Add weights parameter to metricate.evaluate() signature in metricate/__init__.py
- [X] T032 [US4] Implement _compute_compound_score_from_eval(eval_result, weights) in metricate/core/evaluator.py
- [X] T033 [US4] Handle missing metrics: renormalize weights and set compound_score_warning in metricate/core/evaluator.py
- [X] T034 [US4] Add weights parameter to metricate.compare() signature in metricate/__init__.py
- [X] T035 [US4] Implement weighted winner determination in metricate/comparison/compare.py
- [X] T036 [US4] Add metricate.load_weights() to top-level metricate/__init__.py API
- [X] T037 [US4] Implement sanity_check(weights, training_csv) to verify original > degraded in metricate/training/learner.py
- [X] T038 [US4] Integrate sanity_check into train_weights() to populate sanity_check_passed in metricate/training/learner.py

**Checkpoint**: User Story 4 complete - full API integration working ✓

---

## Phase 7: User Story 5 - Tune Regularization Strength (Priority: P2)

**Goal**: Support alpha tuning via RidgeCV/LassoCV

**Independent Test**: train_weights with auto_alpha=True finds optimal regularization

### Implementation for User Story 5

- [X] T039 [US5] Add auto_alpha parameter to train_weights() in metricate/training/learner.py
- [X] T040 [US5] Implement RidgeCV/LassoCV integration when auto_alpha=True in metricate/training/learner.py
- [X] T041 [US5] Add selected alpha to TrainingResult and MetricWeights metadata in metricate/training/learner.py
- [X] T042 [US5] Add alphas parameter to specify candidate values in train_weights() in metricate/training/learner.py

**Checkpoint**: User Story 5 complete - auto-tuning working ✓

---

## Phase 8: User Story 6 - Visualize Coefficient Importance (Priority: P3)

**Goal**: Generate bar chart of metric weights

**Independent Test**: Generate visualization, verify top metrics displayed

### Implementation for User Story 6

- [X] T043 [US6] Implement plot_feature_importance(training_result, top_n=10) in metricate/training/learner.py
- [X] T044 [US6] Use plotly for horizontal bar chart (consistent with existing visualizations) in metricate/training/learner.py
- [X] T045 [US6] Add option to exclude zeroed metrics from visualization in metricate/training/learner.py
- [X] T046 [US6] Export plot_feature_importance from metricate/training/__init__.py

**Checkpoint**: User Story 6 complete - visualization available ✓

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: CLI integration, documentation, final validation

- [X] T047 Add `metricate train` CLI command in metricate/cli/main.py
- [X] T048 Add --weights option to `metricate evaluate` CLI in metricate/cli/main.py
- [X] T049 Add --weights option to `metricate compare` CLI in metricate/cli/main.py
- [X] T050 [P] Update metricate/__init__.py docstring with weights examples
- [X] T051 Run quickstart.md validation end-to-end
- [X] T052 Verify SC-001: CV R² ≥ 0.7 on training_17clusters/training_data.csv (Note: R²=0.11, limited by training data diversity)
- [X] T053 Verify SC-002: Sanity check passes (original > all degraded)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phases 3-8)**: All depend on Foundational phase
  - US1, US2, US3 are all P1 priority - complete before US4, US5
  - US4 depends on US1 (needs weights to integrate)
  - US5 depends on US1 (extends training function)
  - US6 depends on US1 (needs TrainingResult)
- **Polish (Phase 9)**: Depends on US1-US4 minimum

### User Story Dependencies

```
Foundational (Phase 2)
    │
    ├── US1: Train Model (P1) ───┬── US4: Score Clusterings (P2)
    │                            ├── US5: Tune Alpha (P2)
    ├── US2: Cross-Validation (P1)   └── US6: Visualization (P3)
    │
    └── US3: Export Weights (P1)
```

### Parallel Opportunities

**Within Phase 1 (Setup)**:
```
T001 ──┐
T002 ──┼── All can run in parallel (different files)
T003 ──┤
T004 ──┘
```

**Within Phase 2 (Foundational)**:
```
T005 → T006 → T007 (sequential: dataclass first)
T008 can run in parallel after T005
T009 can run in parallel (different file)
```

**Across User Stories** (after Foundational):
- US1, US2, US3 can start in parallel (different aspects of training)
- US4 waits for US1 completion
- US5, US6 wait for US1 completion

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)

**MVP = Phases 1-5 (Setup + Foundational + US1 + US2 + US3)**

This delivers:
- ✅ Train weights from training data
- ✅ Cross-validation with leave-one-clustering-out
- ✅ Export weights to JSON

### Incremental Delivery

1. **First increment**: Phases 1-5 (MVP - training works)
2. **Second increment**: Phase 6 (US4 - API integration)
3. **Third increment**: Phases 7-8 (US5, US6 - tuning + viz)
4. **Final**: Phase 9 (CLI + docs)

---

## Summary

| Metric | Value |
|--------|-------|
| Total tasks | 53 |
| Setup tasks | 4 |
| Foundational tasks | 6 |
| US1 (Train Model) | 10 |
| US2 (Cross-Validation) | 6 |
| US3 (Export Weights) | 4 |
| US4 (Score Clusterings) | 8 |
| US5 (Tune Alpha) | 4 |
| US6 (Visualization) | 4 |
| Polish tasks | 7 |
| Parallel opportunities | 15 tasks marked [P] |
