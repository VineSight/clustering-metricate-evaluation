# Tasks: Labricate Weighted Evaluation & Computation Modes

**Input**: Design documents from `/specs/005-labricate-weighted-eval/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Tests**: Included as specified in contracts/testing.md (TC-001 to TC-027)

> **Note**: Test IDs use TC-XXX prefix (e.g., TC-001) to distinguish from task IDs (T001-T063).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3, US4) - only for story phases
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify project readiness (no new setup required - extending existing project)

- [x] T001 Verify existing metricate package structure and dependencies
- [x] T002 [P] Confirm metricate/training/weights.py exports: MetricWeights, load_weights, compute_compound_score, validate_weights_schema
- [x] T003 [P] Confirm metricate/core/reference.py has METRIC_REFERENCE with skip_large flags

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: All user stories depend on these foundational modules

### 2.1 Create modes.py Module

- [x] T004 Create metricate/labricate/core/modes.py with ComputationMode type alias
- [x] T005 Implement get_expensive_metrics() using METRIC_REFERENCE skip_large in metricate/labricate/core/modes.py
- [x] T006 Implement apply_mode_exclusions() for light/heavy filtering in metricate/labricate/core/modes.py
- [x] T007 [P] Add modes.py exports to metricate/labricate/core/__init__.py

### 2.2 Create scoring.py Module

- [x] T008 Create metricate/labricate/core/scoring.py with WeightCoverageWarning class
- [x] T009 Implement compute_run_scores() in metricate/labricate/core/scoring.py
- [x] T010 Implement find_best_run() with tie detection in metricate/labricate/core/scoring.py
- [x] T011 Implement check_weight_coverage() for FR-017 warning in metricate/labricate/core/scoring.py
- [x] T012 [P] Add scoring.py exports to metricate/labricate/core/__init__.py

### 2.3 Add BestRunInfo Dataclass

- [x] T013 Add BestRunInfo dataclass to metricate/labricate/core/experiment.py
- [x] T014 Implement BestRunInfo.__str__() for tie-aware display in metricate/labricate/core/experiment.py

### 2.4 Foundational Tests

- [x] T015 [P] Create tests/unit/test_labricate_modes.py with TC-001 to TC-005
- [x] T016 [P] Create tests/unit/test_labricate_scoring.py with TC-006 to TC-014
- [x] T016a [P] Add test_find_best_run_returns_none_when_all_failed (TC-012) to tests/unit/test_labricate_scoring.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Weighted Experiment Evaluation (Priority: P1) 🎯 MVP

**Goal**: Provide weights JSON to get compound scores and identify best run

**Independent Test**: Run experiment with weights JSON, verify results include compound_score and best_run field

### Tests for User Story 1

- [x] T017 [P] [US1] Add test_experiment_accepts_weights_path (TC-015) to tests/unit/test_labricate_weights.py
- [x] T018 [P] [US1] Add test_experiment_accepts_weights_dict (TC-016) to tests/unit/test_labricate_weights.py
- [x] T019 [P] [US1] Add test_experiment_without_weights_unchanged (TC-018) to tests/unit/test_labricate_weights.py

### Implementation for User Story 1

- [x] T020 [US1] Add weights parameter to Experiment.__init__() in metricate/labricate/core/experiment.py
- [x] T021 [US1] Add self._weights attribute loading via load_weights() in metricate/labricate/core/experiment.py
- [x] T022 [US1] Add compound_score field to RunResult dataclass in metricate/labricate/core/experiment.py
- [x] T023 [US1] Call compute_run_scores() after runs complete in Experiment.run() in metricate/labricate/core/experiment.py
- [x] T024 [US1] Call find_best_run() to populate result.best_run in Experiment.run() in metricate/labricate/core/experiment.py
- [x] T025 [US1] Add same weights/scoring logic to run_grid() in metricate/labricate/core/experiment.py
- [x] T026 [US1] Update metricate/labricate/__init__.py to export BestRunInfo

**Checkpoint**: User Story 1 complete - experiments with weights produce compound_score and best_run

---

## Phase 4: User Story 3 - Best Run Display in Results (Priority: P1)

> **Note**: US3 is implemented before US2 because US3 depends on compound_score from US1. US2 (modes) is independent and can be implemented in parallel with US1/US3 if desired.

**Goal**: Best run prominently displayed and accessible in all output formats

**Independent Test**: Run any experiment, verify result.best_run populated and displayed in summary/DataFrame/JSON

### Tests for User Story 3

- [x] T027 [P] [US3] Add test_to_dataframe_includes_compound_score (TC-019) to tests/unit/test_labricate_weights.py
- [x] T028 [P] [US3] Add test_to_dataframe_includes_is_best_run (TC-020) to tests/unit/test_labricate_weights.py
- [x] T029 [P] [US3] Add test_is_best_run_handles_ties (TC-021) to tests/unit/test_labricate_weights.py

### Implementation for User Story 3

- [x] T030 [US3] Add best_run field to ExperimentResult dataclass in metricate/labricate/core/experiment.py
- [x] T031 [US3] Modify to_dataframe() to include compound_score column in metricate/labricate/core/experiment.py
- [x] T032 [US3] Modify to_dataframe() to include is_best_run boolean column in metricate/labricate/core/experiment.py
- [x] T033 [US3] Print best run config when verbose=True in Experiment.run() in metricate/labricate/core/experiment.py
- [x] T034 [US3] Add best_run to JSON output in metricate/labricate/output/storage.py
- [x] T035 [US3] Implement get_best_run() to use compound_score when weights provided in metricate/labricate/core/experiment.py

**Checkpoint**: User Story 3 complete - best_run accessible via .best_run, to_dataframe(), JSON

---

## Phase 5: User Story 2 - Computation Mode Selection (Priority: P2)

**Goal**: Choose light (fast) or heavy (comprehensive) computation modes

**Independent Test**: Run identical experiments with mode="light" vs mode="heavy", verify light excludes expensive metrics

### Tests for User Story 2

- [x] T036 [P] [US2] Add test for mode parameter in run() to tests/unit/test_labricate_modes.py
- [x] T037 [P] [US2] Add test for light mode excluding 6 metrics to tests/unit/test_labricate_modes.py
- [x] T038 [P] [US2] Add test for include_metrics precedence over mode to tests/unit/test_labricate_modes.py

### Implementation for User Story 2

- [x] T039 [US2] Add mode parameter to Experiment.run() signature in metricate/labricate/core/experiment.py
- [x] T040 [US2] Add best_metric parameter to Experiment.run() signature in metricate/labricate/core/experiment.py
- [x] T041 [US2] Apply mode exclusions via apply_mode_exclusions() before evaluation in metricate/labricate/core/experiment.py
- [x] T042 [US2] Add weight coverage warning when mode=light with weights in metricate/labricate/core/experiment.py
- [x] T043 [US2] Add same mode/best_metric params to run_grid() in metricate/labricate/core/experiment.py

**Checkpoint**: User Story 2 complete - mode="light" excludes expensive metrics

---

## Phase 6: User Story 4 - Weights JSON Validation (Priority: P2)

**Goal**: Clear validation errors for malformed weights files

**Independent Test**: Provide various malformed weights files, verify clear actionable error messages

### Tests for User Story 4

- [x] T044 [P] [US4] Add test_experiment_validates_weights_schema (TC-017) to tests/unit/test_labricate_weights.py
- [x] T045 [P] [US4] Add test for missing coefficients error message to tests/unit/test_labricate_weights.py
- [x] T046 [P] [US4] Add test for invalid _norm suffix error message to tests/unit/test_labricate_weights.py

### Implementation for User Story 4

- [x] T047 [US4] Validate weights dict via validate_weights_schema() in Experiment.__init__() in metricate/labricate/core/experiment.py
- [x] T048 [US4] Raise ValueError with clear message for missing coefficients in metricate/labricate/core/experiment.py
- [x] T049 [US4] Raise ValueError with clear message for invalid coefficient keys in metricate/labricate/core/experiment.py

**Checkpoint**: User Story 4 complete - validation errors include actionable guidance

---

## Phase 7: CLI Integration & Polish

**Purpose**: CLI support and cross-cutting improvements

### CLI Options

- [x] T050 Add --weights/-w option to experiment command in metricate/cli/labricate.py
- [x] T051 Add --mode/-m option with choices light/heavy to experiment command in metricate/cli/labricate.py
- [x] T052 Pass weights and mode params to Experiment() in CLI handler in metricate/cli/labricate.py
- [x] T053 Display best_run in CLI output when experiment completes in metricate/cli/labricate.py

### CLI Tests

- [x] T054 [P] Add test_cli_weights_option (TC-025) to tests/unit/test_labricate_cli.py
- [x] T055 [P] Add test_cli_mode_option (TC-026) to tests/unit/test_labricate_cli.py
- [x] T056 [P] Add test_cli_json_output_includes_best_run (TC-027) to tests/unit/test_labricate_cli.py

### Integration Tests

- [x] T057 Create tests/integration/test_labricate_weighted_experiment.py
- [x] T058 [P] Add test_weighted_experiment_end_to_end (TC-022) to tests/integration/test_labricate_weighted_experiment.py
- [x] T058a [P] Add test_light_mode_faster_than_heavy (TC-023) to tests/integration/test_labricate_weighted_experiment.py
- [x] T059 [P] Add test_weight_coverage_warning_displayed (TC-024) to tests/integration/test_labricate_weighted_experiment.py

### Documentation & Cleanup

- [x] T060 [P] Update docs/labricate.md with weights and mode examples
- [x] T061 [P] Update docs/cli-reference.md with --weights and --mode options
- [x] T062 Run quickstart.md validation - verify all code snippets work
- [x] T063 Run full test suite to confirm SC-005 (existing functionality unchanged)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phase 3 (US1)**: Depends on Phase 2 completion
- **Phase 4 (US3)**: Depends on Phase 3 (needs compound_score for best_run)
- **Phase 5 (US2)**: Depends on Phase 2 only - can parallel with US1/US3
- **Phase 6 (US4)**: Depends on Phase 3 (needs weights in Experiment)
- **Phase 7 (Polish)**: Depends on all user stories complete

### User Story Dependencies

```
Phase 2 (Foundational)
        │
        ├─────────────────┬─────────────────┐
        ▼                 ▼                 ▼
   Phase 3 (US1)     Phase 5 (US2)    [independent]
        │                 │
        ▼                 │
   Phase 4 (US3)          │
        │                 │
        ▼                 │
   Phase 6 (US4)          │
        │                 │
        └────────┬────────┘
                 ▼
          Phase 7 (Polish)
```

### Parallel Opportunities

**Within Phase 2:**
```
T004-T006 (modes.py)  ||  T008-T012 (scoring.py)  ||  T013-T014 (BestRunInfo)
T015 (modes tests)    ||  T016, T016a (scoring tests)
```

**Within Phase 3:**
```
T017 || T018 || T019 (all tests in parallel)
```

**Within Phase 7:**
```
T054 || T055 || T056 (CLI tests)
T058 || T058a || T059 (integration tests)
T060 || T061 (docs)
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 3)

1. Complete Phase 1: Setup verification
2. Complete Phase 2: Foundational modules
3. Complete Phase 3: User Story 1 (weighted eval)
4. Complete Phase 4: User Story 3 (best_run display)
5. **STOP and VALIDATE**: Test weighted experiments independently
6. MVP delivers: weights → compound_score → best_run identification

### Incremental Delivery

| Increment | Stories | Delivers |
|-----------|---------|----------|
| MVP | US1 + US3 | Weighted scoring + best run display |
| +Modes | US2 | Light/heavy computation modes |
| +Validation | US4 | Clear weights validation errors |
| +CLI | Phase 7 | Full CLI support |

### Task Count by Phase

| Phase | Tasks | Parallel |
|-------|-------|----------|
| Phase 1: Setup | 3 | 2 |
| Phase 2: Foundational | 14 | 7 |
| Phase 3: US1 | 10 | 3 |
| Phase 4: US3 | 9 | 3 |
| Phase 5: US2 | 8 | 3 |
| Phase 6: US4 | 6 | 3 |
| Phase 7: Polish | 15 | 10 |
| **Total** | **65** | **31** |

---

## Notes

- Test IDs reference contracts/testing.md (TC-001 to TC-027)
- All tasks include exact file paths per plan.md structure
- [P] tasks can run in parallel (different files, no deps)
- Commit after each task or logical group
- Verify tests fail before implementing (TDD)
- SC-005: Run existing tests frequently to confirm no regression
