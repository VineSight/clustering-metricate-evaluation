# Tasks: Labricate - Hyperparameter Experimentation Framework

**Input**: Design documents from `/specs/004-labricate/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: ✅ Included per user request - tests added between phases

**Organization**: Tasks grouped by user story for independent implementation and testing

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, etc.) - only for user story phases
- Exact file paths included in descriptions

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create directory structure and configure dependencies

- [x] T001 Create labricate submodule structure in metricate/labricate/
- [x] T002 Add dependencies (bertopic>=0.15, tqdm>=4.60) to pyproject.toml
- [x] T003 [P] Create metricate/labricate/__init__.py with module exports
- [x] T004 [P] Create metricate/labricate/core/__init__.py
- [x] T005 [P] Create metricate/labricate/pipelines/__init__.py
- [x] T006 [P] Create metricate/labricate/output/__init__.py
- [x] T007 [P] Create metricate/labricate/utils/__init__.py
- [x] T008 [P] Create tests/unit/labricate/__init__.py and conftest.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T009 Implement config validation with dot-notation path resolver in metricate/labricate/core/config.py
- [x] T010 [P] Define Pipeline protocol in metricate/labricate/pipelines/base.py
- [x] T011 [P] Implement embeddings loader (CSV/NPY/DataFrame) in metricate/labricate/core/loader.py
- [x] T012 [P] Implement timing/progress utilities in metricate/labricate/utils/logging.py
- [x] T013 Define result dataclasses (PipelineResult, RunResult, ExperimentResult) in metricate/labricate/core/experiment.py

### Tests: Foundational Infrastructure

- [x] T014 [P] Test config validation and dot-notation resolver in tests/unit/labricate/test_config.py
- [x] T015 [P] Test embeddings loader for all input formats in tests/unit/labricate/test_loader.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Single Hyperparameter Experiment (Priority: P1) 🎯 MVP

**Goal**: Run single-parameter experiments, evaluate with Metricate, save results

**Independent Test**: Provide embeddings, config, parameter range; verify all runs execute with correct values and results are saved

### Tests for User Story 1

> **Write tests FIRST, ensure they FAIL before implementation**

- [x] T016 [P] [US1] Test Experiment.run() executes correct number of runs in tests/unit/labricate/test_experiment.py
- [x] T017 [P] [US1] Test parameter values are correctly applied per run in tests/unit/labricate/test_experiment.py
- [x] T018 [P] [US1] Test Metricate integration returns expected metric results in tests/unit/labricate/test_experiment.py

### Implementation for User Story 1

- [x] T019 [US1] Implement Experiment class with __init__ in metricate/labricate/core/experiment.py
- [x] T020 [US1] Implement Experiment.run() for single-param experiments in metricate/labricate/core/experiment.py
- [x] T021 [US1] Implement Metricate evaluation integration in metricate/labricate/core/experiment.py
- [x] T022 [US1] Implement ExperimentSummary generation in metricate/labricate/core/experiment.py
- [x] T023 [US1] Add progress bar and verbose timing output in metricate/labricate/core/experiment.py

**Checkpoint**: User Story 1 functional - can run single-param experiments and get evaluated results

---

## Phase 4: User Story 2 - Default BERTopic Pipeline (Priority: P1) 🎯 MVP

**Goal**: Provide default UMAP → HDBSCAN/K-Means pipeline via BERTopic library

**Independent Test**: Run default pipeline with embeddings and config; verify valid cluster labels and reduced embeddings

### Tests for User Story 2

- [x] T024 [P] [US2] Test BERTopicPipeline with HDBSCAN config in tests/unit/labricate/test_pipelines.py
- [x] T025 [P] [US2] Test BERTopicPipeline with K-Means config in tests/unit/labricate/test_pipelines.py
- [x] T026 [P] [US2] Test pipeline applies all UMAP/HDBSCAN parameters in tests/unit/labricate/test_pipelines.py
- [x] T027 [P] [US2] Test pipeline respects random_seed for reproducibility in tests/unit/labricate/test_pipelines.py

### Implementation for User Story 2

- [x] T028 [US2] Implement BERTopicPipeline class wrapping BERTopic library in metricate/labricate/pipelines/bertopic.py
- [x] T029 [US2] Implement UMAP model configuration (n_neighbors, n_components, min_dist, metric, repulsion_strength, low_memory) in metricate/labricate/pipelines/bertopic.py
- [x] T030 [US2] Implement HDBSCAN model configuration (min_cluster_size, min_samples, cluster_selection_method, metric) in metricate/labricate/pipelines/bertopic.py
- [x] T031 [US2] Implement K-Means model configuration (n_clusters) in metricate/labricate/pipelines/bertopic.py
- [x] T032 [US2] Implement output extraction (topic_model.topics_, topic_model.umap_model.embedding_) in metricate/labricate/pipelines/bertopic.py
- [x] T033 [US2] Wire BERTopicPipeline as default in Experiment class in metricate/labricate/core/experiment.py

**Checkpoint**: User Stories 1 & 2 functional - can run full experiments with default pipeline

---

## Phase 5: User Story 3 - Multiple Input Formats (Priority: P2)

**Goal**: Accept embeddings as NumPy array, CSV file, or DataFrame

**Independent Test**: Provide embeddings in each format; verify all are accepted and produce identical results

### Tests for User Story 3

- [x] T034 [P] [US3] Test load_embeddings with NumPy array input in tests/unit/labricate/test_loader.py
- [x] T035 [P] [US3] Test load_embeddings with CSV file path in tests/unit/labricate/test_loader.py
- [x] T036 [P] [US3] Test load_embeddings with pandas DataFrame in tests/unit/labricate/test_loader.py
- [x] T037 [P] [US3] Test identical results across all input formats in tests/unit/labricate/test_loader.py

### Implementation for User Story 3

- [x] T038 [US3] Extend loader.py to detect input type and dispatch correctly in metricate/labricate/core/loader.py
- [x] T039 [US3] Implement CSV column detection (dim_* or numeric) in metricate/labricate/core/loader.py
- [x] T040 [US3] Add validation for embedding shape consistency in metricate/labricate/core/loader.py

**Checkpoint**: User Story 3 functional - all input formats accepted ✅

---

## Phase 6: User Story 4 - Custom Pipeline Integration (Priority: P2)

**Goal**: Allow users to provide custom pipeline functions

**Independent Test**: Provide custom pipeline function; verify Labricate calls it correctly and processes outputs

### Tests for User Story 4

- [x] T041 [P] [US4] Test Experiment accepts custom pipeline function in tests/unit/labricate/test_experiment.py
- [x] T042 [P] [US4] Test custom pipeline output validation in tests/unit/labricate/test_experiment.py
- [x] T043 [P] [US4] Test error handling for invalid custom pipeline outputs in tests/unit/labricate/test_experiment.py

### Implementation for User Story 4

- [x] T044 [US4] Implement custom pipeline acceptance in Experiment.__init__() in metricate/labricate/core/experiment.py
- [x] T045 [US4] Implement output shape validation (labels 1D, embeddings 2D) in metricate/labricate/core/experiment.py
- [x] T046 [US4] Add error messages for custom pipeline failures in metricate/labricate/core/experiment.py

**Checkpoint**: User Story 4 functional - custom pipelines supported ✅

---

## Phase 7: User Story 6 - Output and Visualization (Priority: P2)

**Goal**: Save results (JSON/CSV) and generate visualization charts

**Independent Test**: Complete an experiment; verify output files are created and charts are generated

### Tests for User Story 6

- [x] T047 [P] [US6] Test JSON output format in tests/unit/labricate/test_storage.py
- [x] T048 [P] [US6] Test CSV output format in tests/unit/labricate/test_storage.py
- [x] T049 [P] [US6] Test hierarchical directory structure creation in tests/unit/labricate/test_storage.py
- [x] T050 [P] [US6] Test line chart generation in tests/unit/labricate/test_visualization.py

### Implementation for User Story 6

- [x] T051 [US6] Implement JSON result storage in metricate/labricate/output/storage.py
- [x] T052 [US6] Implement CSV result storage in metricate/labricate/output/storage.py
- [x] T053 [US6] Implement hierarchical directory structure (experiments/<name>/<timestamp>/) in metricate/labricate/output/storage.py
- [x] T054 [US6] Implement intermediate clustering CSV output per run in metricate/labricate/output/storage.py
- [x] T055 [US6] Implement line chart visualization (metric vs parameter) in metricate/labricate/output/visualization.py
- [x] T056 [US6] Wire storage and visualization into Experiment completion in metricate/labricate/core/experiment.py

**Checkpoint**: User Story 6 functional - results saved and visualized ✅

---

## Phase 8: User Story 5 - Grid Search Experiment (Priority: P3)

**Goal**: Run experiments varying multiple parameters simultaneously

**Independent Test**: Specify two parameters with multiple values; verify all combinations tested

### Tests for User Story 5

- [x] T057 [P] [US5] Test Experiment.run_grid() executes all combinations in tests/unit/labricate/test_experiment.py
- [x] T058 [P] [US5] Test grid search with 2 parameters produces correct run count in tests/unit/labricate/test_experiment.py
- [x] T059 [P] [US5] Test heatmap generation for 2-param grids in tests/unit/labricate/test_visualization.py

### Implementation for User Story 5

- [x] T060 [US5] Implement Experiment.run_grid() method in metricate/labricate/core/experiment.py
- [x] T061 [US5] Implement parameter combination generation (itertools.product) in metricate/labricate/core/experiment.py
- [x] T062 [US5] Implement heatmap visualization for 2-param grids in metricate/labricate/output/visualization.py
- [x] T063 [US5] Implement tabular-only output for 3+ param grids in metricate/labricate/output/visualization.py

**Checkpoint**: User Story 5 functional - grid search works ✅

---

## Phase 9: Advanced Features (P2 Supporting)

**Goal**: Parallelism, checkpoint/resume, error handling

### Tests: Advanced Features

- [x] T064 [P] Test parallel execution with ThreadPoolExecutor in tests/unit/labricate/test_parallel.py
- [x] T065 [P] Test worker count capping at CPU count in tests/unit/labricate/test_parallel.py
- [x] T066 [P] Test checkpoint save after each run in tests/unit/labricate/test_checkpoint.py
- [x] T067 [P] Test resume from checkpoint skips completed runs in tests/unit/labricate/test_checkpoint.py
- [x] T068 [P] Test config mismatch detection on resume in tests/unit/labricate/test_checkpoint.py
- [x] T069 [P] Test error_handling="continue" logs and continues in tests/unit/labricate/test_experiment.py
- [x] T070 [P] Test error_handling="fail_fast" stops immediately in tests/unit/labricate/test_experiment.py

### Implementation: Advanced Features

- [x] T071 Implement ThreadPoolExecutor wrapper in metricate/labricate/utils/parallel.py
- [x] T072 Implement worker count capping with warning in metricate/labricate/utils/parallel.py
- [x] T073 Implement checkpoint save/load in metricate/labricate/core/checkpoint.py
- [x] T074 Implement config hash for mismatch detection in metricate/labricate/core/checkpoint.py
- [x] T075 Implement resume logic with --force flag in metricate/labricate/core/experiment.py
- [x] T076 Implement error_handling modes in Experiment.run() in metricate/labricate/core/experiment.py
- [x] T077 Implement ground truth handling for supervised metrics in metricate/labricate/core/experiment.py
- [x] T078 Implement metric filtering (include/exclude) in metricate/labricate/core/experiment.py

**Checkpoint**: Advanced features functional - parallelism, resume, error handling work ✅

---

## Phase 10: User Story 7 - CLI Interface (Priority: P3)

**Goal**: Provide command-line interface for experiments

**Independent Test**: Run experiment via CLI; verify same results as Python API

### Tests for User Story 7

- [x] T079 [P] [US7] Test CLI argument parsing in tests/unit/labricate/test_cli.py
- [x] T080 [P] [US7] Test CLI --help output in tests/unit/labricate/test_cli.py
- [x] T081 [P] [US7] Test CLI produces same results as Python API in tests/unit/labricate/test_cli.py

### Implementation for User Story 7

- [x] T082 [US7] Create metricate labricate command group in metricate/cli/labricate.py
- [x] T083 [US7] Implement labricate experiment subcommand in metricate/cli/labricate.py
- [x] T084 [US7] Implement labricate resume subcommand in metricate/cli/labricate.py
- [x] T085 [US7] Implement labricate validate subcommand in metricate/cli/labricate.py
- [x] T086 [US7] Implement --grid mode for multi-param experiments in metricate/cli/labricate.py
- [x] T087 [US7] Register labricate group in metricate/cli/main.py

**Checkpoint**: User Story 7 functional - CLI works ✅

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, documentation, validation

- [X] T088 [P] Update metricate/__init__.py to re-export labricate module
- [X] T089 [P] Add type hints throughout labricate module
- [X] T090 [P] Add docstrings per contracts/python-api.md
- [X] T091 Run quickstart.md validation (all examples work)
- [X] T092 [P] Update README.md with labricate usage
- [X] T093 Final integration test: full experiment with visualization

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational) → [All User Stories can proceed in parallel]
                                         → Phase 11 (Polish) after desired stories complete
```

### User Story Dependencies

| Story | Priority | Dependencies | Can Parallelize With |
|-------|----------|--------------|----------------------|
| US1 (Single-param) | P1 MVP | Phase 2 only | US2 |
| US2 (BERTopic) | P1 MVP | Phase 2 only | US1 |
| US3 (Input formats) | P2 | Phase 2 only | US4, US6 |
| US4 (Custom pipeline) | P2 | US1 | US3, US6 |
| US6 (Output/Viz) | P2 | US1 | US3, US4 |
| US5 (Grid search) | P3 | US1, US6 | US7 |
| US7 (CLI) | P3 | US1 | US5 |
| Phase 9 (Advanced) | P2 | US1 | US3-US7 |

### Parallel Opportunities per Phase

**Phase 1**: T003-T008 (all module __init__.py files)
**Phase 2**: T010-T012 (protocol, loader, logging); T014-T015 (tests)
**Phase 3**: T016-T018 (US1 tests)
**Phase 4**: T024-T027 (US2 tests)
**Phase 5**: T034-T037 (US3 tests)
**Phase 6**: T041-T043 (US4 tests)
**Phase 7**: T047-T050 (US6 tests)
**Phase 8**: T057-T059 (US5 tests)
**Phase 9**: T064-T070 (advanced feature tests)
**Phase 10**: T079-T081 (US7 tests)
**Phase 11**: T088-T090, T092 (polish tasks)

---

## Implementation Strategy

### MVP Scope (Recommended First Delivery)

**Phases 1-4**: Setup + Foundational + US1 + US2 = **33 tasks**

This delivers:
- Working single-param experiments
- Default BERTopic pipeline
- Metricate evaluation
- Basic results output

### Incremental Additions

1. **+Input Flexibility**: Phase 5 (US3) - 7 tasks
2. **+Custom Pipelines**: Phase 6 (US4) - 6 tasks
3. **+Output/Viz**: Phase 7 (US6) - 10 tasks
4. **+Grid Search**: Phase 8 (US5) - 7 tasks
5. **+Advanced Features**: Phase 9 - 15 tasks
6. **+CLI**: Phase 10 (US7) - 9 tasks
7. **+Polish**: Phase 11 - 6 tasks

---

## Summary

| Phase | Purpose | Tasks | Tests |
|-------|---------|-------|-------|
| 1 | Setup | 8 | 0 |
| 2 | Foundational | 5 | 2 |
| 3 | US1 Single-param (P1) | 5 | 3 |
| 4 | US2 BERTopic (P1) | 6 | 4 |
| 5 | US3 Input formats (P2) | 3 | 4 |
| 6 | US4 Custom pipeline (P2) | 3 | 3 |
| 7 | US6 Output/Viz (P2) | 6 | 4 |
| 8 | US5 Grid search (P3) | 4 | 3 |
| 9 | Advanced features (P2) | 8 | 7 |
| 10 | US7 CLI (P3) | 6 | 3 |
| 11 | Polish | 6 | 0 |
| **Total** | | **60** | **33** |

**Grand Total**: 93 tasks (60 implementation + 33 tests)
