# Feature Specification: Labricate Weighted Evaluation & Computation Modes

**Feature Branch**: `005-labricate-weighted-eval`  
**Created**: March 26, 2026  
**Status**: Draft  
**Input**: User description: "Add weighted evaluation support and computation modes to Labricate experiments"

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Weighted Experiment Evaluation (Priority: P1)

As a data scientist running clustering experiments, I want to provide a JSON weights file so that experiment results include compound scores and the best run is clearly identified based on my trained quality model.

**Why this priority**: Weighted evaluation is the core feature that transforms raw metrics into actionable decisions. Without it, users must manually interpret 36 metrics to find the best configuration. This delivers immediate value by automating "which config is best" based on learned weights.

**Independent Test**: Can be fully tested by running an experiment with a weights JSON file and verifying the results include compound scores and a `best_run` field. Delivers clear "winner" identification.

**Acceptance Scenarios**:

1. **Given** an Experiment with a valid weights JSON path, **When** I call `exp.run()`, **Then** each run's results include a `compound_score` computed using the weights
2. **Given** an experiment result with weights, **When** I access `result.best_run`, **Then** it returns the RunResult with the highest compound score
3. **Given** a weights JSON file, **When** I run via CLI with `--weights path/to/weights.json`, **Then** output includes compound scores and best run identification
4. **Given** invalid weights JSON (missing coefficients), **When** I provide it to Experiment, **Then** a clear validation error is raised before the experiment starts
5. **Given** an experiment completes with weights, **When** `verbose=True`, **Then** the best run configuration is printed to console

---

### User Story 2 - Computation Mode Selection (Priority: P2)

As a user iterating quickly on hyperparameters, I want to choose between "light" (fast) and "heavy" (comprehensive) computation modes so that I can balance speed vs. completeness based on my workflow stage.

**Why this priority**: Computation time is a major friction point. Users often want quick iteration (light mode) during exploration, then comprehensive evaluation (heavy mode) for final decisions. This enables 10x faster experiments during exploration.

**Independent Test**: Can be tested by running identical experiments with `mode="light"` vs `mode="heavy"` and verifying light mode excludes expensive metrics and runs faster.

**Acceptance Scenarios**:

1. **Given** `mode="light"` in Experiment.run(), **When** the experiment executes, **Then** O(n²) expensive metrics (Gamma, Tau, G-plus, Point-Biserial, McClain-Rao, NIVA) are automatically excluded
2. **Given** `mode="heavy"` in Experiment.run(), **When** the experiment executes, **Then** all metrics are computed including expensive O(n²) metrics
3. **Given** no mode specified, **When** experiment runs, **Then** default mode is "heavy" for comprehensive evaluation
4. **Given** `mode="light"` with explicit `include_metrics`, **When** experiment runs, **Then** `include_metrics` takes precedence over mode defaults
5. **Given** CLI experiment command, **When** I pass `--mode light` or `--mode heavy`, **Then** appropriate metrics are included/excluded

---

### User Story 3 - Best Run Display in Results (Priority: P1)

As a user reviewing experiment results, I want the best run to be prominently displayed and accessible so that I can immediately see the optimal configuration without manual analysis.

**Why this priority**: The "best config" is the primary output users want. Making it prominent reduces cognitive load and accelerates decision-making.

**Independent Test**: Can be tested by running any experiment and verifying `result.best_run` is populated and `result.summary` includes best run info.

**Acceptance Scenarios**:

1. **Given** a completed experiment, **When** I access `result.best_run`, **Then** it returns the RunResult with best metric value (highest Silhouette by default, or highest compound_score if weights provided)
2. **Given** experiment results with weights, **When** printed via `print(result.summary)`, **Then** best run config and compound score are displayed
3. **Given** experiment DataFrame via `result.to_dataframe()`, **When** I examine it, **Then** there's a `is_best_run` boolean column marking the best row
4. **Given** results saved to JSON, **When** I load the file, **Then** `best_run` is a top-level field with run_id and param_values
5. **Given** multiple metrics requested, **When** finding best run without weights, **Then** user can specify `best_metric` parameter (default: "Silhouette")

---

### User Story 4 - Weights JSON Validation (Priority: P2)

As a user providing a weights file, I want clear validation errors so that I can quickly fix any issues with my weights JSON before wasting time on a failed experiment.

**Why this priority**: Poor error messages waste user time. Upfront validation with clear messages prevents confusion and debugging.

**Independent Test**: Can be tested by providing various malformed weights files and verifying clear, actionable error messages.

**Acceptance Scenarios**:

1. **Given** weights JSON missing required `coefficients` field, **When** provided to Experiment, **Then** error message says "weights JSON missing required field 'coefficients'"
2. **Given** weights JSON with non-numeric coefficient values, **When** provided to Experiment, **Then** error message identifies the invalid field
3. **Given** weights JSON with coefficient keys not ending in `_norm`, **When** provided to Experiment, **Then** error message explains the naming convention
4. **Given** a valid weights JSON, **When** provided to Experiment, **Then** no error is raised and weights are loaded successfully

---

### Edge Cases

- What happens when weights JSON references metrics that weren't computed? → Compute compound score using only available metrics, log warning for missing ones
- What happens when all runs fail? → `best_run` is None, warning message displayed
- What happens with ties in compound score? → Return first run with highest score AND explicitly mention the tie in output (e.g., "Best run: run_id=3 (tied with run_id=7)")
- What happens with mode="light" but user explicitly adds Gamma to include_metrics? → User's explicit include takes precedence
- What happens when user excludes metrics that are dominant in the weights? → Display warning that compound scores may be skewed/less informative (e.g., "Warning: Excluded metrics ['Silhouette', 'Davies-Bouldin'] account for 65% of weight coefficients. Compound scores may be unreliable.")

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a `weights` parameter in Experiment constructor as path string or dict
- **FR-002**: System MUST validate weights JSON against schema before experiment starts
- **FR-003**: System MUST compute `compound_score` for each run when weights are provided
- **FR-004**: System MUST populate `best_run` field in ExperimentResult with highest-scoring run
- **FR-005**: System MUST support `mode` parameter with values "light" or "heavy"
- **FR-006**: Light mode MUST exclude expensive metrics: Gamma, Tau, G-plus, Point-Biserial, McClain-Rao, NIVA
- **FR-007**: System MUST display best run configuration when `verbose=True` and experiment completes
- **FR-008**: ExperimentResult.to_dataframe() MUST include `compound_score` column when weights used
- **FR-009**: ExperimentResult.to_dataframe() MUST include `is_best_run` boolean column
- **FR-010**: CLI MUST support `--weights` option for experiment command
- **FR-011**: CLI MUST support `--mode` option with choices "light" and "heavy"
- **FR-012**: JSON output MUST include `best_run` top-level field when experiment completes
- **FR-013**: System MUST use existing metricate `METRIC_REFERENCE` to determine which metrics have `skip_large: True` for light mode
- **FR-014**: When weights provided, `get_best_run()` MUST use compound_score by default instead of single metric
- **FR-015**: `best_run` MUST include the best hyperparameter values (param_values dict) that achieved the highest compound score
- **FR-016**: When multiple runs have identical best compound scores (tie), system MUST report all tied run IDs in the output
- **FR-017**: System MUST warn users when excluded metrics account for significant weight in the provided weights file (threshold: >30% of total absolute coefficient weight)

### Key Entities

- **Weights JSON**: User-provided file with coefficients for computing compound scores. Must follow metricate's weights schema with `coefficients` (metric_name_norm → float), `bias` (float), and optional metadata.
- **Computation Mode**: String enum ("light" or "heavy") controlling which metrics to compute. Light excludes expensive O(n²) metrics marked with `skip_large: True` in metricate reference.
- **Best Run**: The RunResult with optimal score. Determined by compound_score when weights provided, or specified metric (default Silhouette) otherwise. Contains the best hyperparameter values (param_values) that achieved the highest score. If ties exist, all tied run IDs are reported.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can identify the best configuration from experiment results in under 5 seconds (vs. manual DataFrame analysis taking 30+ seconds). *Measured as: time from `result` object access to reading `best_run.param_values`.*
- **SC-002**: Light mode experiments complete at least 30% faster than heavy mode on datasets with 5,000+ points. *Measured as: wall-clock time on reference dataset (5000 samples, 50 dimensions) with identical hyperparameter grid.*
- **SC-003**: 100% of weight validation errors include actionable guidance (field name, expected format)
- **SC-004**: Best run is correctly identified in all test cases (highest compound_score when weights used, highest Silhouette otherwise)
- **SC-005**: Existing Labricate functionality (run, run_grid, resume) continues working unchanged when weights/mode not specified

---

## Assumptions

- Users already have trained weights from metricate's `train_weights()` or manually crafted weights JSON
- The expensive metrics (Gamma, Tau, etc.) from metricate's `skip_large: True` reference are the correct set for light mode exclusion
- Silhouette is a reasonable default metric for best-run determination when no weights provided
- Existing metricate `MetricWeights` class and `load_weights()` function should be reused for weights handling

---

## Clarifications

### Session 2026-03-26

- Q: What should happen when user excludes metrics that are dominant in the weights? → A: Display warning that compound scores may be skewed (threshold: >30% of total weight)
- Q: What should best_run return? → A: The best hyperparameter values (param_values dict) with highest compound score
- Q: How should ties in compound score be handled? → A: Mention all tied run IDs in output, not silent first-pick
