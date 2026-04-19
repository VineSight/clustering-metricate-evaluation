# Feature Specification: Parallel Workers for Labricate Experiments

**Feature Branch**: `006-parallel-workers`
**Created**: 2026-04-16
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fast Single-Parameter Experiment (Priority: P1)

A researcher has a single-parameter scan with many values to test (e.g., 8+ values of `min_cluster_size`). They set `n_workers=4` so that 4 values are evaluated simultaneously, reducing total wall-clock time. The results come back in the same order as the input values, identical to what sequential execution would produce.

**Why this priority**: Single-parameter experiments are the most common use of Labricate. Researchers regularly scan 8–20 values and currently wait for each to finish before the next begins. This is the highest-frequency pain point.

**Independent Test**: Run an 8-value single-parameter experiment with `n_workers=4` on any dataset. Measure total time and compare to `n_workers=1`. Verify results are ordered by input value sequence.

**Acceptance Scenarios**:

1. **Given** an experiment with 8 values and `n_workers=4`, **When** `run()` is called, **Then** runs execute concurrently and total time is ≤60% of sequential time on the same machine.
2. **Given** an experiment with `n_workers=1` (default), **When** `run()` is called, **Then** behavior is identical to the current implementation with no performance regression.
3. **Given** results from a parallel run and a sequential run with identical inputs, **When** compared, **Then** metric values and run IDs are identical in the same order.

---

### User Story 2 - Fast Grid Search Experiment (Priority: P2)

A researcher runs a grid search over multiple parameters (e.g., 4 values × 2 values = 8 combinations). They set `n_workers=4` so that 4 combinations run simultaneously, cutting multi-hour experiments down to minutes.

**Why this priority**: Grid searches multiply the number of runs combinatorially. Even small grids (8–64 combinations) take significantly longer than single-parameter scans. Parallelism has the highest absolute time savings here.

**Independent Test**: Run an 8-combination grid search with `n_workers=4`. Measure speedup vs sequential. Confirm result order matches combination order from `itertools.product`.

**Acceptance Scenarios**:

1. **Given** a grid search with 8 combinations and `n_workers=4`, **When** `run_grid()` is called, **Then** combinations execute concurrently and total time is ≤60% of sequential time.
2. **Given** `n_workers=1` on a grid search, **When** run, **Then** output is identical to current sequential grid search behavior.
3. **Given** a parallel grid search result, **When** inspected, **Then** run IDs and param values follow the same ordering as sequential `itertools.product` enumeration.

---

### User Story 3 - Graceful Failure Handling in Parallel Runs (Priority: P3)

A researcher's parallel experiment encounters a failure in one run (e.g., an invalid hyperparameter combination causes a pipeline error). The system saves results for all successful runs and clearly marks the failed ones, so the researcher doesn't lose hours of completed work.

**Why this priority**: Without this, any single failure would discard all parallel work. This is a correctness requirement for the feature to be safe to use in practice.

**Independent Test**: Configure a run where one specific parameter value is known to fail. Run with `n_workers=4` and `error_handling="continue"`. Verify the remaining runs complete and their results appear in the output.

**Acceptance Scenarios**:

1. **Given** a parallel experiment where one run fails and `error_handling="continue"`, **When** the experiment completes, **Then** all other runs' results are saved and the failed run is recorded with an error status.
2. **Given** a parallel experiment where one run fails and `error_handling="fail_fast"`, **When** the failure occurs, **Then** execution stops, the error is surfaced to the caller, and any already-completed run results are preserved.
3. **Given** a completed experiment with some failures, **When** the user inspects the output, **Then** failed runs are distinguishable from successful ones (e.g., status field or absence of metrics).

---

### Edge Cases

- What happens when `n_workers` exceeds the number of available CPUs? → Silently capped to CPU count, warning emitted.
- What happens when `n_workers=0` or negative? → Treated as `n_workers=1` (sequential) with a warning.
- What happens when all workers fail? → An exception is raised (same as current fail behavior) and no successful results are saved.
- What happens when `n_workers=1`? → Exactly equivalent to current sequential execution — no concurrency overhead introduced.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST execute experiment runs concurrently when `n_workers > 1` in both `run()` (single-parameter) and `run_grid()` (multi-parameter grid).
- **FR-002**: System MUST default to sequential execution when `n_workers` is not specified.
- **FR-003**: System MUST cap `n_workers` to the number of available CPU cores and emit a warning when the requested value exceeds capacity.
- **FR-004**: System MUST treat `n_workers ≤ 0` as `n_workers=1` and continue execution.
- **FR-005**: System MUST preserve all successful run results when one or more runs fail and `error_handling="continue"`.
- **FR-006**: System MUST stop pending work and surface the first error when a run fails and `error_handling="fail_fast"`.
- **FR-007**: System MUST return results in the same order as the input values or combinations, regardless of which runs finish first.
- **FR-008**: System MUST mark failed runs with an error status distinguishable from successful runs in the output.

### Key Entities

- **Experiment**: Orchestrates runs; accepts `n_workers` and `error_handling` to control parallel behavior.
- **Run**: A single evaluation of one hyperparameter value or combination; the atomic unit of parallelization.
- **RunResult**: The output of one run; carries success/failure status and, on success, all computed metrics.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Parallel execution with 4 workers reduces total experiment time by at least 40% compared to sequential execution on experiments with 8 or more runs.
- **SC-002**: Results produced by parallel execution are identical to results produced by sequential execution for the same inputs and random seeds.
- **SC-003**: When one run fails in a 10-run parallel experiment with `error_handling="continue"`, the remaining 9 results are saved and accessible.
- **SC-004**: Experiments with `n_workers=1` complete with no measurable performance regression compared to the pre-change implementation.
- **SC-005**: An over-subscription warning is emitted within 1 second of experiment start when `n_workers` exceeds available cores.
- **SC-006**: Results are always returned in input order, regardless of which worker finishes first.

## Assumptions

- Thread-based concurrency (not process-based) is the correct model for this feature, because experiment pipelines involve complex in-memory objects that are not safe to serialize across processes.
- The unit of parallelization is a complete run (pipeline execution + metric evaluation), not a sub-step within a run.
- Result ordering is by input position (run ID), not by completion time.
- The existing `ParallelExecutor` class in `metricate/labricate/utils/parallel.py` is the correct primitive — it already handles worker capping, error modes, and ordering.
- Checkpoint behavior is unchanged: parallel runs do not alter the existing per-run checkpoint format.
