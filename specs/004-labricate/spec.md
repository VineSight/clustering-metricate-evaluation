# Feature Specification: Labricate - Hyperparameter Experimentation Framework

**Feature Branch**: `004-labricate`  
**Created**: March 18, 2026  
**Status**: Draft  
**Input**: User description: "Framework for running clustering pipeline experiments with varying hyperparameters, evaluating results with Metricate, and comparing outcomes"

## Clarifications

### Session 2026-03-18

- Q: Should pipeline output just labels or labels + embeddings? → A: Both cluster labels AND reduced embeddings
- Q: How to determine "optimal" hyperparameter value? → A: No automatic determination; present comprehensive results for user interpretation
- Q: What hyperparameters should BERTopic pipeline support? → A: UMAP (n_neighbors, n_components, min_dist, metric, repulsion_strength), HDBSCAN (min_cluster_size, min_samples, cluster_selection_method), K-Means (n_clusters)
- Q: Should pipeline support HDBSCAN, K-Means, or both? → A: User chooses one via config
- Q: Single hyperparameter experiments or grid search? → A: Both - single hyperparameter as primary, grid search as advanced option
- Q: Output format and storage? → A: Both JSON and CSV (user selects), save intermediate clustering outputs, hierarchical directory structure
- Q: Custom pipeline function signature? → A: `(embeddings: np.ndarray, config: dict) -> tuple[labels, reduced_embeddings]`
- Q: How to integrate with Metricate? → A: Direct import (`metricate.evaluate()`)
- Q: What interfaces to provide? → A: Python API first, then CLI; no Web UI for now
- Q: What embedding input formats? → A: All (NumPy array, CSV file, DataFrame) with internal conversion
- Q: Visualization output? → A: Basic line charts showing metric values vs hyperparameter values
- Q: Parallel execution? → A: Optional with configurable workers (default=1)
- Q: Progress feedback level? → A: Verbose logging with progress bar and step-by-step timing
- Q: Error handling for partial failures? → A: Configurable (default: continue & report failures at end)
- Q: Module location? → A: `metricate/labricate/` submodule (revised from sibling folder to comply with single-package constitution)

### Session 2026-03-18 (Clarification)

- Q: How should users specify which hyperparameter to vary? → A: Dot notation path (e.g., `"umap.n_neighbors"`, `"hdbscan.min_cluster_size"`)
- Q: What should happen when a user specifies an invalid parameter path? → A: Fail fast with clear error message listing invalid paths before running any experiments
- Q: Should users be able to filter which Metricate metrics are calculated? → A: Yes, optional filtering (include/exclude specific metrics); default calculates all 34 metrics
- Q: Should experiments be resumable if interrupted? → A: Yes, checkpoint-based resume; save progress after each run and allow resuming from last checkpoint
- Q: Should Labricate enforce reproducibility via random seeds? → A: Optional seed with default value (e.g., 42); reproducible by default, users can set to null for stochastic behavior

### Session 2026-03-18 (BERTopic Parity)

- Q: Should we add `umap.low_memory` and `hdbscan.metric` for full BERTopic parity? → A: Yes, add both parameters (`umap.low_memory`: bool, `hdbscan.metric`: str)
- Q: How should grid search (3+ params) visualizations be handled? → A: No heatmaps for 3+ params; show tabular results only (heatmaps for 2-param grids)
- Q: How should ground truth labels be used when provided? → A: If ground truth provided, automatically enable supervised metrics (ARI, NMI, etc.) alongside unsupervised; default expectation is no ground truth (unsupervised only)
- Q: How should n_workers be handled if user requests more than CPU count? → A: Cap at CPU count and emit warning if user requested more
- Q: What happens if user tries to resume with a modified config? → A: Detect mismatch, warn, require `--force` flag to override (starts fresh)

### Session 2026-03-22 (BERTopic Library)

- Q: Should we use the BERTopic library instead of direct UMAP + HDBSCAN? → A: Yes, use BERTopic library as the default pipeline (leverage its modularity, accept extra dependencies)
- Q: How to extract clustering outputs from BERTopic for Metricate? → A: Access BERTopic internals: `topic_model.umap_model.embedding_` for reduced embeddings, `topic_model.topics_` for labels
- Q: Should BERTopic run full topic representation (c-TF-IDF)? → A: Configurable; skip by default for speed, option to enable for users who want topic words
- Q: How to handle BERTopic's document requirement with pre-computed embeddings? → A: Pass pre-computed embeddings + placeholder empty docs (`fit_transform([""] * n, embeddings=embeddings)`)
- Q: What BERTopic install variant to use? → A: Minimal install (`pip install bertopic`), no embedding backends; users provide pre-computed embeddings

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Hyperparameter Experiment (Priority: P1)

A data scientist wants to determine the best `min_cluster_size` for their clustering pipeline. They provide embeddings and a base configuration, specify that `hdbscan.min_cluster_size` should vary from 5 to 60 in steps of 5 using dot notation to pinpoint the parameter, and Labricate runs 12 pipeline executions with all other parameters held constant. Each run's clustering is evaluated by Metricate, and results are presented in a comparison table with visualizations.

**Why this priority**: This is the core functionality - the primary use case for hyperparameter experimentation. Without single-parameter experiments, the framework has no value.

**Independent Test**: Provide embeddings, config, and parameter range; verify all runs execute with correct parameter values and results are saved.

**Acceptance Scenarios**:

1. **Given** a user provides embeddings, a base config, and specifies parameter `"hdbscan.min_cluster_size"` with values `[5, 10, 15, 20, 25, 30]`, **When** they run the experiment, **Then** 6 pipeline runs execute with the specified values while other parameters remain constant.

2. **Given** an experiment completes, **When** results are saved, **Then** a hierarchical directory structure is created with: experiment config, intermediate clustering CSVs for each run, Metricate evaluation results for each run, and a combined comparison file.

3. **Given** an experiment completes, **When** the user views results, **Then** they see a comparison table showing all metrics for each hyperparameter value, allowing them to interpret which value performs best.

4. **Given** an experiment is running, **When** the user observes progress, **Then** they see a progress bar indicating current run (e.g., "Run 3/6: hdbscan.min_cluster_size=15") and verbose timing logs for each step (UMAP, clustering, evaluation).

---

### User Story 2 - Default BERTopic Pipeline (Priority: P1)

A researcher wants to experiment with clustering but doesn't have their own pipeline implementation. They use Labricate's default BERTopic pipeline which handles UMAP dimensionality reduction followed by either HDBSCAN or K-Means clustering. The pipeline accepts a comprehensive configuration and outputs cluster labels with reduced embeddings.

**Why this priority**: The default pipeline is essential for users to start experimenting immediately without building their own pipeline. Core functionality.

**Independent Test**: Run the default pipeline with embeddings and config; verify it produces valid cluster labels and reduced embeddings.

**Acceptance Scenarios**:

1. **Given** a user provides embeddings and a BERTopic config with `clustering_algorithm: "hdbscan"`, **When** the pipeline runs, **Then** UMAP reduction is applied followed by HDBSCAN clustering, outputting labels and reduced embeddings.

2. **Given** a user provides embeddings and a BERTopic config with `clustering_algorithm: "kmeans"` and `n_clusters: 10`, **When** the pipeline runs, **Then** UMAP reduction is applied followed by K-Means clustering with 10 clusters.

3. **Given** a BERTopic config specifies UMAP parameters (`n_neighbors: 15`, `n_components: 5`, `min_dist: 0.1`, `repulsion_strength: 1.0`), **When** the pipeline runs, **Then** UMAP uses exactly those parameters.

4. **Given** a BERTopic config specifies HDBSCAN parameters (`min_cluster_size: 10`, `min_samples: 5`, `cluster_selection_method: "eom"`), **When** the pipeline runs, **Then** HDBSCAN uses exactly those parameters.

---

### User Story 3 - Multiple Input Formats (Priority: P2)

A data scientist has embeddings in different formats depending on their workflow. They may have a NumPy array from a previous computation, a CSV file from a data export, or a pandas DataFrame from their analysis pipeline. Labricate accepts all these formats and converts them internally.

**Why this priority**: Flexibility in input formats reduces friction and makes the framework accessible to different workflows. Extends core functionality.

**Independent Test**: Provide embeddings in each supported format; verify all are accepted and produce identical results.

**Acceptance Scenarios**:

1. **Given** a user provides embeddings as a NumPy array of shape `(n_samples, n_dims)`, **When** they create an experiment, **Then** the embeddings are accepted without conversion.

2. **Given** a user provides a CSV file path with embedding columns (e.g., `dim_0`, `dim_1`, ...), **When** they create an experiment, **Then** the file is loaded and embeddings are extracted.

3. **Given** a user provides a pandas DataFrame with embedding columns, **When** they create an experiment, **Then** embeddings are extracted from the DataFrame.

4. **Given** embeddings in any format, **When** processed by the same pipeline and config, **Then** results are identical regardless of input format.

---

### User Story 4 - Custom Pipeline Integration (Priority: P2)

An advanced user has their own clustering pipeline with custom preprocessing or a different algorithm. They provide a pipeline function that conforms to Labricate's interface, and Labricate uses it instead of the default BERTopic pipeline for all experiment runs.

**Why this priority**: Extensibility for advanced users. Allows Labricate to be used with any clustering approach, not just BERTopic.

**Independent Test**: Provide a custom pipeline function; verify Labricate calls it correctly and processes its outputs.

**Acceptance Scenarios**:

1. **Given** a user provides a custom pipeline function with signature `(embeddings: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]`, **When** an experiment runs, **Then** Labricate calls the custom function instead of the default pipeline.

2. **Given** a custom pipeline returns `(labels, reduced_embeddings)`, **When** Labricate processes the output, **Then** it correctly pairs labels with reduced embeddings for Metricate evaluation.

3. **Given** a custom pipeline raises an exception, **When** error handling is set to "continue", **Then** Labricate logs the error, skips that run, and continues with remaining runs.

---

### User Story 5 - Grid Search Experiment (Priority: P3)

A researcher wants to explore the interaction between two hyperparameters: `n_neighbors` and `min_cluster_size`. They specify value ranges for both, and Labricate runs all combinations (grid search), evaluates each, and presents a comprehensive comparison including heatmap-style results.

**Why this priority**: Advanced functionality that extends single-parameter experiments. Useful for understanding parameter interactions but more complex.

**Independent Test**: Specify two parameters with multiple values; verify all combinations are tested and results show the full grid.

**Acceptance Scenarios**:

1. **Given** a user specifies `n_neighbors: [10, 15, 20]` and `min_cluster_size: [5, 10]`, **When** running a grid search experiment, **Then** 6 runs execute covering all combinations.

2. **Given** a grid search completes, **When** viewing results, **Then** the user sees a matrix/table showing metric values for each parameter combination.

3. **Given** a grid search with 3 parameters of 4 values each (64 combinations), **When** parallelism is enabled with 4 workers, **Then** runs execute in parallel batches, reducing total experiment time.

---

### User Story 6 - Experiment Output and Visualization (Priority: P2)

After an experiment completes, a data scientist wants to analyze results. They access structured output files (JSON/CSV) and visualization charts showing how each metric varies with the hyperparameter values.

**Why this priority**: Output and visualization are essential for interpreting experiment results. Core to the user's decision-making process.

**Independent Test**: Complete an experiment; verify output files are created correctly and charts are generated.

**Acceptance Scenarios**:

1. **Given** an experiment completes with output format "json", **When** saving results, **Then** a JSON file is created containing: experiment config, all run configs, and all metric results.

2. **Given** an experiment completes with output format "csv", **When** saving results, **Then** a CSV file is created with rows for each run and columns for config values and metric scores.

3. **Given** an experiment completes, **When** visualizations are generated, **Then** line charts are created showing each metric's value plotted against the varying hyperparameter.

4. **Given** a 2-parameter grid search experiment completes, **When** visualizations are generated, **Then** heatmaps show metric values across the parameter grid. For 3+ parameters, only tabular results are provided (no heatmaps).

---

### User Story 7 - CLI Interface (Priority: P3)

A data scientist wants to run experiments from the command line without writing Python code. They use Labricate's CLI to specify embeddings file, config file, experiment parameters, and output location.

**Why this priority**: CLI extends accessibility for users who prefer command-line workflows or want to integrate with shell scripts.

**Independent Test**: Run an experiment via CLI commands; verify it produces the same results as the Python API.

**Acceptance Scenarios**:

1. **Given** a user runs `labricate experiment --embeddings data.csv --config config.json --param "hdbscan.min_cluster_size" --values 5,10,15,20`, **When** the command completes, **Then** an experiment runs with the specified parameter values.

2. **Given** a user runs `labricate experiment --help`, **When** viewing output, **Then** they see documentation for all available options including parallelism, output format, and error handling.

3. **Given** a user specifies `--output-format csv --output-dir ./results`, **When** the experiment completes, **Then** results are saved as CSV in the specified directory.

---

### Edge Cases

- What happens when a user specifies an invalid parameter path (e.g., `"umap.invalid_param"` or `"nonexistent.n_neighbors"`)? System validates all paths before running and fails fast with clear error listing invalid paths.
- What happens when HDBSCAN produces 0 valid clusters (all points are noise)? System should log the failure and continue with remaining runs (if configured).
- What happens when a parameter value is invalid for the algorithm (e.g., `min_cluster_size=0`)? System should validate config before running and report errors.
- How does the system handle very large embedding matrices that don't fit in memory? System should warn about memory requirements; parallelism should be limited.
- What happens when the user specifies conflicting parameters (e.g., HDBSCAN params with K-Means algorithm)? System should warn about unused parameters.
- How does the system handle a custom pipeline that returns mismatched array sizes? System should validate output shapes and raise clear errors.
- What happens when Metricate evaluation fails for a specific clustering? System should log the failure, save partial results, and continue.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Experiment Execution (P1)

- **FR-001**: System MUST accept embeddings in three formats: NumPy array `(n_samples, n_dims)`, CSV file path, or pandas DataFrame.
- **FR-002**: System MUST accept a base configuration as a Python dict or JSON file.
- **FR-003**: System MUST support single-hyperparameter experiments where one parameter (specified via dot notation path, e.g., `"umap.n_neighbors"` or `"hdbscan.min_cluster_size"`) varies across user-defined values while others remain constant.
- **FR-003a**: System MUST validate all parameter paths before executing any experiment runs; invalid paths MUST cause immediate failure with a clear error message listing all invalid paths.
- **FR-004**: System MUST execute pipeline runs sequentially by default, with each run using a different value for the varying parameter.
- **FR-005**: System MUST evaluate each pipeline output using Metricate via direct import (`metricate.evaluate()`).
- **FR-005a**: System MUST support optional metric filtering, allowing users to specify which metrics to include or exclude from evaluation (default: calculate all 34 metrics).
- **FR-005b**: When ground truth labels are provided, system MUST automatically include supervised metrics (ARI, NMI, etc.) in evaluation; when not provided (default), only unsupervised metrics are calculated.
- **FR-006**: System MUST save experiment results including: experiment configuration, per-run configurations, per-run metric results, and comparison summary.
- **FR-006a**: System MUST save a checkpoint file after each completed run, recording which runs have finished and their results.
- **FR-006b**: System MUST support resuming an interrupted experiment from the last checkpoint, skipping already-completed runs.
- **FR-006c**: When resuming, system MUST detect existing checkpoint in output directory and prompt user to resume or start fresh.
- **FR-006d**: When resuming, if config differs from checkpoint's original config, system MUST warn and require `--force` flag to start fresh; without `--force`, abort with error.

#### Default BERTopic Pipeline (P1)

- **FR-007**: System MUST provide a default BERTopic-style pipeline that performs UMAP dimensionality reduction followed by clustering.
- **FR-008**: Default pipeline MUST support UMAP parameters: `n_neighbors`, `n_components`, `min_dist`, `metric`, `repulsion_strength`, `low_memory`.
- **FR-009**: Default pipeline MUST support clustering algorithm selection via `clustering_algorithm` config field: "hdbscan" or "kmeans".
- **FR-010**: When `clustering_algorithm="hdbscan"`, pipeline MUST use parameters: `min_cluster_size`, `min_samples`, `cluster_selection_method`, `metric`.
- **FR-011**: When `clustering_algorithm="kmeans"`, pipeline MUST use parameter: `n_clusters`.
- **FR-012**: Default pipeline MUST return a tuple of `(labels: np.ndarray, reduced_embeddings: np.ndarray)`.
- **FR-012a**: Default pipeline MUST support a `random_seed` config parameter (default: 42) for reproducibility; if set to null, runs are stochastic.
- **FR-012b**: When `random_seed` is provided, pipeline MUST apply it to UMAP and clustering algorithms to ensure reproducible results.

#### Custom Pipeline Support (P2)

- **FR-013**: System MUST accept an optional custom pipeline function with signature: `(embeddings: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]`.
- **FR-014**: When a custom pipeline is provided, system MUST use it instead of the default BERTopic pipeline.
- **FR-015**: System MUST validate custom pipeline output shapes: labels must be 1D with length `n_samples`, reduced_embeddings must be 2D with shape `(n_samples, n_dims)`.

#### Grid Search (P3)

- **FR-016**: System MUST support grid search experiments where multiple parameters vary simultaneously.
- **FR-017**: Grid search MUST execute all combinations of specified parameter values.
- **FR-018**: Grid search results MUST be presented in a format showing metric values for each parameter combination.

#### Output and Storage (P2)

- **FR-019**: System MUST save results in user-selected format: JSON, CSV, or both.
- **FR-020**: System MUST save intermediate clustering outputs (CSV files) for each pipeline run.
- **FR-021**: System MUST use hierarchical directory structure: `experiments/<experiment_name>/<timestamp>/`.
- **FR-022**: Directory structure MUST contain: `config.json` (experiment config), `runs/` (intermediate outputs), `results.{json,csv}` (combined results), `visualizations/` (charts), `checkpoint.json` (resume state).
- **FR-023**: System MUST generate line charts showing metric values plotted against hyperparameter values.

#### Parallelism (P2)

- **FR-024**: System MUST support optional parallel execution of pipeline runs.
- **FR-025**: Parallelism MUST be configurable via `n_workers` parameter (default=1, sequential).
- **FR-025a**: If `n_workers` exceeds CPU count, system MUST cap at CPU count and emit a warning.
- **FR-026**: System MUST handle parallel execution failures gracefully, ensuring partial results are saved.

#### Progress and Logging (P2)

- **FR-027**: System MUST display a progress bar showing current run number and parameter values using dot notation (e.g., "Run 3/12: hdbscan.min_cluster_size=15").
- **FR-028**: System MUST log verbose timing information for each step: UMAP duration, clustering duration, Metricate evaluation duration.
- **FR-029**: System MUST log total experiment duration upon completion.

#### Error Handling (P2)

- **FR-030**: System MUST support configurable error handling: "fail_fast" or "continue" (default).
- **FR-031**: When `error_handling="continue"`, system MUST skip failed runs, log errors, and continue with remaining runs.
- **FR-032**: When `error_handling="fail_fast"`, system MUST stop immediately on first error and save partial results.
- **FR-033**: System MUST report all failures in the final results summary.

#### Python API (P1)

- **FR-034**: System MUST be importable as a Python module: `from labricate import Experiment, BERTopicPipeline`.
- **FR-035**: System MUST provide `Experiment` class for configuring and running experiments.
- **FR-036**: System MUST provide `BERTopicPipeline` class for the default pipeline.
- **FR-037**: API MUST return results as a structured object containing DataFrames and metadata.

#### CLI (P3)

- **FR-038**: System MUST provide CLI command: `labricate experiment` for running experiments.
- **FR-039**: CLI MUST support flags: `--embeddings`, `--config`, `--param`, `--values`, `--output-format`, `--output-dir`, `--workers`, `--error-handling`, `--include-metrics`, `--exclude-metrics`, `--resume`.
- **FR-040**: CLI MUST support `--help` flag with documentation for all options.

### Non-Functional Requirements

- **NFR-001**: Single-parameter experiment with 10 runs on 10,000 points SHOULD complete within 10 minutes (excluding Metricate O(n²) metrics).
- **NFR-002**: System SHOULD handle embedding matrices up to 100,000 rows without memory errors (with sequential execution).
- **NFR-003**: Progress updates SHOULD appear at least every 5 seconds during long-running operations.
- **NFR-004**: System SHOULD provide clear error messages indicating the specific issue when invalid input is provided.

### Key Entities

- **Experiment**: A complete hyperparameter search session containing: embeddings, base config, parameter(s) to vary, value ranges, pipeline function, and results. Key attributes: experiment_name, timestamp, total_runs, completed_runs, failed_runs.
- **ExperimentConfig**: Configuration for an experiment containing: base pipeline config, varying parameter name(s), value range(s), output settings, parallelism settings, error handling mode.
- **PipelineConfig**: Configuration for a single pipeline run containing all hyperparameters: UMAP settings, clustering algorithm, clustering settings.
- **PipelineResult**: Output from a single pipeline run containing: labels array, reduced embeddings array, config used, timing metadata.
- **ExperimentResult**: Complete experiment output containing: all run results, Metricate evaluations, comparison data, visualizations.
- **BERTopicPipeline**: Default pipeline implementation performing UMAP + HDBSCAN/K-Means clustering.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can run a single-parameter experiment with 10 values and receive comparison results for all 34 Metricate metrics.
- **SC-002**: Default BERTopic pipeline correctly applies all specified UMAP and clustering parameters.
- **SC-003**: Users can provide custom pipeline functions and have them executed correctly within experiments.
- **SC-004**: Grid search experiments execute all parameter combinations (verified by run count = product of value counts).
- **SC-005**: Intermediate clustering CSVs are saved and can be loaded independently for further analysis.
- **SC-006**: Generated line charts correctly show metric trends across hyperparameter values.
- **SC-007**: Parallel execution with 4 workers reduces total experiment time by at least 50% compared to sequential (on 8+ run experiments).
- **SC-008**: Failed runs are logged and do not prevent successful runs from completing (when configured).

## Assumptions

- Users have pre-computed embeddings (before dimensionality reduction) in a supported format.
- Metricate is installed and importable in the same environment as Labricate.
- UMAP and HDBSCAN/scikit-learn are available as dependencies.
- Users understand clustering evaluation metrics and can interpret Metricate output.
- The primary use case is text embeddings from NLP pipelines, but the framework is agnostic to embedding source.
- Default BERTopic pipeline does not include the topic representation step (TF-IDF, etc.) - only clustering.

## Default Configuration Schema

```json
{
  "random_seed": 42,
  "umap": {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "repulsion_strength": 1.0,
    "low_memory": false
  },
  "clustering_algorithm": "hdbscan",
  "hdbscan": {
    "min_cluster_size": 10,
    "min_samples": 10,
    "cluster_selection_method": "eom",
    "metric": "euclidean"
  },
  "kmeans": {
    "n_clusters": 10
  }
}
```

> **Note**: Set `"random_seed": null` for stochastic (non-reproducible) behavior.

## Parameter Path Specification

Users specify which hyperparameter to vary using **dot notation paths** that match the nested config structure:

### Available Parameter Paths

| Path | Type | Description |
|------|------|-------------|
| `umap.n_neighbors` | int | Number of neighbors for UMAP |
| `umap.n_components` | int | Target dimensionality for UMAP |
| `umap.min_dist` | float | Minimum distance between points in UMAP |
| `umap.metric` | str | Distance metric for UMAP |
| `umap.repulsion_strength` | float | Repulsion strength for UMAP |
| `umap.low_memory` | bool | Enable low-memory mode for large datasets (slower but uses less RAM) |
| `hdbscan.min_cluster_size` | int | Minimum cluster size for HDBSCAN |
| `hdbscan.min_samples` | int | Minimum samples for HDBSCAN core points |
| `hdbscan.cluster_selection_method` | str | HDBSCAN cluster selection method ("eom" or "leaf") |
| `hdbscan.metric` | str | Distance metric for HDBSCAN (default: "euclidean") |
| `kmeans.n_clusters` | int | Number of clusters for K-Means |
| `clustering_algorithm` | str | Which clustering algorithm to use ("hdbscan" or "kmeans") |
| `random_seed` | int/null | Random seed for reproducibility (default: 42); set to null for stochastic |

### Usage Examples

**Single parameter experiment (Python API):**
```python
experiment.run(
    param="umap.n_neighbors",
    values=[5, 10, 15, 20, 30]
)
```

**Single parameter experiment (CLI):**
```bash
labricate experiment --param "hdbscan.min_cluster_size" --values 5,10,15,20
```

**Grid search (multiple parameters):**
```python
experiment.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20],
        "hdbscan.min_cluster_size": [5, 10, 15]
    }
)
```

### Validation Rules

- All parameter paths are validated before experiment execution
- Invalid paths (non-existent section or parameter) cause immediate failure with clear error message
- Path format must be `"<section>.<parameter>"` for nested params or `"<parameter>"` for top-level params
- Custom pipelines may define their own parameter paths; validation uses the provided config structure

## Module Structure

> **Note**: Per project constitution (single package), Labricate is implemented as a submodule of `metricate`.

```
metricate/
├── labricate/             # Experimentation submodule
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── experiment.py      # Experiment class
│   │   ├── config.py          # Config handling and validation
│   │   └── loader.py          # Input format loading
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── base.py            # Pipeline interface/protocol
│   │   └── bertopic.py        # Default BERTopic pipeline
│   ├── output/
│   │   ├── __init__.py
│   │   ├── storage.py         # Results saving (JSON/CSV)
│   │   └── visualization.py   # Chart generation
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # Progress bar and timing
│       └── parallel.py        # Parallel execution
└── cli/
    └── main.py                # Extended with labricate subcommand
```
