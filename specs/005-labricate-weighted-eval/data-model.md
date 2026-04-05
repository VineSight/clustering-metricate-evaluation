# Data Model: Labricate Weighted Evaluation

**Phase**: 1 (Design)  
**Date**: 2026-03-26

## Entities

### 1. BestRunInfo (NEW)

Represents the optimal run from an experiment with tie awareness.

```python
@dataclass
class BestRunInfo:
    """Information about the best-performing experiment run."""
    
    run_id: int
    """Primary best run identifier (1-indexed)."""
    
    param_values: dict[str, Any]
    """Hyperparameter values that achieved best score. FR-015."""
    
    score: float
    """The score value (compound_score if weights, else metric value)."""
    
    score_type: str
    """Either 'compound_score' or the metric name used for ranking."""
    
    tied_run_ids: list[int]
    """List of all run IDs with identical best score. Empty if no ties. FR-016."""
```

**Validation Rules**:
- `run_id` must be positive integer
- `param_values` must not be empty
- `score` must be finite float
- `tied_run_ids` must not contain `run_id` (it's implicit)

**JSON Representation**:
```json
{
  "run_id": 3,
  "param_values": {"hdbscan.min_cluster_size": 15},
  "score": 0.847,
  "score_type": "compound_score",
  "tied_run_ids": [7]
}
```

---

### 2. ComputationMode (NEW)

Enum-like type for controlling metric computation scope.

```python
from typing import Literal

ComputationMode = Literal["light", "heavy"]
"""
Computation mode for experiments.

- "light": Excludes expensive O(n²) metrics (Gamma, Tau, G-plus, 
           Point-Biserial, McClain-Rao, NIVA) for faster iteration
- "heavy": Computes all metrics for comprehensive evaluation (DEFAULT)
"""
```

**State Transitions**: N/A (stateless)

---

### 3. RunResult (MODIFIED)

Existing entity, extended with compound_score.

```python
@dataclass
class RunResult:
    """Complete result for a single experiment run."""
    
    run_id: int
    param_values: dict[str, Any]
    pipeline_result: PipelineResult
    metrics: list[MetricResult]
    
    # NEW FIELD
    compound_score: float | None = None
    """Weighted quality score (0-1) if weights provided, else None."""
```

---

### 4. ExperimentResult (MODIFIED)

Existing entity, extended with best_run and is_best tracking.

```python
@dataclass
class ExperimentResult:
    """Complete experiment output."""
    
    experiment_id: str
    experiment_name: str
    config: dict[str, Any]
    runs: list[RunResult]
    summary: ExperimentSummary
    output_path: str | None = None
    
    # NEW FIELD
    best_run: BestRunInfo | None = None
    """Best run information. None if all runs failed."""
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        NEW columns when weights used:
        - compound_score: float (weighted quality score)
        - is_best_run: bool (True for best run, handles ties)
        """
        ...
```

---

### 5. Experiment (MODIFIED)

Existing class, extended constructor and methods.

```python
class Experiment:
    """Hyperparameter experimentation orchestrator."""
    
    def __init__(
        self,
        embeddings: np.ndarray | pd.DataFrame | str | Path,
        config: dict[str, Any] | str | Path,
        name: str | None = None,
        output_dir: str | Path = "./experiments",
        output_format: Literal["json", "csv", "both"] = "json",
        pipeline: Pipeline | Callable | None = None,
        
        # NEW PARAMETER
        weights: str | Path | dict[str, Any] | None = None,
        """Weights for compound scoring. Path to JSON or dict. FR-001."""
    ) -> None:
        ...
        
        # NEW ATTRIBUTE
        self._weights: MetricWeights | None
        """Loaded and validated weights, or None."""
    
    def run(
        self,
        param: str,
        values: list[Any],
        n_workers: int = 1,
        error_handling: Literal["continue", "fail_fast"] = "continue",
        include_metrics: list[str] | None = None,
        exclude_metrics: list[str] | None = None,
        ground_truth: np.ndarray | None = None,
        resume: bool = False,
        force: bool = False,
        verbose: bool = True,
        
        # NEW PARAMETERS
        mode: ComputationMode = "heavy",
        """Computation mode. FR-005, FR-006."""
        
        best_metric: str = "Silhouette",
        """Metric for best_run when no weights. FR-005 acceptance 5."""
    ) -> ExperimentResult:
        ...
```

---

### 6. WeightCoverageWarning (NEW)

Runtime warning for dominant excluded metrics.

```python
class WeightCoverageWarning(UserWarning):
    """Warning when excluded metrics account for significant weight."""
    
    excluded_metrics: list[str]
    """Metric names that were excluded."""
    
    excluded_weight_pct: float
    """Percentage of total weight from excluded metrics."""
```

---

## Relationships

```
┌──────────────────┐          ┌─────────────────┐
│    Experiment    │ ◆─────── │  MetricWeights  │
│                  │ 0..1     │   (existing)    │
│  - _weights      │          └─────────────────┘
│  - run()         │
│  - run_grid()    │
└────────┬─────────┘
         │ produces
         ▼
┌──────────────────┐          ┌─────────────────┐
│ ExperimentResult │ ◆─────── │   BestRunInfo   │
│                  │ 0..1     │      (NEW)      │
│  - best_run      │          └─────────────────┘
│  - runs          │
└────────┬─────────┘
         │ 1..*
         ▼
┌──────────────────┐
│    RunResult     │
│                  │
│  - compound_score│  (NEW field)
│  - metrics       │
└──────────────────┘
```

## Validation Rules Summary

| Entity | Field | Rule |
|--------|-------|------|
| BestRunInfo | run_id | > 0 |
| BestRunInfo | param_values | len > 0 |
| BestRunInfo | score | is_finite |
| Experiment | weights | Valid schema if provided (FR-002) |
| Experiment | mode | ∈ {"light", "heavy"} |
| RunResult | compound_score | [0, 1] if present |
