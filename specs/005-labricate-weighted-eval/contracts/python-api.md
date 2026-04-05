# Python API Contract: Labricate Weighted Evaluation

**Version**: 1.0  
**Date**: 2026-03-26

## Module: metricate.labricate

### Experiment Class

#### Constructor

```python
def __init__(
    self,
    embeddings: np.ndarray | pd.DataFrame | str | Path,
    config: dict[str, Any] | str | Path,
    name: str | None = None,
    output_dir: str | Path = "./experiments",
    output_format: Literal["json", "csv", "both"] = "json",
    pipeline: Pipeline | Callable | None = None,
    weights: str | Path | dict[str, Any] | None = None,  # NEW
) -> None:
    """
    Initialize an experiment.
    
    Args:
        embeddings: Input embeddings (array, DataFrame, or file path).
        config: Base pipeline configuration (dict or JSON path).
        name: Experiment name (auto-generated if None).
        output_dir: Directory for results.
        output_format: Output format ("json", "csv", "both").
        pipeline: Custom pipeline function (default: BERTopicPipeline).
        weights: NEW - Weights for compound scoring. Accepts:
            - str/Path: Path to weights JSON file
            - dict: Weights dictionary with 'coefficients' and 'bias'
            - None: No weighted scoring (current behavior)
    
    Raises:
        ValueError: If weights JSON is invalid (FR-002).
        FileNotFoundError: If weights file path doesn't exist.
    """
```

#### run() Method

```python
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
    mode: Literal["light", "heavy"] = "heavy",  # NEW
    best_metric: str = "Silhouette",  # NEW
) -> ExperimentResult:
    """
    Run a single-parameter experiment.
    
    Args:
        param: Dot-notation parameter path.
        values: List of values to test.
        n_workers: Number of parallel workers.
        error_handling: How to handle failed runs.
        include_metrics: Specific metrics to calculate.
        exclude_metrics: Metrics to skip.
        ground_truth: Ground truth labels for supervised metrics.
        resume: Whether to resume from checkpoint.
        force: Force start fresh if config mismatch.
        verbose: Whether to print progress.
        mode: NEW - Computation mode (FR-005, FR-006):
            - "light": Excludes expensive O(n²) metrics
            - "heavy": All metrics (DEFAULT)
        best_metric: NEW - Metric for best_run when no weights.
    
    Returns:
        ExperimentResult with runs, metrics, and best_run.
    
    Behavior:
        - If mode="light" and include_metrics specified, include_metrics
          takes precedence (user's explicit choice wins).
        - If weights provided and mode="light", warns if excluded metrics
          account for >30% of weight (FR-017).
        - best_run uses compound_score if weights, else best_metric.
    """
```

#### run_grid() Method

```python
def run_grid(
    self,
    params: dict[str, list[Any]],
    n_workers: int = 1,
    error_handling: Literal["continue", "fail_fast"] = "continue",
    include_metrics: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
    ground_truth: np.ndarray | None = None,
    resume: bool = False,
    force: bool = False,
    verbose: bool = True,
    mode: Literal["light", "heavy"] = "heavy",  # NEW
    best_metric: str = "Silhouette",  # NEW
) -> ExperimentResult:
    """
    Run a grid search over multiple parameters.
    
    Same NEW parameters as run().
    """
```

---

### ExperimentResult Class

```python
@dataclass
class ExperimentResult:
    experiment_id: str
    experiment_name: str
    config: dict[str, Any]
    runs: list[RunResult]
    summary: ExperimentSummary
    output_path: str | None = None
    best_run: BestRunInfo | None = None  # NEW (FR-004)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with columns:
            - run_id: int
            - [param_name]: Any (for each varied parameter)
            - n_clusters: int
            - n_noise: int
            - status: str
            - [metric_name]: float (for each computed metric)
            - compound_score: float | None  # NEW (FR-008)
            - is_best_run: bool  # NEW (FR-009)
        """
    
    def get_best_run(
        self,
        metric: str | None = None,
        direction: str | None = None,
    ) -> RunResult:
        """
        Get the run with the best value.
        
        Args:
            metric: Metric name. If None and weights provided, uses
                    compound_score. Otherwise defaults to Silhouette.
            direction: "higher" or "lower". Auto-detected if None.
        
        Returns:
            RunResult with best metric value.
        
        Note:
            For compound_score, direction is always "higher".
        """
```

---

### RunResult Class

```python
@dataclass
class RunResult:
    run_id: int
    param_values: dict[str, Any]
    pipeline_result: PipelineResult
    metrics: list[MetricResult]
    compound_score: float | None = None  # NEW (FR-003)
```

---

### BestRunInfo Class (NEW)

```python
@dataclass
class BestRunInfo:
    """Information about the best-performing experiment run."""
    
    run_id: int
    """Primary best run identifier."""
    
    param_values: dict[str, Any]
    """Best hyperparameter values (FR-015)."""
    
    score: float
    """The score value (compound_score or metric value)."""
    
    score_type: str
    """'compound_score' or metric name."""
    
    tied_run_ids: list[int]
    """All tied run IDs, empty if no ties (FR-016)."""
    
    def __str__(self) -> str:
        """
        Human-readable representation.
        
        Examples:
            "Best run: run_id=3 (compound_score=0.847)"
            "Best run: run_id=3 (tied with run_id=7)"
        """
```

---

## Module: metricate.labricate.core.modes (NEW)

```python
from typing import Literal

ComputationMode = Literal["light", "heavy"]

def get_expensive_metrics() -> list[str]:
    """
    Return metric names with skip_large=True from METRIC_REFERENCE.
    
    Returns:
        List of metric names: ["Gamma", "G-plus", "Tau", 
                               "Point-Biserial", "McClain-Rao", "NIVA"]
    """

def apply_mode_exclusions(
    mode: ComputationMode,
    exclude_metrics: list[str] | None,
    include_metrics: list[str] | None = None,
) -> list[str] | None:
    """
    Compute final exclusion list based on mode and user preferences.
    
    Args:
        mode: "light" or "heavy"
        exclude_metrics: User-provided exclusions
        include_metrics: User-provided inclusions (takes precedence)
    
    Returns:
        Final exclusion list, or None if no exclusions.
    
    Behavior:
        - heavy mode: Return exclude_metrics unchanged
        - light mode: Add expensive metrics to exclusions,
                      UNLESS metric is in include_metrics
    """
```

---

## Module: metricate.labricate.core.scoring (NEW)

```python
from metricate.training.weights import MetricWeights

def compute_run_scores(
    runs: list[RunResult],
    weights: MetricWeights,
) -> None:
    """
    Compute compound_score for each run (mutates runs).
    
    Args:
        runs: List of RunResult to score.
        weights: Weights for compound score calculation.
    
    Side Effects:
        Sets compound_score on each run with completed status.
    """

def find_best_run(
    runs: list[RunResult],
    weights: MetricWeights | None,
    best_metric: str = "Silhouette",
) -> BestRunInfo | None:
    """
    Determine best run with tie detection.
    
    Args:
        runs: Completed runs to evaluate.
        weights: If provided, uses compound_score.
        best_metric: Metric to use if no weights.
    
    Returns:
        BestRunInfo with tie information, or None if no completed runs.
    """

def check_weight_coverage(
    weights: MetricWeights,
    excluded_metrics: list[str],
    threshold: float = 0.30,
) -> str | None:
    """
    Check if excluded metrics account for significant weight.
    
    Args:
        weights: Weights with coefficients.
        excluded_metrics: Metrics being excluded.
        threshold: Warning threshold (default 30%).
    
    Returns:
        Warning message if threshold exceeded, else None.
    
    Example return:
        "Warning: Excluded metrics ['Gamma', 'Tau'] account for 45% of 
         weight coefficients. Compound scores may be unreliable."
    """
```

---

## Error Contracts

### WeightsValidationError

Raised when weights JSON fails schema validation (FR-002).

```python
# Missing required field
ValueError: "Weights JSON missing required field 'coefficients'"

# Invalid coefficient type
ValueError: "Coefficient value must be numeric: Silhouette_norm"

# Invalid naming convention
ValueError: "All coefficient keys must end with '_norm', got: ['Silhouette']"
```

### WeightCoverageWarning

Warning when excluded metrics dominate weights (FR-017).

```python
import warnings
warnings.warn(
    "Excluded metrics ['Gamma', 'Tau'] account for 45% of weight "
    "coefficients. Compound scores may be unreliable.",
    WeightCoverageWarning,
)
```
