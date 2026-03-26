# Python API Contract: Labricate

**Date**: March 18, 2026  
**Feature**: 004-labricate

---

## Module Exports

```python
from metricate.labricate import (
    # Core classes
    Experiment,
    BERTopicPipeline,
    
    # Result types
    ExperimentResult,
    RunResult,
    PipelineResult,
    
    # Config helpers
    load_config,
    validate_config,
    
    # Utilities
    load_embeddings,
)
```

---

## Experiment Class

### Constructor

```python
class Experiment:
    def __init__(
        self,
        embeddings: np.ndarray | pd.DataFrame | str,  # array, DataFrame, or CSV path
        config: dict | str,                            # config dict or JSON path
        name: str | None = None,                       # experiment name (auto-generated if None)
        output_dir: str = "./experiments",
        output_format: Literal["json", "csv", "both"] = "json",
        pipeline: Callable | None = None,              # custom pipeline (default: BERTopicPipeline)
    ) -> None:
        """
        Initialize an experiment.
        
        Args:
            embeddings: Input embeddings in any supported format
            config: Base pipeline configuration
            name: Experiment name (default: auto-generated timestamp)
            output_dir: Directory for results
            output_format: Output format for results
            pipeline: Custom pipeline function (default: BERTopicPipeline)
        
        Raises:
            ValueError: If embeddings format is invalid
            ValueError: If config is invalid
            FileNotFoundError: If file paths don't exist
        """
```

### Methods

```python
def run(
    self,
    param: str,
    values: list,
    n_workers: int = 1,
    error_handling: Literal["continue", "fail_fast"] = "continue",
    include_metrics: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
    ground_truth: np.ndarray | None = None,
    resume: bool = False,
    force: bool = False,
) -> ExperimentResult:
    """
    Run a single-parameter experiment.
    
    Args:
        param: Dot-notation parameter path (e.g., "hdbscan.min_cluster_size")
        values: List of values to test
        n_workers: Number of parallel workers (1 = sequential)
        error_handling: How to handle failed runs
        include_metrics: Specific metrics to calculate (None = all)
        exclude_metrics: Metrics to skip
        ground_truth: Ground truth labels for supervised metrics (None = unsupervised only)
        resume: Whether to resume from checkpoint if exists
        force: Force start fresh if config mismatch on resume
    
    Returns:
        ExperimentResult with all run results and metrics
    
    Raises:
        ValueError: If param path is invalid
        ValueError: If values list is empty
        ExperimentError: If all runs fail (when error_handling="fail_fast")
    
    Example:
        >>> exp = Experiment(embeddings, config)
        >>> result = exp.run(
        ...     param="hdbscan.min_cluster_size",
        ...     values=[5, 10, 15, 20, 25, 30]
        ... )
        >>> print(result.summary)
    """

def run_grid(
    self,
    params: dict[str, list],
    n_workers: int = 1,
    error_handling: Literal["continue", "fail_fast"] = "continue",
    include_metrics: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
    ground_truth: np.ndarray | None = None,
    resume: bool = False,
    force: bool = False,
) -> ExperimentResult:
    """
    Run a grid search experiment over multiple parameters.
    
    Args:
        params: Dict mapping parameter paths to value lists
        n_workers: Number of parallel workers
        error_handling: How to handle failed runs
        include_metrics: Specific metrics to calculate
        exclude_metrics: Metrics to skip
        resume: Whether to resume from checkpoint
    
    Returns:
        ExperimentResult with all combination results
    
    Example:
        >>> result = exp.run_grid(
        ...     params={
        ...         "umap.n_neighbors": [10, 15, 20],
        ...         "hdbscan.min_cluster_size": [5, 10, 15]
        ...     },
        ...     n_workers=4
        ... )
    """

def validate_param(self, param: str) -> bool:
    """
    Validate a parameter path against the config.
    
    Args:
        param: Dot-notation parameter path
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If path is invalid (with detailed message)
    """
```

---

## BERTopicPipeline Class

```python
class BERTopicPipeline:
    """
    Default clustering pipeline using BERTopic library.
    
    Wraps BERTopic's UMAP + HDBSCAN/K-Means pipeline, extracting
    cluster labels and reduced embeddings for Metricate evaluation.
    Topic representation (c-TF-IDF) is skipped by default for speed.
    """
    
    def __call__(
        self,
        embeddings: np.ndarray,
        config: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute the BERTopic pipeline.
        
        Args:
            embeddings: Input embeddings (n_samples, n_dims)
            config: Pipeline configuration dict
        
        Returns:
            Tuple of (labels, reduced_embeddings)
            - labels: 1D array of cluster assignments (from topic_model.topics_)
            - reduced_embeddings: 2D array of UMAP-reduced embeddings
              (from topic_model.umap_model.embedding_)
        
        Raises:
            ValueError: If config is invalid
            RuntimeError: If clustering produces no valid clusters
        
        Notes:
            - Uses placeholder empty docs since users provide pre-computed embeddings
            - Topic representation (c-TF-IDF) is skipped unless enable_topic_representation=True
        """
    
    @staticmethod
    def default_config() -> dict:
        """Return the default configuration."""
        return {
            "random_seed": 42,
            "umap": {
                "n_neighbors": 15,
                "n_components": 5,
                "min_dist": 0.0,
                "metric": "cosine",
                "repulsion_strength": 1.0,
                "low_memory": False
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
            },
            "enable_topic_representation": False,
            "calculate_probabilities": False
        }
```

---

## Custom Pipeline Protocol

```python
from typing import Protocol

class Pipeline(Protocol):
    """Protocol for custom pipeline functions."""
    
    def __call__(
        self,
        embeddings: np.ndarray,
        config: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute the pipeline.
        
        Args:
            embeddings: Input embeddings (n_samples, n_dims)
            config: Configuration dict (structure defined by pipeline)
        
        Returns:
            Tuple of (labels, reduced_embeddings)
            - labels: 1D int array of cluster assignments, length n_samples
            - reduced_embeddings: 2D float array, shape (n_samples, n_dims)
        """
        ...
```

---

## Result Types

```python
@dataclass
class ExperimentResult:
    """Complete experiment results."""
    experiment_id: str
    experiment_name: str
    config: dict
    runs: list[RunResult]
    summary: ExperimentSummary
    output_path: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
    
    def get_best_run(self, metric: str) -> RunResult:
        """Get run with best value for specified metric."""
    
    def plot_metric(self, metric: str, save_path: str | None = None) -> None:
        """Plot metric values across parameter values."""


@dataclass
class RunResult:
    """Single run result."""
    run_id: int
    param_values: dict[str, Any]
    pipeline_result: PipelineResult
    metrics: dict[str, float]
    status: Literal["completed", "failed", "skipped"]
    error: str | None


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    labels: np.ndarray
    reduced_embeddings: np.ndarray
    n_clusters: int
    n_noise: int
    timing: dict[str, float]


@dataclass
class ExperimentSummary:
    """Experiment summary statistics."""
    total_runs: int
    completed_runs: int
    failed_runs: int
    skipped_runs: int
    total_duration_seconds: float
```

---

## Utility Functions

```python
def load_embeddings(
    source: np.ndarray | pd.DataFrame | str,
    embedding_cols: list[str] | None = None
) -> np.ndarray:
    """
    Load embeddings from various formats.
    
    Args:
        source: Embeddings as array, DataFrame, or file path
        embedding_cols: Column names for embeddings (auto-detected if None)
    
    Returns:
        NumPy array of shape (n_samples, n_dims)
    """

def load_config(path: str) -> dict:
    """Load configuration from JSON file."""

def validate_config(config: dict) -> list[str]:
    """
    Validate configuration structure.
    
    Returns:
        List of validation error messages (empty if valid)
    """
```

---

## Exceptions

```python
class LabricateError(Exception):
    """Base exception for Labricate errors."""

class ConfigValidationError(LabricateError):
    """Invalid configuration."""

class InvalidParameterPathError(LabricateError):
    """Invalid dot-notation parameter path."""

class PipelineExecutionError(LabricateError):
    """Pipeline execution failed."""

class ExperimentError(LabricateError):
    """Experiment-level error."""
```

---

## Usage Examples

### Basic Single-Parameter Experiment

```python
from metricate.labricate import Experiment

# Load embeddings (any format)
embeddings = np.load("embeddings.npy")

# Define base config
config = {
    "random_seed": 42,
    "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0},
    "clustering_algorithm": "hdbscan",
    "hdbscan": {"min_cluster_size": 10, "min_samples": 5}
}

# Create and run experiment
exp = Experiment(embeddings, config, name="cluster_size_sweep")
result = exp.run(
    param="hdbscan.min_cluster_size",
    values=[5, 10, 15, 20, 25, 30, 40, 50]
)

# Analyze results
print(result.summary)
df = result.to_dataframe()
best = result.get_best_run("Silhouette")
print(f"Best Silhouette at min_cluster_size={best.param_values['hdbscan.min_cluster_size']}")
```

### Grid Search with Parallelism

```python
result = exp.run_grid(
    params={
        "umap.n_neighbors": [10, 15, 20, 30],
        "hdbscan.min_cluster_size": [5, 10, 15, 20]
    },
    n_workers=4,
    error_handling="continue"
)
```

### Custom Pipeline

```python
def my_pipeline(embeddings: np.ndarray, config: dict):
    # Custom preprocessing
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Custom reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=config["pca"]["n_components"])
    reduced = pca.fit_transform(normalized)
    
    # Custom clustering
    from sklearn.cluster import AgglomerativeClustering
    clusterer = AgglomerativeClustering(n_clusters=config["agg"]["n_clusters"])
    labels = clusterer.fit_predict(reduced)
    
    return labels, reduced

exp = Experiment(
    embeddings,
    config={"pca": {"n_components": 10}, "agg": {"n_clusters": 5}},
    pipeline=my_pipeline
)
```
