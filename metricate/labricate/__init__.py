"""Labricate - Hyperparameter Experimentation Framework for Clustering Pipelines.

Labricate provides tools for running experiments with varying hyperparameters
on clustering pipelines, evaluating results with Metricate, and comparing outcomes.

Example:
    >>> from metricate.labricate import Experiment, BERTopicPipeline
    >>> import numpy as np
    >>> 
    >>> embeddings = np.random.randn(1000, 384)
    >>> config = {
    ...     "umap": {"n_neighbors": 15, "n_components": 5},
    ...     "clustering_algorithm": "hdbscan",
    ...     "hdbscan": {"min_cluster_size": 10}
    ... }
    >>> exp = Experiment(embeddings, config)
    >>> result = exp.run(param="hdbscan.min_cluster_size", values=[5, 10, 15, 20])
"""

from metricate.labricate.core.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentSummary,
    PipelineResult,
    RunResult,
)
from metricate.labricate.core.config import (
    load_config,
    validate_config,
)
from metricate.labricate.core.loader import load_embeddings
from metricate.labricate.core.modes import (
    ComputationMode,
    apply_mode_exclusions,
    get_expensive_metrics,
)
from metricate.labricate.core.scoring import (
    BestRunInfo,
    WeightCoverageWarning,
    check_weight_coverage,
    compute_run_scores,
    find_best_run,
)
from metricate.labricate.pipelines.bertopic import BERTopicPipeline

__all__ = [
    # Core classes
    "BERTopicPipeline",
    "Experiment",
    # Result types
    "BestRunInfo",
    "ExperimentResult",
    "ExperimentSummary",
    "PipelineResult",
    "RunResult",
    # Modes
    "ComputationMode",
    "apply_mode_exclusions",
    "get_expensive_metrics",
    # Scoring
    "WeightCoverageWarning",
    "check_weight_coverage",
    "compute_run_scores",
    "find_best_run",
    # Config helpers
    "load_config",
    "validate_config",
    # Utilities
    "load_embeddings",
]
