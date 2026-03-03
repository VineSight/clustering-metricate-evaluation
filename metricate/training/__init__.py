"""
Training dataset generation and weight learning module for metricate.

This module provides functions to:
1. Generate labeled training datasets from clustering CSVs by systematically 
   degrading good clusterings and calculating metrics.
2. Train regression models (Ridge/Lasso) to learn optimal metric weights.
3. Export and load trained weights for scoring new clusterings.

Example (Generate Training Data):
    >>> import metricate
    >>> result = metricate.generate_training_data("clustering.csv", "./output/")
    >>> df = result.to_dataframe()
    >>> result.to_csv("training_dataset.csv")

Example (Train Weights):
    >>> from metricate.training import train_weights
    >>> result = train_weights("training_data.csv", regularization="ridge")
    >>> result.weights.save("weights.json")
"""

from metricate.training.generator import (
    generate_training_data,
    generate_training_data_batch,
)
from metricate.training.learner import (
    CVResult,
    TrainingResult,
    cross_validate_weights,
    export_weights,
    plot_feature_importance,
    sanity_check,
    train_weights,
)
from metricate.training.result import TrainingDataResult
from metricate.training.weights import (
    MetricWeights,
    compute_compound_score,
    load_weights,
    validate_weights_schema,
)

__all__ = [
    # Training data generation
    "generate_training_data",
    "generate_training_data_batch",
    "TrainingDataResult",
    # Weight learning
    "train_weights",
    "cross_validate_weights",
    "export_weights",
    "plot_feature_importance",
    "sanity_check",
    "TrainingResult",
    "CVResult",
    # Weights dataclass and utilities
    "MetricWeights",
    "load_weights",
    "compute_compound_score",
    "validate_weights_schema",
]
