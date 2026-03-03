"""
Metricate: A clustering evaluation toolkit.

Evaluate clustering quality with 36 metrics, compare clusterings,
generate degraded datasets for testing metric robustness, and train
quality scoring models using machine learning.

Example:
    >>> import metricate
    >>> result = metricate.evaluate("clustering.csv")
    >>> print(result.to_table())

    >>> # Compare two clusterings
    >>> comparison = metricate.compare("v1.csv", "v2.csv")
    >>> print(f"Winner: {comparison.winner}")

    >>> # Generate degraded datasets
    >>> result = metricate.degrade("data.csv", "./output/")

    >>> # Train metric weights for compound scoring
    >>> training_result = metricate.train_weights("training_data.csv")
    >>> print(f"CV R²: {training_result.cv_scores['r2_mean']:.3f}")
    >>> training_result.save_weights("weights.json")

    >>> # Evaluate with learned weights
    >>> weights = metricate.load_weights("weights.json")
    >>> result = metricate.evaluate("clustering.csv", weights=weights)
    >>> print(f"Compound score: {result.compound_score:.3f}")

    >>> # Compare with compound scoring
    >>> comparison = metricate.compare("v1.csv", "v2.csv", weights=weights)
    >>> print(f"Weighted winner: {comparison.weighted_winner}")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from metricate.training.weights import MetricWeights

__version__ = "0.1.0"


def evaluate(
    csv_path: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    weights: "MetricWeights | None" = None,
    final_score: bool = False,
):
    """
    Evaluate a single clustering and return all metric scores.

    Computes up to 36 clustering quality metrics on the provided dataset.
    Automatically detects label and embedding columns if not specified.

    Args:
        csv_path: Path to CSV file containing clustering data.
        exclude: List of metric names to skip (e.g., ["Gamma", "Tau"]).
        force_all: If True, compute O(n²) metrics even for large datasets.
        label_col: Name of cluster label column (auto-detected if None).
        embedding_cols: List of embedding column names (auto-detected if None).
        weights: Optional MetricWeights for computing compound score.
        final_score: If True and no weights provided, returns an unweighted average
            of all normalized metrics as final_score. NOT RECOMMENDED for production use.

    Returns:
        EvaluationResult: Object containing metric scores with methods:
            - to_table(): Formatted string table
            - to_dataframe(): pandas DataFrame
            - to_json(): JSON string
            - to_csv(): CSV string
            If weights provided, also includes compound_score.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        InvalidCSVError: If file is not valid CSV.
        ColumnNotFoundError: If required columns not found.
        InsufficientClustersError: If fewer than 2 clusters.

    Example:
        >>> result = metricate.evaluate("clustering.csv")
        >>> print(result.to_table())
        >>> df = result.to_dataframe()
        >>>
        >>> # With weights for compound score
        >>> weights = metricate.load_weights("weights.json")
        >>> result = metricate.evaluate("clustering.csv", weights=weights)
        >>> print(f"Compound score: {result.compound_score:.3f}")
    """
    from metricate.core.evaluator import evaluate as _evaluate

    return _evaluate(
        csv_path,
        exclude=exclude,
        force_all=force_all,
        label_col=label_col,
        embedding_cols=embedding_cols,
        weights=weights,
        final_score=final_score,
    )


def compare(
    csv_path_a: str | Path,
    csv_path_b: str | Path,
    *,
    exclude: list[str] | None = None,
    force_all: bool = False,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    name_a: str = "A",
    name_b: str = "B",
    weights: "MetricWeights | None" = None,
):
    """
    Compare two clusterings and determine the winner.

    Evaluates both clusterings and determines which is better based on
    the majority of metrics. Each metric "votes" for the better clustering
    according to its direction preference (higher/lower is better).

    If weights are provided, also determines winner by compound score.

    Args:
        csv_path_a: Path to first CSV file.
        csv_path_b: Path to second CSV file.
        exclude: List of metric names to skip.
        force_all: If True, compute O(n²) metrics even for large datasets.
        label_col: Name of cluster label column (auto-detected if None).
        embedding_cols: List of embedding column names (auto-detected if None).
        name_a: Display name for first clustering (default: "A").
        name_b: Display name for second clustering (default: "B").
        weights: Optional MetricWeights for weighted winner determination.

    Returns:
        ComparisonResult: Object containing comparison with attributes:
            - winner: Name of the winning clustering
            - wins: Dict with win counts {"A": n, "B": m, "Tie": t}
            - metric_winners: Dict mapping metric name to winner
            - weighted_winner: Winner by compound score (if weights provided)
            - to_table(): Formatted comparison table
            - to_dataframe(): pandas DataFrame

    Example:
        >>> result = metricate.compare("v1.csv", "v2.csv")
        >>> print(f"Winner: {result.winner}")
        >>> print(result.to_table())
        >>>
        >>> # With weights for weighted comparison
        >>> weights = metricate.load_weights("weights.json")
        >>> result = metricate.compare("v1.csv", "v2.csv", weights=weights)
        >>> print(f"Weighted winner: {result.weighted_winner}")
    """
    from metricate.comparison.compare import compare as _compare

    return _compare(
        csv_path_a,
        csv_path_b,
        exclude=exclude,
        force_all=force_all,
        label_col=label_col,
        embedding_cols=embedding_cols,
        name_a=name_a,
        name_b=name_b,
        weights=weights,
    )


def degrade(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    levels: list[str] | None = None,
    types: list[str] | None = None,
    visualize: bool = True,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    random_seed: int = 42,
):
    """
    Generate degraded versions of a clustering dataset.

    Creates multiple degraded versions of the input clustering at various
    intensity levels using different degradation strategies. Useful for
    testing how metrics respond to systematic quality degradation.

    Args:
        csv_path: Path to input CSV with clustering data.
        output_dir: Directory to write degraded datasets.
        levels: Intensity levels (default: ["5pct", "10pct", "25pct", "50pct"]).
        types: Degradation types to apply (default: all 19 types).
        visualize: If True, generate HTML visualizations.
        label_col: Name of cluster label column (auto-detected if None).
        embedding_cols: List of embedding column names (auto-detected if None).
        random_seed: Random seed for reproducibility.

    Returns:
        DegradationResult: Object containing:
            - output_dir: Path to output directory
            - degradations: List of DegradationEntry objects
            - manifest_path: Path to manifest.json
            - csv_files: List of generated CSV paths
            - summary(): Text summary of generation

    Example:
        >>> result = metricate.degrade("clustering.csv", "./output/")
        >>> print(result.summary())
        >>> print(f"Generated {len(result.csv_files)} files")
    """
    from metricate.degradation.toolkit import degrade as _degrade

    return _degrade(
        csv_path,
        output_dir,
        levels=levels,
        types=types,
        visualize=visualize,
        label_col=label_col,
        embedding_cols=embedding_cols,
        random_seed=random_seed,
    )


def list_metrics(include_reference: bool = False) -> list[str] | pd.DataFrame:
    """
    List all available clustering metrics.

    Args:
        include_reference: If True, return DataFrame with full metadata
            (range, direction, tier, complexity). If False, return list of names.

    Returns:
        List of metric names, or DataFrame with columns:
            - metric: Metric name
            - range: Value range (e.g., "[-1, 1]")
            - direction: "higher" or "lower" is better
            - tier: Category (Original, Tier1, Tier2, Tier3, External)
            - complexity: Time complexity (O(n) or O(n²))

    Example:
        >>> metrics = metricate.list_metrics()
        >>> print(f"{len(metrics)} metrics available")
        >>> df = metricate.list_metrics(include_reference=True)
    """
    from metricate.core.reference import METRIC_REFERENCE

    if include_reference:
        data = []
        for name, info in METRIC_REFERENCE.items():
            data.append(
                {
                    "metric": name,
                    "range": info["range"],
                    "direction": info["direction"],
                    "tier": info["tier"],
                    "complexity": info["complexity"],
                }
            )
        return pd.DataFrame(data)
    return list(METRIC_REFERENCE.keys())


def list_degradations() -> dict[str, list[str]]:
    """
    List all available degradation types organized by category.

    Returns:
        Dict mapping category names to lists of degradation types:
            - Label Manipulation: label_swap_random, label_swap_neighboring, ...
            - Cluster Structure: merge_random, split_largest, ...
            - Point Manipulation: noise_injection, random_removal, ...
            - Cluster Removal: remove_smallest_clusters, ...
            - Embedding Manipulation: embedding_perturbation, ...

    Example:
        >>> types = metricate.list_degradations()
        >>> for category, degradations in types.items():
        ...     print(f"{category}: {len(degradations)} types")
    """
    from metricate.degradation.toolkit import DEGRADATION_TYPES

    return dict(DEGRADATION_TYPES)


def generate_training_data(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    topic: str | None = None,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
):
    """
    Generate a training dataset from a single clustering CSV.

    Creates degraded versions of the input clustering at various intensity
    levels, calculates metrics for both original and degraded versions,
    and returns a structured dataset for ML training.

    Args:
        csv_path: Path to input clustering CSV.
        output_dir: Directory for degraded CSVs and outputs.
        types: Degradation types to apply (None = all 19).
        levels: Degradation levels (None = ["5pct", "10pct", "25pct", "50pct"]).
        exclude: Metrics to exclude from calculation.
        force_all: Compute O(n²) metrics on large datasets.
        topic: Manual topic assignment (None = extract from filename).
        random_seed: Seed for reproducibility.
        label_col: Cluster label column (auto-detected if None).
        embedding_cols: Embedding columns (auto-detected if None).

    Returns:
        TrainingDataResult: Object containing:
            - records: List of record dicts (one per clustering version)
            - to_dataframe(): Convert to pandas DataFrame
            - to_csv(path): Export to CSV file
            - to_parquet(path): Export to Parquet file
            - summary(): Human-readable summary

    Example:
        >>> result = metricate.generate_training_data("clustering.csv", "./output/")
        >>> df = result.to_dataframe()
        >>> result.to_csv("training_dataset.csv")
    """
    from metricate.training.generator import (
        generate_training_data as _generate_training_data,
    )

    return _generate_training_data(
        csv_path,
        output_dir,
        types=types,
        levels=levels,
        exclude=exclude,
        force_all=force_all,
        topic=topic,
        random_seed=random_seed,
        label_col=label_col,
        embedding_cols=embedding_cols,
    )


def generate_training_data_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    topic_mapping: dict[str, str] | None = None,
    types: list[str] | None = None,
    levels: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    random_seed: int = 42,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
):
    """
    Generate training data from all CSVs in a directory.

    Processes each clustering file in the input directory, generating
    degraded versions and calculating metrics, then combines all results
    into a single training dataset.

    Args:
        input_dir: Directory containing clustering CSVs.
        output_dir: Directory for degraded CSVs and outputs.
        topic_mapping: Dict mapping filename to topic.
        types: Degradation types to apply (None = all 19).
        levels: Degradation levels (None = ["5pct", "10pct", "25pct", "50pct"]).
        exclude: Metrics to exclude from calculation.
        force_all: Compute O(n²) metrics on large datasets.
        random_seed: Seed for reproducibility.
        label_col: Cluster label column (auto-detected if None).
        embedding_cols: Embedding columns (auto-detected if None).

    Returns:
        TrainingDataResult: Combined results from all files.

    Example:
        >>> result = metricate.generate_training_data_batch("./clusterings/", "./output/")
        >>> print(result.summary())
        >>> result.to_csv("full_training_dataset.csv")
    """
    from metricate.training.generator import (
        generate_training_data_batch as _generate_training_data_batch,
    )

    return _generate_training_data_batch(
        input_dir,
        output_dir,
        topic_mapping=topic_mapping,
        types=types,
        levels=levels,
        exclude=exclude,
        force_all=force_all,
        random_seed=random_seed,
        label_col=label_col,
        embedding_cols=embedding_cols,
    )


def train_weights(
    csv_path: str | Path,
    *,
    regularization: str = "ridge",
    alpha: float = 1.0,
    auto_alpha: bool = False,
    alphas: list[float] | None = None,
    run_cv: bool = True,
    cv_splits: int = 5,
    run_sanity_check: bool = True,
):
    """
    Train a regression model to learn optimal metric weights for quality scoring.

    Uses Ridge (default) or Lasso regularization to learn coefficients that
    weight each metric's contribution to a compound quality score.

    The learned formula is:
        score = clip(Σ(weight_i × metric_i) + bias, 0, 1)

    Args:
        csv_path: Path to training dataset CSV with normalized metrics and quality_score.
        regularization: Type of regularization ("ridge" or "lasso").
        alpha: Regularization strength (ignored if auto_alpha=True).
        auto_alpha: If True, use cross-validation to select optimal alpha.
        alphas: Candidate alpha values for auto-tuning.
        run_cv: If True, run leave-one-clustering-out cross-validation.
        cv_splits: Number of CV folds (default: 5).
        run_sanity_check: If True, verify original scores > all degraded scores.

    Returns:
        TrainingResult: Object containing:
            - weights: MetricWeights with coefficients and bias
            - feature_importance: Ranked list of (metric, weight) tuples
            - zeroed_metrics: Metrics with zero weight (Lasso only)
            - cv_scores: Cross-validation metrics (R², MAE, RMSE)
            - cv_results: Per-fold CVResult objects
            - sanity_check_passed: True if original > all degraded
            - sanity_failures: List of violations (if any)

    Example:
        >>> import metricate
        >>> result = metricate.train_weights("training_data.csv")
        >>> print(f"CV R²: {result.cv_scores['r2_mean']:.3f}")
        >>> print(f"Sanity check: {'PASS' if result.sanity_check_passed else 'FAIL'}")
        >>> result.weights.save("weights.json")
    """
    from metricate.training.learner import train_weights as _train_weights

    return _train_weights(
        csv_path,
        regularization=regularization,
        alpha=alpha,
        auto_alpha=auto_alpha,
        alphas=alphas,
        run_cv=run_cv,
        cv_splits=cv_splits,
        run_sanity_check=run_sanity_check,
    )


def load_weights(path: str | Path):
    """
    Load learned metric weights from a JSON file.

    Args:
        path: Path to JSON file containing weights.

    Returns:
        MetricWeights: Object containing coefficients, bias, and metadata.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If JSON is invalid or missing required fields.

    Example:
        >>> weights = metricate.load_weights("weights.json")
        >>> result = metricate.evaluate("clustering.csv", weights=weights)
        >>> print(f"Compound score: {result.compound_score:.3f}")
    """
    from metricate.training.weights import load_weights as _load_weights

    return _load_weights(path)


__all__ = [
    "evaluate",
    "compare",
    "degrade",
    "list_metrics",
    "list_degradations",
    "generate_training_data",
    "generate_training_data_batch",
    "train_weights",
    "load_weights",
    "__version__",
]
