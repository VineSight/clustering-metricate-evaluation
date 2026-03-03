"""
Core evaluation engine for calculating clustering metrics.

This module provides the main entry point for evaluating clustering quality
using the 34 implemented metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from metricate.core.exceptions import (
    ComputationError,
    InsufficientClustersError,
)
from metricate.core.loader import load_csv
from metricate.core.metrics import (
    METRIC_FUNCTIONS,
    precompute_all,
)
from metricate.core.reference import (
    LARGE_DATASET_SKIP,
    LARGE_DATASET_THRESHOLD,
    METRIC_REFERENCE,
)
from metricate.output.report import EvaluationResult, MetricValue

if TYPE_CHECKING:
    from metricate.training.weights import MetricWeights


def filter_noise_points(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove noise points (cluster_id=-1) from the data.

    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        labels: Cluster labels array

    Returns:
        Tuple of (filtered_embeddings, filtered_labels, noise_count)
    """
    noise_mask = labels == -1
    noise_count = noise_mask.sum()

    if noise_count > 0:
        valid_mask = ~noise_mask
        return embeddings[valid_mask], labels[valid_mask], int(noise_count)

    return embeddings, labels, 0


def calculate_all_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    exclude: list[str] | None = None,
    force_all: bool = False,
) -> EvaluationResult:
    """
    Calculate all internal clustering metrics for the given data.

    This function handles precomputation of shared values and calls each
    metric function with the appropriate parameters.

    Args:
        X: Embedding matrix (n_samples, n_features)
        labels: Cluster labels array
        exclude: List of metric names to skip
        force_all: If True, compute O(n²) metrics even on large datasets

    Returns:
        EvaluationResult containing all computed metrics
    """
    exclude = set(exclude or [])
    result = EvaluationResult()

    n_samples = len(labels)
    n_clusters = len(np.unique(labels))

    # Add metadata
    result.metadata = {
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "n_clusters": n_clusters,
    }

    # Check if we should auto-skip expensive metrics
    auto_skip = set()
    if n_samples > LARGE_DATASET_THRESHOLD and not force_all:
        auto_skip = set(LARGE_DATASET_SKIP)
        result.add_warning(
            f"Dataset has {n_samples:,} rows (>{LARGE_DATASET_THRESHOLD:,}). "
            f"Auto-skipping O(n²) metrics: {', '.join(sorted(auto_skip))}. "
            "Use force_all=True to override."
        )

    # Warn about unrecognized exclusions
    all_known = set(METRIC_FUNCTIONS.keys())
    unknown = exclude - all_known
    if unknown:
        result.add_warning(
            f"Unrecognized metric names in exclude list: {', '.join(sorted(unknown))}"
        )

    # Precompute shared values
    try:
        precomputed = precompute_all(X, labels)
    except Exception as e:
        raise ComputationError(
            metric="precomputation", message=f"Failed to precompute shared values: {e}"
        )

    # Calculate each metric
    for metric_name, metric_func in METRIC_FUNCTIONS.items():
        # Check if excluded
        if metric_name in exclude:
            result.add_metric(
                MetricValue(
                    metric=metric_name,
                    value=None,
                    computed=False,
                    skip_reason="User excluded",
                )
            )
            continue

        # Check if auto-skipped
        if metric_name in auto_skip:
            result.add_metric(
                MetricValue(
                    metric=metric_name,
                    value=None,
                    computed=False,
                    skip_reason=f"Auto-skipped (n > {LARGE_DATASET_THRESHOLD:,})",
                )
            )
            continue

        # Calculate the metric
        try:
            # Some metrics don't need X/labels (e.g., gamma_index uses precomputed S_plus/S_minus)
            if metric_name in ("Gamma", "G-plus", "Tau"):
                value = metric_func(**precomputed)
            else:
                value = metric_func(X, labels, **precomputed)

            # Handle inf/nan
            if np.isnan(value) or np.isinf(value):
                result.add_metric(
                    MetricValue(
                        metric=metric_name,
                        value=None,
                        computed=False,
                        skip_reason=f"Computation returned {value}",
                    )
                )
            else:
                result.add_metric(
                    MetricValue(
                        metric=metric_name,
                        value=float(value),
                        computed=True,
                    )
                )
        except Exception as e:
            result.add_metric(
                MetricValue(
                    metric=metric_name,
                    value=None,
                    computed=False,
                    skip_reason=f"Error: {str(e)[:50]}",
                )
            )

    return result


def compute_compound_score_from_eval(
    result: EvaluationResult,
    weights: "MetricWeights",
) -> tuple[float, str | None]:
    """
    Compute compound score from evaluation result and weights.

    Normalizes raw metric values to 0-1 range (using direction awareness)
    and computes weighted sum.

    Args:
        result: EvaluationResult with computed metrics
        weights: MetricWeights with coefficients and bias

    Returns:
        Tuple of (compound_score, warning_message).
        Warning is set if metrics are missing.
    """
    from metricate.training.weights import compute_compound_score

    # Build normalized metrics dict
    metrics_norm: dict[str, float] = {}

    for mv in result.computed_metrics():
        metric_name = mv.metric
        value = mv.value

        if value is None:
            continue

        # Get direction from reference
        ref = METRIC_REFERENCE.get(metric_name, {})
        direction = ref.get("direction", "higher")
        range_str = ref.get("range", "[0, ∞)")

        # Normalize to 0-1 range (higher is always better after normalization)
        norm_value = _normalize_metric_value(value, range_str, direction)

        # Add with _norm suffix to match weights coefficients
        metrics_norm[f"{metric_name}_norm"] = norm_value

    # Use weights module to compute score with renormalization for missing metrics
    return compute_compound_score(metrics_norm, weights)


def _normalize_metric_value(value: float, range_str: str, direction: str) -> float:
    """
    Normalize a metric value to 0-1 range where higher is better.

    Args:
        value: Raw metric value
        range_str: Range string from reference (e.g., "[-1, 1]", "[0, ∞)")
        direction: "higher" or "lower" indicating which direction is better

    Returns:
        Normalized value in [0, 1] where higher is better
    """
    # Parse common range patterns
    if range_str == "[-1, 1]":
        # Map [-1, 1] to [0, 1]
        norm = (value + 1) / 2
    elif range_str == "[0, 1]":
        norm = value
    elif range_str in ("[0, ∞)", "(0, ∞)"):
        # For unbounded positive ranges, use sigmoid-like transformation
        # Values around 1 → 0.5, larger values → 1
        norm = value / (1 + value) if value >= 0 else 0.0
    elif range_str in ("(-∞, ∞)", "ℝ"):
        # For fully unbounded, use tanh
        norm = (np.tanh(value) + 1) / 2
    else:
        # Default: assume [0, 1] or clamp
        norm = max(0.0, min(1.0, value))

    # Flip if lower is better so that higher normalized value = better quality
    if direction == "lower":
        norm = 1.0 - norm

    return float(np.clip(norm, 0.0, 1.0))


def compute_unweighted_final_score(result: EvaluationResult) -> tuple[float, str]:
    """
    Compute unweighted average of all normalized metrics.

    ⚠️  WARNING: This is NOT a recommended approach for production use!

    This function normalizes all computed metrics to [0, 1] (where higher is better)
    and returns their simple arithmetic mean. This approach has significant limitations:

    - Equal weighting assumes all metrics are equally important (they are not)
    - Different metrics measure different aspects of clustering quality
    - Some metrics are highly correlated, causing implicit overweighting
    - The normalization for unbounded metrics uses heuristic transformations
    - No validation that the score correlates with actual clustering quality

    For reliable quality scoring, train proper weights using metricate.train_weights()
    with labeled data that represents your use case.

    Args:
        result: EvaluationResult with computed metrics

    Returns:
        Tuple of (final_score, warning_message)
    """
    normalized_values: list[float] = []

    for mv in result.computed_metrics():
        metric_name = mv.metric
        value = mv.value

        if value is None:
            continue

        # Get direction from reference
        ref = METRIC_REFERENCE.get(metric_name, {})
        direction = ref.get("direction", "higher")
        range_str = ref.get("range", "[0, ∞)")

        # Normalize to 0-1 range (higher is always better after normalization)
        norm_value = _normalize_metric_value(value, range_str, direction)
        normalized_values.append(norm_value)

    if not normalized_values:
        return 0.0, "No metrics available for final score computation"

    final_score = float(np.mean(normalized_values))

    # THE BIG SCARY WARNING
    warning = (
        "⚠️  UNWEIGHTED FINAL SCORE - NOT RECOMMENDED FOR PRODUCTION USE ⚠️\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "This score is a simple average of {} normalized metrics.\n"
        "It assumes all metrics are equally important, which is INCORRECT.\n"
        "\n"
        "PROBLEMS WITH THIS APPROACH:\n"
        "  • Metrics are NOT equally important for quality assessment\n"
        "  • Many metrics are correlated → implicit overweighting\n"
        "  • Unbounded metrics use heuristic normalization\n"
        "  • No validation against actual clustering quality\n"
        "\n"
        "FOR RELIABLE SCORING:\n"
        "  Use metricate.train_weights() to learn proper metric weights\n"
        "  from labeled training data for your specific use case.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ).format(len(normalized_values))

    return final_score, warning


def evaluate(
    csv_path: str | Path,
    *,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    weights: "MetricWeights | None" = None,
    final_score: bool = False,
) -> EvaluationResult:
    """
    Evaluate clustering quality for a CSV file.

    This is the main public API for single-file evaluation.

    Args:
        csv_path: Path to the CSV file containing clustering data
        label_col: Name of the cluster label column (auto-detected if not provided)
        embedding_cols: List of embedding column names (auto-detected if not provided)
        exclude: List of metric names to skip
        force_all: If True, compute O(n²) metrics even on large datasets (>50k rows)
        weights: Optional MetricWeights for computing compound score
        final_score: If True and no weights provided, computes an unweighted average
            of all normalized metrics. NOT RECOMMENDED for production use.

    Returns:
        EvaluationResult containing all computed metrics with ranges and directions.
        If weights provided, also includes compound_score and compound_score_warning.
        If final_score=True and no weights, includes final_score and final_score_warning.

    Raises:
        FileNotFoundError: If CSV file does not exist
        InvalidCSVError: If CSV is malformed
        ColumnNotFoundError: If required columns not found
        InsufficientClustersError: If fewer than 2 clusters

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
    # Load and validate data
    data = load_csv(csv_path, label_col=label_col, embedding_cols=embedding_cols)

    # Filter noise points
    X, labels, noise_count = filter_noise_points(data.embeddings, data.labels)

    # Check we still have enough clusters after noise removal
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise InsufficientClustersError(
            required=2,
            actual=n_clusters,
            message="After removing noise points, fewer than 2 clusters remain",
        )

    # Calculate metrics
    result = calculate_all_metrics(X, labels, exclude=exclude, force_all=force_all)

    # Add file metadata
    result.metadata["source_file"] = str(csv_path)
    result.metadata["label_col"] = data.label_col
    result.metadata["embedding_cols"] = data.embedding_cols

    # Add noise warning if applicable
    if noise_count > 0:
        result.add_warning(f"Excluded {noise_count:,} noise points (cluster_id=-1) from evaluation")
        result.metadata["noise_points_excluded"] = noise_count

    # Compute compound score if weights provided
    if weights is not None:
        result.compound_score, result.compound_score_warning = compute_compound_score_from_eval(
            result, weights
        )

    # Compute unweighted final score if requested and no weights provided
    if final_score and weights is None:
        result.final_score, result.final_score_warning = compute_unweighted_final_score(result)

    return result
