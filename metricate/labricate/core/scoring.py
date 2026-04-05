"""Scoring utilities for Labricate weighted evaluation.

Provides compound score computation, best run identification with tie detection,
and weight coverage warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from metricate.core.evaluator import _normalize_metric_value
from metricate.core.reference import METRIC_REFERENCE
from metricate.training.weights import MetricWeights, compute_compound_score

if TYPE_CHECKING:
    from metricate.labricate.core.experiment import RunResult

__all__ = [
    "BestRunInfo",
    "WeightCoverageWarning",
    "compute_run_scores",
    "find_best_run",
    "check_weight_coverage",
]


@dataclass
class BestRunInfo:
    """Information about the best run in an experiment.

    Attributes:
        run_id: ID of the best run.
        param_values: Hyperparameter values that achieved the best score.
        score: The best score value (compound_score or single metric).
        score_type: Type of score used ("compound_score" or metric name).
        tied_run_ids: List of run IDs that tied with the best run (empty if no ties).
    """

    run_id: int
    param_values: dict[str, Any]
    score: float
    score_type: str
    tied_run_ids: list[int]

    def __str__(self) -> str:
        """Return human-readable string with tie information."""
        base = (
            f"BestRunInfo(\n"
            f"    run_id={self.run_id},\n"
            f"    param_values={self.param_values},\n"
            f"    score={self.score:.4f} ({self.score_type})"
        )
        if self.tied_run_ids:
            tied_str = ", ".join(str(rid) for rid in self.tied_run_ids)
            base += f",\n    tied with run_ids=[{tied_str}]"
        base += "\n)"
        return base


class WeightCoverageWarning(UserWarning):
    """Warning when excluded metrics account for significant weight."""

    pass


def compute_run_scores(
    runs: list[RunResult],
    weights: MetricWeights,
) -> None:
    """Compute compound scores for all completed runs.

    Modifies runs in-place by setting their compound_score attribute.
    Failed or skipped runs are left with compound_score=None.

    Args:
        runs: List of RunResult objects from an experiment.
        weights: MetricWeights for computing compound scores.
    """
    for run in runs:
        if run.pipeline_result.status != "completed":
            run.compound_score = None
            continue

        # Build normalized metrics dict from run metrics
        # Normalize raw values to 0-1 range using metric reference, then add _norm suffix
        metrics_norm: dict[str, float] = {}
        for metric in run.metrics:
            # Get normalization info from reference
            ref = METRIC_REFERENCE.get(metric.name, {})
            range_str = ref.get("range", "[0, 1]")
            direction = ref.get("direction", "higher")

            # Normalize the raw metric value to 0-1 range
            norm_value = _normalize_metric_value(metric.value, range_str, direction)

            # Check both with and without _norm suffix in weights
            if metric.name in weights.coefficients:
                metrics_norm[metric.name] = norm_value
            elif f"{metric.name}_norm" in weights.coefficients:
                metrics_norm[f"{metric.name}_norm"] = norm_value

        if not metrics_norm:
            run.compound_score = None
            continue

        try:
            score, _ = compute_compound_score(metrics_norm, weights, warn_on_missing=False)
            run.compound_score = score
        except ValueError:
            run.compound_score = None


def find_best_run(
    runs: list[RunResult],
    weights: MetricWeights | None = None,
    best_metric: str = "Silhouette",
) -> BestRunInfo | None:
    """Find the best run based on compound_score or a single metric.

    Args:
        runs: List of RunResult objects from an experiment.
        weights: If provided, use compound_score. Otherwise use best_metric.
        best_metric: Metric name to use when weights not provided (default: Silhouette).

    Returns:
        BestRunInfo with best run details, or None if no completed runs.
        Ties are detected and reported in tied_run_ids.
    """
    # Filter to completed runs
    completed_runs = [r for r in runs if r.pipeline_result.status == "completed"]
    if not completed_runs:
        return None

    if weights is not None:
        # Use compound_score
        scored_runs = [(r, r.compound_score) for r in completed_runs if r.compound_score is not None]
        if not scored_runs:
            return None

        score_type = "compound_score"
    else:
        # Use single metric
        scored_runs = []
        for run in completed_runs:
            for metric in run.metrics:
                if metric.name == best_metric:
                    scored_runs.append((run, metric.value))
                    break

        if not scored_runs:
            return None

        score_type = best_metric

    # Find max score
    max_score = max(score for _, score in scored_runs)

    # Find all runs with max score (to detect ties)
    best_runs = [(run, score) for run, score in scored_runs if score == max_score]

    # First one is the "best", others are ties
    best_run, best_score = best_runs[0]
    tied_run_ids = [run.run_id for run, _ in best_runs[1:]]

    return BestRunInfo(
        run_id=best_run.run_id,
        param_values=best_run.param_values.copy(),
        score=best_score,
        score_type=score_type,
        tied_run_ids=tied_run_ids,
    )


def check_weight_coverage(
    weights: MetricWeights,
    excluded_metrics: list[str],
    threshold: float = 0.30,
) -> str | None:
    """Check if excluded metrics account for significant weight.

    Issues a warning if excluded metrics represent more than threshold
    of total absolute weight in the weights coefficients.

    Args:
        weights: MetricWeights to check.
        excluded_metrics: List of metric names being excluded.
        threshold: Warning threshold (default 0.30 = 30%).

    Returns:
        Warning message if threshold exceeded, None otherwise.
    """
    coefficients = weights.coefficients

    # Calculate total absolute weight
    total_weight = sum(abs(v) for v in coefficients.values())
    if total_weight == 0:
        return None

    # Calculate excluded absolute weight
    excluded_weight = 0.0
    excluded_found: list[str] = []
    for metric_name in excluded_metrics:
        # Try both with and without _norm suffix
        norm_key = f"{metric_name}_norm" if not metric_name.endswith("_norm") else metric_name
        if norm_key in coefficients:
            excluded_weight += abs(coefficients[norm_key])
            excluded_found.append(metric_name)

    if not excluded_found:
        return None

    # Calculate percentage
    excluded_percentage = excluded_weight / total_weight

    if excluded_percentage > threshold:
        pct_str = f"{excluded_percentage * 100:.0f}%"
        metrics_str = ", ".join(sorted(excluded_found))
        return (
            f"Warning: Excluded metrics [{metrics_str}] account for {pct_str} "
            f"of weight coefficients. Compound scores may be unreliable."
        )

    return None
