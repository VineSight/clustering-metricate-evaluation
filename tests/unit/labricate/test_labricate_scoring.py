"""Tests for metricate.labricate.core.scoring module.

Test cases TC-006 through TC-014 from contracts/testing.md.
"""

from typing import Any

import numpy as np
import pytest

from metricate.labricate.core.experiment import (
    MetricResult,
    PipelineResult,
    RunResult,
)
from metricate.labricate.core.scoring import (
    BestRunInfo,
    WeightCoverageWarning,
    check_weight_coverage,
    compute_run_scores,
    find_best_run,
)
from metricate.labricate.utils.logging import TimingInfo
from metricate.training.weights import MetricWeights


def make_timing_info() -> TimingInfo:
    """Create a dummy TimingInfo for tests."""
    return TimingInfo(
        bertopic_seconds=0.5,
        evaluation_seconds=0.3,
        total_seconds=1.0,
    )


def make_pipeline_result(
    run_id: int = 1,
    status: str = "completed",
    error: str | None = None,
) -> PipelineResult:
    """Create a PipelineResult for tests."""
    return PipelineResult(
        run_id=run_id,
        config={"test": True},
        labels=np.array([0, 0, 1, 1, 2, 2]),
        reduced_embeddings=np.random.randn(6, 5),
        n_clusters=3,
        n_noise=0,
        timing=make_timing_info(),
        status=status,
        error=error,
    )


def make_run_result(
    run_id: int = 1,
    param_values: dict[str, Any] | None = None,
    silhouette: float = 0.5,
    davies_bouldin: float = 1.0,
    compound_score: float | None = None,
    status: str = "completed",
) -> RunResult:
    """Factory for RunResult in tests."""
    if param_values is None:
        param_values = {"hdbscan.min_cluster_size": 10}

    metrics = [
        MetricResult(name="Silhouette", value=silhouette, direction="higher"),
        MetricResult(name="Davies-Bouldin", value=davies_bouldin, direction="lower"),
    ]

    run = RunResult(
        run_id=run_id,
        param_values=param_values,
        pipeline_result=make_pipeline_result(run_id, status),
        metrics=metrics,
        compound_score=compound_score,
    )
    return run


def make_failed_run_result(run_id: int = 1) -> RunResult:
    """Factory for failed RunResult."""
    return RunResult(
        run_id=run_id,
        param_values={"hdbscan.min_cluster_size": 10},
        pipeline_result=make_pipeline_result(run_id, status="failed", error="Test error"),
        metrics=[],
        compound_score=None,
    )


class TestComputeRunScores:
    """Tests for compute_run_scores function."""

    def test_compute_run_scores_sets_compound_score(self):
        """TC-006: Verify compound_score is computed for completed runs."""
        weights = MetricWeights(
            coefficients={"Silhouette_norm": 0.5, "Davies-Bouldin_norm": -0.3},
            bias=0.2,
        )
        runs = [make_run_result(run_id=1, silhouette=0.8, davies_bouldin=0.4)]

        compute_run_scores(runs, weights)

        assert runs[0].compound_score is not None
        assert 0 <= runs[0].compound_score <= 1

    def test_compute_run_scores_skips_failed_runs(self):
        """TC-007: Failed runs should not have compound_score."""
        weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)
        runs = [make_failed_run_result(run_id=1)]

        compute_run_scores(runs, weights)

        assert runs[0].compound_score is None

    def test_compute_run_scores_handles_missing_metrics(self):
        """Runs missing required metrics should have None compound_score."""
        # Weights expect metrics not in run
        weights = MetricWeights(
            coefficients={"Unknown_metric_norm": 1.0},
            bias=0,
        )
        runs = [make_run_result(run_id=1)]

        compute_run_scores(runs, weights)

        assert runs[0].compound_score is None

    def test_compute_run_scores_multiple_runs(self):
        """Multiple runs should all get scores computed."""
        weights = MetricWeights(
            coefficients={"Silhouette_norm": 1.0},
            bias=0,
        )
        runs = [
            make_run_result(run_id=1, silhouette=0.5),
            make_run_result(run_id=2, silhouette=0.7),
            make_failed_run_result(run_id=3),
        ]

        compute_run_scores(runs, weights)

        assert runs[0].compound_score is not None
        assert runs[1].compound_score is not None
        assert runs[2].compound_score is None


class TestFindBestRun:
    """Tests for find_best_run function."""

    def test_find_best_run_uses_compound_score(self):
        """TC-008: Best run should be highest compound_score when weights provided."""
        weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)
        runs = [
            make_run_result(run_id=1, compound_score=0.7),
            make_run_result(run_id=2, compound_score=0.9),
            make_run_result(run_id=3, compound_score=0.8),
        ]

        best = find_best_run(runs, weights)

        assert best is not None
        assert best.run_id == 2
        assert best.score == 0.9
        assert best.score_type == "compound_score"

    def test_find_best_run_uses_metric_without_weights(self):
        """TC-009: Best run should use specified metric when no weights."""
        runs = [
            make_run_result(run_id=1, silhouette=0.6),
            make_run_result(run_id=2, silhouette=0.8),
        ]

        best = find_best_run(runs, weights=None, best_metric="Silhouette")

        assert best is not None
        assert best.run_id == 2
        assert best.score_type == "Silhouette"

    def test_find_best_run_detects_ties(self):
        """TC-010: Ties should be reported in tied_run_ids."""
        runs = [
            make_run_result(run_id=1, compound_score=0.8),
            make_run_result(run_id=2, compound_score=0.9),
            make_run_result(run_id=3, compound_score=0.9),  # Tie with run 2
        ]
        weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)

        best = find_best_run(runs, weights)

        assert best is not None
        assert best.run_id == 2  # First with max score
        assert 3 in best.tied_run_ids

    def test_find_best_run_includes_param_values(self):
        """TC-011: Best run must include hyperparameter values."""
        runs = [
            make_run_result(
                run_id=1,
                param_values={"hdbscan.min_cluster_size": 15},
                compound_score=0.9,
            )
        ]
        weights = MetricWeights(coefficients={"Silhouette_norm": 1.0}, bias=0)

        best = find_best_run(runs, weights)

        assert best is not None
        assert best.param_values == {"hdbscan.min_cluster_size": 15}

    def test_find_best_run_returns_none_when_all_failed(self):
        """TC-012: No completed runs should return None."""
        runs = [make_failed_run_result(run_id=1)]

        best = find_best_run(runs, weights=None)

        assert best is None

    def test_find_best_run_returns_none_for_empty_list(self):
        """Empty run list should return None."""
        best = find_best_run([], weights=None)

        assert best is None

    def test_find_best_run_no_ties_empty_tied_run_ids(self):
        """When no ties, tied_run_ids should be empty."""
        runs = [
            make_run_result(run_id=1, silhouette=0.5),
            make_run_result(run_id=2, silhouette=0.8),
            make_run_result(run_id=3, silhouette=0.6),
        ]

        best = find_best_run(runs, weights=None, best_metric="Silhouette")

        assert best is not None
        assert best.run_id == 2
        assert best.tied_run_ids == []


class TestCheckWeightCoverage:
    """Tests for check_weight_coverage function."""

    def test_check_weight_coverage_warns_above_threshold(self):
        """TC-013: Warning when excluded metrics > 30% of weight."""
        weights = MetricWeights(
            coefficients={
                "Gamma_norm": 0.4,  # Will be excluded
                "Silhouette_norm": 0.6,
            },
            bias=0,
        )

        warning = check_weight_coverage(weights, ["Gamma"], threshold=0.30)

        assert warning is not None
        assert "40%" in warning
        assert "Gamma" in warning

    def test_check_weight_coverage_no_warning_below_threshold(self):
        """TC-014: No warning when excluded metrics < 30% of weight."""
        weights = MetricWeights(
            coefficients={
                "Gamma_norm": 0.2,  # Will be excluded
                "Silhouette_norm": 0.8,
            },
            bias=0,
        )

        warning = check_weight_coverage(weights, ["Gamma"], threshold=0.30)

        assert warning is None

    def test_check_weight_coverage_handles_norm_suffix(self):
        """Should work with or without _norm suffix in excluded_metrics."""
        weights = MetricWeights(
            coefficients={
                "Gamma_norm": 0.5,
                "Silhouette_norm": 0.5,
            },
            bias=0,
        )

        # Without _norm suffix
        warning = check_weight_coverage(weights, ["Gamma"], threshold=0.30)
        assert warning is not None
        assert "50%" in warning

    def test_check_weight_coverage_no_matching_excluded(self):
        """No warning when excluded metrics not in weights."""
        weights = MetricWeights(
            coefficients={"Silhouette_norm": 1.0},
            bias=0,
        )

        warning = check_weight_coverage(weights, ["Gamma"], threshold=0.30)

        assert warning is None

    def test_check_weight_coverage_multiple_excluded(self):
        """Should sum weights of multiple excluded metrics."""
        weights = MetricWeights(
            coefficients={
                "Gamma_norm": 0.2,
                "Tau_norm": 0.2,
                "Silhouette_norm": 0.6,
            },
            bias=0,
        )

        warning = check_weight_coverage(weights, ["Gamma", "Tau"], threshold=0.30)

        assert warning is not None
        assert "40%" in warning

    def test_check_weight_coverage_zero_total_weight(self):
        """Should return None when all weights are zero."""
        weights = MetricWeights(
            coefficients={"Silhouette_norm": 0.0},
            bias=0,
        )

        warning = check_weight_coverage(weights, ["Silhouette"], threshold=0.30)

        assert warning is None


class TestBestRunInfo:
    """Tests for BestRunInfo dataclass."""

    def test_best_run_info_str_no_ties(self):
        """String representation without ties."""
        info = BestRunInfo(
            run_id=1,
            param_values={"n_clusters": 5},
            score=0.85,
            score_type="compound_score",
            tied_run_ids=[],
        )

        result = str(info)

        assert "run_id=1" in result
        assert "0.8500" in result
        assert "compound_score" in result
        assert "tied" not in result.lower()

    def test_best_run_info_str_with_ties(self):
        """String representation with ties."""
        info = BestRunInfo(
            run_id=1,
            param_values={"n_clusters": 5},
            score=0.85,
            score_type="compound_score",
            tied_run_ids=[2, 3],
        )

        result = str(info)

        assert "run_id=1" in result
        assert "tied with run_ids=[2, 3]" in result


class TestWeightCoverageWarning:
    """Tests for WeightCoverageWarning class."""

    def test_is_user_warning_subclass(self):
        """WeightCoverageWarning should be a UserWarning subclass."""
        assert issubclass(WeightCoverageWarning, UserWarning)

    def test_can_raise_warning(self):
        """Can be raised as a warning."""
        with pytest.warns(WeightCoverageWarning):
            import warnings

            warnings.warn("Test warning", WeightCoverageWarning, stacklevel=2)
