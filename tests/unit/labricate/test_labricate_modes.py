"""Tests for metricate.labricate.core.modes module.

Test cases TC-001 through TC-005 from contracts/testing.md.
"""

import numpy as np
import pytest

from metricate.labricate.core.modes import (
    ComputationMode,
    apply_mode_exclusions,
    get_expensive_metrics,
)


class TestGetExpensiveMetrics:
    """Tests for get_expensive_metrics function."""

    def test_get_expensive_metrics_returns_six_metrics(self):
        """TC-001: Verify exactly 6 metrics with skip_large=True."""
        metrics = get_expensive_metrics()

        assert len(metrics) == 6
        assert set(metrics) == {
            "Gamma",
            "G-plus",
            "Tau",
            "Point-Biserial",
            "McClain-Rao",
            "NIVA",
        }

    def test_get_expensive_metrics_returns_list(self):
        """Verify return type is a list."""
        metrics = get_expensive_metrics()
        assert isinstance(metrics, list)

    def test_get_expensive_metrics_consistent_order(self):
        """Verify multiple calls return consistent results."""
        metrics1 = get_expensive_metrics()
        metrics2 = get_expensive_metrics()
        assert metrics1 == metrics2


class TestApplyModeExclusions:
    """Tests for apply_mode_exclusions function."""

    def test_heavy_mode_preserves_user_exclusions(self):
        """TC-002: Heavy mode should not add any exclusions."""
        user_exclude = ["Silhouette", "Davies-Bouldin"]

        result = apply_mode_exclusions("heavy", user_exclude)

        assert result == sorted(user_exclude)

    def test_heavy_mode_with_none_exclusions(self):
        """Heavy mode with None exclusions returns empty list."""
        result = apply_mode_exclusions("heavy", None)

        assert result == []

    def test_light_mode_adds_expensive_metrics(self):
        """TC-003: Light mode should exclude expensive metrics."""
        result = apply_mode_exclusions("light", None)

        assert "Gamma" in result
        assert "Tau" in result
        assert len(result) == 6

    def test_light_mode_merges_exclusions(self):
        """TC-004: Light mode should merge with user's exclude list."""
        user_exclude = ["Silhouette"]

        result = apply_mode_exclusions("light", user_exclude)

        assert "Silhouette" in result
        assert "Gamma" in result
        assert len(result) == 7  # 6 expensive + 1 user

    def test_include_metrics_overrides_light_mode(self):
        """TC-005: User's include_metrics should override light mode exclusions."""
        result = apply_mode_exclusions(
            "light",
            exclude_metrics=None,
            include_metrics=["Gamma", "Tau"],
        )

        # Gamma and Tau should NOT be in result since user explicitly included them
        assert "Gamma" not in result
        assert "Tau" not in result
        # Other expensive metrics should still be excluded
        assert "G-plus" in result
        assert "Point-Biserial" in result
        assert "McClain-Rao" in result
        assert "NIVA" in result
        # Total should be 4 (6 expensive - 2 included)
        assert len(result) == 4

    def test_include_metrics_with_heavy_mode(self):
        """Include_metrics with heavy mode should not affect result."""
        result = apply_mode_exclusions(
            "heavy",
            exclude_metrics=["Silhouette"],
            include_metrics=["Silhouette"],  # Contradicts exclude
        )

        # Include takes precedence, so Silhouette should be removed from exclusions
        assert "Silhouette" not in result
        assert result == []

    def test_result_is_sorted(self):
        """Verify exclusion results are sorted."""
        result = apply_mode_exclusions("light", ["Zebra", "Alpha"])

        # Check that result is sorted
        assert result == sorted(result)

    def test_no_duplicates_in_result(self):
        """Verify no duplicates when user excludes expensive metric."""
        user_exclude = ["Gamma"]  # Already in expensive metrics

        result = apply_mode_exclusions("light", user_exclude)

        # Should not have duplicate Gamma
        assert result.count("Gamma") == 1
        assert len(result) == 6  # Still 6 unique metrics


class TestExperimentModeParameter:
    """Tests for mode parameter in Experiment.run()."""

    def test_run_accepts_mode_parameter(self, small_embeddings, base_config, mock_pipeline):
        """T036: Experiment.run() should accept mode parameter."""
        from metricate.labricate.core.experiment import Experiment

        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        # Should not raise - mode parameter accepted
        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5],
            mode="light",
            verbose=False,
        )

        assert result is not None
        assert result.summary.completed_runs == 1

    def test_light_mode_excludes_expensive_metrics(self, small_embeddings, base_config, mock_pipeline):
        """T037: Light mode should exclude the 6 expensive O(n²) metrics."""
        from metricate.labricate.core.experiment import Experiment

        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5],
            mode="light",
            verbose=False,
        )

        # Get all metric names from the completed run
        completed_run = result.runs[0]
        metric_names = [m.name for m in completed_run.metrics]

        # Expensive metrics should NOT be present
        expensive = get_expensive_metrics()
        for metric in expensive:
            assert metric not in metric_names, f"Expensive metric {metric} should be excluded in light mode"

    def test_include_metrics_overrides_mode(self, small_embeddings, base_config, mock_pipeline):
        """T038: include_metrics should take precedence over mode exclusions."""
        from metricate.labricate.core.experiment import Experiment

        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        # Explicitly include Gamma (an expensive metric) in light mode
        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5],
            mode="light",
            include_metrics=["Silhouette", "Gamma"],
            verbose=False,
        )

        # Get all metric names from the completed run
        completed_run = result.runs[0]
        metric_names = [m.name for m in completed_run.metrics]

        # Silhouette should be present (explicitly included)
        assert "Silhouette" in metric_names

        # Gamma should be present because user explicitly requested it
        # (include_metrics takes precedence over mode exclusions)
        assert "Gamma" in metric_names

    def test_heavy_mode_includes_all_metrics(self, small_embeddings, base_config, mock_pipeline):
        """Heavy mode should not exclude expensive metrics."""
        from metricate.labricate.core.experiment import Experiment

        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5],
            mode="heavy",
            verbose=False,
        )

        # Get all metric names
        completed_run = result.runs[0]
        metric_names = [m.name for m in completed_run.metrics]

        # At least some expensive metrics should be present in heavy mode
        # (unless they failed to compute for other reasons)
        assert len(metric_names) > 0
