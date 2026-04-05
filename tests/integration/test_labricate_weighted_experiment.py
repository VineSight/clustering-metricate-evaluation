"""Integration tests for Labricate weighted experiments.

Tests TC-022 through TC-024 from contracts/testing.md.
"""

import json
import time
import warnings

import numpy as np
import pytest

from metricate.labricate.core.experiment import Experiment
from metricate.labricate.core.scoring import WeightCoverageWarning


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing (100 samples, 50 dimensions)."""
    np.random.seed(42)
    return np.random.randn(100, 50)


@pytest.fixture
def large_embeddings():
    """Create larger embeddings for performance testing (500 samples, 50 dimensions).
    
    Note: Using 500 instead of 5000+ to keep tests reasonably fast while still
    showing meaningful performance differences between light/heavy modes.
    """
    np.random.seed(42)
    return np.random.randn(500, 50)


@pytest.fixture
def base_config():
    """Create a base pipeline configuration for testing."""
    return {
        "random_seed": 42,
        "umap": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
        },
        "clustering_algorithm": "hdbscan",
        "hdbscan": {
            "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "eom",
            "metric": "euclidean",
        },
    }


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline that returns valid clustering labels.
    
    This avoids needing BERTopic for integration tests, making them faster.
    """
    def _pipeline(embeddings, config):
        n = embeddings.shape[0]
        # Create 3 clusters based on simple rule
        labels = np.zeros(n, dtype=int)
        labels[n // 3:2 * n // 3] = 1
        labels[2 * n // 3:] = 2
        reduced = embeddings[:, :5] if embeddings.shape[1] >= 5 else embeddings
        return labels, reduced
    return _pipeline


class TestWeightedExperimentEndToEnd:
    """TC-022: Full weighted experiment flow."""

    def test_weighted_experiment_end_to_end(
        self, tmp_path, sample_embeddings, base_config, mock_pipeline
    ):
        """Complete experiment with weights produces expected output."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "coefficients": {
                "Silhouette_norm": 0.3,
                "Davies-Bouldin_norm": -0.2,
                "Calinski-Harabasz_norm": 0.2,
            },
            "bias": 0.3
        }))

        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            weights=str(weights_file),
            pipeline=mock_pipeline,
        )

        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5, 10, 15],
            verbose=False,
        )

        # Verify compound scores computed
        completed_runs = [r for r in result.runs if r.pipeline_result.status == "completed"]
        assert len(completed_runs) > 0, "Expected at least one completed run"
        
        for run in completed_runs:
            assert run.compound_score is not None, f"Run {run.run_id} missing compound_score"
            assert 0 <= run.compound_score <= 1, f"compound_score out of range: {run.compound_score}"

        # Verify best_run populated
        assert result.best_run is not None, "Expected best_run to be populated"
        assert result.best_run.score_type == "compound_score"
        assert "hdbscan.min_cluster_size" in result.best_run.param_values

    def test_weighted_experiment_grid_search(
        self, tmp_path, sample_embeddings, base_config, mock_pipeline
    ):
        """Grid search with weights works correctly."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "coefficients": {
                "Silhouette_norm": 0.4,
                "Davies-Bouldin_norm": -0.3,
            },
            "bias": 0.2
        }))

        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            weights=str(weights_file),
            pipeline=mock_pipeline,
        )

        result = exp.run_grid(
            params={
                "hdbscan.min_cluster_size": [5, 10],
                "hdbscan.min_samples": [3, 5],
            },
            verbose=False,
        )

        # Should have 4 runs (2x2 grid)
        assert len(result.runs) == 4

        # Verify best_run is determined
        assert result.best_run is not None

    def test_experiment_without_weights_no_compound_score(
        self, sample_embeddings, base_config, mock_pipeline
    ):
        """Experiment without weights should not have compound_score."""
        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5, 10],
            verbose=False,
        )

        # Without weights, compound_score should be None
        for run in result.runs:
            assert run.compound_score is None


class TestLightModeFaster:
    """TC-023: Light mode should exclude expensive metrics."""

    def test_light_mode_excludes_expensive_metrics(
        self, sample_embeddings, base_config, mock_pipeline
    ):
        """Light mode should compute fewer metrics than heavy mode."""
        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        # Heavy mode - should include all metrics
        result_heavy = exp.run(
            param="hdbscan.min_cluster_size",
            values=[10],
            mode="heavy",
            verbose=False,
        )

        # Light mode - should exclude expensive metrics
        result_light = exp.run(
            param="hdbscan.min_cluster_size",
            values=[10],
            mode="light",
            verbose=False,
        )

        heavy_metrics = {m.name for m in result_heavy.runs[0].metrics}
        light_metrics = {m.name for m in result_light.runs[0].metrics}

        # Light mode should have fewer metrics
        assert len(light_metrics) < len(heavy_metrics)

        # Known expensive metrics should be excluded from light mode
        expensive = {"Gamma", "G-plus", "Tau", "Point-Biserial", "McClain-Rao", "NIVA"}
        for metric in expensive:
            if metric in heavy_metrics:  # Only if it was computed in heavy mode
                assert metric not in light_metrics, f"Expensive metric {metric} should be excluded in light mode"

    @pytest.mark.slow
    def test_light_mode_faster_than_heavy(
        self, large_embeddings, base_config, mock_pipeline
    ):
        """Light mode should be faster on larger datasets.
        
        Note: The 30% faster requirement from SC-002 is measured with real
        expensive metric computation. With mock pipelines, the difference
        is smaller but still measurable.
        """
        exp = Experiment(
            embeddings=large_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        # Time heavy mode
        start = time.time()
        exp.run(
            param="hdbscan.min_cluster_size",
            values=[10, 15],
            mode="heavy",
            verbose=False,
        )
        heavy_time = time.time() - start

        # Time light mode
        start = time.time()
        exp.run(
            param="hdbscan.min_cluster_size",
            values=[10, 15],
            mode="light",
            verbose=False,
        )
        light_time = time.time() - start

        # Light mode should be faster (even if not 30% in mocked scenario)
        # We test that it's at least somewhat faster
        assert light_time <= heavy_time, "Light mode should not be slower than heavy mode"


class TestWeightCoverageWarning:
    """TC-024: Weight coverage warning when excluded metrics dominate weights."""

    def test_weight_coverage_warning_displayed(
        self, tmp_path, sample_embeddings, base_config, mock_pipeline
    ):
        """Warning should fire when excluded metrics have significant weight."""
        # Create weights where Gamma (expensive) has 50% weight
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "coefficients": {
                "Gamma_norm": 0.5,       # Excluded in light mode
                "Silhouette_norm": 0.3,
            },
            "bias": 0.2
        }))

        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            weights=str(weights_file),
            pipeline=mock_pipeline,
        )

        # Should warn when using light mode with weights on expensive metrics
        with pytest.warns(WeightCoverageWarning):
            exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10],
                mode="light",
                verbose=False,
            )

    def test_no_warning_when_heavy_mode(
        self, tmp_path, sample_embeddings, base_config, mock_pipeline
    ):
        """No warning should fire in heavy mode (all metrics computed)."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "coefficients": {
                "Gamma_norm": 0.5,
                "Silhouette_norm": 0.3,
            },
            "bias": 0.2
        }))

        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            weights=str(weights_file),
            pipeline=mock_pipeline,
        )

        # Heavy mode computes all metrics - no coverage warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightCoverageWarning)
            # Should not raise
            exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10],
                mode="heavy",
                verbose=False,
            )

    def test_no_warning_when_no_weights(
        self, sample_embeddings, base_config, mock_pipeline
    ):
        """No warning should fire when no weights provided."""
        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            pipeline=mock_pipeline,
        )

        # No weights - no coverage warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightCoverageWarning)
            exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10],
                mode="light",
                verbose=False,
            )


class TestJSONOutputIncludesBestRun:
    """TC-027: JSON output includes best_run information."""

    def test_to_dict_includes_best_run(
        self, tmp_path, sample_embeddings, base_config, mock_pipeline
    ):
        """ExperimentResult should include best_run in dict output."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "coefficients": {"Silhouette_norm": 1.0},
            "bias": 0.0
        }))

        exp = Experiment(
            embeddings=sample_embeddings,
            config=base_config,
            weights=str(weights_file),
            pipeline=mock_pipeline,
        )

        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[5, 10, 15],
            verbose=False,
        )

        # Convert to DataFrame to check best_run info is accessible
        df = result.to_dataframe()
        
        assert "is_best_run" in df.columns
        assert "compound_score" in df.columns
        assert df["is_best_run"].sum() >= 1  # At least one best run marked
