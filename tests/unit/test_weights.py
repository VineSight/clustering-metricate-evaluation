"""
Unit tests for metricate.training.weights module.

Tests the MetricWeights dataclass, load/save functionality,
and compound score computation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from metricate.training.weights import (
    MetricWeights,
    compute_compound_score,
    load_weights,
)


# ============================================================================
# MetricWeights dataclass tests
# ============================================================================


class TestMetricWeightsInit:
    """Tests for MetricWeights initialization and validation."""

    def test_valid_initialization(self):
        """MetricWeights initializes with valid coefficients."""
        weights = MetricWeights(
            coefficients={"Silhouette_norm": 0.5, "Davies-Bouldin_norm": 0.3},
            bias=0.1,
        )
        assert weights.coefficients["Silhouette_norm"] == 0.5
        assert weights.bias == 0.1
        assert weights.version == "1.0"

    def test_empty_coefficients_raises(self):
        """MetricWeights raises ValueError for empty coefficients."""
        with pytest.raises(ValueError, match="at least one metric"):
            MetricWeights(coefficients={}, bias=0.0)

    def test_invalid_coefficient_keys_raises(self):
        """MetricWeights raises ValueError for keys not ending in _norm."""
        with pytest.raises(ValueError, match="_norm"):
            MetricWeights(
                coefficients={"Silhouette": 0.5},  # Missing _norm suffix
                bias=0.0,
            )


class TestMetricWeightsSerialization:
    """Tests for MetricWeights to_dict, to_json, save methods."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample weights for testing."""
        return MetricWeights(
            coefficients={
                "Silhouette_norm": 0.15,
                "Davies-Bouldin_norm": 0.10,
                "R-squared_norm": 0.08,
            },
            bias=0.25,
            regularization="ridge",
            alpha=1.0,
            cv_r2=0.85,
            training_samples=74,
        )

    def test_to_dict(self, sample_weights):
        """to_dict returns complete dictionary representation."""
        # TODO: Implement test after to_dict is implemented
        pass

    def test_to_json(self, sample_weights):
        """to_json returns valid JSON string."""
        # TODO: Implement test after to_json is implemented
        pass

    def test_save_and_load_roundtrip(self, sample_weights, tmp_path):
        """Weights can be saved and loaded back identically."""
        # TODO: Implement test after save/load are implemented
        pass


# ============================================================================
# load_weights tests
# ============================================================================


class TestLoadWeights:
    """Tests for load_weights function."""

    def test_load_valid_json(self, tmp_path):
        """load_weights loads valid JSON file."""
        # TODO: Implement test after load_weights is implemented
        pass

    def test_load_missing_file_raises(self):
        """load_weights raises FileNotFoundError for missing file."""
        # TODO: Implement test after load_weights is implemented
        pass

    def test_load_invalid_json_raises(self, tmp_path):
        """load_weights raises ValueError for invalid JSON."""
        # TODO: Implement test after load_weights is implemented
        pass


# ============================================================================
# compute_compound_score tests
# ============================================================================


class TestComputeCompoundScore:
    """Tests for compute_compound_score function."""

    @pytest.fixture
    def weights(self):
        """Create weights for testing."""
        return MetricWeights(
            coefficients={
                "Silhouette_norm": 0.4,
                "Davies-Bouldin_norm": 0.3,
                "R-squared_norm": 0.2,
            },
            bias=0.1,
        )

    def test_all_metrics_present(self, weights):
        """Compound score computed correctly with all metrics."""
        # TODO: Implement test after compute_compound_score is implemented
        pass

    def test_missing_metrics_renormalized(self, weights):
        """Missing metrics trigger renormalization and warning."""
        # TODO: Implement test after compute_compound_score is implemented
        pass

    def test_score_clipped_to_bounds(self, weights):
        """Score is clipped to [0, 1] range."""
        # TODO: Implement test after compute_compound_score is implemented
        pass
