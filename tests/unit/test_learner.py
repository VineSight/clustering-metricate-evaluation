"""
Unit tests for metricate.training.learner module.

Tests the training pipeline, cross-validation, and TrainingResult.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metricate.training.learner import (
    CVResult,
    TrainingResult,
    cross_validate_weights,
    train_weights,
)


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def sample_training_csv(tmp_path):
    """Create a minimal training CSV for testing."""
    # Create synthetic training data with normalized metrics
    data = {
        "clustering_name": [
            "test_original",
            "test_label_swap_5pct",
            "test_label_swap_10pct",
            "test_label_swap_25pct",
        ],
        "quality_score": [1.0, 0.95, 0.90, 0.75],
        "Silhouette_norm": [0.9, 0.85, 0.7, 0.5],
        "Davies-Bouldin_norm": [0.85, 0.8, 0.65, 0.45],
        "R-squared_norm": [0.95, 0.9, 0.8, 0.6],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "training_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# TrainingResult tests
# ============================================================================


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_dataclass_fields(self):
        """TrainingResult has expected fields."""
        # Just verify the dataclass structure exists
        from metricate.training.weights import MetricWeights

        weights = MetricWeights(
            coefficients={"Silhouette_norm": 0.5},
            bias=0.1,
        )
        result = TrainingResult(weights=weights)
        assert result.weights == weights
        assert result.cv_scores == {}
        assert result.feature_importance == []
        assert result.sanity_check_passed is True


class TestCVResult:
    """Tests for CVResult dataclass."""

    def test_dataclass_fields(self):
        """CVResult has expected fields."""
        result = CVResult(
            fold=0,
            held_out_group="test_clustering",
            train_size=60,
            test_size=14,
            r2=0.85,
            rmse=0.08,
            mae=0.06,
        )
        assert result.fold == 0
        assert result.r2 == 0.85


# ============================================================================
# train_weights tests
# ============================================================================


class TestTrainWeights:
    """Tests for train_weights function."""

    def test_train_ridge_basic(self, sample_training_csv):
        """train_weights with Ridge produces valid result."""
        # TODO: Implement test after train_weights is implemented
        pass

    def test_train_lasso_produces_zeros(self, sample_training_csv):
        """train_weights with Lasso zeros some coefficients."""
        # TODO: Implement test after train_weights is implemented
        pass

    def test_feature_importance_sorted(self, sample_training_csv):
        """feature_importance sorted by absolute value descending."""
        # TODO: Implement test after train_weights is implemented
        pass

    def test_missing_file_raises(self):
        """train_weights raises FileNotFoundError for missing file."""
        # TODO: Implement test after train_weights is implemented
        pass


# ============================================================================
# cross_validate_weights tests
# ============================================================================


class TestCrossValidateWeights:
    """Tests for cross_validate_weights function."""

    def test_groupkfold_respects_groups(self, sample_training_csv):
        """CV folds respect clustering groups."""
        # TODO: Implement test after cross_validate_weights is implemented
        pass

    def test_returns_per_fold_metrics(self, sample_training_csv):
        """Returns CVResult for each fold."""
        # TODO: Implement test after cross_validate_weights is implemented
        pass


# ============================================================================
# Private function tests
# ============================================================================


class TestExtractFeatures:
    """Tests for _extract_features helper."""

    def test_extracts_norm_columns_only(self):
        """Only *_norm columns are extracted as features."""
        # TODO: Implement test after _extract_features is implemented
        pass


class TestExtractGroups:
    """Tests for _extract_groups helper."""

    def test_parses_base_clustering_name(self):
        """Extracts base name from clustering_name column."""
        # TODO: Implement test after _extract_groups is implemented
        pass
