"""Tests for weights integration in Labricate experiments.

Test cases TC-015 through TC-018 from contracts/testing.md.
"""

import json

import pytest

from metricate.labricate.core.experiment import Experiment


class TestExperimentWeightsInit:
    """Tests for Experiment weights parameter in __init__."""

    def test_experiment_accepts_weights_path(self, tmp_path, small_embeddings, base_config):
        """TC-015: Experiment should load weights from file path."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(
            json.dumps({"coefficients": {"Silhouette_norm": 1.0}, "bias": 0.0})
        )

        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            weights=str(weights_file),
        )

        assert exp._weights is not None
        assert exp._weights.coefficients["Silhouette_norm"] == 1.0

    def test_experiment_accepts_weights_dict(self, small_embeddings, base_config):
        """TC-016: Experiment should accept weights as dict."""
        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            weights={"coefficients": {"Silhouette_norm": 0.5}, "bias": 0.3},
        )

        assert exp._weights is not None
        assert exp._weights.bias == 0.3

    def test_experiment_validates_weights_schema(self, small_embeddings, base_config):
        """TC-017: Invalid weights should raise ValueError with clear message."""
        with pytest.raises(ValueError, match="coefficients"):
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights={"bias": 0.5},  # Missing coefficients
            )

    def test_experiment_missing_coefficients_error_message(self, small_embeddings, base_config):
        """T045: Missing coefficients error should include actionable guidance."""
        with pytest.raises(ValueError) as exc_info:
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights={"bias": 0.5},  # Missing coefficients
            )

        error_message = str(exc_info.value)
        # Error should mention what's missing
        assert "coefficients" in error_message.lower()
        # Error should be clear and actionable
        assert "missing" in error_message.lower() or "required" in error_message.lower()

    def test_experiment_missing_bias_error_message(self, small_embeddings, base_config):
        """Missing bias error should include actionable guidance."""
        with pytest.raises(ValueError) as exc_info:
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights={"coefficients": {"Silhouette_norm": 0.5}},  # Missing bias
            )

        error_message = str(exc_info.value)
        # Error should mention bias is missing
        assert "bias" in error_message.lower()

    def test_experiment_invalid_coefficient_value_error(self, small_embeddings, base_config):
        """T046: Non-numeric coefficient values should raise clear error."""
        with pytest.raises(ValueError) as exc_info:
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights={
                    "coefficients": {"Silhouette_norm": "not_a_number"},
                    "bias": 0.5
                },
            )

        error_message = str(exc_info.value)
        # Error should indicate the issue with coefficient value
        assert "coefficient" in error_message.lower() or "numeric" in error_message.lower()

    def test_experiment_empty_coefficients_error(self, small_embeddings, base_config):
        """Empty coefficients dict should raise clear error."""
        with pytest.raises(ValueError) as exc_info:
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights={"coefficients": {}, "bias": 0.5},
            )

        error_message = str(exc_info.value)
        # Error should indicate empty/no coefficients
        assert "coefficient" in error_message.lower()
        assert "empty" in error_message.lower() or "at least one" in error_message.lower()

    def test_experiment_without_weights_unchanged(self, small_embeddings, base_config):
        """TC-018: Existing behavior should work when weights not provided."""
        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
        )

        assert exp._weights is None

    def test_experiment_weights_path_not_found(self, small_embeddings, base_config):
        """Non-existent weights file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights="/nonexistent/path/weights.json",
            )

    def test_experiment_weights_invalid_json(self, tmp_path, small_embeddings, base_config):
        """Invalid JSON in weights file should raise ValueError."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text("not valid json {")

        with pytest.raises(ValueError, match="Invalid JSON"):
            Experiment(
                embeddings=small_embeddings,
                config=base_config,
                weights=str(weights_file),
            )

    def test_experiment_weights_with_metadata(self, tmp_path, small_embeddings, base_config):
        """Weights with full metadata should load correctly."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(
            json.dumps({
                "version": "1.0",
                "coefficients": {
                    "Silhouette_norm": 0.3,
                    "Davies-Bouldin_norm": -0.2,
                },
                "bias": 0.5,
                "metadata": {
                    "regularization": "ridge",
                    "alpha": 1.0,
                    "cv_r2": 0.85,
                },
            })
        )

        exp = Experiment(
            embeddings=small_embeddings,
            config=base_config,
            weights=str(weights_file),
        )

        assert exp._weights is not None
        assert exp._weights.coefficients["Silhouette_norm"] == 0.3
        assert exp._weights.coefficients["Davies-Bouldin_norm"] == -0.2
        assert exp._weights.bias == 0.5


class TestToDataFrameWithWeights:
    """Tests for to_dataframe() with compound_score and is_best_run columns."""

    def test_to_dataframe_includes_compound_score(self, experiment_result_with_weights):
        """TC-019: to_dataframe should include compound_score column when weights used."""
        df = experiment_result_with_weights.to_dataframe()

        assert "compound_score" in df.columns
        # All completed runs should have compound_score values
        completed_mask = df["status"] == "completed"
        assert df.loc[completed_mask, "compound_score"].notna().all()

    def test_to_dataframe_includes_is_best_run(self, experiment_result_with_weights):
        """TC-020: to_dataframe should include is_best_run boolean column."""
        df = experiment_result_with_weights.to_dataframe()

        assert "is_best_run" in df.columns
        assert df["is_best_run"].dtype == bool
        # Exactly one run should be marked as best (or ties handled)
        assert df["is_best_run"].sum() >= 1

    def test_is_best_run_handles_ties(self, experiment_result_with_ties):
        """TC-021: is_best_run should be True for all tied runs."""
        df = experiment_result_with_ties.to_dataframe()

        assert "is_best_run" in df.columns
        # Multiple runs should be marked as best when there are ties
        best_count = df["is_best_run"].sum()
        assert best_count == 2  # Our fixture has 2 tied runs

    def test_to_dataframe_without_weights_has_null_compound_score(self, experiment_result_no_weights):
        """compound_score should be None/NaN when no weights provided."""
        df = experiment_result_no_weights.to_dataframe()

        assert "compound_score" in df.columns
        assert df["compound_score"].isna().all()

    def test_to_dataframe_without_weights_no_best_run(self, experiment_result_no_weights):
        """is_best_run should be False for all when no weights/best_run."""
        df = experiment_result_no_weights.to_dataframe()

        assert "is_best_run" in df.columns
        # No best run when no weights - all False
        assert not df["is_best_run"].any()
