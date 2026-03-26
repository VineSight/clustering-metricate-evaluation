"""Tests for Experiment class and single-parameter experiments (US1)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metricate.labricate.core.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentSummary,
    MetricResult,
    PipelineResult,
    RunResult,
)

# Patch path for metricate.evaluate (imported inside run())
EVALUATE_PATCH = "metricate.evaluate"


class TestExperimentInit:
    """Tests for Experiment.__init__()."""

    def test_init_with_array_and_dict(self, sample_embeddings, base_config):
        """Experiment initializes with array and config dict."""
        exp = Experiment(sample_embeddings, base_config)
        assert exp.embeddings.shape == sample_embeddings.shape
        assert exp.n_samples == 100
        assert "umap" in exp.config

    def test_init_with_csv_path(self, embeddings_csv, base_config):
        """Experiment initializes with CSV path."""
        exp = Experiment(embeddings_csv, base_config)
        assert exp.n_samples == 100

    def test_init_with_dataframe(self, embeddings_dataframe, base_config):
        """Experiment initializes with DataFrame."""
        exp = Experiment(embeddings_dataframe, base_config)
        assert exp.n_samples == 100

    def test_init_default_name_generated(self, sample_embeddings, base_config):
        """Default experiment name is generated if not provided."""
        exp = Experiment(sample_embeddings, base_config)
        assert exp.name.startswith("experiment_")

    def test_init_custom_name(self, sample_embeddings, base_config):
        """Custom experiment name is preserved."""
        exp = Experiment(sample_embeddings, base_config, name="my_experiment")
        assert exp.name == "my_experiment"

    def test_init_invalid_config_raises(self, sample_embeddings):
        """Invalid config raises ValueError."""
        invalid_config = {"clustering_algorithm": "invalid"}
        with pytest.raises(ValueError, match="Invalid config"):
            Experiment(sample_embeddings, invalid_config)

    def test_init_default_pipeline_is_bertopic(self, sample_embeddings, base_config):
        """Default pipeline is BERTopicPipeline."""
        exp = Experiment(sample_embeddings, base_config)
        from metricate.labricate.pipelines.bertopic import BERTopicPipeline

        assert isinstance(exp.pipeline, BERTopicPipeline)


class TestCustomPipeline:
    """Tests for custom pipeline integration (US4)."""

    def test_accepts_custom_pipeline_function(self, sample_embeddings, base_config):
        """T041: Experiment accepts custom pipeline function."""
        def custom_pipeline(embeddings, config):
            n_samples = embeddings.shape[0]
            labels = np.zeros(n_samples, dtype=np.int64)
            reduced = np.random.randn(n_samples, 5)
            return labels, reduced

        exp = Experiment(sample_embeddings, base_config, pipeline=custom_pipeline)
        assert exp.pipeline is custom_pipeline

    def test_accepts_custom_pipeline_class(self, sample_embeddings, base_config):
        """T041: Experiment accepts custom pipeline class instance."""
        class CustomPipeline:
            def __call__(self, embeddings, config):
                n_samples = embeddings.shape[0]
                labels = np.zeros(n_samples, dtype=np.int64)
                reduced = np.random.randn(n_samples, 5)
                return labels, reduced

        custom = CustomPipeline()
        exp = Experiment(sample_embeddings, base_config, pipeline=custom)
        assert exp.pipeline is custom

    def test_custom_pipeline_output_validation(self, small_embeddings, base_config):
        """T042: Custom pipeline output is validated for correct shapes."""
        def valid_pipeline(embeddings, config):
            n_samples = embeddings.shape[0]
            labels = np.zeros(n_samples, dtype=np.int64)
            reduced = np.random.randn(n_samples, 5)
            return labels, reduced

        exp = Experiment(small_embeddings, base_config, pipeline=valid_pipeline)

        # Mock metricate to avoid actual evaluation
        mock_metrics = [MagicMock(metric="silhouette", value=0.5, direction="higher")]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        assert result.runs[0].pipeline_result.status == "completed"

    def test_custom_pipeline_invalid_labels_shape_raises(self, small_embeddings, base_config):
        """T043: Error handling for invalid labels shape (not 1D)."""
        def bad_pipeline(embeddings, config):
            n_samples = embeddings.shape[0]
            labels = np.zeros((n_samples, 2))  # Wrong: 2D instead of 1D
            reduced = np.random.randn(n_samples, 5)
            return labels, reduced

        exp = Experiment(small_embeddings, base_config, pipeline=bad_pipeline)

        with patch(EVALUATE_PATCH):
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        # Run should fail with validation error
        assert result.runs[0].pipeline_result.status == "failed"
        assert "1D" in result.runs[0].pipeline_result.error

    def test_custom_pipeline_invalid_embeddings_shape_raises(self, small_embeddings, base_config):
        """T043: Error handling for invalid reduced embeddings shape (not 2D)."""
        def bad_pipeline(embeddings, config):
            n_samples = embeddings.shape[0]
            labels = np.zeros(n_samples, dtype=np.int64)
            reduced = np.random.randn(n_samples)  # Wrong: 1D instead of 2D
            return labels, reduced

        exp = Experiment(small_embeddings, base_config, pipeline=bad_pipeline)

        with patch(EVALUATE_PATCH):
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        # Run should fail with validation error
        assert result.runs[0].pipeline_result.status == "failed"
        assert "2D" in result.runs[0].pipeline_result.error

    def test_custom_pipeline_wrong_n_samples_raises(self, small_embeddings, base_config):
        """T043: Error handling for mismatched sample count."""
        def bad_pipeline(embeddings, config):
            n_samples = embeddings.shape[0]
            labels = np.zeros(n_samples + 10)  # Wrong: mismatched count
            reduced = np.random.randn(n_samples, 5)
            return labels, reduced

        exp = Experiment(small_embeddings, base_config, pipeline=bad_pipeline)

        with patch(EVALUATE_PATCH):
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        # Run should fail with validation error
        assert result.runs[0].pipeline_result.status == "failed"
        assert "n_samples" in result.runs[0].pipeline_result.error

    def test_custom_pipeline_exception_handled(self, small_embeddings, base_config):
        """T043: Exception in custom pipeline is caught and recorded."""
        def broken_pipeline(embeddings, config):
            raise RuntimeError("Pipeline exploded!")

        exp = Experiment(small_embeddings, base_config, pipeline=broken_pipeline)

        result = exp.run(
            param="hdbscan.min_cluster_size",
            values=[10],
            verbose=False,
        )

        assert result.runs[0].pipeline_result.status == "failed"
        assert "Pipeline exploded!" in result.runs[0].pipeline_result.error

    def test_custom_pipeline_non_array_labels_raises(self, small_embeddings, base_config):
        """T043: Error handling for non-array labels."""
        def bad_pipeline(embeddings, config):
            n_samples = embeddings.shape[0]
            labels = list(range(n_samples))  # Wrong: list instead of array
            reduced = np.random.randn(n_samples, 5)
            return labels, reduced

        exp = Experiment(small_embeddings, base_config, pipeline=bad_pipeline)

        with patch(EVALUATE_PATCH):
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        # Run should fail with validation error
        assert result.runs[0].pipeline_result.status == "failed"
        assert "numpy array" in result.runs[0].pipeline_result.error


class TestExperimentValidateParam:
    """Tests for Experiment.validate_param()."""

    def test_validate_valid_param(self, sample_embeddings, base_config):
        """Valid parameter path returns True."""
        exp = Experiment(sample_embeddings, base_config)
        assert exp.validate_param("hdbscan.min_cluster_size") is True

    def test_validate_invalid_param_raises(self, sample_embeddings, base_config):
        """Invalid parameter path raises ValueError."""
        exp = Experiment(sample_embeddings, base_config)
        with pytest.raises(ValueError, match="Invalid parameter path"):
            exp.validate_param("invalid.path")


class TestExperimentRunSingleParam:
    """Tests for Experiment.run() with single parameter (US1 Tests)."""

    def test_run_executes_correct_number_of_runs(self, small_embeddings, base_config):
        """T016: run() executes correct number of runs."""
        exp = Experiment(small_embeddings, base_config)

        # Mock the pipeline and metricate to avoid actual computation
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            np.zeros(50),  # labels
            np.random.randn(50, 5),  # reduced embeddings
        )
        exp.pipeline = mock_pipeline

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={"silhouette": 0.5})
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10, 15],
                verbose=False,
            )

        assert len(result.runs) == 3
        assert mock_pipeline.call_count == 3

    def test_run_applies_parameter_values_correctly(self, small_embeddings, base_config):
        """T017: Parameter values are correctly applied per run."""
        exp = Experiment(small_embeddings, base_config)
        test_values = [5, 10, 15, 20]

        # Track configs passed to pipeline
        configs_used = []

        def capture_config(embeddings, config):
            configs_used.append(config.copy())
            return np.zeros(50), np.random.randn(50, 5)

        mock_pipeline = MagicMock(side_effect=capture_config)
        exp.pipeline = mock_pipeline

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={"silhouette": 0.5})
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=test_values,
                verbose=False,
            )

        # Verify each run used the correct parameter value
        for i, config in enumerate(configs_used):
            assert config["hdbscan"]["min_cluster_size"] == test_values[i]

        # Verify result param_values match
        for i, run in enumerate(result.runs):
            assert run.param_values["hdbscan.min_cluster_size"] == test_values[i]

    def test_run_returns_metricate_metrics(self, small_embeddings, base_config):
        """T018: Metricate integration returns expected metric results."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            np.zeros(50),
            np.random.randn(50, 5),
        )
        exp.pipeline = mock_pipeline

        # Create mock MetricValue objects (matching metricate's EvaluationResult.computed_metrics())
        mock_metrics = [
            MagicMock(metric="silhouette", value=0.75, direction="higher"),
            MagicMock(metric="calinski_harabasz", value=150.0, direction="higher"),
            MagicMock(metric="davies_bouldin", value=0.8, direction="lower"),
        ]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        # Check metrics are captured
        run = result.runs[0]
        metric_names = {m.name for m in run.metrics}
        assert "silhouette" in metric_names
        assert "calinski_harabasz" in metric_names
        assert "davies_bouldin" in metric_names

    def test_run_empty_values_raises(self, small_embeddings, base_config):
        """run() raises ValueError for empty values list."""
        exp = Experiment(small_embeddings, base_config)
        with pytest.raises(ValueError, match="values list cannot be empty"):
            exp.run(param="hdbscan.min_cluster_size", values=[], verbose=False)

    def test_run_invalid_param_raises(self, small_embeddings, base_config):
        """run() raises ValueError for invalid parameter path."""
        exp = Experiment(small_embeddings, base_config)
        with pytest.raises(ValueError, match="Invalid parameter path"):
            exp.run(param="invalid.path", values=[10], verbose=False)

    def test_run_generates_experiment_id(self, small_embeddings, base_config):
        """run() generates unique experiment ID."""
        exp = Experiment(small_embeddings, base_config, name="test_exp")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (np.zeros(50), np.random.randn(50, 5))
        exp.pipeline = mock_pipeline

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={})
            result = exp.run(param="hdbscan.min_cluster_size", values=[10], verbose=False)

        assert result.experiment_id.startswith("test_exp_")

    def test_run_creates_summary(self, small_embeddings, base_config):
        """run() creates ExperimentSummary with correct counts."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (np.zeros(50), np.random.randn(50, 5))
        exp.pipeline = mock_pipeline

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={})
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10, 15],
                verbose=False,
            )

        assert result.summary.total_runs == 3
        assert result.summary.completed_runs == 3
        assert result.summary.failed_runs == 0


class TestExperimentErrorHandling:
    """Tests for error handling in Experiment.run()."""

    def test_run_continue_on_error(self, small_embeddings, base_config):
        """error_handling='continue' logs error and continues."""
        exp = Experiment(small_embeddings, base_config)

        call_count = [0]

        def failing_pipeline(embeddings, config):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated failure")
            return np.zeros(50), np.random.randn(50, 5)

        exp.pipeline = MagicMock(side_effect=failing_pipeline)

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={})
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10, 15],
                error_handling="continue",
                verbose=False,
            )

        assert len(result.runs) == 3
        assert result.summary.completed_runs == 2
        assert result.summary.failed_runs == 1
        assert result.runs[1].pipeline_result.status == "failed"

    def test_run_fail_fast_on_error(self, small_embeddings, base_config):
        """error_handling='fail_fast' raises immediately."""
        exp = Experiment(small_embeddings, base_config)

        call_count = [0]

        def failing_pipeline(embeddings, config):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated failure")
            return np.zeros(50), np.random.randn(50, 5)

        exp.pipeline = MagicMock(side_effect=failing_pipeline)

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={})
            with pytest.raises(RuntimeError, match="Run 2 failed"):
                exp.run(
                    param="hdbscan.min_cluster_size",
                    values=[5, 10, 15],
                    error_handling="fail_fast",
                    verbose=False,
                )


class TestExperimentResultDataclass:
    """Tests for ExperimentResult dataclass."""

    def test_to_dataframe(self, small_embeddings, base_config):
        """to_dataframe() returns valid DataFrame."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (np.zeros(50), np.random.randn(50, 5))
        exp.pipeline = mock_pipeline

        # Create mock MetricValue objects
        mock_metrics = [MagicMock(metric="silhouette", value=0.5, direction="higher")]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[5, 10],
                verbose=False,
            )

        df = result.to_dataframe()
        assert len(df) == 2
        assert "run_id" in df.columns
        assert "hdbscan.min_cluster_size" in df.columns
        assert "silhouette" in df.columns

    def test_get_best_run_higher(self, small_embeddings, base_config):
        """get_best_run() returns run with highest metric value."""
        # Create synthetic results
        runs = [
            RunResult(
                run_id=1,
                param_values={"p": 5},
                pipeline_result=PipelineResult(
                    run_id=1,
                    config={},
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=5,
                    n_noise=0,
                    timing=MagicMock(),
                ),
                metrics=[MetricResult(name="silhouette", value=0.5)],
            ),
            RunResult(
                run_id=2,
                param_values={"p": 10},
                pipeline_result=PipelineResult(
                    run_id=2,
                    config={},
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=5,
                    n_noise=0,
                    timing=MagicMock(),
                ),
                metrics=[MetricResult(name="silhouette", value=0.8)],
            ),
        ]

        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            config={},
            runs=runs,
            summary=ExperimentSummary(
                total_runs=2,
                completed_runs=2,
                failed_runs=0,
                skipped_runs=0,
                total_duration_seconds=1.0,
            ),
        )

        best = result.get_best_run("silhouette", direction="higher")
        assert best.run_id == 2

    def test_get_best_run_lower(self):
        """get_best_run() returns run with lowest metric value when direction='lower'."""
        runs = [
            RunResult(
                run_id=1,
                param_values={"p": 5},
                pipeline_result=PipelineResult(
                    run_id=1,
                    config={},
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=5,
                    n_noise=0,
                    timing=MagicMock(),
                ),
                metrics=[MetricResult(name="davies_bouldin", value=0.8, direction="lower")],
            ),
            RunResult(
                run_id=2,
                param_values={"p": 10},
                pipeline_result=PipelineResult(
                    run_id=2,
                    config={},
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=5,
                    n_noise=0,
                    timing=MagicMock(),
                ),
                metrics=[MetricResult(name="davies_bouldin", value=0.5, direction="lower")],
            ),
        ]

        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            config={},
            runs=runs,
            summary=ExperimentSummary(
                total_runs=2,
                completed_runs=2,
                failed_runs=0,
                skipped_runs=0,
                total_duration_seconds=1.0,
            ),
        )

        best = result.get_best_run("davies_bouldin", direction="lower")
        assert best.run_id == 2

    def test_get_best_run_no_completed_raises(self):
        """get_best_run() raises ValueError if no completed runs."""
        runs = [
            RunResult(
                run_id=1,
                param_values={"p": 5},
                pipeline_result=PipelineResult(
                    run_id=1,
                    config={},
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=0,
                    n_noise=0,
                    timing=MagicMock(),
                    status="failed",
                ),
                metrics=[],
            ),
        ]

        result = ExperimentResult(
            experiment_id="test",
            experiment_name="test",
            config={},
            runs=runs,
            summary=ExperimentSummary(
                total_runs=1,
                completed_runs=0,
                failed_runs=1,
                skipped_runs=0,
                total_duration_seconds=1.0,
            ),
        )

        with pytest.raises(ValueError, match="No completed runs"):
            result.get_best_run("silhouette")


class TestExperimentCustomPipeline:
    """Tests for custom pipeline support."""

    def test_custom_pipeline_function(self, small_embeddings, base_config):
        """Experiment accepts custom pipeline function."""

        def custom_pipeline(embeddings, config):
            return np.zeros(len(embeddings)), np.random.randn(len(embeddings), 3)

        exp = Experiment(small_embeddings, base_config, pipeline=custom_pipeline)

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = MagicMock(metrics={"silhouette": 0.5})
            result = exp.run(
                param="hdbscan.min_cluster_size",
                values=[10],
                verbose=False,
            )

        assert len(result.runs) == 1
        assert result.runs[0].pipeline_result.status == "completed"


class TestExperimentGridSearch:
    """Tests for grid search experiments (US5)."""

    def test_run_grid_executes_all_combinations(self, small_embeddings, base_config):
        """T057: run_grid() executes all parameter combinations."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            np.zeros(50, dtype=np.int64),
            np.random.randn(50, 5),
        )
        exp.pipeline = mock_pipeline

        mock_metrics = [MagicMock(metric="silhouette", value=0.5, direction="higher")]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run_grid(
                params={
                    "hdbscan.min_cluster_size": [5, 10],
                    "umap.n_neighbors": [10, 15],
                },
                verbose=False,
            )

        # 2 x 2 = 4 combinations
        assert len(result.runs) == 4
        assert mock_pipeline.call_count == 4

    def test_run_grid_correct_run_count_2_params(self, small_embeddings, base_config):
        """T058: Grid search with 2 parameters produces correct run count."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            np.zeros(50, dtype=np.int64),
            np.random.randn(50, 5),
        )
        exp.pipeline = mock_pipeline

        mock_metrics = [MagicMock(metric="silhouette", value=0.5, direction="higher")]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run_grid(
                params={
                    "hdbscan.min_cluster_size": [5, 10, 15],
                    "umap.n_neighbors": [10, 15, 20, 25],
                },
                verbose=False,
            )

        # 3 x 4 = 12 combinations
        assert len(result.runs) == 12
        assert result.summary.total_runs == 12

    def test_run_grid_param_values_correct(self, small_embeddings, base_config):
        """run_grid() stores correct param values for each run."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            np.zeros(50, dtype=np.int64),
            np.random.randn(50, 5),
        )
        exp.pipeline = mock_pipeline

        mock_metrics = [MagicMock(metric="silhouette", value=0.5, direction="higher")]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run_grid(
                params={
                    "hdbscan.min_cluster_size": [5, 10],
                    "umap.n_neighbors": [15, 20],
                },
                verbose=False,
            )

        # Check all combinations are present
        param_combinations = [
            (r.param_values["hdbscan.min_cluster_size"], r.param_values["umap.n_neighbors"])
            for r in result.runs
        ]
        expected = [(5, 15), (5, 20), (10, 15), (10, 20)]
        assert sorted(param_combinations) == sorted(expected)

    def test_run_grid_empty_params_raises(self, small_embeddings, base_config):
        """run_grid() raises ValueError for empty params dict."""
        exp = Experiment(small_embeddings, base_config)
        with pytest.raises(ValueError, match="params.*empty"):
            exp.run_grid(params={}, verbose=False)

    def test_run_grid_empty_values_raises(self, small_embeddings, base_config):
        """run_grid() raises ValueError for param with empty values."""
        exp = Experiment(small_embeddings, base_config)
        with pytest.raises(ValueError, match="values.*empty"):
            exp.run_grid(
                params={"hdbscan.min_cluster_size": []},
                verbose=False,
            )

    def test_run_grid_invalid_param_raises(self, small_embeddings, base_config):
        """run_grid() raises ValueError for invalid parameter path."""
        exp = Experiment(small_embeddings, base_config)
        with pytest.raises(ValueError, match="Invalid parameter"):
            exp.run_grid(
                params={"invalid.path": [5, 10]},
                verbose=False,
            )

    def test_run_grid_three_params(self, small_embeddings, base_config):
        """run_grid() handles 3+ parameters correctly."""
        exp = Experiment(small_embeddings, base_config)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            np.zeros(50, dtype=np.int64),
            np.random.randn(50, 5),
        )
        exp.pipeline = mock_pipeline

        mock_metrics = [MagicMock(metric="silhouette", value=0.5, direction="higher")]
        mock_eval_result = MagicMock()
        mock_eval_result.computed_metrics.return_value = mock_metrics

        with patch(EVALUATE_PATCH) as mock_eval:
            mock_eval.return_value = mock_eval_result
            result = exp.run_grid(
                params={
                    "hdbscan.min_cluster_size": [5, 10],
                    "umap.n_neighbors": [10, 15],
                    "umap.n_components": [3, 5],
                },
                verbose=False,
            )

        # 2 x 2 x 2 = 8 combinations
        assert len(result.runs) == 8
