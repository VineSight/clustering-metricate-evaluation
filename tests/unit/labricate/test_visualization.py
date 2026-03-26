"""Tests for Labricate visualization (US6)."""

import numpy as np
import pytest

from metricate.labricate.core.experiment import (
    ExperimentResult,
    ExperimentSummary,
    MetricResult,
    PipelineResult,
    RunResult,
)
from metricate.labricate.output.visualization import (
    plot_heatmap,
    plot_metric_vs_param,
)
from metricate.labricate.utils.logging import TimingInfo


@pytest.fixture
def sample_experiment_result():
    """Create a sample ExperimentResult for testing."""
    runs = []
    for i, val in enumerate([5, 10, 15, 20], start=1):
        runs.append(
            RunResult(
                run_id=i,
                param_values={"hdbscan.min_cluster_size": val},
                pipeline_result=PipelineResult(
                    run_id=i,
                    config={"hdbscan": {"min_cluster_size": val}},
                    labels=np.zeros(10, dtype=np.int64),
                    reduced_embeddings=np.random.randn(10, 3),
                    n_clusters=3,
                    n_noise=1,
                    timing=TimingInfo(
                        bertopic_seconds=1.0,
                        evaluation_seconds=0.5,
                        total_seconds=1.5,
                    ),
                ),
                metrics=[
                    MetricResult(
                        name="silhouette",
                        value=0.5 + i * 0.1,
                        direction="higher",
                    ),
                    MetricResult(
                        name="davies_bouldin",
                        value=0.5 - i * 0.05,
                        direction="lower",
                    ),
                ],
            )
        )

    return ExperimentResult(
        experiment_id="exp_20260323_120000",
        experiment_name="test_experiment",
        config={"clustering_algorithm": "hdbscan"},
        runs=runs,
        summary=ExperimentSummary(
            total_runs=4,
            completed_runs=4,
            failed_runs=0,
            skipped_runs=0,
            total_duration_seconds=6.0,
        ),
    )


class TestPlotMetricVsParam:
    """Tests for line chart generation (T050)."""

    def test_creates_figure(self, sample_experiment_result):
        """Line chart creates a matplotlib figure."""
        fig = plot_metric_vs_param(
            sample_experiment_result,
            metric="silhouette",
            param="hdbscan.min_cluster_size",
        )
        assert fig is not None
        # Check it's a matplotlib figure
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path, sample_experiment_result):
        """Line chart can be saved to file."""
        output_path = tmp_path / "chart.png"
        fig = plot_metric_vs_param(
            sample_experiment_result,
            metric="silhouette",
            param="hdbscan.min_cluster_size",
            output_path=output_path,
        )
        assert output_path.exists()

    def test_uses_correct_data_points(self, sample_experiment_result):
        """Line chart plots correct param values vs metric values."""
        fig = plot_metric_vs_param(
            sample_experiment_result,
            metric="silhouette",
            param="hdbscan.min_cluster_size",
        )
        # Get the axes and line data
        ax = fig.axes[0]
        line = ax.lines[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # Check x values (param values)
        assert list(x_data) == [5, 10, 15, 20]
        # Check y values (metric values)
        assert list(y_data) == [0.6, 0.7, 0.8, 0.9]

    def test_sets_axis_labels(self, sample_experiment_result):
        """Line chart has correct axis labels."""
        fig = plot_metric_vs_param(
            sample_experiment_result,
            metric="silhouette",
            param="hdbscan.min_cluster_size",
        )
        ax = fig.axes[0]
        # Labels have underscores replaced with spaces
        assert "min cluster size" in ax.get_xlabel().lower()
        assert "silhouette" in ax.get_ylabel().lower()

    def test_sets_title(self, sample_experiment_result):
        """Line chart has title with experiment name."""
        fig = plot_metric_vs_param(
            sample_experiment_result,
            metric="silhouette",
            param="hdbscan.min_cluster_size",
        )
        ax = fig.axes[0]
        title = ax.get_title()
        assert "silhouette" in title.lower() or "test_experiment" in title.lower()

    def test_raises_for_missing_metric(self, sample_experiment_result):
        """Raises ValueError for metric not in results."""
        with pytest.raises(ValueError, match="not found"):
            plot_metric_vs_param(
                sample_experiment_result,
                metric="nonexistent_metric",
                param="hdbscan.min_cluster_size",
            )


class TestPlotHeatmap:
    """Tests for heatmap visualization (for grid search)."""

    @pytest.fixture
    def grid_experiment_result(self):
        """Create a sample grid search result for heatmap testing."""
        runs = []
        run_id = 1
        for min_cluster in [5, 10, 15]:
            for n_neighbors in [10, 15, 20]:
                runs.append(
                    RunResult(
                        run_id=run_id,
                        param_values={
                            "hdbscan.min_cluster_size": min_cluster,
                            "umap.n_neighbors": n_neighbors,
                        },
                        pipeline_result=PipelineResult(
                            run_id=run_id,
                            config={},
                            labels=np.zeros(10, dtype=np.int64),
                            reduced_embeddings=np.random.randn(10, 3),
                            n_clusters=3,
                            n_noise=0,
                            timing=TimingInfo(
                                bertopic_seconds=1.0,
                                evaluation_seconds=0.5,
                                total_seconds=1.5,
                            ),
                        ),
                        metrics=[
                            MetricResult(
                                name="silhouette",
                                value=(min_cluster + n_neighbors) / 100.0,
                                direction="higher",
                            ),
                        ],
                    )
                )
                run_id += 1

        return ExperimentResult(
            experiment_id="grid_exp",
            experiment_name="grid_test",
            config={},
            runs=runs,
            summary=ExperimentSummary(
                total_runs=9,
                completed_runs=9,
                failed_runs=0,
                skipped_runs=0,
                total_duration_seconds=13.5,
            ),
        )

    def test_creates_figure(self, grid_experiment_result):
        """Heatmap creates a matplotlib figure."""
        fig = plot_heatmap(
            grid_experiment_result,
            metric="silhouette",
            param_x="hdbscan.min_cluster_size",
            param_y="umap.n_neighbors",
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path, grid_experiment_result):
        """Heatmap can be saved to file."""
        output_path = tmp_path / "heatmap.png"
        fig = plot_heatmap(
            grid_experiment_result,
            metric="silhouette",
            param_x="hdbscan.min_cluster_size",
            param_y="umap.n_neighbors",
            output_path=output_path,
        )
        assert output_path.exists()

    def test_heatmap_dimensions_match_grid(self, grid_experiment_result):
        """Heatmap has correct dimensions for 3x3 grid."""
        fig = plot_heatmap(
            grid_experiment_result,
            metric="silhouette",
            param_x="hdbscan.min_cluster_size",
            param_y="umap.n_neighbors",
        )
        ax = fig.axes[0]
        # Get the heatmap data (the image)
        images = ax.images
        assert len(images) == 1
        heatmap_data = images[0].get_array()
        assert heatmap_data.shape == (3, 3)  # 3 y values x 3 x values
