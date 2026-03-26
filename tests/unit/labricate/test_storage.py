"""Tests for Labricate output storage (US6)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metricate.labricate.core.experiment import (
    ExperimentResult,
    ExperimentSummary,
    MetricResult,
    PipelineResult,
    RunResult,
)
from metricate.labricate.output.storage import (
    create_experiment_directory,
    save_results_csv,
    save_results_json,
    save_run_csv,
)
from metricate.labricate.utils.logging import TimingInfo


@pytest.fixture
def sample_experiment_result():
    """Create a sample ExperimentResult for testing."""
    runs = [
        RunResult(
            run_id=1,
            param_values={"hdbscan.min_cluster_size": 5},
            pipeline_result=PipelineResult(
                run_id=1,
                config={"hdbscan": {"min_cluster_size": 5}},
                labels=np.array([0, 0, 1, 1, -1]),
                reduced_embeddings=np.random.randn(5, 3),
                n_clusters=2,
                n_noise=1,
                timing=TimingInfo(
                    bertopic_seconds=1.5,
                    evaluation_seconds=0.5,
                    total_seconds=2.0,
                ),
            ),
            metrics=[
                MetricResult(name="silhouette", value=0.75, direction="higher"),
                MetricResult(name="davies_bouldin", value=0.3, direction="lower"),
            ],
        ),
        RunResult(
            run_id=2,
            param_values={"hdbscan.min_cluster_size": 10},
            pipeline_result=PipelineResult(
                run_id=2,
                config={"hdbscan": {"min_cluster_size": 10}},
                labels=np.array([0, 0, 0, 1, 1]),
                reduced_embeddings=np.random.randn(5, 3),
                n_clusters=2,
                n_noise=0,
                timing=TimingInfo(
                    bertopic_seconds=1.2,
                    evaluation_seconds=0.4,
                    total_seconds=1.6,
                ),
            ),
            metrics=[
                MetricResult(name="silhouette", value=0.85, direction="higher"),
                MetricResult(name="davies_bouldin", value=0.25, direction="lower"),
            ],
        ),
    ]

    return ExperimentResult(
        experiment_id="exp_20260323_120000",
        experiment_name="test_experiment",
        config={"clustering_algorithm": "hdbscan"},
        runs=runs,
        summary=ExperimentSummary(
            total_runs=2,
            completed_runs=2,
            failed_runs=0,
            skipped_runs=0,
            total_duration_seconds=3.6,
        ),
    )


class TestSaveResultsJSON:
    """Tests for JSON output format (T047)."""

    def test_save_creates_json_file(self, tmp_path, sample_experiment_result):
        """JSON file is created with correct name."""
        output_path = save_results_json(sample_experiment_result, tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".json"

    def test_json_contains_experiment_metadata(self, tmp_path, sample_experiment_result):
        """JSON contains experiment ID, name, and config."""
        output_path = save_results_json(sample_experiment_result, tmp_path)
        with open(output_path) as f:
            data = json.load(f)

        assert data["experiment_id"] == "exp_20260323_120000"
        assert data["experiment_name"] == "test_experiment"
        assert "config" in data
        assert data["config"]["clustering_algorithm"] == "hdbscan"

    def test_json_contains_runs(self, tmp_path, sample_experiment_result):
        """JSON contains all run results."""
        output_path = save_results_json(sample_experiment_result, tmp_path)
        with open(output_path) as f:
            data = json.load(f)

        assert "runs" in data
        assert len(data["runs"]) == 2
        assert data["runs"][0]["run_id"] == 1
        assert data["runs"][1]["run_id"] == 2

    def test_json_contains_metrics(self, tmp_path, sample_experiment_result):
        """JSON contains metric values for each run."""
        output_path = save_results_json(sample_experiment_result, tmp_path)
        with open(output_path) as f:
            data = json.load(f)

        run1_metrics = {m["name"]: m["value"] for m in data["runs"][0]["metrics"]}
        assert run1_metrics["silhouette"] == 0.75
        assert run1_metrics["davies_bouldin"] == 0.3

    def test_json_contains_summary(self, tmp_path, sample_experiment_result):
        """JSON contains experiment summary."""
        output_path = save_results_json(sample_experiment_result, tmp_path)
        with open(output_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert data["summary"]["total_runs"] == 2
        assert data["summary"]["completed_runs"] == 2
        assert data["summary"]["total_duration_seconds"] == 3.6


class TestSaveResultsCSV:
    """Tests for CSV output format (T048)."""

    def test_save_creates_csv_file(self, tmp_path, sample_experiment_result):
        """CSV file is created with correct name."""
        output_path = save_results_csv(sample_experiment_result, tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_csv_has_correct_columns(self, tmp_path, sample_experiment_result):
        """CSV contains run_id, param values, and metrics columns."""
        output_path = save_results_csv(sample_experiment_result, tmp_path)
        df = pd.read_csv(output_path)

        assert "run_id" in df.columns
        assert "hdbscan.min_cluster_size" in df.columns
        assert "silhouette" in df.columns
        assert "davies_bouldin" in df.columns
        assert "n_clusters" in df.columns
        assert "n_noise" in df.columns

    def test_csv_has_correct_rows(self, tmp_path, sample_experiment_result):
        """CSV has one row per run."""
        output_path = save_results_csv(sample_experiment_result, tmp_path)
        df = pd.read_csv(output_path)

        assert len(df) == 2
        assert df.loc[0, "run_id"] == 1
        assert df.loc[1, "run_id"] == 2

    def test_csv_metric_values_correct(self, tmp_path, sample_experiment_result):
        """CSV metric values match original results."""
        output_path = save_results_csv(sample_experiment_result, tmp_path)
        df = pd.read_csv(output_path)

        assert df.loc[0, "silhouette"] == 0.75
        assert df.loc[1, "silhouette"] == 0.85


class TestCreateExperimentDirectory:
    """Tests for hierarchical directory structure (T049)."""

    def test_creates_experiment_directory(self, tmp_path):
        """Creates experiments/<name>/<timestamp> directory."""
        exp_dir = create_experiment_directory(
            base_dir=tmp_path,
            experiment_name="my_experiment",
            timestamp="20260323_120000",
        )
        assert exp_dir.exists()
        assert exp_dir.is_dir()

    def test_directory_structure_correct(self, tmp_path):
        """Directory has correct hierarchical structure."""
        exp_dir = create_experiment_directory(
            base_dir=tmp_path,
            experiment_name="my_experiment",
            timestamp="20260323_120000",
        )
        # Should be: tmp_path/experiments/my_experiment/20260323_120000/
        expected = tmp_path / "experiments" / "my_experiment" / "20260323_120000"
        assert exp_dir == expected

    def test_creates_parent_directories(self, tmp_path):
        """Creates all parent directories if they don't exist."""
        exp_dir = create_experiment_directory(
            base_dir=tmp_path,
            experiment_name="nested/experiment",
            timestamp="20260323_120000",
        )
        assert exp_dir.exists()

    def test_auto_generates_timestamp(self, tmp_path):
        """Auto-generates timestamp if not provided."""
        exp_dir = create_experiment_directory(
            base_dir=tmp_path,
            experiment_name="my_experiment",
        )
        assert exp_dir.exists()
        # Should have a timestamp-like directory name (YYYYMMDD_HHMMSS)
        assert len(exp_dir.name) == 15  # 20260323_120000 format


class TestSaveRunCSV:
    """Tests for intermediate clustering CSV output (T054)."""

    def test_save_creates_csv_file(self, tmp_path, sample_experiment_result):
        """CSV file for run is created."""
        run = sample_experiment_result.runs[0]
        output_path = save_run_csv(run, tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_csv_contains_labels(self, tmp_path, sample_experiment_result):
        """CSV contains cluster_id column."""
        run = sample_experiment_result.runs[0]
        output_path = save_run_csv(run, tmp_path)
        df = pd.read_csv(output_path)

        assert "cluster_id" in df.columns
        assert list(df["cluster_id"]) == [0, 0, 1, 1, -1]

    def test_csv_contains_reduced_embeddings(self, tmp_path, sample_experiment_result):
        """CSV contains reduced embedding columns."""
        run = sample_experiment_result.runs[0]
        output_path = save_run_csv(run, tmp_path)
        df = pd.read_csv(output_path)

        # Should have dim_0, dim_1, dim_2 columns
        assert "dim_0" in df.columns
        assert "dim_1" in df.columns
        assert "dim_2" in df.columns

    def test_csv_file_named_with_run_id(self, tmp_path, sample_experiment_result):
        """CSV file is named with run ID."""
        run = sample_experiment_result.runs[0]
        output_path = save_run_csv(run, tmp_path)
        assert "run_001" in output_path.name
