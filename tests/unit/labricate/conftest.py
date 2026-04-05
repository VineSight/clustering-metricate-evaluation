"""Pytest fixtures for Labricate tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing (100 samples, 384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(100, 384)


@pytest.fixture
def small_embeddings():
    """Create small embeddings for fast tests (50 samples, 50 dimensions)."""
    np.random.seed(42)
    return np.random.randn(50, 50)


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
            "repulsion_strength": 1.0,
            "low_memory": False,
        },
        "clustering_algorithm": "hdbscan",
        "hdbscan": {
            "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "eom",
            "metric": "euclidean",
        },
        "kmeans": {
            "n_clusters": 10,
        },
    }


@pytest.fixture
def kmeans_config():
    """Create a K-Means pipeline configuration for testing."""
    return {
        "random_seed": 42,
        "umap": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
        },
        "clustering_algorithm": "kmeans",
        "kmeans": {
            "n_clusters": 5,
        },
    }


@pytest.fixture
def embeddings_csv(tmp_path, sample_embeddings):
    """Create a CSV file with embeddings."""
    n_samples, n_dims = sample_embeddings.shape
    df = pd.DataFrame(
        {f"dim_{i}": sample_embeddings[:, i] for i in range(n_dims)}
    )
    csv_path = tmp_path / "embeddings.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def embeddings_npy(tmp_path, sample_embeddings):
    """Create a NPY file with embeddings."""
    npy_path = tmp_path / "embeddings.npy"
    np.save(npy_path, sample_embeddings)
    return npy_path


@pytest.fixture
def embeddings_npz(tmp_path, sample_embeddings):
    """Create a NPZ file with embeddings."""
    npz_path = tmp_path / "embeddings.npz"
    np.savez(npz_path, embeddings=sample_embeddings)
    return npz_path


@pytest.fixture
def embeddings_dataframe(sample_embeddings):
    """Create a pandas DataFrame with embeddings."""
    n_samples, n_dims = sample_embeddings.shape
    return pd.DataFrame(
        {f"dim_{i}": sample_embeddings[:, i] for i in range(n_dims)}
    )


@pytest.fixture
def ground_truth_labels():
    """Create ground truth labels for supervised metrics."""
    np.random.seed(42)
    return np.random.randint(0, 5, 100)


@pytest.fixture
def experiment_result_with_weights():
    """Create an ExperimentResult with weights and compound_score populated."""
    from metricate.labricate.core.experiment import (
        ExperimentResult,
        ExperimentSummary,
        MetricResult,
        PipelineResult,
        RunResult,
    )
    from metricate.labricate.core.scoring import BestRunInfo
    from metricate.labricate.utils.logging import TimingInfo

    runs = [
        RunResult(
            run_id=1,
            param_values={"min_cluster_size": 5},
            pipeline_result=PipelineResult(
                run_id=1,
                config={},
                labels=np.array([0, 0, 1, 1, -1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=1,
                timing=TimingInfo(bertopic_seconds=1.0, evaluation_seconds=0.5, total_seconds=1.5),
                status="completed",
            ),
            metrics=[
                MetricResult(name="Silhouette", value=0.5, direction="higher"),
                MetricResult(name="Davies-Bouldin", value=0.8, direction="lower"),
            ],
            compound_score=0.75,
        ),
        RunResult(
            run_id=2,
            param_values={"min_cluster_size": 10},
            pipeline_result=PipelineResult(
                run_id=2,
                config={},
                labels=np.array([0, 0, 0, 1, 1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=0,
                timing=TimingInfo(bertopic_seconds=1.2, evaluation_seconds=0.4, total_seconds=1.6),
                status="completed",
            ),
            metrics=[
                MetricResult(name="Silhouette", value=0.6, direction="higher"),
                MetricResult(name="Davies-Bouldin", value=0.7, direction="lower"),
            ],
            compound_score=0.85,  # Best score
        ),
    ]

    return ExperimentResult(
        experiment_id="test_exp_001",
        experiment_name="test_experiment",
        config={"param": "min_cluster_size", "values": [5, 10]},
        runs=runs,
        summary=ExperimentSummary(
            total_runs=2,
            completed_runs=2,
            failed_runs=0,
            skipped_runs=0,
            total_duration_seconds=3.1,
        ),
        best_run=BestRunInfo(
            run_id=2,
            param_values={"min_cluster_size": 10},
            score=0.85,
            score_type="compound_score",
            tied_run_ids=[],
        ),
    )


@pytest.fixture
def experiment_result_with_ties():
    """Create an ExperimentResult with tied compound_scores."""
    from metricate.labricate.core.experiment import (
        ExperimentResult,
        ExperimentSummary,
        MetricResult,
        PipelineResult,
        RunResult,
    )
    from metricate.labricate.core.scoring import BestRunInfo
    from metricate.labricate.utils.logging import TimingInfo

    runs = [
        RunResult(
            run_id=1,
            param_values={"min_cluster_size": 5},
            pipeline_result=PipelineResult(
                run_id=1,
                config={},
                labels=np.array([0, 0, 1, 1, -1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=1,
                timing=TimingInfo(bertopic_seconds=1.0, evaluation_seconds=0.5, total_seconds=1.5),
                status="completed",
            ),
            metrics=[MetricResult(name="Silhouette", value=0.5, direction="higher")],
            compound_score=0.80,  # Tied
        ),
        RunResult(
            run_id=2,
            param_values={"min_cluster_size": 10},
            pipeline_result=PipelineResult(
                run_id=2,
                config={},
                labels=np.array([0, 0, 0, 1, 1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=0,
                timing=TimingInfo(bertopic_seconds=1.2, evaluation_seconds=0.4, total_seconds=1.6),
                status="completed",
            ),
            metrics=[MetricResult(name="Silhouette", value=0.6, direction="higher")],
            compound_score=0.80,  # Tied
        ),
        RunResult(
            run_id=3,
            param_values={"min_cluster_size": 15},
            pipeline_result=PipelineResult(
                run_id=3,
                config={},
                labels=np.array([0, 0, 0, 0, 1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=0,
                timing=TimingInfo(bertopic_seconds=1.1, evaluation_seconds=0.3, total_seconds=1.4),
                status="completed",
            ),
            metrics=[MetricResult(name="Silhouette", value=0.55, direction="higher")],
            compound_score=0.70,  # Not tied - lower
        ),
    ]

    return ExperimentResult(
        experiment_id="test_exp_ties",
        experiment_name="test_experiment_ties",
        config={"param": "min_cluster_size", "values": [5, 10, 15]},
        runs=runs,
        summary=ExperimentSummary(
            total_runs=3,
            completed_runs=3,
            failed_runs=0,
            skipped_runs=0,
            total_duration_seconds=4.5,
        ),
        best_run=BestRunInfo(
            run_id=1,
            param_values={"min_cluster_size": 5},
            score=0.80,
            score_type="compound_score",
            tied_run_ids=[2],  # Run 2 is tied
        ),
    )


@pytest.fixture
def experiment_result_no_weights():
    """Create an ExperimentResult without weights (no compound_score)."""
    from metricate.labricate.core.experiment import (
        ExperimentResult,
        ExperimentSummary,
        MetricResult,
        PipelineResult,
        RunResult,
    )
    from metricate.labricate.utils.logging import TimingInfo

    runs = [
        RunResult(
            run_id=1,
            param_values={"min_cluster_size": 5},
            pipeline_result=PipelineResult(
                run_id=1,
                config={},
                labels=np.array([0, 0, 1, 1, -1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=1,
                timing=TimingInfo(bertopic_seconds=1.0, evaluation_seconds=0.5, total_seconds=1.5),
                status="completed",
            ),
            metrics=[
                MetricResult(name="Silhouette", value=0.5, direction="higher"),
            ],
            compound_score=None,  # No weights
        ),
        RunResult(
            run_id=2,
            param_values={"min_cluster_size": 10},
            pipeline_result=PipelineResult(
                run_id=2,
                config={},
                labels=np.array([0, 0, 0, 1, 1]),
                reduced_embeddings=np.random.randn(5, 2),
                n_clusters=2,
                n_noise=0,
                timing=TimingInfo(bertopic_seconds=1.2, evaluation_seconds=0.4, total_seconds=1.6),
                status="completed",
            ),
            metrics=[
                MetricResult(name="Silhouette", value=0.6, direction="higher"),
            ],
            compound_score=None,  # No weights
        ),
    ]

    return ExperimentResult(
        experiment_id="test_exp_no_weights",
        experiment_name="test_experiment_no_weights",
        config={"param": "min_cluster_size", "values": [5, 10]},
        runs=runs,
        summary=ExperimentSummary(
            total_runs=2,
            completed_runs=2,
            failed_runs=0,
            skipped_runs=0,
            total_duration_seconds=3.1,
        ),
        best_run=None,  # No best_run without weights
    )


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline that returns simple clustering results."""

    def pipeline_fn(embeddings, config):
        """Mock pipeline that returns deterministic clusters."""
        n_samples = embeddings.shape[0]
        # Simple clustering: split into 2 clusters based on first embedding dimension
        labels = np.zeros(n_samples, dtype=int)
        labels[n_samples // 2:] = 1

        # Reduce to 5 dimensions
        reduced = embeddings[:, :5] if embeddings.shape[1] >= 5 else embeddings

        return labels, reduced

    return pipeline_fn
