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
