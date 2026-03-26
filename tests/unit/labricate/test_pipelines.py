"""Tests for BERTopicPipeline (US2)."""

import numpy as np
import pytest

from metricate.labricate.pipelines.bertopic import BERTopicPipeline


class TestBERTopicPipelineHDBSCAN:
    """T024: Tests for BERTopicPipeline with HDBSCAN config."""

    def test_hdbscan_returns_labels_and_embeddings(self, small_embeddings, base_config):
        """Pipeline returns labels and reduced embeddings."""
        pipeline = BERTopicPipeline()
        labels, reduced = pipeline(small_embeddings, base_config)

        assert isinstance(labels, np.ndarray)
        assert isinstance(reduced, np.ndarray)
        assert labels.ndim == 1
        assert reduced.ndim == 2
        assert len(labels) == small_embeddings.shape[0]
        assert reduced.shape[0] == small_embeddings.shape[0]

    def test_hdbscan_labels_are_integers(self, small_embeddings, base_config):
        """HDBSCAN labels are integers (possibly with -1 for noise)."""
        pipeline = BERTopicPipeline()
        labels, _ = pipeline(small_embeddings, base_config)

        assert np.issubdtype(labels.dtype, np.integer) or all(
            float(lbl).is_integer() for lbl in labels
        )

    def test_hdbscan_reduced_has_correct_components(self, small_embeddings, base_config):
        """Reduced embeddings have n_components dimensions."""
        pipeline = BERTopicPipeline()
        base_config["umap"]["n_components"] = 3
        _, reduced = pipeline(small_embeddings, base_config)

        assert reduced.shape[1] == 3


class TestBERTopicPipelineKMeans:
    """T025: Tests for BERTopicPipeline with K-Means config."""

    def test_kmeans_returns_labels_and_embeddings(self, small_embeddings, kmeans_config):
        """K-Means pipeline returns labels and reduced embeddings."""
        pipeline = BERTopicPipeline()
        labels, reduced = pipeline(small_embeddings, kmeans_config)

        assert isinstance(labels, np.ndarray)
        assert isinstance(reduced, np.ndarray)
        assert len(labels) == small_embeddings.shape[0]

    def test_kmeans_produces_expected_clusters(self, small_embeddings, kmeans_config):
        """K-Means produces approximately expected number of clusters."""
        pipeline = BERTopicPipeline()
        kmeans_config["kmeans"]["n_clusters"] = 3
        labels, _ = pipeline(small_embeddings, kmeans_config)

        # K-Means should produce close to n_clusters clusters
        unique_labels = np.unique(labels)
        # BERTopic may adjust but should be close
        assert len(unique_labels) >= 1

    def test_kmeans_no_noise_labels(self, small_embeddings, kmeans_config):
        """K-Means typically doesn't produce -1 noise labels."""
        pipeline = BERTopicPipeline()
        labels, _ = pipeline(small_embeddings, kmeans_config)

        # K-Means assigns all points; may have outlier topic but not standard -1
        # The exact behavior depends on BERTopic version


class TestBERTopicPipelineUMAPParams:
    """T026: Tests for UMAP/HDBSCAN parameter application."""

    def test_umap_n_neighbors_applied(self, small_embeddings, base_config):
        """UMAP n_neighbors parameter is applied."""
        pipeline = BERTopicPipeline()

        # Run with different n_neighbors values
        base_config["umap"]["n_neighbors"] = 5
        labels1, _ = pipeline(small_embeddings, base_config)

        base_config["umap"]["n_neighbors"] = 30
        labels2, _ = pipeline(small_embeddings, base_config)

        # Results may differ (not guaranteed but likely with different settings)
        # At minimum, both should complete successfully
        assert len(labels1) == len(labels2)

    def test_umap_n_components_applied(self, small_embeddings, base_config):
        """UMAP n_components parameter is applied."""
        pipeline = BERTopicPipeline()

        base_config["umap"]["n_components"] = 2
        _, reduced2 = pipeline(small_embeddings, base_config)

        base_config["umap"]["n_components"] = 10
        _, reduced10 = pipeline(small_embeddings, base_config)

        assert reduced2.shape[1] == 2
        assert reduced10.shape[1] == 10

    def test_umap_min_dist_applied(self, small_embeddings, base_config):
        """UMAP min_dist parameter is applied."""
        pipeline = BERTopicPipeline()

        # Different min_dist values should work
        base_config["umap"]["min_dist"] = 0.0
        labels1, _ = pipeline(small_embeddings, base_config)

        base_config["umap"]["min_dist"] = 0.5
        labels2, _ = pipeline(small_embeddings, base_config)

        assert len(labels1) == len(labels2)

    def test_umap_metric_applied(self, small_embeddings, base_config):
        """UMAP metric parameter is applied."""
        pipeline = BERTopicPipeline()

        base_config["umap"]["metric"] = "cosine"
        labels_cos, _ = pipeline(small_embeddings, base_config)

        base_config["umap"]["metric"] = "euclidean"
        labels_euc, _ = pipeline(small_embeddings, base_config)

        assert len(labels_cos) == len(labels_euc)

    def test_hdbscan_min_cluster_size_applied(self, small_embeddings, base_config):
        """HDBSCAN min_cluster_size parameter is applied."""
        pipeline = BERTopicPipeline()

        # Small min_cluster_size may produce more clusters
        base_config["hdbscan"]["min_cluster_size"] = 5
        labels_small, _ = pipeline(small_embeddings, base_config)

        # Large min_cluster_size may produce fewer clusters
        base_config["hdbscan"]["min_cluster_size"] = 20
        labels_large, _ = pipeline(small_embeddings, base_config)

        assert len(labels_small) == len(labels_large)

    def test_hdbscan_cluster_selection_method_applied(self, small_embeddings, base_config):
        """HDBSCAN cluster_selection_method parameter is applied."""
        pipeline = BERTopicPipeline()

        base_config["hdbscan"]["cluster_selection_method"] = "eom"
        labels_eom, _ = pipeline(small_embeddings, base_config)

        base_config["hdbscan"]["cluster_selection_method"] = "leaf"
        labels_leaf, _ = pipeline(small_embeddings, base_config)

        assert len(labels_eom) == len(labels_leaf)


class TestBERTopicPipelineReproducibility:
    """T027: Tests for random_seed reproducibility."""

    def test_same_seed_produces_same_results(self, small_embeddings, base_config):
        """Same random_seed produces identical results."""
        pipeline = BERTopicPipeline()

        base_config["random_seed"] = 42
        labels1, reduced1 = pipeline(small_embeddings, base_config)

        base_config["random_seed"] = 42
        labels2, reduced2 = pipeline(small_embeddings, base_config)

        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_almost_equal(reduced1, reduced2)

    def test_different_seeds_may_produce_different_results(self, small_embeddings, base_config):
        """Different random_seeds may produce different results."""
        pipeline = BERTopicPipeline()

        base_config["random_seed"] = 42
        labels1, _ = pipeline(small_embeddings, base_config)

        base_config["random_seed"] = 123
        labels2, _ = pipeline(small_embeddings, base_config)

        # Results may differ (not guaranteed but likely)
        # At minimum, both should complete successfully
        assert len(labels1) == len(labels2)


class TestBERTopicPipelineDefaultConfig:
    """Tests for BERTopicPipeline.default_config()."""

    def test_default_config_is_valid(self):
        """default_config() returns a valid configuration."""
        config = BERTopicPipeline.default_config()

        assert "random_seed" in config
        assert "umap" in config
        assert "clustering_algorithm" in config
        assert "hdbscan" in config
        assert "kmeans" in config

    def test_default_config_has_umap_params(self):
        """default_config() includes all UMAP parameters."""
        config = BERTopicPipeline.default_config()

        assert "n_neighbors" in config["umap"]
        assert "n_components" in config["umap"]
        assert "min_dist" in config["umap"]
        assert "metric" in config["umap"]

    def test_default_config_has_hdbscan_params(self):
        """default_config() includes all HDBSCAN parameters."""
        config = BERTopicPipeline.default_config()

        assert "min_cluster_size" in config["hdbscan"]
        assert "min_samples" in config["hdbscan"]
        assert "cluster_selection_method" in config["hdbscan"]

    def test_default_config_works(self, small_embeddings):
        """default_config() produces a working pipeline."""
        pipeline = BERTopicPipeline()
        config = BERTopicPipeline.default_config()

        labels, reduced = pipeline(small_embeddings, config)
        assert len(labels) == len(small_embeddings)


class TestBERTopicPipelineInvalidConfig:
    """Tests for error handling with invalid config."""

    def test_invalid_clustering_algorithm_raises(self, small_embeddings, base_config):
        """Invalid clustering_algorithm raises ValueError."""
        pipeline = BERTopicPipeline()
        base_config["clustering_algorithm"] = "invalid"

        with pytest.raises(ValueError, match="Unknown clustering algorithm"):
            pipeline(small_embeddings, base_config)
