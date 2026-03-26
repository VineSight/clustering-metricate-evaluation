"""BERTopic-based pipeline for Labricate.

Provides the default clustering pipeline using BERTopic library
with custom UMAP and HDBSCAN/K-Means models.
"""

from typing import Any

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class BERTopicPipeline:
    """Default clustering pipeline using BERTopic library.

    Wraps BERTopic's UMAP + HDBSCAN/K-Means pipeline, extracting
    cluster labels and reduced embeddings for Metricate evaluation.
    Topic representation (c-TF-IDF) is skipped by default for speed.

    Example:
        >>> pipeline = BERTopicPipeline()
        >>> labels, reduced = pipeline(embeddings, config)
    """

    @staticmethod
    def default_config() -> dict[str, Any]:
        """Return default pipeline configuration."""
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
                "min_samples": 10,
                "cluster_selection_method": "eom",
                "metric": "euclidean",
            },
            "kmeans": {
                "n_clusters": 10,
            },
            "enable_topic_representation": False,
            "calculate_probabilities": False,
        }

    def __call__(
        self,
        embeddings: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the BERTopic pipeline.

        Args:
            embeddings: Input embeddings (n_samples, n_dims).
            config: Pipeline configuration dict.

        Returns:
            Tuple of (labels, reduced_embeddings):
                - labels: 1D array of cluster assignments (from topic_model.topics_)
                - reduced_embeddings: 2D array of UMAP-reduced embeddings

        Raises:
            ValueError: If config is invalid.
            RuntimeError: If clustering fails.
        """
        n_samples = embeddings.shape[0]
        random_seed = config.get("random_seed", 42)

        # Create UMAP model with config
        umap_config = config.get("umap", {})
        umap_model = UMAP(
            n_neighbors=umap_config.get("n_neighbors", 15),
            n_components=umap_config.get("n_components", 5),
            min_dist=umap_config.get("min_dist", 0.0),
            metric=umap_config.get("metric", "cosine"),
            repulsion_strength=umap_config.get("repulsion_strength", 1.0),
            low_memory=umap_config.get("low_memory", False),
            random_state=random_seed,
        )

        # Create clustering model based on algorithm choice
        clustering_algorithm = config.get("clustering_algorithm", "hdbscan")

        if clustering_algorithm == "hdbscan":
            hdbscan_config = config.get("hdbscan", {})
            clustering_model = HDBSCAN(
                min_cluster_size=hdbscan_config.get("min_cluster_size", 10),
                min_samples=hdbscan_config.get("min_samples", 10),
                cluster_selection_method=hdbscan_config.get("cluster_selection_method", "eom"),
                metric=hdbscan_config.get("metric", "euclidean"),
                prediction_data=True,
            )
        elif clustering_algorithm == "kmeans":
            kmeans_config = config.get("kmeans", {})
            clustering_model = KMeans(
                n_clusters=kmeans_config.get("n_clusters", 10),
                random_state=random_seed,
                n_init=10,
            )
        else:
            raise ValueError(
                f"Unknown clustering algorithm: {clustering_algorithm}. "
                f"Expected 'hdbscan' or 'kmeans'."
            )

        # Create a simple vectorizer that works with placeholder docs
        # Using min_df=1 ensures even single-word docs are processed
        vectorizer_model = CountVectorizer(min_df=1, stop_words=None)

        # Create BERTopic model
        # Skip topic representation for speed unless explicitly enabled
        calculate_probabilities = config.get("calculate_probabilities", False)

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=clustering_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=calculate_probabilities,
            verbose=False,
        )

        # Fit with pre-computed embeddings + placeholder docs
        # Use simple placeholder text (not empty) for c-TF-IDF to work
        placeholder_docs = [f"doc{i}" for i in range(n_samples)]
        topic_model.fit_transform(placeholder_docs, embeddings=embeddings)

        # Extract outputs
        labels = np.array(topic_model.topics_)
        reduced_embeddings = topic_model.umap_model.embedding_

        return labels, reduced_embeddings
