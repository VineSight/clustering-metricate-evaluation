"""Tests for Labricate configuration handling."""

import json

import pytest

from metricate.labricate.core.config import (
    DEFAULT_CONFIG,
    get_param,
    load_config,
    resolve_path,
    set_param,
    validate_config,
    validate_param_paths,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_dict(self, base_config):
        """Test loading from a dict."""
        config = load_config(base_config)
        assert config["umap"]["n_neighbors"] == base_config["umap"]["n_neighbors"]
        assert config["clustering_algorithm"] == "hdbscan"

    def test_load_from_json_file(self, tmp_path, base_config):
        """Test loading from JSON file."""
        json_path = tmp_path / "config.json"
        with open(json_path, "w") as f:
            json.dump(base_config, f)

        config = load_config(json_path)
        assert config["umap"]["n_neighbors"] == base_config["umap"]["n_neighbors"]

    def test_load_merges_with_defaults(self):
        """Test that partial config merges with defaults."""
        partial = {"umap": {"n_neighbors": 20}}
        config = load_config(partial)

        # Overridden value
        assert config["umap"]["n_neighbors"] == 20
        # Default values preserved
        assert config["umap"]["n_components"] == DEFAULT_CONFIG["umap"]["n_components"]
        assert config["clustering_algorithm"] == DEFAULT_CONFIG["clustering_algorithm"]

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_load_invalid_type_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError):
            load_config(123)  # type: ignore


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_hdbscan_config(self, base_config):
        """Test valid HDBSCAN config passes validation."""
        errors = validate_config(base_config)
        assert errors == []

    def test_valid_kmeans_config(self, kmeans_config):
        """Test valid K-Means config passes validation."""
        errors = validate_config(kmeans_config)
        assert errors == []

    def test_invalid_clustering_algorithm(self, base_config):
        """Test invalid clustering_algorithm fails validation."""
        base_config["clustering_algorithm"] = "invalid"
        errors = validate_config(base_config)
        assert len(errors) == 1
        assert "clustering_algorithm" in errors[0]

    def test_invalid_min_cluster_size(self, base_config):
        """Test min_cluster_size < 2 fails validation."""
        base_config["hdbscan"]["min_cluster_size"] = 1
        errors = validate_config(base_config)
        assert len(errors) == 1
        assert "min_cluster_size" in errors[0]

    def test_invalid_cluster_selection_method(self, base_config):
        """Test invalid cluster_selection_method fails validation."""
        base_config["hdbscan"]["cluster_selection_method"] = "invalid"
        errors = validate_config(base_config)
        assert len(errors) == 1
        assert "cluster_selection_method" in errors[0]

    def test_invalid_kmeans_n_clusters(self, kmeans_config):
        """Test invalid n_clusters fails validation."""
        kmeans_config["kmeans"]["n_clusters"] = 0
        errors = validate_config(kmeans_config)
        assert len(errors) == 1
        assert "n_clusters" in errors[0]

    def test_invalid_umap_n_neighbors(self, base_config):
        """Test invalid n_neighbors fails validation."""
        base_config["umap"]["n_neighbors"] = 1
        errors = validate_config(base_config)
        assert len(errors) == 1
        assert "n_neighbors" in errors[0]


class TestResolvePath:
    """Tests for dot-notation path resolution."""

    def test_resolve_top_level(self, base_config):
        """Test resolving top-level key."""
        parent, key = resolve_path(base_config, "clustering_algorithm")
        assert key == "clustering_algorithm"
        assert parent[key] == "hdbscan"

    def test_resolve_nested(self, base_config):
        """Test resolving nested key."""
        parent, key = resolve_path(base_config, "umap.n_neighbors")
        assert key == "n_neighbors"
        assert parent[key] == 15

    def test_resolve_invalid_section_raises(self, base_config):
        """Test that invalid section raises ValueError."""
        with pytest.raises(ValueError, match="section 'invalid' not found"):
            resolve_path(base_config, "invalid.n_neighbors")

    def test_resolve_invalid_key_raises(self, base_config):
        """Test that invalid key raises ValueError."""
        with pytest.raises(ValueError, match="key 'invalid' not found"):
            resolve_path(base_config, "umap.invalid")


class TestGetSetParam:
    """Tests for get_param and set_param functions."""

    def test_get_param(self, base_config):
        """Test getting parameter value."""
        value = get_param(base_config, "umap.n_neighbors")
        assert value == 15

    def test_set_param(self, base_config):
        """Test setting parameter value."""
        new_config = set_param(base_config, "umap.n_neighbors", 30)
        assert new_config["umap"]["n_neighbors"] == 30
        # Original unchanged
        assert base_config["umap"]["n_neighbors"] == 15

    def test_set_param_top_level(self, base_config):
        """Test setting top-level parameter."""
        new_config = set_param(base_config, "random_seed", 123)
        assert new_config["random_seed"] == 123


class TestValidateParamPaths:
    """Tests for validate_param_paths function."""

    def test_all_valid_paths(self, base_config):
        """Test that valid paths return empty list."""
        paths = ["umap.n_neighbors", "hdbscan.min_cluster_size", "random_seed"]
        invalid = validate_param_paths(base_config, paths)
        assert invalid == []

    def test_some_invalid_paths(self, base_config):
        """Test that invalid paths are returned."""
        paths = ["umap.n_neighbors", "invalid.path", "hdbscan.invalid"]
        invalid = validate_param_paths(base_config, paths)
        assert "invalid.path" in invalid
        assert "hdbscan.invalid" in invalid
        assert "umap.n_neighbors" not in invalid
