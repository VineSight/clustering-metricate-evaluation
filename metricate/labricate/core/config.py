"""Configuration handling and validation for Labricate.

Provides dot-notation path resolution and config validation.
"""

import copy
import json
from pathlib import Path
from typing import Any


# Default pipeline configuration
DEFAULT_CONFIG: dict[str, Any] = {
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

# Valid parameter paths for validation
VALID_PARAM_PATHS: set[str] = {
    "random_seed",
    "umap.n_neighbors",
    "umap.n_components",
    "umap.min_dist",
    "umap.metric",
    "umap.repulsion_strength",
    "umap.low_memory",
    "clustering_algorithm",
    "hdbscan.min_cluster_size",
    "hdbscan.min_samples",
    "hdbscan.cluster_selection_method",
    "hdbscan.metric",
    "kmeans.n_clusters",
    "enable_topic_representation",
    "calculate_probabilities",
}


def load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load configuration from dict or JSON file path.
    
    Args:
        config: Configuration dict or path to JSON file.
        
    Returns:
        Configuration dictionary with defaults applied.
        
    Raises:
        FileNotFoundError: If config path doesn't exist.
        ValueError: If config is not a valid dict or path.
    """
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict or path to JSON file, got {type(config)}")
    
    # Merge with defaults (user config overrides defaults)
    return _merge_config(DEFAULT_CONFIG, config)


def _merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override config into defaults."""
    result = copy.deepcopy(defaults)
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a pipeline configuration.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    # Check clustering_algorithm
    algorithm = config.get("clustering_algorithm")
    if algorithm not in ("hdbscan", "kmeans"):
        errors.append(
            f"clustering_algorithm must be 'hdbscan' or 'kmeans', got '{algorithm}'"
        )
    
    # Check HDBSCAN config if algorithm is hdbscan
    if algorithm == "hdbscan":
        hdbscan_config = config.get("hdbscan", {})
        min_cluster_size = hdbscan_config.get("min_cluster_size", 10)
        if min_cluster_size < 2:
            errors.append(
                f"hdbscan.min_cluster_size must be >= 2, got {min_cluster_size}"
            )
        
        cluster_selection = hdbscan_config.get("cluster_selection_method", "eom")
        if cluster_selection not in ("eom", "leaf"):
            errors.append(
                f"hdbscan.cluster_selection_method must be 'eom' or 'leaf', "
                f"got '{cluster_selection}'"
            )
    
    # Check K-Means config if algorithm is kmeans
    if algorithm == "kmeans":
        kmeans_config = config.get("kmeans", {})
        n_clusters = kmeans_config.get("n_clusters")
        if n_clusters is None or n_clusters < 1:
            errors.append(
                f"kmeans.n_clusters must be >= 1, got {n_clusters}"
            )
    
    # Check UMAP config
    umap_config = config.get("umap", {})
    n_neighbors = umap_config.get("n_neighbors", 15)
    if n_neighbors < 2:
        errors.append(f"umap.n_neighbors must be >= 2, got {n_neighbors}")
    
    n_components = umap_config.get("n_components", 5)
    if n_components < 1:
        errors.append(f"umap.n_components must be >= 1, got {n_components}")
    
    min_dist = umap_config.get("min_dist", 0.0)
    if min_dist < 0:
        errors.append(f"umap.min_dist must be >= 0, got {min_dist}")
    
    return errors


def resolve_path(config: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    """Resolve a dot-notation path to (parent_dict, key).
    
    Args:
        config: Configuration dictionary.
        path: Dot-notation path (e.g., "umap.n_neighbors").
        
    Returns:
        Tuple of (parent dictionary, final key).
        
    Raises:
        ValueError: If path is invalid.
    """
    parts = path.split(".")
    current = config
    
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ValueError(
                f"Invalid path: '{path}' - '{part}' is not a dict section"
            )
        if part not in current:
            raise ValueError(
                f"Invalid path: '{path}' - section '{part}' not found"
            )
        current = current[part]
    
    final_key = parts[-1]
    if not isinstance(current, dict):
        raise ValueError(
            f"Invalid path: '{path}' - parent is not a dict"
        )
    if final_key not in current:
        raise ValueError(
            f"Invalid path: '{path}' - key '{final_key}' not found"
        )
    
    return current, final_key


def get_param(config: dict[str, Any], path: str) -> Any:
    """Get a parameter value using dot notation.
    
    Args:
        config: Configuration dictionary.
        path: Dot-notation path (e.g., "umap.n_neighbors").
        
    Returns:
        The parameter value.
        
    Raises:
        ValueError: If path is invalid.
    """
    parent, key = resolve_path(config, path)
    return parent[key]


def set_param(config: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Set a parameter value using dot notation.
    
    Args:
        config: Configuration dictionary.
        path: Dot-notation path (e.g., "umap.n_neighbors").
        value: New value to set.
        
    Returns:
        New configuration with updated value (original unchanged).
        
    Raises:
        ValueError: If path is invalid.
    """
    config = copy.deepcopy(config)
    parent, key = resolve_path(config, path)
    parent[key] = value
    return config


def validate_param_paths(config: dict[str, Any], paths: list[str]) -> list[str]:
    """Validate that all parameter paths exist in config.
    
    Args:
        config: Configuration dictionary.
        paths: List of dot-notation paths to validate.
        
    Returns:
        List of invalid paths (empty if all valid).
    """
    invalid = []
    for path in paths:
        try:
            resolve_path(config, path)
        except ValueError:
            invalid.append(path)
    return invalid
