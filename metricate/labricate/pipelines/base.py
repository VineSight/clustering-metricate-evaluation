"""Base pipeline protocol for Labricate.

Defines the interface that all pipeline implementations must follow.
"""

from typing import Any, Protocol

import numpy as np


class Pipeline(Protocol):
    """Protocol for clustering pipelines.
    
    A pipeline takes embeddings and configuration, returning cluster labels
    and optionally reduced embeddings.
    
    Example:
        >>> class MyPipeline:
        ...     def __call__(
        ...         self, 
        ...         embeddings: np.ndarray, 
        ...         config: dict
        ...     ) -> tuple[np.ndarray, np.ndarray]:
        ...         # Your clustering logic here
        ...         labels = ...
        ...         reduced = ...
        ...         return labels, reduced
    """
    
    def __call__(
        self,
        embeddings: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute the pipeline.
        
        Args:
            embeddings: Input embeddings array of shape (n_samples, n_dims).
            config: Pipeline configuration dictionary.
            
        Returns:
            Tuple of:
                - labels: 1D array of cluster assignments, shape (n_samples,).
                  Use -1 for noise points.
                - reduced_embeddings: 2D array of reduced embeddings,
                  shape (n_samples, n_components).
                  
        Raises:
            ValueError: If embeddings or config are invalid.
            RuntimeError: If clustering fails.
        """
        ...


def validate_pipeline_output(
    labels: np.ndarray,
    reduced_embeddings: np.ndarray,
    n_samples: int,
) -> list[str]:
    """Validate pipeline output shapes and types.
    
    Args:
        labels: Cluster labels from pipeline.
        reduced_embeddings: Reduced embeddings from pipeline.
        n_samples: Expected number of samples.
        
    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    
    # Check labels
    if not isinstance(labels, np.ndarray):
        errors.append(f"labels must be numpy array, got {type(labels)}")
    elif labels.ndim != 1:
        errors.append(f"labels must be 1D array, got {labels.ndim}D")
    elif len(labels) != n_samples:
        errors.append(
            f"labels length ({len(labels)}) != n_samples ({n_samples})"
        )
    
    # Check reduced embeddings
    if not isinstance(reduced_embeddings, np.ndarray):
        errors.append(
            f"reduced_embeddings must be numpy array, got {type(reduced_embeddings)}"
        )
    elif reduced_embeddings.ndim != 2:
        errors.append(
            f"reduced_embeddings must be 2D array, got {reduced_embeddings.ndim}D"
        )
    elif reduced_embeddings.shape[0] != n_samples:
        errors.append(
            f"reduced_embeddings rows ({reduced_embeddings.shape[0]}) != "
            f"n_samples ({n_samples})"
        )
    
    return errors
