"""Embeddings loading utilities for Labricate.

Supports loading embeddings from NumPy arrays, CSV files, NPY files,
NPZ files, and pandas DataFrames.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_embeddings(
    source: np.ndarray | pd.DataFrame | str | Path,
    array_key: str | None = None,
) -> np.ndarray:
    """Load embeddings from various sources.
    
    Supports:
    - NumPy array: returned as-is (validated)
    - pandas DataFrame: extracted from columns
    - CSV file path: loaded and extracted from columns
    - NPY file path: loaded directly
    - NPZ file path: loaded with optional key selection
    
    Args:
        source: Embeddings source (array, DataFrame, or file path).
        array_key: For NPZ files, the key of the array to load.
            If None, uses 'embeddings' or the first array found.
            
    Returns:
        2D NumPy array of shape (n_samples, n_dims).
        
    Raises:
        ValueError: If source format is invalid or embeddings are wrong shape.
        FileNotFoundError: If file path doesn't exist.
    """
    if isinstance(source, np.ndarray):
        return _validate_embeddings_array(source)
    
    if isinstance(source, pd.DataFrame):
        return _load_from_dataframe(source)
    
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _load_from_csv(path)
        elif suffix == ".npy":
            return _load_from_npy(path)
        elif suffix == ".npz":
            return _load_from_npz(path, array_key)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: .csv, .npy, .npz"
            )
    
    raise ValueError(
        f"Unsupported source type: {type(source)}. "
        f"Expected np.ndarray, pd.DataFrame, or file path."
    )


def _validate_embeddings_array(arr: np.ndarray) -> np.ndarray:
    """Validate that array is a proper embeddings matrix."""
    if arr.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D array, got {arr.ndim}D with shape {arr.shape}"
        )
    
    if arr.shape[0] == 0:
        raise ValueError("Embeddings array is empty (0 samples)")
    
    if arr.shape[1] == 0:
        raise ValueError("Embeddings array has 0 dimensions")
    
    # Convert to float64 if needed
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    
    # Check for NaN/Inf
    if np.any(np.isnan(arr)):
        raise ValueError("Embeddings contain NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError("Embeddings contain infinite values")
    
    return arr


def _load_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Load embeddings from pandas DataFrame.
    
    Extracts columns that look like embedding dimensions:
    - Columns named dim_0, dim_1, ... dim_N
    - Or columns named embedding_0, embedding_1, ...
    - Or all numeric columns if above patterns not found
    """
    # Try dim_* pattern first
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if dim_cols:
        # Sort by dimension number
        dim_cols = sorted(dim_cols, key=lambda x: int(x.split("_")[1]))
        return _validate_embeddings_array(df[dim_cols].values)
    
    # Try embedding_* pattern
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    if emb_cols:
        emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_")[1]))
        return _validate_embeddings_array(df[emb_cols].values)
    
    # Fall back to all numeric columns (excluding common non-embedding columns)
    exclude = {"cluster_id", "label", "index", "id", "sample_id"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() not in exclude
    ]
    
    if not numeric_cols:
        raise ValueError(
            "Could not find embedding columns in DataFrame. "
            "Expected columns like 'dim_0', 'embedding_0', or numeric columns."
        )
    
    return _validate_embeddings_array(df[numeric_cols].values)


def _load_from_csv(path: Path) -> np.ndarray:
    """Load embeddings from CSV file."""
    df = pd.read_csv(path)
    return _load_from_dataframe(df)


def _load_from_npy(path: Path) -> np.ndarray:
    """Load embeddings from NPY file."""
    arr = np.load(path)
    return _validate_embeddings_array(arr)


def _load_from_npz(path: Path, array_key: str | None = None) -> np.ndarray:
    """Load embeddings from NPZ file."""
    data = np.load(path)
    
    if array_key is not None:
        if array_key not in data.files:
            raise ValueError(
                f"Array key '{array_key}' not found in NPZ file. "
                f"Available keys: {data.files}"
            )
        return _validate_embeddings_array(data[array_key])
    
    # Try 'embeddings' key first
    if "embeddings" in data.files:
        return _validate_embeddings_array(data["embeddings"])
    
    # Try 'data' key
    if "data" in data.files:
        return _validate_embeddings_array(data["data"])
    
    # Use first array
    if len(data.files) == 0:
        raise ValueError("NPZ file is empty")
    
    first_key = data.files[0]
    return _validate_embeddings_array(data[first_key])


def detect_ground_truth(
    source: pd.DataFrame | str | Path,
    label_column: str = "cluster_id",
) -> np.ndarray | None:
    """Detect ground truth labels if present in source.
    
    Args:
        source: DataFrame or CSV file path.
        label_column: Name of the label column to look for.
        
    Returns:
        1D array of labels if found, None otherwise.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists() or path.suffix.lower() != ".csv":
            return None
        df = pd.read_csv(path)
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        return None
    
    if label_column in df.columns:
        return df[label_column].values.astype(np.int64)
    
    return None
