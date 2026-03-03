"""
CSV data loading and validation for clustering evaluation.

This module handles loading clustering data from CSV files, validating
the data format, and auto-detecting column roles when not specified.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from metricate.core.exceptions import (
    ColumnNotFoundError,
    DimensionMismatchError,
    FileNotFoundError,
    InsufficientClustersError,
    InvalidCSVError,
)

# Common patterns for detecting column roles
LABEL_PATTERNS = [
    r"^cluster(?:_?id)?$",
    r"^(?:new_)?cluster$",
    r"^label(?:s)?$",
    r"^class$",
    r"^group(?:_?id)?$",
    r"^assignment$",
    r"^cluster_id$",
]

EMBEDDING_PATTERNS = [
    r"^(?:x|dim|d|emb|embed|embedding|component|pc|umap|tsne)_?\d+$",
    r"^\d+$",  # Just numeric column names
    r"^reduced_\d+$",
]

EXCLUDE_PATTERNS = [
    r"^(?:id|index|row|unnamed).*$",
    r"^text$",
    r"^content$",
    r"^title$",
    r"^post_id$",
    r"^platform$",
    r"^batch$",
    r"^.*_title$",
    r"^original_cluster$",
    r"^new_silhouette$",
]


def is_string_array_column(series: pd.Series) -> bool:
    """Check if a column contains string-encoded arrays like '[1.2, 3.4, ...]'."""
    # Check if dtype is object or string
    if series.dtype not in (object, "object", "str", "string"):
        # Also check for pandas string dtype
        if not pd.api.types.is_string_dtype(series):
            return False
    # Check first non-null value
    sample = series.dropna().iloc[0] if not series.dropna().empty else None
    if sample is None:
        return False
    if isinstance(sample, str) and sample.strip().startswith("[") and sample.strip().endswith("]"):
        return True
    return False


def parse_string_array(value: str) -> list[float]:
    """Parse a string-encoded array like '[1.2, 3.4, ...]' into a list of floats."""
    import ast

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Try manual parsing for malformed strings
        cleaned = value.strip("[]").split(",")
        return [float(x.strip()) for x in cleaned if x.strip()]


def expand_string_array_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Expand a string-encoded array column into multiple numeric columns.

    Args:
        df: DataFrame containing the column
        col: Name of the column to expand

    Returns:
        Tuple of (modified DataFrame, list of new column names)
    """
    # Parse the first row to determine array length
    sample = parse_string_array(df[col].iloc[0])
    n_dims = len(sample)

    # Create new column names
    new_cols = [f"{col}_{i}" for i in range(n_dims)]

    # Parse all values and create new columns
    parsed = df[col].apply(parse_string_array)
    for i, new_col in enumerate(new_cols):
        df[new_col] = parsed.apply(lambda x: x[i] if i < len(x) else np.nan)

    return df, new_cols


class ClusteringData:
    """Container for validated clustering data."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_col: str,
        embedding_cols: list[str],
        source_path: Path | None = None,
        original_df: pd.DataFrame | None = None,
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.label_col = label_col
        self.embedding_cols = embedding_cols
        self.source_path = source_path
        self.original_df = original_df

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def n_features(self) -> int:
        return self.embeddings.shape[1]

    @property
    def n_clusters(self) -> int:
        return len(np.unique(self.labels))

    @property
    def cluster_sizes(self) -> dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts, strict=False))


def detect_columns(
    df: pd.DataFrame,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Auto-detect label and embedding columns from a DataFrame.

    Args:
        df: Input DataFrame
        label_col: Override for label column (if known)
        embedding_cols: Override for embedding columns (if known)

    Returns:
        Tuple of (label_col, embedding_cols)

    Raises:
        ColumnNotFoundError: If required columns cannot be detected
    """
    columns = list(df.columns)

    # Detect label column if not specified
    if label_col is None:
        for pattern in LABEL_PATTERNS:
            for col in columns:
                if re.match(pattern, col.lower()):
                    label_col = col
                    break
            if label_col:
                break

        if label_col is None:
            raise ColumnNotFoundError(
                "label",
                available=columns,
                message="Could not auto-detect label column. "
                "Please specify --label-col explicitly.",
            )

    # Validate label column exists
    if label_col not in columns:
        raise ColumnNotFoundError(label_col, available=columns)

    # Detect embedding columns if not specified
    if embedding_cols is None:
        detected = []
        for col in columns:
            col_lower = col.lower()

            # Skip excluded columns
            if any(re.match(p, col_lower) for p in EXCLUDE_PATTERNS):
                continue

            # Skip label column
            if col == label_col:
                continue

            # Check if numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Check against embedding patterns
            for pattern in EMBEDDING_PATTERNS:
                if re.match(pattern, col_lower):
                    detected.append(col)
                    break
            else:
                # If no pattern matched but it's numeric and not excluded, include it
                if pd.api.types.is_numeric_dtype(df[col]):
                    detected.append(col)

        if not detected:
            # Fallback: use all numeric columns except label
            detected = [
                c for c in columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not detected:
            raise ColumnNotFoundError(
                "embedding",
                available=columns,
                message="Could not find numeric columns for embeddings.",
            )

        embedding_cols = detected

    # Validate embedding columns exist
    missing = [c for c in embedding_cols if c not in columns]
    if missing:
        raise ColumnNotFoundError(
            missing[0], available=columns, message=f"Embedding columns not found: {missing}"
        )

    return label_col, embedding_cols


def validate_data(
    df: pd.DataFrame,
    label_col: str,
    embedding_cols: list[str],
) -> list[str]:
    """
    Validate clustering data and return any warnings.

    Validation rules:
    - V-001: File must be valid CSV
    - V-002: Must have at least 2 data rows
    - V-003: Label column must exist
    - V-004: At least 2 embedding columns
    - V-005: No NaN in label column
    - V-006: No NaN in embedding columns (or warn)
    - V-007: At least 2 unique labels
    - V-008: Labels must be convertible to int

    Returns:
        List of warning messages
    """
    warnings = []

    # V-002: Minimum rows
    if len(df) < 2:
        raise InvalidCSVError("CSV must contain at least 2 data rows", details={"rows": len(df)})

    # V-004: Minimum embedding dimensions
    if len(embedding_cols) < 2:
        raise InvalidCSVError(
            "Need at least 2 embedding dimensions",
            details={"dimensions": len(embedding_cols), "columns": list(embedding_cols)},
        )

    # V-005: No NaN in labels
    nan_labels = df[label_col].isna().sum()
    if nan_labels > 0:
        raise InvalidCSVError(
            f"Label column '{label_col}' contains {nan_labels} missing values",
            details={"column": label_col, "missing": nan_labels},
        )

    # V-006: Check NaN in embeddings
    for col in embedding_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            warnings.append(
                f"Column '{col}' has {nan_count} missing values ({nan_count / len(df) * 100:.1f}%)"
            )

    # V-007: At least 2 clusters
    n_unique = df[label_col].nunique()
    if n_unique < 2:
        raise InsufficientClustersError(
            required=2, actual=n_unique, message="Need at least 2 clusters for meaningful metrics"
        )

    # V-008: Labels convertible to int
    try:
        df[label_col].astype(int)
    except (ValueError, TypeError) as e:
        raise InvalidCSVError(
            f"Label column '{label_col}' cannot be converted to integers",
            details={"column": label_col, "error": str(e)},
        )

    return warnings


def load_csv(
    path: str | Path,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    drop_nan: bool = True,
) -> ClusteringData:
    """
    Load and validate clustering data from a CSV file.

    Args:
        path: Path to CSV file
        label_col: Name of cluster label column (auto-detected if not specified)
        embedding_cols: List of embedding column names (auto-detected if not specified)
        drop_nan: If True, drop rows with NaN in embedding columns

    Returns:
        ClusteringData object with validated embeddings and labels

    Raises:
        FileNotFoundError: If file does not exist
        InvalidCSVError: If file is not valid CSV
        ColumnNotFoundError: If required columns not found
        InsufficientClustersError: If fewer than 2 clusters
    """
    path = Path(path)

    # V-001: File exists
    if not path.exists():
        raise FileNotFoundError(path)

    # V-001: Valid CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise InvalidCSVError(
            f"Failed to parse CSV: {e}", details={"path": str(path), "error": str(e)}
        )

    if df.empty:
        raise InvalidCSVError("CSV file is empty", details={"path": str(path)})

    # Check for string-encoded array columns and expand them
    # Prefer 'reduced_embedding' over 'embedding' if both exist
    string_array_cols = []
    for col in ["reduced_embedding", "embedding"]:
        if col in df.columns and is_string_array_column(df[col]):
            string_array_cols.append(col)

    expanded_cols = []
    # Expand if embedding_cols is None, or if it contains a single string-array column
    if string_array_cols:
        if embedding_cols is None:
            # Use the first available string array column (reduced_embedding preferred)
            col_to_expand = string_array_cols[0]
            df, expanded_cols = expand_string_array_column(df, col_to_expand)
            embedding_cols = expanded_cols
        elif len(embedding_cols) == 1 and embedding_cols[0] in string_array_cols:
            # Explicitly passed a single string-array column, expand it
            col_to_expand = embedding_cols[0]
            df, expanded_cols = expand_string_array_column(df, col_to_expand)
            embedding_cols = expanded_cols

    # Auto-detect columns
    label_col, embedding_cols = detect_columns(df, label_col, embedding_cols)

    # Validate data
    warnings = validate_data(df, label_col, embedding_cols)

    # Handle NaN values
    if drop_nan:
        mask = df[embedding_cols].notna().all(axis=1)
        dropped = (~mask).sum()
        if dropped > 0:
            warnings.append(f"Dropped {dropped} rows with NaN embeddings")
        df = df[mask]

    # Extract arrays
    embeddings = df[embedding_cols].values.astype(np.float64)
    labels = df[label_col].values.astype(int)

    data = ClusteringData(
        embeddings=embeddings,
        labels=labels,
        label_col=label_col,
        embedding_cols=list(embedding_cols),
        source_path=path,
        original_df=df,
    )

    return data


def load_comparison_pair(
    path_a: str | Path,
    path_b: str | Path,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
) -> tuple[ClusteringData, ClusteringData]:
    """
    Load two datasets for comparison, ensuring compatible structure.

    Args:
        path_a: Path to first CSV file
        path_b: Path to second CSV file
        label_col: Label column name (same for both)
        embedding_cols: Embedding columns (same for both)

    Returns:
        Tuple of (data_a, data_b)

    Raises:
        DimensionMismatchError: If datasets have incompatible structures
    """
    data_a = load_csv(path_a, label_col, embedding_cols)
    data_b = load_csv(path_b, label_col, embedding_cols)

    # Validate compatible dimensions
    if data_a.n_features != data_b.n_features:
        raise DimensionMismatchError(
            dimensions_a=data_a.n_features,
            dimensions_b=data_b.n_features,
        )

    return data_a, data_b
