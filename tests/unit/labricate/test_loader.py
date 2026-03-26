"""Tests for Labricate embeddings loader."""

import numpy as np
import pandas as pd
import pytest

from metricate.labricate.core.loader import (
    detect_ground_truth,
    load_embeddings,
)


class TestLoadEmbeddingsArray:
    """Tests for loading from NumPy arrays."""

    def test_load_valid_array(self, sample_embeddings):
        """Test loading valid 2D array."""
        result = load_embeddings(sample_embeddings)
        assert np.array_equal(result, sample_embeddings)

    def test_load_int_array_converts_to_float(self):
        """Test that integer arrays are converted to float."""
        int_array = np.array([[1, 2, 3], [4, 5, 6]])
        result = load_embeddings(int_array)
        assert result.dtype == np.float64

    def test_load_1d_array_raises(self):
        """Test that 1D array raises ValueError."""
        with pytest.raises(ValueError, match="2D array"):
            load_embeddings(np.array([1, 2, 3]))

    def test_load_3d_array_raises(self):
        """Test that 3D array raises ValueError."""
        with pytest.raises(ValueError, match="2D array"):
            load_embeddings(np.random.randn(10, 10, 10))

    def test_load_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            load_embeddings(np.array([]).reshape(0, 10))

    def test_load_array_with_nan_raises(self):
        """Test that array with NaN raises ValueError."""
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="NaN"):
            load_embeddings(arr)

    def test_load_array_with_inf_raises(self):
        """Test that array with Inf raises ValueError."""
        arr = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(ValueError, match="infinite"):
            load_embeddings(arr)


class TestLoadEmbeddingsDataFrame:
    """Tests for loading from pandas DataFrames."""

    def test_load_df_with_dim_columns(self, embeddings_dataframe):
        """Test loading DataFrame with dim_* columns."""
        result = load_embeddings(embeddings_dataframe)
        assert result.shape == (100, 384)

    def test_load_df_with_embedding_columns(self, sample_embeddings):
        """Test loading DataFrame with embedding_* columns."""
        n_samples, n_dims = sample_embeddings.shape
        df = pd.DataFrame(
            {f"embedding_{i}": sample_embeddings[:, i] for i in range(n_dims)}
        )
        result = load_embeddings(df)
        assert result.shape == sample_embeddings.shape

    def test_load_df_with_numeric_columns(self, sample_embeddings):
        """Test loading DataFrame with generic numeric columns."""
        n_samples, n_dims = sample_embeddings.shape
        # Use arbitrary column names
        df = pd.DataFrame(
            {f"col_{i}": sample_embeddings[:, i] for i in range(n_dims)}
        )
        result = load_embeddings(df)
        assert result.shape == sample_embeddings.shape

    def test_load_df_excludes_label_columns(self, sample_embeddings):
        """Test that cluster_id/label columns are excluded."""
        n_samples, n_dims = sample_embeddings.shape
        df = pd.DataFrame(
            {
                "cluster_id": np.zeros(n_samples),
                "label": np.zeros(n_samples),
                **{f"dim_{i}": sample_embeddings[:, i] for i in range(n_dims)},
            }
        )
        result = load_embeddings(df)
        assert result.shape == sample_embeddings.shape

    def test_load_df_no_numeric_columns_raises(self):
        """Test that DataFrame with no numeric columns raises."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        with pytest.raises(ValueError, match="Could not find embedding columns"):
            load_embeddings(df)


class TestLoadEmbeddingsCSV:
    """Tests for loading from CSV files."""

    def test_load_csv_with_dim_columns(self, embeddings_csv):
        """Test loading CSV with dim_* columns."""
        result = load_embeddings(embeddings_csv)
        assert result.shape == (100, 384)

    def test_load_csv_nonexistent_raises(self, tmp_path):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent.csv")


class TestLoadEmbeddingsNPY:
    """Tests for loading from NPY files."""

    def test_load_npy(self, embeddings_npy):
        """Test loading NPY file."""
        result = load_embeddings(embeddings_npy)
        assert result.shape == (100, 384)

    def test_load_npy_nonexistent_raises(self, tmp_path):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "nonexistent.npy")


class TestLoadEmbeddingsNPZ:
    """Tests for loading from NPZ files."""

    def test_load_npz_default_key(self, embeddings_npz):
        """Test loading NPZ with 'embeddings' key."""
        result = load_embeddings(embeddings_npz)
        assert result.shape == (100, 384)

    def test_load_npz_custom_key(self, tmp_path, sample_embeddings):
        """Test loading NPZ with custom key."""
        npz_path = tmp_path / "custom.npz"
        np.savez(npz_path, my_data=sample_embeddings)
        result = load_embeddings(npz_path, array_key="my_data")
        assert result.shape == sample_embeddings.shape

    def test_load_npz_invalid_key_raises(self, embeddings_npz):
        """Test that invalid array key raises ValueError."""
        with pytest.raises(ValueError, match="not found in NPZ"):
            load_embeddings(embeddings_npz, array_key="invalid")

    def test_load_npz_first_array_fallback(self, tmp_path, sample_embeddings):
        """Test fallback to first array when no 'embeddings' key."""
        npz_path = tmp_path / "first.npz"
        np.savez(npz_path, first_array=sample_embeddings)
        result = load_embeddings(npz_path)
        assert result.shape == sample_embeddings.shape


class TestLoadEmbeddingsUnsupported:
    """Tests for unsupported formats."""

    def test_unsupported_file_extension_raises(self, tmp_path):
        """Test that unsupported extension raises ValueError."""
        txt_path = tmp_path / "data.txt"
        txt_path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_embeddings(txt_path)

    def test_unsupported_type_raises(self):
        """Test that unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_embeddings([1, 2, 3])  # type: ignore


class TestDetectGroundTruth:
    """Tests for ground truth detection."""

    def test_detect_from_dataframe(self, sample_embeddings):
        """Test detecting ground truth from DataFrame."""
        df = pd.DataFrame(
            {
                "cluster_id": np.arange(100),
                **{f"dim_{i}": sample_embeddings[:, i] for i in range(10)},
            }
        )
        labels = detect_ground_truth(df)
        assert labels is not None
        assert len(labels) == 100

    def test_detect_from_csv(self, tmp_path, sample_embeddings):
        """Test detecting ground truth from CSV."""
        csv_path = tmp_path / "with_labels.csv"
        df = pd.DataFrame(
            {
                "cluster_id": np.arange(100),
                **{f"dim_{i}": sample_embeddings[:, i] for i in range(10)},
            }
        )
        df.to_csv(csv_path, index=False)
        labels = detect_ground_truth(csv_path)
        assert labels is not None
        assert len(labels) == 100

    def test_detect_no_label_column_returns_none(self, sample_embeddings):
        """Test that missing label column returns None."""
        df = pd.DataFrame(
            {f"dim_{i}": sample_embeddings[:, i] for i in range(10)}
        )
        labels = detect_ground_truth(df)
        assert labels is None

    def test_detect_unsupported_type_returns_none(self):
        """Test that unsupported type returns None."""
        labels = detect_ground_truth(np.array([1, 2, 3]))  # type: ignore
        assert labels is None


class TestIdenticalResultsAcrossFormats:
    """Tests for US3: identical results across all input formats."""

    def test_array_and_dataframe_identical(self, sample_embeddings):
        """Test NumPy array and DataFrame produce identical results."""
        # Load from array
        arr_result = load_embeddings(sample_embeddings)
        
        # Create DataFrame with dim_* columns
        df = pd.DataFrame(
            {f"dim_{i}": sample_embeddings[:, i] for i in range(sample_embeddings.shape[1])}
        )
        df_result = load_embeddings(df)
        
        assert np.allclose(arr_result, df_result)
        assert arr_result.shape == df_result.shape

    def test_csv_and_array_identical(self, tmp_path, sample_embeddings):
        """Test CSV and NumPy array produce identical results."""
        # Load from array
        arr_result = load_embeddings(sample_embeddings)
        
        # Save to CSV and load
        csv_path = tmp_path / "embeddings.csv"
        df = pd.DataFrame(
            {f"dim_{i}": sample_embeddings[:, i] for i in range(sample_embeddings.shape[1])}
        )
        df.to_csv(csv_path, index=False)
        csv_result = load_embeddings(csv_path)
        
        assert np.allclose(arr_result, csv_result)
        assert arr_result.shape == csv_result.shape

    def test_npy_and_array_identical(self, tmp_path, sample_embeddings):
        """Test NPY and NumPy array produce identical results."""
        # Load from array
        arr_result = load_embeddings(sample_embeddings)
        
        # Save to NPY and load
        npy_path = tmp_path / "embeddings.npy"
        np.save(npy_path, sample_embeddings)
        npy_result = load_embeddings(npy_path)
        
        assert np.allclose(arr_result, npy_result)
        assert arr_result.shape == npy_result.shape

    def test_npz_and_array_identical(self, tmp_path, sample_embeddings):
        """Test NPZ and NumPy array produce identical results."""
        # Load from array
        arr_result = load_embeddings(sample_embeddings)
        
        # Save to NPZ and load
        npz_path = tmp_path / "embeddings.npz"
        np.savez(npz_path, embeddings=sample_embeddings)
        npz_result = load_embeddings(npz_path)
        
        assert np.allclose(arr_result, npz_result)
        assert arr_result.shape == npz_result.shape

    def test_all_formats_produce_same_shape(self, tmp_path, sample_embeddings):
        """Test all formats produce identical shape and values."""
        expected_shape = sample_embeddings.shape
        
        # Array
        arr_result = load_embeddings(sample_embeddings)
        
        # DataFrame
        df = pd.DataFrame(
            {f"dim_{i}": sample_embeddings[:, i] for i in range(sample_embeddings.shape[1])}
        )
        df_result = load_embeddings(df)
        
        # CSV
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        csv_result = load_embeddings(csv_path)
        
        # NPY
        npy_path = tmp_path / "test.npy"
        np.save(npy_path, sample_embeddings)
        npy_result = load_embeddings(npy_path)
        
        # NPZ
        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, embeddings=sample_embeddings)
        npz_result = load_embeddings(npz_path)
        
        # All should have same shape
        assert arr_result.shape == expected_shape
        assert df_result.shape == expected_shape
        assert csv_result.shape == expected_shape
        assert npy_result.shape == expected_shape
        assert npz_result.shape == expected_shape
        
        # All should be numerically identical (or very close for CSV due to float precision)
        assert np.allclose(arr_result, df_result)
        assert np.allclose(arr_result, csv_result)
        assert np.allclose(arr_result, npy_result)
        assert np.allclose(arr_result, npz_result)
