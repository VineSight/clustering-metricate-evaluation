"""Tests for checkpoint/resume functionality (Phase 9)."""

import json

import numpy as np
import pytest

from metricate.labricate.core.checkpoint import (
    Checkpoint,
    compute_config_hash,
    load_checkpoint,
    save_checkpoint,
)
from metricate.labricate.core.experiment import (
    MetricResult,
    PipelineResult,
    RunResult,
)
from metricate.labricate.utils.logging import TimingInfo


@pytest.fixture
def sample_run_result():
    """Create a sample RunResult for checkpoint testing."""
    return RunResult(
        run_id=1,
        param_values={"hdbscan.min_cluster_size": 10},
        pipeline_result=PipelineResult(
            run_id=1,
            config={"hdbscan": {"min_cluster_size": 10}},
            labels=np.array([0, 0, 1, 1, -1]),
            reduced_embeddings=np.random.randn(5, 3),
            n_clusters=2,
            n_noise=1,
            timing=TimingInfo(
                bertopic_seconds=1.0,
                evaluation_seconds=0.5,
                total_seconds=1.5,
            ),
        ),
        metrics=[
            MetricResult(name="silhouette", value=0.5, direction="higher"),
        ],
    )


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        "clustering_algorithm": "hdbscan",
        "umap": {"n_neighbors": 15, "n_components": 5},
        "hdbscan": {"min_cluster_size": 10},
    }


class TestComputeConfigHash:
    """Tests for config hash computation (T074)."""

    def test_same_config_same_hash(self, sample_config):
        """Same config produces same hash."""
        hash1 = compute_config_hash(sample_config)
        hash2 = compute_config_hash(sample_config)
        assert hash1 == hash2

    def test_different_config_different_hash(self, sample_config):
        """Different config produces different hash."""
        import copy

        modified = copy.deepcopy(sample_config)
        modified["hdbscan"]["min_cluster_size"] = 20
        hash1 = compute_config_hash(sample_config)
        hash2 = compute_config_hash(modified)
        assert hash1 != hash2

    def test_key_order_independent(self):
        """Hash is independent of key ordering."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}
        assert compute_config_hash(config1) == compute_config_hash(config2)

    def test_returns_hex_string(self, sample_config):
        """Returns a hex string."""
        hash_val = compute_config_hash(sample_config)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex length


class TestSaveCheckpoint:
    """Tests for checkpoint saving (T066)."""

    def test_save_creates_file(self, tmp_path, sample_run_result, sample_config):
        """Checkpoint file is created."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )
        assert checkpoint_path.exists()

    def test_save_contains_config_hash(self, tmp_path, sample_run_result, sample_config):
        """Checkpoint contains config hash."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )

        data = json.loads(checkpoint_path.read_text())
        assert "config_hash" in data
        assert data["config_hash"] == compute_config_hash(sample_config)

    def test_save_contains_completed_run_ids(
        self, tmp_path, sample_run_result, sample_config
    ):
        """Checkpoint contains completed run IDs."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )

        data = json.loads(checkpoint_path.read_text())
        assert "completed_run_ids" in data
        assert 1 in data["completed_run_ids"]


class TestLoadCheckpoint:
    """Tests for checkpoint loading (T067)."""

    def test_load_returns_checkpoint(self, tmp_path, sample_run_result, sample_config):
        """Loading returns Checkpoint object."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )

        checkpoint = load_checkpoint(checkpoint_path)
        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.experiment_id == "exp_001"
        assert 1 in checkpoint.completed_run_ids

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading nonexistent file returns None."""
        checkpoint = load_checkpoint(tmp_path / "nonexistent.json")
        assert checkpoint is None

    def test_load_corrupt_returns_none(self, tmp_path):
        """Loading corrupt file returns None."""
        checkpoint_path = tmp_path / "corrupt.json"
        checkpoint_path.write_text("not valid json")
        checkpoint = load_checkpoint(checkpoint_path)
        assert checkpoint is None


class TestConfigMismatch:
    """Tests for config mismatch detection (T068)."""

    def test_mismatch_detected(self, tmp_path, sample_run_result, sample_config):
        """Config mismatch is detected."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )

        checkpoint = load_checkpoint(checkpoint_path)

        # Create modified config
        modified_config = sample_config.copy()
        modified_config["hdbscan"]["min_cluster_size"] = 99

        assert not checkpoint.matches_config(modified_config)

    def test_match_detected(self, tmp_path, sample_run_result, sample_config):
        """Matching config is detected."""
        checkpoint_path = tmp_path / "checkpoint.json"
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )

        checkpoint = load_checkpoint(checkpoint_path)
        assert checkpoint.matches_config(sample_config)


class TestCheckpointIntegration:
    """Integration tests for checkpoint/resume workflow."""

    def test_checkpoint_skips_completed_runs(
        self, tmp_path, sample_run_result, sample_config
    ):
        """Checkpoint correctly identifies runs to skip."""
        checkpoint_path = tmp_path / "checkpoint.json"

        # Save checkpoint with run 1 completed
        save_checkpoint(
            checkpoint_path,
            experiment_id="exp_001",
            config=sample_config,
            completed_runs=[sample_run_result],
        )

        checkpoint = load_checkpoint(checkpoint_path)

        # Run 1 should be skipped, runs 2, 3 should not
        assert checkpoint.should_skip_run(1)
        assert not checkpoint.should_skip_run(2)
        assert not checkpoint.should_skip_run(3)
