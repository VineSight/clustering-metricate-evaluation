"""Checkpoint and resume functionality for Labricate experiments.

Provides save/load for experiment state to enable resuming
interrupted experiments.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metricate.labricate.core.experiment import RunResult

logger = logging.getLogger(__name__)


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a stable hash of a configuration dict.

    The hash is independent of key ordering.

    Args:
        config: Configuration dictionary.

    Returns:
        SHA256 hex digest string.
    """

    def sort_dict(d: Any) -> Any:
        """Recursively sort dict keys for stable hashing."""
        if isinstance(d, dict):
            return {k: sort_dict(v) for k, v in sorted(d.items())}
        if isinstance(d, list):
            return [sort_dict(v) for v in d]
        return d

    sorted_config = sort_dict(config)
    config_json = json.dumps(sorted_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode()).hexdigest()


@dataclass
class Checkpoint:
    """Represents a saved experiment checkpoint."""

    experiment_id: str
    config_hash: str
    completed_run_ids: list[int]
    timestamp: str

    def matches_config(self, config: dict[str, Any]) -> bool:
        """Check if checkpoint matches a configuration.

        Args:
            config: Configuration to compare.

        Returns:
            True if config hash matches.
        """
        return self.config_hash == compute_config_hash(config)

    def should_skip_run(self, run_id: int) -> bool:
        """Check if a run should be skipped (already completed).

        Args:
            run_id: Run ID to check.

        Returns:
            True if run was already completed.
        """
        return run_id in self.completed_run_ids


def save_checkpoint(
    path: str | Path,
    experiment_id: str,
    config: dict[str, Any],
    completed_runs: list[RunResult],
) -> None:
    """Save a checkpoint to disk.

    Args:
        path: Path to checkpoint file.
        experiment_id: Experiment identifier.
        config: Experiment configuration.
        completed_runs: List of completed RunResult objects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "experiment_id": experiment_id,
        "config_hash": compute_config_hash(config),
        "completed_run_ids": [r.run_id for r in completed_runs],
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }

    path.write_text(json.dumps(checkpoint_data, indent=2))
    logger.debug(f"Checkpoint saved: {path}")


def load_checkpoint(path: str | Path) -> Checkpoint | None:
    """Load a checkpoint from disk.

    Args:
        path: Path to checkpoint file.

    Returns:
        Checkpoint object, or None if file doesn't exist or is corrupt.
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return Checkpoint(
            experiment_id=data["experiment_id"],
            config_hash=data["config_hash"],
            completed_run_ids=data["completed_run_ids"],
            timestamp=data.get("timestamp", ""),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load checkpoint {path}: {e}")
        return None


def get_checkpoint_path(experiment_dir: str | Path) -> Path:
    """Get the standard checkpoint path for an experiment directory.

    Args:
        experiment_dir: Experiment output directory.

    Returns:
        Path to checkpoint.json file.
    """
    return Path(experiment_dir) / "checkpoint.json"
