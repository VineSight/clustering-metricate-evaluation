"""Storage utilities for Labricate experiment results.

Provides JSON and CSV output formats and directory structure management.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from metricate.labricate.core.experiment import ExperimentResult, RunResult


def create_experiment_directory(
    base_dir: str | Path,
    experiment_name: str,
    timestamp: str | None = None,
) -> Path:
    """Create hierarchical experiment directory structure.

    Creates: base_dir/experiments/<experiment_name>/<timestamp>/

    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        timestamp: Optional timestamp string (auto-generated if None).

    Returns:
        Path to the created experiment directory.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = Path(base_dir) / "experiments" / experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_results_json(
    result: "ExperimentResult",
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save experiment results to JSON file.

    Args:
        result: ExperimentResult to save.
        output_dir: Directory to save the file.
        filename: Optional filename (default: results.json).

    Returns:
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "results.json"

    output_path = output_dir / filename

    # Convert result to serializable dict
    data = _result_to_dict(result)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)

    return output_path


def save_results_csv(
    result: "ExperimentResult",
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save experiment results summary to CSV file.

    Creates a CSV with one row per run, columns for param values and metrics.

    Args:
        result: ExperimentResult to save.
        output_dir: Directory to save the file.
        filename: Optional filename (default: results.csv).

    Returns:
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "results.csv"

    output_path = output_dir / filename

    # Use the ExperimentResult's to_dataframe method
    df = result.to_dataframe()
    df.to_csv(output_path, index=False)

    return output_path


def save_run_csv(
    run: "RunResult",
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save intermediate clustering results for a single run.

    Creates a CSV with cluster_id and reduced embedding columns.

    Args:
        run: RunResult to save.
        output_dir: Directory to save the file.
        filename: Optional filename (default: run_<run_id>.csv).

    Returns:
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"run_{run.run_id:03d}.csv"

    output_path = output_dir / filename

    # Create DataFrame with labels and reduced embeddings
    pr = run.pipeline_result
    n_dims = pr.reduced_embeddings.shape[1] if pr.reduced_embeddings.ndim == 2 else 0

    data = {"cluster_id": pr.labels}
    for i in range(n_dims):
        data[f"dim_{i}"] = pr.reduced_embeddings[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    return output_path


def _result_to_dict(result: "ExperimentResult") -> dict[str, Any]:
    """Convert ExperimentResult to a serializable dictionary."""
    data = {
        "experiment_id": result.experiment_id,
        "experiment_name": result.experiment_name,
        "config": result.config,
        "runs": [_run_to_dict(run) for run in result.runs],
        "summary": {
            "total_runs": result.summary.total_runs,
            "completed_runs": result.summary.completed_runs,
            "failed_runs": result.summary.failed_runs,
            "skipped_runs": result.summary.skipped_runs,
            "total_duration_seconds": result.summary.total_duration_seconds,
        },
        "output_path": result.output_path,
    }
    # Add best_run if present (FR-004)
    if result.best_run is not None:
        data["best_run"] = {
            "run_id": result.best_run.run_id,
            "param_values": result.best_run.param_values,
            "score": result.best_run.score,
            "score_type": result.best_run.score_type,
            "tied_run_ids": result.best_run.tied_run_ids,
        }
    else:
        data["best_run"] = None
    return data


def _run_to_dict(run: "RunResult") -> dict[str, Any]:
    """Convert RunResult to a serializable dictionary."""
    pr = run.pipeline_result
    return {
        "run_id": run.run_id,
        "param_values": run.param_values,
        "compound_score": run.compound_score,  # None if no weights
        "pipeline_result": {
            "run_id": pr.run_id,
            "config": pr.config,
            "n_clusters": pr.n_clusters,
            "n_noise": pr.n_noise,
            "timing": {
                "bertopic_seconds": pr.timing.bertopic_seconds,
                "evaluation_seconds": pr.timing.evaluation_seconds,
                "total_seconds": pr.timing.total_seconds,
            },
            "status": pr.status,
            "error": pr.error,
        },
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "range": list(m.range),
                "direction": m.direction,
            }
            for m in run.metrics
        ],
    }


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types and other special objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
