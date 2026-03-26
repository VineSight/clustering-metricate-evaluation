"""Experiment orchestration and result types for Labricate.

Provides the Experiment class for running hyperparameter experiments
and dataclasses for storing results.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from metricate.labricate.core.config import (
    load_config,
    resolve_path,
    set_param,
    validate_config,
    validate_param_paths,
)
from metricate.labricate.core.loader import load_embeddings
from metricate.labricate.pipelines.base import validate_pipeline_output
from metricate.labricate.utils.logging import (
    TimingInfo,
    format_duration,
    log_run_complete,
    log_run_start,
    setup_progress,
    timer,
)

if TYPE_CHECKING:
    from metricate.labricate.pipelines.base import Pipeline


@dataclass
class MetricResult:
    """Single metric evaluation result from Metricate."""

    name: str
    value: float
    range: tuple[float, float] = (0.0, 1.0)
    direction: str = "higher"  # "higher" or "lower" is better


@dataclass
class PipelineResult:
    """Output from a single pipeline run."""

    run_id: int
    config: dict[str, Any]
    labels: np.ndarray
    reduced_embeddings: np.ndarray
    n_clusters: int
    n_noise: int
    timing: TimingInfo
    status: str = "completed"  # "completed", "failed", "skipped"
    error: str | None = None


@dataclass
class RunResult:
    """Complete result for a single experiment run."""

    run_id: int
    param_values: dict[str, Any]
    pipeline_result: PipelineResult
    metrics: list[MetricResult]


@dataclass
class ExperimentSummary:
    """Aggregated experiment statistics."""

    total_runs: int
    completed_runs: int
    failed_runs: int
    skipped_runs: int
    total_duration_seconds: float

    def __str__(self) -> str:
        return (
            f"ExperimentSummary(\n"
            f"    total_runs={self.total_runs},\n"
            f"    completed_runs={self.completed_runs},\n"
            f"    failed_runs={self.failed_runs},\n"
            f"    total_duration_seconds={self.total_duration_seconds:.1f}\n"
            f")"
        )


@dataclass
class ExperimentResult:
    """Complete experiment output."""

    experiment_id: str
    experiment_name: str
    config: dict[str, Any]
    runs: list[RunResult]
    summary: ExperimentSummary
    output_path: str | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with one row per run, columns for param values and metrics.
        """
        rows = []
        for run in self.runs:
            row = {"run_id": run.run_id}
            row.update(run.param_values)
            row["n_clusters"] = run.pipeline_result.n_clusters
            row["n_noise"] = run.pipeline_result.n_noise
            row["status"] = run.pipeline_result.status
            for metric in run.metrics:
                row[metric.name] = metric.value
            rows.append(row)
        return pd.DataFrame(rows)

    def get_best_run(
        self,
        metric: str,
        direction: str | None = None,
    ) -> RunResult:
        """Get the run with the best value for a metric.

        Args:
            metric: Metric name to optimize.
            direction: "higher" or "lower". Auto-detected if None.

        Returns:
            RunResult with best metric value.

        Raises:
            ValueError: If metric not found in results.
        """
        completed_runs = [r for r in self.runs if r.pipeline_result.status == "completed"]
        if not completed_runs:
            raise ValueError("No completed runs to compare")

        # Find direction from first run's metrics if not provided
        if direction is None:
            for m in completed_runs[0].metrics:
                if m.name == metric:
                    direction = m.direction
                    break
            if direction is None:
                direction = "higher"  # Default assumption

        # Find best run
        def get_metric_value(run: RunResult) -> float:
            for m in run.metrics:
                if m.name == metric:
                    return m.value
            raise ValueError(f"Metric '{metric}' not found in run {run.run_id}")

        if direction == "higher":
            return max(completed_runs, key=get_metric_value)
        else:
            return min(completed_runs, key=get_metric_value)


class Experiment:
    """Hyperparameter experimentation orchestrator.

    Example:
        >>> exp = Experiment(embeddings, config)
        >>> result = exp.run(param="hdbscan.min_cluster_size", values=[5, 10, 15, 20])
        >>> print(result.summary)
    """

    def __init__(
        self,
        embeddings: np.ndarray | pd.DataFrame | str | Path,
        config: dict[str, Any] | str | Path,
        name: str | None = None,
        output_dir: str | Path = "./experiments",
        output_format: Literal["json", "csv", "both"] = "json",
        pipeline: Pipeline | Callable | None = None,
    ) -> None:
        """Initialize an experiment.

        Args:
            embeddings: Input embeddings (array, DataFrame, or file path).
            config: Base pipeline configuration (dict or JSON path).
            name: Experiment name (auto-generated if None).
            output_dir: Directory for results.
            output_format: Output format ("json", "csv", "both").
            pipeline: Custom pipeline function (default: BERTopicPipeline).

        Raises:
            ValueError: If embeddings or config are invalid.
            FileNotFoundError: If file paths don't exist.
        """
        # Load embeddings
        self.embeddings = load_embeddings(embeddings)
        self.n_samples = self.embeddings.shape[0]

        # Load and validate config
        self.config = load_config(config)
        errors = validate_config(self.config)
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        # Set experiment metadata
        self.name = name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir)
        self.output_format = output_format

        # Set pipeline (lazy import to avoid circular deps)
        if pipeline is None:
            from metricate.labricate.pipelines.bertopic import BERTopicPipeline

            self.pipeline = BERTopicPipeline()
        else:
            self.pipeline = pipeline

    def validate_param(self, param: str) -> bool:
        """Validate a parameter path against the config.

        Args:
            param: Dot-notation parameter path.

        Returns:
            True if valid.

        Raises:
            ValueError: If path is invalid.
        """
        try:
            resolve_path(self.config, param)
            return True
        except ValueError as e:
            raise ValueError(f"Invalid parameter path: {e}") from e

    def run(
        self,
        param: str,
        values: list[Any],
        n_workers: int = 1,
        error_handling: Literal["continue", "fail_fast"] = "continue",
        include_metrics: list[str] | None = None,
        exclude_metrics: list[str] | None = None,
        ground_truth: np.ndarray | None = None,
        resume: bool = False,
        force: bool = False,
        verbose: bool = True,
    ) -> ExperimentResult:
        """Run a single-parameter experiment.

        Args:
            param: Dot-notation parameter path (e.g., "hdbscan.min_cluster_size").
            values: List of values to test.
            n_workers: Number of parallel workers (1 = sequential).
            error_handling: How to handle failed runs ("continue" or "fail_fast").
            include_metrics: Specific metrics to calculate (None = all).
            exclude_metrics: Metrics to skip.
            ground_truth: Ground truth labels for supervised metrics.
            resume: Whether to resume from checkpoint.
            force: Force start fresh if config mismatch on resume.
            verbose: Whether to print progress.

        Returns:
            ExperimentResult with all run results and metrics.

        Raises:
            ValueError: If param path is invalid or values is empty.
        """
        # Validate inputs
        if not values:
            raise ValueError("values list cannot be empty")

        invalid_paths = validate_param_paths(self.config, [param])
        if invalid_paths:
            raise ValueError(f"Invalid parameter path: '{param}'")

        # Generate experiment ID
        experiment_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Import metricate for evaluation
        from metricate import evaluate

        # Track results
        runs: list[RunResult] = []
        total_runs = len(values)
        completed = 0
        failed = 0
        start_time = datetime.now()

        # Progress bar
        pbar = setup_progress(total_runs, desc=f"Running {self.name}", disable=not verbose)

        for run_id, value in enumerate(values, start=1):
            # Log start
            log_run_start(run_id, total_runs, param, value, verbose)

            # Create run config
            run_config = set_param(self.config, param, value)

            try:
                # Run pipeline with timing
                with timer() as pipeline_timer:
                    labels, reduced_embeddings = self.pipeline(self.embeddings, run_config)

                # Validate pipeline output (T045: output shape validation)
                validation_errors = validate_pipeline_output(
                    labels, reduced_embeddings, self.n_samples
                )
                if validation_errors:
                    raise ValueError(
                        f"Invalid pipeline output: {'; '.join(validation_errors)}"
                    )

                # Count clusters and noise
                unique_labels = np.unique(labels)
                n_clusters = len([lbl for lbl in unique_labels if lbl >= 0])
                n_noise = int(np.sum(labels == -1))

                # Evaluate with Metricate
                with timer() as eval_timer:
                    import tempfile
                    
                    # Create DataFrame for Metricate
                    eval_df = pd.DataFrame(
                        {
                            "cluster_id": labels,
                            **{
                                f"dim_{i}": reduced_embeddings[:, i]
                                for i in range(reduced_embeddings.shape[1])
                            },
                        }
                    )

                    # Save to temp file for metricate (it expects CSV path)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", delete=False
                    ) as tmp:
                        eval_df.to_csv(tmp.name, index=False)
                        tmp_path = tmp.name

                    try:
                        # Run evaluation
                        eval_result = evaluate(
                            tmp_path,
                            label_col="cluster_id",
                            exclude=exclude_metrics,
                        )
                    finally:
                        # Clean up temp file
                        Path(tmp_path).unlink(missing_ok=True)

                # Create timing info
                timing = TimingInfo(
                    bertopic_seconds=pipeline_timer["elapsed"],
                    evaluation_seconds=eval_timer["elapsed"],
                    total_seconds=pipeline_timer["elapsed"] + eval_timer["elapsed"],
                )

                # Create pipeline result
                pipeline_result = PipelineResult(
                    run_id=run_id,
                    config=run_config,
                    labels=labels,
                    reduced_embeddings=reduced_embeddings,
                    n_clusters=n_clusters,
                    n_noise=n_noise,
                    timing=timing,
                    status="completed",
                )

                # Extract metrics from Metricate result
                # EvaluationResult.metrics is a list of MetricValue objects
                metrics: list[MetricResult] = []
                for mv in eval_result.computed_metrics():
                    if include_metrics and mv.metric not in include_metrics:
                        continue
                    metrics.append(
                        MetricResult(
                            name=mv.metric,
                            value=float(mv.value) if mv.value is not None else 0.0,
                            range=(0.0, 1.0),  # Default, could parse mv.range
                            direction=mv.direction or "higher",
                        )
                    )

                # Create run result
                run_result = RunResult(
                    run_id=run_id,
                    param_values={param: value},
                    pipeline_result=pipeline_result,
                    metrics=metrics,
                )
                runs.append(run_result)
                completed += 1

                # Log completion
                log_run_complete(run_id, timing, n_clusters, n_noise, verbose)

            except Exception as e:
                failed += 1

                # Create failed result
                pipeline_result = PipelineResult(
                    run_id=run_id,
                    config=run_config,
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=0,
                    n_noise=0,
                    timing=TimingInfo(),
                    status="failed",
                    error=str(e),
                )

                run_result = RunResult(
                    run_id=run_id,
                    param_values={param: value},
                    pipeline_result=pipeline_result,
                    metrics=[],
                )
                runs.append(run_result)

                if error_handling == "fail_fast":
                    pbar.close()
                    raise RuntimeError(f"Run {run_id} failed: {e}") from e
                else:
                    if verbose:
                        from tqdm import tqdm

                        tqdm.write(f"  ✗ Run {run_id} failed: {e}")

            pbar.update(1)

        pbar.close()

        # Create summary
        total_duration = (datetime.now() - start_time).total_seconds()
        summary = ExperimentSummary(
            total_runs=total_runs,
            completed_runs=completed,
            failed_runs=failed,
            skipped_runs=0,
            total_duration_seconds=total_duration,
        )

        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=self.name,
            config={
                "base_config": self.config,
                "param": param,
                "values": values,
                "output_format": self.output_format,
                "n_workers": n_workers,
                "error_handling": error_handling,
            },
            runs=runs,
            summary=summary,
        )

        if verbose:
            print(f"\n✓ Experiment complete: {completed}/{total_runs} runs succeeded")
            print(f"  Total duration: {format_duration(total_duration)}")

        return result

    def run_grid(
        self,
        params: dict[str, list[Any]],
        n_workers: int = 1,
        error_handling: Literal["continue", "fail_fast"] = "continue",
        include_metrics: list[str] | None = None,
        exclude_metrics: list[str] | None = None,
        ground_truth: np.ndarray | None = None,
        resume: bool = False,
        force: bool = False,
        verbose: bool = True,
    ) -> ExperimentResult:
        """Run a grid search over multiple parameters.

        Args:
            params: Dict mapping param paths to lists of values.
                Example: {"hdbscan.min_cluster_size": [5, 10], "umap.n_neighbors": [10, 15]}
            n_workers: Number of parallel workers (1 = sequential).
            error_handling: How to handle failed runs ("continue" or "fail_fast").
            include_metrics: Specific metrics to calculate (None = all).
            exclude_metrics: Metrics to skip.
            ground_truth: Ground truth labels for supervised metrics.
            resume: Whether to resume from checkpoint.
            force: Force start fresh if config mismatch on resume.
            verbose: Whether to print progress.

        Returns:
            ExperimentResult with all run results and metrics.

        Raises:
            ValueError: If params is empty, any param path is invalid, or values is empty.
        """
        import itertools

        # Validate inputs
        if not params:
            raise ValueError("params dict cannot be empty")

        for param, values in params.items():
            if not values:
                raise ValueError(f"values list for param '{param}' cannot be empty")

        # Validate all parameter paths
        invalid_paths = validate_param_paths(self.config, list(params.keys()))
        if invalid_paths:
            raise ValueError(f"Invalid parameter paths: {', '.join(invalid_paths)}")

        # Generate all parameter combinations using itertools.product
        param_names = list(params.keys())
        param_values_lists = [params[name] for name in param_names]
        combinations = list(itertools.product(*param_values_lists))

        # Generate experiment ID
        experiment_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Import metricate for evaluation
        from metricate import evaluate

        # Track results
        runs: list[RunResult] = []
        total_runs = len(combinations)
        completed = 0
        failed = 0
        start_time = datetime.now()

        # Progress bar
        pbar = setup_progress(total_runs, desc=f"Grid search {self.name}", disable=not verbose)

        for run_id, combo_values in enumerate(combinations, start=1):
            # Create param_values dict for this combination
            param_values_dict = dict(zip(param_names, combo_values, strict=True))

            if verbose:
                params_str = ", ".join(f"{k}={v}" for k, v in param_values_dict.items())
                from tqdm import tqdm
                tqdm.write(f"Run {run_id}/{total_runs}: {params_str}")

            # Create run config by applying all param values
            run_config = self.config.copy()
            for param, value in param_values_dict.items():
                run_config = set_param(run_config, param, value)

            try:
                # Run pipeline with timing
                with timer() as pipeline_timer:
                    labels, reduced_embeddings = self.pipeline(self.embeddings, run_config)

                # Validate pipeline output
                validation_errors = validate_pipeline_output(
                    labels, reduced_embeddings, self.n_samples
                )
                if validation_errors:
                    raise ValueError(
                        f"Invalid pipeline output: {'; '.join(validation_errors)}"
                    )

                # Count clusters and noise
                unique_labels = np.unique(labels)
                n_clusters = len([lbl for lbl in unique_labels if lbl >= 0])
                n_noise = int(np.sum(labels == -1))

                # Evaluate with Metricate
                with timer() as eval_timer:
                    import tempfile

                    # Create DataFrame for Metricate
                    eval_df = pd.DataFrame(
                        {
                            "cluster_id": labels,
                            **{
                                f"dim_{i}": reduced_embeddings[:, i]
                                for i in range(reduced_embeddings.shape[1])
                            },
                        }
                    )

                    # Save to temp file for metricate (it expects CSV path)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", delete=False
                    ) as tmp:
                        eval_df.to_csv(tmp.name, index=False)
                        tmp_path = tmp.name

                    try:
                        # Run evaluation
                        eval_result = evaluate(
                            tmp_path,
                            label_col="cluster_id",
                            exclude=exclude_metrics,
                        )
                    finally:
                        # Clean up temp file
                        Path(tmp_path).unlink(missing_ok=True)

                # Create timing info
                timing = TimingInfo(
                    bertopic_seconds=pipeline_timer["elapsed"],
                    evaluation_seconds=eval_timer["elapsed"],
                    total_seconds=pipeline_timer["elapsed"] + eval_timer["elapsed"],
                )

                # Create pipeline result
                pipeline_result = PipelineResult(
                    run_id=run_id,
                    config=run_config,
                    labels=labels,
                    reduced_embeddings=reduced_embeddings,
                    n_clusters=n_clusters,
                    n_noise=n_noise,
                    timing=timing,
                    status="completed",
                )

                # Extract metrics from Metricate result
                metrics: list[MetricResult] = []
                for mv in eval_result.computed_metrics():
                    if include_metrics and mv.metric not in include_metrics:
                        continue
                    metrics.append(
                        MetricResult(
                            name=mv.metric,
                            value=float(mv.value) if mv.value is not None else 0.0,
                            range=(0.0, 1.0),
                            direction=mv.direction or "higher",
                        )
                    )

                # Create run result
                run_result = RunResult(
                    run_id=run_id,
                    param_values=param_values_dict,
                    pipeline_result=pipeline_result,
                    metrics=metrics,
                )
                runs.append(run_result)
                completed += 1

                # Log completion
                log_run_complete(run_id, timing, n_clusters, n_noise, verbose)

            except Exception as e:
                failed += 1

                # Create failed result
                pipeline_result = PipelineResult(
                    run_id=run_id,
                    config=run_config,
                    labels=np.array([]),
                    reduced_embeddings=np.array([[]]),
                    n_clusters=0,
                    n_noise=0,
                    timing=TimingInfo(),
                    status="failed",
                    error=str(e),
                )

                run_result = RunResult(
                    run_id=run_id,
                    param_values=param_values_dict,
                    pipeline_result=pipeline_result,
                    metrics=[],
                )
                runs.append(run_result)

                if error_handling == "fail_fast":
                    pbar.close()
                    raise RuntimeError(f"Run {run_id} failed: {e}") from e
                else:
                    if verbose:
                        from tqdm import tqdm

                        tqdm.write(f"  ✗ Run {run_id} failed: {e}")

            pbar.update(1)

        pbar.close()

        # Create summary
        total_duration = (datetime.now() - start_time).total_seconds()
        summary = ExperimentSummary(
            total_runs=total_runs,
            completed_runs=completed,
            failed_runs=failed,
            skipped_runs=0,
            total_duration_seconds=total_duration,
        )

        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=self.name,
            config={
                "base_config": self.config,
                "params": params,
                "output_format": self.output_format,
                "n_workers": n_workers,
                "error_handling": error_handling,
            },
            runs=runs,
            summary=summary,
        )

        if verbose:
            print(f"\n✓ Grid search complete: {completed}/{total_runs} runs succeeded")
            print(f"  Total duration: {format_duration(total_duration)}")

        return result
