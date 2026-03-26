"""Logging and progress utilities for Labricate.

Provides progress bars, timing, and formatted output.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class TimingInfo:
    """Execution timing breakdown."""
    
    bertopic_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    total_seconds: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"BERTopic: {format_duration(self.bertopic_seconds)} | "
            f"Evaluation: {format_duration(self.evaluation_seconds)} | "
            f"Total: {format_duration(self.total_seconds)}"
        )


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted string like "1.23s", "1m 23s", or "1h 2m 3s".
    """
    if seconds < 0:
        return "0.0s"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def setup_progress(
    total: int,
    desc: str = "Running experiments",
    disable: bool = False,
) -> tqdm:
    """Create a progress bar for experiment runs.
    
    Args:
        total: Total number of iterations.
        desc: Description to show.
        disable: Whether to disable the progress bar.
        
    Returns:
        tqdm progress bar instance.
    """
    return tqdm(
        total=total,
        desc=desc,
        unit="run",
        disable=disable,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def log_run_start(
    run_id: int,
    total_runs: int,
    param_path: str,
    param_value: any,
    verbose: bool = True,
) -> None:
    """Log the start of a pipeline run.
    
    Args:
        run_id: Current run number (1-indexed).
        total_runs: Total number of runs.
        param_path: Parameter being varied.
        param_value: Current parameter value.
        verbose: Whether to print.
    """
    if verbose:
        tqdm.write(f"Run {run_id}/{total_runs}: {param_path}={param_value}")


def log_run_complete(
    run_id: int,
    timing: TimingInfo,
    n_clusters: int,
    n_noise: int,
    verbose: bool = True,
) -> None:
    """Log completion of a pipeline run.
    
    Args:
        run_id: Current run number (1-indexed).
        timing: Timing information for the run.
        n_clusters: Number of clusters found.
        n_noise: Number of noise points.
        verbose: Whether to print.
    """
    if verbose:
        tqdm.write(
            f"  → {n_clusters} clusters, {n_noise} noise | {timing}"
        )


def log_timing(
    stage: str,
    seconds: float,
    verbose: bool = True,
) -> None:
    """Log timing for a specific stage.
    
    Args:
        stage: Name of the stage (e.g., "UMAP", "Clustering").
        seconds: Duration in seconds.
        verbose: Whether to print.
    """
    if verbose:
        tqdm.write(f"  {stage}: {format_duration(seconds)}")


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager for timing code blocks.
    
    Example:
        >>> with timer() as t:
        ...     # do something
        >>> print(t["elapsed"])
        
    Yields:
        Dict with 'elapsed' key containing duration in seconds.
    """
    result: dict = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
