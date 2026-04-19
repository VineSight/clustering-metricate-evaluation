"""Parallel execution utilities for Labricate.

Provides a joblib-based wrapper for running experiment runs in parallel
with proper error handling and worker count capping.

Note: joblib's default "loky" backend spawns separate processes, giving
each worker its own Numba runtime.  This avoids the crash that occurs
when Numba's workqueue threading layer is accessed concurrently from
multiple Python threads.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Literal, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def cap_worker_count(requested: int) -> int:
    """Cap worker count to available CPU cores.

    Args:
        requested: Requested number of workers.

    Returns:
        Actual number of workers (capped at CPU count, minimum 1).
    """
    cpu_count = os.cpu_count() or 1

    if requested <= 0:
        return 1

    if requested > cpu_count:
        logger.warning(
            f"Requested {requested} workers, capping to {cpu_count} (CPU count)"
        )
        return cpu_count

    return requested


class ParallelExecutor:
    """Joblib-based parallel executor with error handling.

    Uses joblib's "loky" backend (process-based) so that each worker
    has its own Numba runtime, avoiding workqueue threading conflicts.

    Example:
        >>> executor = ParallelExecutor(n_workers=4)
        >>> results = executor.map(lambda x: x * 2, [1, 2, 3])
        >>> print(results)  # [2, 4, 6]
    """

    def __init__(
        self,
        n_workers: int = 1,
        error_handling: Literal["continue", "fail_fast"] = "continue",
    ) -> None:
        """Initialize parallel executor.

        Args:
            n_workers: Number of parallel workers (capped at CPU count).
            error_handling: How to handle errors:
                - "continue": Record exception, continue with other tasks
                - "fail_fast": Raise on first error
        """
        self.n_workers = cap_worker_count(n_workers)
        self.error_handling = error_handling

    def __enter__(self) -> ParallelExecutor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""

    def map(
        self,
        func: Callable[[T], R],
        items: list[T],
        on_complete: Callable[[], None] | None = None,
    ) -> list[R | Exception]:
        """Execute function on items in parallel.

        Args:
            func: Function to apply to each item.
            items: List of items to process.
            on_complete: Optional callback invoked in the main process each
                time a single item finishes.  Use this to update a progress
                bar without sharing it with worker processes.

        Returns:
            List of results in same order as input.
            In "continue" mode, failed items contain the Exception.

        Raises:
            Exception: In "fail_fast" mode, re-raises first exception.
        """
        if self.n_workers == 1:
            # Sequential execution
            return self._map_sequential(func, items, on_complete=on_complete)

        return self._map_parallel(func, items, on_complete=on_complete)

    def _map_sequential(
        self,
        func: Callable[[T], R],
        items: list[T],
        on_complete: Callable[[], None] | None = None,
    ) -> list[R | Exception]:
        """Execute sequentially (n_workers=1)."""
        results: list[R | Exception] = []
        for item in items:
            try:
                results.append(func(item))
            except Exception as e:
                if self.error_handling == "fail_fast":
                    raise
                results.append(e)
            if on_complete:
                on_complete()
        return results

    def _map_parallel(
        self,
        func: Callable[[T], R],
        items: list[T],
        on_complete: Callable[[], None] | None = None,
    ) -> list[R | Exception]:
        """Execute in parallel using joblib (loky process-based backend).

        Each worker process has its own Numba runtime, so there is no
        workqueue threading conflict even when pipelines use UMAP or
        other Numba-parallel code.

        Results are collected in completion order for live on_complete
        callbacks, then reordered to match input order before returning.
        """
        from joblib import Parallel, delayed

        # Each task returns (original_index, result) so we can reorder later
        def _indexed(idx: int, item: T) -> tuple[int, R]:
            return idx, func(item)

        def _indexed_safe(idx: int, item: T) -> tuple[int, R | Exception]:
            try:
                return idx, func(item)
            except Exception as e:
                return idx, e

        worker = _indexed if self.error_handling == "fail_fast" else _indexed_safe

        results: list[R | Exception | None] = [None] * len(items)

        # return_as="generator_unordered" yields each result as it finishes
        gen = Parallel(n_jobs=self.n_workers, return_as="generator_unordered")(
            delayed(worker)(idx, item) for idx, item in enumerate(items)
        )

        for idx, result in gen:
            results[idx] = result
            if on_complete:
                on_complete()

        return results  # type: ignore


def run_parallel(
    func: Callable[[T], R],
    items: list[T],
    n_workers: int = 1,
    error_handling: Literal["continue", "fail_fast"] = "continue",
) -> list[R | Exception]:
    """Convenience function for parallel execution.

    Args:
        func: Function to apply to each item.
        items: List of items to process.
        n_workers: Number of parallel workers.
        error_handling: "continue" or "fail_fast".

    Returns:
        List of results in same order as input.
    """
    with ParallelExecutor(n_workers=n_workers, error_handling=error_handling) as ex:
        return ex.map(func, items)
