"""Parallel execution utilities for Labricate.

Provides a ThreadPoolExecutor wrapper for running experiment
runs in parallel with proper error handling and worker count capping.

Note: ThreadPoolExecutor is used instead of ProcessPoolExecutor because
experiment pipelines involve complex objects (BERTopic models, numpy arrays)
that are easier to share via threads than to pickle across processes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Wrapper around ProcessPoolExecutor with error handling.

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
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> ParallelExecutor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the executor if active."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map(
        self,
        func: Callable[[T], R],
        items: list[T],
    ) -> list[R | Exception]:
        """Execute function on items in parallel.

        Args:
            func: Function to apply to each item.
            items: List of items to process.

        Returns:
            List of results in same order as input.
            In "continue" mode, failed items contain the Exception.

        Raises:
            Exception: In "fail_fast" mode, re-raises first exception.
        """
        if self.n_workers == 1:
            # Sequential execution
            return self._map_sequential(func, items)

        return self._map_parallel(func, items)

    def _map_sequential(
        self,
        func: Callable[[T], R],
        items: list[T],
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
        return results

    def _map_parallel(
        self,
        func: Callable[[T], R],
        items: list[T],
    ) -> list[R | Exception]:
        """Execute in parallel using ThreadPoolExecutor."""
        results: list[R | Exception | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks with their indices
            future_to_idx = {
                executor.submit(func, item): idx for idx, item in enumerate(items)
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    if self.error_handling == "fail_fast":
                        # Cancel remaining futures
                        for f in future_to_idx:
                            f.cancel()
                        raise
                    results[idx] = e

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
