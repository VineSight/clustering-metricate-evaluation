"""Tests for parallel execution utilities (Phase 9)."""

import os

import pytest

from metricate.labricate.utils.parallel import (
    ParallelExecutor,
    cap_worker_count,
    run_parallel,
)


class TestCapWorkerCount:
    """Tests for worker count capping (T065)."""

    def test_cap_to_cpu_count(self):
        """Worker count is capped to CPU count."""
        cpu_count = os.cpu_count() or 1
        result = cap_worker_count(cpu_count + 10)
        assert result <= cpu_count

    def test_returns_requested_if_below_cap(self):
        """Returns requested count if below CPU count."""
        result = cap_worker_count(2)
        assert result == min(2, os.cpu_count() or 1)

    def test_minimum_one_worker(self):
        """At least 1 worker is always returned."""
        result = cap_worker_count(0)
        assert result >= 1

    def test_negative_defaults_to_one(self):
        """Negative values default to 1."""
        result = cap_worker_count(-5)
        assert result >= 1

    def test_warns_when_capping(self, caplog):
        """Warning logged when worker count is capped."""
        cpu_count = os.cpu_count() or 1
        with caplog.at_level("WARNING"):
            cap_worker_count(cpu_count + 100)
        # Should log a warning about capping
        assert any("cap" in r.message.lower() for r in caplog.records) or cpu_count >= 100


class TestParallelExecutor:
    """Tests for ProcessPoolExecutor wrapper (T064)."""

    def test_execute_single_task(self):
        """Executes single task correctly."""

        def task_fn(x):
            return x * 2

        executor = ParallelExecutor(n_workers=2)
        results = executor.map(task_fn, [5])
        assert results == [10]

    def test_execute_multiple_tasks(self):
        """Executes multiple tasks in parallel."""

        def task_fn(x):
            return x * x

        executor = ParallelExecutor(n_workers=2)
        results = executor.map(task_fn, [1, 2, 3, 4, 5])
        assert results == [1, 4, 9, 16, 25]

    def test_preserves_order(self):
        """Results are returned in input order."""

        def task_fn(x):
            return x

        executor = ParallelExecutor(n_workers=4)
        inputs = list(range(20))
        results = executor.map(task_fn, inputs)
        assert results == inputs

    def test_handles_exceptions_continue_mode(self):
        """In continue mode, failed tasks are recorded but don't stop execution."""

        def task_fn(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        executor = ParallelExecutor(n_workers=2, error_handling="continue")
        results = executor.map(task_fn, [1, 2, 3, 4, 5])

        # Check results: 3 should have an error
        assert len(results) == 5
        assert results[0] == 2
        assert results[1] == 4
        assert isinstance(results[2], Exception)
        assert results[3] == 8
        assert results[4] == 10

    def test_handles_exceptions_fail_fast_mode(self):
        """In fail_fast mode, stops on first error."""

        def task_fn(x):
            if x == 2:
                raise ValueError("Test error")
            return x * 2

        executor = ParallelExecutor(n_workers=1, error_handling="fail_fast")
        with pytest.raises(ValueError, match="Test error"):
            executor.map(task_fn, [1, 2, 3, 4, 5])

    def test_context_manager(self):
        """Can be used as context manager."""

        def task_fn(x):
            return x

        with ParallelExecutor(n_workers=2) as executor:
            results = executor.map(task_fn, [1, 2, 3])
        assert results == [1, 2, 3]


class TestRunParallel:
    """Tests for convenience function (T064)."""

    def test_run_parallel_simple(self):
        """Convenience function works for simple cases."""
        results = run_parallel(lambda x: x * 2, [1, 2, 3], n_workers=2)
        assert results == [2, 4, 6]

    def test_run_parallel_single_worker(self):
        """Single worker mode works (sequential fallback)."""
        results = run_parallel(lambda x: x + 1, [1, 2, 3], n_workers=1)
        assert results == [2, 3, 4]
