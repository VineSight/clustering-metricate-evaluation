"""Computation mode handling for Labricate experiments.

Provides light/heavy mode filtering to control which metrics are computed.
Light mode excludes expensive O(n²) metrics for faster iteration.
"""

from __future__ import annotations

from typing import Literal

from metricate.core.reference import METRIC_REFERENCE

__all__ = [
    "ComputationMode",
    "get_expensive_metrics",
    "apply_mode_exclusions",
]

# Type alias for computation mode
ComputationMode = Literal["light", "heavy"]


def get_expensive_metrics() -> list[str]:
    """Get list of expensive metrics that should be excluded in light mode.

    Returns:
        List of metric names that have skip_large=True in METRIC_REFERENCE.
        These are O(n²) metrics: Gamma, G-plus, Tau, Point-Biserial, McClain-Rao, NIVA.

    Example:
        >>> metrics = get_expensive_metrics()
        >>> len(metrics)
        6
        >>> "Gamma" in metrics
        True
    """
    return [
        name
        for name, info in METRIC_REFERENCE.items()
        if info.get("skip_large", False)
    ]


def apply_mode_exclusions(
    mode: ComputationMode,
    exclude_metrics: list[str] | None = None,
    include_metrics: list[str] | None = None,
) -> list[str]:
    """Apply computation mode exclusions to build final exclude list.

    Handles the precedence rules:
    1. Heavy mode: Only user's exclude_metrics apply
    2. Light mode: User's exclude_metrics + expensive metrics
    3. If include_metrics specified: Remove those from exclusions (user intent takes precedence)

    Args:
        mode: Computation mode - "light" or "heavy".
        exclude_metrics: User-specified metrics to exclude.
        include_metrics: User-specified metrics to include (overrides mode defaults).

    Returns:
        List of metric names to exclude.

    Example:
        >>> apply_mode_exclusions("heavy", ["Silhouette"])
        ['Silhouette']
        >>> apply_mode_exclusions("light", None)
        ['Gamma', 'G-plus', 'Tau', 'Point-Biserial', 'McClain-Rao', 'NIVA']
        >>> apply_mode_exclusions("light", None, include_metrics=["Gamma"])
        ['G-plus', 'Tau', 'Point-Biserial', 'McClain-Rao', 'NIVA']
    """
    # Start with user exclusions
    exclusions: set[str] = set(exclude_metrics or [])

    # Add expensive metrics in light mode
    if mode == "light":
        exclusions.update(get_expensive_metrics())

    # Remove any metrics the user explicitly wants to include
    if include_metrics:
        exclusions -= set(include_metrics)

    return sorted(exclusions)
