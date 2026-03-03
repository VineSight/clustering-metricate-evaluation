"""Degradation generation functionality."""

from metricate.degradation.toolkit import (
    ALL_DEGRADATION_TYPES,
    DEFAULT_DEGRADATION_TYPES,
    DEFAULT_LEVELS,
    DEGRADATION_TYPES,
    EXCLUDED_DEGRADATION_TYPES,
    LEVEL_FRACTIONS,
    DegradationConfig,
    DegradationEntry,
    DegradationResult,
    degrade,
)

__all__ = [
    "degrade",
    "DegradationConfig",
    "DegradationEntry",
    "DegradationResult",
    "DEGRADATION_TYPES",
    "ALL_DEGRADATION_TYPES",
    "DEFAULT_DEGRADATION_TYPES",
    "EXCLUDED_DEGRADATION_TYPES",
    "DEFAULT_LEVELS",
    "LEVEL_FRACTIONS",
]
