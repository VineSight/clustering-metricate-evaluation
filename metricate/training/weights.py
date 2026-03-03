"""
Metric weights for compound clustering quality scoring.

This module provides the MetricWeights dataclass for storing learned
coefficients that combine individual metrics into a single quality score.

Example:
    >>> from metricate.training.weights import MetricWeights, load_weights
    >>> weights = load_weights("weights.json")
    >>> score = weights.compute_score({"Silhouette_norm": 0.8, ...})
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "MetricWeights",
    "load_weights",
    "compute_compound_score",
    "validate_weights_schema",
]

logger = logging.getLogger(__name__)


@dataclass
class MetricWeights:
    """
    Learned coefficients for computing compound clustering quality score.

    The compound score is computed as:
        score = clip(Σ(weight_i × metric_i_norm) + bias, 0, 1)

    Attributes:
        coefficients: Mapping of metric name → weight (e.g., {"Silhouette_norm": 0.15})
        bias: Intercept term from regression model
        version: Schema version for forward compatibility (e.g., "1.0")
        regularization: Type of regularization used ("ridge" or "lasso")
        alpha: Regularization strength used during training
        created_at: ISO 8601 timestamp of when weights were trained
        training_samples: Number of samples used for training
        cv_r2: Cross-validation R² score
        non_zero_count: Number of non-zero coefficients
    """

    coefficients: dict[str, float]
    bias: float
    version: str = "1.0"
    regularization: str = "ridge"
    alpha: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_samples: int = 0
    cv_r2: float = 0.0
    non_zero_count: int = 0

    def __post_init__(self) -> None:
        """Validate weights after initialization."""
        if not self.coefficients:
            raise ValueError("coefficients must contain at least one metric")
        if not all(k.endswith("_norm") for k in self.coefficients):
            invalid = [k for k in self.coefficients if not k.endswith("_norm")]
            raise ValueError(f"All coefficient keys must end with '_norm', got: {invalid}")

    def to_dict(self, *, schema_compliant: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Args:
            schema_compliant: If True, nest metadata under 'metadata' key
                              to match weights-schema.json. If False, flatten
                              all fields at top level for easier access.

        Returns:
            Dictionary representation of weights.
        """
        if schema_compliant:
            # Match weights-schema.json with nested metadata
            return {
                "version": self.version,
                "coefficients": self.coefficients.copy(),
                "bias": self.bias,
                "metadata": {
                    "regularization": self.regularization,
                    "alpha": self.alpha,
                    "created_at": self.created_at,
                    "training_samples": self.training_samples,
                    "cv_r2": self.cv_r2,
                    "non_zero_count": self.non_zero_count,
                },
            }
        else:
            # Flat structure for internal use
            return {
                "version": self.version,
                "regularization": self.regularization,
                "alpha": self.alpha,
                "created_at": self.created_at,
                "training_samples": self.training_samples,
                "cv_r2": self.cv_r2,
                "non_zero_count": self.non_zero_count,
                "coefficients": self.coefficients.copy(),
                "bias": self.bias,
            }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path, *, validate: bool = True) -> None:
        """
        Save weights to JSON file.

        Args:
            path: File path to save weights to.
            validate: If True, validate against schema before saving.

        Raises:
            ValueError: If validate=True and schema validation fails.
        """
        path = Path(path)

        # Get schema-compliant dict for validation and saving
        data = self.to_dict(schema_compliant=True)

        if validate:
            is_valid, errors = validate_weights_schema(data)
            if not is_valid:
                raise ValueError(f"Schema validation failed: {errors}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved weights to {path}")


def load_weights(path: str | Path) -> MetricWeights:
    """
    Load MetricWeights from a JSON file.

    Args:
        path: Path to JSON file containing weights.

    Returns:
        MetricWeights instance loaded from file.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If JSON is invalid or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in weights file: {e}") from e

    # Validate required fields
    required_fields = ["coefficients", "bias"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Weights file missing required fields: {missing}")

    # Support both nested (schema-compliant) and flat metadata formats
    metadata = data.get("metadata", {})

    return MetricWeights(
        coefficients=data["coefficients"],
        bias=data["bias"],
        version=data.get("version", "1.0"),
        regularization=metadata.get("regularization", data.get("regularization", "ridge")),
        alpha=metadata.get("alpha", data.get("alpha", 1.0)),
        created_at=metadata.get("created_at", data.get("created_at", "")),
        training_samples=metadata.get("training_samples", data.get("training_samples", 0)),
        cv_r2=metadata.get("cv_r2", data.get("cv_r2", 0.0)),
        non_zero_count=metadata.get("non_zero_count", data.get("non_zero_count", 0)),
    )


def validate_weights_schema(weights_dict: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a weights dictionary against the expected schema.

    Args:
        weights_dict: Dictionary representation of weights (from to_dict()).

    Returns:
        Tuple of (is_valid, list of error messages).
        Empty error list means validation passed.
    """
    errors: list[str] = []

    # Required fields
    if "coefficients" not in weights_dict:
        errors.append("Missing required field: coefficients")
    elif not isinstance(weights_dict["coefficients"], dict):
        errors.append("coefficients must be a dictionary")
    elif len(weights_dict["coefficients"]) == 0:
        errors.append("coefficients must contain at least one metric")
    else:
        # Check coefficient keys end with _norm
        for key in weights_dict["coefficients"]:
            if not key.endswith("_norm"):
                errors.append(f"Coefficient key must end with '_norm': {key}")
            if not isinstance(weights_dict["coefficients"][key], (int, float)):
                errors.append(f"Coefficient value must be numeric: {key}")

    if "bias" not in weights_dict:
        errors.append("Missing required field: bias")
    elif not isinstance(weights_dict["bias"], (int, float)):
        errors.append("bias must be a number")

    # Version format
    if "version" in weights_dict:
        import re
        if not re.match(r"^\d+\.\d+$", str(weights_dict["version"])):
            errors.append(f"version must match pattern 'X.Y': {weights_dict['version']}")

    # Metadata validation (if present)
    metadata = weights_dict.get("metadata", {})
    if metadata:
        if "regularization" in metadata:
            if metadata["regularization"] not in ("ridge", "lasso"):
                errors.append(f"regularization must be 'ridge' or 'lasso': {metadata['regularization']}")
        if "alpha" in metadata:
            if not isinstance(metadata["alpha"], (int, float)) or metadata["alpha"] < 0:
                errors.append(f"alpha must be a non-negative number: {metadata['alpha']}")
        if "training_samples" in metadata:
            if not isinstance(metadata["training_samples"], int) or metadata["training_samples"] < 1:
                errors.append(f"training_samples must be a positive integer: {metadata['training_samples']}")
        if "cv_r2" in metadata:
            if not isinstance(metadata["cv_r2"], (int, float)) or metadata["cv_r2"] > 1:
                errors.append(f"cv_r2 must be a number <= 1: {metadata['cv_r2']}")

    return len(errors) == 0, errors


def compute_compound_score(
    metrics: dict[str, float],
    weights: MetricWeights,
    *,
    warn_on_missing: bool = True,
) -> tuple[float, str | None]:
    """
    Compute compound quality score from normalized metrics and weights.

    Args:
        metrics: Dictionary of metric_name → normalized value (0-1 range).
        weights: MetricWeights with coefficients and bias.
        warn_on_missing: If True, emit warning when metrics are missing.

    Returns:
        Tuple of (compound_score, warning_message).
        - compound_score is clipped to [0, 1] range
        - warning_message is None if all metrics present, else lists missing metrics

    Note:
        If some metrics are missing, remaining weights are renormalized
        to maintain the same overall scale.
    """
    coefficients = weights.coefficients
    warning_message: str | None = None

    # Check for missing metrics
    expected_metrics = set(coefficients.keys())
    available_metrics = set(metrics.keys())
    missing_metrics = expected_metrics - available_metrics

    if missing_metrics:
        # Renormalize remaining weights
        present_metrics = expected_metrics & available_metrics
        if not present_metrics:
            raise ValueError("No matching metrics between weights and input")

        # Compute sum of absolute weights for present metrics
        present_weight_sum = sum(abs(coefficients[m]) for m in present_metrics)
        total_weight_sum = sum(abs(v) for v in coefficients.values())

        if present_weight_sum == 0:
            # All present metrics have zero weight - just use bias
            score = max(0.0, min(1.0, weights.bias))
            warning_message = f"Missing metrics: {sorted(missing_metrics)}. All remaining weights are zero."
            if warn_on_missing:
                warnings.warn(warning_message)
            return score, warning_message

        # Renormalization factor to scale remaining weights
        renorm_factor = total_weight_sum / present_weight_sum if total_weight_sum > 0 else 1.0

        # Compute weighted sum with renormalized weights
        weighted_sum = sum(
            coefficients[m] * renorm_factor * metrics[m]
            for m in present_metrics
        )

        warning_message = f"Missing metrics: {sorted(missing_metrics)}. Remaining weights renormalized."
        if warn_on_missing:
            warnings.warn(warning_message)
    else:
        # All metrics present - standard weighted sum
        weighted_sum = sum(
            coefficients[m] * metrics[m]
            for m in expected_metrics
        )

    # Add bias and clip to [0, 1]
    score = weighted_sum + weights.bias
    score = max(0.0, min(1.0, score))

    return score, warning_message
