"""
Output dataclasses for metric evaluation results.

This module provides structured containers for metric evaluation results
with serialization methods for various output formats.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from metricate.core.reference import METRIC_REFERENCE


@dataclass
class MetricValue:
    """Container for a single metric evaluation result."""

    metric: str
    value: float | None
    range: str = ""
    direction: str = ""
    tier: str = ""
    computed: bool = True
    skip_reason: str | None = None

    def __post_init__(self):
        """Populate metadata from reference if not provided."""
        if self.metric in METRIC_REFERENCE:
            ref = METRIC_REFERENCE[self.metric]
            if not self.range:
                self.range = ref.get("range", "")
            if not self.direction:
                self.direction = ref.get("direction", "")
            if not self.tier:
                self.tier = ref.get("tier", "")

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "metric": self.metric,
            "value": self.value,
            "range": self.range,
            "direction": self.direction,
            "tier": self.tier,
            "computed": self.computed,
            "skip_reason": self.skip_reason,
        }


@dataclass
class EvaluationResult:
    """Container for complete metric evaluation results."""

    metrics: list[MetricValue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    compound_score: float | None = None
    compound_score_warning: str | None = None
    final_score: float | None = None
    final_score_warning: str | None = None

    def add_metric(self, metric_value: MetricValue) -> None:
        """Add a metric result to the evaluation."""
        self.metrics.append(metric_value)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def get_metric(self, name: str) -> MetricValue | None:
        """Get a specific metric by name."""
        for m in self.metrics:
            if m.metric == name:
                return m
        return None

    def computed_metrics(self) -> list[MetricValue]:
        """Return only metrics that were successfully computed."""
        return [m for m in self.metrics if m.computed and m.value is not None]

    def skipped_metrics(self) -> list[MetricValue]:
        """Return only metrics that were skipped."""
        return [m for m in self.metrics if not m.computed or m.value is None]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame."""
        data = []
        for m in self.metrics:
            row = {
                "Metric": m.metric,
                "Value": m.value,
                "Range": m.range,
                "Direction": m.direction,
                "Tier": m.tier,
            }
            if not m.computed:
                row["Status"] = f"Skipped: {m.skip_reason}"
            else:
                row["Status"] = "Computed"
            data.append(row)
        return pd.DataFrame(data)

    def to_table(self, format: str = "simple") -> str:
        """
        Generate a formatted table string.

        Args:
            format: Table format ('simple', 'grid', 'markdown')

        Returns:
            Formatted table string
        """
        df = self.to_dataframe()

        if format == "markdown":
            return _markdown_table(df)
        elif format == "grid":
            return _grid_table(df)
        else:  # simple
            return _simple_table(df)

    def to_dict(self) -> dict:
        """Convert entire result to dictionary."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2, include_metadata: bool = True) -> str:
        """
        Convert evaluation result to JSON string.

        Args:
            indent: JSON indentation level
            include_metadata: Whether to include metadata and warnings

        Returns:
            JSON string representation
        """
        import json
        data = {
            "metrics": [m.to_dict() for m in self.metrics],
        }
        if include_metadata:
            data["metadata"] = self.metadata
            if self.warnings:
                data["warnings"] = self.warnings
        return json.dumps(data, indent=indent, default=str)

    def to_csv(self) -> str:
        """
        Convert evaluation result to CSV string.

        Returns:
            CSV string representation
        """
        df = self.to_dataframe()
        return df.to_csv(index=False)

    def summary(self) -> dict:
        """Generate a summary of the evaluation."""
        computed = self.computed_metrics()
        skipped = self.skipped_metrics()

        return {
            "total_metrics": len(self.metrics),
            "computed": len(computed),
            "skipped": len(skipped),
            "warnings": len(self.warnings),
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Container for comparing two or more clustering evaluations."""

    evaluations: dict[str, EvaluationResult] = field(default_factory=dict)
    baseline_name: str | None = None
    winner: str | None = None
    wins: dict[str, int] = field(default_factory=dict)
    metric_winners: dict[str, str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    winner_margin: int = 0
    weighted_winner: str | None = None
    compound_scores: dict[str, float] = field(default_factory=dict)

    def add_evaluation(self, name: str, result: EvaluationResult) -> None:
        """Add an evaluation result with a label."""
        self.evaluations[name] = result
        if self.baseline_name is None:
            self.baseline_name = name

    def set_baseline(self, name: str) -> None:
        """Set which evaluation is the baseline for comparisons."""
        if name in self.evaluations:
            self.baseline_name = name
        else:
            raise ValueError(f"Evaluation '{name}' not found")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a comparison DataFrame with metrics as rows and evaluations as columns.
        """
        if not self.evaluations:
            return pd.DataFrame()

        # Get all unique metric names
        all_metrics = set()
        for eval_result in self.evaluations.values():
            for m in eval_result.metrics:
                all_metrics.add(m.metric)

        # Build comparison data
        data = []
        for metric_name in sorted(all_metrics):
            row = {"Metric": metric_name}
            for eval_name, eval_result in self.evaluations.items():
                m = eval_result.get_metric(metric_name)
                row[eval_name] = m.value if m and m.computed else None
            data.append(row)

        return pd.DataFrame(data)

    def compute_deltas(self) -> pd.DataFrame:
        """
        Compute deltas between each evaluation and the baseline.

        Returns:
            DataFrame with delta columns for each non-baseline evaluation
        """
        if not self.baseline_name or self.baseline_name not in self.evaluations:
            raise ValueError("Baseline not set or not found")

        df = self.to_dataframe()
        baseline = df[self.baseline_name]

        for eval_name in self.evaluations:
            if eval_name != self.baseline_name:
                df[f"{eval_name}_delta"] = df[eval_name] - baseline
                # Percentage change
                with pd.option_context("mode.use_inf_as_na", True):
                    df[f"{eval_name}_pct"] = (
                        (df[eval_name] - baseline) / baseline.abs() * 100
                    ).fillna(0)

        return df

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "baseline": self.baseline_name,
            "evaluations": {name: result.to_dict() for name, result in self.evaluations.items()},
        }

    def to_table(self, format: str = "simple", show_winner: bool = True) -> str:
        """
        Format comparison result as a table.

        Args:
            format: Table format ('simple', 'markdown', 'grid')
            show_winner: Whether to include winner column

        Returns:
            Formatted table string
        """
        from metricate.comparison.compare import compare_to_table
        return compare_to_table(self, format=format, show_winner=show_winner)


def _simple_table(df: pd.DataFrame) -> str:
    """Generate a simple text table."""
    if df.empty:
        return "No data"

    # Calculate column widths
    cols = list(df.columns)
    widths = {}
    for col in cols:
        max_width = max(len(str(col)), int(df[col].astype(str).str.len().max()))
        widths[col] = min(max_width, 30)  # Cap at 30 chars

    # Build header
    header = " | ".join(str(col).ljust(widths[col]) for col in cols)
    separator = "-+-".join("-" * widths[col] for col in cols)

    # Build rows
    rows = []
    for _, row in df.iterrows():
        row_str = " | ".join(str(row[col])[: widths[col]].ljust(widths[col]) for col in cols)
        rows.append(row_str)

    return "\n".join([header, separator] + rows)


def _grid_table(df: pd.DataFrame) -> str:
    """Generate a grid-style table with box drawing characters."""
    if df.empty:
        return "No data"

    cols = list(df.columns)
    widths = {}
    for col in cols:
        max_width = max(len(str(col)), df[col].astype(str).str.len().max())
        widths[col] = min(max_width + 2, 32)  # Padding + cap

    # Box drawing
    top = "┌" + "┬".join("─" * widths[col] for col in cols) + "┐"
    mid = "├" + "┼".join("─" * widths[col] for col in cols) + "┤"
    bot = "└" + "┴".join("─" * widths[col] for col in cols) + "┘"

    # Header
    header = (
        "│"
        + "│".join(f" {str(col)[: widths[col] - 2].center(widths[col] - 2)} " for col in cols)
        + "│"
    )

    # Rows
    rows = []
    for _, row in df.iterrows():
        row_str = (
            "│"
            + "│".join(
                f" {str(row[col])[: widths[col] - 2].ljust(widths[col] - 2)} " for col in cols
            )
            + "│"
        )
        rows.append(row_str)

    return "\n".join([top, header, mid] + rows + [bot])


def _markdown_table(df: pd.DataFrame) -> str:
    """Generate a markdown table."""
    if df.empty:
        return "No data"

    cols = list(df.columns)

    # Header
    header = "| " + " | ".join(str(col) for col in cols) + " |"
    separator = "|" + "|".join("---" for _ in cols) + "|"

    # Rows
    lines = [header, separator]
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[col]) for col in cols) + " |"
        lines.append(row_str)

    return "\n".join(lines)
