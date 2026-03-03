"""
Clustering comparison functionality.

This module provides functions for comparing two clusterings and
determining which one performs better across the metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from metricate.core.evaluator import evaluate as _evaluate
from metricate.core.exceptions import DimensionMismatchError
from metricate.core.loader import load_csv
from metricate.core.reference import METRIC_REFERENCE
from metricate.output.report import ComparisonResult

if TYPE_CHECKING:
    from metricate.training.weights import MetricWeights


def determine_winner(
    metric_name: str,
    value_a: float | None,
    value_b: float | None,
) -> str | None:
    """
    Determine which clustering wins for a specific metric.

    Args:
        metric_name: Name of the metric
        value_a: Value for clustering A
        value_b: Value for clustering B

    Returns:
        'A', 'B', 'tie', or None if comparison not possible
    """
    if value_a is None or value_b is None:
        return None

    ref = METRIC_REFERENCE.get(metric_name, {})
    direction = ref.get("direction", "higher")

    if direction == "higher":
        if value_a > value_b:
            return "A"
        elif value_b > value_a:
            return "B"
        else:
            return "tie"
    else:  # lower is better
        if value_a < value_b:
            return "A"
        elif value_b < value_a:
            return "B"
        else:
            return "tie"


def compare(
    csv_path_a: str | Path,
    csv_path_b: str | Path,
    *,
    label_col: str | None = None,
    embedding_cols: list[str] | None = None,
    exclude: list[str] | None = None,
    force_all: bool = False,
    name_a: str = "A",
    name_b: str = "B",
    weights: MetricWeights | None = None,
) -> ComparisonResult:
    """
    Compare two clusterings and determine the overall winner.

    Args:
        csv_path_a: Path to first CSV file
        csv_path_b: Path to second CSV file
        label_col: Name of the cluster label column (auto-detected if not provided)
        embedding_cols: List of embedding column names (auto-detected if not provided)
        exclude: List of metric names to skip
        force_all: If True, compute O(n²) metrics even on large datasets
        name_a: Label for first clustering (default: "A")
        name_b: Label for second clustering (default: "B")
        weights: Optional MetricWeights for weighted winner determination

    Returns:
        ComparisonResult with both evaluations, per-metric winners, and overall winner.
        If weights provided, includes weighted_winner and compound_scores.

    Raises:
        FileNotFoundError: If either CSV file does not exist
        DimensionMismatchError: If embedding dimensions don't match

    Example:
        >>> result = metricate.compare("v1.csv", "v2.csv")
        >>> print(f"Winner: {result.winner}")
        >>> print(result.to_table())
        >>>
        >>> # With weights for weighted comparison
        >>> weights = metricate.load_weights("weights.json")
        >>> result = metricate.compare("v1.csv", "v2.csv", weights=weights)
        >>> print(f"Weighted winner: {result.weighted_winner}")
    """
    # Load datasets to check dimensions before evaluation
    data_a = load_csv(csv_path_a, label_col=label_col, embedding_cols=embedding_cols)
    data_b = load_csv(csv_path_b, label_col=label_col, embedding_cols=embedding_cols)

    result = ComparisonResult()

    # Validate dimensions match
    if data_a.n_features != data_b.n_features:
        raise DimensionMismatchError(
            dimensions_a=data_a.n_features,
            dimensions_b=data_b.n_features,
        )

    # Warn about row count differences (FR-009, FR-009a)
    if data_a.n_samples != data_b.n_samples:
        pct_diff = (
            abs(data_a.n_samples - data_b.n_samples) / max(data_a.n_samples, data_b.n_samples) * 100
        )
        result.warnings = [
            f"Row counts differ: {name_a}={data_a.n_samples:,}, {name_b}={data_b.n_samples:,} "
            f"({pct_diff:.1f}% difference). Metrics are still comparable but may reflect different scales."
        ]

    # Evaluate both clusterings (with weights if provided)
    eval_a = _evaluate(
        csv_path_a,
        label_col=label_col,
        embedding_cols=embedding_cols,
        exclude=exclude,
        force_all=force_all,
        weights=weights,
    )
    eval_b = _evaluate(
        csv_path_b,
        label_col=label_col,
        embedding_cols=embedding_cols,
        exclude=exclude,
        force_all=force_all,
        weights=weights,
    )

    result.add_evaluation(name_a, eval_a)
    result.add_evaluation(name_b, eval_b)
    result.set_baseline(name_a)

    # Determine per-metric winners
    wins_a = 0
    wins_b = 0
    ties = 0

    metric_winners = {}
    for metric in eval_a.metrics:
        metric_name = metric.metric
        val_a = metric.value

        metric_b = eval_b.get_metric(metric_name)
        val_b = metric_b.value if metric_b else None

        winner = determine_winner(metric_name, val_a, val_b)
        metric_winners[metric_name] = winner

        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        elif winner == "tie":
            ties += 1

    # Store comparison metadata
    result.metric_winners = metric_winners
    result.wins = {name_a: wins_a, name_b: wins_b, "tie": ties}

    # Determine overall winner (by metric voting)
    if wins_a > wins_b:
        result.winner = name_a
    elif wins_b > wins_a:
        result.winner = name_b
    else:
        result.winner = "tie"

    result.winner_margin = abs(wins_a - wins_b)

    # Determine weighted winner if weights provided
    if weights is not None:
        score_a = eval_a.compound_score
        score_b = eval_b.compound_score

        if score_a is not None and score_b is not None:
            result.compound_scores = {name_a: score_a, name_b: score_b}

            if score_a > score_b:
                result.weighted_winner = name_a
            elif score_b > score_a:
                result.weighted_winner = name_b
            else:
                result.weighted_winner = "tie"

    return result


def compare_to_table(
    result: ComparisonResult,
    format: str = "simple",
    show_winner: bool = True,
) -> str:
    """
    Format comparison result as a table.

    Args:
        result: ComparisonResult to format
        format: Table format ('simple', 'markdown', 'grid')
        show_winner: Whether to include winner column

    Returns:
        Formatted table string
    """
    eval_names = list(result.evaluations.keys())
    if len(eval_names) != 2:
        raise ValueError("Comparison table requires exactly 2 evaluations")

    name_a, name_b = eval_names
    eval_a = result.evaluations[name_a]
    eval_b = result.evaluations[name_b]

    rows = []
    for metric in eval_a.metrics:
        metric_name = metric.metric
        val_a = metric.value

        metric_b = eval_b.get_metric(metric_name)
        val_b = metric_b.value if metric_b else None

        # Format values
        def fmt(v):
            if v is None:
                return "—"
            if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
                return f"{v:.4e}"
            return f"{v:.6f}"

        row = {
            "Metric": metric_name,
            name_a: fmt(val_a),
            name_b: fmt(val_b),
        }

        if show_winner:
            winner = result.metric_winners.get(metric_name)
            if winner == "A":
                row["Winner"] = name_a
            elif winner == "B":
                row["Winner"] = name_b
            elif winner == "tie":
                row["Winner"] = "tie"
            else:
                row["Winner"] = "—"

        rows.append(row)

    # Add summary row
    summary = {
        "Metric": "--- TOTAL ---",
        name_a: str(result.wins.get(name_a, 0)),
        name_b: str(result.wins.get(name_b, 0)),
    }
    if show_winner:
        summary["Winner"] = f"→ {result.winner}"
    rows.append(summary)

    if format == "markdown":
        return _markdown_table(rows)
    elif format == "grid":
        return _grid_table(rows)
    else:
        return _simple_table(rows)


def _simple_table(rows: list[dict]) -> str:
    """Generate a simple text table."""
    if not rows:
        return "No data"

    cols = list(rows[0].keys())
    widths = {}
    for col in cols:
        max_width = max(len(str(col)), max(len(str(row.get(col, ""))) for row in rows))
        widths[col] = min(max_width, 30)

    header = " | ".join(str(col).ljust(widths[col]) for col in cols)
    separator = "-+-".join("-" * widths[col] for col in cols)

    lines = [header, separator]
    for row in rows:
        row_str = " | ".join(
            str(row.get(col, ""))[: widths[col]].ljust(widths[col]) for col in cols
        )
        lines.append(row_str)

    return "\n".join(lines)


def _markdown_table(rows: list[dict]) -> str:
    """Generate a markdown table."""
    if not rows:
        return "No data"

    cols = list(rows[0].keys())
    header = "| " + " | ".join(str(col) for col in cols) + " |"
    separator = "|" + "|".join("---" for _ in cols) + "|"

    lines = [header, separator]
    for row in rows:
        row_str = "| " + " | ".join(str(row.get(col, "")) for col in cols) + " |"
        lines.append(row_str)

    return "\n".join(lines)


def _grid_table(rows: list[dict]) -> str:
    """Generate a grid-style table."""
    if not rows:
        return "No data"

    cols = list(rows[0].keys())
    widths = {}
    for col in cols:
        max_width = max(len(str(col)), max(len(str(row.get(col, ""))) for row in rows))
        widths[col] = min(max_width + 2, 32)

    top = "┌" + "┬".join("─" * widths[col] for col in cols) + "┐"
    mid = "├" + "┼".join("─" * widths[col] for col in cols) + "┤"
    bot = "└" + "┴".join("─" * widths[col] for col in cols) + "┘"

    header = (
        "│"
        + "│".join(f" {str(col)[: widths[col] - 2].center(widths[col] - 2)} " for col in cols)
        + "│"
    )

    lines = [top, header, mid]
    for row in rows:
        row_str = (
            "│"
            + "│".join(
                f" {str(row.get(col, ''))[: widths[col] - 2].ljust(widths[col] - 2)} "
                for col in cols
            )
            + "│"
        )
        lines.append(row_str)
    lines.append(bot)

    return "\n".join(lines)
