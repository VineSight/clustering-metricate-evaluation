"""Visualization utilities for Labricate experiment results.

Provides line charts for single-parameter experiments and
heatmaps for grid search experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from metricate.labricate.core.experiment import ExperimentResult


def plot_metric_vs_param(
    result: ExperimentResult,
    metric: str,
    param: str,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> plt.Figure:
    """Plot a line chart of metric values vs parameter values.

    Args:
        result: ExperimentResult containing runs to plot.
        metric: Name of the metric to plot on Y axis.
        param: Parameter path to plot on X axis.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches (width, height).
        title: Optional custom title (auto-generated if None).

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If metric is not found in results.
    """
    # Extract param values and metric values from completed runs
    completed_runs = [r for r in result.runs if r.pipeline_result.status == "completed"]

    if not completed_runs:
        raise ValueError("No completed runs to plot")

    # Check that metric exists (compound_score is a special case)
    metric_found = metric == "compound_score"
    if not metric_found:
        for run in completed_runs:
            for m in run.metrics:
                if m.name == metric:
                    metric_found = True
                    break
            if metric_found:
                break

    if not metric_found:
        raise ValueError(f"Metric '{metric}' not found in experiment results")

    # Extract data points
    x_values = []
    y_values = []
    for run in completed_runs:
        # Get param value
        param_value = run.param_values.get(param)
        if param_value is None:
            continue

        # Get metric value - check compound_score as special case
        metric_value = None
        if metric == "compound_score" and run.compound_score is not None:
            metric_value = run.compound_score
        else:
            for m in run.metrics:
                if m.name == metric:
                    metric_value = m.value
                    break

        if metric_value is not None:
            x_values.append(param_value)
            y_values.append(metric_value)

    # Sort by x values for proper line plotting
    sorted_pairs = sorted(zip(x_values, y_values))
    x_values = [p[0] for p in sorted_pairs]
    y_values = [p[1] for p in sorted_pairs]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line with markers
    ax.plot(x_values, y_values, marker="o", linewidth=2, markersize=8)

    # Set labels
    param_name = param.split(".")[-1]  # Get last part of path
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())

    # Set title
    if title is None:
        title = f"{metric.replace('_', ' ').title()} vs {param_name.replace('_', ' ').title()}"
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    fig.tight_layout()

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_heatmap(
    result: ExperimentResult,
    metric: str,
    param_x: str,
    param_y: str,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot a heatmap of metric values for two parameters.

    Args:
        result: ExperimentResult containing grid search runs.
        metric: Name of the metric for the heatmap values.
        param_x: Parameter path for X axis.
        param_y: Parameter path for Y axis.
        output_path: Optional path to save the figure.
        figsize: Figure size in inches (width, height).
        title: Optional custom title (auto-generated if None).
        cmap: Colormap name for the heatmap.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If metric is not found or insufficient data for heatmap.
    """
    completed_runs = [r for r in result.runs if r.pipeline_result.status == "completed"]

    if not completed_runs:
        raise ValueError("No completed runs to plot")

    # Extract unique x and y values
    x_values_set: set[Any] = set()
    y_values_set: set[Any] = set()
    data_points: dict[tuple[Any, Any], float] = {}

    for run in completed_runs:
        x_val = run.param_values.get(param_x)
        y_val = run.param_values.get(param_y)

        if x_val is None or y_val is None:
            continue

        x_values_set.add(x_val)
        y_values_set.add(y_val)

        # Get metric value - check compound_score as special case
        if metric == "compound_score" and run.compound_score is not None:
            data_points[(x_val, y_val)] = run.compound_score
        else:
            for m in run.metrics:
                if m.name == metric:
                    data_points[(x_val, y_val)] = m.value
                    break

    if len(x_values_set) < 2 or len(y_values_set) < 2:
        raise ValueError(
            f"Insufficient data for heatmap: need at least 2 values for each parameter"
        )

    # Sort values
    x_values = sorted(x_values_set)
    y_values = sorted(y_values_set)

    # Create 2D array for heatmap
    heatmap_data = np.zeros((len(y_values), len(x_values)))
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            heatmap_data[i, j] = data_points.get((x_val, y_val), np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_yticklabels([str(v) for v in y_values])

    # Set axis labels
    param_x_name = param_x.split(".")[-1]
    param_y_name = param_y.split(".")[-1]
    ax.set_xlabel(param_x_name.replace("_", " ").title())
    ax.set_ylabel(param_y_name.replace("_", " ").title())

    # Add value annotations
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                text = ax.text(
                    j, i, f"{value:.3f}",
                    ha="center", va="center",
                    color="white" if value > np.nanmean(heatmap_data) else "black",
                    fontsize=8,
                )

    # Set title
    if title is None:
        title = f"{metric.replace('_', ' ').title()} Heatmap"
    ax.set_title(title)

    # Tight layout
    fig.tight_layout()

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
